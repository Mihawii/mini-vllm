"""Continuous batching scheduler.

Static batching waits for a full batch, runs it to completion, then starts
the next one. A short request stuck next to a long one wastes its slot until
the whole batch drains. Real inference servers (vLLM, TGI, Orca) schedule at
ITERATION level instead: every model step, the batch is rebuilt from whatever
requests are alive right now.

This module implements that idea on top of our GenerationEngine primitives:

  - new requests are PREFILLED individually and their KV caches are merged
    into the live batch (left-padded so sequence ends stay aligned),
  - every tick runs ONE batched decode step, so each active request gains
    exactly one token,
  - finished requests leave the batch immediately and their cache rows are
    dropped; waiting requests are admitted the moment a slot frees up.

The same Scheduler runs in two settings: driven by a wall-clock traffic file
(`mini-vllm simulate`) and inside a daemon thread behind the HTTP server,
where concurrent API calls land in one batch automatically.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import torch
import torch.nn.functional as F

from mini_vllm.engine.batching import next_position_ids
from mini_vllm.engine.generation import GenerationEngine, GenerationResult, _StopTracker
from mini_vllm.engine.kv_cache import (
    LegacyCache,
    cache_seq_len,
    from_legacy,
    pad_cache_left,
    select_rows,
    to_legacy,
    trim_left_padding,
)
from mini_vllm.engine.sampling import SamplingParams, make_generator, sample_token

_id_counter = itertools.count()


class RequestState(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class Request:
    """One generation request flowing through the scheduler."""

    prompt: str
    params: SamplingParams
    request_id: str = field(default_factory=lambda: f"req-{next(_id_counter)}")
    state: RequestState = RequestState.WAITING

    # filled during processing
    prompt_ids: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    finish_reason: str = "length"
    error: str | None = None

    # streaming: consumers read ("delta", str) items, then one terminal
    # ("done", GenerationResult) or ("error", message).
    output_queue: queue.Queue = field(default_factory=queue.Queue)

    # timing (perf_counter seconds)
    submitted_at: float = 0.0
    started_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float = 0.0

    # internals
    tracker: _StopTracker | None = None
    generator: torch.Generator | None = None

    @property
    def queue_time_s(self) -> float:
        return max(self.started_at - self.submitted_at, 0.0)

    @property
    def ttft_s(self) -> float | None:
        """Time to first token, measured from submission."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.submitted_at

    def to_result(self) -> GenerationResult:
        return GenerationResult(
            text=self.tracker.text if self.tracker else "",
            token_ids=list(self.generated),
            prompt_tokens=len(self.prompt_ids),
            completion_tokens=len(self.generated),
            finish_reason=self.finish_reason,
            latency_s=max(self.finished_at - self.submitted_at, 0.0),
            prefill_s=self.ttft_s or 0.0,
            decode_s=max(self.finished_at - self.started_at, 0.0),
        )


@dataclass
class TickStats:
    """One row of the scheduler timeline, recorded every step."""

    tick: int
    t_ms: float  # since scheduler start
    queue_depth: int
    active: int
    step_tokens: int
    cache_len: int

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "t_ms": round(self.t_ms, 2),
            "queue_depth": self.queue_depth,
            "active": self.active,
            "step_tokens": self.step_tokens,
            "cache_len": self.cache_len,
        }


class Scheduler:
    """Iteration-level scheduler over a GenerationEngine.

    Thread contract: submit() may be called from any thread; step() must only
    run on one thread (the model thread). Batched state (cache, mask, logits)
    belongs exclusively to the model thread.
    """

    def __init__(self, engine: GenerationEngine, max_batch_size: int = 8, on_event=None):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.on_event = on_event  # callback(name, request) for metrics

        self._waiting: deque[Request] = deque()
        self._waiting_lock = threading.Lock()
        self.active: list[Request] = []

        # batched decode state; rows are parallel to self.active
        self._past: LegacyCache | None = None
        self._mask: torch.Tensor | None = None
        self._next_logits: torch.Tensor | None = None

        self.ticks: list[TickStats] = []
        self._tick_count = 0
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def submit(self, request: Request) -> Request:
        """Validate and enqueue a request. Safe to call from any thread."""
        request.submitted_at = time.perf_counter()
        try:
            request.prompt_ids = self.engine._encode_prompt(
                request.prompt, request.params.max_new_tokens
            )
        except ValueError as exc:
            request.state = RequestState.FAILED
            request.error = str(exc)
            request.finished_at = time.perf_counter()
            request.output_queue.put(("error", str(exc)))
            self._emit("failed", request)
            return request
        request.tracker = _StopTracker(stop=request.params.stop)
        request.generator = make_generator(request.params.seed, self.engine.device)
        with self._waiting_lock:
            self._waiting.append(request)
        self._emit("submitted", request)
        return request

    def has_work(self) -> bool:
        with self._waiting_lock:
            waiting = bool(self._waiting)
        return waiting or bool(self.active)

    def queue_depth(self) -> int:
        with self._waiting_lock:
            return len(self._waiting)

    @torch.inference_mode()
    def step(self) -> TickStats:
        """One scheduler tick: admit, prefill, batched decode, retire."""
        self._admit_waiting()
        step_tokens = self._decode_step() if self.active else 0
        stats = TickStats(
            tick=self._tick_count,
            t_ms=(time.perf_counter() - self._t0) * 1000.0,
            queue_depth=self.queue_depth(),
            active=len(self.active),
            step_tokens=step_tokens,
            cache_len=cache_seq_len(self._past) if self._past is not None else 0,
        )
        self._tick_count += 1
        self.ticks.append(stats)
        return stats

    # ------------------------------------------------------------------
    # admission and prefill
    # ------------------------------------------------------------------

    def _admit_waiting(self) -> None:
        while len(self.active) < self.max_batch_size:
            with self._waiting_lock:
                if not self._waiting:
                    return
                request = self._waiting.popleft()
            self._prefill(request)

    def _prefill(self, request: Request) -> None:
        """Run the prompt through the model once and join the live batch."""
        device = self.engine.device
        input_ids = torch.tensor([request.prompt_ids], dtype=torch.long, device=device)
        mask_row = torch.ones(len(request.prompt_ids), dtype=torch.long, device=device)

        out = self.engine.model(
            input_ids=input_ids,
            attention_mask=mask_row.unsqueeze(0),
            use_cache=True,
        )
        new_cache = to_legacy(out.past_key_values)
        new_logits = out.logits[:, -1, :].float()

        if self._past is None:
            self._past = new_cache
            self._mask = mask_row.unsqueeze(0)
            self._next_logits = new_logits
        else:
            # The newcomer's history is shorter or longer than the live batch;
            # left-pad whichever is shorter so all sequence ends stay aligned,
            # then stack along the batch dimension. Padded K/V positions are
            # zeros, but the attention mask keeps them out of the softmax.
            batch_len = cache_seq_len(self._past)
            new_len = cache_seq_len(new_cache)
            target = max(batch_len, new_len)
            past = pad_cache_left(self._past, target - batch_len)
            new = pad_cache_left(new_cache, target - new_len)
            self._past = tuple(
                (torch.cat([pk, nk], dim=0), torch.cat([pv, nv], dim=0))
                for (pk, pv), (nk, nv) in zip(past, new)
            )
            self._mask = torch.cat(
                [
                    F.pad(self._mask, (target - batch_len, 0)),
                    F.pad(mask_row.unsqueeze(0), (target - new_len, 0)),
                ],
                dim=0,
            )
            self._next_logits = torch.cat([self._next_logits, new_logits], dim=0)

        request.state = RequestState.RUNNING
        request.started_at = time.perf_counter()
        self.active.append(request)
        self._emit("started", request)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def _decode_step(self) -> int:
        """Sample one token for every active request, then advance the batch."""
        device = self.engine.device
        eos = self.engine.tokenizer.eos_token_id
        produced = 0

        # Sample per row: each request carries its own sampling parameters,
        # seed and repetition-penalty context, so rows are independent.
        for row, request in enumerate(self.active):
            context = torch.tensor(request.prompt_ids + request.generated, device=device)
            token = sample_token(
                self._next_logits[row : row + 1], request.params, [context], request.generator
            )
            token_id = int(token[0])

            if eos is not None and token_id == eos:
                self._finish(request, "stop")
                continue
            request.generated.append(token_id)
            produced += 1
            if request.first_token_at is None:
                request.first_token_at = time.perf_counter()
            decoded = self.engine.tokenizer.decode(request.generated, skip_special_tokens=True)
            delta = request.tracker.update(decoded)
            if delta:
                request.output_queue.put(("delta", delta))
            if request.tracker.hit:
                self._finish(request, "stop")
            elif len(request.generated) >= request.params.max_new_tokens:
                self._finish(request, "length")

        # Rebuild the batch from survivors only: finished rows leave NOW,
        # which is exactly what makes this "continuous" batching.
        keep_rows = [i for i, r in enumerate(self.active) if r.state is RequestState.RUNNING]
        survivors = [self.active[i] for i in keep_rows]
        self.active = survivors

        if not survivors:
            self._past, self._mask, self._next_logits = None, None, None
            return produced

        keep = torch.tensor(keep_rows, dtype=torch.long)
        self._past = select_rows(self._past, keep)
        self._mask = self._mask[keep]
        self._past, self._mask = trim_left_padding(self._past, self._mask)

        # One forward pass advances every survivor by its newest token.
        feed = torch.tensor(
            [[r.generated[-1]] for r in survivors], dtype=torch.long, device=device
        )
        self._mask = torch.cat(
            [self._mask, torch.ones((len(survivors), 1), dtype=self._mask.dtype, device=device)],
            dim=-1,
        )
        out = self.engine.model(
            input_ids=feed,
            attention_mask=self._mask,
            position_ids=next_position_ids(self._mask),
            past_key_values=from_legacy(self._past),
            use_cache=True,
        )
        self._past = to_legacy(out.past_key_values)
        self._next_logits = out.logits[:, -1, :].float()
        return produced

    def _finish(self, request: Request, reason: str) -> None:
        request.finish_reason = reason
        request.state = RequestState.FINISHED
        request.finished_at = time.perf_counter()
        tail = request.tracker.flush()
        if tail:
            request.output_queue.put(("delta", tail))
        request.output_queue.put(("done", request.to_result()))
        self._emit("finished", request)

    def fail_all_active(self, message: str) -> None:
        """Systemic failure (e.g. OOM mid-step): notify every live request."""
        for request in self.active:
            request.state = RequestState.FAILED
            request.error = message
            request.finished_at = time.perf_counter()
            request.output_queue.put(("error", message))
            self._emit("failed", request)
        self.active = []
        self._past, self._mask, self._next_logits = None, None, None

    def _emit(self, event: str, request: Request) -> None:
        if self.on_event is not None:
            self.on_event(event, request)


class SchedulerLoop:
    """Runs Scheduler.step() on a dedicated daemon thread.

    The HTTP server submits requests from asyncio handlers; this thread is the
    only one that touches the model. An Event wakes the loop the moment work
    arrives so idle polling stays cheap.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mini-vllm-scheduler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)

    def submit(self, request: Request) -> Request:
        self.scheduler.submit(request)
        self._wake.set()
        return request

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.scheduler.has_work():
                try:
                    self.scheduler.step()
                except Exception as exc:  # noqa: BLE001 - surface to callers, keep serving
                    self.scheduler.fail_all_active(f"scheduler step failed: {exc!r}")
            else:
                self._wake.wait(timeout=0.05)
                self._wake.clear()
