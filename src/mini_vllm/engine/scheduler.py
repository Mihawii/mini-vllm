"""Continuous batching scheduler.

Static batching waits for a full batch, runs it to completion, then starts
the next one. A short request stuck next to a long one wastes its slot until
the whole batch drains. Real inference servers (vLLM, TGI, Orca) schedule at
ITERATION level instead: every model step, the batch is rebuilt from
whatever requests are alive right now.

This scheduler implements that idea plus three production mechanics on top:

Pluggable cache storage
    Each request's KV cache lives in a backend (contiguous tensors, or a
    paged block pool; see cache_backend.py). Every tick the scheduler asks
    the backend to assemble the live batch with left-padding.

Chunked prefill
    With `prefill_chunk_size > 0`, a long prompt is processed a chunk per
    tick instead of in one big forward pass, so decoding requests stall for
    at most one chunk, never a whole prompt.

Preemption
    When the paged pool runs out of blocks, the most recently started
    request is evicted: its blocks are freed and it goes back to the front
    of the queue. Nothing is lost; on re-admission its prompt plus the
    tokens it already generated are prefilled again (vLLM calls this
    recompute mode).

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

from mini_vllm.engine.batching import next_position_ids
from mini_vllm.engine.cache_backend import PoolExhausted, make_backend
from mini_vllm.engine.generation import GenerationEngine, GenerationResult, _StopTracker
from mini_vllm.engine.kv_cache import LegacyCache, from_legacy, to_legacy
from mini_vllm.engine.sampling import SamplingParams, make_generator, sample_token

_id_counter = itertools.count()


class RequestState(StrEnum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
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
    preemptions: int = 0

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
    next_logits: torch.Tensor | None = None  # [1, vocab], set after prefill/decode
    pending_prefill: list[int] = field(default_factory=list)  # ids not yet prefilled

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
    prefilling: int = 0
    preemptions: int = 0  # cumulative
    pool_used_blocks: int = 0
    pool_utilization: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "t_ms": round(self.t_ms, 2),
            "queue_depth": self.queue_depth,
            "active": self.active,
            "step_tokens": self.step_tokens,
            "cache_len": self.cache_len,
            "prefilling": self.prefilling,
            "preemptions": self.preemptions,
            "pool_used_blocks": self.pool_used_blocks,
            "pool_utilization": self.pool_utilization,
        }


class Scheduler:
    """Iteration-level scheduler over a GenerationEngine.

    Thread contract: submit() may be called from any thread; step() must only
    run on one thread (the model thread). Cache state belongs exclusively to
    the model thread.
    """

    def __init__(
        self,
        engine: GenerationEngine,
        max_batch_size: int = 8,
        on_event=None,
        cache_backend: str = "contiguous",
        block_size: int = 16,
        pool_blocks: int = 256,
        prefill_chunk_size: int = 0,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.on_event = on_event  # callback(name, request) for metrics
        self.backend = make_backend(cache_backend, block_size=block_size, num_blocks=pool_blocks)
        self.prefill_chunk_size = prefill_chunk_size

        self._waiting: deque[Request] = deque()
        self._waiting_lock = threading.Lock()
        self.active: list[Request] = []  # state RUNNING, in batch row order
        self.prefilling: list[Request] = []  # state PREFILLING (chunked mode)

        self.preemption_count = 0
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
        return waiting or bool(self.active) or bool(self.prefilling)

    def queue_depth(self) -> int:
        with self._waiting_lock:
            return len(self._waiting)

    @torch.inference_mode()
    def step(self) -> TickStats:
        """One scheduler tick: admit, prefill (full or one chunk), decode."""
        self._admit_waiting()
        if self.prefilling:
            self._prefill_chunk_tick()
        step_tokens = self._decode_step() if self.active else 0

        pool = self.backend.memory_stats()
        stats = TickStats(
            tick=self._tick_count,
            t_ms=(time.perf_counter() - self._t0) * 1000.0,
            queue_depth=self.queue_depth(),
            active=len(self.active),
            step_tokens=step_tokens,
            cache_len=max((self.backend.seq_len(r.request_id) for r in self.active), default=0),
            prefilling=len(self.prefilling),
            preemptions=self.preemption_count,
            pool_used_blocks=pool.get("used_blocks", 0),
            pool_utilization=pool.get("utilization", 0.0),
        )
        self._tick_count += 1
        self.ticks.append(stats)
        return stats

    # ------------------------------------------------------------------
    # admission and prefill
    # ------------------------------------------------------------------

    def _admit_waiting(self) -> None:
        while len(self.active) + len(self.prefilling) < self.max_batch_size:
            with self._waiting_lock:
                if not self._waiting:
                    return
                request = self._waiting.popleft()
            # Recompute mode after preemption: everything generated so far
            # becomes part of the text to prefill, so no work is re-sampled.
            ids = request.prompt_ids + request.generated
            if self.prefill_chunk_size > 0 and len(ids) > self.prefill_chunk_size:
                request.pending_prefill = ids
                request.state = RequestState.PREFILLING
                self.backend.add(request.request_id)
                self.prefilling.append(request)
            else:
                self._prefill_full(request, ids)

    def _prefill_full(self, request: Request, ids: list[int]) -> None:
        """One forward pass over all ids, then join the decode batch."""
        device = self.engine.device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        out = self.engine.model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        new_kv = to_legacy(out.past_key_values)

        self.backend.add(request.request_id)
        if not self._store_with_preemption(request, new_kv):
            return
        request.next_logits = out.logits[:, -1, :].float()
        self._mark_running(request)

    def _prefill_chunk_tick(self) -> None:
        """Process ONE chunk of ONE prefilling request (FIFO), bounding how
        long the decode batch can be stalled by a large prompt."""
        request = self.prefilling[0]
        device = self.engine.device
        done = self.backend.seq_len(request.request_id)
        chunk = request.pending_prefill[done : done + self.prefill_chunk_size]

        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        mask = torch.ones((1, done + len(chunk)), dtype=torch.long, device=device)
        positions = torch.arange(done, done + len(chunk), device=device).unsqueeze(0)
        if done > 0:
            past, _ = self.backend.gather([request.request_id])
            out = self.engine.model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=from_legacy(past),
                use_cache=True,
            )
        else:
            out = self.engine.model(
                input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=True
            )

        # The returned cache covers past + chunk; only the chunk is new.
        full = to_legacy(out.past_key_values)
        new_kv = tuple((k[:, :, -len(chunk):, :], v[:, :, -len(chunk):, :]) for k, v in full)
        if not self._store_with_preemption(request, new_kv):
            return
        if self.backend.seq_len(request.request_id) >= len(request.pending_prefill):
            request.pending_prefill = []
            request.next_logits = out.logits[:, -1, :].float()
            self.prefilling.remove(request)
            self._mark_running(request)

    def _mark_running(self, request: Request) -> None:
        request.state = RequestState.RUNNING
        request.started_at = time.perf_counter()
        self.active.append(request)
        self._emit("started", request)

    # ------------------------------------------------------------------
    # preemption
    # ------------------------------------------------------------------

    def _store_with_preemption(self, request: Request, new_kv: LegacyCache) -> bool:
        """backend.append with eviction on PoolExhausted.

        Victims are the most recently started requests (least work lost).
        The storing request itself can be the victim; its sampled tokens are
        already in `generated`, so recompute loses nothing. Returns False if
        the request was preempted or failed instead of stored.
        """
        while True:
            try:
                self.backend.append(request.request_id, new_kv)
                return True
            except PoolExhausted as exc:
                victim = self._pick_victim()
                if victim is None:
                    request.state = RequestState.FAILED
                    request.error = (
                        f"KV pool too small for this request alone: {exc}. "
                        "Raise --pool-blocks or --block-size."
                    )
                    request.finished_at = time.perf_counter()
                    self.backend.drop(request.request_id)
                    if request in self.active:
                        self.active.remove(request)
                    if request in self.prefilling:
                        self.prefilling.remove(request)
                    request.output_queue.put(("error", request.error))
                    self._emit("failed", request)
                    return False
                self._preempt(victim)
                if victim is request:
                    return False

    def _pick_victim(self) -> Request | None:
        candidates = self.active + self.prefilling
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.started_at or r.submitted_at)

    def _preempt(self, victim: Request) -> None:
        """Free the victim's blocks and put it back at the front of the queue."""
        self.backend.drop(victim.request_id)
        if victim in self.active:
            self.active.remove(victim)
        if victim in self.prefilling:
            self.prefilling.remove(victim)
        victim.state = RequestState.WAITING
        victim.next_logits = None
        victim.pending_prefill = []
        victim.preemptions += 1
        self.preemption_count += 1
        with self._waiting_lock:
            self._waiting.appendleft(victim)
        self._emit("preempted", victim)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def _decode_step(self) -> int:
        """Sample one token for every running request, then advance the batch."""
        device = self.engine.device
        eos = self.engine.tokenizer.eos_token_id
        produced = 0

        # Sample per row: each request carries its own sampling parameters,
        # seed and repetition-penalty context, so rows are independent.
        for request in list(self.active):
            context = torch.tensor(request.prompt_ids + request.generated, device=device)
            token = sample_token(request.next_logits, request.params, [context], request.generator)
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

        # Finished rows are gone already; rebuild the batch from survivors.
        # This per-tick reassembly is what lets requests join and leave
        # freely: there is no long-lived batched tensor to keep in sync.
        survivors = [r for r in self.active if r.state is RequestState.RUNNING]
        self.active = survivors
        if not survivors:
            return produced

        past, mask = self.backend.gather([r.request_id for r in survivors])
        feed = torch.tensor(
            [[r.generated[-1]] for r in survivors], dtype=torch.long, device=device
        )
        mask = torch.cat(
            [mask, torch.ones((len(survivors), 1), dtype=mask.dtype, device=device)], dim=-1
        )
        out = self.engine.model(
            input_ids=feed,
            attention_mask=mask,
            position_ids=next_position_ids(mask),
            past_key_values=from_legacy(past),
            use_cache=True,
        )
        returned = to_legacy(out.past_key_values)
        logits = out.logits[:, -1, :].float()

        # Store each survivor's NEW column (the last position) and logits.
        # A preemption inside this loop only drops rows that come after the
        # victim was chosen; their tokens are preserved in `generated`.
        for row, request in enumerate(list(survivors)):
            if request.state is not RequestState.RUNNING:
                continue  # preempted while storing an earlier row
            new_kv = tuple(
                (k[row : row + 1, :, -1:, :], v[row : row + 1, :, -1:, :]) for k, v in returned
            )
            if self._store_with_preemption(request, new_kv):
                request.next_logits = logits[row : row + 1]
        return produced

    def _finish(self, request: Request, reason: str) -> None:
        request.finish_reason = reason
        request.state = RequestState.FINISHED
        request.finished_at = time.perf_counter()
        self.backend.drop(request.request_id)
        tail = request.tracker.flush()
        if tail:
            request.output_queue.put(("delta", tail))
        request.output_queue.put(("done", request.to_result()))
        self._emit("finished", request)

    def fail_all_active(self, message: str) -> None:
        """Systemic failure (e.g. OOM mid-step): notify every live request."""
        for request in self.active + self.prefilling:
            request.state = RequestState.FAILED
            request.error = message
            request.finished_at = time.perf_counter()
            self.backend.drop(request.request_id)
            request.output_queue.put(("error", message))
            self._emit("failed", request)
        self.active = []
        self.prefilling = []

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
