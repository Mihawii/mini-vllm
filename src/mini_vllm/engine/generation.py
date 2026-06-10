"""The custom autoregressive decoding loop.

This is the heart of mini-vLLM. The main path never calls model.generate();
it drives the model with plain forward passes in two phases:

Prefill
    One forward pass over the whole prompt. The model computes hidden states
    for every prompt position in parallel and (when caching) stores each
    layer's keys/values. The logits at the LAST position give the
    distribution for the first new token.

Decode
    One forward pass per new token. With a KV cache we feed ONLY the newest
    token: attention still sees the full history because the keys/values of
    all earlier positions are read from the cache instead of being recomputed.
    Without the cache we must re-feed the entire growing sequence each step,
    which is the O(T^2) behavior the benchmark makes visible.

Sampling happens between steps: logits -> repetition penalty -> temperature
-> top-k -> top-p -> multinomial draw (or argmax when temperature is 0).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from mini_vllm.engine.batching import (
    collate_left_pad,
    next_position_ids,
    position_ids_from_mask,
)
from mini_vllm.engine.kv_cache import from_legacy, to_legacy
from mini_vllm.engine.model_loader import LoadedModel
from mini_vllm.engine.sampling import SamplingParams, make_generator, sample_token
from mini_vllm.utils.timing import Timer


class PromptTooLongError(ValueError):
    """Raised when prompt + requested tokens exceed the model's context window."""


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" (EOS or stop string) or "length"
    latency_s: float
    prefill_s: float = 0.0
    decode_s: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return self.completion_tokens / self.latency_s


@dataclass
class _StopTracker:
    """Incremental detokenization with stop-string handling.

    Streamed deltas hold back max(len(stop)) - 1 characters so a stop string
    split across several tokens is never partially emitted to the client.
    """

    stop: list[str]
    emitted: str = ""
    text: str = ""
    hit: bool = False
    _holdback: int = field(init=False)

    def __post_init__(self) -> None:
        self._holdback = max((len(s) for s in self.stop), default=0)
        self._holdback = max(self._holdback - 1, 0)

    def update(self, decoded: str) -> str:
        """Feed the full decoded text so far; return the safe-to-stream delta."""
        self.text = decoded
        for s in self.stop:
            idx = decoded.find(s)
            if idx != -1:
                self.text = decoded[:idx]
                self.hit = True
                break
        safe_until = len(self.text) if self.hit else max(len(self.text) - self._holdback, 0)
        delta = self.text[len(self.emitted):safe_until]
        if delta:
            self.emitted = self.text[:safe_until]
        return delta

    def flush(self) -> str:
        delta = self.text[len(self.emitted):]
        self.emitted = self.text
        return delta


class GenerationEngine:
    """Drives a LoadedModel with our own prefill/decode loop."""

    def __init__(self, loaded: LoadedModel):
        self.lm = loaded
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.device = loaded.device

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str, max_new_tokens: int) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("Prompt produced no tokens; provide a non-empty prompt.")
        limit = self.lm.context_length
        if limit is not None and len(ids) + max_new_tokens > limit:
            raise PromptTooLongError(
                f"Prompt has {len(ids)} tokens and max_new_tokens={max_new_tokens}, "
                f"but the model context window is {limit} tokens. "
                "Shorten the prompt or lower max_new_tokens."
            )
        return ids

    # ------------------------------------------------------------------
    # single-sequence generation (the educational core)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        use_kv_cache: bool = True,
        stream_callback: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        params = params or SamplingParams()
        prompt_ids = self._encode_prompt(prompt, params.max_new_tokens)
        eos = self.tokenizer.eos_token_id
        generator = make_generator(params.seed, self.device)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)

        total_timer = Timer()
        total_timer.__enter__()

        # ---- PREFILL: process the whole prompt in one parallel pass ----
        with Timer() as prefill_timer:
            out = self.model(input_ids=input_ids, attention_mask=mask, use_cache=use_kv_cache)
        # Logits have shape [1, prompt_len, vocab]; only the last position
        # predicts an unseen token, so that row seeds the decode loop.
        next_logits = out.logits[:, -1, :].float()
        past = to_legacy(out.past_key_values) if use_kv_cache else None

        generated: list[int] = []
        tracker = _StopTracker(stop=params.stop)
        finish_reason = "length"

        # ---- DECODE: one token per iteration ----
        with Timer() as decode_timer:
            for step in range(params.max_new_tokens):
                context = torch.tensor(prompt_ids + generated, device=self.device)
                token = sample_token(next_logits, params, [context], generator)
                token_id = int(token[0])

                # EOS means the model chose to stop; do not append the marker.
                if eos is not None and token_id == eos:
                    finish_reason = "stop"
                    break

                generated.append(token_id)
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                delta = tracker.update(decoded)
                if stream_callback and delta:
                    stream_callback(delta)
                if tracker.hit:
                    finish_reason = "stop"
                    break
                if step == params.max_new_tokens - 1:
                    break  # budget exhausted; skip the forward pass for a token we will not use

                if use_kv_cache:
                    # Feed ONLY the new token. Its K/V get appended to the
                    # cache; attention reads every earlier position from there.
                    mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=self.device)], dim=-1)
                    out = self.model(
                        input_ids=token.view(1, 1),
                        attention_mask=mask,
                        position_ids=next_position_ids(mask),
                        past_key_values=from_legacy(past),
                        use_cache=True,
                    )
                    past = to_legacy(out.past_key_values)
                else:
                    # No cache: re-run the FULL sequence. Every step repeats
                    # all earlier K/V work, so steps get slower as text grows.
                    full = torch.tensor([prompt_ids + generated], dtype=torch.long, device=self.device)
                    out = self.model(input_ids=full, use_cache=False)
                next_logits = out.logits[:, -1, :].float()

        if stream_callback:
            tail = tracker.flush()
            if tail:
                stream_callback(tail)

        total_timer.__exit__()
        return GenerationResult(
            text=tracker.text,
            token_ids=generated,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            latency_s=total_timer.elapsed,
            prefill_s=prefill_timer.elapsed,
            decode_s=decode_timer.elapsed,
        )

    # ------------------------------------------------------------------
    # static batching: several prompts decoded in lockstep
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        params: SamplingParams | None = None,
        use_kv_cache: bool = True,
    ) -> list[GenerationResult]:
        params = params or SamplingParams()
        if not prompts:
            return []
        id_lists = [self._encode_prompt(p, params.max_new_tokens) for p in prompts]
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        generator = make_generator(params.seed, self.device)
        batch = len(prompts)

        total_timer = Timer()
        total_timer.__enter__()

        # Left padding lines every prompt's last token up at the right edge.
        input_ids, mask = collate_left_pad(id_lists, pad, self.device)
        with Timer() as prefill_timer:
            out = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids_from_mask(mask),
                use_cache=use_kv_cache,
            )
        next_logits = out.logits[:, -1, :].float()
        past = to_legacy(out.past_key_values) if use_kv_cache else None

        generated: list[list[int]] = [[] for _ in range(batch)]
        trackers = [_StopTracker(stop=params.stop) for _ in range(batch)]
        finished = [False] * batch
        reasons = ["length"] * batch

        with Timer() as decode_timer:
            for step in range(params.max_new_tokens):
                contexts = [
                    torch.tensor(ids + gen, device=self.device)
                    for ids, gen in zip(id_lists, generated)
                ]
                tokens = sample_token(next_logits, params, contexts, generator)

                for row in range(batch):
                    if finished[row]:
                        continue
                    token_id = int(tokens[row])
                    if eos is not None and token_id == eos:
                        finished[row] = True
                        reasons[row] = "stop"
                        continue
                    generated[row].append(token_id)
                    trackers[row].update(
                        self.tokenizer.decode(generated[row], skip_special_tokens=True)
                    )
                    if trackers[row].hit:
                        finished[row] = True
                        reasons[row] = "stop"

                if all(finished) or step == params.max_new_tokens - 1:
                    break

                # Finished rows keep riding along with pad tokens; their
                # outputs are frozen so the extra positions are harmless.
                feed = torch.tensor(
                    [[pad if finished[r] else generated[r][-1]] for r in range(batch)],
                    dtype=torch.long,
                    device=self.device,
                )
                if use_kv_cache:
                    mask = torch.cat(
                        [mask, torch.ones((batch, 1), dtype=mask.dtype, device=self.device)], dim=-1
                    )
                    out = self.model(
                        input_ids=feed,
                        attention_mask=mask,
                        position_ids=next_position_ids(mask),
                        past_key_values=from_legacy(past),
                        use_cache=True,
                    )
                    past = to_legacy(out.past_key_values)
                else:
                    full_ids, full_mask = collate_left_pad(
                        [ids + gen for ids, gen in zip(id_lists, generated)], pad, self.device
                    )
                    out = self.model(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        position_ids=position_ids_from_mask(full_mask),
                        use_cache=False,
                    )
                next_logits = out.logits[:, -1, :].float()

        total_timer.__exit__()
        share = total_timer.elapsed
        return [
            GenerationResult(
                text=trackers[row].text,
                token_ids=generated[row],
                prompt_tokens=len(id_lists[row]),
                completion_tokens=len(generated[row]),
                finish_reason=reasons[row],
                latency_s=share,
                prefill_s=prefill_timer.elapsed,
                decode_s=decode_timer.elapsed,
            )
            for row in range(batch)
        ]

    # ------------------------------------------------------------------
    # compatibility mode: Hugging Face generate(), used ONLY for parity
    # checks and debugging. Never called by the native path above.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_hf(self, prompt: str, params: SamplingParams | None = None) -> GenerationResult:
        params = params or SamplingParams()
        prompt_ids = self._encode_prompt(prompt, params.max_new_tokens)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        kwargs: dict = {
            "max_new_tokens": params.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": params.repetition_penalty,
        }
        if params.greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                do_sample=True,
                temperature=params.temperature,
                top_k=params.top_k or 0,
                top_p=params.top_p,
            )

        with Timer() as timer:
            output = self.model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), **kwargs
            )
        new_ids = output[0, len(prompt_ids):].tolist()
        eos = self.tokenizer.eos_token_id
        finish = "stop" if eos in new_ids else "length"
        new_ids = [t for t in new_ids if t != eos]
        return GenerationResult(
            text=self.tokenizer.decode(new_ids, skip_special_tokens=True),
            token_ids=new_ids,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(new_ids),
            finish_reason=finish,
            latency_s=timer.elapsed,
        )
