"""Speculative decoding.

The idea (Leviathan et al., Chen et al.): a small DRAFT model proposes a few
tokens cheaply, and the large TARGET model verifies all of them in ONE
forward pass instead of one pass per token. A modified rejection-sampling
rule decides which proposals survive, and it is lossless: the committed
token stream follows exactly the distribution the target alone would have
produced. Under greedy decoding that guarantee becomes testable equality,
and this repo's tests assert it token for token.

One round:

1. The draft autoregressively proposes gamma tokens x_1..x_gamma, keeping
   its probability distribution q_i for each.
2. The target runs one forward over its unseen suffix plus all gamma
   proposals, yielding p_i for every proposal position plus one extra row.
3. Each x_i is accepted with probability min(1, p_i(x_i) / q_i(x_i)).
   At the first rejection, a replacement token is drawn from
   normalize(max(0, p_i - q_i)) and the rest of the draft is discarded.
   If everything is accepted, a BONUS token is sampled from the extra row,
   so a fully successful round commits gamma + 1 tokens.
4. Both models' KV caches roll back to the committed length (a slice for
   us); rejected positions' cache entries are invalid by definition.

Greedy specialization: q is a point mass, so the rule reduces to "accept
while the target's argmax equals the draft token", and the replacement and
bonus are the target argmax. Output is then identical to target-only greedy
decoding, which is the cornerstone correctness test.

The economics are honest: each round costs gamma draft forwards plus one
target forward. With a CPU target only ~2x the draft's cost (gpt2 vs
distilgpt2), the breakeven acceptance rate is high; the win grows with the
target/draft size ratio and on GPUs where a forward pass has high fixed
cost. The benchmark reports whichever way the numbers land.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_vllm.engine.generation import GenerationEngine, GenerationResult, _StopTracker
from mini_vllm.engine.kv_cache import LegacyCache, from_legacy, slice_cache, to_legacy
from mini_vllm.engine.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    apply_temperature,
    filter_top_k,
    filter_top_p,
    make_generator,
)
from mini_vllm.utils.timing import Timer


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    bonus: int = 0
    rounds: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def to_dict(self) -> dict:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "bonus": self.bonus,
            "rounds": self.rounds,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "acceptance_rate": round(self.acceptance_rate, 3),
        }


class SpeculativeEngine:
    """Drives a target GenerationEngine with proposals from a draft engine."""

    def __init__(self, target: GenerationEngine, draft: GenerationEngine, gamma: int = 4):
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {gamma}")
        self.target = target
        self.draft = draft
        self.gamma = gamma
        self._check_tokenizers()

    def _check_tokenizers(self) -> None:
        t, d = self.target.tokenizer, self.draft.tokenizer
        probe = "speculative decoding tokenizer probe"
        if t.vocab_size != d.vocab_size or t.encode(probe) != d.encode(probe):
            raise ValueError(
                "Draft and target must share a tokenizer (same vocab, same "
                f"encoding). Got {self.target.lm.name} vs {self.draft.lm.name}."
            )

    # ------------------------------------------------------------------

    def _forward(self, engine: GenerationEngine, tokens: list[int], past: LegacyCache | None,
                 past_len: int) -> tuple[LegacyCache, torch.Tensor]:
        """One forward over `tokens` given a cache of length past_len.
        Returns (new cache, logits for every fed position [n, vocab])."""
        device = engine.device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        mask = torch.ones((1, past_len + len(tokens)), dtype=torch.long, device=device)
        positions = torch.arange(past_len, past_len + len(tokens), device=device).unsqueeze(0)
        out = engine.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=from_legacy(past) if past is not None else None,
            use_cache=True,
        )
        return to_legacy(out.past_key_values), out.logits[0].float()

    def _dist(self, logits_row: torch.Tensor, params: SamplingParams,
              context: list[int], device: torch.device) -> torch.Tensor:
        """The engine's sampling pipeline as an explicit distribution [vocab]."""
        logits = logits_row.unsqueeze(0)
        if params.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(
                logits, [torch.tensor(context, device=device)], params.repetition_penalty
            )
        logits = apply_temperature(logits, params.temperature)
        logits = filter_top_k(logits, params.top_k)
        logits = filter_top_p(logits, params.top_p)
        return torch.softmax(logits, dim=-1).squeeze(0)

    def _argmax(self, logits_row: torch.Tensor, params: SamplingParams,
                context: list[int], device: torch.device) -> int:
        logits = logits_row.unsqueeze(0)
        if params.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(
                logits, [torch.tensor(context, device=device)], params.repetition_penalty
            )
        return int(logits.argmax(dim=-1)[0])

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(self, prompt: str, params: SamplingParams | None = None
                 ) -> tuple[GenerationResult, SpecStats]:
        params = params or SamplingParams()
        prompt_ids = self.target._encode_prompt(prompt, params.max_new_tokens)
        device = self.target.device
        eos = self.target.tokenizer.eos_token_id
        generator = make_generator(params.seed, device)
        greedy = params.greedy

        committed = list(prompt_ids)  # prompt + all committed completions
        generated: list[int] = []
        tracker = _StopTracker(stop=params.stop)
        stats = SpecStats()
        finish_reason = "length"
        done = False

        target_past: LegacyCache | None = None
        target_len = 0
        draft_past: LegacyCache | None = None
        draft_len = 0

        timer = Timer()
        timer.__enter__()
        while len(generated) < params.max_new_tokens and not done:
            stats.rounds += 1
            gamma = min(self.gamma, params.max_new_tokens - len(generated))

            # ---- 1) draft proposes gamma tokens, one forward each ----
            draft_tokens: list[int] = []
            q_rows: list[torch.Tensor | None] = []
            feed = committed[draft_len:]
            for _ in range(gamma):
                draft_past, logits = self._forward(self.draft, feed, draft_past, draft_len)
                draft_len += len(feed)
                stats.draft_forwards += 1
                row = logits[-1]
                context = committed + draft_tokens
                if greedy:
                    token = self._argmax(row, params, context, device)
                    q_rows.append(None)
                else:
                    q = self._dist(row, params, context, device)
                    token = int(torch.multinomial(q, 1, generator=generator))
                    q_rows.append(q)
                draft_tokens.append(token)
                feed = [token]
            stats.proposed += gamma

            # ---- 2) target verifies everything in one forward ----
            t_feed = committed[target_len:] + draft_tokens
            target_past, t_logits = self._forward(self.target, t_feed, target_past, target_len)
            target_len += len(t_feed)
            stats.target_forwards += 1
            # Row that PREDICTS draft_tokens[i] sits right before it:
            base = len(t_feed) - gamma - 1

            # ---- 3) acceptance loop ----
            n_accepted = 0
            replacement: int | None = None
            for i, token in enumerate(draft_tokens):
                row = t_logits[base + i]
                context = committed + draft_tokens[:i]
                if greedy:
                    target_choice = self._argmax(row, params, context, device)
                    if target_choice == token:
                        n_accepted += 1
                        continue
                    replacement = target_choice
                    break
                p = self._dist(row, params, context, device)
                q = q_rows[i]
                ratio = p[token] / q[token] if q[token] > 0 else torch.tensor(0.0)
                u = torch.rand((), generator=generator)
                if u < torch.clamp(ratio, max=1.0):
                    n_accepted += 1
                    continue
                residual = torch.clamp(p - q, min=0.0)
                total = residual.sum()
                replacement = int(
                    torch.multinomial(residual / total, 1, generator=generator)
                    if total > 0
                    else torch.multinomial(p, 1, generator=generator)
                )
                break
            stats.accepted += n_accepted

            if replacement is None:
                # every proposal survived: the extra target row is free
                row = t_logits[base + gamma]
                context = committed + draft_tokens
                if greedy:
                    extra = self._argmax(row, params, context, device)
                else:
                    extra = int(
                        torch.multinomial(self._dist(row, params, context, device), 1,
                                          generator=generator)
                    )
                new_tokens = draft_tokens + [extra]
                stats.bonus += 1
            else:
                new_tokens = draft_tokens[:n_accepted] + [replacement]

            # ---- 4) roll caches back to the committed boundary ----
            keep = len(committed) + n_accepted
            target_past = slice_cache(target_past, min(keep, target_len))
            target_len = min(keep, target_len)
            draft_past = slice_cache(draft_past, min(keep, draft_len))
            draft_len = min(keep, draft_len)

            # ---- 5) commit tokens with the usual stop rules ----
            for token in new_tokens:
                if eos is not None and token == eos:
                    finish_reason = "stop"
                    done = True
                    break
                generated.append(token)
                committed.append(token)
                tracker.update(self.target.tokenizer.decode(generated, skip_special_tokens=True))
                if tracker.hit:
                    finish_reason = "stop"
                    done = True
                    break
                if len(generated) >= params.max_new_tokens:
                    done = True
                    break
        timer.__exit__()

        tracker.flush()
        result = GenerationResult(
            text=tracker.text,
            token_ids=generated,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            latency_s=timer.elapsed,
        )
        return result, stats
