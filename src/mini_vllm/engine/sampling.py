"""Sampling: how a token gets picked from the model's output distribution.

Every function here is a pure transform on a logits tensor of shape
[batch, vocab]. The pipeline applied at each decode step is:

    repetition penalty -> temperature -> top-k filter -> top-p filter -> sample

Keeping these as small pure functions makes them unit-testable without a
model and keeps the decode loop in generation.py readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    """Validated sampling configuration for one request.

    temperature == 0 means greedy decoding (argmax), matching common API
    conventions. top_k == 0 disables the top-k filter; top_p == 1.0 disables
    the nucleus filter.
    """

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables it), got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if isinstance(self.stop, str):
            self.stop = [self.stop]

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def apply_repetition_penalty(
    logits: torch.Tensor, context_ids: list[torch.Tensor], penalty: float
) -> torch.Tensor:
    """Discourage tokens that already appeared in each row's context.

    The CTRL-paper rule: positive logits are divided by the penalty, negative
    logits are multiplied, so repeated tokens always lose probability mass.
    context_ids holds one 1-D tensor of token ids per batch row.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for row, ids in enumerate(context_ids):
        if ids.numel() == 0:
            continue
        unique = ids.unique()
        scores = logits[row, unique]
        logits[row, unique] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sharpen (<1) or flatten (>1) the distribution. Caller handles t == 0."""
    if temperature == 1.0:
        return logits
    return logits / temperature


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest logits per row; everything else gets -inf."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability reaches p. The highest-probability token always survives.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Drop a token only if the mass BEFORE it already reached p. This keeps
    # the first token whose inclusion crosses the threshold.
    drop = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return sorted_logits.gather(-1, sorted_idx.argsort(-1))


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    context_ids: list[torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id for every row of a [batch, vocab] logits tensor."""
    if context_ids is not None and params.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, context_ids, params.repetition_penalty)

    if params.greedy:
        return logits.argmax(dim=-1)

    logits = apply_temperature(logits, params.temperature)
    logits = filter_top_k(logits, params.top_k)
    logits = filter_top_p(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    """Seeded RNG for reproducible sampling. Returns None when unseeded."""
    if seed is None:
        return None
    # torch.Generator on MPS is unsupported for multinomial; CPU generators
    # work everywhere because we sample on the logits' device only for CUDA.
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed)
    return gen
