"""KV cache plumbing.

What the cache is
-----------------
A transformer layer computes attention between the new token's query and the
keys/values of every earlier position. Without a cache, generating token N
recomputes keys and values for all N-1 earlier tokens at every layer, so a
T-token generation costs O(T^2) forward work. With a cache we store each
layer's keys and values the first time a position is processed and reuse
them, so each decode step only computes the NEW token's K/V. That is why
cached decoding feeds the model exactly one token per step.

Layout
------
We normalize everything to the "legacy" representation because it is plain
tensors and easy to reason about:

    cache  = tuple over layers of (keys, values)
    keys   : [batch, num_heads, seq_len, head_dim]
    values : [batch, num_heads, seq_len, head_dim]

Hugging Face models return Cache objects (DynamicCache in our case) whose
internals have changed several times across releases. The two functions
to_legacy / from_legacy are the only place in the codebase that touches that
API surface, so version drift stays contained here.

Batching caches
---------------
The continuous batching scheduler needs to merge caches of requests that have
different lengths. We left-pad the sequence dimension with zeros and rely on
the attention mask to exclude those positions: a masked position never enters
the softmax, so the zero K/V rows are never read.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache

# tuple over layers of (keys, values), each [batch, heads, seq, head_dim]
LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def to_legacy(cache) -> LegacyCache | None:
    """Extract plain (keys, values) tensors from whatever the model returned."""
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return cache
    if hasattr(cache, "layers"):  # transformers >= 4.54 DynamicCache layout
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache"):  # older DynamicCache layout
        return tuple(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"Unsupported cache type: {type(cache).__name__}")


def from_legacy(legacy: LegacyCache | None) -> DynamicCache | None:
    """Wrap plain tensors back into the Cache object the model expects."""
    if legacy is None:
        return None
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(legacy):
        cache.update(keys, values, layer_idx)
    return cache


def cache_seq_len(legacy: LegacyCache) -> int:
    return legacy[0][0].shape[2]


def cache_batch_size(legacy: LegacyCache) -> int:
    return legacy[0][0].shape[0]


def pad_cache_left(legacy: LegacyCache, pad_len: int) -> LegacyCache:
    """Grow the sequence dimension on the left with zeros.

    F.pad's spec runs from the last dimension backward, so (0, 0, pad_len, 0)
    means: head_dim unchanged, seq_len gets pad_len new positions on the left.
    """
    if pad_len <= 0:
        return legacy
    return tuple(
        (F.pad(k, (0, 0, pad_len, 0)), F.pad(v, (0, 0, pad_len, 0))) for k, v in legacy
    )


def merge_caches(
    caches: list[LegacyCache], masks: list[torch.Tensor]
) -> tuple[LegacyCache, torch.Tensor]:
    """Stack several single-request caches into one batched cache.

    Each request may have a different sequence length (different prompt sizes,
    different numbers of generated tokens). Shorter caches are left-padded to
    the longest one and the returned attention mask marks which positions are
    real. masks[i] is the 1-D attention mask for request i.
    """
    lengths = [cache_seq_len(c) for c in caches]
    target = max(lengths)
    padded = [pad_cache_left(c, target - length) for c, length in zip(caches, lengths)]

    merged: LegacyCache = tuple(
        (
            torch.cat([p[layer][0] for p in padded], dim=0),
            torch.cat([p[layer][1] for p in padded], dim=0),
        )
        for layer in range(len(caches[0]))
    )
    batched_mask = torch.stack(
        [F.pad(m, (target - m.shape[0], 0)) for m in masks], dim=0
    )
    return merged, batched_mask


def select_rows(legacy: LegacyCache, keep: torch.Tensor) -> LegacyCache:
    """Keep only the given batch rows (used to retire finished requests)."""
    return tuple((k[keep], v[keep]) for k, v in legacy)


def trim_left_padding(
    legacy: LegacyCache, mask: torch.Tensor
) -> tuple[LegacyCache, torch.Tensor]:
    """Drop leading columns that are padding in EVERY row.

    After short requests retire, the batch may carry columns no row uses.
    Trimming them keeps memory and attention cost proportional to the longest
    LIVE request. This is a tiny taste of why vLLM manages cache memory in
    pages instead of one contiguous block per batch.
    """
    real = mask.any(dim=0).nonzero()
    if real.numel() == 0:
        return legacy, mask
    first = int(real[0])
    if first == 0:
        return legacy, mask
    trimmed = tuple((k[:, :, first:, :], v[:, :, first:, :]) for k, v in legacy)
    return trimmed, mask[:, first:]
