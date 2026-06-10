"""Batch collation helpers.

Decoder-only models predict the next token from the LAST sequence position,
so a batch of different-length prompts must be LEFT-padded: real tokens sit
flush against the right edge and every row's "next token" lives at the same
index. The attention mask tells the model which positions are padding, and
position_ids must be derived from that mask so padding does not advance the
position counter.
"""

from __future__ import annotations

import torch


def collate_left_pad(
    id_lists: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token id lists into [batch, max_len] input ids + attention mask."""
    max_len = max(len(ids) for ids in id_lists)
    input_ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for row, ids in enumerate(id_lists):
        if ids:
            input_ids[row, -len(ids):] = torch.tensor(ids, dtype=torch.long)
            mask[row, -len(ids):] = 1
    return input_ids.to(device), mask.to(device)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Positions count only REAL tokens.

    With left padding, row [0,0,1,1,1] must get positions [0,0,0,1,2] (the
    clamp keeps padding at 0; it is masked out anyway). cumsum over the mask
    does exactly that. Passing explicit position_ids matters: the model's
    default arange(past_len, past_len + n) assumption is wrong for padded rows.
    """
    return (mask.long().cumsum(dim=-1) - 1).clamp(min=0)


def next_position_ids(mask: torch.Tensor) -> torch.Tensor:
    """Position of the single NEW token per row given the full mask so far.

    mask has shape [batch, cache_len + 1] (it already includes the new token).
    The new token's position equals the count of real tokens before it.
    """
    return (mask.long().sum(dim=-1, keepdim=True) - 1).clamp(min=0)
