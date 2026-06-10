"""Cache storage backends for the scheduler.

The scheduler stores each request's KV cache through one of these backends
and asks for a batched view once per tick. Two implementations:

ContiguousBackend
    One tensor per layer per request, grown with torch.cat. Simple, and the
    append cost is O(T) every step because cat copies the whole tensor.

PagedBackend
    A preallocated pool of fixed-size blocks per layer plus a free list.
    Each request holds a block table (list of block ids). Appending one
    token writes one column into the tail block: O(1), no copies, no
    fragmentation, and freed blocks are reusable the moment a request
    finishes. This is the memory model vLLM built paged attention around.

The honest difference from vLLM: their attention kernel reads blocks in
place through the block table. We drive stock Hugging Face models, which
need contiguous [batch, heads, seq, head_dim] tensors, so gather() copies
blocks into that shape every tick. The management layer is real; the
kernel-level zero-copy read is not ours to give without custom kernels.
"""

from __future__ import annotations

import math

import torch

from mini_vllm.engine.kv_cache import LegacyCache, merge_caches


class PoolExhausted(RuntimeError):
    """Raised when the paged pool has no free block; the scheduler preempts."""


class ContiguousBackend:
    """v1 storage: per-request contiguous tensors, concatenated on append."""

    name = "contiguous"

    def __init__(self) -> None:
        self._caches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def add(self, request_id: str) -> None:
        self._caches[request_id] = []

    def append(self, request_id: str, new_kv: LegacyCache) -> None:
        """new_kv: per layer (k, v) with shape [1, heads, n, head_dim], n >= 1."""
        stored = self._caches[request_id]
        if not stored:
            self._caches[request_id] = [(k.clone(), v.clone()) for k, v in new_kv]
            return
        self._caches[request_id] = [
            (torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2))
            for (sk, sv), (k, v) in zip(stored, new_kv)
        ]

    def gather(self, request_ids: list[str]) -> tuple[LegacyCache, torch.Tensor]:
        caches = [tuple(self._caches[rid]) for rid in request_ids]
        device = caches[0][0][0].device
        masks = [
            torch.ones(self.seq_len(rid), dtype=torch.long, device=device)
            for rid in request_ids
        ]
        return merge_caches(caches, masks)

    def seq_len(self, request_id: str) -> int:
        stored = self._caches.get(request_id)
        if not stored:
            return 0
        return stored[0][0].shape[2]

    def drop(self, request_id: str) -> None:
        self._caches.pop(request_id, None)

    def memory_stats(self) -> dict:
        elements = sum(
            k.numel() + v.numel() for layers in self._caches.values() for k, v in layers
        )
        any_tensor = next(
            (layers[0][0] for layers in self._caches.values() if layers), None
        )
        esize = any_tensor.element_size() if any_tensor is not None else 4
        return {
            "backend": self.name,
            "requests": len(self._caches),
            "cache_bytes": elements * esize,
        }


class PagedBackend:
    """Block-pool storage with per-request block tables.

    Pool shapes are discovered lazily from the first cache tensors seen, so
    any causal LM works, including GQA models whose KV head count differs
    from the attention head count.
    """

    name = "paged"

    def __init__(self, block_size: int = 16, num_blocks: int = 256) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("block_size and num_blocks must be >= 1")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._k_pools: list[torch.Tensor] | None = None  # per layer [N, H, bs, D]
        self._v_pools: list[torch.Tensor] | None = None
        self._free: list[int] = []
        self._tables: dict[str, list[int]] = {}
        self._lens: dict[str, int] = {}

    # ------------------------------------------------------------------

    def _init_pools(self, new_kv: LegacyCache) -> None:
        self._k_pools, self._v_pools = [], []
        for k, v in new_kv:
            _, heads, _, head_dim = k.shape
            self._k_pools.append(
                torch.zeros(self.num_blocks, heads, self.block_size, head_dim,
                            dtype=k.dtype, device=k.device)
            )
            self._v_pools.append(
                torch.zeros(self.num_blocks, heads, self.block_size, head_dim,
                            dtype=v.dtype, device=v.device)
            )
        self._free = list(range(self.num_blocks))

    def add(self, request_id: str) -> None:
        self._tables[request_id] = []
        self._lens[request_id] = 0

    def append(self, request_id: str, new_kv: LegacyCache) -> None:
        """Write n new token columns into the request's blocks.

        Checks capacity up front and raises PoolExhausted before writing
        anything, so a failed append leaves the pool consistent and the
        scheduler can preempt a victim and retry.
        """
        if self._k_pools is None:
            self._init_pools(new_kv)
        n = new_kv[0][0].shape[2]
        length = self._lens[request_id]
        have = len(self._tables[request_id]) * self.block_size - length
        needed_blocks = max(0, math.ceil((n - have) / self.block_size))
        if needed_blocks > len(self._free):
            raise PoolExhausted(
                f"need {needed_blocks} blocks, {len(self._free)} free "
                f"(pool: {self.num_blocks} blocks x {self.block_size} tokens)"
            )
        for _ in range(needed_blocks):
            self._tables[request_id].append(self._free.pop())

        table = self._tables[request_id]
        for layer, (k, v) in enumerate(new_kv):
            for col in range(n):
                pos = length + col
                block = table[pos // self.block_size]
                offset = pos % self.block_size
                self._k_pools[layer][block, :, offset, :] = k[0, :, col, :]
                self._v_pools[layer][block, :, offset, :] = v[0, :, col, :]
        self._lens[request_id] = length + n

    def gather(self, request_ids: list[str]) -> tuple[LegacyCache, torch.Tensor]:
        """Assemble contiguous per-request caches from blocks, then left-pad
        merge them into one batch (reusing the same merge as v1)."""
        caches: list[LegacyCache] = []
        masks: list[torch.Tensor] = []
        device = self._k_pools[0].device
        for rid in request_ids:
            table = self._tables[rid]
            length = self._lens[rid]
            per_layer = []
            for layer in range(len(self._k_pools)):
                k = torch.cat([self._k_pools[layer][b] for b in table], dim=1)[None, :, :length, :]
                v = torch.cat([self._v_pools[layer][b] for b in table], dim=1)[None, :, :length, :]
                per_layer.append((k, v))
            caches.append(tuple(per_layer))
            masks.append(torch.ones(length, dtype=torch.long, device=device))
        return merge_caches(caches, masks)

    def seq_len(self, request_id: str) -> int:
        return self._lens.get(request_id, 0)

    def drop(self, request_id: str) -> None:
        table = self._tables.pop(request_id, None)
        self._lens.pop(request_id, None)
        if table:
            self._free.extend(table)

    def blocks_of(self, request_id: str) -> int:
        return len(self._tables.get(request_id, []))

    def memory_stats(self) -> dict:
        if self._k_pools is None:
            return {
                "backend": self.name,
                "block_size": self.block_size,
                "num_blocks": self.num_blocks,
                "used_blocks": 0,
                "utilization": 0.0,
                "pool_bytes": 0,
                "per_request_blocks": {},
            }
        used = self.num_blocks - len(self._free)
        pool_bytes = sum(p.numel() * p.element_size() for p in self._k_pools)
        pool_bytes += sum(p.numel() * p.element_size() for p in self._v_pools)
        return {
            "backend": self.name,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "used_blocks": used,
            "utilization": round(used / self.num_blocks, 3),
            "pool_bytes": pool_bytes,
            "per_request_blocks": {rid: len(t) for rid, t in self._tables.items()},
        }


def make_backend(name: str, block_size: int = 16, num_blocks: int = 256):
    if name == "contiguous":
        return ContiguousBackend()
    if name == "paged":
        return PagedBackend(block_size=block_size, num_blocks=num_blocks)
    raise ValueError(f"Unknown cache backend '{name}'. Expected: contiguous, paged.")
