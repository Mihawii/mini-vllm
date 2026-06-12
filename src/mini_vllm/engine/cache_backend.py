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

import hashlib
import math

import torch

from mini_vllm.engine.kv_cache import LegacyCache, merge_caches


class PoolExhausted(RuntimeError):
    """Raised when the paged pool has no free block; the scheduler preempts."""


def block_hash(parent: bytes | None, tokens: tuple[int, ...]) -> bytes:
    """Content address of one full block: chained over the whole prefix.

    Hashing (parent_hash, block_tokens) means a block's identity covers every
    token before it, so equal hashes imply equal prefixes. Same scheme as
    vLLM's automatic prefix caching, at educational scale.
    """
    payload = (parent or b"root") + b"|" + ",".join(map(str, tokens)).encode()
    return hashlib.sha256(payload).digest()


class ContiguousBackend:
    """v1 storage: per-request contiguous tensors, concatenated on append."""

    name = "contiguous"

    def __init__(self) -> None:
        self._caches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def add(self, request_id: str) -> None:
        self._caches[request_id] = []

    def append(self, request_id: str, new_kv: LegacyCache, tokens: list[int] | None = None) -> None:
        """new_kv: per layer (k, v) with shape [1, heads, n, head_dim], n >= 1.
        tokens is accepted for interface parity with PagedBackend (which uses
        it for prefix hashing) and ignored here."""
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

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 256,
        enable_prefix_caching: bool = False,
    ) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("block_size and num_blocks must be >= 1")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self._k_pools: list[torch.Tensor] | None = None  # per layer [N, H, bs, D]
        self._v_pools: list[torch.Tensor] | None = None
        # Free queue: ref==0 blocks in eviction order (index 0 evicts first).
        # vLLM uses a doubly linked list for O(1) unlink; a plain list is fine
        # at this pool size and much easier to read.
        self._free: list[int] = []
        self._tables: dict[str, list[int]] = {}
        self._lens: dict[str, int] = {}
        # prefix caching state: shared blocks carry a reference count, full
        # blocks get a chained content hash, and the hash table maps prefixes
        # to reusable blocks even after their request finished.
        self._ref: dict[int, int] = {}
        self._hash_of: dict[int, bytes] = {}
        self._cached: dict[bytes, int] = {}
        self._req_tokens: dict[str, list[int]] = {}
        self.prefix_queries = 0
        self.prefix_hit_tokens = 0

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
        self._req_tokens[request_id] = []

    def _allocate_block(self) -> int:
        """Pop the next-to-evict free block; reusing it invalidates any
        prefix-cache entry it carried (that is the eviction)."""
        block = self._free.pop(0)
        stale = self._hash_of.pop(block, None)
        if stale is not None and self._cached.get(stale) == block:
            del self._cached[stale]
        self._ref[block] = 1
        return block

    def match_prefix(self, request_id: str, token_ids: list[int], max_tokens: int) -> int:
        """Reuse cached blocks whose chained hashes match this prompt.

        Walks full blocks only (a partial block has no registered hash),
        bounded by max_tokens so the scheduler can keep at least one token
        for real prefill (the model must still produce next-token logits).
        Hits are "touched": ref count up, unlinked from the free queue, and
        adopted into this request's block table. Returns matched tokens.
        """
        self.prefix_queries += 1
        if not self.enable_prefix_caching or self._k_pools is None:
            return 0
        matched_blocks: list[int] = []
        parent: bytes | None = None
        limit = (min(len(token_ids), max_tokens) // self.block_size) * self.block_size
        for start in range(0, limit, self.block_size):
            chunk = tuple(token_ids[start : start + self.block_size])
            h = block_hash(parent, chunk)
            block = self._cached.get(h)
            if block is None:
                break
            matched_blocks.append(block)
            parent = h
        if not matched_blocks:
            return 0
        for block in matched_blocks:
            if self._ref.get(block, 0) == 0 and block in self._free:
                self._free.remove(block)
            self._ref[block] = self._ref.get(block, 0) + 1
        matched = len(matched_blocks) * self.block_size
        self._tables[request_id] = list(matched_blocks)
        self._lens[request_id] = matched
        self._req_tokens[request_id] = list(token_ids[:matched])
        self.prefix_hit_tokens += matched
        return matched

    def append(self, request_id: str, new_kv: LegacyCache, tokens: list[int] | None = None) -> None:
        """Write n new token columns into the request's blocks.

        Checks capacity up front and raises PoolExhausted before writing
        anything, so a failed append leaves the pool consistent and the
        scheduler can preempt a victim and retry. When prefix caching is on
        and `tokens` is provided, every block that becomes full here gets a
        chained content hash and joins the reuse table. Shared (matched)
        blocks are never written: they are full by construction, and writes
        only ever land in the tail block, which is allocated fresh per
        request.
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
            self._tables[request_id].append(self._allocate_block())

        table = self._tables[request_id]
        for layer, (k, v) in enumerate(new_kv):
            for col in range(n):
                pos = length + col
                block = table[pos // self.block_size]
                offset = pos % self.block_size
                self._k_pools[layer][block, :, offset, :] = k[0, :, col, :]
                self._v_pools[layer][block, :, offset, :] = v[0, :, col, :]
        self._lens[request_id] = length + n

        if self.enable_prefix_caching and tokens is not None:
            self._req_tokens[request_id].extend(tokens)
            self._register_full_blocks(request_id, first_new=length)

    def _register_full_blocks(self, request_id: str, first_new: int) -> None:
        """Hash and publish blocks that became full in the latest append."""
        table = self._tables[request_id]
        all_tokens = self._req_tokens[request_id]
        total = self._lens[request_id]
        first_block = first_new // self.block_size
        for idx in range(first_block, total // self.block_size):
            block = table[idx]
            if block in self._hash_of:
                continue  # already published (e.g. it was a matched block)
            chunk = tuple(all_tokens[idx * self.block_size : (idx + 1) * self.block_size])
            parent = self._hash_of.get(table[idx - 1]) if idx > 0 else None
            if idx > 0 and parent is None:
                return  # parent unhashed (prefix caching enabled mid-flight)
            h = block_hash(parent, chunk)
            self._hash_of[block] = h
            # An identical block may already be published by another request;
            # keep the first registration, leave this one as a private copy.
            self._cached.setdefault(h, block)

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
        self._req_tokens.pop(request_id, None)
        if not table:
            return
        # Deepest blocks rejoin the free queue first (vLLM's ordering): a
        # block far into a sequence hashes more context and is the least
        # likely to be reused, so it should be the first eviction candidate.
        # Hashes stay registered; that persistence IS the prefix cache.
        for block in reversed(table):
            self._ref[block] = self._ref.get(block, 1) - 1
            if self._ref[block] <= 0:
                self._ref[block] = 0
                self._free.append(block)

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
        stats = {
            "backend": self.name,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "used_blocks": used,
            "utilization": round(used / self.num_blocks, 3),
            "pool_bytes": pool_bytes,
            "per_request_blocks": {rid: len(t) for rid, t in self._tables.items()},
        }
        if self.enable_prefix_caching:
            stats["prefix"] = {
                "queries": self.prefix_queries,
                "hit_tokens": self.prefix_hit_tokens,
                "cached_blocks": len(self._cached),
            }
        return stats


def make_backend(
    name: str,
    block_size: int = 16,
    num_blocks: int = 256,
    enable_prefix_caching: bool = False,
):
    if name == "contiguous":
        return ContiguousBackend()
    if name == "paged":
        return PagedBackend(
            block_size=block_size,
            num_blocks=num_blocks,
            enable_prefix_caching=enable_prefix_caching,
        )
    raise ValueError(f"Unknown cache backend '{name}'. Expected: contiguous, paged.")
