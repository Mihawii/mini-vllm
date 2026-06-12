"""Prefix caching: chained hashes, ref counts, eviction, end-to-end parity."""

import torch

from mini_vllm.engine.cache_backend import PagedBackend, block_hash
from mini_vllm.engine.sampling import SamplingParams
from mini_vllm.engine.scheduler import Request, RequestState, Scheduler


def fake_kv(tokens: int, layers: int = 2, heads: int = 2, dim: int = 4):
    return tuple(
        (torch.randn(1, heads, tokens, dim), torch.randn(1, heads, tokens, dim))
        for _ in range(layers)
    )


def greedy_request(prompt: str, n: int = 8) -> Request:
    return Request(prompt=prompt, params=SamplingParams(max_new_tokens=n, temperature=0.0))


def run_until_done(scheduler: Scheduler, limit: int = 2000) -> None:
    ticks = 0
    while scheduler.has_work():
        scheduler.step()
        ticks += 1
        assert ticks < limit, "scheduler did not converge"


# ---------------------------------------------------------------------------
# backend unit tests
# ---------------------------------------------------------------------------


def test_chained_hash_is_prefix_sensitive():
    a = block_hash(None, (1, 2, 3, 4))
    same = block_hash(None, (1, 2, 3, 4))
    other_tokens = block_hash(None, (1, 2, 3, 5))
    other_parent = block_hash(a, (1, 2, 3, 4))
    assert a == same
    assert a != other_tokens
    assert a != other_parent


def test_match_reuses_full_blocks_and_refcounts():
    backend = PagedBackend(block_size=4, num_blocks=16, enable_prefix_caching=True)
    ids = list(range(100, 110))  # 10 tokens: 2 full blocks + 2 leftover

    backend.add("a")
    backend.append("a", fake_kv(10), ids)
    assert backend.memory_stats()["prefix"]["cached_blocks"] == 2

    backend.add("b")
    matched = backend.match_prefix("b", ids, max_tokens=len(ids) - 1)
    assert matched == 8  # both full blocks, never the partial tail
    # shared blocks are the same physical ids in both tables
    assert backend._tables["b"] == backend._tables["a"][:2]
    assert all(backend._ref[blk] == 2 for blk in backend._tables["b"])

    # request b writes its own suffix into a fresh tail block
    backend.append("b", fake_kv(2), ids[8:10])
    assert backend._tables["b"][2] not in backend._tables["a"]

    # gather for b must equal a's first 10 tokens worth of cache
    cache_b, _ = backend.gather(["b"])
    cache_a, _ = backend.gather(["a"])
    for (kb, _vb), (ka, _va) in zip(cache_b, cache_a):
        assert torch.equal(kb[:, :, :8, :], ka[:, :, :8, :])


def test_drop_keeps_cache_until_evicted():
    backend = PagedBackend(block_size=4, num_blocks=4, enable_prefix_caching=True)
    ids = list(range(8))
    backend.add("a")
    backend.append("a", fake_kv(8), ids)
    backend.drop("a")

    # blocks are free but still matchable
    backend.add("b")
    assert backend.match_prefix("b", ids + [99], max_tokens=8) == 8
    backend.drop("b")

    # exhaust the pool with a new request: the old blocks get evicted and
    # their hashes must disappear with them
    backend.add("c")
    backend.append("c", fake_kv(16), list(range(200, 216)))
    backend.add("d")
    assert backend.match_prefix("d", ids + [99], max_tokens=8) == 0


def test_refcounted_blocks_survive_other_request_finishing():
    backend = PagedBackend(block_size=4, num_blocks=8, enable_prefix_caching=True)
    ids = list(range(50, 58))
    backend.add("a")
    backend.append("a", fake_kv(8), ids)
    backend.add("b")
    assert backend.match_prefix("b", ids + [1, 2], max_tokens=9) == 8
    backend.drop("a")  # b still holds refs; blocks must not hit the free queue
    assert all(backend._ref[blk] == 1 for blk in backend._tables["b"])
    assert not set(backend._tables["b"]) & set(backend._free)


# ---------------------------------------------------------------------------
# scheduler end-to-end (tiny model)
# ---------------------------------------------------------------------------

SHARED = "The greenhouse robot inspected the tomatoes and wrote a careful report about"


def test_prefix_caching_preserves_output(engine):
    """Reusing cached prefix blocks must never change what gets generated."""
    cold = Scheduler(engine, max_batch_size=2, cache_backend="paged", block_size=4)
    r_cold = greedy_request(SHARED, n=8)
    cold.submit(r_cold)
    run_until_done(cold)

    warm = Scheduler(
        engine, max_batch_size=2, cache_backend="paged", block_size=4,
        enable_prefix_caching=True,
    )
    first = greedy_request(SHARED, n=8)
    second = greedy_request(SHARED, n=8)
    warm.submit(first)
    run_until_done(warm)
    warm.submit(second)
    run_until_done(warm)

    assert first.generated == r_cold.generated
    assert second.generated == r_cold.generated
    stats = warm.backend.memory_stats()["prefix"]
    assert stats["hit_tokens"] > 0, "second request should have hit the prefix cache"


def test_prefix_hit_skips_prefill_work(engine):
    """A warm identical prompt must reach RUNNING with almost no prefill:
    only the suffix beyond the matched blocks is computed."""
    scheduler = Scheduler(
        engine, max_batch_size=2, cache_backend="paged", block_size=4,
        enable_prefix_caching=True,
    )
    a = greedy_request(SHARED, n=4)
    scheduler.submit(a)
    run_until_done(scheduler)

    b = greedy_request(SHARED, n=4)
    scheduler.submit(b)
    scheduler.step()  # admit + suffix prefill in one pass
    prompt_len = len(b.prompt_ids)
    matched = scheduler.backend.prefix_hit_tokens
    assert matched >= ((prompt_len - 1) // 4) * 4
    run_until_done(scheduler)
    assert b.generated == a.generated


def test_prefix_caching_with_preemption_stays_consistent(engine):
    """Memory pressure + shared blocks: everything completes, outputs match
    the unpressured run, and the pool drains to fully free."""
    reference = Scheduler(
        engine, max_batch_size=4, cache_backend="paged", block_size=4, pool_blocks=64,
        enable_prefix_caching=True,
    )
    ref_reqs = [greedy_request(SHARED + f" {i}", n=10) for i in range(3)]
    for r in ref_reqs:
        reference.submit(r)
    run_until_done(reference)

    tight = Scheduler(
        engine, max_batch_size=4, cache_backend="paged", block_size=4, pool_blocks=14,
        enable_prefix_caching=True,
    )
    reqs = [greedy_request(SHARED + f" {i}", n=10) for i in range(3)]
    for r in reqs:
        tight.submit(r)
    run_until_done(tight)

    for got, want in zip(reqs, ref_reqs):
        assert got.state is RequestState.FINISHED
        assert got.generated == want.generated
    assert all(ref >= 0 for ref in tight.backend._ref.values())
    # every request finished, so the whole pool must be back on the free queue
    assert len(tight.backend._free) == tight.backend.num_blocks
    assert len(set(tight.backend._free)) == len(tight.backend._free), "free queue has duplicates"


def test_prefix_disabled_changes_nothing(engine):
    """Flag off: behavior must be byte-identical to v2 paged scheduling."""
    plain = Scheduler(engine, max_batch_size=2, cache_backend="paged", block_size=4)
    a = greedy_request("Hello world from the cache", n=6)
    plain.submit(a)
    run_until_done(plain)
    assert "prefix" not in plain.backend.memory_stats()
    assert a.state is RequestState.FINISHED
