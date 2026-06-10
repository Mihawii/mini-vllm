"""Paged KV cache backend, preemption, and chunked prefill."""

import pytest
import torch

from mini_vllm.engine.cache_backend import ContiguousBackend, PagedBackend, PoolExhausted
from mini_vllm.engine.sampling import SamplingParams
from mini_vllm.engine.scheduler import Request, RequestState, Scheduler


def greedy_request(prompt: str, n: int = 8) -> Request:
    return Request(prompt=prompt, params=SamplingParams(max_new_tokens=n, temperature=0.0))


def run_until_done(scheduler: Scheduler, limit: int = 2000) -> int:
    ticks = 0
    while scheduler.has_work():
        scheduler.step()
        ticks += 1
        assert ticks < limit, "scheduler did not converge"
    return ticks


def fake_kv(n_layers: int = 2, heads: int = 2, tokens: int = 5, dim: int = 4):
    return tuple(
        (torch.randn(1, heads, tokens, dim), torch.randn(1, heads, tokens, dim))
        for _ in range(n_layers)
    )


# ---------------------------------------------------------------------------
# backend unit tests (no model needed)
# ---------------------------------------------------------------------------


def test_paged_roundtrip_matches_contiguous():
    """Whatever goes in must come out identical from both backends."""
    kv_a, kv_b = fake_kv(tokens=5), fake_kv(tokens=9)
    step = fake_kv(tokens=1)

    paged = PagedBackend(block_size=4, num_blocks=16)
    contig = ContiguousBackend()
    for backend in (paged, contig):
        backend.add("a")
        backend.append("a", kv_a)
        backend.add("b")
        backend.append("b", kv_b)
        backend.append("a", step)

    p_cache, p_mask = paged.gather(["a", "b"])
    c_cache, c_mask = contig.gather(["a", "b"])
    assert torch.equal(p_mask, c_mask)
    for (pk, pv), (ck, cv) in zip(p_cache, c_cache):
        assert torch.equal(pk, ck)
        assert torch.equal(pv, cv)


def test_paged_block_accounting():
    backend = PagedBackend(block_size=4, num_blocks=8)
    backend.add("a")
    backend.append("a", fake_kv(tokens=5))  # 5 tokens -> 2 blocks
    assert backend.blocks_of("a") == 2
    assert backend.memory_stats()["used_blocks"] == 2
    backend.append("a", fake_kv(tokens=1))  # fits in block 2 (6/8)
    assert backend.blocks_of("a") == 2
    backend.drop("a")
    stats = backend.memory_stats()
    assert stats["used_blocks"] == 0
    assert stats["utilization"] == 0.0


def test_paged_pool_exhaustion_is_atomic():
    backend = PagedBackend(block_size=4, num_blocks=2)
    backend.add("a")
    backend.append("a", fake_kv(tokens=8))  # exactly fills the pool
    backend.add("b")
    with pytest.raises(PoolExhausted):
        backend.append("b", fake_kv(tokens=1))
    # the failed append must not have leaked partial state
    assert backend.seq_len("b") == 0
    backend.drop("a")
    backend.append("b", fake_kv(tokens=1))  # now it fits
    assert backend.seq_len("b") == 1


# ---------------------------------------------------------------------------
# scheduler integration (tiny model)
# ---------------------------------------------------------------------------


def test_paged_scheduler_matches_contiguous_greedy(engine):
    """The storage backend must never change what gets generated."""
    prompts = ["Hello world", "The cat sat on", "One two three four five"]

    def run(backend: str):
        scheduler = Scheduler(engine, max_batch_size=4, cache_backend=backend, block_size=4)
        requests = [greedy_request(p, n=10) for p in prompts]
        for r in requests:
            scheduler.submit(r)
        run_until_done(scheduler)
        return [r.generated for r in requests]

    assert run("paged") == run("contiguous")


def test_preemption_recovers_and_completes(engine):
    """A pool too small for the whole workload forces evictions; every
    request must still finish with exactly the unpreempted greedy output."""
    prompts = ["Hello world", "The quick brown fox", "Tokens are fun"]
    reference = [
        engine.generate(p, SamplingParams(max_new_tokens=12, temperature=0.0)).token_ids
        for p in prompts
    ]

    scheduler = Scheduler(
        engine, max_batch_size=4, cache_backend="paged", block_size=4, pool_blocks=10
    )
    requests = [greedy_request(p, n=12) for p in prompts]
    for r in requests:
        scheduler.submit(r)
    run_until_done(scheduler)

    assert scheduler.preemption_count > 0, "pool was sized to force preemption"
    for r, ref in zip(requests, reference):
        assert r.state is RequestState.FINISHED
        assert r.generated == ref, f"preemption changed output for {r.prompt!r}"
    assert sum(r.preemptions for r in requests) == scheduler.preemption_count


def test_pool_too_small_for_one_request_fails_clearly(engine):
    scheduler = Scheduler(
        engine, max_batch_size=2, cache_backend="paged", block_size=2, pool_blocks=2
    )
    request = greedy_request("This prompt is definitely longer than four tokens", n=8)
    scheduler.submit(request)
    while scheduler.has_work():
        scheduler.step()
    assert request.state is RequestState.FAILED
    assert "pool too small" in request.error.lower()


def test_chunked_prefill_matches_unchunked(engine):
    prompt = "word " * 40  # 40 prompt tokens, chunk size 8 -> 5 chunks
    reference = engine.generate(prompt, SamplingParams(max_new_tokens=8, temperature=0.0))

    scheduler = Scheduler(engine, max_batch_size=2, prefill_chunk_size=8)
    request = greedy_request(prompt, n=8)
    scheduler.submit(request)
    run_until_done(scheduler)
    assert request.generated == reference.token_ids


def test_chunked_prefill_does_not_stall_decode(engine):
    """While a long prompt prefills chunk by chunk, an already-running
    request must keep gaining tokens every tick."""
    scheduler = Scheduler(engine, max_batch_size=4, prefill_chunk_size=8)
    short = greedy_request("Hi there", n=30)
    scheduler.submit(short)
    scheduler.step()  # short is running
    tokens_before = len(short.generated)

    long = greedy_request("word " * 64, n=4)
    scheduler.submit(long)
    # 64 prompt tokens / 8 per chunk = 8 prefill ticks; decode must advance
    # in lockstep with them instead of waiting for the whole prompt.
    for _ in range(8):
        scheduler.step()
        if long.state is RequestState.RUNNING:
            break
        assert len(short.generated) > tokens_before
        tokens_before = len(short.generated)
    run_until_done(scheduler)
    assert short.state is RequestState.FINISHED
    assert long.state is RequestState.FINISHED
    assert len(long.generated) == 4


def test_chunked_prefill_with_paged_backend(engine):
    prompt = "many words " * 20
    reference = engine.generate(prompt, SamplingParams(max_new_tokens=6, temperature=0.0))
    scheduler = Scheduler(
        engine, max_batch_size=2, cache_backend="paged", block_size=4,
        pool_blocks=64, prefill_chunk_size=8,
    )
    request = greedy_request(prompt, n=6)
    scheduler.submit(request)
    run_until_done(scheduler)
    assert request.generated == reference.token_ids


def test_tick_stats_report_pool_state(engine):
    scheduler = Scheduler(engine, max_batch_size=2, cache_backend="paged", block_size=4)
    scheduler.submit(greedy_request("Hello", n=4))
    stats = scheduler.step()
    assert stats.pool_used_blocks > 0
    assert 0 < stats.pool_utilization <= 1
