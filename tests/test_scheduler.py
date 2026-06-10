"""Scheduler behavior: admission, interleaving, retirement, streaming, errors."""

import pytest

from mini_vllm.engine.sampling import SamplingParams
from mini_vllm.engine.scheduler import Request, RequestState, Scheduler, SchedulerLoop


def make_request(prompt: str, n: int = 8, **kwargs) -> Request:
    return Request(prompt=prompt, params=SamplingParams(max_new_tokens=n, temperature=0.0, **kwargs))


def run_until_done(scheduler: Scheduler, limit: int = 500) -> int:
    ticks = 0
    while scheduler.has_work():
        scheduler.step()
        ticks += 1
        assert ticks < limit, "scheduler did not converge"
    return ticks


def test_all_requests_complete(engine):
    scheduler = Scheduler(engine, max_batch_size=4)
    requests = [make_request(p, n=6) for p in ["Hello", "The cat", "One two three"]]
    for r in requests:
        scheduler.submit(r)
    run_until_done(scheduler)
    for r in requests:
        assert r.state is RequestState.FINISHED
        assert 0 < len(r.generated) <= 6
        kind, payload = r.output_queue.queue[-1]
        assert kind == "done"
        assert payload.completion_tokens == len(r.generated)


def test_single_slot_scheduler_matches_single_generation(engine):
    """With max_batch_size=1 the scheduler degenerates to sequential decoding,
    so its greedy output must exactly match GenerationEngine.generate."""
    scheduler = Scheduler(engine, max_batch_size=1)
    req = make_request("The quick brown fox", n=10)
    scheduler.submit(req)
    run_until_done(scheduler)
    single = engine.generate("The quick brown fox", SamplingParams(max_new_tokens=10, temperature=0.0))
    assert req.generated == single.token_ids


def test_batched_scheduler_matches_single_generation(engine):
    """Greedy decoding must survive batching: rows see the same logits thanks
    to left-padding masks and per-row position ids."""
    prompts = ["Hello world", "A b c d e", "Tokens"]
    scheduler = Scheduler(engine, max_batch_size=4)
    requests = [make_request(p, n=8) for p in prompts]
    for r in requests:
        scheduler.submit(r)
    run_until_done(scheduler)
    for r, p in zip(requests, prompts):
        single = engine.generate(p, SamplingParams(max_new_tokens=8, temperature=0.0))
        assert r.generated == single.token_ids, f"batched output diverged for {p!r}"


def test_late_arrival_joins_running_batch(engine):
    """The continuous-batching property: a request submitted mid-flight joins
    the active batch instead of waiting for it to drain."""
    scheduler = Scheduler(engine, max_batch_size=4)
    first = make_request("Hello there friend", n=20)
    scheduler.submit(first)
    for _ in range(3):
        scheduler.step()
    late = make_request("Late arrival", n=4)
    scheduler.submit(late)
    stats = scheduler.step()
    # both are decoding in ONE batch while the first is still unfinished
    assert stats.active == 2
    assert first.state is RequestState.RUNNING
    run_until_done(scheduler)
    assert first.state is RequestState.FINISHED
    assert late.state is RequestState.FINISHED


def test_queue_respects_max_batch_size(engine):
    scheduler = Scheduler(engine, max_batch_size=2)
    requests = [make_request(f"prompt {i}", n=6) for i in range(5)]
    for r in requests:
        scheduler.submit(r)
    stats = scheduler.step()
    assert stats.active == 2
    assert stats.queue_depth == 3
    run_until_done(scheduler)
    assert all(r.state is RequestState.FINISHED for r in requests)


def test_finished_requests_leave_batch_early(engine):
    """A short request must retire while a long one keeps decoding."""
    scheduler = Scheduler(engine, max_batch_size=4)
    short = make_request("Hi", n=2)
    long = make_request("Hello", n=15)
    scheduler.submit(short)
    scheduler.submit(long)
    saw_solo_long = False
    while scheduler.has_work():
        stats = scheduler.step()
        if short.state is RequestState.FINISHED and long.state is RequestState.RUNNING:
            assert stats.active <= 1 or stats.tick <= 2
            saw_solo_long = True
    assert saw_solo_long
    assert len(short.generated) == 2
    assert len(long.generated) == 15


def test_streaming_deltas_concatenate_to_final_text(engine):
    scheduler = Scheduler(engine, max_batch_size=2)
    req = make_request("Hello world", n=8)
    scheduler.submit(req)
    run_until_done(scheduler)
    deltas = []
    result = None
    while not req.output_queue.empty():
        kind, payload = req.output_queue.get()
        if kind == "delta":
            deltas.append(payload)
        elif kind == "done":
            result = payload
    assert result is not None
    assert "".join(deltas) == result.text


def test_invalid_request_fails_fast(engine):
    scheduler = Scheduler(engine, max_batch_size=2)
    bad = make_request("", n=4)
    scheduler.submit(bad)
    assert bad.state is RequestState.FAILED
    assert bad.error
    kind, payload = bad.output_queue.get_nowait()
    assert kind == "error"
    assert not scheduler.has_work()


def test_metrics_collector_tracks_lifecycle(engine):
    from mini_vllm.metrics import MetricsCollector

    collector = MetricsCollector()
    scheduler = Scheduler(engine, max_batch_size=2, on_event=collector.on_event)
    for i in range(3):
        scheduler.submit(make_request(f"hello {i}", n=4))
    run_until_done(scheduler)
    snap = collector.snapshot()
    assert snap["requests"]["total"] == 3
    assert snap["requests"]["completed"] == 3
    assert snap["requests"]["active"] == 0
    assert snap["requests"]["queued"] == 0
    assert snap["tokens"]["generated"] == 12
    assert snap["latency_s"]["p50"] > 0


def test_scheduler_loop_thread_roundtrip(engine):
    """End-to-end through the daemon thread, the way the HTTP server uses it."""
    loop = SchedulerLoop(Scheduler(engine, max_batch_size=2))
    loop.start()
    try:
        req = make_request("Hello thread", n=5)
        loop.submit(req)
        kind, payload = req.output_queue.get(timeout=10)
        seen = [(kind, payload)]
        while seen[-1][0] == "delta":
            seen.append(req.output_queue.get(timeout=10))
        assert seen[-1][0] == "done"
        assert seen[-1][1].completion_tokens == 5
    finally:
        loop.stop()


def test_tick_stats_recorded(engine):
    scheduler = Scheduler(engine, max_batch_size=2)
    scheduler.submit(make_request("Hello", n=3))
    run_until_done(scheduler)
    assert scheduler.ticks
    assert scheduler.ticks[0].active == 1
    assert all(t.t_ms >= 0 for t in scheduler.ticks)


@pytest.mark.parametrize("n_requests,batch", [(6, 3)])
def test_stress_interleaving(engine, n_requests, batch):
    """Mixed lengths arriving in waves; everything must finish exactly once."""
    scheduler = Scheduler(engine, max_batch_size=batch)
    requests = [make_request(f"prompt number {i}", n=2 + (i % 5) * 3) for i in range(n_requests)]
    for r in requests[: n_requests // 2]:
        scheduler.submit(r)
    for _ in range(2):
        scheduler.step()
    for r in requests[n_requests // 2 :]:
        scheduler.submit(r)
    run_until_done(scheduler)
    for r in requests:
        assert r.state is RequestState.FINISHED
        assert len(r.generated) == r.params.max_new_tokens  # tiny-gpt2 never emits EOS
