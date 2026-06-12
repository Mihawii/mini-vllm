"""Speculative decoding: greedy parity is the whole point."""

from types import SimpleNamespace

import pytest

from mini_vllm.engine.sampling import SamplingParams
from mini_vllm.engine.speculative import SpeculativeEngine


def greedy(n=16, **kw):
    return SamplingParams(max_new_tokens=n, temperature=0.0, **kw)


def test_greedy_parity_with_plain_decoding(engine):
    """With the draft equal to the target, every proposal must be accepted
    and the output must equal plain greedy decoding token for token. This is
    the lossless-ness guarantee made testable."""
    spec = SpeculativeEngine(target=engine, draft=engine, gamma=4)
    result, stats = spec.generate("The quick brown fox", greedy(16))
    plain = engine.generate("The quick brown fox", greedy(16))
    assert result.token_ids == plain.token_ids
    assert stats.acceptance_rate > 0.95
    assert stats.target_forwards < 16, "verification must batch several tokens per forward"


def test_budget_and_stats_consistency(engine):
    spec = SpeculativeEngine(target=engine, draft=engine, gamma=3)
    result, stats = spec.generate("Hello world", greedy(10))
    assert result.completion_tokens == len(result.token_ids) <= 10
    assert stats.proposed >= stats.accepted
    assert stats.rounds >= 1
    assert stats.draft_forwards >= stats.proposed  # one draft forward per proposal


def test_stop_string_parity(engine):
    base = engine.generate("Hello", greedy(10))
    if len(base.text) < 5:
        pytest.skip("tiny model output too short")
    stop = base.text[2:5]
    spec = SpeculativeEngine(target=engine, draft=engine, gamma=4)
    result, _ = spec.generate("Hello", greedy(10, stop=[stop]))
    expected = engine.generate("Hello", greedy(10, stop=[stop]))
    assert result.text == expected.text
    assert result.finish_reason == "stop"


def test_sampled_mode_runs(engine):
    spec = SpeculativeEngine(target=engine, draft=engine, gamma=3)
    params = SamplingParams(max_new_tokens=12, temperature=0.9, top_p=0.9, seed=11)
    result, stats = spec.generate("Once upon a time", params)
    assert result.completion_tokens > 0
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_tokenizer_mismatch_rejected(engine):
    class FakeTok:
        vocab_size = 999

        def encode(self, text):
            return [1, 2, 3]

    fake = SimpleNamespace(tokenizer=FakeTok(), lm=SimpleNamespace(name="fake"))
    with pytest.raises(ValueError, match="share a tokenizer"):
        SpeculativeEngine(target=engine, draft=fake, gamma=2)


def test_gamma_validation(engine):
    with pytest.raises(ValueError):
        SpeculativeEngine(target=engine, draft=engine, gamma=0)
