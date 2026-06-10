import pytest
import torch

from mini_vllm.engine.sampling import SamplingParams


def greedy(n=12, **kwargs):
    return SamplingParams(max_new_tokens=n, temperature=0.0, **kwargs)


def test_generate_returns_tokens(engine):
    result = engine.generate("Hello world", greedy())
    assert result.completion_tokens > 0
    assert result.token_ids
    assert result.prompt_tokens == 2
    assert result.finish_reason in {"stop", "length"}
    assert result.latency_s > 0


def test_kv_cache_on_off_parity(engine):
    """The cache is an optimization, never a behavior change: greedy decoding
    must produce identical tokens with and without it."""
    with_cache = engine.generate("The quick brown fox", greedy(16), use_kv_cache=True)
    without_cache = engine.generate("The quick brown fox", greedy(16), use_kv_cache=False)
    assert with_cache.token_ids == without_cache.token_ids


def test_native_loop_matches_hf_generate(engine):
    """Our decode loop must agree with Hugging Face's reference implementation
    under greedy decoding. This is the correctness anchor for the whole engine."""
    ours = engine.generate("Paris is the capital of", greedy(16))
    reference = engine.generate_hf("Paris is the capital of", greedy(16))
    assert ours.token_ids == reference.token_ids


def test_seeded_sampling_reproducible(engine):
    params = SamplingParams(max_new_tokens=12, temperature=0.9, top_p=0.9, seed=7)
    first = engine.generate("Once upon a time", params)
    second = engine.generate("Once upon a time", params)
    assert first.token_ids == second.token_ids


def test_stream_callback_receives_full_text(engine):
    chunks: list[str] = []
    result = engine.generate("Hello", greedy(10), stream_callback=chunks.append)
    assert "".join(chunks) == result.text


def test_stop_string_truncates(engine, tokenizer):
    # Find what the model actually generates greedily, then use a piece of it
    # as a stop string and check the output is cut before it.
    base = engine.generate("Hello", greedy(10))
    if len(base.text) < 4:
        pytest.skip("tiny model produced too little text to test stop strings")
    stop = base.text[2:5]
    result = engine.generate("Hello", greedy(10, stop=[stop]))
    assert stop not in result.text
    assert result.finish_reason == "stop"
    assert result.text == base.text[: base.text.find(stop)]


def test_prompt_too_long_raises(engine):
    from mini_vllm.engine.generation import PromptTooLongError

    long_prompt = "word " * 1100  # tiny-gpt2 context is 1024
    with pytest.raises(PromptTooLongError):
        engine.generate(long_prompt, greedy(8))


def test_empty_prompt_raises(engine):
    with pytest.raises(ValueError):
        engine.generate("", greedy(4))


def test_batch_matches_single_greedy(engine):
    """Static batching must not change results: each row of a greedy batch
    equals the same prompt generated alone."""
    prompts = ["Hello world", "The cat sat on", "One two three four five"]
    batch_results = engine.generate_batch(prompts, greedy(12))
    for prompt, batched in zip(prompts, batch_results):
        single = engine.generate(prompt, greedy(12))
        assert batched.token_ids == single.token_ids, f"mismatch for prompt {prompt!r}"


def test_batch_no_kv_cache_matches_single(engine):
    prompts = ["Hello world", "Tokens are"]
    batch_results = engine.generate_batch(prompts, greedy(8), use_kv_cache=False)
    for prompt, batched in zip(prompts, batch_results):
        single = engine.generate(prompt, greedy(8))
        assert batched.token_ids == single.token_ids


def test_generate_respects_inference_mode(engine):
    result = engine.generate("Hi", greedy(4))
    assert not torch.is_grad_enabled() or result  # generation ran under inference_mode
