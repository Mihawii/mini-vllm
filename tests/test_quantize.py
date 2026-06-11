"""Quantization experiment: conversion parity, int8 smoke, size reduction."""

import pytest

from mini_vllm.engine.generation import GenerationEngine
from mini_vllm.engine.model_loader import ModelLoadError, load_model
from mini_vllm.engine.quantize import convert_conv1d_to_linear, state_dict_bytes
from mini_vllm.engine.sampling import SamplingParams

TEST_MODEL = "sshleifer/tiny-gpt2"


def greedy(n=12):
    return SamplingParams(max_new_tokens=n, temperature=0.0)


def test_conv1d_to_linear_preserves_output():
    """The Conv1D -> Linear rewrite must be numerically identical; only the
    int8 step afterwards is allowed to change anything."""
    loaded = load_model(TEST_MODEL, device="cpu", dtype="float32")
    engine = GenerationEngine(loaded)
    before = engine.generate("The quick brown fox", greedy()).token_ids

    converted = convert_conv1d_to_linear(loaded.model)
    assert converted > 0, "tiny-gpt2 should contain Conv1D modules"
    after = engine.generate("The quick brown fox", greedy()).token_ids
    assert after == before


def test_int8_quantizes_and_generates():
    """Mechanism check on the tiny model. Note: tiny-gpt2 is 99% embedding
    table, which dynamic quantization does not touch, so its checkpoint can
    even GROW from per-module overhead; size reduction is asserted nowhere
    here and reported honestly by the benchmark on real models instead."""
    int8 = load_model(TEST_MODEL, device="cpu", dtype="float32", quantize="int8")
    assert int8.metadata()["quantization"] == "int8"

    quantized_modules = [
        m for m in int8.model.modules()
        if type(m).__module__.startswith("torch.ao.nn.quantized")
    ]
    assert quantized_modules, "no Linear layer was actually quantized"

    result = GenerationEngine(int8).generate("Hello world", greedy(8))
    assert result.completion_tokens > 0
    assert state_dict_bytes(int8.model) > 0


def test_quantize_rejects_non_cpu():
    with pytest.raises(ModelLoadError):
        load_model(TEST_MODEL, device="cpu", dtype="float32", quantize="int4")


def test_quantize_compare_scenario():
    from mini_vllm.benchmark.runner import run_quantize_compare

    report = run_quantize_compare(TEST_MODEL, requests=2, max_new_tokens=6)
    assert report["float32"]["throughput_tok_s"] > 0
    assert report["int8"]["throughput_tok_s"] > 0
    assert 0.0 <= report["token_agreement"] <= 1.0
    assert report["float32"]["checkpoint_mb"] > 0
    assert report["int8"]["checkpoint_mb"] > 0
