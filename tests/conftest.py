"""Shared fixtures.

All model-backed tests use sshleifer/tiny-gpt2 (about 100K parameters). It
produces meaningless text but exercises the exact same code paths as a real
model, downloads in seconds, and keeps the suite fast on a laptop CPU.
"""

from __future__ import annotations

import pytest

TEST_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def loaded_model():
    from mini_vllm.engine.model_loader import load_model

    return load_model(TEST_MODEL, device="cpu", dtype="float32")


@pytest.fixture(scope="session")
def engine(loaded_model):
    from mini_vllm.engine.generation import GenerationEngine

    return GenerationEngine(loaded_model)


@pytest.fixture(scope="session")
def tokenizer(loaded_model):
    return loaded_model.tokenizer
