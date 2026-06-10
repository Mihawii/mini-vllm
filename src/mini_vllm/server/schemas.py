"""Pydantic schemas for the HTTP API.

The /v1 endpoints follow the OpenAI request/response shapes closely enough
that standard clients work, plus a `mini_vllm` extension block with timing
detail. Divergences are documented in docs/api.md.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from mini_vllm.engine.sampling import SamplingParams


class _CommonParams(BaseModel):
    model: str | None = Field(None, description="Informational; one model is loaded per server.")
    max_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=4.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0, description="Extension: 0 disables top-k.")
    repetition_penalty: float = Field(1.1, gt=0.0, description="Extension.")
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False

    def to_sampling_params(self) -> SamplingParams:
        stop = self.stop if isinstance(self.stop, list) else ([self.stop] if self.stop else [])
        return SamplingParams(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=stop,
            seed=self.seed,
        )


class CompletionRequest(_CommonParams):
    prompt: str = Field(..., min_length=1)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_CommonParams):
    messages: list[ChatMessage] = Field(..., min_length=1)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MiniVllmExtras(BaseModel):
    """Timing detail the OpenAI schema has no place for."""

    latency_ms: float
    ttft_ms: float | None = None
    queue_ms: float | None = None
    tokens_per_second: float


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage
    mini_vllm: MiniVllmExtras


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage
    mini_vllm: MiniVllmExtras


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TokenDetail(BaseModel):
    index: int
    token_id: int
    token: str
    text: str


class TokenizeResponse(BaseModel):
    count: int
    token_ids: list[int]
    tokens: list[TokenDetail]
    roundtrip: str


class BenchmarkRequest(BaseModel):
    prompt: str = "Explain what an inference engine does."
    requests: int = Field(4, ge=1, le=16)
    max_new_tokens: int = Field(32, ge=1, le=256)


class BenchmarkResponse(BaseModel):
    requests: int
    max_new_tokens: int
    total_time_s: float
    total_tokens: int
    throughput_tok_s: float
    latency_ms_avg: float
    latency_ms_p95: float


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-vllm"
    metadata: dict


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
