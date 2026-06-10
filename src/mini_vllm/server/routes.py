"""HTTP endpoints.

Every generation request, streamed or not, flows through the same
SchedulerLoop. Two API calls arriving together therefore decode in one
batch; that is the continuous batching path, not a special case.
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import Request as HttpRequest
from fastapi.responses import PlainTextResponse, StreamingResponse

from mini_vllm.engine.scheduler import Request
from mini_vllm.metrics.collector import percentile
from mini_vllm.metrics.storage import append_jsonl, latest_file
from mini_vllm.server import schemas
from mini_vllm.server.streaming import (
    chat_chunk,
    completion_chunk,
    new_chunk_id,
    sse,
    stream_sse,
)

router = APIRouter()

REQUEST_TIMEOUT_S = 300.0


def _ctx(http: HttpRequest):
    return http.app.state.ctx


def _log_request(ctx, endpoint: str, request: Request) -> None:
    result = request.to_result()
    append_jsonl(
        Path(ctx.config.log_dir) / "requests.jsonl",
        {
            "ts": time.time(),
            "endpoint": endpoint,
            "request_id": request.request_id,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": round(result.latency_s * 1000, 1),
            "ttft_ms": round((request.ttft_s or 0) * 1000, 1),
            "finish_reason": result.finish_reason,
        },
    )


async def _submit_and_wait(ctx, prompt: str, params) -> Request:
    """Send one request through the scheduler and wait for completion."""
    request = Request(prompt=prompt, params=params)
    ctx.loop.submit(request)
    while True:
        try:
            kind, payload = await asyncio.to_thread(
                partial(request.output_queue.get, timeout=REQUEST_TIMEOUT_S)
            )
        except Exception as exc:
            raise HTTPException(status_code=504, detail="timed out waiting for the scheduler") from exc
        if kind == "done":
            return request
        if kind == "error":
            # failures before the first decode step are caller errors
            # (empty prompt, context overflow); later ones are server faults.
            status = 400 if request.started_at == 0.0 else 500
            raise HTTPException(status_code=status, detail=payload)
        # kind == "delta": non-streaming callers just wait for the result


def _usage(result) -> schemas.Usage:
    return schemas.Usage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )


def _extras(request: Request) -> schemas.MiniVllmExtras:
    result = request.to_result()
    return schemas.MiniVllmExtras(
        latency_ms=round(result.latency_s * 1000, 1),
        ttft_ms=round((request.ttft_s or 0) * 1000, 1) if request.ttft_s else None,
        queue_ms=round(request.queue_time_s * 1000, 1),
        tokens_per_second=round(result.tokens_per_second, 2),
    )


# ---------------------------------------------------------------------------
# health / models / metrics
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(http: HttpRequest) -> dict:
    ctx = _ctx(http)
    return {
        "status": "ok",
        "model": ctx.loaded.name,
        "device": ctx.loaded.device_name,
        "uptime_s": round(time.time() - ctx.started_at, 1),
    }


@router.get("/models", response_model=schemas.ModelList)
@router.get("/v1/models", response_model=schemas.ModelList)
async def models(http: HttpRequest) -> schemas.ModelList:
    ctx = _ctx(http)
    return schemas.ModelList(
        data=[schemas.ModelCard(id=ctx.loaded.name, metadata=ctx.loaded.metadata())]
    )


@router.get("/metrics")
async def metrics(http: HttpRequest) -> dict:
    ctx = _ctx(http)
    snap = ctx.collector.snapshot()
    snap["model"] = ctx.loaded.name
    snap["device"] = ctx.loaded.device_name
    snap["scheduler"] = {
        "max_batch_size": ctx.scheduler.max_batch_size,
        "active": len(ctx.scheduler.active),
        "queue_depth": ctx.scheduler.queue_depth(),
        "ticks": len(ctx.scheduler.ticks),
    }
    return snap


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def metrics_prometheus(http: HttpRequest) -> str:
    return _ctx(http).collector.prometheus()


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


@router.post("/tokenize", response_model=schemas.TokenizeResponse)
async def tokenize(http: HttpRequest, body: schemas.TokenizeRequest) -> schemas.TokenizeResponse:
    tokenizer = _ctx(http).loaded.tokenizer
    infos = tokenizer.breakdown(body.text)
    ids = [i.token_id for i in infos]
    return schemas.TokenizeResponse(
        count=len(ids),
        token_ids=ids,
        tokens=[schemas.TokenDetail(**vars(i)) for i in infos],
        roundtrip=tokenizer.decode(ids, skip_special_tokens=False),
    )


# ---------------------------------------------------------------------------
# completions
# ---------------------------------------------------------------------------


@router.post("/v1/completions")
async def completions(http: HttpRequest, body: schemas.CompletionRequest):
    ctx = _ctx(http)
    try:
        params = body.to_sampling_params()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.stream:
        request = Request(prompt=body.prompt, params=params)
        ctx.loop.submit(request)
        chunks = completion_chunk(ctx.loaded.name, new_chunk_id("cmpl"))
        return StreamingResponse(
            stream_sse(request, chunks, on_done=lambda r: _log_request(ctx, "/v1/completions", r)),
            media_type="text/event-stream",
        )

    request = await _submit_and_wait(ctx, body.prompt, params)
    result = request.to_result()
    _log_request(ctx, "/v1/completions", request)
    return schemas.CompletionResponse(
        model=ctx.loaded.name,
        choices=[schemas.CompletionChoice(text=result.text, finish_reason=result.finish_reason)],
        usage=_usage(result),
        mini_vllm=_extras(request),
    )


@router.post("/v1/chat/completions")
async def chat_completions(http: HttpRequest, body: schemas.ChatCompletionRequest):
    ctx = _ctx(http)
    try:
        params = body.to_sampling_params()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prompt = ctx.loaded.tokenizer.apply_chat_template([m.model_dump() for m in body.messages])

    if body.stream:
        request = Request(prompt=prompt, params=params)
        ctx.loop.submit(request)
        chunks = chat_chunk(ctx.loaded.name, new_chunk_id("chatcmpl"))
        return StreamingResponse(
            stream_sse(request, chunks, on_done=lambda r: _log_request(ctx, "/v1/chat/completions", r)),
            media_type="text/event-stream",
        )

    request = await _submit_and_wait(ctx, prompt, params)
    result = request.to_result()
    _log_request(ctx, "/v1/chat/completions", request)
    return schemas.ChatCompletionResponse(
        model=ctx.loaded.name,
        choices=[
            schemas.ChatChoice(
                message=schemas.ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=_usage(result),
        mini_vllm=_extras(request),
    )


# ---------------------------------------------------------------------------
# inline benchmark
# ---------------------------------------------------------------------------


@router.post("/benchmark", response_model=schemas.BenchmarkResponse)
async def benchmark(http: HttpRequest, body: schemas.BenchmarkRequest) -> schemas.BenchmarkResponse:
    """Small capped load test: N concurrent requests through the scheduler."""
    ctx = _ctx(http)
    params = schemas._CommonParams(
        max_tokens=body.max_new_tokens, temperature=0.8, seed=0
    ).to_sampling_params()

    start = time.perf_counter()
    waits = [_submit_and_wait(ctx, body.prompt, params) for _ in range(body.requests)]
    finished = await asyncio.gather(*waits)
    total = time.perf_counter() - start

    latencies = sorted(r.to_result().latency_s * 1000 for r in finished)
    tokens = sum(r.to_result().completion_tokens for r in finished)
    return schemas.BenchmarkResponse(
        requests=body.requests,
        max_new_tokens=body.max_new_tokens,
        total_time_s=round(total, 3),
        total_tokens=tokens,
        throughput_tok_s=round(tokens / total, 2) if total > 0 else 0.0,
        latency_ms_avg=round(sum(latencies) / len(latencies), 1),
        latency_ms_p95=round(percentile(latencies, 0.95), 1),
    )


# ---------------------------------------------------------------------------
# dashboard data sources
# ---------------------------------------------------------------------------


@router.get("/benchmark/results")
async def benchmark_results() -> list[dict]:
    """Saved benchmark runs (newest first) for the dashboard."""
    results_dir = Path("benchmarks/results")
    if not results_dir.exists():
        return []
    payload = []
    for path in sorted(results_dir.glob("bench-*.json"), reverse=True)[:10]:
        try:
            payload.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return payload


@router.get("/simulations/latest")
async def simulations_latest() -> dict:
    path = latest_file("logs/simulations", "sim-*.json")
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No simulation found. Run: mini-vllm simulate examples/traffic.json",
        )
    return json.loads(path.read_text())


__all__ = ["router", "sse"]
