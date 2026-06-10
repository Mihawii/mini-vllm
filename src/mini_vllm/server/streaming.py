"""Server-Sent Events bridge.

The scheduler thread pushes ("delta", text) items into each request's queue.
These helpers read that queue from asyncio handlers (via a worker thread, so
the event loop never blocks) and encode OpenAI-style streaming chunks:

    data: {"choices": [{"text": "..."}], ...}\n\n
    ...
    data: [DONE]\n\n
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from functools import partial

from mini_vllm.engine.scheduler import Request

STREAM_TIMEOUT_S = 300.0


def sse(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def completion_chunk(model: str, chunk_id: str) -> Callable[[str | None, str | None], dict]:
    def make(delta: str | None, finish_reason: str | None) -> dict:
        return {
            "id": chunk_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": delta or "", "finish_reason": finish_reason}],
        }

    return make


def chat_chunk(model: str, chunk_id: str) -> Callable[[str | None, str | None], dict]:
    first = True

    def make(delta: str | None, finish_reason: str | None) -> dict:
        nonlocal first
        content: dict = {}
        if first:
            content["role"] = "assistant"
            first = False
        if delta:
            content["content"] = delta
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": content, "finish_reason": finish_reason}],
        }

    return make


def new_chunk_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


async def stream_sse(
    request: Request,
    make_chunk: Callable[[str | None, str | None], dict],
    on_done: Callable[[Request], None] | None = None,
) -> AsyncIterator[str]:
    """Yield SSE strings for one request until it finishes or fails."""
    while True:
        try:
            kind, payload = await asyncio.to_thread(
                partial(request.output_queue.get, timeout=STREAM_TIMEOUT_S)
            )
        except Exception:
            yield sse({"error": {"message": "stream timed out waiting for the scheduler"}})
            yield sse("[DONE]")
            return

        if kind == "delta":
            yield sse(make_chunk(payload, None))
        elif kind == "done":
            yield sse(make_chunk(None, payload.finish_reason))
            if on_done is not None:
                on_done(request)
            yield sse("[DONE]")
            return
        else:  # error
            yield sse({"error": {"message": payload}})
            yield sse("[DONE]")
            return
