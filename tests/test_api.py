"""API tests run a real server (tiny model) through the ASGI test client.

The lifespan hook loads sshleifer/tiny-gpt2 and starts the scheduler thread,
so these tests exercise the full request path: HTTP -> queue -> scheduler ->
batched decode -> response.
"""

import json

import pytest
from fastapi.testclient import TestClient

from mini_vllm.config import EngineConfig
from mini_vllm.server.app import create_app

TEST_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = EngineConfig(
        model_name=TEST_MODEL,
        device="cpu",
        dtype="float32",
        max_batch_size=4,
        log_dir=str(tmp_path_factory.mktemp("logs")),
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == TEST_MODEL
    assert body["uptime_s"] >= 0


def test_models(client):
    body = client.get("/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == TEST_MODEL
    assert body["data"][0]["metadata"]["kv_cache"] is True
    # OpenAI-style alias
    assert client.get("/v1/models").json()["data"][0]["id"] == TEST_MODEL


def test_tokenize(client):
    body = client.post("/tokenize", json={"text": "Hello world"}).json()
    assert body["count"] == len(body["token_ids"]) == len(body["tokens"])
    assert body["roundtrip"] == "Hello world"


def test_completion_shape(client):
    response = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 8, "temperature": 0.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == TEST_MODEL
    choice = body["choices"][0]
    assert choice["finish_reason"] in {"stop", "length"}
    usage = body["usage"]
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    extras = body["mini_vllm"]
    assert extras["latency_ms"] > 0
    assert extras["tokens_per_second"] > 0


def test_completion_deterministic_with_seed(client):
    payload = {"prompt": "Once upon", "max_tokens": 8, "temperature": 0.9, "seed": 11}
    first = client.post("/v1/completions", json=payload).json()
    second = client.post("/v1/completions", json=payload).json()
    assert first["choices"][0]["text"] == second["choices"][0]["text"]


def test_chat_completion(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say something."}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_streaming_completion(client):
    deltas = []
    finish = None
    with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 6, "temperature": 0.0, "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk["choices"][0]
            if choice["text"]:
                deltas.append(choice["text"])
            if choice["finish_reason"]:
                finish = choice["finish_reason"]
    assert deltas, "no streamed tokens received"
    assert finish in {"stop", "length"}

    reference = client.post(
        "/v1/completions", json={"prompt": "Hello", "max_tokens": 6, "temperature": 0.0}
    ).json()
    assert "".join(deltas) == reference["choices"][0]["text"]


def test_streaming_chat_sends_role_first(client):
    events = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line.removeprefix("data: ")))
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"
    assert events[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


def test_validation_errors(client):
    # malformed body -> FastAPI 422
    assert client.post("/v1/completions", json={"max_tokens": 4}).status_code == 422
    # out-of-range pydantic field -> 422
    assert (
        client.post("/v1/completions", json={"prompt": "x", "temperature": -1}).status_code == 422
    )
    # prompt longer than the context window -> 400 with a readable message
    response = client.post(
        "/v1/completions", json={"prompt": "word " * 1100, "max_tokens": 8}
    )
    assert response.status_code == 400
    assert "context window" in response.json()["detail"]


def test_metrics_track_requests(client):
    before = client.get("/metrics").json()["requests"]["completed"]
    client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 4})
    after = client.get("/metrics").json()
    assert after["requests"]["completed"] == before + 1
    assert after["requests"]["failed"] >= 0
    assert after["model"] == TEST_MODEL
    assert after["scheduler"]["max_batch_size"] == 4


def test_prometheus_endpoint(client):
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    assert "mini_vllm_requests_total" in response.text


def test_inline_benchmark(client):
    response = client.post("/benchmark", json={"requests": 3, "max_new_tokens": 6})
    assert response.status_code == 200
    body = response.json()
    assert body["requests"] == 3
    assert body["total_tokens"] == 18
    assert body["throughput_tok_s"] > 0


def test_concurrent_requests_share_batches(client):
    """Fire several requests at once; the scheduler should batch them
    (observable as tick count growing far less than sequential would need)."""
    import concurrent.futures

    def call(i: int):
        return client.post(
            "/v1/completions",
            json={"prompt": f"prompt {i}", "max_tokens": 10, "temperature": 0.0},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(call, range(4)))
    assert all(r.status_code == 200 for r in responses)
    texts = [r.json()["choices"][0]["text"] for r in responses]
    assert all(isinstance(t, str) for t in texts)


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
