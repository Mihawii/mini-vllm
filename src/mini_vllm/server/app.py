"""FastAPI application factory.

The lifespan hook loads the model once and starts the scheduler thread; every
endpoint then talks to that single engine. Uvicorn workers must stay at 1
(one model, one scheduler); concurrency comes from continuous batching, not
from process replication.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from mini_vllm import __version__
from mini_vllm.config import EngineConfig

_DESCRIPTION = """
An educational LLM inference engine with a custom decoding loop, KV cache,
continuous batching, SSE streaming, and OpenAI-style endpoints.

Generation endpoints: `/v1/completions`, `/v1/chat/completions` (set
`"stream": true` for Server-Sent Events). Introspection: `/health`,
`/models`, `/metrics`, `/metrics/prometheus`, `/tokenize`, `/benchmark`.
UI: [/dashboard](/dashboard).
"""

STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"


def create_app(config: EngineConfig | None = None) -> FastAPI:
    config = config or EngineConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import transformers

        from mini_vllm.engine.generation import GenerationEngine
        from mini_vllm.engine.model_loader import load_model
        from mini_vllm.engine.scheduler import Scheduler, SchedulerLoop
        from mini_vllm.metrics import MetricsCollector
        from mini_vllm.utils.logging import get_logger

        log = get_logger("mini_vllm.server")
        transformers.logging.set_verbosity_error()

        log.info("Loading model %s ...", config.model_name)
        loaded = load_model(config.model_name, device=config.device, dtype=config.dtype)
        engine = GenerationEngine(loaded)
        collector = MetricsCollector()
        scheduler = Scheduler(
            engine, max_batch_size=config.max_batch_size, on_event=collector.on_event
        )
        loop = SchedulerLoop(scheduler)
        loop.start()
        log.info(
            "Ready: %s on %s (%s), max batch size %d",
            loaded.name,
            loaded.device_name,
            str(loaded.dtype).removeprefix("torch."),
            config.max_batch_size,
        )

        app.state.ctx = SimpleNamespace(
            config=config,
            loaded=loaded,
            engine=engine,
            scheduler=scheduler,
            loop=loop,
            collector=collector,
            started_at=time.time(),
        )
        yield
        loop.stop()

    app = FastAPI(
        title="mini-vLLM",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )

    from mini_vllm.server.routes import router

    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard/")

    return app
