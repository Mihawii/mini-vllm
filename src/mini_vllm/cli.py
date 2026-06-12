"""mini-vLLM command line interface.

Commands are added per phase:
  inspect / tokenize / generate  - engine basics
  batch / simulate               - batching and the continuous-batching scheduler
  serve / stats                  - HTTP server and live metrics
  bench / bench-report           - benchmark suite
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mini_vllm.config import DEFAULT_MODEL, EngineConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = typer.Typer(
    name="mini-vllm",
    help="An educational LLM inference engine: custom decoding, KV cache, batching, streaming.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _quiet_transformers() -> None:
    import transformers

    transformers.logging.set_verbosity_error()


def _load_engine(model: str, device: str, dtype: str, quantize: str | None = None):
    """Load a model behind a spinner and wrap it in a GenerationEngine."""
    from mini_vllm.engine.generation import GenerationEngine
    from mini_vllm.engine.model_loader import ModelLoadError, load_model

    _quiet_transformers()
    try:
        with console.status(f"[bold cyan]Loading {model}..."):
            loaded = load_model(model, device=device, dtype=dtype, quantize=quantize)
    except (ModelLoadError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    return GenerationEngine(loaded)


def _sampling_params(
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop: list[str] | None,
    seed: int | None,
):
    from mini_vllm.engine.sampling import SamplingParams

    try:
        return SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=list(stop or []),
            seed=seed,
        )
    except ValueError as exc:
        console.print(f"[bold red]Invalid sampling parameters:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _stats_line(result) -> str:
    return (
        f"[dim]prompt[/] [bold]{result.prompt_tokens}[/] tok   "
        f"[dim]generated[/] [bold]{result.completion_tokens}[/] tok   "
        f"[dim]latency[/] [bold]{result.latency_s:.2f}s[/]   "
        f"[dim]speed[/] [bold]{result.tokens_per_second:.1f}[/] tok/s   "
        f"[dim]finish[/] [bold]{result.finish_reason}[/]"
    )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Hugging Face model id."),
    device: str = typer.Option("auto", help="auto | cpu | cuda | mps"),
    dtype: str = typer.Option("auto", help="auto | float32 | float16 | bfloat16"),
) -> None:
    """Load a model and print its metadata."""
    from mini_vllm.engine.model_loader import format_param_count

    engine = _load_engine(model, device, dtype)
    meta = engine.lm.metadata()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Model", meta["model"])
    table.add_row("Architecture", meta["architecture"])
    table.add_row("Parameters", f"{format_param_count(meta['parameters'])}  ({meta['parameters']:,})")
    table.add_row("Device", meta["device"])
    table.add_row("DType", meta["dtype"])
    table.add_row("Context length", str(meta["context_length"]))
    table.add_row("Vocab size", f"{meta['vocab_size']:,}")
    table.add_row("KV cache", "supported" if meta["kv_cache"] else "not supported")
    console.print(Panel(table, title="[bold cyan]Model Inspection", border_style="cyan", expand=False))


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


@app.command()
def tokenize(
    text: str = typer.Argument(..., help="Text to tokenize."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Tokenizer to use."),
) -> None:
    """Show how text splits into tokens (ids, BPE pieces, decoded fragments)."""
    from mini_vllm.engine.tokenizer import TokenizerWrapper

    _quiet_transformers()
    tok = TokenizerWrapper(model)
    infos = tok.breakdown(text)
    ids = [i.token_id for i in infos]
    roundtrip = tok.decode(ids, skip_special_tokens=False)

    table = Table(title=f"[bold cyan]{len(infos)} tokens[/] ({model})", border_style="dim")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Token ID", justify="right", style="cyan")
    table.add_column("BPE piece", style="magenta")
    table.add_column("Text", style="green")
    for info in infos:
        table.add_row(str(info.index), str(info.token_id), info.token, repr(info.text))
    console.print(table)
    console.print(f"[dim]Token IDs:[/] {ids}")
    ok = "[green]exact[/]" if roundtrip == text else "[yellow]lossy[/]"
    console.print(f"[dim]Roundtrip:[/] {roundtrip!r}  ({ok})")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt text."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", "-n"),
    temperature: float = typer.Option(0.8, help="0 = greedy decoding."),
    top_k: int = typer.Option(0, help="Keep only the k most likely tokens (0 disables)."),
    top_p: float = typer.Option(0.95, help="Nucleus sampling mass (1.0 disables)."),
    repetition_penalty: float = typer.Option(1.1),
    stop: list[str] = typer.Option(None, "--stop", help="Stop string (repeatable)."),
    seed: int = typer.Option(None, help="Seed for reproducible sampling."),
    stream: bool = typer.Option(False, "--stream", help="Print tokens as they are generated."),
    kv_cache: bool = typer.Option(True, "--kv-cache/--no-kv-cache", help="Toggle the KV cache."),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("auto"),
    quantize: str = typer.Option(None, help="int8 = dynamic int8 quantization (CPU only)."),
    draft_model: str = typer.Option(
        None, "--draft-model", help="Enable speculative decoding with this draft model."
    ),
    gamma: int = typer.Option(4, help="Draft tokens proposed per speculative round."),
    engine_mode: str = typer.Option(
        "native", "--engine", help="native = our decode loop, hf = model.generate() fallback."
    ),
) -> None:
    """Generate a completion with the custom decoding loop."""
    from mini_vllm.engine.generation import PromptTooLongError

    engine = _load_engine(model, device, dtype, quantize=quantize)
    params = _sampling_params(max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop, seed)

    if draft_model:
        from mini_vllm.engine.speculative import SpeculativeEngine

        draft = _load_engine(draft_model, device, dtype)
        try:
            spec = SpeculativeEngine(target=engine, draft=draft, gamma=gamma)
            result, spec_stats = spec.generate(prompt, params)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(Panel(result.text or "[dim](empty)", title="[bold]speculative completion", border_style="cyan"))
        console.print(_stats_line(result))
        console.print(
            f"[dim]draft[/] [bold]{draft_model}[/]   [dim]gamma[/] [bold]{gamma}[/]   "
            f"[dim]acceptance[/] [bold]{spec_stats.acceptance_rate:.0%}[/]   "
            f"[dim]target forwards[/] [bold]{spec_stats.target_forwards}[/] "
            f"[dim]for[/] [bold]{result.completion_tokens}[/] [dim]tokens[/]"
        )
        return

    try:
        if engine_mode == "hf":
            result = engine.generate_hf(prompt, params)
            console.print(Panel(result.text or "[dim](empty)", title="[bold]hf fallback output", border_style="yellow"))
        elif stream:
            console.print(f"[bold cyan]{prompt}[/]", end="")
            result = engine.generate(
                prompt,
                params,
                use_kv_cache=kv_cache,
                stream_callback=lambda delta: print(delta, end="", flush=True),
            )
            print()
        else:
            result = engine.generate(prompt, params, use_kv_cache=kv_cache)
            console.print(Panel(result.text or "[dim](empty)", title="[bold]completion", border_style="cyan"))
    except (PromptTooLongError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    cache_note = "" if engine_mode == "hf" else f"   [dim]kv-cache[/] [bold]{'on' if kv_cache else 'off'}[/]"
    console.print(_stats_line(result) + cache_note)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@app.command()
def batch(
    prompts_file: Path = typer.Argument(..., help="JSON file containing a list of prompts."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", "-n"),
    temperature: float = typer.Option(0.8),
    top_p: float = typer.Option(0.95),
    seed: int = typer.Option(None),
    kv_cache: bool = typer.Option(True, "--kv-cache/--no-kv-cache"),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("auto"),
) -> None:
    """Generate completions for several prompts in one padded batch."""
    if not prompts_file.exists():
        console.print(f"[bold red]Error:[/] file not found: {prompts_file}")
        raise typer.Exit(code=1)
    prompts = json.loads(prompts_file.read_text())
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        console.print("[bold red]Error:[/] expected a JSON list of prompt strings.")
        raise typer.Exit(code=1)

    engine = _load_engine(model, device, dtype)
    params = _sampling_params(max_new_tokens, temperature, 0, top_p, 1.1, None, seed)

    from mini_vllm.utils.timing import Timer

    with Timer() as t:
        results = engine.generate_batch(prompts, params, use_kv_cache=kv_cache)

    total_tokens = sum(r.completion_tokens for r in results)
    for prompt, result in zip(prompts, results):
        console.print(Panel(
            f"[dim]{prompt}[/]\n\n{result.text or '[dim](empty)'}",
            title=f"[bold]{result.completion_tokens} tok  |  finish: {result.finish_reason}",
            border_style="cyan",
        ))
    console.print(
        f"[bold]Batch of {len(prompts)}[/]   latency [bold]{t.elapsed:.2f}s[/]   "
        f"total [bold]{total_tokens}[/] tok   throughput [bold]{total_tokens / t.elapsed:.1f}[/] tok/s"
    )


# ---------------------------------------------------------------------------
# simulate (continuous batching demo)
# ---------------------------------------------------------------------------


@app.command()
def simulate(
    traffic_file: Path = typer.Argument(..., help="JSON list of {at_ms, prompt, max_new_tokens}."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    max_batch_size: int = typer.Option(4, help="Scheduler slot count."),
    temperature: float = typer.Option(0.8),
    seed: int = typer.Option(0, help="Seed shared by all simulated requests."),
    cache_backend: str = typer.Option("contiguous", help="KV storage: contiguous | paged."),
    block_size: int = typer.Option(16, help="Tokens per block (paged backend)."),
    pool_blocks: int = typer.Option(256, help="Total blocks in the pool (paged backend)."),
    prefill_chunk_size: int = typer.Option(0, help="Prefill chunk in tokens (0 = off)."),
    prefix_caching: bool = typer.Option(False, "--prefix-caching/--no-prefix-caching", help="Reuse KV blocks across shared prefixes (paged backend)."),
    out_dir: Path = typer.Option(Path("logs/simulations"), help="Where JSONL metrics land."),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("auto"),
) -> None:
    """Replay a traffic file through the continuous batching scheduler.

    Requests arrive on a wall-clock timeline while earlier ones are still
    decoding; the scheduler grows and shrinks the live batch every tick.
    """
    import time as _time

    from mini_vllm.engine.scheduler import Request, Scheduler
    from mini_vllm.metrics.storage import timestamp_slug, write_jsonl

    if not traffic_file.exists():
        console.print(f"[bold red]Error:[/] file not found: {traffic_file}")
        raise typer.Exit(code=1)
    spec = json.loads(traffic_file.read_text())

    engine = _load_engine(model, device, dtype)
    scheduler = Scheduler(
        engine,
        max_batch_size=max_batch_size,
        cache_backend=cache_backend,
        block_size=block_size,
        pool_blocks=pool_blocks,
        prefill_chunk_size=prefill_chunk_size,
        enable_prefix_caching=prefix_caching,
    )

    pending: list[tuple[float, Request]] = []
    for i, item in enumerate(spec):
        params = _sampling_params(
            item.get("max_new_tokens", 48), item.get("temperature", temperature),
            0, item.get("top_p", 0.95), 1.1, None, seed,
        )
        req = Request(prompt=item["prompt"], params=params, request_id=f"req-{i}")
        pending.append((float(item.get("at_ms", 0)), req))
    pending.sort(key=lambda pair: pair[0])
    arrivals = {req.request_id: at for at, req in pending}
    requests = [req for _, req in pending]

    console.print(
        f"[bold cyan]Simulating[/] {len(pending)} requests, max batch size {max_batch_size}"
    )
    t0 = _time.perf_counter()
    while pending or scheduler.has_work():
        now_ms = (_time.perf_counter() - t0) * 1000.0
        while pending and pending[0][0] <= now_ms:
            _, req = pending.pop(0)
            scheduler.submit(req)
        if scheduler.has_work():
            scheduler.step()
        elif pending:
            _time.sleep(min(0.002, max(pending[0][0] - now_ms, 0.1) / 1000.0))
    makespan_ms = (_time.perf_counter() - t0) * 1000.0

    # ---- per-request table ----
    table = Table(title="[bold cyan]Request timeline", border_style="dim")
    for col, justify in [
        ("id", "left"), ("arrive ms", "right"), ("queue ms", "right"), ("ttft ms", "right"),
        ("latency ms", "right"), ("tok", "right"), ("tok/s", "right"), ("finish", "left"), ("output", "left"),
    ]:
        table.add_column(col, justify=justify)
    for req in requests:
        res = req.to_result()
        snippet = (res.text[:42] + "...") if len(res.text) > 45 else res.text
        table.add_row(
            req.request_id,
            f"{arrivals[req.request_id]:.0f}",
            f"{req.queue_time_s * 1000:.0f}",
            f"{(req.ttft_s or 0) * 1000:.0f}",
            f"{res.latency_s * 1000:.0f}",
            str(res.completion_tokens),
            f"{res.tokens_per_second:.1f}",
            res.finish_reason if req.error is None else f"[red]{req.error[:30]}",
            snippet.replace("\n", " "),
        )
    console.print(table)

    # ---- batch occupancy timeline ----
    ticks = scheduler.ticks
    paged = cache_backend == "paged"
    shown = ticks if len(ticks) <= 28 else [ticks[i] for i in range(0, len(ticks), len(ticks) // 28 + 1)]
    timeline = Table(title="[bold cyan]Scheduler ticks  (█ active  ░ queued)", border_style="dim")
    timeline.add_column("t (ms)", justify="right")
    timeline.add_column("active", justify="right")
    timeline.add_column("queue", justify="right")
    timeline.add_column("cache len", justify="right")
    if paged:
        timeline.add_column("pool blocks", justify="right")
    timeline.add_column("batch")
    for tk in shown:
        row = [
            f"{tk.t_ms:.0f}", str(tk.active), str(tk.queue_depth), str(tk.cache_len),
        ]
        if paged:
            row.append(f"[magenta]{tk.pool_used_blocks}[/] ({tk.pool_utilization:.0%})")
        row.append("[green]" + "█" * tk.active + "[/][yellow]" + "░" * tk.queue_depth)
        timeline.add_row(*row)
    console.print(timeline)

    total_tokens = sum(len(r.generated) for r in requests)
    busy = [tk.active for tk in ticks if tk.active > 0]
    avg_active = sum(busy) / len(busy) if busy else 0.0
    pool = scheduler.backend.memory_stats()
    extra = f"   [bold]preemptions[/] {scheduler.preemption_count}" if scheduler.preemption_count else ""
    if pool.get("backend") == "paged":
        peak = max((tk.pool_used_blocks for tk in ticks), default=0)
        extra += (
            f"   [bold]pool peak[/] {peak}/{pool['num_blocks']} blocks "
            f"({pool['block_size']} tok each)"
        )
        prefix = pool.get("prefix")
        if prefix and prefix["hit_tokens"]:
            extra += f"   [bold]prefix hits[/] {prefix['hit_tokens']} tok reused"
    console.print(
        f"[bold]makespan[/] {makespan_ms:.0f} ms   [bold]tokens[/] {total_tokens}   "
        f"[bold]throughput[/] {total_tokens / (makespan_ms / 1000):.1f} tok/s   "
        f"[bold]avg active batch[/] {avg_active:.2f}   [bold]ticks[/] {len(ticks)}" + extra
    )

    # ---- persist metrics ----
    slug = timestamp_slug()
    ticks_path = out_dir / f"sim-{slug}-ticks.jsonl"
    write_jsonl(ticks_path, [tk.to_dict() for tk in ticks])
    summary = {
        "model": model,
        "max_batch_size": max_batch_size,
        "makespan_ms": round(makespan_ms, 1),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / (makespan_ms / 1000), 2),
        "avg_active_batch": round(avg_active, 2),
        "preemptions": scheduler.preemption_count,
        "prefill_chunk_size": prefill_chunk_size,
        "pool": pool,
        "requests": [
            {
                "id": req.request_id,
                "prompt": req.prompt,
                "arrive_ms": arrivals[req.request_id],
                "queue_ms": round(req.queue_time_s * 1000, 1),
                "ttft_ms": round((req.ttft_s or 0) * 1000, 1),
                "latency_ms": round(req.to_result().latency_s * 1000, 1),
                "tokens": len(req.generated),
                "finish_reason": req.finish_reason,
                "text": req.tracker.text if req.tracker else "",
            }
            for req in requests
        ],
        "ticks": [tk.to_dict() for tk in ticks],
    }
    summary_path = out_dir / f"sim-{slug}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"[dim]Saved tick metrics to {ticks_path} and summary to {summary_path}[/]")


# ---------------------------------------------------------------------------
# bench / bench-report
# ---------------------------------------------------------------------------


@app.command()
def bench(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    requests: int = typer.Option(10, help="Requests per scenario."),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", "-n"),
    concurrency: int = typer.Option(4, help="Scheduler slots for the continuous batching test."),
    compare_kv_cache: bool = typer.Option(
        True, "--compare-kv-cache/--no-compare-kv-cache", help="Measure cache on vs off."
    ),
    compare_quantize: bool = typer.Option(
        False, "--compare-quantize", help="Also measure dynamic int8 vs fp32 (CPU only)."
    ),
    baselines: bool = typer.Option(
        False, "--baselines", help="Compare HF generate(), no-cache, cached, and batched."
    ),
    speculative_draft: str = typer.Option(
        None, "--speculative-draft", help="Measure speculative decoding with this draft model."
    ),
    gamma: int = typer.Option(4, help="Draft tokens per speculative round."),
    batch_sizes: str = typer.Option("1,2,4,8", help="Comma-separated static batch sizes."),
    prompt_lengths: str = typer.Option(
        "", help="Comma-separated prompt token counts for the prefill sweep (empty = skip)."
    ),
    out_dir: Path = typer.Option(Path("benchmarks/results")),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("auto"),
) -> None:
    """Run the benchmark suite and save JSON + CSV results."""
    from mini_vllm.benchmark.report import render_markdown
    from mini_vllm.benchmark.runner import run_benchmark, save_results

    engine = _load_engine(model, device, dtype)
    sizes = [int(s) for s in batch_sizes.split(",") if s.strip()]
    lengths = [int(s) for s in prompt_lengths.split(",") if s.strip()]

    with console.status("[bold cyan]Benchmarking...") as status:
        report = run_benchmark(
            engine,
            requests=requests,
            max_new_tokens=max_new_tokens,
            concurrency=concurrency,
            compare_kv_cache=compare_kv_cache,
            batch_sizes=sizes,
            prompt_lengths=lengths or None,
            compare_quantize=compare_quantize,
            baselines=baselines,
            speculative_draft=speculative_draft,
            gamma=gamma,
            on_progress=lambda label: status.update(f"[bold cyan]Benchmarking: {label}..."),
        )
    json_path, csv_path = save_results(report, out_dir)

    from rich.markdown import Markdown

    console.print(Markdown(render_markdown(report)))
    console.print(f"[dim]Saved {json_path} and {csv_path}[/]")


@app.command(name="bench-report")
def bench_report(
    results_dir: Path = typer.Option(Path("benchmarks/results")),
    write: bool = typer.Option(True, "--write/--no-write", help="Also write report.md."),
) -> None:
    """Render the most recent benchmark run as a Markdown report."""
    from rich.markdown import Markdown

    from mini_vllm.benchmark.report import latest_report, render_markdown

    report = latest_report(results_dir)
    if report is None:
        console.print(
            f"[bold red]Error:[/] no benchmark results in {results_dir}. Run `mini-vllm bench` first."
        )
        raise typer.Exit(code=1)
    markdown = render_markdown(report)
    console.print(Markdown(markdown))
    if write:
        out = results_dir / "report.md"
        out.write_text(markdown)
        console.print(f"[dim]Wrote {out}[/]")


# ---------------------------------------------------------------------------
# serve / stats
# ---------------------------------------------------------------------------


@app.command()
def serve(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    host: str = typer.Option("127.0.0.1", help="Bind address (0.0.0.0 to expose)."),
    port: int = typer.Option(8000),
    max_batch_size: int = typer.Option(8, help="Continuous batching slot count."),
    cache_backend: str = typer.Option("contiguous", help="KV storage: contiguous | paged."),
    block_size: int = typer.Option(16, help="Tokens per block (paged backend)."),
    pool_blocks: int = typer.Option(256, help="Total blocks in the pool (paged backend)."),
    prefill_chunk_size: int = typer.Option(0, help="Prefill chunk in tokens (0 = whole prompt at once)."),
    prefix_caching: bool = typer.Option(False, "--prefix-caching/--no-prefix-caching", help="Reuse KV blocks across shared prompt prefixes (paged backend)."),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("auto"),
) -> None:
    """Start the OpenAI-compatible HTTP server with the dashboard."""
    import uvicorn

    from mini_vllm.server.app import create_app

    config = EngineConfig(
        model_name=model,
        device=device,
        dtype=dtype,
        host=host,
        port=port,
        max_batch_size=max_batch_size,
        cache_backend=cache_backend,
        block_size=block_size,
        pool_blocks=pool_blocks,
        prefill_chunk_size=prefill_chunk_size,
        prefix_caching=prefix_caching,
    )
    console.print(
        f"[bold cyan]mini-vLLM[/] serving [bold]{model}[/] at "
        f"[bold]http://{host}:{port}[/]  (docs: /docs, dashboard: /dashboard)"
    )
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


@app.command()
def stats(
    url: str = typer.Option("http://127.0.0.1:8000", help="Base URL of a running server."),
) -> None:
    """Fetch /metrics from a running server and render them."""
    import httpx

    try:
        snap = httpx.get(f"{url}/metrics", timeout=5.0).raise_for_status().json()
    except httpx.HTTPError as exc:
        console.print(f"[bold red]Error:[/] could not reach {url}/metrics ({exc})")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"[bold cyan]mini-vLLM stats[/]  {snap.get('model', '')}", border_style="dim")
    table.add_column("metric", style="dim")
    table.add_column("value", style="bold", justify="right")
    table.add_row("device", str(snap.get("device")))
    table.add_row("uptime", f"{snap.get('uptime_s', 0):.0f}s")
    requests = snap.get("requests", {})
    for key in ("total", "queued", "active", "completed", "failed"):
        table.add_row(f"requests {key}", str(requests.get(key, 0)))
    tokens = snap.get("tokens", {})
    table.add_row("prompt tokens", f"{tokens.get('prompt', 0):,}")
    table.add_row("generated tokens", f"{tokens.get('generated', 0):,}")
    table.add_row("tokens/s (lifetime)", str(tokens.get("per_second_lifetime", 0)))
    latency = snap.get("latency_s", {})
    table.add_row("latency avg", f"{latency.get('avg', 0) * 1000:.0f} ms")
    table.add_row("latency p50", f"{latency.get('p50', 0) * 1000:.0f} ms")
    table.add_row("latency p95", f"{latency.get('p95', 0) * 1000:.0f} ms")
    ttft = snap.get("ttft_s", {})
    table.add_row("ttft p50", f"{ttft.get('p50', 0) * 1000:.0f} ms")
    scheduler = snap.get("scheduler", {})
    table.add_row("batch (active/max)", f"{scheduler.get('active', 0)}/{scheduler.get('max_batch_size', 0)}")
    table.add_row("scheduler ticks", str(scheduler.get("ticks", 0)))
    console.print(table)


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":
    sys.exit(app())
