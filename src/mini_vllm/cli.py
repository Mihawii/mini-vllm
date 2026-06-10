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


def _load_engine(model: str, device: str, dtype: str):
    """Load a model behind a spinner and wrap it in a GenerationEngine."""
    from mini_vllm.engine.generation import GenerationEngine
    from mini_vllm.engine.model_loader import ModelLoadError, load_model

    _quiet_transformers()
    try:
        with console.status(f"[bold cyan]Loading {model}..."):
            loaded = load_model(model, device=device, dtype=dtype)
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
    engine_mode: str = typer.Option(
        "native", "--engine", help="native = our decode loop, hf = model.generate() fallback."
    ),
) -> None:
    """Generate a completion with the custom decoding loop."""
    from mini_vllm.engine.generation import PromptTooLongError

    engine = _load_engine(model, device, dtype)
    params = _sampling_params(max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop, seed)

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
    out_dir: Path = typer.Option(Path("logs/simulations"), help="Where JSONL metrics land."),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("auto"),
) -> None:
    """Replay a traffic file through the continuous batching scheduler.

    Requests arrive on a wall-clock timeline while earlier ones are still
    decoding; the scheduler grows and shrinks the live batch every tick.
    """
    import time as _time

    from mini_vllm.engine.sampling import SamplingParams
    from mini_vllm.engine.scheduler import Request, Scheduler
    from mini_vllm.metrics.storage import timestamp_slug, write_jsonl

    if not traffic_file.exists():
        console.print(f"[bold red]Error:[/] file not found: {traffic_file}")
        raise typer.Exit(code=1)
    spec = json.loads(traffic_file.read_text())

    engine = _load_engine(model, device, dtype)
    scheduler = Scheduler(engine, max_batch_size=max_batch_size)

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
    shown = ticks if len(ticks) <= 28 else [ticks[i] for i in range(0, len(ticks), len(ticks) // 28 + 1)]
    timeline = Table(title="[bold cyan]Scheduler ticks  (█ active  ░ queued)", border_style="dim")
    timeline.add_column("t (ms)", justify="right")
    timeline.add_column("active", justify="right")
    timeline.add_column("queue", justify="right")
    timeline.add_column("cache len", justify="right")
    timeline.add_column("batch")
    for tk in shown:
        timeline.add_row(
            f"{tk.t_ms:.0f}", str(tk.active), str(tk.queue_depth), str(tk.cache_len),
            "[green]" + "█" * tk.active + "[/][yellow]" + "░" * tk.queue_depth,
        )
    console.print(timeline)

    total_tokens = sum(len(r.generated) for r in requests)
    busy = [tk.active for tk in ticks if tk.active > 0]
    avg_active = sum(busy) / len(busy) if busy else 0.0
    console.print(
        f"[bold]makespan[/] {makespan_ms:.0f} ms   [bold]tokens[/] {total_tokens}   "
        f"[bold]throughput[/] {total_tokens / (makespan_ms / 1000):.1f} tok/s   "
        f"[bold]avg active batch[/] {avg_active:.2f}   [bold]ticks[/] {len(ticks)}"
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


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":
    sys.exit(app())
