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


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":
    sys.exit(app())
