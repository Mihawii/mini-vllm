"""Generate SVG terminal captures for the README and docs.

Runs real CLI commands, records their ANSI output with a Rich recording
console, and saves SVG files that render natively on GitHub. Browser
screenshots (dashboard, /docs) are listed in docs/screenshots.md and need a
human with a browser.

Usage:
    uv run python scripts/make_screenshots.py [--fast]

--fast uses sshleifer/tiny-gpt2 everywhere (CI-friendly); the default uses
distilbert/distilgpt2 so the generated text in the captures reads like text.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

FAST = "--fast" in sys.argv
MODEL = "sshleifer/tiny-gpt2" if FAST else "distilbert/distilgpt2"


def run(args: list[str]) -> str:
    env = dict(os.environ)
    env.update({"FORCE_COLOR": "1", "COLUMNS": "100", "TERM": "xterm-256color"})
    result = subprocess.run(
        args, capture_output=True, text=True, env=env, cwd=ROOT, timeout=900
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def capture(name: str, cmd: list[str], title: str) -> None:
    print(f"[capture] {name}: {' '.join(cmd)}")
    output = run(cmd)
    console = Console(record=True, width=100)
    console.print(Text(f"$ {' '.join(cmd)}", style="bold green"))
    console.print(Text.from_ansi(output))
    ASSETS.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(ASSETS / f"{name}.svg"), title=title)
    print(f"  -> docs/assets/{name}.svg")


def main() -> None:
    uv = ["uv", "run", "mini-vllm"]

    capture(
        "inspect",
        uv + ["inspect", "--model", MODEL],
        "mini-vllm inspect",
    )
    capture(
        "tokenize",
        uv + ["tokenize", "The greenhouse robot inspected the tomatoes.", "--model", MODEL],
        "mini-vllm tokenize",
    )
    capture(
        "generate",
        uv + [
            "generate", "An inference engine is", "--model", MODEL,
            "-n", "60", "--seed", "7", "--temperature", "0.8",
        ],
        "mini-vllm generate",
    )
    capture(
        "simulate",
        uv + ["simulate", "examples/traffic.json", "--model", MODEL, "--seed", "0"],
        "mini-vllm simulate (continuous batching)",
    )
    capture(
        "bench-report",
        uv + ["bench-report", "--no-write"],
        "mini-vllm bench-report",
    )
    print("done")


if __name__ == "__main__":
    main()
