.PHONY: install test lint format serve bench screenshots clean

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

serve:
	uv run mini-vllm serve --model distilbert/distilgpt2

bench:
	uv run mini-vllm bench --model distilbert/distilgpt2 --requests 10 --max-new-tokens 64 --compare-kv-cache

screenshots:
	uv run python scripts/make_screenshots.py

clean:
	rm -rf .pytest_cache .ruff_cache dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
