# CPU-only image. The CUDA wheels are ~7 GB; pinning the CPU index keeps
# this image around 1.5 GB. The model downloads on first start and lands in
# the HF_HOME volume if you mount one.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

ENV MINI_VLLM_HOST=0.0.0.0 \
    MINI_VLLM_PORT=8000 \
    MINI_VLLM_MODEL=distilbert/distilgpt2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["mini-vllm", "serve", "--host", "0.0.0.0", "--model", "distilbert/distilgpt2"]
