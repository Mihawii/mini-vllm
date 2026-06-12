#!/usr/bin/env bash
# Builds docs/assets/demo.mp4 from live browser recordings + caption cards.
# Prereqs: a mini-vllm server on :8742 with bench results and a simulation
# saved, plus playwright-core installed in scripts' working env (see
# docs/demo_video.md). The ffmpeg shipped with Playwright is reused.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${1:-http://127.0.0.1:8742}"
WORK="$(mktemp -d /tmp/mini-vllm-video.XXXX)"
FFMPEG="$(command -v ffmpeg || ls "$HOME"/Library/Caches/ms-playwright/ffmpeg-*/ffmpeg-mac 2>/dev/null | head -1)"
[ -x "$FFMPEG" ] || { echo "no ffmpeg found (need system ffmpeg or the Playwright bundle)"; exit 1; }

echo "workdir: $WORK"
node "$ROOT/scripts/video/record.js" "$ROOT" "$BASE" "$WORK"

norm() { # input, output, [duration for stills]
  if [[ "$1" == *.png ]]; then
    "$FFMPEG" -y -loglevel error -loop 1 -t "$3" -i "$1" \
      -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=#f1efe8,format=yuv420p,fps=30" \
      -c:v libx264 -preset fast -crf 22 "$2"
  else
    "$FFMPEG" -y -loglevel error -i "$1" \
      -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p,fps=30" \
      -c:v libx264 -preset fast -crf 22 -an "$2"
  fi
}

cd "$WORK"
norm card-title.png      01.mp4 5
norm seg-playground.webm 02.mp4
norm term-generate.png   03.mp4 4.5
norm seg-tokenizer.webm  04.mp4
norm card-batching.png   05.mp4 5
norm term-simulate.png   06.mp4 5.5
norm seg-scheduler.webm  07.mp4
norm card-bench.png      08.mp4 5
norm term-bench.png      09.mp4 5
norm seg-benchmarks.webm 10.mp4
norm card-api.png        11.mp4 7
norm card-closing.png    12.mp4 5

for f in 01.mp4 02.mp4 03.mp4 04.mp4 05.mp4 06.mp4 07.mp4 08.mp4 09.mp4 10.mp4 11.mp4 12.mp4; do echo "file '$WORK/$f'"; done > list.txt
"$FFMPEG" -y -loglevel error -f concat -safe 0 -i list.txt -c copy "$ROOT/docs/assets/demo.mp4"

"$FFMPEG" -i "$ROOT/docs/assets/demo.mp4" 2>&1 | grep Duration
ls -lh "$ROOT/docs/assets/demo.mp4"
