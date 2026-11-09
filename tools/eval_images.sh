#!/usr/bin/env bash
# Read every image in tests/eval_local (or the folder given) with the app's image chain and score the
# prompts against the answer files (<image>.txt). Needs Ollama running with the image models.
# Env passes through to live_image: IMAGE_MODEL, VISION_ENCODER, PC_NO_POLISH, ...
set -euo pipefail
cd "$(dirname "$0")/.."
dir="${1:-tests/eval_local}"
images=$(find "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | sort | paste -sd:)
[ -n "$images" ] || { echo "no images in $dir"; exit 1; }
out="target/eval_images_$(date +%Y%m%d_%H%M%S).txt"
IMAGE_TEST="$images" cargo test --release live_image -- --ignored --nocapture --test-threads=1 | tee "$out"
python3 tools/score_image_prompts.py "$out" "${IMAGE_MODEL:-default}"
echo "raw output: $out"
