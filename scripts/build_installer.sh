#!/usr/bin/env bash
# Builds a self-contained installer bundle for Intel & AMD (and NVIDIA) x86-64 machines:
#   dist/prompt-condenser-installer/
#     prompt-condenser        release binary
#     models/                 pre-packaged GGUF models (registered into Ollama by install.sh)
#     install.sh              installs Ollama (dependency) + the app into ~/.local
#   dist/prompt-condenser-installer.tar.gz
#
# Usage: scripts/build_installer.sh [--no-models] [--low-only]
#   TARGET_CPU=x86-64-v3 (default, AVX2 — Intel Haswell+/AMD Zen+). Use x86-64-v2 for older CPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist/prompt-condenser-installer"
MODELS="$DIST/models"
TARGET_CPU="${TARGET_CPU:-x86-64-v3}"

WITH_MODELS=1
WITH_HIGH=1
for arg in "$@"; do
  case "$arg" in
    --no-models) WITH_MODELS=0 ;;
    --low-only) WITH_HIGH=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

HF="https://huggingface.co"
LOW_FILE="Qwen2.5-1.5B-Instruct-abliterated.Q4_K_M.gguf"
LOW_URL="$HF/mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF/resolve/main/$LOW_FILE"
HIGH_FILE="Qwen2.5-3B-Instruct-abliterated.Q4_K_M.gguf"
HIGH_URL="$HF/mradermacher/Qwen2.5-3B-Instruct-abliterated-GGUF/resolve/main/$HIGH_FILE"

echo "==> Building release binary (target-cpu=$TARGET_CPU)"
(cd "$ROOT" && RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=$TARGET_CPU" cargo build --release)

mkdir -p "$MODELS"
install -m 755 "$ROOT/target/release/prompt-condenser" "$DIST/prompt-condenser"
install -m 755 "$ROOT/install.sh" "$DIST/install.sh"

fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "    already have $(basename "$dest")"
    return
  fi
  echo "    downloading $(basename "$dest")"
  # -C - resumes an interrupted download from the .part file
  curl -fL --retry 3 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

if [[ $WITH_MODELS -eq 1 ]]; then
  echo "==> Packaging models into $MODELS"
  fetch "$LOW_URL" "$MODELS/$LOW_FILE"
  if [[ $WITH_HIGH -eq 1 ]]; then
    fetch "$HIGH_URL" "$MODELS/$HIGH_FILE"
  fi
else
  echo "==> Skipping models (--no-models); the app can download them through Ollama"
fi

echo "==> Creating tarball"
tar -C "$ROOT/dist" -czf "$ROOT/dist/prompt-condenser-installer.tar.gz" prompt-condenser-installer

echo
echo "Done:"
du -sh "$DIST" "$ROOT/dist/prompt-condenser-installer.tar.gz"
echo "Install with: $DIST/install.sh"
