#!/usr/bin/env bash
# Builds a self-contained installer bundle for Intel & AMD (and NVIDIA) x86-64 machines:
#   dist/prompt-condenser-installer/
#     prompt-condenser        release binary
#     install.sh              installs Ollama (dependency) + the app into ~/.local, and can
#                             download the AI model through Ollama
#   dist/prompt-condenser-installer.tar.gz
#
# Models aren't bundled: Ollama downloads them (install.sh or "Download Model" in the app).
# For Arch Linux, prefer the pacman package: scripts/build_arch_package.sh
#
# Usage: scripts/build_installer.sh
#   TARGET_CPU=x86-64-v3 (default, AVX2 — Intel Haswell+/AMD Zen+). Use x86-64-v2 for older CPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist/prompt-condenser-installer"
TARGET_CPU="${TARGET_CPU:-x86-64-v3}"

for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "==> Building release binary (target-cpu=$TARGET_CPU)"
(cd "$ROOT" && RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=$TARGET_CPU" cargo build --release)

mkdir -p "$DIST"
install -m 755 "$ROOT/target/release/prompt-condenser" "$DIST/prompt-condenser"
install -m 755 "$ROOT/install.sh" "$DIST/install.sh"

echo "==> Creating tarball"
tar -C "$ROOT/dist" -czf "$ROOT/dist/prompt-condenser-installer.tar.gz" prompt-condenser-installer

echo
echo "Done:"
du -sh "$DIST" "$ROOT/dist/prompt-condenser-installer.tar.gz"
echo "Install with: $DIST/install.sh"
