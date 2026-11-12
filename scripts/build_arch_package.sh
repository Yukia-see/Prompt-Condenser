#!/usr/bin/env bash
# Builds the Arch Linux package (pacman) for Prompt Condenser.
#   scripts/build_arch_package.sh            -> packaging/arch/prompt-condenser-<ver>-1-x86_64.pkg.tar.zst
#   scripts/build_arch_package.sh --install  -> also installs it, pulling in its dependencies
#                                               (ollama, ollama-vulkan, ...) with pacman
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKGDIR="$ROOT/packaging/arch"
NAME=prompt-condenser
VER="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"

INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --install|-i) INSTALL=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

sed -i "s/^pkgver=.*/pkgver=$VER/" "$PKGDIR/PKGBUILD"

echo "==> Packing sources ($NAME $VER)"
tar -C "$ROOT" \
  --exclude=./.git --exclude=./target --exclude=./dist \
  --exclude=./packaging/arch/src --exclude=./packaging/arch/pkg \
  --exclude='*.pkg.tar.zst' --exclude='./packaging/arch/*.tar.gz' \
  --exclude=./docs/reports --exclude=./.agents --exclude=__pycache__ --exclude='*.zip' \
  --exclude='./*.jpg' --exclude='./packaging/arch/*.jpg' --exclude='./packaging/arch/PROGRAM RESULT' \
  --exclude='./Untitled Document' --exclude=./prompt_condenser.rs \
  --exclude=./tests/tagger_shootout \
  --transform "s,^\.,$NAME-$VER," \
  -czf "$PKGDIR/$NAME-$VER.tar.gz" .

cd "$PKGDIR"
if [[ $INSTALL -eq 1 ]]; then
  # -s installs missing dependencies, -i installs the package (both ask for sudo)
  makepkg -sfci
else
  makepkg -sfc
fi
ls -lh "$PKGDIR"/*.pkg.tar.zst
