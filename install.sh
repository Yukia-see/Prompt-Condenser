#!/usr/bin/env bash
# Installs ComfyUI Prompt Condenser for the current user and sets up its AI dependency:
#   1. Ollama + Vulkan GPU support (ollama, ollama-vulkan) via pacman, service enabled
#   2. binary -> ~/.local/bin/prompt-condenser, menu entry in ~/.local/share/applications
#   3. bundled models (models/*.gguf) registered in Ollama as
#      prompt-condenser-low / prompt-condenser-high
#
# Run from the installer bundle (scripts/build_installer.sh) or from the source tree
# after `cargo build --release`.
# Usage: ./install.sh [--uninstall]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
DESKTOP_FILE="$DATA_DIR/applications/prompt-condenser.desktop"
OLLAMA_URL="http://localhost:11434"

ask() { # ask "question" -> 0 for yes
  local reply
  read -r -p "$1 [Y/n] " reply || reply=n
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

ollama_up() { curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

if [[ "${1:-}" == "--uninstall" ]]; then
  rm -f "$BIN_DIR/prompt-condenser" "$DESKTOP_FILE"
  echo "Removed binary and menu entry."
  if command -v ollama >/dev/null && ollama_up && ask "Also remove the prompt-condenser models from Ollama?"; then
    ollama rm prompt-condenser-low prompt-condenser-high 2>/dev/null || true
  fi
  echo "Ollama itself was left installed (sudo pacman -R ollama-vulkan ollama to remove it)."
  exit 0
fi

# ---------------------------------------------------------------- dependency: Ollama
echo "==> Checking Ollama (AI engine, Vulkan GPU via ollama-vulkan)"
if command -v pacman >/dev/null; then
  missing=()
  pacman -Q ollama >/dev/null 2>&1 || missing+=(ollama)
  pacman -Q ollama-vulkan >/dev/null 2>&1 || missing+=(ollama-vulkan)
  if (( ${#missing[@]} )); then
    if ask "Install ${missing[*]} with pacman (needs sudo)?"; then
      sudo pacman -S --needed "${missing[@]}"
    else
      echo "    Skipped. The rule-based condenser still works; the Deep AI Scan needs Ollama."
    fi
  else
    echo "    ollama + ollama-vulkan already installed"
  fi
elif ! command -v ollama >/dev/null; then
  echo "    Ollama not found and this isn't an Arch system."
  echo "    Install it from https://ollama.com (with GPU support for your card), then re-run."
fi

if command -v ollama >/dev/null && ! ollama_up; then
  if command -v systemctl >/dev/null && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
    if ask "Enable and start the Ollama service (with Vulkan enabled)?"; then
      # Some Ollama releases only use the Vulkan backend when this is set; harmless otherwise.
      sudo mkdir -p /etc/systemd/system/ollama.service.d
      printf '[Service]\nEnvironment="OLLAMA_VULKAN=1"\n' | sudo tee /etc/systemd/system/ollama.service.d/vulkan.conf >/dev/null
      sudo systemctl daemon-reload
      sudo systemctl enable --now ollama
    fi
  else
    echo "    Start Ollama with: OLLAMA_VULKAN=1 ollama serve"
  fi
  for _ in $(seq 1 20); do ollama_up && break; sleep 0.5; done
fi

# ---------------------------------------------------------------- app
if [[ -x "$HERE/prompt-condenser" ]]; then
  BIN_SRC="$HERE/prompt-condenser"
elif [[ -x "$HERE/target/release/prompt-condenser" ]]; then
  BIN_SRC="$HERE/target/release/prompt-condenser"
else
  echo "No prompt-condenser binary found next to install.sh or in target/release." >&2
  echo "Run 'cargo build --release' or scripts/build_installer.sh first." >&2
  exit 1
fi

echo "==> Installing binary to $BIN_DIR"
mkdir -p "$BIN_DIR"
install -m 755 "$BIN_SRC" "$BIN_DIR/prompt-condenser"

echo "==> Creating menu entry"
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Prompt Condenser
GenericName=ComfyUI Prompt Condenser
Comment=Condense ComfyUI prompts and fix framing / body-part conflicts
Exec=$BIN_DIR/prompt-condenser
Icon=accessories-text-editor
Terminal=false
Categories=Graphics;Utility;
DESKTOP
command -v update-desktop-database >/dev/null && update-desktop-database "$(dirname "$DESKTOP_FILE")" >/dev/null 2>&1 || true

# ---------------------------------------------------------------- bundled models -> Ollama
register() { # register <ollama-name> <gguf path>
  local name="$1" gguf="$2" modelfile
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$name:latest"; then
    echo "    $name already in Ollama"
    return
  fi
  echo "    $name <- $(basename "$gguf")"
  modelfile="$(mktemp)"
  # ChatML template for Qwen2.5 (the app sends a system + user message)
  cat > "$modelfile" <<MODELFILE
FROM $gguf
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
MODELFILE
  ollama create "$name" -f "$modelfile"
  rm -f "$modelfile"
}

shopt -s nullglob
low=("$HERE"/models/*1.5B*.gguf)
high=("$HERE"/models/*3B*.gguf)
if (( ${#low[@]} + ${#high[@]} )); then
  if command -v ollama >/dev/null && ollama_up; then
    echo "==> Registering bundled models in Ollama"
    (( ${#low[@]} )) && register prompt-condenser-low "${low[0]}"
    (( ${#high[@]} )) && register prompt-condenser-high "${high[0]}"
  else
    echo "==> Ollama isn't running, so bundled models weren't registered. Re-run install.sh once it is."
  fi
else
  echo "==> No bundled models; use 'Download Model' in the app (Ollama downloads them)."
fi

echo
echo "Installed. Launch 'Prompt Condenser' from your app menu or run: prompt-condenser"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Note: $BIN_DIR is not on your PATH." ;;
esac
