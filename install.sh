#!/usr/bin/env bash
# Installs ComfyUI Prompt Condenser for the current user and sets up its AI dependency:
#   1. Ollama + Vulkan GPU support (ollama, ollama-vulkan) via pacman, service enabled
#   2. binary -> ~/.local/bin/prompt-condenser, menu entry in ~/.local/share/applications
#   3. optionally downloads the recommended AI model (Llama 3.1 8B Abliterated, uncensored)
#      through Ollama (other models: "Download Model" in the app)
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
    # Only the models registered under this app's own names (the current one, then older ones).
    # General models it can use (Llama 3.1 8B, Gemma 4 12B, Qwen-VL, Moondream) are left alone,
    # since other apps may use them. One `ollama rm` each: it stops at the first missing name.
    for m in prompt-condenser-gemma4-e4b prompt-condenser-gemma4-e2b-q8 prompt-condenser-gemma4-12b; do
      if ollama rm "$m" >/dev/null 2>&1; then echo "Removed $m"; fi
    done
    echo "General models (Llama 3.1 8B, Gemma 4 12B, vision models) were kept; remove them with 'ollama rm <name>'."
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

# ---------------------------------------------------------------- AI model (through Ollama)
MODEL_NAME=richardyoung/llama-3.1-8b-instruct-abliterated
MODEL_SRC=richardyoung/llama-3.1-8b-instruct-abliterated:latest
if command -v ollama >/dev/null && ollama_up; then
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL_NAME:latest"; then
    echo "==> $MODEL_NAME is already in Ollama"
  elif ask "Download the recommended AI model now (Llama 3.1 8B Abliterated, ~4.9 GB, needs ~5 GB VRAM)?"; then
    ollama pull "$MODEL_SRC"
  else
    echo "    Skipped. Pick a model in the app and click 'Download Model' (Good: Gemma 4 E4B ~3.5 GB, Best: Gemma 4 12B Abliterated ~8 GB)."
  fi
else
  echo "==> Ollama isn't running; download a model later from the app ('Download Model')."
fi

echo
echo "Installed. Launch 'Prompt Condenser' from your app menu or run: prompt-condenser"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Note: $BIN_DIR is not on your PATH." ;;
esac
