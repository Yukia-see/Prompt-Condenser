# Prompt Condenser

A desktop app for **ComfyUI / Stable Diffusion prompts**. Paste a prompt, and it:

- **Condenses** it: removes duplicate tags and filler like `8k` and `ultra detailed`, and clamps runaway weights like `((((masterpiece))))`.
- **Fixes framing conflicts** for your checkpoint (Anima, Pony V6, Illustrious/NoobAI, SDXL, SD 1.5, Flux). For example, it swaps `ass` for framing-safe tags in an *Upper Body* shot, and removes single-angle tags from a *360° turnaround*. Every change is a card you can accept or undo.
- **Describes the scene** your prompt creates, live, and points out what's missing (lighting, background, expression…).
- **Recommends a negative prompt** that fits your checkpoint and camera view.
- **Runs a Deep AI Scan** (optional) through [Ollama](https://ollama.com) with Vulkan GPU support. It returns a scene analysis, an improved prompt, a tailored negative prompt and individual suggestions. The instructions adapt to your prompt and to the model.
- **Extracts Danbooru tags from an image** with the WD EVA02-Large tagger, which runs locally on the CPU.
- **Camera angle and pose pickers** with common prompting and photography terms (Dutch angle, worm's-eye, contrapposto, dynamic pose, …).
- **Dynamic prompts:** an optional `{a|b|c}` variation view (pose, angle, expression, lighting) for the ComfyUI Dynamic Prompts extension.

Built with Rust and [Slint](https://slint.dev) (Material 3 dark).

## Install (Arch Linux)

Download the package from the [latest release](https://github.com/Yukia-see/Prompt-Condenser/releases/latest), then:

```sh
sudo pacman -U prompt-condenser-*-x86_64.pkg.tar.zst
sudo systemctl enable --now ollama      # needed for the Deep AI Scan
```

pacman pulls in the dependencies, including `ollama`, `ollama-vulkan` and `vulkan-tools`. Launch **Prompt Condenser** from your app menu, or run `prompt-condenser`. You can also start it with a prompt: `prompt-condenser "1girl, solo, …"`.

Models download from inside the app the first time you need them:
- **Deep AI Scan:** pick a model in the *AI model* list and click **Download Model**. All models are uncensored Gemma 4:
  - **Maximum Fast — Gemma 4 12B** (~8 GB VRAM): recommended; the best results in our benchmarks.
  - **Balanced — Gemma 4 E4B** (~3.5 GB VRAM): faster, but adds fewer details.
  - **Light — Gemma 4 E2B Q8** (~3 GB VRAM): for weak GPUs.
- **Image → Danbooru tags:** click **From Image…**, then **Download Tagger** (1.2 GB, once).

The package also adds an Ollama systemd drop-in that waits for the Vulkan GPU at boot, so Ollama doesn't fall back to the CPU when it starts before the GPU driver is ready.

## Build from source

```sh
cargo build --release                  # binary in target/release/prompt-condenser
cargo test                             # unit tests
scripts/build_arch_package.sh          # Arch package in packaging/arch/
scripts/build_arch_package.sh --install   # build and install with dependencies
```

Requires a recent stable Rust (built and tested with 1.99). The ONNX Runtime used by the tagger is downloaded and statically linked at build time.
