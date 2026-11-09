# Prompt Condenser

A desktop app for **ComfyUI / Stable Diffusion prompts**. Paste a prompt, and it:

- **Condenses** it: removes duplicate tags and filler like `8k` and `ultra detailed`, and clamps runaway weights like `((((masterpiece))))`.
- **Fixes framing conflicts** for your checkpoint (Anima, Pony V6, Illustrious/NoobAI, SDXL, SD 1.5, Flux). For example, it swaps `ass` for framing-safe tags in an *Upper Body* shot, and removes single-angle tags from a *360° turnaround*. Every change is a card you can accept or undo.
- **Describes the scene** your prompt creates, live, and points out what's missing (lighting, background, expression…).
- **Recommends a negative prompt** that fits your checkpoint and camera view.
- **Runs a Deep AI Scan** (optional) through [Ollama](https://ollama.com) with Vulkan GPU support. It returns a scene analysis, an improved prompt, a tailored negative prompt and individual suggestions. The instructions adapt to your prompt and to the model.
  - **AI Detail** dropdown: *Faithful* (fix problems only), *Balanced* (fix and fill gaps), or *Rich* (a detailed, descriptive rewrite: hair, outfit materials, pose, setting, light, camera and atmosphere, about 40–60 tags).
- **Extracts Danbooru tags from an image** with the WD EVA02-Large tagger, which runs locally on the CPU.
  - A **Tagger / AI** switch: AI mode has a vision model (Qwen2.5-VL, Moondream or Gemma 4 12B) look at the image and write a full prompt in your checkpoint's style, using the tagger's tags as hints. With it on, Deep Scan reads the image first.
- **LoRA tags box** (under Keep & Emphasize): its tags (`<lora:…>`, trigger words, the details your LoRA needs) go into every output word for word and are never changed. They replace contradicting tags in the prompt (LoRA tag `large breasts` replaces `gigantic breasts`), in the rule-based output and in the AI's rewrite.
- **Danbooru Words** (button next to Checkpoint Profile): describe something the model may not understand, such as unusual words, metaphors or prose, and the AI turns it into real Danbooru tags. It picks them from Danbooru's tag list, checks them against it, and shows what each phrase became ("peek out from behind a door, all shy" → `peeking, open door, shy`). For Anima, SDXL and Flux it can add a natural-language description as well.
- **1 to 4 Characters tabs:** write each character in their own box, each with its own LoRA box, plus one **Scene** box for what they all share (setting, camera, light, what they do together). The After prompt keeps each character's tags together: quality, the count (`2girls`, `1girl, 1boy`…) and interactions, then each character in turn (separated by `BREAK` on CLIP checkpoints), then the shared camera, setting and light. So hair, eyes and outfits don't bleed between them. Rules, Deep Scan, cards and the scene description all work per character.
  - On **Anima** (which ignores BREAK) the characters are written the way Anima understands: *"the girl on the left is … with …, wearing …, the girl in the middle …"*. Pony, Illustrious, SDXL and SD 1.5 keep `BREAK`.
  - **Split Image:** one picture with several characters is separated by itself. A local anime person detector finds them left to right, the Danbooru tagger reads each one, and in AI mode a vision model checks who is who (it handles overlapping poses and mirror images too) and fills every character plus the Scene. Results are the same every time you read the same image.
  - **+ Char 1…4 images:** build a scene from separate reference images, each filling only its own character.
  - **Identity Only:** an image supplies just who the character is and what they wear (hair, eyes, body, outfit), not its pose, camera or background, so your own scene stays in charge.
  - **Read** reads everything you loaded, in order: the Scene / Pose reference first, then characters 1 to 4. Nothing runs until you press it.
- **Scene / Pose reference image:** load a picture and keep only what it *does*: each character's pose and action, who holds what, gaze, camera framing, setting and lighting, never who is in it. **⚡ Auto Scene (Fast)** does it locally in seconds without any AI. The **Scene Panel** shows who does what, lets you move actions between characters, and gives every character their own **View**, **Angle** and **Pose** dropdowns.
- **Body-type check:** `petite` (or slim, slender, skinny) next to large breasts or wide hips makes the model shrink the chest and hips you asked for, so it's replaced with `short`, with a card to undo it.
- **Material 3 look with six themes** (OLED Pitch Black, Deep Violet, Cyber Emerald, Ocean Blue, Crimson Ember, Mocha Gold). Your checkpoint, AI mode, view, angle, pose, theme and prompts are remembered between sessions.
- **Arranged layout** (Options): the After prompt as grouped lines in the order the model weighs most (quality, LoRA, who, looks, outfit, expression and pose, camera, setting and light, style), easy to read at a glance and still a normal prompt to ComfyUI.
- **Natural-language bits** (Options; Anima, SDXL, Flux): adds 1–2 sentences for what tags get across badly, such as interactions, who has which features, composition and mood. Deep Scan writes them; without a scan, a rule-based sentence is used.
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
- **Deep AI Scan:** pick a model in the *AI model* list and click **Download Model**. All models are uncensored (abliterated):
  - **Best — Gemma 4 12B Abliterated** (~8 GB VRAM): the largest and most detailed.
  - **Better — Llama 3.1 8B Abliterated** (~5 GB VRAM): a middle size.
  - **Good — Gemma 4 E4B** (~3.5 GB VRAM): fast, light on memory.
- **Reading images with AI** uses a vision model that you download once with Ollama: `ollama pull huihui_ai/qwen2.5-vl-abliterated:3b` (recommended) or `ollama pull moondream:1.8b`. Without one, the app falls back to the tagger (or to the AI model itself if it can see images). Vision models are unloaded right after they've read the image, to leave room for the text model.
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
