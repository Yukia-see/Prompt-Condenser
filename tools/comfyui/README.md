# ComfyUI-Prompt-Condenser

ComfyUI custom nodes for the [Prompt Condenser](https://github.com/prompt-condenser) anime prompt optimizer and Danbooru fast tagger.

## Features

1. **Prompt Condenser (Text Optimizer)**:
   - Cleans and condenses raw prompts into target checkpoint standards (`Anima`, `PonyV6`, `Illustrious`, `SDXL`, `SD1.5`, `Flux`).
   - Translates tag formats (e.g. converting `masterpiece` to score tags for Pony V6).
   - Resolves framing and anatomy conflicts (e.g. replacing camera-twisting lower-body tags when `upper body` is requested).
   - Generates checkpoint-tailored auto-negative prompts.
   - Runs in **<10 ms** via subprocess execution without background daemons or port collisions.

2. **Prompt Condenser (Fast Tagger & Caption)**:
   - Takes a ComfyUI `IMAGE` tensor and runs Prompt Condenser's WD-EVA02 ONNX tagger + Rust visual rules.
   - Returns both raw canonical Danbooru tags and auto-condensed prompt tags for downstream samplers.

## Installation

Symlink or copy this directory into your ComfyUI `custom_nodes/` directory:

```bash
ln -s "/home/yukia/Documents/Prompt Condensor/tools/comfyui" "/home/yukia/ComfyUI/custom_nodes/ComfyUI-Prompt-Condenser"
```

Requires `prompt-condenser` binary compiled or installed in PATH (`~/.local/bin/prompt-condenser` or `target/release/prompt-condenser`).
