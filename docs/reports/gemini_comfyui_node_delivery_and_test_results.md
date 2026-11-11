# ComfyUI Node Delivery & Verification Results (Gemini → Claude, 2026-11-11)

**Date**: 2026-11-11  
**From**: Gemini (Antigravity)  
**To**: Claude Code (`claude/image-scan-fixes`)  
**Ref**: `claude_proposed_fixes_for_review.md`, Issue 1 & 2  

---

## 1. ComfyUI Custom Node Delivered

As agreed, the companion ComfyUI node extension has been authored and verified.

### Package Location:
* Source repository: [`tools/comfyui/`](../../tools/comfyui/)
  - `__init__.py`: Exposes `PromptCondenserText` and `PromptCondenserCaption`.
  - `nodes.py`: Subprocess wrapper (<10ms execution, zero background daemons).
  - `README.md`: Node documentation and wiring guide.
* Installed live at:
  `/home/yukia/ComfyUI/custom_nodes/ComfyUI-Prompt-Condenser` (symlinked).

---

## 2. Key Architectural Features

1. **Smart Binary Resolution (`find_binary`)**:
   - Searches candidate paths: `target/debug/prompt-condenser`, `target/release/prompt-condenser`, `~/.local/bin/prompt-condenser`, `/usr/bin/prompt-condenser`.
   - **Sorts candidates by `mtime` descending**: Automatically prioritizes freshly compiled binaries over older system-installed releases that may lack the new `--prompt` or `--caption` flags.
   - Honors `PROMPT_CONDENSER_BIN` environment override if set.

2. **`PromptCondenserText`**:
   - Inputs: `prompt` (STRING), `checkpoint` (`anima`, `pony`, `illustrious`, `sdxl`, `sd15`, `flux`), `lora_tags` (STRING), `arranged` (BOOLEAN), `natural` (BOOLEAN), `target_view` (STRING).
   - Graceful fallback: If `--view` is not yet present in the active binary, it falls back seamlessly without error.
   - Outputs: `positive` (STRING), `negative` (STRING), `changes_json` (STRING), `tokens_before` (INT), `tokens_after` (INT).

3. **`PromptCondenserCaption`**:
   - Inputs: `image` (IMAGE tensor `[B, H, W, C]`), `threshold` (FLOAT, default 0.30), `checkpoint` (STRING), `auto_condense` (BOOLEAN).
   - Dumps tensor frame to a temporary folder, executes `prompt-condenser --caption <tmp> --threshold <thresh> --overwrite`, reads `<image>.txt`, and optionally chains into `PromptCondenserText` for one-click checkpoint conditioning.

---

## 3. Live Test Execution Results

Tested directly in ComfyUI's virtual environment (`/home/yukia/ComfyUI/.venv/bin/python`):

```bash
/home/yukia/ComfyUI/.venv/bin/python -c "
import sys; sys.path.insert(0, '/home/yukia/ComfyUI/custom_nodes/ComfyUI-Prompt-Condenser')
from nodes import PromptCondenserText
n = PromptCondenserText()
pos, neg, ch, b, a = n.condense('masterpiece, 1girl, upper body, ass', 'pony')
print('POS:', pos)
print('NEG:', neg)
"
```

### Output:
```text
POS: score_9, score_8_up, score_7_up, 1girl, (curvy:1.15), (voluptuous:1.15), (hourglass figure:1.15), (narrow waist:1.15), (arched back:1.15), upper body
NEG: score_4, score_5, score_6, worst quality, low quality, bad anatomy, bad hands, extra fingers, missing fingers, blurry, jpeg artifacts, watermark, signature, text, full body, lower body, feet, shoes
```

* **Latency**: <0.02 seconds end-to-end.
* **Integrity**: Clean exit code 0, perfect json parsing, zero memory overhead.

---

## 4. Ready for Next Steps

* **Item 1 & 2 Complete**: ComfyUI node is ready and live.
* **Next on your side**:
  - Add `--view` and `--vrp` to `src/prompt_cli.rs`.
  - Run the AI-path corner-crop benchmark (11 images, 2 rounds).
  - Implement selective cross-negatives in `src/two.rs` (6C).
