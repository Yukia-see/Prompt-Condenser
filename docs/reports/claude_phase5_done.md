# Phase 5 done on Claude's side: region editor and the ComfyUI interface (Claude → Gemini, 2026-11-11)

Commits `4a0e6a1` (region editor) and `775f2d1` (`--prompt`), local, not pushed yet. 219 tests pass.

## Region editor (Slint)
"✎ Edit regions" next to "🗺 Copy VRP regions" on the 2-4 Characters tabs opens an overlay: the picture (or a blank
canvas with its shape) and one colored box per character (red / blue / green / yellow, as in the AI split). Drag a
box to move it, drag its corner handle to resize, "↺ Reset" goes back to the detector boxes / even strips. "Copy VRP
regions" (in the overlay and in the toolbar) uses the edited boxes; a new Split Image clears the edits.
`two::vrp_regions` + `two::vrp_state_json_from` split the old builder; JSON format unchanged (vrp_state v2, regions
clamped to the canvas). Not clicked through yet (no window access from here): please try it in the next build.

## ComfyUI node interface (for your node draft)
Both work without a window or Ollama, so the node can call the binary as a subprocess:
- **Text:** `prompt-condenser --prompt "<text>" | - [--checkpoint anima|pony|illustrious|sdxl|sd15|flux]
  [--lora "<tags>"] [--arranged] [--natural]` → JSON on stdout:
  `{"checkpoint", "positive", "negative", "changes": [{"original","suggestion","reason"}], "tokens_before",
  "tokens_after"}`. `positive` = the window's rule-based After view with every suggestion applied; `negative` is
  empty for Flux. Errors go to stderr with exit code 1. Example (Pony, "masterpiece, 1girl, upper body, ass, smile,
  smile"): score tags replace masterpiece, ass becomes a framing-safe body description, the duplicate goes.
- **Images:** `prompt-condenser --caption <folder> [--threshold 0.30] [--overwrite] [--recursive]` writes
  `<image>.txt` per image (fast Tagger mode). For a single image in a node: copy it into a temp folder, run
  `--caption`, read the `.txt`.
- Not exposed: the AI Deep Scan and the AI image read (they need Ollama and take 30+ s); say if the node needs them.

Roadmap status: steps 1-5 done on Claude's side. Open: your ComfyUI node; the corner-crop decision for the AI path
(needs a full-pipeline benchmark); pushing and building (user's call).
