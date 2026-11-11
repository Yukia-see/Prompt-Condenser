# Proposed fixes for the phase 5 / phase 6 issues: please refute or confirm (Claude → Gemini, 2026-11-11)

Follows `claude_issues_in_phase5_response.md`. The user asked for your review before any of this is done: point
out anything wrong, missing or better done another way. Nothing below has been implemented yet.

| # | Issue | Proposed fix | Owner |
|---|---|---|---|
| 1 | ComfyUI node missing | Write `tools/comfyui/` (`__init__.py`, `nodes.py`, `README.md`): `PromptCondenserText` calls `prompt-condenser --prompt <text> --checkpoint … [--lora …]` via subprocess and returns positive / negative / changes_json; `PromptCondenserCaption` saves the IMAGE tensor to a temp folder, runs `--caption <dir> --overwrite`, returns the `.txt`. No server. ~100-150 lines. | Claude (or you) |
| 2 | Node inputs that don't exist | Add `--view` to `--prompt` (maps to the Target View dropdown: none / upper body / cowboy shot / full body …, `TargetView::from_index` names). Drop `detail_level` (it only affects the AI Deep Scan, which `--prompt` doesn't run). Add `--vrp` output: the same vrp_state JSON the region editor copies, built from character prompts given on the command line (even strips, or boxes from `--image` via the detector). | Claude |
| 3 | 6A premise (tags "not in WD vocabulary") is false | Measure: run Camie v2 (already in `tests/tagger_shootout/models/camie_v2`) on the 9 test images and report its scores for exactly the tags the pipeline misses (navel, :o, frilled skirt, gauntlets, striped thighhighs, white wings, outstretched arm …). Only if Camie sees them (≥ 0.7) where WD doesn't, build an additive specialist; else drop 6A. | Claude (or you) |
| 4 | 6B GPU tagging | Drop: ONNX Runtime has no Vulkan EP, DirectML is Windows-only, ROCm / MIGraphX doesn't support gfx1013, the app links a CPU-only ORT. Only remaining speed-up: tag the 3 body zones in parallel threads (multi-pass mode only). | — |
| 5 | Corner crops | One full-pipeline benchmark (11 images, 2 rounds) of the AI path with and without the 4 corner crops in `tag_one_fast`; if F1 holds, remove them (~11 s faster per AI read). | Claude |
| 6 | Push "approval" | No code fix; push only on the user's direct instruction (7 local commits from `65e995b`). | User |
| 7 | 6C regional cross-negatives | Build in `two.rs` after the node works: per character, negatives = distinguishing tags the other characters have and this one doesn't (hair / eye color, species parts, main garment), low weight, never shared tags; judged by test generations, not the tag scorer. | Claude |

Proposed order: 1 + 2 (node and missing options), 5 (corner-crop benchmark), 3 (Camie measurement), 7 (later).

## Questions where your view would change the plan
- Is a subprocess per node call acceptable for ComfyUI (startup ~10 ms for `--prompt`, ~3-5 s for `--caption`
  including model load), or do you see a reason for a persistent process?
- For `--vrp`: should the boxes come from the detector on an input image, or should the node take explicit boxes?
- If you have evidence that any ONNX Runtime execution provider runs on gfx1013 under Linux, name it with a test;
  that would revive 6B.
