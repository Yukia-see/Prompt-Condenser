# Steps 1 and 2 done: fast Tagger mode and batch captioning (Claude → Gemini, 2026-11-11)

## Step 1: fast Tagger mode (`65e995b`)
The slow part of an image read was the tagging itself: `tag_one_fast` runs the tagger 8+ times (full image, 4
corner crops, 3 zones per character), ~20 of the ~32 s. On the 11 test images:
| | F1 | time per image |
|---|---|---|
| full AI pipeline (tagger → ToriiGate → E4B) | 78% | ~32 s |
| tagger all passes + cleanup rules | 79% | ~25 s |
| tagger full image + body zones + rules | 79% | ~15 s |
| **tagger one pass + rules** (now Tagger mode) | **78%** | **~4.6 s** |
Threshold sweep for one pass + rules: 0.25 77.8%, **0.30 79.2%**, 0.35 78.5%, 0.40 76.6%. In the 0.25-0.35 band
no kind of tag is reliable (expression 60%, clothing 47%, background 25%, hair/eyes 12% correct), so no
per-category thresholds. Single-model VLMs aren't needed for the speed goal.

## Step 2: batch captioning (`1450789`)
`prompt-condenser --caption <folder> [--threshold 0.30] [--overwrite] [--recursive]` writes `<image>.txt`
(unescaped Danbooru captions) with the fast mode, ~2.5-8 s per image, no window, no Ollama; existing captions
are skipped unless `--overwrite`. So the batch pipeline from your roadmap exists now; a ComfyUI node could call
this command line (or a future `--prompt` mode) instead of needing a daemon.

Next: step 3, the Identity / Outfit / Atmosphere transfer presets.
