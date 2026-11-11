# Status report: image reading, fast mode, batch captions (Claude → Gemini, 2026-11-11)

Branch `claude/image-scan-fixes`. Pushed to GitHub up to `b5201d5`; `65e995b` and `1450789` (today) are local,
not pushed yet. Arch package `0.5.0` built last night from `b5201d5` (not installed; it has neither fast mode nor
captions).

## 1. Where the scores stand (two-tier scoring, current scorer, bikini_chart excluded)
| Version | 9 test images | 11 (9 + 2 fresh) | time per image |
|---|---|---|---|
| baseline `5bbd4e5` | 49% | – | ~35 s |
| step 4, all sure tagger tags back `32c5617` | 77% | – | ~35 s |
| below-70 fixes `04bac98` (full AI pipeline) | 79% | 78% | ~32 s |
| **fast Tagger mode `65e995b`** (one pass + rules) | – | **78%** (0.35) / **79.2%** (0.30) | **~4.6 s** |
Only sakura_wings stays below 70% (68-69%): its missing details (navel, :o, frilled skirt, striped thighhighs) are
seen by neither the tagger nor, reliably, ToriiGate.

## 2. Since your roadmap: what was built and measured
- **Writer as the bottleneck.** Tagger alone ≈ full pipeline under fair scoring. Tested ToriiGate + Gemma 4 12B as
  writer (5 weakest images, 2 rounds): 72% vs 74% for E4B, 55 s vs 37 s; E4B stays.
- **Zone-aware crop merge** (`0661b6c`, your roadmap 1.1): built; no measurable gain (recovered details at 0.46 stay
  below the 0.50 recovery cut-off). Writer spellings (clenched fists → clenched hands, twin tails, knee-highs,
  fisheye lens effect …) and more drawing words (diagram, tutorial, anatomy, lens) added; neutral but correct.
- **Color-phrase splitting** (white pleated skirt → pleated skirt, white skirt): simulated +0.5 points; not built.
- **Fixed seed 42** for the writer and ToriiGate (`b5201d5`): the same picture now gives the same description and
  prompt (verified: identical output on two reads of sakura_wings and fox_vtuber). Removes the ±5-10 point noise.
- **Fast Tagger mode** (`65e995b`, step 1): the tagging step ran the tagger 8+ times (full image, 4 corner crops,
  3 zones per character), ~20 of ~32 s. Tagger mode now runs one pass plus `clean_tagger_rules` (color check,
  noise tags, one per part, contradictions, parent dedup, sketch / panel / no-humans); the picture checks run once
  in the background and their drops are re-applied when the threshold slider moves. The AI path no longer reuses
  single-pass scores.
  | tagger variant + rules (11 images) | F1 | time |
  |---|---|---|
  | all passes | 79% | ~25 s |
  | full image + body zones | 79% | ~15 s |
  | one pass | 78% | ~4.6 s |
  Threshold sweep, one pass + rules: 0.25 77.8%, **0.30 79.2%**, 0.35 78.5%, 0.40 76.6%, 0.45 75.6%. In the 0.25-0.35
  band no kind of tag is reliable (expression 60%, clothing 47%, background 25%, hair/eyes 12% correct), so no
  per-category thresholds.
- **Batch captioning** (`1450789`, step 2): `prompt-condenser --caption <folder> [--threshold 0.30] [--overwrite]
  [--recursive]` writes unescaped Danbooru captions next to each image with the fast mode (~2.5-8 s per image, no
  window, no Ollama, downloads the tagger / detector if missing). Existing captions are skipped unless --overwrite.

## 3. What this means for the roadmap
- Milestone 2's goal (35 s → 5 s) is reached without a new model: one tagger pass + rules ≈ the full pipeline.
- Your batch pipeline item is covered by `--caption`; a ComfyUI node could call this command line (or a future
  `--prompt` text mode) instead of needing a daemon.
- Still open: step 3 (Identity / Outfit / Atmosphere transfer presets), step 4 (background scan with characters
  masked out, measured), later the region editor and the ComfyUI node. Skipped: heads-tall, single-model VLM.
- Open option: the AI path's own tagging could drop the 4 corner crops (full image + zones scored the same in
  tagger-only tests) to save ~10 s per AI read; needs a full-pipeline benchmark first.

## 4. Questions for you
1. ComfyUI node: is calling the binary per prompt / image (`--caption`, plus a `--prompt` mode I'd add) good enough
   for your draft, or do you need a persistent local server?
2. Transfer presets (step 3): Outfit transfer = character image's identity + body, reference image's clothing and
   accessories only; Atmosphere = reference image's light, palette and setting only. Any other split you'd want?
3. Should the UI default threshold move to 0.30 too? It also sets the hints for the AI path, which was measured at
   0.35, so it would need a full-pipeline run first.

Raw data: `tests/eval_local/results/` (run files), `CHANGELOG.md` (every measurement with commit ids).
