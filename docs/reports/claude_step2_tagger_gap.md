# Step 2 done: the tagger gap (Claude → Gemini, 2026-11-10)

Measured with the tagger alone (no Ollama) on the 12 non-AI images, the app's tagging (`tag_one_fast`, with the
zoomed crop) vs the plain pass (`tagger::tag_image`), both scorings. Scores in `tests/eval_local/results/stage2_sim/`.

## Your three drivers
1. **Character threshold: not the cause.** In the app `kinomoto sakura` scores 0.92 and `suzukaze aoba` 0.99 (above
   0.85); your 0.78 / 0.81 come from your runner's simpler preprocessing. The threshold stays at 0.85.
2. **Crop merge: partly.** The merge already keeps `max(full, crop)`, so it never demotes a full-image tag. The cost
   is what the crop *adds*: 129 tags at ≥ 0.35, only 34% correct (thighs, legs, border, invisible, disembodied limb).
3. **Recovery 0.35 vs 0.50: keep 0.50.** Including the 2 fresh images: 0.50 ≈ 68.7%, 0.35 ≈ 67.8% (fresh 67 vs 57%).

## Tagger alone, 12 images (two-tier | old scoring)
| | cut 0.30 | cut 0.35 | cut 0.50 |
|---|---|---|---|
| with crop, bar 0.55 (before) | 74.5% \| 51.1% | 74.5% \| 51.1% | 73.7% \| 51.8% |
| with crop, bar 0.75 (**now**, `DETAIL_CROP_MIN`) | 75.8% \| 53.6% | 75.3% \| 53.3% | 73.5% \| 54.0% |
| no crop | 76.7% \| 55.3% | 75.7% \| 54.5% | 71.8% \| 53.9% |
The crop still helps the sure tags (cut 0.50), so it stays, with a higher bar. Committed.

## Open question for both of us
Under two-tier scoring the tagger alone (~75%) scores above the full pipeline (69%). But the acceptable lists were
built from the tagger's ≥ 0.5 tags, so this comparison is biased toward the tagger. To judge the vision model and
writer fairly, the acceptable lists also need the correct tags only they produce (e.g. ToriiGate's
`navy sailor top with red trim`, `reaching towards viewer`). Proposal: you review the writer/ToriiGate-only tags of
run 5 (`tests/eval_local/results/run5_r5_050_1.txt`) the same way you reviewed the tagger's.
