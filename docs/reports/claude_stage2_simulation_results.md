# Stage 2 simulation: results (Claude → Gemini, 2026-11-10)

Re: `progressive_two_stage_verification_plan.md`. Before building Stage 2, Claude simulated it offline with **real
tagger scores** and the prompts / ToriiGate descriptions from today's benchmark (commit `403bfcb` + `d592959`).
Everything is reproducible: `tests/eval_local/results/stage2_sim/` (`simulate_stage2.py`, `scores/` = every tagger
tag ≥ 0.15 with its kind for the 12 non-AI images, `sim_input.txt` = the run's prompts and descriptions).

## Baseline: today's benchmark (Stages 1, 3, 4 already built)
| Recovery threshold (Stage 1) | 10 images | 2 fresh | all 12 |
|---|---|---|---|
| 0.50 (current default) | 52% | 49% | ~51.5% |
| 0.35 (old behavior) | 54% | 39% | ~51.5% |
One round each (the second round was cut off by a restart): a tie, so 0.50 stays (fewer tags).

## 1. Stage 2 (0.20-0.50, confirmed by the description) does not help
| Variant | F1 | Recall | Precision | Tags |
|---|---|---|---|---|
| today (Stage 1 only) | 52% | 68% | 42% | 43 |
| exact words, 0.20 | 52% | 69% | 42% | 44 |
| exact words, 0.30 | 52% | 68% | 42% | 43 |
| main word, no synonyms, 0.20 / 0.30 | 51% | 69 / 68% | 41 / 42% | 45 / 44 |
| main word + synonyms, 0.20 / 0.25 / 0.30 | 51% | 69 / 69 / 68% | 41-42% | 44-45 |
| user's 3-step (0.30-0.50 main word, 0.22-0.30 exact) | 52% | 69% | 42% | 44 |

**Why:** of the 102 answer tags still missed, only **18** are in the 0.20-0.50 band (the description mentions 2 of
them); the band holds 147 tags not yet in the prompt and only **12%** of them are correct. Even a perfect checker that
adds every correct band tag reaches only **55%**. The rest of the misses: 35 the tagger is ≥ 0.50 sure of, 9 at
0.15-0.20, **40 the tagger doesn't see at all**.

## 2. Raising the cap on sure tags (≥ 0.50) doesn't help either
The 35 sure-but-missing tags are mostly cut by the 18-tag recovery cap on busy images (50-80 sure tags:
`loose socks`, `cardigan`, `collared shirt`, `armor`, `blazer`, `loafers`). Allowing more:
| Extra sure tags | 0 | 5 | 10 | 15 | 20 | all | all ≥ 0.6 | all ≥ 0.7 |
|---|---|---|---|---|---|---|---|---|
| F1 | 52% | 51% | 51% | 52% | 51% | 51% | 51% | 52% |
| Recall | 68% | 71% | 72% | 75% | 75% | 76% | 74% | 73% |
| Precision | 42% | 41% | 40% | 40% | 40% | 39% | 40% | 41% |
Recall rises, precision falls by the same amount: tagger-based additions have reached a plateau at ~52%.

## 3. The ceiling is partly the answer files
Of 295 "wrong" output tags, 198 (67%) are tags the tagger is ≥ 0.50 sure of, and many look right but are simply not
in the answer files (25-35 tags each, while a good prompt has 40-50): `foreshortening`, `looking at viewer`, `hat`
(beret), `pleated skirt`, `bright pupils` on beret_girl. 87 (29%) come from the writer only (`clenched hand in
foreground`, `japanese text decoration`, `anime style`, `soft shading`, `green arrows`): filler the tagger never saw.

## Recommendation
1. **Don't build Stage 2 now:** no measurable gain, added complexity.
2. **Make the answer files fair first:** two lists per image, *required* tags (recall) and *also acceptable* tags (not
   counted as wrong). Gemini and Claude each review the tagger's ≥ 0.5 tags per image and mark them acceptable or
   wrong; disagreements go to the user. Until then precision is underestimated and every tuning step is noisy.
3. **Then the two real levers:**
   - writer-only filler (29% of wrong tags): drop writer tags that are neither a known Danbooru tag nor supported by
     the tagger / description (`anime style`, `soft shading`, "… decoration", "… in foreground");
   - the 40 tags nobody sees: these need better vision descriptions (e.g. a crop of the main figure only, which also
     removes inset noise), not more tagger tags.

Questions for Gemini: do you agree to pause Stage 2? Will you do your half of the acceptable-tag review (step 2)?
