# Below-70 fixes: results (Claude → Gemini, 2026-11-10)

Commit `04bac98`. Built: footwear check (bottom strip of the main figures), diagram-inset check (main figures =
boxes ≥ 30% of the biggest: the inset figures are detected as people too, which first blocked the check on
dark_armor), description tags (navel, midriff, outstretched arm, reaching towards viewer), invisible skipped,
"trim" no longer filler, parent tags counted in the scorer. Fix 4 (tall pictures) dropped: `tag_one_fast`
already tags three body zones of tall figures, and fox_vtuber's choker / flower stay invisible to the tagger.

Measured as the user asked: the 5 affected images first (2 rounds), then the other 7 as a regression check.
| Image | step 4 | new (avg of 2) | change |
|---|---|---|---|
| sakura_wings | 57% | 68% | +11 |
| dark_armor | 67% | 74% | +7 |
| fox_vtuber | 71% | 72% | +2 |
| torso_chart | 70% | 73% | +3 |
| bikini_chart | 76% | 71% | -5 (writer added abs, light skin, line art in both rounds; no rule removed anything) |
| the other 7 | — | — | -3 to +3 (noise); footwear kept where it's really there |
10 images: 77 → 78% (step 4 re-scored with the parent rule). Raw: `tests/eval_local/results/run9_*.txt`.
