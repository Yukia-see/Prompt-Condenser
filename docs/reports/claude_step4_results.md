# Step 4 results: every sure tagger tag back (Claude → Gemini, 2026-11-10)

Commit `32c5617`: with scores known, all tagger tags ≥ 0.50 come back after the writer (guard 60; without scores
the old cap of 18). Two rounds each, cleaned acceptable lists. Raw: `tests/eval_local/results/run7_step4_*.txt`.

| | step 3 (round 1 / 2) | **step 4 (round 1 / 2)** |
|---|---|---|
| 10 images F1 | 74% / 72% | **75% / 75%** |
| recall | 70% / 68% | 72% / 73% |
| precision | 79% / 78% | 79% / 78% |
| tags per prompt | ~41 | ~49 |
| 2 fresh F1 | 60% / 62% | **67% / 68%** |

Per image (average of 2 rounds): office_two_girls +10, dark_armor +6, bikini_chart +4, gym_girl +3,
schoolgirl_two_panels +2, rouge_bat +1, chibi_pink 0, fox_vtuber -2, sakura_wings -2, beret_girl -5.
Recall rose without losing precision, unlike the offline estimate (-3 to -6 precision): the cleaned lists and the
named-uniform / filler rules keep the extra tags mostly correct. Busy pictures gain most, as expected.

Overall on the same scoring: baseline `5bbd4e5` 42% → now 75% (10 images). The tagger alone scores ~78%, so
the remaining gap is ~3 points of recall. Next candidates: the WD + Redstonexs combination (your benchmark), tested
in the real app.
