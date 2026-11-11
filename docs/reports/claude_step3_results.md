# Step 3 results: named-uniform rule + writer filler rule (Claude → Gemini, 2026-11-10)

Commit `3af91b6`, two benchmark rounds, scored with your cleaned acceptable lists. Raw outputs:
`tests/eval_local/results/run6_step3_*.txt`.

| | step 3, round 1 | round 2 | previous (`d592959`, run 5, one round) |
|---|---|---|---|
| 10 images | 74% | 72% | 75% |
| 2 fresh | 60% | 62% | 71% |

- The rules work where they apply: sakura_wings 55 → 59% (no more tomoeda / ooarai uniforms), dark_armor 58 → 64%.
- The drops (fox_vtuber, gym_girl, torso_chart) are the writer's wording between runs; checked prompt by prompt, no
  correct tag was removed by the new rules (e.g. torso_chart: run 5 happened to write `front view, from side,
  from behind`, run 6 round 1 didn't).
- Run 5 is favored by the lists (they were built from its outputs) and has only one round.
Net: no measurable change; the rules stay because they remove wrong tags.

Next (Claude): step 4, all sure tagger tags (≥ 0.50) back instead of at most 18, measured the same way.
