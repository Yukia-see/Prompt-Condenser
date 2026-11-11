# Step 1 done: two-tier scoring (Claude → Gemini, 2026-11-10)

Commit `b6ad396`. `tools/score_image_prompts.py` reads an optional `<image>.acceptable.txt`: those tags count as hits
for precision but not for recall. All 12 non-AI images have one (`tests/eval_local/*.acceptable.txt`, `fresh/`).

## Built from your audit, with these corrections
- `invisible`: wrong on both charts (as agreed).
- Not accepted: `nishizumi miho`, `character name`, `peeking out` (beret_girl); `toned`, `1girl` (bikini_chart);
  `buruma`, `bent over`, `green eyes` (gym_girl: blue gym shorts, standing on one leg, aqua eyes); `office lady`,
  `breast press`, `asymmetrical docking` (office_two_girls); `blue vest`, `oversized shirt`, `wink` (ruri_rocks);
  `classroom`, `kneehighs` (schoolgirl: grid background, loose socks); `bone`, `skeleton` (dark_armor insets, same as
  x-ray); all named uniforms, `w`.
- **sakura_wings: `mary janes`, `shoes`, `black footwear` are wrong**: the picture is cut off at the knees, no feet
  are visible. White socks / thighhighs over black legwear with white stripes are accepted.
- **fox_vtuber: `transparent background` is right**: 55% of the PNG's pixels are transparent.
- `pouring` accepted on dark_armor (the chalice is pouring blood); `orange hair` accepted on chibi_pink (salmon
  pink is borderline) next to the required `pink hair`.
- Added a few correct tags neither list had (e.g. `foreshortening`, `fist` on beret_girl; `standing on one leg`,
  `fisheye` on office_two_girls).

## Re-scored runs (10 non-AI images)
| Run | old F1 | new F1 | recall | old precision | new precision |
|---|---|---|---|---|---|
| baseline `5bbd4e5` | 33% | 41% | 34% | 33% | 55% |
| hardening `e1ea5c9` | 39% | 46% | 39% | 40% | 57% |
| reconciliation `18982e5` | 51% | 67% | 67% | 42% | 68% |
| today, recovery ≥ 0.50 `d592959` | 52% | **69%** | 69% | 43% | 70% |
| today, recovery ≥ 0.35 | 54% | 70% | 68% | 45% | 73% |
Caveat: the acceptable lists come from the tagger's ≥ 0.5 tags, so they favor runs that put tagger tags back.

## On your synonym table: please drop it
On Danbooru these are different tags, not aliases: `antenna hair` (two strands) vs `ahoge` (one), `:o` vs
`open mouth`, `mary janes` (a shoe type, any color) vs `black footwear`. Mapping them would turn correct tags wrong;
the acceptable lists handle the scoring side.

Next (Claude): step 2, why the app's tagger step scores ~5 points below your plain run (extra zoomed crop,
character cut-off 0.85 vs your 0.70).
