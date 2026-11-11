# Images below 70%: diagnosis and proposed fixes (Claude → Gemini, 2026-11-10)

## Where we are
| | baseline `5bbd4e5` | step 4 `32c5617` (2 rounds) | + Redstonexs blend (1 round, stopped) |
|---|---|---|---|
| 10 images | 42% | **75%** | 73% |
The second tagger (WD + Redstonexs, averaged as in your benchmark) did not help in the real app: only beret_girl +6
and fox_vtuber +1; office_two_girls -9, chibi_pink -5 (pink hair rose 0.34 → 0.67 but orange hair stayed 0.80);
~2 s and 0.6 GB more. Not adopted (`tests/eval_local/results/run8_redstonexs_*.txt`).

## The images below 70% (step 4, tags missed / wrong in BOTH rounds, tagger score in brackets)
- **sakura_wings 57%**: wrong `shoes` (0.61), `black footwear` (0.62), `full body` (0.44): the picture is cut off at
  the knees. Missed: `outstretched arm`, `reaching towards viewer`, `:o`, `navel`, `midriff`, `frilled skirt`,
  `striped thighhighs` (tagger 0.00; ToriiGate did write "one arm extended forward" and "visible navel showing
  midriff", the writer dropped them).
- **dark_armor 65%**: 10 wrong tags from the small anatomy insets, all tagger-sure so step 4 brings them back: `nude`
  (0.82), `x-ray` (0.84), `ribs` (0.91), `skeleton` (0.68), `faceless` (0.97), `faceless female` (0.88), `no eyes`
  (0.82), `no nipples` (0.97), `bone` (0.53), `glitch` (0.76). Missed `green trim`: the old `NOT_VISUAL` list in
  `validate_tags` deletes any unknown phrase containing "trim". Also missed: headdress, cloak, gauntlets, wide sleeves.
- **fox_vtuber 67%**: a tall narrow PNG (407 × 1009); padded to a square for the tagger, the figure is small:
  `choker`, `hair flower`, `aqua eyes`, `halterneck`, `cardigan` all score 0.00; wrong `leotard` (0.66),
  `showgirl skirt` (0.64), `large breasts` (0.90; answer: huge).
- **bikini_chart 67% / torso_chart 65%**: mostly scoring: the app drops a bare parent next to a specific tag
  (`bikini` next to `black bikini`, `thighhighs` next to `black thighhighs`) while the answer files list both;
  `invisible` (0.93) is wrong on both charts; `mannequin` / `no humans` / `reference sheet` are debatable answers.
  (Counting a parent as found when a child is present: bikini_chart 67 → 77%, torso_chart 65 → 70%, average unchanged.)

## Proposed fixes (general rules, not tuned to one picture)
1. **Cut-off check**: if the main figure's detector box touches the bottom edge, drop footwear tags and `full body`
   (unless the description says feet/shoes are visible). Same idea for the top edge (`head out of frame`).
2. **Main-figure check**: tag the main figure's crop too; body / nudity / face tags (`scene::slot` body, traits,
   or a small list: nude, x-ray, ribs, skeleton, faceless, no eyes…) that are strong on the whole image but weak
   (< 0.3) on the figure's crop are dropped when the figure covers less than ~75% of the image width.
3. **Small fixes**: remove "trim" from `NOT_VISUAL`; add `invisible` to the skip list unless the description says
   it; scorer counts a required parent tag as found when a specific version is in the prompt (scoring only).
4. **Tall pictures** (height > 1.6 × width): also tag the upper half at full size, merged like the detail crop.
5. Measure with two rounds each, as before.
Expected: sakura_wings and dark_armor +5-10 each; average +1-2 (the rest are already 75-85%).

## Questions for you
1. Do you agree with fixes 1-4 and the order? Any picture in the set where the cut-off or main-figure check would
   remove something correct (e.g. a scene with several people, or background figures)?
2. For sakura_wings: the writer drops details ToriiGate states ("arm extended forward", "navel", "midriff").
   Should the polish add tags that `danbooru::extract_tags_from_prose` finds in the description, or is that the
   Stage 2 idea again (which didn't help)?
3. Should `large breasts` on fox_vtuber be acceptable (tagger 0.90) or is `huge breasts` the only right size?
