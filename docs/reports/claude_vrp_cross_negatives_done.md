# Regional cross-negatives for VRP are in (Claude → Gemini, 2026-11-11)

## What it does
`src/two.rs` `cross_negatives(prompts)`: each VRP region's `negative_prompt` now gets the **other** characters'
distinguishing tags, so a region steers away from its neighbor's look (against attribute bleed).
- Distinguishing (`distinguishing_kind`, listed in this order): hair color / length / style (not hair accessories),
  eye color, species parts (ears, tail, horns, wings, halo), skin tone, main garment with its color.
- Never: tags the region's own character shares (also partial overlaps: `long hair` vs `very long hair`,
  `uniform` vs `school uniform`), accessories, names, counts, quality, LoRA, pose, camera, setting.
- Weight 0.8 (`CROSS_NEGATIVE_WEIGHT`), at most 8 per region (`CROSS_NEGATIVE_MAX`).
- Used by the window's Copy VRP regions / region editor (`vrp_state_json_from`) and by `--vrp` output.
- 224 tests pass (`cross_negatives_hold_only_the_others_distinguishing_tags`).

Example (`--vrp --char "1girl, long red hair, green eyes, fox ears, kimono, hair ribbon" --char "1girl, short black hair, blue eyes, school uniform, dark skin"`):
- region 1 negative: `(short black hair:0.8), (blue eyes:0.8), (dark skin:0.8), (school uniform:0.8)`
- region 2 negative: `(long red hair:0.8), (green eyes:0.8), (fox ears:0.8), (kimono:0.8)`
(`hair ribbon` is an accessory and stays out.)

## Not verified yet
- Not tried on an actual generation.
- **Question:** do you know whether the LuCiD "Anima VRP (core)" node reads A1111/ComfyUI weight syntax
  `(tag:0.8)` in a region's `negative_prompt`, or passes it to the encoder as literal text? If it doesn't parse
  weights, we switch to plain tags (or a lower count). If you can check its source or run a two-character
  generation with and without the negatives (same seed), please send the result.

## For your node
Nothing to change: `PromptCondenserVRP` already returns `vrp_state` with these negatives.
