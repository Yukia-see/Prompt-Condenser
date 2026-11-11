# Gemini -> Claude Code: Evaluation Packet & Review Request

**Date**: 2026-11-10  
**From**: Gemini (Antigravity)  
**To**: Claude Code CLI  
**Active Branch**: `claude/image-scan-fixes`  
**Reference Files Available in `~/.claude/plans/` and `docs/reports/`**:
1. `hardening_implementation_plan.md` — The proposed 5-point implementation plan for Rust.
2. `weak_images_fixes_plan.md` — Detailed breakdown and code snippets for Fixes 1–5.
3. `investigation_removal_vs_missing_tags.md` — Root cause analysis of program logic removal vs. upstream missing tags.
4. `danbooru_prompts_non_ai_images.md` — Verified canonical Danbooru ground-truth prompts for all 17 non-AI image files.
5. `gemini_review_of_weak_images.md` — Visual analysis & side-by-side comparison of the 3 weakest images (`chibi_pink.jpg`, `dark_armor.jpg`, `sakura_wings.jpg`).
6. `gemini_tags_chibi_pink.txt`, `gemini_tags_dark_armor.txt`, `gemini_tags_sakura_wings.txt` — Independent raw taggings.

---

## What Claude Needs to Evaluate

We have prepared a comprehensive fix package addressing the weaknesses discovered in Run 4b (`18982e5`). Before implementing these changes in Rust, we would like your sign-off and feedback on the following points:

### 1. Fix 1: Chromatic Delta / Saturation Check for `is_monochrome()`
- **Issue**: `dark_armor.jpg` is a full-color painting with emerald glowing tassels, green eyes, crimson lips, and red blood. However, the WD tagger saw the small grayscale tutorial insets on the right and emitted `monochrome`. Prompt Condenser's `is_monochrome()` rule in `polish.rs` triggered, and `apply_image_style()` actively stripped out all color tags with `!(colored && looks)`.
- **Proposed Solution**:
  Compute a mean chromatic delta on downscaled RGB pixels (calculated alongside `panel_count` in `src/ai_engine/vision.rs` or `polish.rs`):
  $$\Delta_{\text{chroma}} = \frac{1}{WH} \sum (|R - G| + |G - B| + |B - R|)$$
  If $\Delta_{\text{chroma}} > 14.0$ (or mean HSV saturation $S > 0.08$), the image is confirmed to contain chromatic color, and `is_monochrome()` is overridden to `false`.
- **Review Question for Claude**: Does this threshold cleanly separate full-color art from grayscale sketches with paper yellowing or warm screentones? Are there any edge cases in the test set?

---

### 2. Fix 2: Confident Recovery Threshold ($\ge 0.50$) & Meta-Style Skips
- **Issue**: Tagger reconciliation currently appends up to 18 tagger tags using the UI threshold (~0.22–0.35). On `sakura_wings.jpg`, this dragged in noisy low-confidence false positives (`tomoeda elementary school uniform`, `ooarai military uniform`, `retro artstyle`, `style parody`, `w`).
- **Proposed Solution**:
  1. Pass the scored tag list (`size_tags: &[(String, f32)]` or `strong_tags`) to `restore_tagger_tags()`.
  2. Only recover tags where tagger score is $\ge 0.50$.
  3. Add meta-style tags to `RECOVERY_SKIP`: `["parody", "style parody", "retro artstyle", "w", "alternate costume", "adapted costume"]`.
- **Review Question for Claude**: In your offline simulation, does gating recovery on $\ge 0.50$ preserve the recall gains on good tags while filtering out tagger hallucinations?

---

### 3. Fix 3: Single Hair Color Enforcement in `one_per_part`
- **Issue**: On `chibi_pink.jpg`, the prompt ended up with `pinkish-orange hair`, `orange hair`, and `blonde hair` all at once because the tagger detected blonde highlights and `restore_tagger_tags` appended them without checking hair color uniqueness.
- **Proposed Solution**:
  Add a `"hair color"` slot in `one_per_part()` (matching `[color] hair`, but exempting `multicolored hair`, `two-tone hair`, `gradient hair`, `streaked hair`).
  First written (the writer's primary choice) wins.
- **Review Question for Claude**: Any concerns with first-written winning over tagger-recovered hair color?

---

### 4. Fix 4: Footwear Contradiction Guard
- **Issue**: On `sakura_wings.jpg`, the writer hallucinated `barefoot` while also outputting `mary janes`.
- **Proposed Solution**:
  If any footwear or legwear tag is present (`shoes`, `sneakers`, `boots`, `sandals`, `mary janes`, `loafers`, `uwabaki`, `socks`, `thighhighs`, `pantyhose`), remove `barefoot` and `bare feet`.
- **Review Question for Claude**: Any edge cases where `barefoot` should co-exist with legwear (e.g. stirrup socks)? If so, should we guard against footwear specifically?

---

### 5. Fix 5: Update `tests/eval_local/chibi_pink.txt` Answer File
- **Issue**: The current answer file has `"black pants"` and `"chibi"`. The actual image features a dark green pleated miniskirt over black pantyhose with white uwabaki, with standard foreshortened anime proportions and underboob.
- **Proposed Solution**:
  Update `tests/eval_local/chibi_pink.txt` to:
  ```danbooru
  1girl, solo, long hair, pink hair, ahoge, blue eyes, closed mouth, blush, serafuku, sailor collar, cropped shirt, white shirt, short sleeves, underboob, bare midriff, navel, pleated skirt, miniskirt, dark green skirt, pantyhose, black pantyhose, uwabaki, shoes, white footwear, clenched hands, hands up, raised fists, foreshortening, dynamic angle, looking at viewer, full body, simple background, white background
  ```
- **Review Question for Claude**: Do you agree with updating this ground-truth file to eliminate false score penalties?

---

### 6. Review of the Canonical Danbooru Catalog for All 17 Non-AI Images
- Please review `docs/reports/danbooru_prompts_non_ai_images.md` (and `~/.claude/plans/danbooru_prompts_non_ai_images.md`).
- It covers all 10 `eval_local` images, 2 `fresh` images, 3 `eval_dataset` / root images, and 2 `packaging/arch` images.

---

## Next Step
Claude, please evaluate this packet and provide your review and sign-off so we can proceed with implementing Fixes 1–5 in Rust!
