# The 3 weakest images: Claude's opinion and plan, plus a tagging check for Gemini (2026-11-10)

## Context
After the AI-generated images were moved out (`tests/eval_local/ai_generated/`), 10 test images remain. Claude rated
each latest prompt (run 4b, commit `18982e5`) 1-10 by eye; three scored below 6. The user wants Gemini's opinion on
the fixes and an independent tagging of those three images (Claude's answer files may be wrong: one already was).

**On approval:** this file is copied to Gemini's folder as `claude_weak_images_plan.md`. No code changes until the
user brings back Gemini's answer.

## The three images
| Test file | Original | Score (by eye) | Raw output |
|---|---|---|---|
| `tests/eval_local/chibi_pink.jpg` | `~/Downloads/HVFV47ibcAAu1E5.jpg` | 3, revised to ~5 | `tests/eval_local/results/run4b_reconcile_18982e5.txt` |
| `tests/eval_local/dark_armor.jpg` | `~/Downloads/HU96SraWAAE2m7H.jpg` | 5 | same |
| `tests/eval_local/sakura_wings.jpg` | `~/Downloads/HUhWMmdbwAAu1h7.jpg` | 5 | same |

## Diagnosis (from the TAGGER / DESC / PROMPT lines)
- **chibi_pink:** Claude's answer file is wrong: she wears a dark green pleated skirt, black pantyhose and white
  uwabaki, with underboob and navel showing (it said "black pants"). Real problems: three hair colors in one prompt
  (pinkish-orange / orange / blonde; the tagger lists orange and blonde) and no `chibi` (the tagger missed it).
- **dark_armor:** a full-color painting, but the tagger said `monochrome` (the small grey anatomy insets), ToriiGate
  was given that tag and also called it a "monochrome sketch", and the sketch rule then stripped colors.
  `nude, x-ray, skeleton, faceless, no eyes, heart (organ)` come from the insets.
- **sakura_wings:** ToriiGate described the outfit correctly (navy sailor top with red trim, white pleated miniskirt,
  black thighhighs with white stripes, wings, arms reaching forward). The tagger guessed named uniforms
  (`tomoeda elementary school uniform`, `ooarai military uniform`) plus `style parody`, `retro artstyle`, `w`,
  `mary janes`, `white socks`, and the new recovery step (top-18 tagger tags) put those wrong tags back.

## Plan (general fixes, none tuned to one picture)
1. **Color check before trusting "monochrome"** (`src/ai_engine/polish.rs`, `src/ai_engine/vision.rs`): measure the
   image's mean saturation next to `panel_count` (same downscaled image). A clearly colorful picture is never
   treated as monochrome: `is_monochrome` gets the flag, the monochrome tags are dropped, the sketch note isn't sent,
   and the tags given to ToriiGate leave out `monochrome` / `greyscale`. Tests with a synthetic color vs grey image.
2. **Recover only fairly sure tagger tags (≥ 0.5)** (`restore_tagger_tags`): pass the scored tags (generalize
   `ImageAiRequest.size_tags` to the scored tag list, filled where `strong_tags` is, `src/main.rs` ~L1219 / L2065 and
   `live_image`) instead of the display-threshold string; also skip style labels (`parody`, `style parody`,
   `retro artstyle`, `w`) in `RECOVERY_SKIP`. Check the offline simulation first to choose the threshold.
3. **One hair color per character** (`one_per_part`, polish.rs): a "hair color" part like the existing eye-color one
   (color word + `hair`; not for `multicolored hair` / `two-tone hair` / `gradient hair` / streaks). The first wins.
4. **Contradictions:** `barefoot` goes when footwear is listed (`one_per_part` or `drop_contradicted_hair`'s
   neighbor).
5. **Fix Claude's chibi_pink answer file** (pleated skirt, green skirt, black pantyhose, uwabaki, underboob, navel,
   crop top, long hair, orange/pink hair), after Gemini's tagging below.
Later, only if still needed: tag the main figure crop (person detector) to drop inset-only tags; a lower threshold
for style tags like `chibi`.

## Request to Gemini
1. **Your opinion** on fixes 1-5 and their order. Anything that would hurt other images?
2. **Tag the three images yourself**, without reading Claude's answer files first: open each image and write the
   Danbooru tags a correct prompt should contain (characters, hair, eyes, outfit layers, pose, framing,
   background, style). Write them to `gemini_tags_chibi_pink.txt`, `gemini_tags_dark_armor.txt` and
   `gemini_tags_sakura_wings.txt` in this folder.
3. **Then compare** with Claude's answer files (`tests/eval_local/<name>.txt`) and with the app's prompt and tagger
   tags in `run4b_reconcile_18982e5.txt`: list tags all three sources agree on, tags only Claude has, tags only you
   have, and tags the app/tagger has that you consider wrong. Put this in `gemini_review_of_weak_images.md`.
   Specific questions: chibi_pink hair color (pink, orange or salmon?) and whether `chibi` applies; dark_armor is
   it monochrome anywhere besides the insets; sakura_wings the outfit name (her battle costume, not a school uniform?).

## Verification (after Gemini's answer, when implementing)
`cargo test` with new tests for 1-4; `tools/eval_images.sh` twice; compare the 10 non-AI images and the 3 targets
with run 4a/4b. No package build.
