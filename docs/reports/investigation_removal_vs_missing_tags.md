# Investigation: Program Removal vs. Upstream Missing

This document diagnoses whether specific missing tags, items, or test images were **actively removed by Prompt Condenser's program logic** or **straight up missing** from upstream models, metadata, or the source images.

---

## Executive Summary Matrix

| Item / Subject | Context | Removed by Program Logic? | Straight Up Missing Upstream? | Exact Pipeline Cause |
| :--- | :--- | :---: | :---: | :--- |
| **Colors on `dark_armor.jpg`** (`green trim`, `green eyes`, `red blood`) | Test Run 4b | **YES (Removed by Program)** | No (Tagger had them) | `is_monochrome()` triggered in [`polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L422), which actively stripped all color tags via `!(colored && looks)`. |
| **`chibi` on `chibi_pink.jpg`** | Prompt output | No | **YES (Straight up missing)** | WD tagger score for `chibi` was below threshold; ToriiGate described full-body foreshortening; character is standard proportions, not super-deformed. |
| **`black pants` on `chibi_pink.jpg`** | Test answer file | No | **YES (Missing from image)** | Source image depicts a dark green pleated miniskirt over black pantyhose. "Black pants" was a human error in Claude's initial ground-truth answer file. |
| **`two_blondes.jpg`** (`HWP1dffawAA5Y7K.jpg`) | Test set filtering | No | **YES (Missing metadata)** | File had zero EXIF/C2PA metadata. Program could not auto-detect AI origin; user manually identified it as AI, and it was moved to `ai_generated/`. |
| **`thighhighs` on `sakura_wings.jpg`** | Prompt output | No | **YES (Straight up missing)** | WD tagger only predicted `socks, white socks`; Gemma-4-E4B writer dropped ToriiGate's description and wrote `barefoot`. |
| **Character Names** (e.g. `lillie (pokemon)`, `suzukaze aoba`) | Image reads | Conditional | Conditional | [`polish.rs` L326](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L326) actively removes character names output by the vision model unless the tagger also recognized them. |
| **Broad Category Tags** (`wings`, `jacket`, `dress`, `pants`, `boots`) | Tagger reconciliation | **YES (Removed by Program)** | No | Explicitly suppressed by [`RECOVERY_SKIP`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L355) so specific tags aren't diluted by generic parents. |

---

## Detailed Case Analysis

### 1. `dark_armor.jpg` Color Tags: Removed by Program Logic

- **What happened**: The image is a full-color painting with emerald green glowing embroidery, green eyes, crimson lips, and red blood dripping into a chalice. However, the final prompt in Run 4b was forced to `monochrome, sketch, lineart` and stripped of all color tags.
- **Root Cause**:
  1. The WD tagger detected `monochrome` with high confidence due to the small grayscale anatomy sketches on the right-hand side.
  2. In [`src/ai_engine/polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L422-L424):
     ```rust
     pub fn is_monochrome(tagger_tags: &str) -> bool {
         tagger_tags.split(',').map(crate::danbooru::key).any(|t| MONOCHROME_TAGS.contains(&t.as_str()))
     }
     ```
  3. When `is_monochrome` is true, [`apply_image_style`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L485-L505) executes:
     ```rust
     tags.retain(|t| {
         // ...
         let colored = words.iter().any(|w| HUE_WORDS.contains(w) || (line_drawing && NEUTRAL_WORDS.contains(w)));
         let looks = crate::parser::parse_prompt(t).into_iter().next().is_some_and(|tok| {
             matches!(crate::scene::Slot(&tok), crate::scene::Slot::Hair | crate::scene::Slot::Eyes | crate::scene::Slot::Outfit)
         });
         !(colored && looks) // Strips colored hair, eyes, and outfit tags!
     });
     ```
  4. **Verdict**: **The program actively removed the tags.**
  5. **Fix**: Add a mean chromatic delta / saturation check on the image pixels so colorful artwork is never treated as monochrome simply because of insets or borders.

---

### 2. `chibi` on `chibi_pink.jpg`: Straight Up Missing

- **What happened**: The final prompt lacked `chibi`.
- **Root Cause**:
  1. **WD Tagger**: Did not output `chibi` above threshold.
  2. **ToriiGate Vision**: Output: `"A full body portrait featuring a single girl with messy pinkish-orange short hair and blue eyes. She is posed dynamically in foreshortening, raising her hands near her face while wearing a white sailor uniform shirt that lifts to reveal underboob and navel."` (No mention of chibi).
  3. **Visual Inspection**: The character is drawn with standard anime body proportions, defined waist, underboob, and legs, captured from an extreme high angle with perspective foreshortening. It is **not** a true super-deformed (2-head-tall) chibi.
  4. **Verdict**: **Straight up missing upstream.** The program did not remove it; the models did not predict it, and the image does not actually depict a true chibi.

---

### 3. `black pants` on `chibi_pink.jpg`: Straight Up Missing from Image

- **What happened**: The original ground-truth test file `tests/eval_local/chibi_pink.txt` contained `black pants`, but neither the tagger nor the writer ever produced it.
- **Root Cause**:
  1. The girl is actually wearing a **dark green pleated miniskirt** over **black pantyhose** with white **uwabaki** shoes.
  2. Claude's initial handwritten answer file mistakenly misidentified the black pantyhose as pants.
  3. **Verdict**: **Missing from the source artwork.** The program was correct; the answer key was wrong.

---

### 4. `two_blondes.jpg` (`HWP1dffawAA5Y7K.jpg`): Straight Up Missing Metadata

- **What happened**: 12 AI images were automatically segregated into `tests/eval_local/ai_generated/`, but `two_blondes.jpg` initially stayed in the test set until commit `1584f71`.
- **Root Cause**:
  1. The 12 AI images had detectable C2PA, SynthID, or ComfyUI metadata blocks (`prompt`, `workflow`).
  2. `two_blondes.jpg` was completely stripped of metadata (`keys=['jfif', ...]`). The automated scanner had no metadata to flag.
  3. You (the user) recognized the visual style as AI-generated and notified Claude. Claude then manually moved the file to `ai_generated/`.
  4. **Verdict**: **Straight up missing metadata.** The program could not detect it without metadata; manual user intervention moved it.

---

### 5. `thighhighs` on `sakura_wings.jpg`: Straight Up Missing from Models

- **What happened**: The prompt lacked `thighhighs` or `striped thighhighs`, and instead included `barefoot`.
- **Root Cause**:
  1. The WD tagger output: `socks, white socks, black footwear, mary janes` (it missed `thighhighs`).
  2. ToriiGate saw them: `"black thigh-high socks with white stripes"`.
  3. Gemma-4-E4B wrote: `"miniskirt, white stripes, wings, hat, ... barefoot, mary janes"`. It failed to convert ToriiGate's prose into the tag `thighhighs`, and hallucinated `barefoot`.
  4. Because the tagger did not have `thighhighs` either, tagger reconciliation could not recover it.
  5. **Verdict**: **Straight up missing upstream from the models.** The program did not strip it.

---

### 6. Broad Tags Suppressed by `RECOVERY_SKIP`

- In [`src/ai_engine/polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L355-L360):
  ```rust
  const RECOVERY_SKIP: &[&str] = &[
      "breasts", "shirt", "skirt", "socks", "shoes", "jewelry", "underwear", "tail", "animal ears",
      "multiple girls", "multiple boys", "thighs", "legs", "ribbon", "bag", "food", "plaid", "clothes",
      "bow", "gloves", "wings", "jacket", "belt", "boots", "dress", "pants", "shorts", "sleeves",
      "virtual youtuber", "alternate breast size", "character name", "copyright name", "artist name",
      "twitter username", "signature", "watermark",
  ];
  ```
  - When the tagger outputs generic tags like `wings`, `jacket`, or `dress`, the recovery step intentionally **removes/skips** them so that more specific tags (`angel wings`, `blue jacket`, `black dress`) aren't duplicated or diluted.
  - **Verdict**: **Actively suppressed by program design.**

---

## Action Plan & Verification

1. **Fix 1 (Chromatic Check)**:
   - Implement `mean_chromatic_delta` in `src/ai_engine/vision.rs` to stop `dark_armor.jpg`'s colors from being removed by `is_monochrome()`.
2. **Fix 2 (Confident Recovery Threshold)**:
   - Gate tagger recovery on score $\ge 0.50$ and add meta-style tags (`parody`, `style parody`, `retro artstyle`, `w`) to `RECOVERY_SKIP`.
3. **Fix 3 (Single Hair Color)**:
   - Add single hair color enforcement in `one_per_part` so `chibi_pink` doesn't retain pink, orange, and blonde simultaneously.
4. **Fix 4 (Contradiction Guard)**:
   - Drop `barefoot` when footwear/socks are present (`sakura_wings`).
5. **Fix 5 (Update Answer File)**:
   - Update `tests/eval_local/chibi_pink.txt` to reflect the verified ground truth.
