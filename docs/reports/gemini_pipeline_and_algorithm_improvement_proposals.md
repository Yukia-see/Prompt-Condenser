# Algorithmic & Rule-Wise Improvements for the Vision Pipeline

**Date**: 2026-11-10  
**Author**: Gemini (Antigravity)  
**Target**: Prompt Condenser Vision Pipeline (`src/ai_engine/polish.rs`, `src/twoimg.rs`, `src/danbooru.rs`)  
**Based on**: Forensic audit of Run 9 results across all 12 test images (`run9_r9_affected_1.txt`, `run9_r9_rest_1.txt`)

---

## 1. Executive Summary

With commit `04bac98` (footwear bottom-strip check, inset diagram suppression, and prose extraction), the pipeline has reached **78% F1** (up from 42% at baseline).

A forensic audit of all remaining false-positive tags and missed tags in Run 9 reveals **5 distinct, actionable structural patterns**:
1. **Unsplit Compound Color/Modifier Garments**: E4B outputs natural English phrases like `white pleated skirt` or `dark blue blazer`, which fail Danbooru exact matching.
2. **Meta & Drawing Instruction Vocabulary Leakage**: Terms like `anatomy diagram`, `figure study`, `tutorial image`, and `letterboxing` slipping into prompts.
3. **Common Danbooru Spelling & Plural Synonyms**: Minor grammatical variances (`clenched fists` vs `clenched hands`, `twin tails` vs `twintails`, `knee-highs` vs `kneehighs`).
4. **Physique & Muscle Hallucination on Exposed Midriffs**: E4B systematically invents `abs`, `toned`, and `athletic` whenever a bikini or bare midriff is visible.
5. **Head & Accessory Resolution Loss on Tall Narrow Ratios**: Full-body square padding on $2.5:1$ images downsamples faces and neckwear below tagger resolution limits.

Implementing these 5 targeted enhancements is projected to lift overall pipeline F1 from **78% $\to$ 82%–84%**.

---

## 2. The 5 Algorithmic & Rule Improvements

### Improvement 1: Compound Garment Modifier Decomposition in `known_pieces`
* **The Root Cause**:
  In [`src/ai_engine/polish.rs:259`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs#L259):
  ```rust
  let only_colors_left = !leftover.is_empty() && leftover.iter().all(|w| COLOR_WORDS.contains(w) || MATERIAL_WORDS.contains(w));
  if pieces.is_empty() || only_colors_left || !head_covered {
      return None;
  }
  ```
  `only_colors_left` was intended to keep descriptive phrases (like `teal bikini top`) for Anima image generation. However, when E4B outputs standard anime outfit elements, this causes non-Danbooru compounds to bypass decomposition:
  * `dark blue blazer` (Danbooru has `blazer` and `blue jacket`; not `dark blue blazer`)
  * `white pleated skirt` (Danbooru has `white skirt` and `pleated skirt`; not `white pleated skirt`)
  * `gold hoop earrings` (Danbooru only has `hoop earrings`)
  * `white sneakers` (Danbooru has `sneakers` and `white footwear`)
  * `red hair bobbles` (Danbooru only has `hair bobbles`)
  * `white gym shirt` (Danbooru has `gym shirt` and `white shirt`)
  * `blue plaid pleated skirt` (Danbooru has `plaid skirt` and `pleated skirt`)
* **The Algorithmic Fix**:
  If the full compound phrase is not in Danbooru vocabulary, **do not leave it whole**. Instead:
  1. Extract the base known garment (`blazer`, `pleated skirt`, `hoop earrings`, `sneakers`, `hair bobbles`).
  2. If the color/material modifier pairs with the base category in Danbooru (e.g., `white skirt`, `white footwear`), emit both canonical pieces (`pleated skirt, white skirt`). Otherwise, emit the base tag.
* **Impact**: Eliminates **8+ wrong tags** and simultaneously scores **4+ missed required tags** across `office_two_girls`, `schoolgirl_two_panels`, `sakura_wings`, and `ruri_rocks`.

---

### Improvement 2: Expansion of Drawing & Meta Filler Vocabulary
* **The Root Cause**:
  Commit `3af91b6` introduced `strip_meta_words` and `META_WORDS`, but the list only includes drawing styles:
  `["style", "styled", "shading", "shaded", "decoration", "decorations", "decorative", "foreground", "rendering", "rendered", "drawn", "artstyle"]`.
* **Current Leaks in Run 9**:
  * `detailed anatomy diagram in background` (`dark_armor`)
  * `figure study`, `detailed anatomy`, `black letterboxing` (`torso_chart`)
  * `tutorial image` (`bikini_chart`)
  * `fisheye lens effect` (`office_two_girls`)
  * `standing on plain background`, `foreshortened hands reaching forward` (`chibi_pink`)
* **The Algorithmic Fix**:
  Expand `META_WORDS` and `NOT_VISUAL` in [`src/ai_engine/polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs) to include technical and instructional nouns:
  ```rust
  const META_WORDS: &[&str] = &[
      "style", "styled", "shading", "shaded", "decoration", "decorations", "decorative",
      "foreground", "rendering", "rendered", "drawn", "artstyle",
      "diagram", "tutorial", "study", "anatomy", "effect", "lens", "letterboxing", "letterboxed"
  ];
  ```
  And map common phrasing in `phrase_to_tag`:
  * `"fisheye lens effect"` $\to$ `"fisheye"`
  * `"black letterboxing"` $\to$ `""` (strip)
  * `"tutorial image"` | `"figure study"` | `"detailed anatomy"` $\to$ `""` (strip)
* **Impact**: Eliminates **6 wrong tags** immediately across chart and complex images.

---

### Improvement 3: High-Frequency Danbooru Synonym & Plural Normalization
* **The Root Cause**:
  The LLM writer occasionally uses natural English plurals or space-separated variations that are cataloged differently in Danbooru's schema:
* **The Mapping Fix (`phrase_to_tag` in `polish.rs`)**:
  ```rust
  "clenched fists" => "clenched hands",
  "twin tails" => "twintails",
  "knee-highs" | "knee highs" => "kneehighs",
  "triangle top" => "triangle bikini",
  "white angel wings" => "white wings",
  "floral pattern" => "flower motif",
  "thick outlines" => "outline",
  "hands resting at sides" => "arms at sides",
  "long hair over one eye" => "hair over one eye",
  ```
* **Impact**: Directly converts 6 false positives into 6 true positive hits.

---

### Improvement 4: Physique & Muscle Hallucination Guard
* **The Root Cause**:
  On `bikini_chart`, Gemma 4 E4B wrote `abs`, `toned`, `athletic`, and `light skin`, causing a **5-point F1 drop** between runs. Whenever an LLM writer sees a character in swimwear or with an exposed stomach, it exhibits a strong prior toward hallucinating fitness descriptors.
* **The Algorithmic Fix**:
  In [`src/ai_engine/polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs), implement an unbacked physique guard:
  ```rust
  const PHYSIQUE_TAGS: &[&str] = &["abs", "toned", "muscular", "athletic", "ribs", "defined abs"];
  ```
  If any tag in `PHYSIQUE_TAGS` is present in the prompt, check `tagger_scores`. If the tagger confidence is $< 0.20$ and the vision description did not explicitly describe abdominal muscles, **drop the tag**.
* **Impact**: Insulates the pipeline from LLM writer stochasticity and recovers the 5-point drop on `bikini_chart`.

---

### Improvement 5: Dedicated Face & Collar Crop for Extreme Aspect Ratios ($h > 1.8w$)
* **The Root Cause**:
  On `fox_vtuber.png` ($407 \times 1009$, $2.5:1$ ratio), padding the full figure to a $448 \times 448$ square compresses the head and neck into an area smaller than $70 \times 70$ pixels. As a result, `aqua eyes`, `black choker`, `hair flower`, and `earrings` are completely unresolvable by the tagger (scoring 0.00).
* **The Algorithmic Fix**:
  In [`src/twoimg.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/twoimg.rs):
  When an image has aspect ratio $h / w > 1.8$, take a **head-and-torso crop** (e.g. top 35% of the detected person box, padded to square) and run a fast tagger pass at native $448 \times 448$.
  * Any accessory or facial tag scoring $\ge 0.50$ on this crop (`choker`, `aqua eyes`, `hair ornament`, `hair flower`, `earrings`) is merged into the tagger pool.
* **Impact**: Solves the resolution bottleneck on tall character portraits and lifts `fox_vtuber` recall by +3–4 tags.

---

## 3. Summary of Expected Gains

| Area | Change | Target Images | Expected F1 Impact |
|---|---|---|---|
| **Garment Modifiers** | Decompose non-Danbooru color compounds into valid pieces | `office_two_girls`, `schoolgirl_two_panels`, `sakura_wings` | **+2–3%** |
| **Meta Vocabulary** | Strip technical/diagram terms (`anatomy`, `tutorial`, `study`) | `dark_armor`, `torso_chart`, `bikini_chart` | **+1–2%** |
| **Synonyms** | Map `clenched fists`, `twin tails`, `knee-highs` | `gym_girl`, `schoolgirl_two_panels`, `sakura_wings` | **+1%** |
| **Muscle Guard** | Filter unbacked `abs` / `toned` unless tagger $\ge 0.20$ | `bikini_chart`, `torso_chart`, female portraits | **+1–2%** |
| **Tall Aspect Crop** | Native $448 \times 448$ head/neck crop on $h/w > 1.8$ | `fox_vtuber`, tall pin-ups | **+2–3%** |
| **Overall Pipeline** | | **All 12 Images** | **78% $\to$ 82%–84%** |
