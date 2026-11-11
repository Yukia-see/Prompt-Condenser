# Engineering Plan: Anatomical Zone Routing & Perceptual Detection

**Date**: 2026-11-10  
**To**: Claude Code (`claude/image-scan-fixes`)  
**From**: Gemini (Antigravity)  
**Ref**: `src/twoimg.rs`, `src/detector.rs`, `src/prompt_condenser.rs`

---

## 1. The Core Insight: Fixing the Discarded Zone Labels

In [`src/twoimg.rs:234`](file:///home/yukia/Documents/Prompt%20Condensor/src/twoimg.rs#L234), the pipeline currently crops the 3 anatomical zones:
```rust
// Current code in twoimg.rs:234:
for (_, zone_img) in detector::three_zone_crops(&image, b) {
    if let Ok(z_tags) = tagger::tag_dynamic(&zone_img) {
        merged = tagger::merge_detail_crop(&merged, &z_tags);
    }
}
```

Notice the `_`! 
[`detector::three_zone_crops`](file:///home/yukia/Documents/Prompt%20Condensor/src/detector.rs#L309) actually produces labeled slices:
* `"Upper"` $\to$ Head, Hair, Eyes, Face, Neck
* `"Middle"` $\to$ Torso, Neckline, Arms, Upper Clothing, Held Items
* `"Lower"` $\to$ Hips, Legs, Footwear, Bottom Garments

By discarding the label and passing all slices to `CropType::CharacterZone`, the tagger was free to hallucinate `thighs` on a head crop or `choker` on a leg crop. To suppress that noise, `DETAIL_CROP_MIN` was raised to **0.75**, which inadvertently threw away genuine fine accessories (like `choker`, `hair flower`, `earrings`, and `:o` which score between 0.50 and 0.70).

---

## 2. Action Item 1: Anatomical Zone Tag Routing

Instead of one generic `CropType::CharacterZone`, route tags based on the slice label:

```rust
// Proposed change in twoimg.rs:
for (label, zone_img) in detector::three_zone_crops(&image, b) {
    if let Ok(z_tags) = tagger::tag_dynamic(&zone_img) {
        merged = tagger::merge_anatomical_zone_crop(&merged, &z_tags, label);
    }
}
```

### Anatomical Tag Filters:
1. **Upper Zone (`label.starts_with("Upper")`)**:
   - **Allowed**: Hair styles/colors, eye colors, expressions, facial features, headwear, hair accessories, earrings, choker, collar, horns, animal ears.
   - **Confidence Threshold**: Lower from `0.75` down to **`0.45`**!
   - *Why safe*: Because body tags (`thighs`, `navel`, `shoes`) cannot physically be admitted from the head slice, lowering the threshold to 0.45 cleanly recovers `choker`, `aqua eyes`, `hair flower` on `fox_vtuber` and `:o` on `sakura_wings` with **zero body noise**.

2. **Middle Zone (`label.starts_with("Middle")`)**:
   - **Allowed**: Tops, shirts, blouses, jackets, vests, cleavage, bare shoulders, navel, midriff, sleeves, gloves, held items.
   - **Threshold**: **`0.50`**.

3. **Lower Zone (`label.starts_with("Lower")`)**:
   - **Allowed**: Skirts, shorts, pants, panties, thighhighs, socks, barefoot, shoes, boots.
   - **Threshold**: **`0.50`**.
   - *Bonus*: Naturally supersedes the manual bottom-strip check since footwear can only ever come from the lower zone.

---

## 3. Action Item 2: Pure Background / Environment Scanning

### The Current Flaw:
`detector::quadrant_crops(&image)` cuts 4 corners (TL, TR, BL, BR). Since the character is centered, the person's body appears in all 4 quadrant crops, causing character clothing to contaminate environment tags.

### The Fix:
The detector already knows where every character box is (`boxes = detector::detect(&image)`).
1. When characters are present, extract environment crops from the canvas **outside** the character bounding box (or black out the character box).
2. Tagger scanning the background-only image sees `office`, `computer`, `desk`, `monitor`, `bookshelf`, `potted plant`, `cherry blossoms`, `pink background` with **zero character body interference**.

---

## 4. Action Item 3: Geometric Body Proportions ("Heads Tall")

Instead of guessing height and build from noisy text tags:
1. Calculate the ratio of total character height to head height:
   $$\text{heads\_tall} = \frac{b.y1 - b.y0}{head\_b.y1 - head\_b.y0}$$
2. **Deterministic Rules**:
   - $\text{heads\_tall} < 3.8 \implies$ Emit `chibi, short stature` (suppress `tall, long legs`).
   - $\text{heads\_tall} > 7.5 \implies$ Emit `tall, long legs` (suppress `petite, short stature`).
3. Pure pixel math: zero hallucinations, zero regex exceptions.

---

## 5. Expected Results

* Recovers small missing accessories on tall images (`fox_vtuber`, `sakura_wings`).
* Eliminates the remaining cross-body false positives without needing high crop thresholds.
* Pushes pipeline F1 past **80%–82%** using clean visual architecture rather than linguistic post-processing.
