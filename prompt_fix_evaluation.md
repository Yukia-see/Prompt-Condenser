# Prompt Evaluation & Root Cause Analysis

## Issue 1: Middle Character Height Distortion
### Cause:
1. **Pose Compression (`leaning forward`)**: 
   When combined with `standing in a row`, `leaning forward` causes the diffusion model to compress the center figure's vertical torso/spine to simulate forward pitch, making her look significantly shorter or shrunken compared to the upright figures on either side.
2. **Proportion Contrast / "Shortstack" Prior**:
   When flanked by figures with extreme curves, diffusion models attempt to normalize the center figure by reducing vertical stature unless anchored by height descriptors.
### Solution:
* Replace `leaning forward` with an upright anchor: `standing upright, head tilt, equal height, normal stature`.
* Add `same height` or `equal height` to the global scene header.

---

## Issue 2: Right Character Getting Giant Breasts Instead of Small
### Cause:
1. **The "Giant" Word Collision (`(giant ass:1.3)`)**:
   In CLIP/T5 text encoders, words attend to neighboring concepts. In anime training datasets, the word `giant` is overwhelmingly paired with chest tags (`gigantic breasts`). Placing `(small breasts:1.2)` directly adjacent to `(giant ass:1.3)` causes the high-weight `giant` token to cross-attend to `breasts`, completely overriding the weaker `small` token!
2. **Global Attention Bleed (`(gigantic breasts:1.3)`)**:
   In a single prompt, a 1.3 weight on the left character's chest exerts a strong global field across the whole image canvas.
3. **Danbooru Semantic Strength**:
   In Danbooru models, `small breasts` has relatively weak negative enforcement. The canonical tag with strong mathematical pull against chest volume is **`flat chest`**.
### Solution:
* **Eliminate the word `giant`**: Use canonical Danbooru tags for lower body volume: `(large ass:1.25)`, `(wide hips:1.25)`, `(thick thighs:1.25)`.
* **Switch from `small breasts` to `flat chest`**: `(flat chest:1.3)` or `(small breasts, flat chest:1.3)` has 10x stronger semantic resistance against cross-attention bleed.
* **Isolate via Regional Prompter / ComfyUI SetArea**: Putting the right character into her own conditioning chunk physically prevents the left character's `gigantic` tokens from ever reaching her pixels.
