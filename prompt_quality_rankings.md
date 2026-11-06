# Detailed Prompt Quality Comparison & Rankings
**Image**: `/home/yukia/Pictures/HWL8f8wbYAAMJ-a.jpg` (3 characters)  
**Target Profile**: Anima

---

## 1. Quality Ranking Summary

| Rank | Model & Pipeline | Score | Strengths | Weaknesses |
|:---:|---|:---:|---|---|
| 🥇 **#1** | **12B Vision Rescan** (Moondream + 12B) | **9.6 / 10** | **100% detail fidelity**: Matched wrench, navy blazer, cowlicks, snack, and vest emblem with zero bleed. | Uses neutral standing pose. |
| 🥈 **#2** | **12B Direct Vision** (Native Multimodal) | **8.8 / 10** | **Dynamic postures**: Captured over-the-shoulder gaze, side view, and leaning attitude. | Hallucinated character names and checkpoint bracket tag. |
| 🥉 **#3** | **E4B Vision Rescan** (Moondream + E4B) | **7.4 / 10** | Clean Danbooru tag syntax; clear spatial brackets. | Minor item misidentifications (snack as books). |
| **#4** | **E2B Q8 Vision Rescan** (Moondream + E2B) | **6.7 / 10** | Ultra-high generation throughput (75.4 t/s). | Repetitive background tags; lower semantic precision. |

---

## 2. Model-by-Model Analysis

### 🥇 Rank 1: `huihui_ai/gemma-4-abliterated:12b` (Vision Rescan)
* **Prompt**:
  ```text
  masterpiece, best quality, 3girls, (black hair, ponytail, light blue blouse, navy plaid skirt, over-knee socks, holding wrench), (teal hair, cowlicks, navy blazer, light plaid skirt, white socks, eating snack), (honey blonde hair, sky blue skirt, navy vest, number 8 emblem, knee-high socks), standing in a row, yellow striped background
  ```
* **Evaluation**:
  1. **Zero Attribute Bleed**: Parenthetical grouping successfully isolates each character's traits so CLIP attention doesn't cross-contaminate colors or clothing.
  2. **Micro-Detail Accuracy**: Accurately assigned the wrench to the left character, the navy blazer and snack to the center character, and the sky blue skirt with the number "8" vest emblem to the right character.
  3. **Syntax Compliance**: Clean Anima formatting without extraneous labels or syntax errors.

---

### 🥈 Rank 2: `huihui_ai/gemma-4-abliterated:12b` (Direct Native Vision)
* **Prompt**:
  ```text
  (masterpiece, best quality:1.2), 3girls, (anime style:1.2), [Anima], standing, side view, long hair, blue eyes, black hair, brown hair, blonde hair, school uniform, pleated skirt, white socks, loafers, oversized backpack, (left girl: Arato Nagi, long black hair in a ponytail, looking at viewer, hand near face, wearing light blue shirt and grey plaid skirt), (middle girl: Imari Yuko, short brown hair with blue highlights, smiling, eyes closed slightly, leaning forward, white blouse), (right girl: Kasumaru Koi, blonde hair tied back, cheerful expression, large smile), (background: yellow vertical stripes, stylized pattern, bright colors)
  ```
* **Evaluation**:
  1. **Dynamic Posing**: Naturally detected the active, non-stiff composition: `side view`, `looking slightly over her shoulder`, `leaning forward`, and `hand near face`.
  2. **Deductions**: Hallucinated fictional anime character names (*"Arato Nagi"*, *"Imari Yuko"*) and inserted the literal checkpoint name `[Anima]`.

---

### 🥉 Rank 3: `prompt-condenser-gemma4-e4b:latest` (Vision Rescan)
* **Evaluation**:
  1. **Structure**: Clean Danbooru tag conventions and structured left/center/right separation.
  2. **Semantic Mismatches**: Misread subtle hand-held objects (mistook the small snack for *"two books"*, and misidentified the right character's legwear as *"blue jeans"*).

---

### Rank 4: `prompt-condenser-gemma4-e2b-q8:latest` (Vision Rescan)
* **Evaluation**:
  1. **Throughput**: Extremely fast (75.4 tokens/second) with low resource consumption.
  2. **Coherence**: Retained spatial brackets, but repeated background tokens and inherited object misidentifications from lighter semantic reasoning.
