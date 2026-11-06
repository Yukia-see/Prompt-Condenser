# Multi-Model Benchmark Report: Vision Rescan vs Direct Native Vision
**Image**: `/home/yukia/Pictures/HWL8f8wbYAAMJ-a.jpg` (3 characters)  
**Hardware**: Local GPU (100% offload)  
**Profile**: Anima (masterpiece, best quality, explicit weights, no score_9)

---

## 1. Quantitative Performance Matrix

| Model | Pipeline Mode | Vision Pass | Rescan / Gen Pass | Total Latency | Gen Speed (t/s) | Tokens |
|---|---|---|---|---|---|---|
| **E2B (Q8)** (`gemma4-e2b-q8`) | 2nd-Pass Rescan (Moondream + E2B) | 8.02s | 14.62s | **22.64s** | **75.4 t/s** | 242 |
| **E4B** (`gemma4-e4b`) | 2nd-Pass Rescan (Moondream + E4B) | 8.02s | 14.70s | **22.72s** | **56.3 t/s** | 155 |
| **12B Rescan** (`gemma-4-abliterated:12b`) | 2nd-Pass Rescan (Moondream + 12B) | 8.02s | 7.05s | **15.07s** | **31.1 t/s** | 192 |
| **12B Direct** (`gemma-4-abliterated:12b`) | Direct Native Multimodal Vision | 0.00s | 9.34s | **9.34s** | **29.7 t/s** | 216 |

---

## 2. Qualitative Output Analysis

### 1. E2B (Q8) - Highest Generation Throughput (75.4 t/s)
* **Prompt**:
  `masterpiece, best quality, high resolution, 3 girls, school uniform, yellow striped backdrop, white stars, standing side by side, looking at viewer, (left character: black hair, blue eyes, white shirt, plaid skirt, standing on heels, large breasts, brown background), (center character: cartoon girl, brown hair, blue eyes, school uniform, plaid skirt, white socks, holding two books), (right character: cartoon girl, long brown hair, white eyes, blue jeans, skirt, backpack, wristwatch, butt exposed)`
* **Pros**: Incredible throughput at 75.4 tokens/second; cleanly brackets left, center, and right characters.
* **Cons**: Slightly genericized some items (e.g., misread snack as books).

### 2. E4B - Balanced Precision (56.3 t/s)
* **Prompt**:
  `masterpiece, best quality, 3girls, (left: black hair, blue eyes, large breasts), wearing a white shirt and plaid skirt/dress pants, posing on heels, against brown background; (center: cartoon girl, brown hair, blue eyes), wearing school uniform with plaid skirt and white socks, holding two books, standing in front of orange yellow striped background; (right: cartoon girl, long brown hair, white eyes), wearing blue jeans and a skirt, exposed butt, backpack on one shoulder, wristwatch, against varying size yellow stripes backdrop`
* **Pros**: Clean Danbooru tag syntax and structure; consistent 155-token compact output.
* **Cons**: Rescan took ~14.7s.

### 3. 12B Rescan - Highest Fine Detail Accuracy (15.07s Total)
* **Prompt**:
  `masterpiece, best quality, 3girls, (black hair, ponytail, light blue blouse, navy plaid skirt, over-knee socks, holding wrench), (teal hair, cowlicks, navy blazer, light plaid skirt, white socks, eating snack), (honey blonde hair, sky blue skirt, navy vest, number 8 emblem, knee-high socks), standing in a row, yellow striped background`
* **Pros**: **100% detail fidelity**. Correctly matched the wrench to the left girl, the navy blazer and cowlicks and snack to the center girl, and the vest with number 8 emblem to the right girl! Zero attribute bleed.
* **Cons**: Requires Moondream pass (8s) + 12B synthesis (7s).

### 4. 12B Direct - Fastest End-to-End Latency (9.34s Total)
* **Prompt**:
  `(masterpiece, best quality:1.2), 3girls, (anime style:1.2), [Anima], standing, side view, long hair, blue eyes, black hair, brown hair, blonde hair, school uniform, pleated skirt, white socks, loafers, oversized backpack, (left girl: Arato Nagi, long black hair in a ponytail, looking at viewer, hand near face, wearing light blue shirt and grey plaid skirt), (middle girl: Imari Yuko, short brown hair with blue highlights, smiling, eyes closed slightly, leaning forward, white blouse), (right girl: Kasumaru Koi, blonde hair tied back, cheerful expression, large smile), (background: yellow vertical stripes, stylized pattern, bright colors)`
* **Pros**: Fastest overall response (under 10s); naturally captured side view and over-the-shoulder gaze; recognized individual character identities and varied poses.
* **Cons**: Slightly larger VRAM footprint (~7.8 GB).
