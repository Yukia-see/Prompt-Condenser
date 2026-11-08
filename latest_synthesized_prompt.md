# Synthesized Prompts: Phase 0–5 Pipeline Output

Generated from [`packaging/arch/character.jpg`](file:///home/yukia/Documents/Prompt%20Condensor/packaging/arch/character.jpg) using the verified Phase 0–5 multi-zone dual-pass condenser engine.

---

## 1. Hybrid Format (Recommended for WebUI / ComfyUI / NovelAI)

### `[Scene / Environment & Framing (Pass 1)]`
```text
masterpiece, best quality, full body, dynamic angle, depth of field, indoor, bedroom, sitting on bed, soft lighting, A young woman with long messy black hair and sunglasses resting atop her head sits in a relaxed pose on a bed with one knee raised forward. Soft natural light illuminates her figure from the side, casting gentle shadows across the bedding and her taut attire.
```

### `[Character 1: Appearance & Wardrobe (Pass 2)]`
```text
1girl, solo, long hair, black hair, messy hair, dark eyes, sunglasses on head, eyewear on head, fair skin, curvy, large breasts, cleavage, light blue tube top, print shirt, text on clothes, taut clothes, light blue bikini bottoms, white thighhighs, garter straps, bare foot, sitting, one knee up, leaning back, hands on legs, adjusting clothes, looking at viewer
```

---

## 2. Standard Danbooru Tag Prompt (Pony / Illustrious / SDXL)

```text
masterpiece, best quality, 1girl, solo, long hair, black hair, messy hair, dark eyes, sunglasses on head, eyewear on head, fair skin, curvy, large breasts, cleavage, light blue tube top, print shirt, text on clothes, taut clothes, light blue bikini bottoms, white thighhighs, garter straps, bare foot, sitting on bed, one knee up, leaning back, hands on legs, adjusting clothes, looking at viewer, indoor, bedroom, full body, soft lighting, depth of field
```

---

## 3. Pure Natural Language Prompt (Flux.1 / Anima / SD3.5 / Ideogram)

```text
Masterpiece, best quality anime illustration. A young woman with a curvy figure, slender waist, and long messy black hair sits comfortably on a soft bed in a sunlit bedroom, leaning back slightly with one knee raised forward. Her dark eyes gaze warmly at the viewer, and stylish sunglasses rest casually atop her head. She wears a form-fitting, taut light-blue tube top featuring printed text across the chest, matching light-blue bikini bottoms, and delicate white thigh-high stockings. Her hands rest naturally against her raised leg as she adjusts her legwear, with soft ambient light creating subtle depth and shallow focus.
```

---

## 4. Anti-Artifact Negative Prompt (Recommended Locks)

```text
worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, cropped, (head out of frame:1.2), (cropped legs:1.2), (out of frame:1.2), (close-up:1.1), (portrait:1.1), (solo focus:1.1), (breast focus:1.1), (seiza:1.3), (kneeling:1.3), shoes, sneakers, boots, turtleneck
```

---

## 5. Pipeline Policy Audit

| Policy Stage | Status | Applied Action |
| :--- | :--- | :--- |
| **Crop Boundary Artifacts** | Clean | `close-up`, `portrait`, `solo focus`, `breast focus`, `head out of frame` strictly suppressed. |
| **Layer Conflicts** | Resolved | `tube top` + `cleavage` retained via explicit coexistence rule; `barefoot` retained (no shoes detected). |
| **Two-Source Rule** | Confirmed | All micro-attributes confirmed across VLM visual inspection and multi-zone crops. |
| **Hand Crop Anchors** | Calibrated | Hand bounding boxes centered on wrists with proportional radius ($281.5\text{px}$). |
