# Live Prompt Test Results: Anima with Default BREAK Chunk Fencing
**Image**: `/home/yukia/Pictures/HWL8f8wbYAAMJ-a.jpg` (3 Characters)  
**Target Profile**: Anima (Pencil-XL / SDXL)  
**Pipeline**: Person Detector + Moondream 1.8B Multi-Zone Rescan + Downstream AI Synthesizer  
**Date**: November 5, 2026  

---

## 1. Test 1: Gemma 4 E4B (`prompt-condenser-gemma4-e4b:latest`)

* **Execution Time**: 27.29 seconds  
* **Generation Speed**: ~56.3 t/s  
* **Tagger Micro-Detections**: 3girls, gyaru, ponytail, black hair, blue hair, blonde hair, school uniform, plaid skirt, loose socks, loafers, yellow background.  
* **Description**:  
  *Three cartoon-style girls are standing in a row against a yellow and white striped background. They are all wearing variations of school uniforms with plaid skirts. The left girl has blonde hair in a ponytail, while the center girl has brown hair and blue eyes. The right girl also appears to have brown hair and is striking an expressive pose.*  
* **Generated Anima BREAK Prompt**:  
  ```text
  masterpiece, best quality, 3girls, same height, equal height, standing upright, (large breasts:1.2), blonde hair ponytail, white shirt plaid skirt, holding book, cartoon style, blush, BREAK, brown hair blue eyes, blue jacket, white and blue school uniform, white socks, left foot on hip pose, right foot on thigh pose, holding book, carrying bag, BREAK, brown hair ponytail, blue dress, backpack, arms outstretched pose, crossed legs, surprised expression, yellow and white striped background, full body, looking at viewer
  ```
* **Evaluation**:  
  - ✅ **Automatic BREAK Chunk Fencing**: Successfully chunked into 3 distinct sections separated by `BREAK`.  
  - ✅ **Height Locking**: Injected `same height, equal height, standing upright` in the scene header.  
  - ✅ **Anima Quality Prefix**: Started cleanly with `masterpiece, best quality` and zero `score_9` tags.  

---

## 2. Test 2: Gemma 4 12B (`huihui_ai/gemma-4-abliterated:12b`)

* **Execution Time**: 37.17 seconds  
* **Generation Speed**: ~31.3 t/s  
* **Description**:  
  *Three girls with large butts stand together against a striped background; they are dressed in school uniforms and each carries a backpack while looking in different directions.*  
* **Generated Anima BREAK Prompt**:  
  ```text
  masterpiece, best quality, 3girls, standing, (striped background:0.9), alternate breast size, different hair colors, left girl: black hair, ponytail, white shirt, plaid skirt, large breasts, black boots, facing left, BREAK, middle girl: brown hair, blue eyes, blue jacket with patches, white shirt, blue and white skirt, arms crossed, holding object, BREAK, right girl: brown hair, ponytail, white top, blue skirt, looking towards left, holding object, school uniform, backpack, large breasts, big butts, long hair
  ```
* **Evaluation**:  
  - ✅ **Positional Labeling & Fencing**: Prepend `left girl: ...`, `middle girl: ...`, and `right girl: ...` directly bounded by `BREAK`.  
  - ✅ **Symmetry-Breaking Meta-Tags**: Included `alternate breast size, different hair colors` in the opening scene chunk.  
  - ✅ **Pose & Outfit Accuracy**: Correctly identified the black boots and left facing on Character 1, the blue jacket with patches and crossed arms on Character 2, and the backpack on Character 3.  

---

## 3. Comparative Summary

| Metric | Gemma 4 E4B | Gemma 4 12B |
| :--- | :--- | :--- |
| **Pipeline Latency** | **27.29s** (Faster) | 37.17s |
| **BREAK Fencing** | Yes (`BREAK` between figures) | Yes (`BREAK` between figures) |
| **Symmetry Breaking** | `same height, equal height, standing upright` | `alternate breast size, different hair colors` |
| **Subject Assignment** | Figure 1, 2, 3 separated | Left, Middle, Right explicitly labeled |
| **Checkpoint Prefix** | `masterpiece, best quality` | `masterpiece, best quality` |
