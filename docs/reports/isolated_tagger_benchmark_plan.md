# Benchmark Plan: 4-Way Standalone Tagger Shootout (Without the Full Chain)

## Goal Description
The user requested an isolated benchmark: **test without the full chain (no ToriiGate vision model, no Gemma LLM writer, no complex polish rules), comparing the taggers by themselves**.

We will test all 4 major anime tagger architectures head-to-head on the test set:
1. **`wd-eva02-large-tagger-v3`** (SmilingWolf) — 10,861 tags (Current Baseline, already local)
2. **`danbooru-tagger-v1`** (Redstonexs) — 24,305 tags (Fine-grained EVA02-Large)
3. **`convnextv2_huge.dbv4-full-onnx`** (AnimeTimm / DeepGHS) — 12,476 tags (Heavy ConvNeXt-V2)
4. **`camie-tagger-v2`** (Camais03) — 70,527 tags (Full-spectrum ViT)

---

## 1. The 4 Competitors in the Shootout

| Model Name | Backbone | Parameters | Tags | Download Size | Input Resolution & Format |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WD-EVA02 v3** | EVA02-Large | ~304M | 10,861 | *Local* (1.2 GB) | $448 \times 448$ BGR |
| **Redstonexs v1** | EVA02-Large (Unfrozen) | ~304M | 24,305 | ~629 MB | $448 \times 448$ BGR/RGB |
| **ConvNeXt-V2 Huge** | ConvNeXt-V2 Huge | ~692M | 12,476 | ~2.6 GB | $512 \times 512$ RGB |
| **Camie v2** | Vision Transformer (ViT) | ~150M | 70,527 | ~752 MB | $448 \times 448$ RGB |

---

## 2. Test Execution Architecture

We will run an isolated benchmark script using `uv` (zero global dependencies) with `onnxruntime`, `pillow`, and `numpy`.

```
                        ┌─── 1. WD-EVA02 v3 (10k tags) ──────────┐
                        │                                        │
[Target Test Images] ───┼─── 2. Redstonexs v1 (24k tags) ────────┼──► [Scorer & Ground Truth]
(chibi_pink,            │                                        │    tools/score_image_prompts.py
 dark_armor,            ├─── 3. ConvNeXt-V2 Huge (12k tags) ─────┤
 sakura_wings,          │                                        │
 + remaining non-AI)    └─── 4. Camie v2 (70k tags) ─────────────┘
```

### Key Questions This Shootout Answers:
1. **Character Recognition**: Which tagger accurately identifies `kinomoto sakura`'s Season 1 battle costume without confusing it with standard school uniforms?
2. **Micro-Accessories**: Does Redstonexs v1 or ConvNeXt-V2 Huge catch the 40 subtle tags that WD-EVA02 missed (e.g. `pocky`, `uwabaki`, `underboob`, `navel`)?
3. **Precision & Insets**: On `dark_armor.jpg`, which tagger focuses on the main full-color armor rather than hallucinating 20+ anatomy tags from the right-hand tutorial insets?
4. **Raw Speed**: Exact milliseconds per image for each model.

---

## 3. Implementation Steps

1. **Setup Download Directory**:
   Create a dedicated directory `tests/tagger_shootout/models/` to hold the ONNX files and CSVs.
2. **Download Model Assets**:
   - `Redstonexs/danbooru-tagger-v1` (`onnx/model.onnx`, `selected_tags.csv`)
   - `itterative/convnextv2_huge.dbv4-full-onnx` (`model.onnx`, `model.onnx_data`, `selected_tags.csv`)
   - `Camais03/camie-tagger-v2` (`camie-tagger-v2.onnx`, `camie-tagger-v2-metadata.json`)
3. **Execution Script (`tools/bench_taggers_shootout.py`)**:
   - Pre-processes each image for each model's native resolution ($448 \times 448$ vs $512 \times 512$).
   - Runs raw ONNX inference across standard confidence thresholds (`0.30`, `0.40`, `0.50`).
   - Evaluates Recall, Precision, F1, and Inference Latency side-by-side against the non-AI answer files.

---

## User Review Required
> [!IMPORTANT]
> The total download for the 3 additional models is approximately **~3.9 GB** (`convnextv2_huge`: 2.6 GB, `camie_v2`: 0.75 GB, `redstonexs`: 0.63 GB).
> Your drive has **213 GB** available.
> 
> Please confirm if you approve downloading these model weights to begin the standalone 4-way shootout!
