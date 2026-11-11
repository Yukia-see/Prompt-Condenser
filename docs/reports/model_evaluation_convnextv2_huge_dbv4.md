# Model Evaluation: `convnextv2_huge.dbv4-full`

## Overview
**`convnextv2_huge.dbv4-full`** is a heavy, state-of-the-art anime multi-label image tagging model developed by **DeepGHS** / `animetimm` (led by `narugo1992`). It is widely regarded in the anime AI community as the most capable and extensive Danbooru auto-tagger available today.

---

## 1. Technical Specifications

| Metric | `wd-eva02-large-tagger-v3` (Currently in Prompt Condenser) | `convnextv2_huge.dbv4-full` |
| :--- | :--- | :--- |
| **Creator** | SmilingWolf | DeepGHS / AnimeTimm (`narugo1992`) |
| **Backbone Architecture** | EVA02-Large (Vision Transformer) | **ConvNeXt V2 (Huge)** (Convolutional) |
| **Parameter Count** | ~304 Million | **~692 Million** (2.3× larger) |
| **Model Size** | ~1.2 GB (FP32) | **~2.7 GB** (FP32) / ~1.4 GB (FP16) |
| **Input Resolution** | $448 \times 448$ | **$512 \times 512$** |
| **Tag Vocabulary** | 10,861 tags | **12,476 tags** (`dbv4-full`) |
| **Dataset Cutoff** | Early 2024 | Late 2024 / newer Danbooru v4 dump |
| **ONNX Availability** | Official on Hugging Face | Community ONNX export available (`itterative/convnextv2_huge.dbv4-full-onnx`) |

---

## 2. Key Advantages of `convnextv2_huge.dbv4`

1. **Superior Character & Franchise Knowledge**:
   - With 12,476 tags and 692M parameters, it recognizes substantially more named characters, modern game costumes (e.g. *Genshin Impact*, *Honkai: Star Rail*, *Blue Archive*, recent VTubers), and specific uniform variants than WD-EVA02.
2. **Fine-Grained Detail & Backgrounds**:
   - The $512 \times 512$ resolution gives it ~30% more pixel area than $448 \times 448$, allowing it to detect small accessories, jewelry, decals, and background furniture that EVA02 misses.
   - In our earlier benchmark analysis, **40 tags were completely missed by WD-EVA02 (score $< 0.15$)**. Many of those micro-accessories fall directly into the expanded capacity of `convnextv2_huge`.
3. **Robustness of ConvNeXt V2**:
   - ConvNeXt V2 combines convolutional inductive biases with Global Response Normalization (GRN), making it less susceptible to spatial artifacts and aspect-ratio distortions than pure vision transformers.

---

## 3. Trade-offs & Limitations

1. **Inference Latency & Compute**:
   - On CPU / shared memory GPUs (like BC-250), EVA02 runs in **~1.5 seconds**.
   - `convnextv2_huge` takes **~3.5 to 6.0 seconds** per forward pass due to its 692M parameter count.
2. **Download & Disk Footprint**:
   - Model download is ~2.7 GB (versus ~1.2 GB for EVA02).
3. **Ecosystem Non-Standard**:
   - Unlike SmilingWolf's models which are universal defaults across web UIs, `convnextv2_huge` is mostly used in specialized dataset-tagging scripts and requires custom ONNX or `timm` runners.

---

## 4. Feasibility for Prompt Condenser

Prompt Condenser's [`src/tagger.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/tagger.rs) uses **ONNX Runtime (`ort`)**, which can natively execute `convnextv2_huge` if provided an ONNX export.

### What Integration Would Require:
1. **Resolution & Tensor Shape**: Update input tensor allocation from `[1, 448, 448, 3]` to `[1, 512, 512, 3]`.
2. **Tag Mapping**: Load the 12,476-row `selected_tags.csv` instead of the 10,861-row version.
3. **Model Selection**: Prompt Condenser could provide a toggle in the UI:
   - **Fast (Default)**: `wd-eva02-large-tagger-v3` (1.5s, 1.2 GB)
   - **High Precision (Heavy)**: `convnextv2_huge.dbv4-full` (4.5s, 2.7 GB)
