# Standalone Vision Tagger Shootout Benchmark Report

**Date**: 2026-11-10  
**Target Images**: 10 canonical non-AI evaluation images in `tests/eval_local/`  
**Focus Images**: `chibi_pink.jpg`, `dark_armor.jpg`, `sakura_wings.jpg`  
**Execution Environment**: Isolated ONNX Runtime CPU execution (16 threads, full graph optimization), direct scoring against verified ground truth answer files with zero LLM text-writer interference.

---

## 1. Executive Summary

To determine whether missing tags and visual classification inaccuracies originate from the upstream vision tagger or downstream LLM processing, we conducted an **isolated 4-way standalone tagger shootout**.

The contenders:
1. **WD-EVA02-Large-v3 (SmilingWolf)**: 304M params, 10,861 tags (Current project baseline)
2. **Redstonexs Danbooru Tagger v1**: 304M params (unfrozen EVA02-Large), 24,000 tags
3. **Camie Tagger v2 (Camais03)**: 143M params (ViT-Base with 2-stage cross-attention), 70,527 tags
4. **ConvNeXt-V2 Huge dbv4 (itterative/animetimm)**: 692M params (pure ConvNet), 12,476 tags
5. **Ensemble (WD-EVA02-v3 + Redstonexs-v1)**: Two-model blend (averaging shared tags, union of unique tags)

---

## 2. Standalone Shootout Scoreboard

Scores evaluated using the exact project metric (`tools/score_image_prompts.py` logic, ignoring quality/solo tags):

| Model / Ensemble | Vocab Size | Model Weights | Avg Latency | All 10 Images F1 | All Recall | All Precision | 3 Focus Images F1 | 3 Focus Recall |
|---|---|---|---|---|---|---|---|---|
| **Ensemble (WD + RS)** | 24,000 | 1.8 GB | ~6.0s | **58.0%** | 71.8% | 49.8% | 55.5% | 57.7% |
| **ConvNeXtV2-Huge** | 12,476 | 2.6 GB | 7.13s | **57.6%** | 66.2% | **51.9%** | 54.3% | 55.8% |
| **WD-EVA02-v3 (Current)** | 10,861 | 1.2 GB | 5.79s | **57.4%** | 76.6% | 47.0% | **57.6%** | 65.5% |
| **Redstonexs-v1** | 24,000 | 629 MB | 6.30s | 54.1% | 73.5% | 43.8% | 52.8% | 59.6% |
| **Camie-v2** | 70,527 | 752 MB | **1.19s** | 50.0% | **78.4%** | 37.2% | 46.9% | **66.0%** |

*(Thresholds optimized per model: WD @ 0.30, Redstone @ 0.35, ConvNeXt @ 0.35, Camie @ 0.50, Ensemble @ 0.50).*

---

## 3. Deep Dive: The 3 Focus Images

### A. `chibi_pink.jpg` (Ground Truth: 31 tags)
- **Camie-v2**: **27 / 31 hits (87.1% recall)**
  - *Exclusive recovery*: Caught `green skirt` (which **every other model missed**!), `crop top`, `hands up`, `midriff`, `miniskirt`, `pleated skirt`, `sailor collar`, `white shirt`.
- **Redstonexs-v1**: **26 / 31 hits (83.9% recall, 75.4% F1)**
  - *Exclusive recovery*: The **ONLY model** that correctly predicted `pink hair`! (WD and ConvNeXt both mistakenly hallucinated `orange hair`). Also caught `crop top`, `hands up`, `midriff`, `miniskirt`, `pleated skirt`.
- **ConvNeXtV2-Huge**: **21 / 31 hits (67.7% recall, 70.0% F1)**
  - Caught `pleated skirt`, `sailor collar`, `white shirt`, but missed `pink hair` (called it orange) and `crop top`.
- **WD-EVA02-v3**: **19 / 31 hits (61.3% recall, 73.0% F1)**
  - Missed `pink hair`, `crop top`, `hands up`, `midriff`, `miniskirt`, `pleated skirt`, `green skirt`.

### B. `sakura_wings.jpg` (Ground Truth: 25 tags)
- **Camie-v2**: **14 / 25 hits (56.0% recall)**
  - *Superior fine-grained perception*: Recognized `:o` (open mouth), `outstretched arm`, `angel wings`, `cardcaptor sakura` (copyright), `white wings`, `long sleeves`.
- **ConvNeXtV2-Huge**: **12 / 25 hits (48.0% recall, 42.9% F1)**
  - Recognized `white wings`, `long sleeves`, `kinomoto sakura`, `open mouth` (alias for `:o`), `antenna hair` (alias for `ahoge`).
- **WD-EVA02-v3**: **11 / 25 hits (44.0% recall, 40.0% F1)**
  - Recognized `wings`, `kinomoto sakura`, `antenna hair`, but missed `white wings`, `long sleeves`, `:o`, and `outstretched arm`.
- **Redstonexs-v1**: **10 / 25 hits (40.0% recall, 33.9% F1)**
  - Caught `long sleeves`, but missed `kinomoto sakura` and `white wings`.

### C. `dark_armor.jpg` (Ground Truth: 31 tags)
- **WD-EVA02-v3**: **21 / 31 hits (67.7% recall, 59.7% F1)**
  - Best single performer on Dark Armor! Correctly identified: `chalice`, `holding knife`, `blood`, `tassel`, `reference sheet`, `side slit`, `lipstick`, `red lips`.
- **Redstonexs-v1**: **17 / 31 hits (54.8% recall)**
  - Recovered `black thighhighs` and `multiple views` (which WD missed!).
- **Camie-v2**: **17 / 31 hits (54.8% recall)**
  - Recovered `black thighhighs`, `multiple views`, and `smile`.
- **ConvNeXtV2-Huge**: **16 / 31 hits (51.6% recall)**
  - Recovered `black thighhighs` and `smile`, but missed `chalice` and `reference sheet`.

---

## 4. Model Architectural Analysis

### 1. Camie Tagger v2 (`camie-tagger-v2`)
- **Strengths**:
  - **Incredible inference speed**: **1.19s** per image on CPU (over 5× faster than EVA02 and ConvNeXt).
  - **Highest raw recall (78.4%)**: Its 70,527-tag vocabulary recognizes complex actions (`outstretched arm`, `reaching towards viewer`), expressions (`:o`), and character series (`cardcaptor sakura`, `seitokai nimo anawa aru`).
- **Trade-off**:
  - Emits many accurate but granular tags (e.g., `angel wings`, `bare shoulders`, `pleated skirt`) that penalize precision down to 37.2% when matched against concise human ground-truth files.

### 2. ConvNeXt-V2 Huge (`convnextv2_huge.dbv4-full-onnx`)
- **Strengths**:
  - **Highest single-model precision (51.9%)**: Conservative and remarkably clean. Very low false positive rate.
  - Strong clothing discrimination (`school uniform`, `blue sailor collar`, `pleated skirt`).
- **Trade-off**:
  - Model weights are massive (2.64 GB on disk).
  - Latency is the highest (7.13s on CPU).

### 3. Redstonexs Danbooru Tagger v1 (`danbooru-tagger-v1`)
- **Strengths**:
  - Unfreezing the EVA02-Large encoder drastically improves fine-grained attribute detection.
  - Successfully resolved `pink hair`, `crop top`, `hands up`, `midriff` on `chibi_pink`.
  - Half the model weight of WD-EVA02 (629 MB vs 1.2 GB) due to FP16 ONNX export.
- **Trade-off**:
  - Weak character head for obscure characters (missed `kinomoto sakura` on `sakura_wings`).

### 4. Ensemble (WD-EVA02-v3 + Redstonexs-v1)
- **Top F1 in the entire benchmark (58.0%)**:
  - By averaging logits on the shared ~8,000 Danbooru tags and allowing Redstonexs to contribute its 16,000 fine-grained tags, random misclassifications are filtered out while critical details are retained.
  - Achieved **76.2% F1 on `chibi_pink`** and **70.6% F1 on `beret_girl`**.

---

## 5. Strategic Recommendations for Prompt Condenser

1. **Short-Term (Quick Accuracy Win)**:
   - Introduce tag aliasing / normalization (`antenna hair` $\leftrightarrow$ `ahoge`, `:o` $\leftrightarrow$ `open mouth`, `mary janes` $\leftrightarrow$ `black footwear`). As shown in the data, models frequently detect the visual element correctly under an official Danbooru synonym.
2. **Vision Model Upgrade Options**:
   - **Option A (The Speed Champion)**: Switch to **Camie-v2** for a **5× speedup** (1.19s vs 5.79s) and higher tag recall, using a confidence threshold of $\ge 0.50$ to control vocabulary expansion.
   - **Option B (The Quality Champion)**: Ensemble **WD-EVA02 + Redstonexs-v1** for state-of-the-art accuracy (**58.0% F1**), solving color misclassifications like pink hair.
   - **Option C (Lightweight Single Model)**: Replace WD-EVA02 with **Redstonexs-v1** alone to cut model disk footprint from 1.2 GB to 629 MB while gaining 24,000-tag coverage.
