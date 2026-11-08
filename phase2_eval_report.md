# Prompt Condenser: Phase 1 Baseline & Multi-Arm Ablation Report

**Date:** Nov 8, 2026
**Images Evaluated:** 3
**Dataset:** `tests/eval_dataset`
**Split:** 2 Tuning / 1 Holdout Confirmation

## 1. Executive Summary & Aggregate Averages

| Arm | Dataset Split | Micro Prec | Micro Rec | Micro F1 | Global Prec | Global Rec | Crop Artifacts | Avg Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **arm1** | Tuning (2/3) | 39.3% | 73.9% | 0.51 | 52.3% | 100.0% | 0 | 13.1s |
| **arm1** | Holdout (1/3) | 15.5% | 24.3% | 0.19 | 72.7% | 80.0% | 0 | 6.5s |
| **arm2** | Tuning (2/3) | 38.4% | 72.4% | 0.50 | 52.3% | 100.0% | 0 | 6.6s |
| **arm2** | Holdout (1/3) | 16.1% | 24.3% | 0.19 | 80.0% | 80.0% | 0 | 7.4s |
| **arm3** | Tuning (2/3) | 33.7% | 73.9% | 0.46 | 47.2% | 100.0% | 0 | 355.5s |
| **arm3** | Holdout (1/3) | 25.6% | 54.1% | 0.35 | 72.7% | 80.0% | 0 | 355.2s |
| **arm4** | Tuning (2/3) | 28.1% | 72.4% | 0.40 | 47.2% | 100.0% | 0 | 399.5s |
| **arm4** | Holdout (1/3) | 25.5% | 64.9% | 0.37 | 83.3% | 100.0% | 0 | 405.0s |

## 2. Per-Image Delta Tables (vs. Arm 1 Baseline)

### Arm 2: Full Frame High-Res (2048px)

| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\Delta$ Micro P | $\Delta$ Micro R | Artifacts (Base $\rightarrow$ Arm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HQwyhBIbAAAguCD.jpg` | Tuning | 47.6% / 81.1% | 46.9% / 81.1% | -0.7% | +0.0% | 0 $\rightarrow$ 0 |
| `HWL8f8wbYAAMJ-a.jpg` | Tuning | 31.0% / 66.7% | 30.0% / 63.6% | -1.0% | -3.0% | 0 $\rightarrow$ 0 |
| `HWP1dffawAA5Y7K.jpg` | Holdout | 15.5% / 24.3% | 16.1% / 24.3% | +0.6% | +0.0% | 0 $\rightarrow$ 0 |

### Arm 3: WD14 Crops Only

| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\Delta$ Micro P | $\Delta$ Micro R | Artifacts (Base $\rightarrow$ Arm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HQwyhBIbAAAguCD.jpg` | Tuning | 47.6% / 81.1% | 42.9% / 81.1% | -4.8% | +0.0% | 0 $\rightarrow$ 0 |
| `HWL8f8wbYAAMJ-a.jpg` | Tuning | 31.0% / 66.7% | 24.4% / 66.7% | -6.5% | +0.0% | 0 $\rightarrow$ 0 |
| `HWP1dffawAA5Y7K.jpg` | Holdout | 15.5% / 24.3% | 25.6% / 54.1% | +10.1% | +29.7% | 0 $\rightarrow$ 0 |

### Arm 4: Full Decomposition (WD14+VLM Crops)

| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\Delta$ Micro P | $\Delta$ Micro R | Artifacts (Base $\rightarrow$ Arm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HQwyhBIbAAAguCD.jpg` | Tuning | 47.6% / 81.1% | 34.5% / 81.1% | -13.1% | +0.0% | 0 $\rightarrow$ 0 |
| `HWL8f8wbYAAMJ-a.jpg` | Tuning | 31.0% / 66.7% | 21.6% / 63.6% | -9.3% | -3.0% | 0 $\rightarrow$ 0 |
| `HWP1dffawAA5Y7K.jpg` | Holdout | 15.5% / 24.3% | 25.5% / 64.9% | +10.0% | +40.5% | 0 $\rightarrow$ 0 |

