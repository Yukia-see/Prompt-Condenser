# Prompt Condenser: Phase 1 Baseline & Multi-Arm Ablation Report

**Date:** Nov 8, 2026
**Images Evaluated:** 3
**Dataset:** `tests/eval_dataset`
**Split:** 2 Tuning / 1 Holdout Confirmation

## 1. Executive Summary & Aggregate Averages

| Arm | Dataset Split | Micro Prec | Micro Rec | Micro F1 | Global Prec | Global Rec | Crop Artifacts | Avg Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **arm1** | Tuning (2/3) | 38.6% | 73.9% | 0.51 | 52.3% | 100.0% | 0 | 8.6s |
| **arm1** | Holdout (1/3) | 17.2% | 27.0% | 0.21 | 72.7% | 80.0% | 0 | 4.7s |
| **arm2** | Tuning (2/3) | 38.4% | 72.4% | 0.50 | 52.3% | 100.0% | 0 | 6.8s |
| **arm2** | Holdout (1/3) | 16.1% | 24.3% | 0.19 | 80.0% | 80.0% | 0 | 7.5s |
| **arm3** | Tuning (2/3) | 33.3% | 73.9% | 0.46 | 47.2% | 100.0% | 6 | 39.8s |
| **arm3** | Holdout (1/3) | 27.3% | 56.8% | 0.37 | 66.7% | 80.0% | 4 | 38.2s |
| **arm4** | Tuning (2/3) | 27.0% | 70.8% | 0.39 | 47.2% | 100.0% | 6 | 88.0s |
| **arm4** | Holdout (1/3) | 29.8% | 67.6% | 0.41 | 76.9% | 100.0% | 4 | 90.0s |

## 2. Per-Image Delta Tables (vs. Arm 1 Baseline)

### Arm 2: Full Frame High-Res (2048px)

| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\Delta$ Micro P | $\Delta$ Micro R | Artifacts (Base $\rightarrow$ Arm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HQwyhBIbAAAguCD.jpg` | Tuning | 46.2% / 81.1% | 46.9% / 81.1% | +0.7% | +0.0% | 0 $\rightarrow$ 0 |
| `HWL8f8wbYAAMJ-a.jpg` | Tuning | 31.0% / 66.7% | 30.0% / 63.6% | -1.0% | -3.0% | 0 $\rightarrow$ 0 |
| `HWP1dffawAA5Y7K.jpg` | Holdout | 17.2% / 27.0% | 16.1% / 24.3% | -1.2% | -2.7% | 0 $\rightarrow$ 0 |

### Arm 3: WD14 Crops Only

| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\Delta$ Micro P | $\Delta$ Micro R | Artifacts (Base $\rightarrow$ Arm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HQwyhBIbAAAguCD.jpg` | Tuning | 46.2% / 81.1% | 42.3% / 81.1% | -3.9% | +0.0% | 0 $\rightarrow$ 4 |
| `HWL8f8wbYAAMJ-a.jpg` | Tuning | 31.0% / 66.7% | 24.4% / 66.7% | -6.5% | +0.0% | 0 $\rightarrow$ 2 |
| `HWP1dffawAA5Y7K.jpg` | Holdout | 17.2% / 27.0% | 27.3% / 56.8% | +10.0% | +29.7% | 0 $\rightarrow$ 4 |

### Arm 4: Full Decomposition (WD14+VLM Crops)

| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\Delta$ Micro P | $\Delta$ Micro R | Artifacts (Base $\rightarrow$ Arm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HQwyhBIbAAAguCD.jpg` | Tuning | 46.2% / 81.1% | 33.3% / 81.1% | -12.8% | +0.0% | 0 $\rightarrow$ 4 |
| `HWL8f8wbYAAMJ-a.jpg` | Tuning | 31.0% / 66.7% | 20.6% / 60.6% | -10.4% | -6.1% | 0 $\rightarrow$ 2 |
| `HWP1dffawAA5Y7K.jpg` | Holdout | 17.2% / 27.0% | 29.8% / 67.6% | +12.5% | +40.5% | 0 $\rightarrow$ 4 |

