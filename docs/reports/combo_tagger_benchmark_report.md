# Combination Tagger Benchmark Report

**Date**: 2026-11-10  
**Dataset**: 10 non-AI images in `tests/eval_local/` (including 3 focus weak images)  
**Metric**: Exact `score_image_prompts.py` logic (excluding quality/solo tags)

## Ensemble Scoreboard

| Tagger / Combination | Best Thr | All 10 F1 | All Recall | Precision | Weak Images F1 | Weak Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **WD-EVA02-v3 (Solo)** | 0.30 | **57.2%** | 76.6% | 46.8% | **57.6%** | 65.5% |
| **Redstonexs-v1 (Solo)** | 0.55 | **54.1%** | 68.5% | 46.0% | **49.7%** | 50.7% |
| **Camie-v2 (Solo)** | 0.55 | **52.7%** | 71.4% | 42.3% | **49.6%** | 60.6% |
| **ConvNeXtV2-Huge (Solo)** | 0.35 | **57.6%** | 66.2% | 51.9% | **54.3%** | 55.8% |
| **Combo: WD + Redstone** | 0.50 | **58.0%** | 71.8% | 49.8% | **55.5%** | 57.7% |
| **Combo: Redstone + Camie** | 0.55 | **55.4%** | 72.3% | 45.8% | **52.2%** | 56.6% |
| **Combo: ConvNeXt + Redstone** | 0.40 | **57.7%** | 75.6% | 47.5% | **53.2%** | 58.8% |
| **Combo: WD + Camie** | 0.50 | **55.1%** | 69.4% | 46.9% | **53.5%** | 55.8% |
| **Tri-Combo: WD + RS + Camie** | 0.50 | **56.6%** | 74.2% | 46.5% | **53.7%** | 59.0% |

## Detailed Per-Image Breakdown for Top 3 Combos

### Combo: WD + Redstone (Optimal Threshold: 0.50)

| Image | Recall | Precision | F1 | Hit / Key |
|---|:---:|:---:|:---:|:---:|
| `beret_girl.jpg` | 90.0% | 58.1% | **70.6%** | 18/20 |
| `bikini_chart.jpg` | 61.5% | 53.3% | **57.1%** | 8/13 |
| `chibi_pink.jpg` | 77.4% | 75.0% | **76.2%** | 24/31 |
| `dark_armor.jpg` | 51.6% | 51.6% | **51.6%** | 16/31 |
| `fox_vtuber.png` | 64.3% | 41.9% | **50.7%** | 18/28 |
| `gym_girl.jpg` | 88.0% | 53.7% | **66.7%** | 22/25 |
| `office_two_girls.jpg` | 76.3% | 44.6% | **56.3%** | 29/38 |
| `rouge_bat.jpg` | 79.2% | 48.7% | **60.3%** | 19/24 |
| `sakura_wings.jpg` | 44.0% | 34.4% | **38.6%** | 11/25 |
| `schoolgirl_two_panels.jpg` | 85.4% | 36.8% | **51.5%** | 35/41 |

### Tri-Combo: WD + RS + Camie (Optimal Threshold: 0.50)

| Image | Recall | Precision | F1 | Hit / Key |
|---|:---:|:---:|:---:|:---:|
| `beret_girl.jpg` | 90.0% | 56.2% | **69.2%** | 18/20 |
| `bikini_chart.jpg` | 76.9% | 43.5% | **55.6%** | 10/13 |
| `chibi_pink.jpg` | 77.4% | 64.9% | **70.6%** | 24/31 |
| `dark_armor.jpg` | 51.6% | 47.1% | **49.2%** | 16/31 |
| `fox_vtuber.png` | 71.4% | 42.6% | **53.3%** | 20/28 |
| `gym_girl.jpg` | 88.0% | 50.0% | **63.8%** | 22/25 |
| `office_two_girls.jpg` | 73.7% | 41.8% | **53.3%** | 28/38 |
| `rouge_bat.jpg` | 79.2% | 45.2% | **57.6%** | 19/24 |
| `sakura_wings.jpg` | 48.0% | 36.4% | **41.4%** | 12/25 |
| `schoolgirl_two_panels.jpg` | 85.4% | 37.6% | **52.2%** | 35/41 |

### Combo: ConvNeXt + Redstone (Optimal Threshold: 0.40)

| Image | Recall | Precision | F1 | Hit / Key |
|---|:---:|:---:|:---:|:---:|
| `beret_girl.jpg` | 90.0% | 51.4% | **65.5%** | 18/20 |
| `bikini_chart.jpg` | 84.6% | 52.4% | **64.7%** | 11/13 |
| `chibi_pink.jpg` | 77.4% | 68.6% | **72.7%** | 24/31 |
| `dark_armor.jpg` | 54.8% | 47.2% | **50.7%** | 17/31 |
| `fox_vtuber.png` | 75.0% | 43.8% | **55.3%** | 21/28 |
| `gym_girl.jpg` | 92.0% | 50.0% | **64.8%** | 23/25 |
| `office_two_girls.jpg` | 81.6% | 47.7% | **60.2%** | 31/38 |
| `rouge_bat.jpg` | 70.8% | 47.2% | **56.7%** | 17/24 |
| `sakura_wings.jpg` | 44.0% | 30.6% | **36.1%** | 11/25 |
| `schoolgirl_two_panels.jpg` | 85.4% | 36.1% | **50.7%** | 35/41 |
