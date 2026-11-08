# Response to Action Plan: Corrections and Further Adjustments

**Date:** Nov 8, 2026  
**From:** Antigravity  
**To:** @Yuka & Project Team  
**Subject:** Evaluation and acceptance of corrections and further adjustments

---

## 1. Executive Summary

We have reviewed `@Yuka`'s document [`Action plan corrections and further adjustments.md`](file:///home/yukia/Documents/Prompt%20Condensor/Action%20plan%20corrections%20and%20further%20adjustments.md). 

**We endorse and accept all 2 corrections and all 5 adjustments without reservation.**

Yuka's review is methodologically rigorous, spots the exact leak mechanism that allowed crop-boundary tags into our live test, provides crucial empirical clarifications on model architectures, and establishes safeguards against overfitting. Below are the verified technical facts and the concrete implementation plan for each adjustment.

---

## 2. Verification of the Two Corrections

### Correction 1: Actual Temperature Values in Code
Yuka correctly noted that the Phase 0 report and pure reference constants did not match the values cited in the methodology document. Here are the exact values verified directly in the codebase:

1. **In App Runtime Layer ([`src/ai_engine.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine.rs)):**
   - **VLM Vision Calls** (`extract_visual_description_from_path` calling Qwen 2.5-VL 3B / Moondream):
     ```rust
     "temperature": 0.05,
     "top_p": 0.85,
     "repeat_penalty": 1.15,
     "seed": 42
     ```
   - **Synthesis LLM Calls** (`generate_prompt_with_retries` calling Gemma-4 12B / 4B):
     ```rust
     "temperature": 0.10,
     "top_p": 0.90,
     "repeat_penalty": 1.18,
     "seed": 42
     ```
2. **In Pure Module ([`reference/prompt_condenser_dual_pass.py`](file:///home/yukia/Documents/Prompt%20Condensor/reference/prompt_condenser_dual_pass.py) & [`src/prompt_condenser.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/prompt_condenser.rs)):**
   - Currently: `MOONDREAM_TEMPERATURE = 0.1` and `GEMMA_TEMPERATURE = 0.35`.
   - **Decision:** As part of Adjustment 4, we will update the module default constants to align with the calibrated runtime values: `VLM_TEMPERATURE = 0.05` and `GEMMA_TEMPERATURE = 0.10`. Both will remain configurable via the `Config` struct.

### Correction 2: Character Detector Identity & Phase 5 Evaluation
- **Model Verification:** Verified in [`src/detector.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/detector.rs#L48-L52). The detector loads `person_detect_v1.3_s.onnx` from [`deepghs/anime_person_detection`](https://huggingface.co/deepghs/anime_person_detection).
- **Confirmation:** In that repository, the `_s` models are indeed **YOLOv8-small** architectures trained on anime illustrations. Yuka's correction is 100% accurate.
- **Evaluation Metric Alignment:** We agree with Yuka's methodology for Phase 5: rather than comparing detector mAP (which is an apples-to-oranges comparison since DWPose produces skeletal keypoints while YOLOv8 produces bounding boxes), we will evaluate the **resulting crop utility and micro-attribute yield**. Specifically, does an anatomically-anchored crop (e.g. wrist keypoint + hand margin) beat a geometric bounding-box third in WD14 tag discovery?

---

## 3. Implementation of the Five Adjustments

### Adjustment 1: Single Implementation of the Tag Policy (Critical)
- **The Issue Identified:** In the live benchmark of [`HWP1dffawAA5Y7K.jpg`](file:///home/yukia/Pictures/HWP1dffawAA5Y7K.jpg), the After output contained crop artifacts (`head out of frame`, `cropped legs`, `out of frame`, `portrait`, `close-up`, `solo focus`, `breast focus`). This happened because [`src/tagger.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/tagger.rs) (`merge_detail_crop` and `merge_quadrant_scene_crop`) used naive score merging (`max(base, crop)`) without filtering crop-boundary artifacts.
- **The Fix:**
  1. The Tag Policy will be implemented **once** as a pure function in [`reference/prompt_condenser_dual_pass.py`](file:///home/yukia/Documents/Prompt%20Condensor/reference/prompt_condenser_dual_pass.py) and generated into [`src/prompt_condenser.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/prompt_condenser.rs).
  2. The app's merge functions in [`src/tagger.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/tagger.rs) and [`src/twoimg.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/twoimg.rs) will delegate directly to this single pure function.
  3. No crop tags will ever enter the tag pool without passing through the policy whitelist/blacklist.
  4. **Phase 2 Done-When:** Unit tests and regression tests prove that crop-boundary artifacts can never reach the output prompt across any pipeline route, and old merge functions either call the pure policy or are deleted.

### Adjustment 2: Resolution Arm in Phase 1 Ablation
- **Rationale:** Qwen 2.5-VL operates with native dynamic resolution (NaViT patch encoding). It does not force square downsampling like WD14 ($448\times 448$). Therefore, Qwen 2.5-VL might detect hairpins and accessories on a native/high-resolution full frame without requiring crops.
- **The Arm Configuration:**
  - **Arm 1 (Baseline):** Full frame at 1024px, no crops (WD14 + VLM).
  - **Arm 2 (High-Res Full Frame):** Full frame at native/2048px, no crops (evaluating native VLM vision acuity).
  - **Arm 3 (WD14 Crops Only):** Full frame VLM + WD14 4-quadrant and 3-zone crops.
  - **Arm 4 (Full Decomposition):** WD14 crops + VLM crops (measuring whether VLM crops justify their compute cost).

### Adjustment 3: Train/Tune (2/3) vs Holdout (1/3) Split
- **Overfitting Prevention:** To prevent tuning the 0.55 confidence threshold to noise on a 30–50 image set, we divide the evaluation set into:
  - **Tuning Set (67%, ~24–34 images):** Used to calibrate crop score thresholds, zone overlaps, and conflict weights.
  - **Holdout Set (33%, ~12–16 images):** Strictly evaluated post-tuning without parameter adjustment.
- **Reporting Requirement:** In addition to aggregate macro/micro precision and recall, the harness will output **per-image delta tables**, exposing regressions on individual images immediately.

### Adjustment 4: Rename Moondream Symbols to `vlm`
- **Scope of Rename:**
  - `moondream_fn` $\rightarrow$ `vlm_fn`
  - `MOONDREAM_TEMPERATURE` $\rightarrow$ `VLM_TEMPERATURE` (default updated to `0.05`)
  - `GEMMA_TEMPERATURE` (default updated to `0.10`)
  - `MOONDREAM_RETRIES` $\rightarrow$ `VLM_RETRIES`
  - `MOONDREAM_RETRY_TEMP_BUMP` $\rightarrow$ `VLM_RETRY_TEMP_BUMP`
  - `dropped_moondream` $\rightarrow$ `dropped_vlm`
  - `negated_moondream` $\rightarrow$ `negated_vlm`
- **Execution:** Executed immediately as a clean, isolated preparatory step. Verified through `reference/test_prompt_condenser.py`, `tools/build.sh`, and `cargo test prompt_condenser_parity`.

### Adjustment 5: Updating the Master Action Plan Document
- [`Antigravity action plan crop stage and tag policy.md`](file:///home/yukia/Documents/Prompt%20Condensor/Antigravity%20action%20plan%20crop%20stage%20and%20tag%20policy.md) is updated immediately with:
  - Closed Phase 0 answers (repo pipeline, detector model, CSV categories, WD14 score handling).
  - Settle of the 3 user decisions.
  - Integration of the 5 adjustments into Phases 1, 2, and 5.

---

## 4. Verification Checkpoints

| Checkpoint | Target | Status |
| :--- | :--- | :--- |
| Python Unit Tests | 27/27 Pass | Verified |
| Rust Parity Check | `cargo test prompt_condenser_parity` Pass | Verified |
| App Compilation | `cargo check --lib` | Verified |
| Artifact Leak Defense | Zero crop-boundary tags permitted | Phase 2 Goal |

---
*Signed by Antigravity*
