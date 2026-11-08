# Antigravity action plan: crop stage and tag policy

Nov 8, 2026 · @Yuka & @Antigravity

## Goal and ground rules

Add a crop stage with a per-crop tag policy to the dual-pass pipeline, in both the Python reference and the Rust module, so crops add correct micro-details without adding crop artifacts. The plan follows the review of the multi-zone methodology document and incorporates the 5 adjustments from the review corrections. Each phase is measured against an evaluation set before the next one starts.

Rules for every phase:

- **Python reference first, then Rust.** Run `./tools/build.sh` to regenerate the Rust data tables and `tests/parity.json`. A phase is done only when `cargo test` and `python3 reference/test_prompt_condenser.py` both pass.
- **Pure logic, injected I/O.** Box math, tag policy and score merging are pure functions with tests. Models are called through closures, as the existing module does.
- **Single implementation of tag policy.** All app routes (`dual_pass`, `detect_n`, `twoimg`, `tagger`) route through the unified pure policy. Legacy merge functions either call it or are deleted. No code path accepts crop tags directly without policy filtering.
- **No new crates in `prompt_condenser.rs`.** Any model-loading dependency (detector, pose) lives in the app.
- **No silent fallbacks.** Every fallback adds a message to `warnings`.
- **Do not log or store images.**
- **Stop and ask on a blocker** instead of guessing.

---

## Phase 0: Repository audit & settled decisions (COMPLETED)

### Answers to Phase 0 Questions

1. **Current Pipeline:**
   - **App runtime (`src/ai_engine.rs`, `src/twoimg.rs`, `src/tagger.rs`):** Runs YOLOv8s anime detection $\rightarrow$ WD14 tagger on full frame + 4 scene quadrants + 3 character zones $\rightarrow$ Qwen 2.5-VL 3B visual inspection $\rightarrow$ Gemma-4 structured prompt synthesis.
   - **Pure module (`reference/prompt_condenser_dual_pass.py`, `src/prompt_condenser.rs`):** Implements dual-pass tag reconciliation (prose-to-tags longest match, conflict resolution, hedging/negation rules).
   - **Convergence goal:** Integrate the multi-zone decomposition and strict tag policy directly into the pure module, and have the app layer delegate its crop merging exclusively to this pure module.
2. **Character Detector:**
   - Model is `person_detect_v1.3_s.onnx` from `deepghs/anime_person_detection`, confirmed as a **YOLOv8-small** anime-specialized detector.
   - Overlapping figures produce overlapping bounding boxes without segmentation.
   - Partial figures (cut off at image boundary) produce truncated boxes.
   - Non-human/empty scenes yield no detections, triggering full-frame fallback.
3. **WD14 Tag Format:**
   - `src/tagger.rs` returns `Vec<(String, f32)>` (tag string with confidence score).
   - The pure module interface is being extended from plain strings to score-aware pairs `(tag, score)`.
4. **Tag Categories from `selected_tags.csv`:**
   - Category 0: General (micro-attributes, clothing, pose, objects).
   - Category 4: Character (specific anime/game character names).
   - Category 9: Rating (`general`, `sensitive`, `questionable`, `explicit`).
5. **Existing Crops in Code:**
   - Scene Quadrants: 4 quadrants with 10% center overlap ($[0, 0.55] \times [0, 0.55]$, etc.).
   - Character Zones: 3 vertical slices of the detected person box: Upper (0–55%), Middle (30–70%), Lower (55–100%).

### Settled User Decisions

- **Pipeline identity:** Dual-pass pipeline augmented with multi-zone WD14 crops and VLM inspection.
- **Framing tags (`full body`, `cowboy shot`, `close-up`):** Routed to `scene_prompt` only, never accepted from crops.
- **Rating tags (`general`, `sensitive`, `questionable`, `explicit`):** Stripped by default.
- **Runtime temperatures:** Calibrated to `0.05` for VLM vision and `0.10` for Gemma synthesis.

---

## Phases 1 to 5

Do the phases in order. Phase 4 does not depend on 2 or 3 and can run in parallel with them.

### Phase 1: Evaluation harness & baseline

Measure before changing anything, so "better" has a quantifiable number.

- [ ] Write one command that runs the pipeline over a folder of images and compares the output tags with a ground-truth tag file per image (30 to 50 verified images).
- [ ] **Data split:** Hold out 1/3 of the dataset as a Confirmation Set (~12–16 images); use 2/3 as a Tuning Set (~24–34 images) to prevent overfitting the 0.55 threshold.
- [ ] Report precision and recall separately for micro-attribute tags and for global tags (framing, camera, lighting, background).
- [ ] Count crop-artifact tags in the output (`head out of frame`, `cropped legs`, `portrait`, `close-up`, `out of frame`, `solo focus`, `breast focus`), and count rating and character-name tags that came from crops.
- [ ] **Ablation arms (including Resolution Arm):**
  - Arm 1: Full frame only (1024px baseline).
  - Arm 2: Full frame at High/Native Resolution (tests native Qwen 2.5-VL dynamic resolution acuity without crops).
  - Arm 3: Full frame + WD14 crops only (evaluates WD14 fixed 448x448 crop utility).
  - Arm 4: Full frame + WD14 crops + VLM crops (full decomposition).
- [ ] Output per-image delta tables against the baseline in addition to summary averages.
- [ ] Save and commit the baseline report from current code.

Done when: one command prints the report with per-image deltas and the baseline is committed.

### Phase 2: Per-crop tag policy and crop stage

Eliminate crop-boundary artifacts and unify tag merging across the application.

- [ ] **Single Policy Implementation:** Pure tag policy implemented once in `reference/prompt_condenser_dual_pass.py` and `src/prompt_condenser.rs`. All app routes (`merge_detail_crop`, `merge_quadrant_scene_crop`, `twoimg.rs`) call this pure function or are deleted.
- [ ] Classify every tag as micro-attribute, global-only, crop-artifact, or rating/character using `selected_tags.csv` categories and explicit lists.
- [ ] **Tag Policy Rules:**
  - Global tags (framing, body size, camera, lighting) come from the full frame only.
  - Crop-artifact tags (`head out of frame`, `cropped legs`, `out of frame`, `portrait`, `close-up`, `solo focus`, `breast focus`) are strictly blacklisted from all crops.
  - Rating and character-name tags are never accepted from crops.
  - Micro-attribute tags from crops require a confidence $\ge 0.55$ (tuned on the 2/3 tuning set).
  - Overlapping zones deduplicate tags (do not add confidences).
- [ ] Pure zone box function: figure box, image size and edge contact in $\rightarrow$ valid zones out. Skip lower zones for half-body or edge-clipped figures.
- [ ] Regenerate Rust tables with `./tools/build.sh` and add parity cases to `tools/gen_parity.py`.

Done when: unit tests prove a crop-artifact tag never reaches the output across any route, parity passes, and evaluation shows zero crop-artifact tags with no precision drop on holdout images.

### Phase 3: Two-source rule and edge cases

- [ ] A tag reported solely by the VLM is retained only if WD14 confirms it or a second crop agrees; unconfirmed VLM tags are logged to `dropped_unconfirmed`.
- [ ] For overlapping figures, blank out or mask the neighboring bounding box before cropping. If impossible, fallback to full-frame for that region with a warning.
- [ ] If no figure is detected, process full frame only and add a warning.

Done when: tests cover each case, parity passes, and evaluation shows fewer unconfirmed tags in output without recall drop on micro-attributes.

### Phase 4: Layer-based conflict rules

- [ ] Replace hardcoded conflict pairs with an external rules file in `reference/`, compiled to Rust via `tools/gen_data.py`.
- [ ] Assign clothing tags a body region and layer (outer, inner, footwear, legwear). An outer layer hides an inner layer only when the inner layer is occluded, rather than blindly deleting it.
- [ ] Add explicit `coexists` list for valid simultaneous tags (e.g. `tube top` + `cleavage`, `boots` + `thighhighs`, `socks` + `shoes`).

Done when: tests verify `tube top` retains `cleavage`, `turtleneck` suppresses `cleavage`, `boots` retains `thighhighs`, and parity passes.

### Phase 5: Pose model and performance budget

- [ ] Compare DWPose with the anime YOLOv8s detector on 20 images from the evaluation set, evaluating resulting crop quality and micro-attribute yield rather than raw mAP.
- [ ] Use keypoints for crops only if they demonstrably beat fixed-ratio zones on held-out images.
- [ ] Hands get a bounding box centered on the wrist with a proportional margin.
- [ ] Optionally export OpenPose-compatible JSON for ControlNet workflows.
- [ ] Profile execution time and VRAM per stage: batch WD14 calls, set Ollama keep-alive, optimize context window for concurrent Qwen/Gemma residency.

Done when: report contains comparison table of crop yields between pose and detector, plus time/VRAM profiles per stage.

---

## Out of scope

- Two-image merge (character image + scene image) and wardrobe completeness rules (separate feature).
- Weighted-tag syntax like `(tag:1.1)` for Anima checkpoints.
- Active-query VLM boxes (deferred until Phase 3 validates two-source confirmation; requires absolute coordinate mapping for Qwen 2.5-VL).
- Aspect-ratio auto-negative prompt without pose gating.
- ComfyUI regional conditioning export (beyond optional OpenPose JSON).
- Unverified marketing claims ("zero hallucination", "300% increased risk").

---

## Report back

Send a short report at the end of each phase:
- Files changed.
- New constants and defaults.
- `cargo test` and Python test results.
- Parity test results (`cargo test prompt_condenser_parity`).
- Evaluation numbers alongside baseline on both Tuning and Holdout sets.
