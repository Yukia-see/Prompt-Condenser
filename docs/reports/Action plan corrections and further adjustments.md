# Action plan: corrections and further adjustments

Nov 8, 2026 · @Yuka

## Corrections to my last message

Two statements in my review of the Phase 0 report were stated with more certainty than I had.

- **The temperatures were not verified.** The 0.05 and 0.10 figures come from the methodology document, and the Phase 0 report does not give the values the code uses. Ask Antigravity for the actual numbers before the baseline, because they decide whether the module's 0.1 and 0.35 defaults are wrong.
- **The Phase 5 baseline may be fine.** I said the current detector "is not YOLOv8s". As far as I know, the `_s` models in that detector family are YOLOv8-small variants trained on anime. The naming may be correct, so confirm it. My other point stands: compare the resulting crops, not detector accuracy, because DWPose gives keypoints and the detector gives boxes.

## Further adjustments to the plan

Five changes, in the order they matter.

1. **Keep one implementation of the policy.** The report says the app has two routes: the dual-pass pipeline and the multi-character tab (`detect_n` plus `twoimg`). If the new policy lives in the module but the old `merge_detail_crop` and `merge_quadrant_scene_crop` stay in app code, the leak survives on one path. Add to the Phase 2 done-when: no code path accepts crop tags without going through the policy, and the old merge functions either call it or are deleted.
2. **Add a resolution arm to the ablation.** As far as I know, Qwen2.5-VL reads images at dynamic resolution, so crops may help it less than they help WD14, which resizes everything to a fixed size. Add "full frame at higher resolution" next to the crop arms in Phase 1. Crops may turn out not to be worth the extra VLM calls.
3. **Tune on one half, confirm on the other.** Thresholds like 0.55 will be tuned on the evaluation set, and with 30 to 50 images they can overfit. Hold out a third of the images, and report per-image differences against the baseline, not only the averages.
4. **Rename the Moondream names.** The module's closure and constants (`moondream`, `MOONDREAM_TEMPERATURE`) now describe Qwen. Rename them to `vlm` in the Python and Rust code, and regenerate the parity cases, in one small separate commit so it does not mix with the Phase 2 changes.
5. **Bring the plan doc up to date.** It still asks the Phase 0 questions, lists the decisions as open, and carries the old Phase 5 wording. Update it with the Phase 0 answers, the settled decisions and these corrections.
