# Implementation Plan: Hardening Tagger-Writer Reconciliation

## Goal Description
This plan addresses the issues where the Prompt Condenser program logic aggressively strips color tags from colorful images (like `dark_armor.jpg`) due to overly sensitive monochrome detection, as well as refining tag recovery to improve overall accuracy and consistency.

## User Review Required
- **IMPORTANT**: The changes to [`polish.rs`](file:///home/yukia/Documents/Prompt%20Condensor/src/ai_engine/polish.rs) are designed to be general. While they should improve accuracy for the 3 weak images, they might slightly change tag outputs for others. I will verify with the existing test set to ensure no regressions.
- **CAUTION**: I am **NOT** building the Arch package (per previous user constraints).

## Open Questions
- Do you have any specific concerns about the order of reconciliation fixes (e.g., should Fix 1 for color detection take absolute priority)?

## Proposed Changes

### [Rust Component: `src/ai_engine/vision.rs`]
- **[MODIFY]** Implement pixel saturation/chromaticity analysis to compute a `mean_chromatic_delta`. This will be used to gate the `is_monochrome` logic.

### [Rust Component: `src/ai_engine/polish.rs`]
- **[MODIFY]**
    - **Fix 1**: Update `is_monochrome` to use `mean_chromatic_delta`. It will only return true if the image is truly low-saturation AND the tagger predicts monochrome.
    - **Fix 2**: Update `restore_tagger_tags` to enforce a $\ge 0.50$ score threshold for recovery and add style tags (`parody`, etc.) to `RECOVERY_SKIP`.
    - **Fix 3**: Add single hair color enforcement in `one_per_part`.
    - **Fix 4**: Add a contradiction guard in `one_per_part` (e.g., dropping `barefoot` if footwear is present).

### [Test Set Updates]
- **[MODIFY]** `tests/eval_local/chibi_pink.txt`: Update the ground truth answer file to the canonical values documented in [`danbooru_prompts_non_ai_images.md`](file:///home/yukia/Documents/Prompt%20Condensor/docs/reports/danbooru_prompts_non_ai_images.md).

## Verification Plan

### Automated Tests
1. `cargo test` (Verify all 205+ unit tests pass).
2. Run `tools/eval_images.sh` on the non-AI image set (the 10 eval_local images + 3 root images + fresh).
3. Compare the output against previous run outputs (`run4b_reconcile_18982e5.txt`).

### Manual Verification
1. Inspect the generated prompt for `dark_armor.jpg` and `chibi_pink.jpg` to ensure colors are no longer stripped and the `black pants` error is resolved.
