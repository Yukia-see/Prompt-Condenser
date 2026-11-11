# Comprehensive UI & Feature Specification: Vision Cleanup, Modular Body Proportions, & Phase 6

**Date**: 2026-11-11  
**From**: Gemini & User  
**To**: Claude Code (`claude/image-scan-fixes`)  
**Target Files**:
- [`ui/app.slint`](../../ui/app.slint) (Vision Toolbar & Body Selectors)
- [`src/profiles.rs`](../../src/profiles.rs) (Bust & Frame Proportion Presets)
- [`src/analyzer.rs`](../../src/analyzer.rs) (Modular Proportion Replacement & Recommendations)
- [`src/main.rs`](../../src/main.rs) (UI Callbacks & Bindings)
- [`src/prompt_cli.rs`](../../src/prompt_cli.rs) (`--view` & `--vrp` Additions)

---

## 1. Executive Summary

This report establishes the complete architectural plan approved by the user. It combines:
1. **Modular Body Proportions (Bust & Frame Split)**: Replacing the single "Body:" dropdown with independent "Bust:" and "Frame:" controls to natively support **Shortstack**, **Pear Shape**, and **Stacked Pear** builds with full customization.
2. **Vision Scanner UI Clean-Up**: Refactoring the 15-button toolbar in `ui/app.slint` into **3 clean visual zones** and a **collapsible Settings drawer**.
3. **ComfyUI Custom Node Delivery Status**: Already built and verified live in `tools/comfyui/` (<0.02s latency).
4. **Phase 6 Roadmap Alignment**: GPU Tagging 6B dropped; 6A measured before coding; AI-path corner-crop benchmark greenlit; 6C regional cross-negatives approved.

---

## 2. Feature Specification A: Modular Body Proportions (Bust & Frame Split)

### 2.1 The Problem
The current single `PROPORTIONS` list forces an artificial choice between upper body and lower body. There is no way to combine **Gigantic Breasts** with **Shortstack** or **Pear-shaped** hips, omitting major anime archetypes.

### 2.2 The Solution: Independent "Bust:" and "Frame:" Dropdowns
In `src/profiles.rs`, replace `PROPORTIONS` with two orthogonal tables:

#### 1. `BUST_TIERS`:
```rust
pub const BUST_TIERS: &[(&str, &str)] = &[
    ("None (keep prompt's bust)", ""),
    ("Flat chest (slender, aa-cup)", "flat chest, aa cup, small underbust, slender torso"),
    ("Small breasts (perky, a/b-cup)", "small breasts, a cup, perky breasts"),
    ("Medium breasts (full, c/d-cup)", "medium breasts, c cup, proportionate body"),
    ("Large breasts (curvy, e/f-cup)", "large breasts, e cup, deep cleavage, voluptuous"),
    ("Huge breasts (heavy, g/h-cup)", "huge breasts, g cup, heavy breasts, deep cleavage"),
    ("Gigantic breasts (hyper, massive)", "gigantic breasts, massive breasts, deep cleavage, voluptuous"),
];
```

#### 2. `FRAME_TIERS`:
```rust
pub const FRAME_TIERS: &[(&str, &str)] = &[
    ("None (keep prompt's frame)", ""),
    ("Proportionate (standard)", "proportionate body"),
    ("Petite (short stature)", "petite, short stature, slender"),
    ("Shortstack (petite, curvy, thick thighs)", "shortstack, short stature, petite, wide hips, thick thighs, narrow waist, curvy"),
    ("Thick thighs & wide hips", "thick thighs, wide hips, plump thighs, curvy"),
    ("Pear-shaped (narrow waist, broad hips, bottom-heavy)", "pear shaped, bottom-heavy, wide hips, broad hips, thick thighs, narrow waist, plump thighs, curvy"),
    ("Hourglass (narrow waist, curvy hips & bust)", "hourglass figure, narrow waist, wide hips, thick thighs, curvy, voluptuous"),
    ("Muscular & athletic (toned, abs)", "muscular female, abs, toned, athletic, defined muscles"),
];
```

### 2.3 Backend Engine Integration (`src/analyzer.rs`)
1. In `ScanConfig`: Replace `proportion: usize` with `bust: usize` and `frame: usize`.
2. In `analyze_prompt_inner`:
   - Replace conflicting bust tags when `config.bust > 0`.
   - Replace conflicting frame tags when `config.frame > 0`.
   - Because `BUST_TIERS` and `FRAME_TIERS` operate on separate tag sets, selecting **Gigantic Breasts** + **Shortstack** cleanly merges both:
     `shortstack, short stature, petite, gigantic breasts, massive breasts, deep cleavage, wide hips, thick thighs, narrow waist, voluptuous, curvy`.
3. Auto-Negatives & Clamping:
   - Preserves Pony V6 clamp (`1.15`) to prevent model burning.
   - Injects proper negatives (`flat hips, thigh gap, thin legs, tall` for Shortstack/Pear).

### 2.4 Slint UI (`ui/app.slint`)
In the toolbar row (around line 1595):
Replace:
```slint
SectionLabel { text: "Body:"; vertical-alignment: center; }
ComboBox {
    horizontal-stretch: 1;
    model: root.proportion_labels;
    current-index <=> root.proportion_index;
    selected => { root.analyze_requested(); }
}
```
With:
```slint
SectionLabel { text: "Bust:"; vertical-alignment: center; }
ComboBox {
    horizontal-stretch: 1;
    model: root.bust_labels;
    current-index <=> root.bust_index;
    selected => { root.analyze_requested(); }
}

SectionLabel { text: "Frame:"; vertical-alignment: center; }
ComboBox {
    horizontal-stretch: 1;
    model: root.frame_labels;
    current-index <=> root.frame_index;
    selected => { root.analyze_requested(); }
}
```

---

## 3. Feature Specification B: Vision Scanner UI Clean-Up

### 3.1 The Problem
Lines ~1704–2050 in `ui/app.slint` pack 15+ widgets into a single 34px horizontal row, creating a chaotic layout with multiple competing action buttons (`Read`, `Auto Scene`, `Redo Vision`, `Transfer`).

### 3.2 The 3-Zone Architecture

```text
Main Bar:
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [🖼 C1] [🌄 Ref] [Status Text...] │ [⚡ Tagger | 🧠 AI] │ [⚡ Action Button] [⚙ Settings ▾] [Scene ▾] │
│          ZONE 1: IMAGES           │   ZONE 2: ENGINE   │        ZONE 3: PRIMARY ACTION & DRAWERS  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

When [⚙ Settings ▾] is toggled (root.image_advanced):
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚙ Vision Settings:                                                                              │
│ Sensitivity (Threshold): [────●─────] 0.35                                                      │
│ Filters: [✓ Identity Only]  [✓ Scene Body Shape]  [✓ 2nd Pass Check]  [↻ Flush Vision Cache]    │
│ Advanced: [✓ Dual-Pass Vision]  [Mode Dropdown ▾]  [✓ LLM Synthesis]                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Zone 1: Unified Image Tray (Left Side)
* **Scene Reference Slot**: Thumbnail with hover `✕` (`remove_scene_ref_requested()`), or `+ Scene` button (`load_scene_ref_requested()`).
* **Character Image / Split Image Slot**:
  - 1 Character tab: `+ Image` button (`load_image_requested()`) / thumbnail with hover `✕` (`remove_image_requested()`).
  - 2–4 Characters tab: `🖼 Split Image…` button / thumbnail with hover `✕`, plus compact `[+ C1]`, `[+ C2]` pills.
* **Status Text**: Shows `root.scene_ref_status` or `root.image_status`.

#### Zone 2: Scan Engine & Presets (Center)
* **Segmented Mode Switch**:
  - `[ ⚡ Fast Tagger ]` (`image_use_ai = false; image_mode_changed();`).
  - `[ 🧠 Combined AI ]` (`image_use_ai = true; image_mode_changed();`).
* **Transfer Preset Selector** (Visible on 1 Character tab when `scene_ref_loaded`):
  - ComboBox: `["Scene & pose", "Identity transfer", "Outfit transfer", "Atmosphere"]` bound to `root.reference_use_index`.

#### Zone 3: Primary Action & Drawers (Right Side)
* **Single Dynamic Action Button**:
  - If `root.tagger_busy`: `"Scanning…"` (primary, disabled).
  - Else if `root.scene_ref_loaded && root.app_tab == 0 && root.reference_use_index > 0`:
    `"⇄ Transfer"`, primary, clicked => `root.transfer_requested()`.
  - Else if `root.scene_ref_loaded && !root.image_loaded && !root.char1_image_loaded`:
    `"⚡ Auto Scene"`, primary, clicked => `root.auto_scene_requested()`.
  - Else:
    `"⚡ Scan Image"`, primary: `root.image_loaded || root.images_pending || root.char1_image_loaded`, clicked => `root.read_image_requested()`.
* **`[ ⚙ Settings ▾ ]` Drawer Toggle**: Toggles `root.image_advanced`.
* **`[ Scene Panel ▾ ]` Drawer Toggle**: Toggles `root.scene_panel_open`.

### 3.3 The `[ ⚙ Settings ▾ ]` Drawer
Rendered when `root.image_advanced` is true:
1. **Sensitivity Slider**:
   - `SectionLabel { text: "SENSITIVITY (THRESHOLD):"; }`
   - `Slider`: `minimum: 0.15; maximum: 0.60; value <=> root.tag_threshold; changed => { root.tag_threshold_changed(); }`
2. **Filters & Quick Actions**:
   - FilterChip: `"Identity Only"` (`image_identity_only`, `identity_changed()`).
   - FilterChip: `"Scene Body Shape"` (`scene_body_shape`, `scene_body_shape_changed()`).
   - FilterChip: `"2nd Pass Visual Check"` (`image_second_pass`, `second_pass_changed()`).
   - PillButton: `"↻ Flush Vision Cache"` (`redo_image_requested()`).
3. **Advanced / Experimental**:
   - FilterChip: `"Dual-Pass Vision"` (`enable_dual_pass`, `dual_pass_changed()`).
   - ComboBox: `root.dual_pass_mode_labels` (`dual_pass_mode_index`).
   - FilterChip: `"LLM Synthesis"` (`use_llm_synthesis`, `llm_synthesis_changed()`).

---

## 4. ComfyUI Node Delivery & CLI Flags

* **Node Status**: Created in `tools/comfyui/` and symlinked to `/home/yukia/ComfyUI/custom_nodes/ComfyUI-Prompt-Condenser/`.
* **Verified Performance**: Tested in ComfyUI venv; execution time is **<0.02s**.
* **Pending CLI additions in `src/prompt_cli.rs`**:
  - Add `--view <view_name>` (mapping to `TargetView`).
  - Add `--vrp` (outputting JSON matching the Region Editor).

---

## 5. Phase 6 Decisions Summary

1. **6B (GPU Tagging)**: Dropped. ONNX Runtime lacks Linux Vulkan support; CPU multi-threading (4.6s) is lightweight and leaves all 16GB APU unified memory for Ollama and `sd-cli`.
2. **6A (Camie-v2)**: Measure Camie-v2 on the 9 test images specifically targeting low-scoring tags before writing any code.
3. **Corner Crops**: Benchmark AI path with vs. without corner crops (11 images, 2 rounds). Drop them if F1 holds (~11s savings).
4. **6C (Regional Cross-Negatives)**: Implement selective cross-negatives in `src/two.rs` for multi-character VRP.

---

Claude, please review this specification and proceed with implementing the UI clean-up and Bust/Frame split!
