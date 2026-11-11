# Task for Claude: Clean Up the Vision Scanner UI in `ui/app.slint`

**Date**: 2026-11-11  
**From**: Gemini & User  
**To**: Claude Code (`claude/image-scan-fixes`)  
**Target File**: [`ui/app.slint`](../../ui/app.slint) (Lines ~1704–2050)  

---

## 1. Problem Statement

The current Vision Scanner toolbar (`SCENE & REFERENCE BAR + LOWERABLE DROPDOWN PANEL`, lines ~1704–2050 in `ui/app.slint`) has become a confusing "15-button traffic jam":
* Four competing action buttons sit side-by-side (`Read (Tagger)`, `Auto Scene (Fast)`, `Redo Vision`, `⇄ Transfer`).
* Engine selector buttons (`Tagger` vs `Combined AI`) are placed *after* the action buttons instead of driving them.
* Advanced toggles (`Dual-Pass Vision`, `LLM Synthesis`, `2nd Pass Visual Check`, `Scene Body Shape`, multiple ComboBoxes) spill out across the horizontal row, causing horizontal overflow or squeezing.
* Tag sensitivity threshold has no slider in the UI despite having the property `tag_threshold` and callback `tag_threshold_changed`.

---

## 2. Redesign Specification: 3 Clean Zones + Collapsible Settings Drawer

Please refactor lines ~1704–2050 in `ui/app.slint` into **3 clear visual zones** in the main bar, plus a **collapsible Settings drawer**:

```text
Main Bar (Single Sleek Container):
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

---

## 3. Detailed Component Breakdown

### Zone 1: Unified Image Tray (Left Side)
* **Scene Reference Slot**:
  - Thumbnail if loaded (with hover `✕` to `remove_scene_ref_requested()`), or `+ Scene` button calling `load_scene_ref_requested()`.
* **Character Image / Split Image Slot**:
  - 1 Character tab: `+ Image` button calling `load_image_requested()` / thumbnail with hover `✕` (`remove_image_requested()`).
  - 2–4 Characters tab: `🖼 Split Image…` button / thumbnail with hover `✕`, plus compact `[+ C1]`, `[+ C2]`, etc. pills.
* **Status Text**:
  - Displays `root.scene_ref_status` or `root.image_status`.

### Zone 2: Scan Engine & Mode (Center)
* **Segmented Pill Selector**:
  - `[ ⚡ Fast Tagger ]` (sets `root.image_use_ai = false; root.image_mode_changed();`, highlighted if `!root.image_use_ai`).
  - `[ 🧠 Combined AI ]` (sets `root.image_use_ai = true; root.image_mode_changed();`, highlighted if `root.image_use_ai`).
* **Transfer Preset Selector** (Only visible on 1 Character tab when `root.scene_ref_loaded`):
  - ComboBox: `["Scene & pose", "Identity transfer", "Outfit transfer", "Atmosphere"]` bound to `root.reference_use_index`.

### Zone 3: Primary Action & Drawers (Right Side)
* **Single Dynamic Action Button**:
  - If `root.tagger_busy`: `"Scanning…"` (primary, disabled).
  - Else if `root.scene_ref_loaded && root.app_tab == 0 && root.reference_use_index > 0`:
    `"⇄ Transfer"`, primary, clicked => `root.transfer_requested()`.
  - Else if `root.scene_ref_loaded && !root.image_loaded && !root.char1_image_loaded`:
    `"⚡ Auto Scene"`, primary, clicked => `root.auto_scene_requested()`.
  - Else:
    `"⚡ Scan Image"`, primary: `root.image_loaded || root.images_pending || root.char1_image_loaded`, clicked => `root.read_image_requested()`.
* **`[ ⚙ Settings ▾ ]` Toggle Button**:
  - PillButton toggling `root.image_advanced = !root.image_advanced`. Text: `root.image_advanced ? "⚙ Settings ▴" : "⚙ Settings ▾"`.
* **`[ Scene Panel ▾ ]` Toggle Button**:
  - Existing toggle for `root.scene_panel_open`.

---

## 4. The `[ ⚙ Settings ▾ ]` Drawer (When `root.image_advanced` is true)

Create a compact container directly beneath the main bar:
1. **Sensitivity Slider**:
   - `SectionLabel { text: "SENSITIVITY (THRESHOLD):"; }`
   - `Slider`: `minimum: 0.15; maximum: 0.60; value <=> root.tag_threshold; changed => { root.tag_threshold_changed(); }`
   - `Text`: e.g. displays threshold value formatted or numeric.
2. **Filters & Quick Actions**:
   - FilterChip: `"Identity Only"` (`checked <=> root.image_identity_only; toggled => { root.identity_changed(); }`).
   - FilterChip: `"Scene Body Shape"` (`checked <=> root.scene_body_shape; toggled => { root.scene_body_shape_changed(); }`).
   - FilterChip: `"2nd Pass Visual Check"` (`checked <=> root.image_second_pass; toggled => { root.second_pass_changed(); }`).
   - PillButton: `"↻ Flush Vision Cache"` (`clicked => { root.redo_image_requested(); }`).
3. **Advanced Experimental Controls**:
   - FilterChip: `"Dual-Pass Vision"` (`checked <=> root.enable_dual_pass; toggled => { root.dual_pass_changed(); }`).
   - If `root.enable_dual_pass`:
     - ComboBox for `root.dual_pass_mode_labels` (`current-index <=> root.dual_pass_mode_index; selected => { root.dual_pass_mode_changed(); }`).
     - FilterChip: `"LLM Synthesis"` (`checked <=> root.use_llm_synthesis; toggled => { root.llm_synthesis_changed(); }`).

---

## 5. Constraint & Compatibility

* Keep all existing Slint callback and property names exactly as they are so `src/main.rs` compiles without modifications.
* Maintain the M3 theme styling palette (`M3.surface-container`, `M3.primary`, `M3.outline-variant`, etc.).
* Run `cargo check` or `cargo test` after editing to ensure the Slint compiler passes without syntax errors.
