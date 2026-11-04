# Backend Handoff Spec for Claude (`ui/app.slint` → `src/`)

The Material 3 + OLED UI overhaul has been implemented in `ui/app.slint`. All existing `AppWindow` properties and callbacks used by `src/main.rs` are preserved 100%, so existing code continues to work.

Below are the new `AppWindow` properties and callbacks in `ui/app.slint` for Claude to wire up in the Rust backend (`src/main.rs`, `src/scene.rs`, `src/ai_engine.rs`):

---

## 1. Scene & Pose Reference Image (`🌄 Scene / Pose Ref…`)

### Slint Properties & Callbacks in `ui/app.slint`
```slint
in-out property <bool> scene_ref_loaded: false;
in-out property <image> scene_ref_thumb;
in-out property <string> scene_ref_status: "";
callback load_scene_ref_requested();
callback remove_scene_ref_requested();
```

### Required Backend Behavior
1. **When no Scene Reference image is loaded (`scene_ref_loaded == false`)**:
   - Complete no-op. Do not modify `scene_prompt` or character prompts, and keep `scene_ref_loaded = false` so the preview drawer in `ui/app.slint` stays collapsed (`0px`).
2. **On `ui.on_load_scene_ref_requested()`**:
   - Open an image file picker (`rfd::FileDialog`).
   - Decode a thumbnail and set `ui.set_scene_ref_thumb(...)` and `ui.set_scene_ref_loaded(true)` (this automatically expands the clickable preview drawer in `ui/app.slint`).
   - Run the active image reader (`image_use_ai == false` → WD14 Tagger on CPU; `image_use_ai == true` → Ollama Vision on GPU/CPU) to extract **only Pose, Action, Camera Framing/Angle, Background Setting, and Lighting** — **strictly stripping all character identity/appearance tags**:
     - Add `scene::filter_scene_and_pose_only(prompt: &str) -> String` in `src/scene.rs`:
       - **Keep only**: `Slot::Pose`, `Slot::Camera`, `Slot::Setting`, `Slot::Light`, `Slot::Style` (excluding artist/character names).
       - **Strip all**: `Slot::Subject` (`1girl`, `2girls`, `solo`, `elf`, etc.), `Slot::Other`, `Slot::Traits`, `Slot::Hair`, `Slot::Eyes`, `Slot::Body`, `Slot::Outfit`, `Slot::Lora`, `Slot::Quality`, `Slot::Meta`.
     - If `image_use_ai == true`, use a dedicated system prompt in `src/ai_engine.rs` that instructs the vision model to output **only** body poses, character interactions/positioning, camera framing/angle, setting/background, and lighting — never hair, eyes, face, species, or clothing.
   - Apply the extracted Pose + Scene tags:
     - On multi-character tabs (`app_tab >= 1`), put/merge the extracted tags into `scene_prompt` (`SHARED PROMPT`) and set `scene_ref_status` to `"Extracted Pose & Scene: " + &tags`.
     - On the 1-character tab (`app_tab == 0`), append/merge the extracted pose & scene tags into `raw_prompt` (or keep them tracked so removing the scene reference cleanly removes them) and set `scene_ref_status`.
   - Trigger `run_analysis(&ui)`.
3. **On `ui.on_remove_scene_ref_requested()`** (triggered when the user clicks directly on the Scene Reference preview image):
   - Set `ui.set_scene_ref_loaded(false)` and `ui.set_scene_ref_status("".into())`.
   - Remove/restore the tags that came from the Scene Reference image and re-run `run_analysis(&ui)`.

---

## 2. Theme Dropdown Persistence & Character 4 Stats

### Slint Properties & Callbacks in `ui/app.slint`
```slint
in-out property <int> theme_index <=> M3.theme-index; // 0=OLED Black, 1=M3 Violet, 2=Cyber Emerald, 3=Ocean Blue, 4=Crimson, 5=Mocha Gold
in-out property <bool> options_expanded: true;
callback theme_changed();
in-out property <string> char4_stats: "";
```

### Required Backend Behavior
1. **Theme Persistence (`Settings` in `src/main.rs`)**:
   - Add `#[serde(default)] theme_index: i32` and `#[serde(default = "default_true")] options_expanded: bool` to `Settings`.
   - Restore `ui.set_theme_index(saved.theme_index)` on startup and save on `ui.on_theme_changed()` / `save_settings(&ui)`.
2. **Character 4 Token Stats**:
   - In `run_analysis(&ui)`, compute token/chunk stats for `char4_prompt` and call `ui.set_char4_stats(...)` just like `char2_stats` and `char3_stats`.
3. **Multiline Per-Character LoRA Fields**:
   - `lora_tags`, `char2_lora_tags`, `char3_lora_tags`, and `char4_lora_tags` are now multiline `TextEdit` boxes underneath each character's prompt in `ui/app.slint`. Ensure newline-separated LoRA lines in `lora_tags` / `char2..4_lora_tags` are normalized with commas when combined into the final prompt.
