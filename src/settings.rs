//! Saved settings (`~/.config/prompt-condenser/settings.json`) and the UI zoom.

use super::*;

fn default_true() -> bool {
    true
}

/// Window layout, connection settings and the prompts, remembered between sessions.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(crate) struct Settings {
    pub(crate) before_ratio: f32,
    pub(crate) side_width: f32,
    pub(crate) cards_strip_height: f32,
    pub(crate) ollama_host: String,
    /// Where the LoRA files are, for trigger-word suggestions (empty: `{comfyui_folder}/models/loras`).
    pub(crate) lora_folder: String,
    /// The ComfyUI folder (empty: `$COMFYUI_PATH`, else `$HOME/ComfyUI`; `lora_meta::comfyui_folder`).
    pub(crate) comfyui_folder: String,
    /// Copy format (`condenser::ExportFormat::from_index`).
    pub(crate) export_index: i32,
    /// Whole-UI zoom on top of the monitor's scale (1.0 = Slint's normal size).
    pub(crate) ui_zoom: f32,
    /// Target checkpoint (0 Anima, 1 Pony, 2 Illustrious, 3 SDXL, 4 SD1.5, 5 Flux).
    pub(crate) checkpoint_index: i32,
    /// AI execution mode (0 GPU, 1 CPU, 2 AI Off).
    pub(crate) ai_mode_index: i32,
    /// Target camera view (0 None, 1 360, 2 Upper Body, etc.).
    pub(crate) target_view_index: i32,
    /// Camera angle (0 None, 1 Straight-on, etc.).
    pub(crate) angle_index: i32,
    /// Pose (0 None, 1 Standing, etc.).
    pub(crate) pose_index: i32,
    /// The old single Body dropdown (0 None, 1 Flat chest … 8 Muscular): only read, to fill
    /// `bust_index` / `frame_index` from an older settings file.
    #[serde(skip_serializing)]
    pub(crate) proportion_index: i32,
    /// Bust dropdown (`profiles::BUSTS`; -1 = not saved yet).
    pub(crate) bust_index: i32,
    /// Frame dropdown (`profiles::FRAMES`; -1 = not saved yet).
    pub(crate) frame_index: i32,
    /// AI detail level (0 Faithful, 1 Balanced, 2 Rich).
    pub(crate) detail_index: i32,
    /// Image reading: false = Danbooru tagger, true = AI.
    pub(crate) image_use_ai: bool,
    /// The Before prompt and the Keep & Emphasize list from the last session.
    pub(crate) raw_prompt: String,
    pub(crate) pinned_emphasis: String,
    pub(crate) lora_tags: String,
    pub(crate) arranged: bool,
    /// Anima Ref Adapter mode: the After prompt keeps only the scene.
    #[serde(default)]
    pub(crate) ref_adapter_mode: bool,
    pub(crate) natural: bool,
    /// 0 = 1 Character, 1 = 2 Characters.
    pub(crate) app_tab: i32,
    pub(crate) char2_prompt: String,
    pub(crate) char2_lora_tags: String,
    pub(crate) char3_prompt: String,
    pub(crate) char3_lora_tags: String,
    pub(crate) char4_prompt: String,
    pub(crate) char4_lora_tags: String,
    pub(crate) char_break: bool,
    pub(crate) scene_prompt: String,
    pub(crate) image_identity_only: bool,
    /// The AI model picked in the dropdown (its Ollama name), restored at startup.
    #[serde(default)]
    pub(crate) ai_model: String,
    pub(crate) image_second_pass: bool,
    pub(crate) scene_body_shape: bool,
    pub(crate) enable_dual_pass: bool,
    pub(crate) dual_pass_mode_index: i32,
    pub(crate) use_llm_synthesis: bool,
    /// THEME dropdown: 0 OLED Pitch Black, 1 M3 Deep Violet, 2 OLED Cyber Emerald,
    /// 3 M3 Ocean Blue, 4 M3 Crimson Ember, 5 M3 Mocha Gold.
    pub(crate) theme_index: i32,
    /// Prompt vs LoRA box split inside each character's card.
    pub(crate) lora_split_ratios: [f32; 4],
    /// The Options drawer open or closed.
    pub(crate) options_expanded: bool,
    /// Output view: 0 Rule-based, 1 AI Improved, 2 Dynamic, 3 Section Blocks.
    pub(crate) after_view: i32,
    /// The Recommended Negative box open (true for settings files from before the accordion; a new
    /// install starts collapsed, see `load_settings`).
    pub(crate) negative_expanded: bool,
    /// Zen mode: the View / Options / Folders rows and the reference bar hidden.
    pub(crate) zen_mode: bool,
    /// Suggest fabric fit and dynamic poses for existing garments/anatomy.
    #[serde(default = "default_true")]
    pub(crate) fit_suggestions: bool,
    /// Allow exaggerated/stylized cartoon proportions for shortstack.
    #[serde(default)]
    pub(crate) exaggerated_proportions: bool,
    /// Saved user preset profiles.
    #[serde(default)]
    pub(crate) user_presets: Vec<crate::profiles::UserPresetProfile>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            before_ratio: 0.4,
            side_width: 320.0,
            cards_strip_height: 250.0,
            ui_zoom: 0.75,
            checkpoint_index: 0,
            ai_mode_index: 0,
            target_view_index: 0,
            angle_index: 0,
            pose_index: 0,
            proportion_index: 0,
            bust_index: -1,
            frame_index: -1,
            detail_index: 1,
            image_use_ai: false,
            raw_prompt: String::new(),
            pinned_emphasis: String::new(),
            lora_tags: String::new(),
            arranged: false,
            ref_adapter_mode: false,
            natural: false,
            app_tab: 0,
            char2_prompt: String::new(),
            char2_lora_tags: String::new(),
            char3_prompt: String::new(),
            char3_lora_tags: String::new(),
            char4_prompt: String::new(),
            char4_lora_tags: String::new(),
            char_break: true,
            scene_prompt: String::new(),
            image_identity_only: false,
            ai_model: String::new(),
            image_second_pass: false,
            scene_body_shape: false,
            // the measured-best chain (tagger -> ToriiGate / checklist -> E4B -> polish) is the default;
            // dual pass stays available under "Advanced"
            enable_dual_pass: false,
            dual_pass_mode_index: 0,
            use_llm_synthesis: true,
            theme_index: 1,
            lora_split_ratios: [0.7; 4],
            options_expanded: false,
            after_view: 3,
            ollama_host: "localhost:11434".into(),
            lora_folder: String::new(),
            comfyui_folder: String::new(),
            export_index: 0,
            negative_expanded: true,
            zen_mode: false,
            fit_suggestions: true,
            exaggerated_proportions: false,
            user_presets: Vec::new(),
        }
    }
}

pub(crate) fn settings_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))?;
    Some(base.join("prompt-condenser").join("settings.json"))
}

/// The AI model saved in settings.json, if any.
pub(crate) fn saved_ai_model() -> Option<String> {
    settings_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<Settings>(&t).ok())
        .map(|st| st.ai_model)
        .filter(|m| !m.is_empty())
}

pub(crate) fn load_settings(ui: &AppWindow) {
    let saved: Option<Settings> = settings_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok());
    // a new install starts with the negative collapsed; older settings files keep it open
    let negative_expanded = saved.as_ref().is_some_and(|s| s.negative_expanded);
    let st = saved.unwrap_or_default();
    ui.set_negative_expanded(negative_expanded);
    ui.set_zen_mode(st.zen_mode);
    ui.set_before_ratio(st.before_ratio.clamp(0.15, 0.8));
    ui.set_side_width(st.side_width.max(160.0));
    ui.set_cards_strip_height(st.cards_strip_height.max(140.0));
    ui.set_ui_zoom(st.ui_zoom.clamp(ZOOM_MIN, ZOOM_MAX));
    ui.set_checkpoint_index(st.checkpoint_index.clamp(0, 5));
    ui.set_ai_mode_index(st.ai_mode_index.clamp(0, 2));
    ui.set_target_view_index(st.target_view_index.clamp(0, 7));
    ui.set_angle_index(st.angle_index.max(0));
    ui.set_pose_index(st.pose_index.max(0));
    let (bust, frame) = if st.bust_index < 0 && st.frame_index < 0 {
        profiles::bust_frame_from_old_body(st.proportion_index)
    } else {
        (st.bust_index, st.frame_index)
    };
    ui.set_bust_index(bust.clamp(0, profiles::BUSTS.len() as i32 - 1));
    ui.set_frame_index(frame.clamp(0, profiles::FRAMES.len() as i32 - 1));
    ui.set_detail_index(st.detail_index.clamp(0, 2));
    ui.set_image_use_ai(st.image_use_ai);
    ui.set_raw_prompt(st.raw_prompt.into());
    ui.set_pinned_emphasis(st.pinned_emphasis.into());
    ui.set_lora_tags(st.lora_tags.into());
    ui.set_arranged_enabled(st.arranged);
    ui.set_fit_suggestions(st.fit_suggestions);
    ui.set_exaggerated_proportions(st.exaggerated_proportions);
    ui.set_ref_adapter_mode(st.ref_adapter_mode);
    ui.set_natural_enabled(st.natural);
    ui.set_app_tab(st.app_tab.clamp(0, 3));
    ui.set_char2_prompt(st.char2_prompt.into());
    ui.set_char2_lora_tags(st.char2_lora_tags.into());
    ui.set_char3_prompt(st.char3_prompt.into());
    ui.set_char3_lora_tags(st.char3_lora_tags.into());
    ui.set_char4_prompt(st.char4_prompt.into());
    ui.set_char4_lora_tags(st.char4_lora_tags.into());
    ui.set_char_break_enabled(st.char_break);
    ui.set_scene_prompt(st.scene_prompt.into());
    ui.set_image_identity_only(st.image_identity_only);
    // the 2nd Pass and Dual-Pass options were removed (2026-11-12): an old saved "on" must not switch them on unseen
    ui.set_image_second_pass(false);
    ui.set_scene_body_shape(st.scene_body_shape);
    ui.set_enable_dual_pass(false);
    ui.set_dual_pass_mode_index(st.dual_pass_mode_index.clamp(0, 1));
    ui.set_use_llm_synthesis(st.use_llm_synthesis);
    ui.set_theme_index(st.theme_index.clamp(0, 5));
    ui.set_options_expanded(st.options_expanded);
    ui.set_after_view(st.after_view.clamp(0, 4));
    let [r1, r2, r3, r4] = st.lora_split_ratios.map(|r| r.clamp(0.2, 0.9));
    ui.set_lora_split_ratio_c1(r1);
    ui.set_lora_split_ratio_c2(r2);
    ui.set_lora_split_ratio_c3(r3);
    ui.set_lora_split_ratio_c4(r4);
    if !st.ollama_host.trim().is_empty() {
        ui.set_ollama_host(st.ollama_host.into());
    }
    ui.set_lora_folder(st.lora_folder.into());
    ui.set_comfyui_folder(st.comfyui_folder.into());
    ui.set_export_index(st.export_index.clamp(0, 3));
}

pub(crate) fn save_settings(ui: &AppWindow) {
    let st = Settings {
        before_ratio: ui.get_before_ratio(),
        side_width: ui.get_side_width(),
        cards_strip_height: ui.get_cards_strip_height(),
        ui_zoom: ui.get_ui_zoom(),
        checkpoint_index: ui.get_checkpoint_index(),
        ai_mode_index: ui.get_ai_mode_index(),
        target_view_index: ui.get_target_view_index(),
        angle_index: ui.get_angle_index(),
        pose_index: ui.get_pose_index(),
        proportion_index: 0,
        bust_index: ui.get_bust_index(),
        frame_index: ui.get_frame_index(),
        detail_index: ui.get_detail_index(),
        image_use_ai: ui.get_image_use_ai(),
        raw_prompt: ui.get_raw_prompt().to_string(),
        pinned_emphasis: ui.get_pinned_emphasis().to_string(),
        lora_tags: ui.get_lora_tags().to_string(),
        arranged: ui.get_arranged_enabled(),
        ref_adapter_mode: ui.get_ref_adapter_mode(),
        natural: ui.get_natural_enabled(),
        app_tab: ui.get_app_tab(),
        char2_prompt: ui.get_char2_prompt().to_string(),
        char2_lora_tags: ui.get_char2_lora_tags().to_string(),
        char3_prompt: ui.get_char3_prompt().to_string(),
        char3_lora_tags: ui.get_char3_lora_tags().to_string(),
        char4_prompt: ui.get_char4_prompt().to_string(),
        char4_lora_tags: ui.get_char4_lora_tags().to_string(),
        char_break: ui.get_char_break_enabled(),
        scene_prompt: ui.get_scene_prompt().to_string(),
        image_identity_only: ui.get_image_identity_only(),
        ai_model: {
            let mut picked = None;
            with_state(|state| picked = selected_choice(ui, state).filter(|c| c.installed).map(|c| c.name));
            picked.or_else(saved_ai_model).unwrap_or_default()
        },
        image_second_pass: ui.get_image_second_pass(),
        scene_body_shape: ui.get_scene_body_shape(),
        enable_dual_pass: ui.get_enable_dual_pass(),
        dual_pass_mode_index: ui.get_dual_pass_mode_index(),
        use_llm_synthesis: ui.get_use_llm_synthesis(),
        theme_index: ui.get_theme_index(),
        options_expanded: ui.get_options_expanded(),
        after_view: ui.get_after_view(),
        lora_split_ratios: [
            ui.get_lora_split_ratio_c1(),
            ui.get_lora_split_ratio_c2(),
            ui.get_lora_split_ratio_c3(),
            ui.get_lora_split_ratio_c4(),
        ],
        ollama_host: ui.get_ollama_host().to_string(),
        lora_folder: ui.get_lora_folder().to_string(),
        comfyui_folder: ui.get_comfyui_folder().to_string(),
        export_index: ui.get_export_index(),
        negative_expanded: ui.get_negative_expanded(),
        zen_mode: ui.get_zen_mode(),
        fit_suggestions: ui.get_fit_suggestions(),
        exaggerated_proportions: ui.get_exaggerated_proportions(),
        user_presets: settings_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|j| serde_json::from_str::<Settings>(&j).ok())
            .map(|s| s.user_presets)
            .unwrap_or_default(),
    };
    if let Some(path) = settings_path() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_string_pretty(&st) {
            let _ = std::fs::write(path, json);
        }
    }
}

/// Internal zoom shown as "100%" in the UI size control.
pub(crate) const ZOOM_BASE: f32 = 0.75;

/// 70% … 200% as shown, in 10% steps.
pub(crate) const ZOOM_MIN: f32 = ZOOM_BASE * 0.7;

pub(crate) const ZOOM_MAX: f32 = ZOOM_BASE * 2.0;

/// Scales the whole UI: Slint's Material widgets have fixed minimum sizes, so a
/// global scale is the only way to make everything smaller together.
/// The monitor's own scale (`base`) is tracked so moving to another screen still works.
pub(crate) struct Zoom {
    pub(crate) base: std::cell::Cell<f32>,
    pub(crate) applied: std::cell::Cell<f32>,
}

impl Zoom {
    pub(crate) fn apply(&self, ui: &AppWindow) {
        let target = self.base.get() * ui.get_ui_zoom();
        self.applied.set(target);
        ui.window().dispatch_event(slint::platform::WindowEvent::ScaleFactorChanged { scale_factor: target });
        ui.set_ui_zoom_percent((ui.get_ui_zoom() / ZOOM_BASE * 100.0).round() as i32);
    }

    /// Called periodically: if the platform reset the scale (new monitor, settings
    /// change), take that as the new base and re-apply the zoom on top of it.
    pub(crate) fn check(&self, ui: &AppWindow) {
        let current = ui.window().scale_factor();
        if (current - self.applied.get()).abs() > 0.001 {
            self.base.set(current);
            self.apply(ui);
        }
    }
}

/// Logical window width below which the top bar and the reference bar use two rows.
pub(crate) const NARROW_WIDTH: f32 = 1250.0;

/// Sets `narrow` from the window's logical width (run by the 250 ms zoom timer; a binding on
/// `root.width` would loop through the window's layout info).
pub(crate) fn update_narrow(ui: &AppWindow) {
    let width = ui.window().size().width as f32 / ui.window().scale_factor().max(0.1);
    let narrow = width < NARROW_WIDTH;
    if ui.get_narrow() != narrow {
        ui.set_narrow(narrow);
    }
}

/// Wires the Options FOLDERS row (LoRA / ComfyUI pickers, Install Node) and the theme.
pub(crate) fn wire_settings_ui(ui: &AppWindow, state: &State) {
    // prompt history overlay (Stash / star / Restore)
    history::wire_history(ui, state);
    // LoRA folder picker (trigger-word suggestions)
    {
        let weak = ui.as_weak();
        ui.on_pick_lora_folder_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut dialog = rfd::FileDialog::new().set_title("Pick your LoRA folder");
            let current = ui.get_lora_folder().to_string();
            if std::path::Path::new(&current).is_dir() {
                dialog = dialog.set_directory(&current);
            }
            if let Some(dir) = dialog.pick_folder() {
                ui.set_lora_folder(dir.to_string_lossy().into_owned().into());
                ui.invoke_analyze_requested();
            }
        });
    }

    // ComfyUI folder picker (the LoRA folder defaults to its models/loras)
    {
        let weak = ui.as_weak();
        ui.on_pick_comfyui_folder_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut dialog = rfd::FileDialog::new().set_title("Pick your ComfyUI folder");
            if let Some(current) = lora_meta::comfyui_folder(&ui.get_comfyui_folder()).filter(|p| p.is_dir()) {
                dialog = dialog.set_directory(current);
            }
            if let Some(dir) = dialog.pick_folder() {
                ui.set_comfyui_folder(dir.to_string_lossy().into_owned().into());
                ui.invoke_analyze_requested();
            }
        });
    }

    // "⚡ Install Node": the ComfyUI add-on into <ComfyUI>/custom_nodes (never through a symlink)
    {
        let weak = ui.as_weak();
        ui.on_install_comfyui_node_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let msg = match lora_meta::comfyui_folder(&ui.get_comfyui_folder()) {
                None => "No ComfyUI folder: pick it with 📁 first.".to_string(),
                Some(folder) => match comfyui::install_comfyui_addon(&folder) {
                    Ok(done) => done.message(),
                    Err(e) => format!("Couldn't install the node: {e:#}"),
                },
            };
            ui.set_status_message(msg.into());
        });
    }

    // --- Theme: saved as soon as it changes ---
    {
        let weak = ui.as_weak();
        ui.on_theme_changed(move || {
            if let Some(ui) = weak.upgrade() {
                save_settings(&ui);
            }
        });
    }
}
