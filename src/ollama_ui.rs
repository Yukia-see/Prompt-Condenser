//! Ollama connection, the model dropdown, the hardware badge and which model reads images.

use super::*;

/// One entry of the AI model dropdown.
#[derive(Clone)]
pub(crate) struct ModelChoice {
    /// Ollama model name.
    pub(crate) name: String,
    /// Set for the pre-packaged tiers; they can be downloaded when missing.
    pub(crate) tier: Option<AiModelTier>,
    pub(crate) installed: bool,
}

impl ModelChoice {
    pub(crate) fn label(&self) -> String {
        match (self.tier, self.installed) {
            (Some(t), true) => t.label().to_string(),
            (Some(t), false) => format!("{} (not installed)", t.label()),
            (None, _) => self.name.clone(),
        }
    }
}

pub(crate) fn format_gpu(gpu: &GpuInfo) -> String {
    if gpu.vram_total_mb > 0 {
        format!(
            "{} • VRAM {:.1} / {:.1} GB",
            gpu.name,
            gpu.vram_used_mb as f64 / 1024.0,
            gpu.vram_total_mb as f64 / 1024.0
        )
    } else {
        gpu.name.clone()
    }
}

/// Probes GPU/VRAM off the UI thread (`vulkaninfo` can take a moment).
pub(crate) fn refresh_hardware_badge(ui: &AppWindow) {
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let badge = format_gpu(&ai_engine::probe_gpu());
        let _ = weak.upgrade_in_event_loop(move |ui| ui.set_hardware_badge(badge.into()));
    });
}

/// The pre-packaged tiers first (installed or not), then everything else Ollama has.
pub(crate) fn build_model_choices(installed: &[OllamaModel]) -> Vec<ModelChoice> {
    let installed: Vec<String> = installed
        .iter()
        .map(|m| m.name.clone())
        .collect();
    let is_tier_match = |m: &str, t: AiModelTier| {
        ai_engine::same_model(m, t.ollama_name())
            || (t == AiModelTier::Best && (m.contains("12b") || m.contains("gemma-4-abliterated")))
    };
    let mut choices: Vec<ModelChoice> = AiModelTier::ALL
        .iter()
        .map(|t| ModelChoice {
            name: t.ollama_name().to_string(),
            tier: Some(*t),
            installed: installed.iter().any(|m| is_tier_match(m, *t)),
        })
        .collect();
    for m in &installed {
        // Hide dedicated vision encoders (Moondream, Qwen-VL, ToriiGate) from the text prompt model dropdown
        if ai_engine::is_toriigate(m) || m.to_lowercase().contains("moondream")
            || (m.to_lowercase().contains("qwen") && (m.to_lowercase().contains("vl") || m.to_lowercase().contains("-vl")))
        {
            continue;
        }
        if !AiModelTier::ALL.iter().any(|t| is_tier_match(m, *t)) {
            choices.push(ModelChoice { name: m.clone(), tier: None, installed: true });
        }
    }
    choices
}

/// Replaces the model dropdown, keeping the current selection when it still exists.
pub(crate) fn set_model_choices(ui: &AppWindow, state: &State, choices: Vec<ModelChoice>) {
    // Keep the user's pick only if it was a real, installed model (not the
    // placeholder shown before Ollama answered).
    let previous = selected_choice(ui, state).filter(|c| c.installed).map(|c| c.name);
    let saved = state.borrow().preferred_model.clone();
    let index = previous
        .and_then(|p| choices.iter().position(|c| ai_engine::same_model(&c.name, &p)))
        // the model picked last time (it used to come back as the last installed tier, the 12B)
        .or_else(|| saved.and_then(|m| choices.iter().position(|c| c.installed && ai_engine::same_model(&c.name, &m))))
        .or_else(|| choices.iter().rposition(|c| c.installed && c.tier.is_some()))
        .or_else(|| choices.iter().position(|c| c.installed))
        .unwrap_or(0);
    let labels: Vec<SharedString> = choices.iter().map(|c| c.label().into()).collect();
    state.borrow_mut().model_choices = choices;
    ui.set_model_labels(ModelRc::new(VecModel::from(labels)));
    ui.set_model_index(index as i32);
    update_download_flag(ui, state);
}

pub(crate) fn selected_choice(ui: &AppWindow, state: &State) -> Option<ModelChoice> {
    state.borrow().model_choices.get(ui.get_model_index().max(0) as usize).cloned()
}

pub(crate) fn update_download_flag(ui: &AppWindow, state: &State) {
    let needs = selected_choice(ui, state).map(|c| !c.installed).unwrap_or(false);
    ui.set_model_needs_download(needs);
}

/// Redoes the vision encoder by invalidating all cached image and reference analysis
/// and re-running the vision model against loaded images.
pub(crate) fn refresh_vision_encoder(ui: &AppWindow, state: &State) {
    if ui.get_tagger_busy() {
        return;
    }
    {
        let mut st = state.borrow_mut();
        st.image_gens.bump_all();
        st.image_ai = None;
        st.two_ai = None;
        st.char_images = [None, None, None, None];
        st.last_tags = None;
        st.two_tags = None;
    }
    let enc_name = vision_model(ui, state).unwrap_or_else(|| "vision encoder".into());
    ui.set_image_status(format!("↻ Redoing Vision Encoder ({enc_name})…").into());

    if ui.get_images_pending() || ui.get_scene_ref_loaded() || ui.get_char1_image_loaded() {
        run_read_queue(ui, state);
    } else if state.borrow().image_path.is_some() {
        if ui.get_app_tab() >= 1 {
            run_two_image(ui, state);
        } else {
            run_ai_image(ui, state);
        }
    }
}

/// Asks Ollama for its models (off the UI thread) and refreshes the dropdown.
/// `quiet` is for background retries: no "Connecting..." flicker.
/// `redo_vision`: if true and images are loaded, automatically re-runs the vision encoder.
pub(crate) fn connect_ollama(ui: &AppWindow, quiet: bool, redo_vision: bool) {
    let host = ui.get_ollama_host().to_string();
    ui.set_ollama_busy(true);
    if !quiet {
        ui.set_ollama_status("Connecting...".into());
    }
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let result = ai_engine::list_ollama_models(&host);
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_ollama_busy(false);
            with_state(|state| match result {
                Ok((base, models)) => {
                    let n = models.len();
                    state.borrow_mut().vision_models = models.iter().filter(|m| m.vision).map(|m| m.name.clone()).collect();
                    set_model_choices(&ui, state, build_model_choices(&models));
                    ui.set_ollama_connected(true);
                    ui.set_ollama_status(format!("Connected to {base} — {n} model(s)").into());

                    if redo_vision {
                        let has_image = state.borrow().image_path.is_some()
                            || state.borrow().scene_ref_path.is_some()
                            || state.borrow().char_paths.iter().any(|p| p.is_some())
                            || ui.get_image_loaded()
                            || ui.get_scene_ref_loaded()
                            || ui.get_char1_image_loaded();
                        if has_image && !ui.get_tagger_busy() {
                            ui.set_image_use_ai(true);
                            refresh_vision_encoder(&ui, state);
                        }
                    }
                }
                Err(_) => {
                    set_model_choices(&ui, state, build_model_choices(&[]));
                    ui.set_ollama_connected(false);
                    let msg = if ai_engine::is_local_host(&host) {
                        "Ollama isn't running on this PC — start it with: systemctl start ollama. Retrying every 5 s..."
                    } else {
                        "No Ollama at that address — on that PC set OLLAMA_HOST=0.0.0.0 and open port 11434. Retrying every 5 s..."
                    };
                    ui.set_ollama_status(msg.into());
                }
            });
        });
    });
}

/// A model name short enough for the status line: `huihui_ai/gemma-4-abliterated:12b` -> `gemma-4-abliterated:12b`.
pub(crate) fn short_model_name(name: &str) -> &str {
    let n = name.rsplit('/').next().unwrap_or(name);
    n.strip_suffix(":latest").unwrap_or(n)
}

/// The text model that writes image prompts: E4B (Good) when installed, else the dropdown's model.
pub(crate) fn default_text_model(ui: &AppWindow, state: &State) -> String {
    let st = state.borrow();
    // Image reads always write with E4B (Good) when it's installed, whatever the dropdown says: it scored best
    // on the image tests (the 12B wrote worse image prompts and shares the memory badly with the tagger and the
    // vision model). The dropdown only picks the Deep Scan's model. E4B can't see images, so the 2nd Pass
    // Visual Check (which attaches the image to the writer) stays off while it writes.
    if let Some(e4b) = st.model_choices.iter().find(|c| c.installed && c.tier == Some(AiModelTier::Good)) {
        return e4b.name.clone();
    }
    st.model_choices
        .get(ui.get_model_index().max(0) as usize)
        .filter(|c| c.installed)
        .map(|c| c.name.clone())
        .or_else(|| st.model_choices.iter().find(|c| c.installed && (c.name.contains("12b") || c.name.contains("llama") || c.name.contains("e4b"))).map(|c| c.name.clone()))
        .or_else(|| st.model_choices.iter().find(|c| c.installed).map(|c| c.name.clone()))
        .unwrap_or_else(|| profiles::AiModelTier::Best.ollama_name().to_string())
}

/// The vision encoder for reading images: defaults to Qwen-VL (abliterated 3B preferred), then Moondream, else selected or first vision model.
pub(crate) fn vision_model(ui: &AppWindow, state: &State) -> Option<String> {
    let st = state.borrow();
    // 0. ToriiGate when installed: an anime captioning model built to use booru tags as hints. In the
    //    tests it beat the Qwen-VL 3B clearly on multi-character images (average F1 23% -> 30%).
    if let Some(m) = st.vision_models.iter().find(|m| ai_engine::is_toriigate(m)) {
        return Some(m.clone());
    }
    // 1. Default to Qwen-VL 3B (especially abliterated) if available, or any Qwen-VL
    if let Some(m) = st.vision_models.iter().find(|m| {
        let l = m.to_lowercase();
        l.contains("qwen") && (l.contains("vl") || l.contains("-vl")) && l.contains("3b")
    }).or_else(|| st.vision_models.iter().find(|m| {
        let l = m.to_lowercase();
        l.contains("qwen") && (l.contains("vl") || l.contains("-vl"))
    })) {
        return Some(m.clone());
    }
    // 2. Fall back to Moondream if available
    if let Some(m) = st.vision_models.iter().find(|m| m.to_lowercase().contains("moondream")) {
        return Some(m.clone());
    }
    let selected = st.model_choices.get(ui.get_model_index().max(0) as usize).map(|c| c.name.clone());
    selected
        .filter(|s| st.vision_models.iter().any(|v| ai_engine::same_model(v, s)))
        .or_else(|| st.vision_models.first().cloned())
}

/// Wires AI mode / model changes, the Ollama connection and model downloads.
pub(crate) fn wire_ollama(ui: &AppWindow, state: &State) {
    // Changing the AI mode or model refreshes the page: AI cards came from the
    // previous model, so they are dropped and the rules re-run.
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_ai_settings_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui); // remember the AI model / mode right away, not only on close
            clear_ai_cards(&state);
            analyze(&ui, &state);
            update_download_flag(&ui, &state);
            ui.set_run_badge("".into());
            let mode = AiMode::from_index(ui.get_ai_mode_index());
            let msg = match (mode, selected_choice(&ui, &state)) {
                (AiMode::Disabled, _) => "AI off — rules only. Refreshed.".to_string(),
                (_, Some(c)) if !c.installed => format!("{} isn't installed yet — click Download Model.", c.label()),
                (_, Some(c)) => format!("Switched to {} ({}) — refreshed.", c.label(), mode.display_name()),
                (_, None) => "No AI model available — connect to Ollama first.".to_string(),
            };
            ui.set_status_message(msg.into());
        });
    }

    // --- Ollama connection ---
    {
        let weak = ui.as_weak();
        ui.on_connect_ollama_requested(move || {
            if let Some(ui) = weak.upgrade() {
                connect_ollama(&ui, false, true);
            }
        });
    }

    // --- Download a pre-packaged model through Ollama ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_download_model_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(tier) = selected_choice(&ui, &state).and_then(|c| c.tier) else { return };
            let host = ui.get_ollama_host().to_string();
            ui.set_ai_busy(true);
            ui.set_status_message(format!("Asking Ollama to download {}...", tier.label()).into());

            let (tx, rx) = mpsc::channel::<String>();
            let progress_weak = weak.clone();
            std::thread::spawn(move || {
                for msg in rx {
                    let _ = progress_weak.upgrade_in_event_loop(move |ui| ui.set_status_message(msg.into()));
                }
            });

            let done_weak = weak.clone();
            std::thread::spawn(move || {
                let result = ai_engine::pull_tier_model(&host, tier, tx);
                let _ = done_weak.upgrade_in_event_loop(move |ui| {
                    ui.set_ai_busy(false);
                    match result {
                        Ok(()) => ui.set_status_message(format!("{} is ready.", tier.label()).into()),
                        Err(e) => ui.set_status_message(format!("Download failed: {e:#}").into()),
                    }
                    connect_ollama(&ui, false, false);
                });
            });
        });
    }
}
