//! The Danbooru Words dialog.

use super::*;

/// Wires the Danbooru Words dialog.
pub(crate) fn wire_translate(ui: &AppWindow, state: &State) {
    // --- Danbooru translator dialog ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_translate_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let text = ui.get_translate_input().to_string();
            if text.trim().is_empty() {
                ui.set_translate_status("Type what you want first.".into());
                return;
            }
            if AiMode::from_index(ui.get_ai_mode_index()) == AiMode::Disabled || !ui.get_ollama_connected() {
                ui.set_translate_status("Needs the AI: turn it on and connect to Ollama.".into());
                return;
            }
            let Some(choice) = selected_choice(&ui, &state).filter(|c| c.installed) else {
                ui.set_translate_status("Pick an installed AI model first.".into());
                return;
            };
            let req = ai_engine::TranslateRequest {
                host: ui.get_ollama_host().to_string(),
                model: choice.name.clone(),
                mode: AiMode::from_index(ui.get_ai_mode_index()),
                text,
                checkpoint: CheckpointProfile::from_index(ui.get_checkpoint_index()),
            };
            ui.set_translate_busy(true);
            ui.set_translate_status(format!("Finding Danbooru tags with {}…", choice.label()).into());
            let weak = weak.clone();
            std::thread::spawn(move || {
                let started = std::time::Instant::now();
                let result = ai_engine::danbooru_translate(&req);
                let secs = started.elapsed().as_secs_f32();
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_translate_busy(false);
                    match result {
                        Ok(r) => {
                            ui.set_translate_tags(r.tags.clone().into());
                            ui.set_translate_natural(r.natural.clone().into());
                            let mut notes: Vec<String> = r.mapping.iter().map(|(f, t)| format!("“{f}” → {t}")).collect();
                            for (w, real) in &r.fixed {
                                notes.push(format!("Corrected “{w}” to the real Danbooru tag “{real}”."));
                            }
                            for (tag, similar) in &r.unknown {
                                notes.push(if similar.is_empty() {
                                    format!("“{tag}” isn't in the common Danbooru list (may work less reliably).")
                                } else {
                                    format!("“{tag}” isn't in the common Danbooru list. Similar real tags: {}", similar.join(", "))
                                });
                            }
                            ui.set_translate_notes(notes.join("\n").into());
                            ui.set_translate_status(format!("Done in {secs:.0} s. Edit the tags if you like, then add them.").into());
                        }
                        Err(e) => ui.set_translate_status(format!("Failed: {e:#}").into()),
                    }
                });
            });
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_translate_apply_requested(move |replace| {
            let Some(ui) = weak.upgrade() else { return };
            let mut add = ui.get_translate_tags().trim().trim_matches(',').trim().to_string();
            let natural = ui.get_translate_natural().trim().to_string();
            let nl = CheckpointProfile::from_index(ui.get_checkpoint_index()).supports_natural_language();
            if nl && ui.get_translate_use_natural() && !natural.is_empty() {
                add = if add.is_empty() { natural } else { format!("{add}, {natural}") };
            }
            if add.is_empty() {
                return;
            }
            let current = ui.get_raw_prompt().to_string();
            let current = current.trim().trim_end_matches(',').trim();
            let prompt = if replace || current.is_empty() { add } else { format!("{current}, {add}") };
            if replace {
                state.borrow_mut().card_choices.clear();
            }
            ui.set_raw_prompt(prompt.into());
            ui.set_translate_open(false);
            analyze(&ui, &state);
            ui.set_status_message(
                if replace { "Before prompt replaced with the Danbooru tags." } else { "Danbooru tags added to the Before prompt." }.into(),
            );
        });
    }
}
