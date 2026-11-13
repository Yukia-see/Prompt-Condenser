//! Cancel and the Deep AI scan.

use super::*;

/// Wires cancel and the Deep AI scan.
pub(crate) fn wire_deep_scan(ui: &AppWindow, state: &State) {
    // --- Cancel (Deep Scan / image read): stops after the current step ---
    {
        let weak = ui.as_weak();
        ui.on_cancel_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            ai_engine::request_cancel();
            // a cancelled read must not hand over to a waiting Deep Scan (which would clear the cancel)
            with_state(|state| {
                let mut st = state.borrow_mut();
                st.scan_after_image = false;
                st.scan_image_done = false;
            });
            ui.set_status_message("Cancelling: stopping after the current step…".into());
            if ui.get_tagger_busy() {
                ui.set_image_status("Cancelling: stopping after the current step…".into());
            }
        });
    }

    // --- Deep AI scan ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_run_ai_scan_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // Coming from an image read that just finished (successfully or not): scan with what it
            // produced. Without this, a failed AI split left the image "unread" and the scan read it
            // again, forever.
            let image_done = std::mem::take(&mut state.borrow_mut().scan_image_done);
            // With an image loaded and the switch on AI, read the image first, then scan.
            if !image_done && ui.get_image_use_ai() && ui.get_image_loaded() {
                if ui.get_tagger_busy() {
                    state.borrow_mut().scan_after_image = true;
                    ui.set_status_message("Deep scan starts as soon as the AI has read the image…".into());
                    return;
                }
                let two_tab = ui.get_app_tab() >= 1;
                let unread = if two_tab { state.borrow().two_ai.is_none() } else { state.borrow().image_ai.is_none() };
                if unread && vision_model(&ui, &state).is_some() {
                    state.borrow_mut().scan_after_image = true;
                    if two_tab {
                        run_two_image(&ui, &state);
                    } else {
                        run_ai_image(&ui, &state);
                    }
                    ui.set_status_message("Reading the image with AI first, then running the deep scan…".into());
                    return;
                }
            }
            let raw = ui.get_raw_prompt().to_string();
            if raw.trim().is_empty() {
                ui.set_status_message("Paste a prompt first.".into());
                return;
            }
            let Some(choice) = selected_choice(&ui, &state).filter(|c| c.installed) else {
                ui.set_status_message("Pick an installed AI model first.".into());
                return;
            };
            let mut config = scan_config(&ui);
            let prompts = char_prompts(&ui, &config);
            let c1 = prompts[0].clone();
            if config.two_char {
                // a pasted "… BREAK …" prompt was split: the scan sees the parts
                config.char2_prompt_raw = prompts.get(1).cloned().unwrap_or_default();
                for (i, p) in prompts.iter().enumerate().skip(2) {
                    if let Some(slot) = config.more_chars.get_mut(i - 2) {
                        slot.0 = p.clone();
                    }
                }
            }
            let scanned_key = ai_key(&ui);
            let req = AiScanRequest {
                raw_prompt: c1,
                config,
                mode: AiMode::from_index(ui.get_ai_mode_index()),
                host: ui.get_ollama_host().to_string(),
                model: choice.name.clone(),
                keep_alive: choice.tier.map(|t| t.keep_alive()).unwrap_or("30m").to_string(),
                think: false,
            };

            ai_engine::reset_cancel();
            ui.set_ai_busy(true);
            ui.set_status_message(
                format!(
                    "Analyzing the scene with {} — bigger models can take up to a minute (the first scan also loads the model)...",
                    choice.label()
                )
                .into(),
            );

            let weak = weak.clone();
            std::thread::spawn(move || {
                let result = ai_engine::run_ai_scan(&req);
                // a big model (10B+, the 12B) frees its memory 2 minutes after the scan instead of
                // staying loaded 30 minutes; a scan again within 2 minutes still finds it loaded
                if ai_engine::is_big_model(&req.model) {
                    ai_engine::set_keep_alive(&req.host, &req.model, "2m");
                }
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_ai_busy(false);
                    // `state` is Rc (UI thread only), so fetch it via a thread-local handle.
                    with_state(|state| {
                        match result {
                            Ok((mut ai_out, placement)) => {
                                ui.set_run_badge(format!("Last scan: {placement}").into());
                                let where_ran = format!("ran on {placement}");
                                if ai_key(&ui) != scanned_key {
                                    ui.set_status_message("Prompt changed during the AI scan — result discarded. Run it again.".into());
                                    return;
                                }
                                analyze(&ui, state);
                                let mut restored_tags = 0usize;
                                let found = {
                                    let mut st = state.borrow_mut();
                                    let tokens = ai_engine::request_tokens(&req);
                                    let rules: Vec<PromptChange> =
                                        st.changes.iter().chain(st.others.iter().flat_map(|o| o.changes.iter())).cloned().collect();
                                    let cards = ai_engine::convert_ai_changes_for(&ai_out.changes, &tokens, &rules, req.config.two_char);
                                    // on the 2 Characters tab, a card about character 2's tags goes with character 2
                                    let norms = |p: &str| parser::parse_prompt(p).into_iter().map(|t| t.normalized).collect::<Vec<_>>();
                                    // on the multi-character tabs, a card about one character's tags goes with that character
                                    let n = req.config.char_count();
                                    let mut boxes = vec![norms(&with_scene(&req.raw_prompt, &req.config))];
                                    boxes.extend((1..n).map(|i| norms(req.config.prompt_of(i))));
                                    let mut lists: Vec<Vec<PromptChange>> = vec![Vec::new(); n];
                                    for c in cards {
                                        let t = norms(&c.original);
                                        let owner = (1..n)
                                            .find(|&i| !t.is_empty() && t.iter().all(|x| boxes[i].contains(x)) && !t.iter().all(|x| boxes[0].contains(x)))
                                            .unwrap_or(0);
                                        lists[owner].push(c);
                                    }
                                    let mut lists = lists.into_iter();
                                    st.ai_changes = lists.next().unwrap_or_default();
                                    for (o, l) in st.others.iter_mut().zip(lists) {
                                        o.ai_changes = l;
                                    }
                                    // a Deep Scan of an image read's prompt only adds context: the read's tags it dropped
                                    // come back (measured 2026-11-12: the 12B dropped 10-25% of correct image tags)
                                    if n == 1 && st.image_prompt.as_deref() == Some(req.raw_prompt.trim()) && !ai_out.improved_prompt.trim().is_empty() {
                                        let dropped = condenser::dropped_tags(&req.raw_prompt, &ai_out.improved_prompt, req.config.checkpoint);
                                        if !dropped.is_empty() {
                                            ai_out.improved_prompt = format!("{}, {}", ai_out.improved_prompt.trim_end().trim_end_matches(','), dropped.join(", "));
                                            restored_tags = dropped.len();
                                        }
                                    }
                                    st.ai_scene = SceneResult::from_ai(&ai_out, &scan_config(&ui));
                                    st.ai_key = Some(ai_key(&ui));
                                    st.ai_changes.len() + st.others.iter().map(|o| o.ai_changes.len()).sum::<usize>()
                                };
                                let has_rewrite = state.borrow().ai_scene.as_ref().map(|s| !s.improved_prompt.is_empty()).unwrap_or(false);
                                if has_rewrite {
                                    if ui.get_after_view() == 3 || ui.get_after_view() == 4 {
                                        ui.set_after_view(4);
                                    } else {
                                        ui.set_after_view(1);
                                    }
                                }
                                render(&ui, state);
                                let msg = match (has_rewrite, found) {
                                    (true, _) => {
                                        let which = if ui.get_after_view() == 4 { "AI Section Blocks" } else { "AI Improved prompt" };
                                        format!("AI scan done — showing the {which}; {found} individual suggestion(s) in the cards • {where_ran}")
                                    }
                                    (false, 0) => format!("AI scan done — nothing beyond the rule cards • {where_ran}"),
                                    (false, n) => format!("AI scan done — {n} AI card(s) added; click to apply • {where_ran}"),
                                };
                                let msg = if restored_tags > 0 {
                                    format!("{msg} • {restored_tags} tag(s) from the image read the scan left out were put back")
                                } else {
                                    msg
                                };
                                ui.set_status_message(msg.into());
                            }
                            Err(e) => {
                                let msg = format!("{e:#}");
                                if msg.starts_with("Couldn't reach Ollama") {
                                    ui.set_ollama_connected(false); // the retry timer reconnects
                                    ui.set_ollama_status("Lost Ollama — retrying every 5 s...".into());
                                }
                                ui.set_status_message(format!("AI scan failed: {msg}").into());
                            }
                        }
                    });
                });
            });
        });
    }
}
