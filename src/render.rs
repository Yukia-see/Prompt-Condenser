//! Rule analysis and rendering: cards, After / negative / structure views, stats, LoRA suggestions.

use super::*;

pub(crate) fn scan_config(ui: &AppWindow) -> ScanConfig {
    ScanConfig {
        checkpoint: CheckpointProfile::from_index(ui.get_checkpoint_index()),
        target_view: TargetView::from_index(ui.get_target_view_index()),
        pinned_emphasis_raw: ui.get_pinned_emphasis().to_string(),
        emphasis_weight: (ui.get_emphasis_weight() * 100.0).round() / 100.0,
        cleanup_enabled: ui.get_cleanup_enabled(),
        resolve_conflicts_enabled: ui.get_resolve_conflicts_enabled(),
        clamp_weights_enabled: ui.get_clamp_weights_enabled(),
        fit_suggestions: ui.get_fit_suggestions(),
        exaggerated_proportions: ui.get_exaggerated_proportions(),
        camera_angle: ui.get_angle_index().max(0) as usize,
        pose: ui.get_pose_index().max(0) as usize,
        bust: ui.get_bust_index().max(0) as usize,
        frame: ui.get_frame_index().max(0) as usize,
        dynamic: ui.get_dynamic_enabled(),
        lora_tags_raw: lora_lines(&ui.get_lora_tags()), arranged: ui.get_arranged_enabled(), natural: ui.get_natural_enabled(), two_char: ui.get_app_tab() >= 1, char2_prompt_raw: ui.get_char2_prompt().to_string(), char2_lora_tags_raw: lora_lines(&ui.get_char2_lora_tags()), char_break: ui.get_char_break_enabled(), scene_prompt_raw: ui.get_scene_prompt().to_string(), more_chars: more_chars(ui),
        detail: crate::profiles::DetailLevel::from_index(ui.get_detail_index()),
    }
}

pub(crate) fn ai_key(ui: &AppWindow) -> AiKey {
    (
        // the tab and character 2's prompt count as part of the prompt
        format!(
            "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}",
            ui.get_app_tab(),
            ui.get_raw_prompt(),
            ui.get_scene_prompt(),
            ui.get_char2_prompt(),
            ui.get_char3_prompt(),
            ui.get_char4_prompt()
        ),
        ui.get_checkpoint_index(),
        ui.get_target_view_index(),
        ui.get_angle_index(),
        ui.get_pose_index(),
        ui.get_bust_index(),
        ui.get_frame_index(),
    )
}

/// Re-runs the rule engine. Resets rule-card choices; AI cards survive unless the
/// prompt, checkpoint or view they were made for has changed.
pub(crate) fn analyze(ui: &AppWindow, state: &State) {
    let config = scan_config(ui);
    let prompts = char_prompts(ui, &config);
    let c1 = with_scene(&prompts[0], &config);
    let mut result = analyzer::analyze_with_rules(&parser::parse_prompt(&c1), &config);
    let mut others = Vec::new();
    if config.two_char {
        // Every other character on its own: its LoRA box, and the scene's framing from
        // character 1; the angle / pose pickers belong to character 1's box.
        for (i, prompt) in prompts.iter().enumerate().skip(1) {
            let mut ci = config.clone();
            ci.lora_tags_raw = config.lora_of(i).to_string();
            ci.camera_angle = 0;
            ci.pose = 0;
            if ci.target_view == TargetView::None {
                ci.target_view = result.active_framing_detected;
            }
            let mut r = analyzer::analyze_with_rules(&parser::parse_prompt(prompt), &ci);
            label_two_char_cards(&mut r, i + 1);
            for c in &mut r.changes {
                c.id += two::card_base(i);
            }
            others.push(CharCards { base_tokens: r.processed_tokens, changes: r.changes, ai_changes: Vec::new() });
        }
        // an interaction typed into several boxes goes once into the shared header (two::combine); say so on
        // the later copies, which are merged (Keep Original puts a copy back)
        let mut boxes = vec![c1.clone()];
        boxes.extend(prompts.iter().skip(1).cloned());
        for (k, (i, raw, first)) in two::repeated_interactions(&boxes).into_iter().enumerate() {
            let Some(other) = others.get_mut(i - 1) else { continue };
            let norm = parser::parse_prompt(&raw).first().map(|t| t.normalized.clone()).unwrap_or_default();
            other.base_tokens.retain(|t| t.normalized != norm);
            other.changes.push(PromptChange {
                id: two::card_base(i) + 900 + k,
                original: raw.clone(),
                action: ChangeAction::MergedDuplicate,
                reason: format!(
                    "Character {}: '{raw}' is also in Character {}'s box. Interaction tags go once into the shared part \
                     of the prompt, so this copy is merged.",
                    i + 1,
                    first + 1
                ),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: 0,
            });
        }
        label_two_char_cards(&mut result, 1);
    }
    let key = ai_key(ui);
    {
        let mut st = state.borrow_mut();
        // the user's card choices survive edits; a new checkpoint or an emptied prompt drops them
        let checkpoint = ui.get_checkpoint_index();
        if st.card_choices_checkpoint != checkpoint || prompts.iter().all(|p| p.trim().is_empty()) {
            st.card_choices.clear();
            st.card_choices_checkpoint = checkpoint;
        }
        apply_card_choices(&mut result.changes, 0, &st.card_choices);
        for (i, o) in others.iter_mut().enumerate() {
            apply_card_choices(&mut o.changes, i + 1, &st.card_choices);
        }
        st.base_tokens = result.processed_tokens;
        st.changes = result.changes;
        let keep_ai = st.ai_key.as_ref() == Some(&key);
        for (i, o) in others.iter_mut().enumerate() {
            if keep_ai {
                o.ai_changes = st.others.get(i).map(|x| x.ai_changes.clone()).unwrap_or_default();
            }
        }
        st.others = others;
        if !keep_ai {
            st.ai_changes.clear();
            st.ai_scene = None;
            st.ai_key = None;
        }
        // Check Result / Ideas cards are about this prompt: a new prompt (or checkpoint, view…) drops them
        if st.extra_key.as_ref() != Some(&key) {
            st.extra_cards.clear();
            st.check_neg.clear();
            st.idea_chip_defaults.clear();
            st.extra_key = None;
        }
    }
    render(ui, state);
    update_lora_suggestions(ui, state);
    show_lora_weights(ui);
    history::touch(ui, state);
}

/// A copied text in the format chosen next to Copy (`condenser::format_prompt`), plus the end of the
/// status message (with a warning when the format can't express part of it).
pub(crate) fn export_text(ui: &AppWindow, text: &str) -> (String, String) {
    let format = condenser::ExportFormat::from_index(ui.get_export_index());
    let note = match condenser::export_warning(text, format) {
        Some(w) => format!(" ({format:?}). {w}"),
        None if format == condenser::ExportFormat::ComfyUi => ".".to_string(),
        None => format!(" ({format:?})."),
    };
    (condenser::format_prompt(text, format), note)
}

/// A rule card's identity across re-analyses: (character index, card kind, the tag it is about).
pub(crate) type CardKey = (usize, String, String);

pub(crate) fn card_key(character: usize, ch: &PromptChange) -> CardKey {
    // a card without an original tag (a Recommendation) is told apart by its reason, which names the rule;
    // its suggestion lists the tags still missing, so it changes as the user types
    // (only its first line: a body-type card lists fit tags below it that change as clothes are added)
    let discriminator = if ch.original.trim().is_empty() {
        ch.reason.lines().next().unwrap_or("").trim().to_lowercase()
    } else {
        ch.original.trim().to_lowercase()
    };
    (character, format!("{:?}", ch.action), discriminator)
}

/// Puts the user's earlier choices back on the matching new rule cards.
pub(crate) fn apply_card_choices(changes: &mut [PromptChange], character: usize, choices: &std::collections::HashMap<CardKey, u8>) {
    for ch in changes.iter_mut() {
        if let Some(mode) = choices.get(&card_key(character, ch)) {
            ch.mode = *mode;
        }
    }
}

/// Character `i`'s LoRA box as typed (several lines).
pub(crate) fn lora_box(ui: &AppWindow, i: usize) -> String {
    match i {
        0 => ui.get_lora_tags(),
        1 => ui.get_char2_lora_tags(),
        2 => ui.get_char3_lora_tags(),
        _ => ui.get_char4_lora_tags(),
    }
    .to_string()
}

/// The "+ tag" chips under each LoRA box: when the LoRA folder or a box's `<lora:…>` names change,
/// the files' training tags are read off the UI thread (`lora_meta`, cached per file); tags already
/// in the box are hidden.
pub(crate) fn update_lora_suggestions(ui: &AppWindow, state: &State) {
    let folder = lora_meta::effective_folder(&ui.get_lora_folder(), &ui.get_comfyui_folder());
    let boxes: Vec<String> = (0..4).map(|i| lora_box(ui, i)).collect();
    let names: Vec<String> = boxes.iter().map(|b| lora_meta::lora_names(b).join("\u{2}")).collect();
    let key = format!("{folder}\u{1}{}", names.join("\u{1}"));
    {
        let mut st = state.borrow_mut();
        if st.lora_suggest_key != key {
            st.lora_suggest_key = key.clone();
            st.lora_suggest = Default::default();
            if !folder.is_empty() && names.iter().any(|n| !n.is_empty()) {
                let weak = ui.as_weak();
                std::thread::spawn(move || {
                    let dir = std::path::PathBuf::from(&folder);
                    let found: Vec<Vec<String>> = boxes.iter().map(|b| lora_meta::suggestions_for_box(&dir, b)).collect();
                    let _ = weak.upgrade_in_event_loop(move |ui| {
                        with_state(|state| {
                            {
                                let mut st = state.borrow_mut();
                                if st.lora_suggest_key != key {
                                    return; // the boxes changed meanwhile
                                }
                                for (i, f) in found.into_iter().enumerate().take(4) {
                                    st.lora_suggest[i] = f;
                                }
                            }
                            show_lora_suggestions(&ui, &state.borrow());
                        })
                    });
                });
            }
        }
    }
    show_lora_suggestions(ui, &state.borrow());
}

pub(crate) fn show_lora_suggestions(ui: &AppWindow, st: &AppState) {
    let chips = |i: usize| {
        let tags = lora_meta::not_in_box(&st.lora_suggest[i], &lora_box(ui, i));
        ModelRc::new(VecModel::from(tags.into_iter().map(SharedString::from).collect::<Vec<_>>()))
    };
    ui.set_lora_suggest_c1(chips(0));
    ui.set_lora_suggest_c2(chips(1));
    ui.set_lora_suggest_c3(chips(2));
    ui.set_lora_suggest_c4(chips(3));
}

/// The − / + weight chips under each LoRA box (`lora_meta::loras_with_weights`); a model is only replaced
/// when its chips changed, so hovering a chip isn't reset by unrelated edits.
pub(crate) fn show_lora_weights(ui: &AppWindow) {
    for i in 0..4 {
        let chips: Vec<LoraWeight> = lora_meta::loras_with_weights(&lora_box(ui, i))
            .into_iter()
            .map(|(name, w)| {
                let short = name.rsplit(['/', '\\']).next().unwrap_or(&name);
                let short = short.strip_suffix(".safetensors").unwrap_or(short);
                LoraWeight { name: short.into(), weight: lora_meta::format_weight(w).into() }
            })
            .collect();
        let current = match i {
            0 => ui.get_lora_weights_c1(),
            1 => ui.get_lora_weights_c2(),
            2 => ui.get_lora_weights_c3(),
            _ => ui.get_lora_weights_c4(),
        };
        if current.iter().eq(chips.iter().cloned()) {
            continue;
        }
        let model = ModelRc::new(VecModel::from(chips));
        match i {
            0 => ui.set_lora_weights_c1(model),
            1 => ui.set_lora_weights_c2(model),
            2 => ui.set_lora_weights_c3(model),
            _ => ui.set_lora_weights_c4(model),
        }
    }
}

/// The prompts to work on, one per character: Before alone on the 1 Character tab, or
/// Before plus the other characters' boxes (a pasted "… BREAK …" prompt is split when
/// they're empty).
pub(crate) fn char_prompts(ui: &AppWindow, config: &ScanConfig) -> Vec<String> {
    let raw = ui.get_raw_prompt().to_string();
    if config.two_char {
        let others: Vec<String> = (1..config.char_count()).map(|i| config.prompt_of(i).to_string()).collect();
        two::prompts(&raw, &others)
    } else {
        vec![raw]
    }
}

/// Character 1's prompt with the Scene box after it (the rule engine sees the scene's
/// framing; `two::combine` moves the scene's tags to the shared part).
pub(crate) fn with_scene(c1: &str, config: &ScanConfig) -> String {
    let scene = config.scene_prompt_raw.trim().trim_matches(',').trim();
    if scene.is_empty() {
        return c1.to_string();
    }
    let c1 = c1.trim().trim_end_matches(',').trim();
    if c1.is_empty() { scene.to_string() } else { format!("{c1}, {scene}") }
}

/// Two characters: "solo" goes (with a card), and every card says whose it is.
pub(crate) fn label_two_char_cards(r: &mut analyzer::RuleAnalysisResult, n: usize) {
    for c in &mut r.changes {
        c.reason = format!("Character {n}: {}", c.reason);
    }
    if r.processed_tokens.iter().any(|t| t.normalized == "solo") {
        r.processed_tokens.retain(|t| t.normalized != "solo");
        let id = r.changes.iter().map(|c| c.id + 1).max().unwrap_or(0);
        r.changes.push(PromptChange {
            id,
            original: "solo".into(),
            action: ChangeAction::PositioningConflict,
            reason: format!("Character {n}: 'solo' removed because several characters are in the scene."),
            suggested_alternative: String::new(),
            framing_alternative: String::new(),
            mode: 0,
        });
    }
}

pub(crate) fn clear_ai_cards(state: &State) {
    let mut st = state.borrow_mut();
    st.ai_changes.clear();
    for o in &mut st.others {
        o.ai_changes.clear();
    }
    st.ai_scene = None;
    st.ai_key = None;
}

/// Rebuilds the After prompt, stats and cards from the current state.
pub(crate) fn render(ui: &AppWindow, state: &State) {
    // a 🎲 Roll preview lasts until the next render (an edit, a view change)
    ui.set_after_rolled(false);
    let config = scan_config(ui);
    let two_char = config.two_char;
    let mut prompts = char_prompts(ui, &config);
    prompts[0] = with_scene(&prompts[0], &config);
    let c1_raw = prompts[0].clone();
    // what the AI views are compared against: every character on the multi-character tabs
    let raw = prompts.join(", ");
    let st = state.borrow();
    // what the cards show (every chip), and what the condenser applies (only the chips left on)
    let shown1: Vec<PromptChange> = st.changes.iter().chain(&st.ai_changes).chain(&st.extra_cards).cloned().collect();
    let shown_others: Vec<Vec<PromptChange>> =
        st.others.iter().map(|o| o.changes.iter().chain(&o.ai_changes).cloned().collect()).collect();
    let changes1 = with_chips(&shown1, 0, &st.chip_off);
    let other_changes: Vec<Vec<PromptChange>> =
        shown_others.iter().enumerate().map(|(i, c)| with_chips(c, i + 1, &st.chip_off)).collect();
    let all_changes: Vec<PromptChange> = changes1.iter().chain(other_changes.iter().flatten()).cloned().collect();
    let check_negative: String = st
        .extra_cards
        .iter()
        .filter(|c| c.mode == 0)
        .filter_map(|c| st.check_neg.get(&c.id).cloned())
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = condenser::build_condensed_output(&c1_raw, &st.base_tokens, &changes1, &config);
    // characters 2..N, each condensed on its own
    let other_outs: Vec<String> = if two_char {
        (1..prompts.len())
            .map(|i| {
                let o = st.others.get(i - 1).cloned().unwrap_or_default();
                let ch = other_changes.get(i - 1).cloned().unwrap_or_default();
                condenser::build_condensed_output(&prompts[i], &o.base_tokens, &ch, &config).final_prompt
            })
            .collect()
    } else {
        Vec::new()
    };
    let stats_of = |i: usize| {
        prompts.get(i).filter(|_| two_char).map(|p| {
            let (t, c) = parser::estimate_clip_tokens(p);
            format_stats(t, c, config.checkpoint)
        })
    };
    let (t2, w2) = stats_of(1).unwrap_or_default();
    let (t3, w3) = stats_of(2).unwrap_or_default();
    let (t4, w4) = stats_of(3).unwrap_or_default();
    ui.set_char2_stats(t2.into());
    ui.set_char2_stats_warn(w2);
    ui.set_char3_stats(t3.into());
    ui.set_char3_stats_warn(w3);
    ui.set_char4_stats(t4.into());
    ui.set_char4_stats_warn(w4);
    // Every view gets the LoRA box word for word, the natural-language sentence (the
    // Deep Scan's, else a rule-based one) and the arranged layout, when those are on.
    let ai_sentence = st.ai_scene.as_ref().map(|sc| sc.natural_sentence.clone()).filter(|s| !s.is_empty());
    let finish = |t: &str, strip: bool| {
        let sentence = ai_sentence.clone().or_else(|| {
            (config.natural && config.checkpoint.supports_natural_language())
                .then(|| scene::prompt_sentence(&parser::parse_prompt(t)))
        });
        analyzer::finish_prompt(t, &config, strip, sentence.as_deref())
    };
    // Several characters: each goes in its own block (see `two::combine`); the
    // rule-based sentence describes all of them.
    let combine = |texts: &[String], strip: bool| {
        let c = two::combine(texts, &config, strip, None);
        let sentence = ai_sentence.clone().or_else(|| {
            (config.natural && config.checkpoint.supports_natural_language())
                .then(|| scene::prompt_sentence_many(&c.chars, &c.shared))
        });
        match sentence {
            Some(s) => two::combine(texts, &config, strip, Some(&s)),
            None => c,
        }
    };
    // the parts of whichever multi-character text is showing, for the live scene card
    let mut shown_parts: Option<two::Combined> = None;
    let rule_texts: Vec<String> = std::iter::once(out.final_prompt.clone()).chain(other_outs.iter().cloned()).collect();
    let rule_prompt = out.final_prompt.clone();
    out.final_prompt = if two_char {
        let c = combine(&rule_texts, true);
        let text = c.text.clone();
        shown_parts = Some(c);
        text
    } else {
        finish(&out.final_prompt, true)
    };
    (out.after_tokens, out.after_chunks) = parser::estimate_clip_tokens(&out.final_prompt);
    let with_loras = |t: String| {
        if two_char {
            // the AI's shared part stays shared (two::combine_ai); the sentence fallback as in `combine`
            let c = two::combine_ai(&t, &prompts, &config, None);
            let sentence = ai_sentence.clone().or_else(|| {
                (config.natural && config.checkpoint.supports_natural_language())
                    .then(|| scene::prompt_sentence_many(&c.chars, &c.shared))
            });
            match sentence {
                Some(s) => two::combine_ai(&t, &prompts, &config, Some(&s)),
                None => c,
            }
        } else {
            two::Combined { text: finish(&t, true), chars: Vec::new(), shared: Vec::new() }
        }
    };

    // Scene analysis card + optional AI Improved view
    let improved = st.ai_scene.as_ref().map(|sc| sc.improved_prompt.clone()).filter(|p| !p.is_empty()).map(with_loras);
    let (improved, improved_parts) = match improved {
        Some(c) => (Some(c.text.clone()), Some(c)),
        None => (None, None),
    };
    ui.set_ai_improved_available(improved.is_some());
    if (improved.is_none() && (ui.get_after_view() == 1 || ui.get_after_view() == 4)) || (!config.dynamic && ui.get_after_view() == 2) {
        if ui.get_after_view() == 4 {
            ui.set_after_view(3);
        } else {
            ui.set_after_view(0);
        }
    }
    match &st.ai_scene {
        Some(sc) => {
            ui.set_scene_summary(sc.summary.clone().into());
            ui.set_scene_details(sc.details.clone().into());
            ui.set_scene_issues(sc.issues.clone().into());
        }
        None => {
            ui.set_scene_summary("".into());
            ui.set_scene_details("".into());
            ui.set_scene_issues("".into());
        }
    }

    // Dynamic view: the AI's variation version if the last scan made one, else rule-based variations
    let dynamic_text = (ui.get_after_view() == 2).then(|| {
        let from_ai = st.ai_scene.as_ref().map(|sc| sc.dynamic_prompt.clone()).filter(|p| !p.trim().is_empty());
        match from_ai {
            Some(p) => {
                let c = with_loras(p);
                let groups = c.text.matches('{').count();
                let p = c.text.clone();
                if two_char {
                    shown_parts = Some(c);
                }
                (p, format!("AI variations: {groups} group(s). Needs the Dynamic Prompts extension / node in ComfyUI."))
            }
            None => {
                // vary the rule output, never the LoRA tags (each character on its own)
                let (p, mut kinds) = dynamic::make_dynamic(&rule_prompt);
                let p = if two_char {
                    let mut varied = vec![p];
                    for t in &rule_texts[1..] {
                        let (pk, kk) = dynamic::make_dynamic(t);
                        varied.push(pk);
                        for k in kk {
                            if !kinds.contains(&k) {
                                kinds.push(k);
                            }
                        }
                    }
                    let c = combine(&varied, true);
                    let text = c.text.clone();
                    shown_parts = Some(c);
                    text
                } else {
                    finish(&p, true)
                };
                let note = if kinds.is_empty() {
                    "Nothing to vary yet: add a pose, angle, expression or lighting tag (or run a Deep AI Scan).".to_string()
                } else {
                    format!("Varies: {}. Needs the Dynamic Prompts extension / node in ComfyUI.", kinds.join(", "))
                };
                (p, note)
            }
        }
    });
    ui.set_dynamic_note(dynamic_text.as_ref().map(|(_, n)| n.clone()).unwrap_or_default().into());
    let (text, markdown, after_tokens, after_chunks) = if let Some((dyn_text, _)) = dynamic_text {
        let (t, c) = parser::estimate_clip_tokens(&dyn_text);
        (dyn_text, String::new(), t, c)
    } else if ui.get_after_view() == 3 {
        // The Scene box was analyzed with character 1 (`with_scene`); Section Blocks re-reads the box as written, so it
        // only gets the box tags the rules kept (a removed `realistic` from a Scene / Pose reference came back here)
        let kept: std::collections::HashSet<String> =
            rule_texts.iter().flat_map(|t| parser::parse_prompt(t)).map(|t| t.normalized).collect();
        let mut block_cfg = config.clone();
        block_cfg.scene_prompt_raw = parser::parse_prompt(&config.scene_prompt_raw)
            .into_iter()
            .filter(|t| kept.contains(&t.normalized))
            .map(|t| t.raw.trim().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let blocks_text = if two_char {
            two::combine_section_blocks(&rule_texts, &block_cfg, true, ai_sentence.as_deref())
        } else {
            two::combine_section_blocks(std::slice::from_ref(&rule_prompt), &block_cfg, true, ai_sentence.as_deref())
        };
        let (t, c) = parser::estimate_clip_tokens(&blocks_text);
        (blocks_text, String::new(), t, c)
    } else if ui.get_after_view() == 4 {
        let ai_raw = st.ai_scene.as_ref().map(|sc| sc.improved_prompt.clone()).unwrap_or_default();
        let blocks_text = if two_char {
            two::combine_section_blocks_ai(&ai_raw, &prompts, &config, ai_sentence.as_deref())
        } else {
            two::combine_section_blocks(&[ai_raw], &config, true, ai_sentence.as_deref())
        };
        let (t, c) = parser::estimate_clip_tokens(&blocks_text);
        (blocks_text, String::new(), t, c)
    } else { match improved.filter(|_| ui.get_after_view() == 1) {
        Some(ai_text) => {
            if two_char {
                shown_parts = improved_parts;
            }
            let dropped = condenser::dropped_tags(&raw, &ai_text, config.checkpoint);
            ui.set_ai_dropped(dropped.join(", ").into());
            let (md, _) = condenser::highlight_against_original(&raw, &ai_text);
            let (t, c) = parser::estimate_clip_tokens(&ai_text);
            (ai_text, md, t, c)
        }
        None => (out.final_prompt.clone(), out.highlighted_markdown.clone(), out.after_tokens, out.after_chunks),
    } };
    if ui.get_after_view() != 1 {
        ui.set_ai_dropped("".into());
    }
    // Anima Ref Adapter mode: the reference image supplies the look, so the prompt keeps only the scene
    let text = if ui.get_ref_adapter_mode() { scene::filter_for_ref_adapter(&text) } else { text };
    render_negative(ui, &config, &text, st.ai_scene.as_ref().filter(|_| ui.get_after_view() == 1 || ui.get_after_view() == 4), &check_negative);
    match shown_parts.as_ref().filter(|_| two_char) {
        Some(c) => set_live_scene(ui, scene::describe_many(&c.chars, &c.shared)),
        None => render_live_scene(ui, &text),
    }
    ui.set_condensed_prompt(text.clone().into());
    // The After box is plain selectable text, so the changed tags are listed under it.
    ui.set_substituted_list(condenser::marked_tags(&markdown, condenser::Highlight::Substituted).join(", ").into());
    ui.set_added_list(condenser::marked_tags(&markdown, condenser::Highlight::Added).join(", ").into());
    ui.set_structure_rows(ModelRc::new(VecModel::from(structure_rows(&out.category_summary))));
    ui.set_category_breakdown(out.category_summary.into());
    let (before_stats, before_warn) = format_stats(out.before_tokens, out.before_chunks, config.checkpoint);
    ui.set_before_stats(before_stats.into());
    ui.set_before_stats_warn(before_warn);
    let (after_stats, after_warn) = format_stats(after_tokens, after_chunks, config.checkpoint);
    ui.set_after_stats(after_stats.into());
    ui.set_after_stats_warn(after_warn);

    ui.set_issue_summary(issue_summary(&all_changes).into());
    push_vrp_preview(ui, &st, if two_char { prompts.len() } else { 0 });

    // the "Send to ComfyUI" bridge: the nodes read this when their live_stream input is on
    let (vrp_state, regions_source) = if two_char { vrp::vrp_json(ui, &st, prompts.len()) } else { (String::new(), "") };
    live_bridge::publish(live_bridge::LivePrompt {
        timestamp: live_bridge::now_secs(),
        positive: text.clone(),
        negative: ui.get_negative_text().to_string(),
        characters: shown_parts
            .as_ref()
            .filter(|_| two_char)
            .map(|c| {
                c.chars
                    .iter()
                    .map(|toks| toks.iter().map(|t| t.format_for_output(true)).collect::<Vec<_>>().join(", "))
                    .collect()
            })
            .unwrap_or_default(),
        view: format!("{:?}", config.target_view),
        checkpoint: format!("{:?}", config.checkpoint),
        vrp_state,
        regions_source: regions_source.to_string(),
        changes: all_changes
            .iter()
            .map(|c| live_bridge::LiveChange {
                original: c.original.clone(),
                suggestion: c.suggested_alternative.clone(),
                reason: c.reason.clone(),
            })
            .collect(),
        tokens_before: out.before_tokens,
        tokens_after: after_tokens,
    });

    let shown: Vec<(usize, &PromptChange)> = shown1
        .iter()
        .map(|c| (0, c))
        .chain(shown_others.iter().enumerate().flat_map(|(i, cs)| cs.iter().map(move |c| (i + 1, c))))
        .collect();
    let (cards, counts) = card_list(&shown, &st.chip_off, st.fixes_expanded, ui.get_card_filter(), two_char);
    ui.set_fixes_expanded(st.fixes_expanded);
    ui.set_card_filter_counts(ModelRc::new(VecModel::from(counts)));
    let model = &st.cards_model;
    if model.row_count() == cards.len() {
        for (i, card) in cards.into_iter().enumerate() {
            if model.row_data(i).as_ref() != Some(&card) {
                model.set_row_data(i, card);
            }
        }
    } else {
        model.set_vec(cards);
    }
}

/// The Organized Structure text ("[Subject] 1girl, solo" per line) as colored rows.
pub(crate) fn structure_rows(summary: &str) -> Vec<StructureRow> {
    use parser::TokenCategory as C;
    let order = [
        C::QualityMeta,
        C::SubjectCount,
        C::PinnedEmphasis,
        C::FramingPose,
        C::CharacterBody,
        C::Attire,
        C::NaturalLanguageClause,
        C::EnvironmentLighting,
        C::LoraOrSpecial,
    ];
    summary
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix('[')?;
            let (label, items) = rest.split_once("] ")?;
            let kind = order.iter().position(|c| c.label() == label).unwrap_or(8) as i32;
            Some(StructureRow { label: label.to_uppercase().into(), items: items.into(), kind })
        })
        .collect()
}

/// Fills the Recommended Negative panel for the prompt currently shown in After.
/// On the AI Improved view it uses the AI's tailored negative (new entries in green).
pub(crate) fn render_negative(ui: &AppWindow, config: &ScanConfig, positive: &str, ai: Option<&SceneResult>, checked: &str) {
    if config.checkpoint.default_negative().is_empty() {
        ui.set_negative_text("".into());
        ui.set_negative_tokens("".into());
        ui.set_negative_note(
            format!("{} ignores negative prompts (it runs at CFG 1), so leave it empty.", config.checkpoint.display_name())
                .into(),
        );
        return;
    }
    let base = condenser::recommended_negative(config, positive);
    let ai_negative = ai
        .map(|sc| condenser::merge_negative(&format!("{}, {}", sc.recommended_negative, base), positive))
        .filter(|_| ai.map(|sc| !sc.recommended_negative.trim().is_empty()).unwrap_or(false));
    let (text, markdown, note) = match ai_negative {
        Some(neg) => {
            let (md, added) = condenser::highlight_against_original(&base, &neg);
            let note = if added > 0 {
                format!("Tailored by AI to the improved prompt: green = {added} AI addition(s) on top of the checkpoint defaults.")
            } else {
                "Tailored by AI to the improved prompt.".to_string()
            };
            (neg, md, note)
        }
        None => {
            let md = parser::parse_prompt(&base)
                .iter()
                .map(|t| condenser::highlight_md(&t.raw, condenser::Highlight::None))
                .collect::<Vec<_>>()
                .join(", ");
            let view = if config.target_view == TargetView::None {
                String::new()
            } else {
                format!(" + {} framing", config.target_view.display_name())
            };
            let tail = if ai.is_some() {
                "The AI didn't return a negative this time, so these are the defaults. Run the scan again to retry."
            } else {
                "Switch to AI Improved after a Deep AI Scan for a negative tailored to the scene."
            };
            let note = format!(
                "{} defaults{view}. Tags your prompt asks for are left out. {tail}",
                config.checkpoint.display_name()
            );
            (base, md, note)
        }
    };
    let _ = markdown; // the negative box is plain selectable text
    // "Check result": what the tagger saw in the generated picture that the prompt contradicts
    let (text, note) = if checked.trim().is_empty() {
        (text, note)
    } else {
        (condenser::merge_negative(&format!("{text}, {checked}"), positive), format!("{note} Check result added: {checked}."))
    };
    ui.set_negative_tokens(format!("~{} tokens", parser::estimate_clip_tokens(&text).0).into());
    ui.set_negative_text(text.into());
    ui.set_negative_note(note.into());
}

/// The status pill's verdict on the cards: "✓ Clean", "N conflicts" (framing / pose / anatomy / size /
/// style conflicts), else "N fixes" (bloat, duplicates, weights, choices, AI cards).
pub(crate) fn issue_summary(changes: &[PromptChange]) -> String {
    let conflicts = changes
        .iter()
        .filter(|c| {
            matches!(
                c.action,
                ChangeAction::FramingSafeSwap
                    | ChangeAction::PositioningConflict
                    | ChangeAction::PoseConflict
                    | ChangeAction::AnatomyConflict
                    | ChangeAction::SizeConflict
                    | ChangeAction::StyleConflict
            )
        })
        .count();
    match (conflicts, changes.len()) {
        (0, 0) => "✓ Clean".to_string(),
        (1, _) => "1 conflict".to_string(),
        (n, _) if n > 1 => format!("{n} conflicts"),
        (_, 1) => "1 fix".to_string(),
        (_, n) => format!("{n} fixes"),
    }
}

/// The mini region preview on the 2-4 Characters tabs: the boxes Copy VRP regions would write
/// (`vrp_current`), on a canvas shaped like the picture. `n` = 0 clears it.
pub(crate) fn push_vrp_preview(ui: &AppWindow, st: &AppState, n: usize) {
    let (rows, aspect) = if n >= 2 {
        let (regions, size) = vrp_current(st, n);
        let (w, h) = size.unwrap_or((1216, 832));
        let rows: Vec<VrpRegion> =
            regions.iter().map(|r| VrpRegion { x: r[0] as f32, y: r[1] as f32, w: r[2] as f32, h: r[3] as f32 }).collect();
        (rows, w as f32 / h.max(1) as f32)
    } else {
        (Vec::new(), 1.46)
    };
    let current = ui.get_vrp_preview();
    if current.row_count() != rows.len() || rows.iter().enumerate().any(|(i, r)| current.row_data(i).as_ref() != Some(r)) {
        ui.set_vrp_preview(ModelRc::new(VecModel::from(rows)));
    }
    ui.set_vrp_preview_aspect(aspect);
}

/// Fills the live scene card from the prompt currently shown in After.
pub(crate) fn render_live_scene(ui: &AppWindow, prompt: &str) {
    set_live_scene(ui, scene::describe(&parser::parse_prompt(prompt)));
}

pub(crate) fn set_live_scene(ui: &AppWindow, d: scene::SceneDescription) {
    let details = d.details.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join("\n");
    ui.set_live_headline(d.headline.into());
    ui.set_live_details(details.into());
    ui.set_live_missing(d.missing.join(", ").into());
}

/// The token readout above a prompt box, and whether to tint it (more than one CLIP chunk).
/// CLIP checkpoints: "~N / 75 tokens · chunk 1, est." (the limit grows by 75 per chunk). Anima and
/// Flux don't read prompts in 75-token chunks: just "~N tokens".
pub(crate) fn format_stats(tokens: usize, chunks: usize, checkpoint: CheckpointProfile) -> (String, bool) {
    if !checkpoint.uses_clip_chunks() {
        return (format!("~{tokens} tokens"), false);
    }
    let chunks = chunks.max(1);
    (format!("~{tokens} / {} tokens · chunk {chunks}, est.", 75 * chunks), chunks > 1)
}

pub(crate) fn to_card(ch: &PromptChange, character_label: String) -> ChangeCardItem {
    let badge_kind = match ch.action {
        ChangeAction::FramingSafeSwap
        | ChangeAction::PositioningConflict
        | ChangeAction::PoseConflict
        | ChangeAction::AnatomyConflict => 0,
        ChangeAction::AiRewrite => 3,
        ChangeAction::KeptEmphasized
        | ChangeAction::AngleChoice
        | ChangeAction::PoseChoice
        | ChangeAction::Recommendation
        | ChangeAction::ProportionChoice => 1,
        ChangeAction::RemovedBloat
        | ChangeAction::MergedDuplicate
        | ChangeAction::WeightClamped
        | ChangeAction::SizeConflict
        | ChangeAction::StyleConflict
        | ChangeAction::ImpliedTag => 2,
        ChangeAction::Idea => 3,
        ChangeAction::CheckResult => 0,
    };
    ChangeCardItem {
        id: ch.id as i32,
        badge: ch.action.badge_text().into(),
        badge_kind,
        character_label: character_label.into(),
        original: ch.original.clone().into(),
        reason: ch.reason.clone().into(),
        suggested_alternative: ch.suggested_alternative.clone().into(),
        framing_alternative: ch.framing_alternative.clone().into(),
        has_framing_alt: !ch.framing_alternative.is_empty(),
        mode: ch.mode as i32,
        chips: ModelRc::default(),
        summary_count: 0,
    }
}

/// Cards whose suggestion is a list of tags to add, picked one by one: AI ideas, suggestion (FIT & DYNAMICS)
/// cards and AI "add" cards with two or more tags.
pub(crate) fn has_chips(ch: &PromptChange) -> bool {
    let adds = match ch.action {
        ChangeAction::Idea | ChangeAction::Recommendation => true,
        ChangeAction::AiRewrite => ch.original.trim().is_empty(),
        _ => false,
    };
    adds && ch.suggested_alternative.split(',').filter(|t| !t.trim().is_empty()).count() >= 2
}

fn chip_tags(ch: &PromptChange) -> Vec<String> {
    ch.suggested_alternative.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect()
}

/// The cards as the condenser should apply them: a chip card adds only its chips left on (none on = not
/// applied).
pub(crate) fn with_chips(
    changes: &[PromptChange],
    character: usize,
    chip_off: &std::collections::HashMap<CardKey, std::collections::HashSet<String>>,
) -> Vec<PromptChange> {
    changes
        .iter()
        .map(|c| {
            let mut c = c.clone();
            if has_chips(&c) {
                if let Some(off) = chip_off.get(&card_key(character, &c)) {
                    let on: Vec<String> = chip_tags(&c).into_iter().filter(|t| !off.contains(t)).collect();
                    if on.is_empty() {
                        c.mode = 2;
                    }
                    c.suggested_alternative = on.join(", ");
                }
            }
            c
        })
        .collect()
}

/// Which filter tab a card belongs to: 1 Conflicts, 2 Fixes (fixes and the user's picks), 3 Suggestions
/// (FIT & DYNAMICS, Check Result), 4 AI. The strip shows them in that order.
pub(crate) fn card_group(ch: &PromptChange) -> i32 {
    match ch.action {
        ChangeAction::FramingSafeSwap
        | ChangeAction::PositioningConflict
        | ChangeAction::PoseConflict
        | ChangeAction::AnatomyConflict
        | ChangeAction::SizeConflict
        | ChangeAction::StyleConflict => 1,
        ChangeAction::RemovedBloat
        | ChangeAction::MergedDuplicate
        | ChangeAction::WeightClamped
        | ChangeAction::ImpliedTag
        | ChangeAction::KeptEmphasized
        | ChangeAction::AngleChoice
        | ChangeAction::PoseChoice
        | ChangeAction::ProportionChoice => 2,
        ChangeAction::Recommendation | ChangeAction::CheckResult => 3,
        ChangeAction::AiRewrite | ChangeAction::Idea => 4,
    }
}

/// Cards Apply all / Skip all may change: not the user's own picks (Angle, Pose, Bust / Frame dropdowns, Keep &
/// Emphasize), which they made on purpose.
pub(crate) fn batchable(ch: &PromptChange) -> bool {
    !matches!(
        ch.action,
        ChangeAction::AngleChoice | ChangeAction::PoseChoice | ChangeAction::ProportionChoice | ChangeAction::KeptEmphasized
    )
}

/// Automatic fixes that fold into one "N small fixes" card while they're applied.
fn is_small_fix(ch: &PromptChange) -> bool {
    ch.mode == 0
        && matches!(ch.action, ChangeAction::RemovedBloat | ChangeAction::MergedDuplicate | ChangeAction::WeightClamped | ChangeAction::ImpliedTag)
}

/// The id of the folded "small fixes" card (no real card has it).
pub(crate) const SUMMARY_CARD_ID: i32 = -1;

/// The strip's cards: sorted by group (conflicts first), small fixes folded into one card unless opened (or
/// the Fixes tab is on), filtered by the tab, chips filled in. Also the tab counts (All, Conflicts, Fixes,
/// Suggestions, AI).
pub(crate) fn card_list(
    shown: &[(usize, &PromptChange)],
    chip_off: &std::collections::HashMap<CardKey, std::collections::HashSet<String>>,
    fixes_expanded: bool,
    filter: i32,
    two_char: bool,
) -> (Vec<ChangeCardItem>, Vec<i32>) {
    let mut counts = vec![shown.len() as i32, 0, 0, 0, 0];
    for (_, c) in shown {
        counts[card_group(c) as usize] += 1;
    }
    let mut sorted: Vec<&(usize, &PromptChange)> = shown.iter().filter(|(_, c)| filter == 0 || card_group(c) == filter).collect();
    sorted.sort_by_key(|(_, c)| card_group(c));
    let fold = !fixes_expanded && filter != 2 && sorted.iter().filter(|(_, c)| is_small_fix(c)).count() >= 2;
    let mut cards = Vec::new();
    let mut summary_at: Option<usize> = None;
    let mut folded: Vec<String> = Vec::new();
    for (character, c) in sorted {
        if fold && is_small_fix(c) {
            summary_at.get_or_insert(cards.len());
            let what = if c.original.trim().is_empty() { c.reason.lines().next().unwrap_or("").to_string() } else { format!("'{}'", c.original.trim()) };
            folded.push(format!("• {}: {what}", c.action.badge_text().to_lowercase()));
            continue;
        }
        // 💡 Ideas and Check Result cards are about the whole prompt, not character 1
        let whole_prompt = matches!(c.action, ChangeAction::Idea | ChangeAction::CheckResult);
        let char_label = if two_char && !whole_prompt { format!("Char {}", character + 1) } else { String::new() };
        let mut card = to_card(c, char_label);
        if has_chips(c) {
            let off = chip_off.get(&card_key(*character, c));
            let chips: Vec<CardChip> =
                chip_tags(c).into_iter().map(|t| CardChip { on: !off.is_some_and(|o| o.contains(&t)), tag: t.into() }).collect();
            card.chips = ModelRc::new(VecModel::from(chips));
        }
        cards.push(card);
    }
    if let Some(at) = summary_at {
        let n = folded.len();
        cards.insert(
            at,
            ChangeCardItem {
                id: SUMMARY_CARD_ID,
                badge: "SMALL FIXES".into(),
                badge_kind: 2,
                character_label: SharedString::default(),
                original: String::new().into(),
                reason: format!("{n} small automatic fixes (already applied):\n{}", folded.join("\n")).into(),
                suggested_alternative: SharedString::default(),
                framing_alternative: SharedString::default(),
                has_framing_alt: false,
                mode: 0,
                chips: ModelRc::default(),
                summary_count: n as i32,
            },
        );
    }
    (cards, counts)
}

/// Wires rule analysis (debounced), card toggles, the After view switch and the copy buttons.
pub(crate) fn wire_cards(ui: &AppWindow, state: &State) {
    // "+ Quality": the checkpoint's missing quality tags in front of Before (an explicit action)
    {
        let weak = ui.as_weak();
        ui.on_add_quality_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let checkpoint = CheckpointProfile::from_index(ui.get_checkpoint_index());
            let (text, added) = analyzer::add_quality_tags(&ui.get_raw_prompt(), checkpoint);
            let msg = if added.is_empty() {
                format!("Before already has {}'s quality tags.", checkpoint.display_name())
            } else {
                ui.set_raw_prompt(text.into());
                ui.invoke_analyze_requested();
                format!("Added {} quality tags: {}.", checkpoint.display_name(), added.join(", "))
            };
            ui.set_status_message(msg.into());
        });
    }
    // --- Rule analysis ---
    // Debounced: typing or dragging the slider fires this on every change, so the
    // analysis runs once input pauses instead of on every keystroke.
    {
        let weak = ui.as_weak();
        let state = state.clone();
        let timer = Rc::new(slint::Timer::default());
        ui.on_analyze_requested(move || {
            let weak = weak.clone();
            let state = state.clone();
            timer.start(slint::TimerMode::SingleShot, std::time::Duration::from_millis(120), move || {
                if let Some(ui) = weak.upgrade() {
                    analyze(&ui, &state);
                }
            });
        });
    }

    // --- Card toggles ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_mode_changed(move |card_id, mode| {
            let Some(ui) = weak.upgrade() else { return };
            {
                let mut st = state.borrow_mut();
                let st = &mut *st;
                // a rule card's choice is remembered for the next analyses (AI cards have their own lifetime)
                let rule_card = std::iter::once((0usize, &st.changes))
                    .chain(st.others.iter().enumerate().map(|(i, o)| (i + 1, &o.changes)))
                    .find_map(|(i, cs)| cs.iter().find(|c| c.id as i32 == card_id).map(|c| card_key(i, c)));
                if let Some(k) = rule_card {
                    st.card_choices.insert(k, mode.clamp(0, 2) as u8);
                }
                if let Some(ch) = st
                    .changes
                    .iter_mut()
                    .chain(st.ai_changes.iter_mut())
                    .chain(st.extra_cards.iter_mut())
                    .chain(st.others.iter_mut().flat_map(|o| o.changes.iter_mut().chain(o.ai_changes.iter_mut())))
                    .find(|c| c.id as i32 == card_id)
                {
                    ch.mode = mode.clamp(0, 2) as u8;
                }
            }
            render(&ui, &state);
        });
    }

    // --- Tag chips on suggestion cards: switch one tag on / off ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_chip_toggled(move |card_id, tag| {
            let Some(ui) = weak.upgrade() else { return };
            {
                let mut st = state.borrow_mut();
                let st = &mut *st;
                let found = std::iter::once((0usize, st.changes.iter().chain(&st.ai_changes).chain(&st.extra_cards).collect::<Vec<_>>()))
                    .chain(st.others.iter().enumerate().map(|(i, o)| (i + 1, o.changes.iter().chain(&o.ai_changes).collect())))
                    .find_map(|(i, cs)| cs.into_iter().find(|c| c.id as i32 == card_id).map(|c| card_key(i, c)));
                if let Some(k) = found {
                    let off = st.chip_off.entry(k).or_default();
                    let tag = tag.to_string();
                    if !off.remove(&tag) {
                        off.insert(tag);
                    }
                }
            }
            render(&ui, &state);
        });
    }

    // --- Tag chips batch toggle (All / None) ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_chips_batch(move |card_id, all_on| {
            let Some(ui) = weak.upgrade() else { return };
            {
                let mut st = state.borrow_mut();
                let st = &mut *st;
                let found = std::iter::once((0usize, st.changes.iter().chain(&st.ai_changes).chain(&st.extra_cards).collect::<Vec<_>>()))
                    .chain(st.others.iter().enumerate().map(|(i, o)| (i + 1, o.changes.iter().chain(&o.ai_changes).collect())))
                    .find_map(|(i, cs)| cs.into_iter().find(|c| c.id as i32 == card_id).map(|c| (card_key(i, c), chip_tags(c))));
                if let Some((k, tags)) = found {
                    if all_on {
                        st.chip_off.remove(&k);
                    } else {
                        st.chip_off.insert(k, tags.into_iter().collect());
                    }
                }
            }
            render(&ui, &state);
        });
    }

    // --- The folded "N small fixes" card opens into separate cards / collapses back ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_expand_fixes(move || {
            let Some(ui) = weak.upgrade() else { return };
            state.borrow_mut().fixes_expanded = true;
            render(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_collapse_fixes(move || {
            let Some(ui) = weak.upgrade() else { return };
            state.borrow_mut().fixes_expanded = false;
            render(&ui, &state);
        });
    }

    // --- Batch card actions: 0 = Apply all in tab, 1 = Skip all in tab, 2 = Reset all ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_batch_action(move |action| {
            let Some(ui) = weak.upgrade() else { return };
            if action == 2 {
                {
                    let mut st = state.borrow_mut();
                    let st = &mut *st;
                    st.card_choices.clear();
                    // chips back to their defaults: every chip on, except the 💡 Ideas options after the first
                    st.chip_off = st.idea_chip_defaults.clone();
                    st.fixes_expanded = false;
                    // opt-in cards (AI, Ideas, Check Result) start unapplied: un-apply them, keep the cards
                    for ch in st.ai_changes.iter_mut().chain(st.extra_cards.iter_mut()).chain(st.others.iter_mut().flat_map(|o| o.ai_changes.iter_mut())) {
                        ch.mode = 2;
                    }
                }
                analyze(&ui, &state);
                ui.set_status_message("Cards reset to their defaults (AI and 💡 Ideas cards kept, unapplied).".into());
                return;
            }
            let target_mode: u8 = if action == 0 { 0 } else { 2 };
            let filter = ui.get_card_filter();
            {
                let mut st = state.borrow_mut();
                let st = &mut *st;
                for ch in st.changes.iter_mut() {
                    if (filter == 0 || card_group(ch) == filter) && batchable(ch) {
                        ch.mode = target_mode;
                        st.card_choices.insert(card_key(0, ch), target_mode);
                    }
                }
                for ch in st.ai_changes.iter_mut().chain(st.extra_cards.iter_mut()) {
                    if (filter == 0 || card_group(ch) == filter) && batchable(ch) {
                        ch.mode = target_mode;
                    }
                }
                for (i, o) in st.others.iter_mut().enumerate() {
                    let char_idx = i + 1;
                    for ch in o.changes.iter_mut() {
                        if (filter == 0 || card_group(ch) == filter) && batchable(ch) {
                            ch.mode = target_mode;
                            st.card_choices.insert(card_key(char_idx, ch), target_mode);
                        }
                    }
                    for ch in o.ai_changes.iter_mut() {
                        if (filter == 0 || card_group(ch) == filter) && batchable(ch) {
                            ch.mode = target_mode;
                        }
                    }
                }
            }
            render(&ui, &state);
        });
    }

    // --- Filter tabs above the cards ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_filter_changed(move |i| {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_card_filter(i.clamp(0, 4));
            render(&ui, &state);
        });
    }

    // --- 💡 Ideas: E4B picks real tags for the parts the prompt leaves out ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_ideas_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // the prompt as shown in After (rules applied), so ideas fit what will be generated
            let prompt = ui.get_condensed_prompt().to_string();
            let mut gaps = ideas::gaps(&prompt);
            if gaps.is_empty() {
                gaps = ideas::refinement_gaps(&prompt);
            }
            if gaps.is_empty() {
                ui.set_status_message(
                    "💡 Ideas: no additional tag menus available (make sure the Danbooru tag list is downloaded).".into(),
                );
                return;
            }
            let model = default_text_model(&ui, &state);
            let (host, mode) = (ui.get_ollama_host().to_string(), AiMode::from_index(ui.get_ai_mode_index()));
            let key = ai_key(&ui);
            ui.set_ideas_busy(true);
            ui.set_status_message(format!("💡 Ideas: asking {model} about the {}…", gaps.iter().map(|g| g.name).collect::<Vec<_>>().join(", ")).into());
            let weak = weak.clone();
            std::thread::spawn(move || {
                let started = std::time::Instant::now();
                let result = ideas::suggest(&host, &model, mode, &prompt, &gaps);
                let secs = started.elapsed().as_secs_f32();
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_ideas_busy(false);
                    with_state(|state| {
                        // the prompt changed while the model was thinking: these ideas are for the old one
                        if ai_key(&ui) != key {
                            ui.set_status_message("💡 Ideas: the prompt changed meanwhile, so the ideas were dropped. Try again.".into());
                            return;
                        }
                        match result {
                            Ok(found) if !found.is_empty() => {
                                let cards = ideas::cards(&found);
                                let n = cards.len();
                                {
                                    let mut st = state.borrow_mut();
                                    st.extra_cards.retain(|c| c.action != ChangeAction::Idea);
                                    // only the first option starts switched on
                                    for (card, (_, options)) in cards.iter().zip(&found) {
                                        let off: std::collections::HashSet<String> = options.iter().skip(1).flatten().cloned().collect();
                                        st.chip_off.insert(card_key(0, card), off.clone());
                                        st.idea_chip_defaults.insert(card_key(0, card), off);
                                    }
                                    st.extra_cards.extend(cards);
                                    st.extra_key = Some(key.clone());
                                }
                                render(&ui, state);
                                ui.set_status_message(format!("💡 Ideas: {n} card(s) in {secs:.1}s. Pick tags and press Add It; nothing changes until you do.").into());
                            }
                            Ok(_) => ui.set_status_message("💡 Ideas: the model's picks weren't on the tag menus, so there's nothing to suggest. Try again.".into()),
                            Err(e) => ui.set_status_message(format!("💡 Ideas failed: {e}").into()),
                        }
                    });
                });
            });
        });
    }

    // --- Rule-based / AI Improved toggle ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_after_view_changed(move |view| {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_after_view(view);
            render(&ui, &state);
        });
    }

    // --- LoRA weight chips: − / + rewrite that LoRA's first weight in its box ---
    {
        let weak = ui.as_weak();
        ui.on_lora_weight_step(move |c, i, delta| {
            let Some(ui) = weak.upgrade() else { return };
            let (c, i) = (c.clamp(0, 3) as usize, i.max(0) as usize);
            // the chip sends ±0.05; the step itself is `lora_meta::WEIGHT_STEP`
            let text: SharedString = lora_meta::step_weight(&lora_box(&ui, c), i, lora_meta::WEIGHT_STEP * delta.signum()).into();
            match c {
                0 => ui.set_lora_tags(text),
                1 => ui.set_char2_lora_tags(text),
                2 => ui.set_char3_lora_tags(text),
                _ => ui.set_char4_lora_tags(text),
            }
            ui.invoke_analyze_requested();
        });
    }

    // --- 🎲 Roll: one pick per {a|b} group of the Dynamic view, shown (and copied) instead of the template ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_roll_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // re-render first, so a second roll starts from the template, not the last pick
            render(&ui, &state);
            let template = ui.get_condensed_prompt().to_string();
            if !template.contains('{') {
                ui.set_status_message("Nothing to roll: the Dynamic prompt has no {a|b} groups.".into());
                return;
            }
            let seed = dynamic::random_seed();
            let rolled = dynamic::roll(&template, seed);
            let config = scan_config(&ui);
            let (t, c) = parser::estimate_clip_tokens(&rolled);
            let (stats, warn) = format_stats(t, c, config.checkpoint);
            ui.set_after_stats(stats.into());
            ui.set_after_stats_warn(warn);
            ui.set_condensed_prompt(rolled.into());
            ui.set_after_rolled(true);
            ui.set_dynamic_note(
                "Rolled preview: one pick per group, the way one image of the batch gets it. 🎲 Roll again for another; Copy copies this pick. Any edit or view change shows the template again."
                    .into(),
            );
        });
    }

    // --- Clipboard ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_copy_output_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let (text, note) = export_text(&ui, &ui.get_condensed_prompt());
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let msg = match st.clipboard.as_mut().map(|c| c.set_text(text)) {
                Some(Ok(())) => format!("Condensed prompt copied to clipboard{note}"),
                Some(Err(e)) => format!("Copy failed: {e}"),
                None => "Copy failed: no clipboard available.".to_string(),
            };
            ui.set_status_message(msg.into());
        });
    }

    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_copy_section_blocks_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let (text, is_ai) = if ui.get_after_view() == 3 || ui.get_after_view() == 4 {
                (ui.get_condensed_prompt().to_string(), ui.get_after_view() == 4)
            } else {
                let config = scan_config(&ui);
                let prompts = char_prompts(&ui, &config);
                let ai_sentence = state.borrow().ai_scene.as_ref().map(|sc| sc.natural_sentence.clone()).filter(|s| !s.is_empty());
                let two_char = ui.get_app_tab() >= 1;
                let use_ai = ui.get_after_view() == 1 && state.borrow().ai_scene.is_some();
                let ai_raw = state.borrow().ai_scene.as_ref().map(|sc| sc.improved_prompt.clone()).filter(|p| !p.trim().is_empty());
                let t = match (use_ai, ai_raw) {
                    (true, Some(ai)) => {
                        if two_char {
                            two::combine_section_blocks_ai(&ai, &prompts, &config, ai_sentence.as_deref())
                        } else {
                            two::combine_section_blocks(&[ai], &config, true, ai_sentence.as_deref())
                        }
                    }
                    _ => {
                        let mut p = prompts.clone();
                        if two_char {
                            p[0] = with_scene(&p[0], &config);
                        }
                        two::combine_section_blocks(&p, &config, true, ai_sentence.as_deref())
                    }
                };
                (t, use_ai)
            };
            let (text, note) = export_text(&ui, &text);
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let msg = match st.clipboard.as_mut().map(|c| c.set_text(text)) {
                Some(Ok(())) => format!("{}Section blocks prompt copied to clipboard{note}", if is_ai { "AI " } else { "" }),
                Some(Err(e)) => format!("Copy failed: {e}"),
                None => "Copy failed: no clipboard available.".to_string(),
            };
            ui.set_status_message(msg.into());
        });
    }

    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_copy_recommended_negative_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let (text, note) = export_text(&ui, &ui.get_negative_text());
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let ok = st.clipboard.as_mut().map(|c| c.set_text(text).is_ok()).unwrap_or(false);
            ui.set_status_message(if ok { format!("Recommended negative prompt copied{note}") } else { "Copy failed.".into() }.into());
        });
    }
}

#[cfg(test)]
mod card_key_tests {
    use super::*;

    #[test]
    fn recommendation_cards_keep_their_own_choice() {
        let config = crate::analyzer::ScanConfig {
            checkpoint: crate::profiles::CheckpointProfile::Anima,
            target_view: crate::profiles::TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            fit_suggestions: true,
            exaggerated_proportions: false,
            camera_angle: 0,
            pose: 0,
            bust: 0,
            frame: 0,
            dynamic: false,
            lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
        };
        let prompt = "1girl, large breasts, shirt, thick thighs, thighhighs";
        let mut changes = analyzer::analyze_with_rules(&parser::parse_prompt(prompt), &config).changes;
        let recs: Vec<usize> = (0..changes.len()).filter(|&i| changes[i].action == analyzer::ChangeAction::Recommendation).collect();
        assert!(recs.len() >= 2, "{changes:?}");
        let (a, b) = (recs[0], recs[1]);
        assert_ne!(card_key(0, &changes[a]), card_key(0, &changes[b]));
        // accept the first one only
        let mut choices = std::collections::HashMap::new();
        choices.insert(card_key(0, &changes[a]), 0u8);
        // the user typed one of the suggested tags: the card's suggestion shrinks, its key stays
        let mut later = analyzer::analyze_with_rules(&parser::parse_prompt(&format!("{prompt}, button gap")), &config).changes;
        apply_card_choices(&mut later, 0, &choices);
        apply_card_choices(&mut changes, 0, &choices);
        assert_eq!(changes[a].mode, 0);
        assert_eq!(changes[b].mode, 2);
        let key_a = card_key(0, &changes[a]);
        let again = later.iter().find(|c| card_key(0, c) == key_a).expect("the accepted card is still there");
        assert_eq!(again.mode, 0, "the choice survived the edit");
        assert!(later.iter().filter(|c| c.action == analyzer::ChangeAction::Recommendation && card_key(0, c) != key_a).all(|c| c.mode == 2));
    }
}
