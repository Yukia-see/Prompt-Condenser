//! Image reads: tagger, AI (vision) reads, multi-character splits, the read queue and their staleness checks.

use super::*;

/// Puts an image's prompt into the Before box, remembering what was there for Undo.
pub(crate) fn set_prompt_from_image(ui: &AppWindow, state: &State, prompt: &str) {
    let current = ui.get_raw_prompt().to_string();
    // remember the prompt from before this image only once, so flipping Tagger / AI
    // doesn't overwrite what Undo goes back to
    let already = state.borrow().prompt_before_image.is_some();
    if !already && !current.trim().is_empty() && current.trim() != prompt.trim() {
        state.borrow_mut().prompt_before_image = Some(current);
        ui.set_image_can_undo(true);
    }
    state.borrow_mut().card_choices.clear();
    ui.set_raw_prompt(prompt.into());
    analyze(ui, state);
}

/// Reads the loaded image with the method the Tagger / AI switch selects, reusing a
/// cached result when there is one.
/// Shows what the loaded image gives in the current mode. With `run`, reads it when
/// that hasn't happened yet (the Read Image button); without, only reuses what was
/// already read, so switching modes never starts a slow run by itself.
pub(crate) fn image_extract(ui: &AppWindow, state: &State, run: bool) {
    if state.borrow().image_path.is_none() {
        return;
    }
    if ui.get_app_tab() >= 1 {
        return two_image_extract(ui, state, run);
    }
    if ui.get_image_use_ai() {
        if run {
            // Explicit run/redo: clear AI cache and re-run with Vision Encoder
            state.borrow_mut().image_ai = None;
            run_ai_image(ui, state);
        } else {
            let cached = state.borrow().image_ai.clone();
            match cached {
                Some(res) => apply_ai_image(ui, state, &res, None),
                None => ui.set_image_status("Press Read Image to read it with AI.".into()),
            }
        }
    } else if run {
        state.borrow_mut().last_tags = None;
        if !tagger::is_installed() {
            ui.set_image_status("Download the Danbooru tagger once (1.2 GB) to extract tags.".into());
        } else {
            run_tagger(ui, state);
        }
    } else if state.borrow().last_tags.is_some() {
        apply_tags(ui, state, Some(0.0));
    } else if !tagger::is_installed() {
        ui.set_image_status("Download the Danbooru tagger once (1.2 GB) to extract tags.".into());
    } else {
        ui.set_image_status("Press Read Image to read it with the tagger.".into());
    }
}

/// 2 Characters tab: split the image into Character 1 | Scene | Character 2, from the
/// cache when this image was read already.
pub(crate) fn two_image_extract(ui: &AppWindow, state: &State, run: bool) {
    let use_ai = ui.get_image_use_ai();
    if run {
        if use_ai {
            state.borrow_mut().two_ai = None;
        } else {
            state.borrow_mut().two_tags = None;
        }
        run_two_image(ui, state);
        return;
    }
    let (tags, ai) = {
        let st = state.borrow();
        (st.two_tags.clone(), st.two_ai.clone())
    };
    match (use_ai, tags, ai) {
        (true, _, Some(ai)) => apply_two(ui, state, &ai, "AI split (cached)".into()),
        (false, Some(t), _) => {
            let split = twoimg::rule_split(&t, ui.get_tag_threshold());
            apply_two(ui, state, &split, two_status(&t, "Tagger split"));
        }
        _ => ui.set_image_status(
            format!("Press Read Image to split it with {}.", if use_ai { "AI" } else { "the tagger" }).into(),
        ),
    }
}

pub(crate) fn two_status(t: &twoimg::TagSplit, what: &str) -> String {
    match twoimg::not_found_note(t) {
        Some(note) => format!("{note}: {what} needs AI mode to separate them"),
        None if t.overlap > 0.25 => format!("{} characters found (some overlap) • {what}", t.wanted),
        None => format!("{} characters found • {what}", t.wanted),
    }
}

/// Detector + tagger (+ the vision model in AI mode), off the UI thread.
pub(crate) fn run_two_image(ui: &AppWindow, state: &State) {
    let Some(path) = state.borrow().image_path.clone() else { return };
    if let Err(e) = twoimg::check_models() {
        ui.set_image_status(format!("{e}").into());
        return;
    }
    let ai_on = ui.get_image_use_ai() && ui.get_ollama_connected();
    let text_model = ai_on.then(|| default_text_model(ui, state));
    let vision_encoder = ai_on.then(|| vision_model(ui, state)).flatten();
    if ui.get_image_use_ai() && text_model.is_none() {
        ui.set_image_status(
            "AI mode needs Ollama and a configured model. Using the tagger split.".into(),
        );
    }
    let (host, mode, threshold) =
        (ui.get_ollama_host().to_string(), AiMode::from_index(ui.get_ai_mode_index()), ui.get_tag_threshold());
    let has_vision = text_model.as_ref().is_some_and(|m| {
        state.borrow().vision_models.iter().any(|v| ai_engine::same_model(v, m))
    });
    let second_pass = ui.get_image_second_pass() && has_vision;
    let scene_body_shape = ui.get_scene_body_shape();
    let n = tab_chars(ui);
    // a cached reading for another number of characters doesn't count
    let cached = state.borrow().two_tags.clone().filter(|t| t.wanted == n);
    let gen = (state.borrow().image_gens, ImageSlots::Main);
    ai_engine::reset_cancel();
    ui.set_tagger_busy(true);
    ui.set_image_status(format!("Finding the {n} characters…").into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let progress = {
            let weak = weak.clone();
            move |msg: String| {
                let _ = weak.upgrade_in_event_loop(move |ui| ui.set_image_status(msg.into()));
            }
        };
        let result = (|| -> anyhow::Result<(twoimg::TagSplit, Option<anyhow::Result<twoimg::SplitPrompt>>)> {
            if !detector::is_installed() {
                let (tx, rx) = mpsc::channel::<String>();
                let p = progress.clone();
                let printer = std::thread::spawn(move || rx.into_iter().for_each(p));
                detector::download(&tx)?;
                drop(tx);
                let _ = printer.join();
            }
            let tags = match cached {
                Some(t) => t,
                None => {
                    progress("Tagging each character…".into());
                    twoimg::read(&path, n, !ai_on)?
                }
            };
            let ai_split = text_model.map(|model| {
                let status_name = match &vision_encoder {
                    Some(enc) if !ai_engine::same_model(enc, &model) => {
                        if second_pass {
                            format!("{enc} + {model} (2nd Pass Vision)")
                        } else {
                            format!("{enc} → {model}")
                        }
                    }
                    _ => model.clone(),
                };
                progress(format!("AI is checking who is who with {status_name}…"));
                ai_engine::split_image(&ai_engine::TwoImageRequest {
                    host,
                    model,
                    vision_encoder,
                    mode,
                    hints: twoimg::rule_split(&tags, threshold),
                    split: tags.clone(),
                    second_pass_vision: second_pass,
                    scene_body_shape,
                })
                .map(|(split, _)| twoimg::settle(split, &tags, threshold))
            });
            Ok((tags, ai_split))
        })();
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            if image_changed_since(&ui, gen) {
                return;
            }
            // Cancelled: no fallback tagger split, no Deep Scan
            if ai_engine::cancelled() {
                with_state(|state| state.borrow_mut().scan_after_image = false);
                ui.set_image_status("Cancelled.".into());
                return;
            }
            with_state(|state| match result {
                Ok((tags, ai_split)) => {
                    state.borrow_mut().two_tags = Some(tags.clone());
                    state.borrow_mut().vrp_regions = None;
                    let (split, how) = match ai_split {
                        Some(Ok(split)) => {
                            state.borrow_mut().two_ai = Some(split.clone());
                            (split, "AI split".to_string())
                        }
                        Some(Err(e)) => (twoimg::rule_split(&tags, ui.get_tag_threshold()), format!("AI failed ({e:#}), tagger split")),
                        None => (twoimg::rule_split(&tags, ui.get_tag_threshold()), "Tagger split".to_string()),
                    };
                    apply_two(&ui, state, &split, format!("{} • {secs:.0} s", two_status(&tags, &how)));
                    if std::mem::take(&mut state.borrow_mut().scan_after_image) {
                        state.borrow_mut().scan_image_done = true; // the scan must not read it again
                        ui.invoke_run_ai_scan_requested();
                    }
                }
                Err(e) => {
                    state.borrow_mut().scan_after_image = false;
                    ui.set_image_status(format!("Reading the image failed: {e:#}").into());
                }
            });
        });
    });
}

/// Fills Character 1 | Scene | Character 2 (remembering what was there, for Undo).
/// Remembers every character box and the Scene box once, for Undo.
pub(crate) fn remember_boxes(ui: &AppWindow, state: &State) {
    let boxes: Vec<String> = (0..4).map(|i| char_box(ui, i)).collect();
    let scene = ui.get_scene_prompt().to_string();
    let had_text = boxes.iter().any(|b| !b.trim().is_empty()) || !scene.trim().is_empty();
    let mut st = state.borrow_mut();
    if st.two_before_image.is_none() && had_text {
        st.two_before_image = Some((boxes, scene));
        ui.set_image_can_undo(true);
    }
}

pub(crate) fn apply_two(ui: &AppWindow, state: &State, split: &twoimg::SplitPrompt, status: String) {
    remember_boxes(ui, state);
    for (i, c) in split.chars.iter().enumerate().take(tab_chars(ui)) {
        set_char_box(ui, i, &identity(ui, c));
    }
    // Identity Only: the image supplies the characters, your Scene box stays
    if !ui.get_image_identity_only() {
        let current_scene = ui.get_scene_prompt().to_string();
        let other_prompts: Vec<String> = (0..tab_chars(ui)).map(|i| char_box(ui, i)).collect();
        let scene_tags: Vec<String> = split.scene.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        let cleaned_scene = sceneref::update_shared_scene_prompt(&current_scene, &scene_tags, &other_prompts);
        ui.set_scene_prompt(cleaned_scene.into());
    }
    ui.set_image_status(status.into());
    analyze(ui, state);
    sync_scene_panel(ui, state);
}

/// Char 1 Image… / Char 2 Image…: one reference image fills one character's box.
/// What the background readers need to call the tagger and the vision model.
#[derive(Clone)]
pub(crate) struct ReadCtx {
    /// The target text model (from dropdown / default tier).
    pub(crate) model: Option<String>,
    /// The vision encoder (defaults to Moondream).
    pub(crate) vision_encoder: Option<String>,
    pub(crate) host: String,
    pub(crate) mode: AiMode,
    pub(crate) threshold: f32,
    pub(crate) config: ScanConfig,
    pub(crate) second_pass: bool,
    pub(crate) scene_body_shape: bool,
    pub(crate) enable_dual_pass: bool,
    pub(crate) use_llm_synthesis: bool,
}

pub(crate) fn effective_scene_body_shape(ui: &AppWindow) -> bool {
    if ui.get_enable_dual_pass() {
        ui.get_dual_pass_mode_index() == 0
    } else {
        ui.get_scene_body_shape()
    }
}

pub(crate) fn read_ctx(ui: &AppWindow, state: &State) -> ReadCtx {
    let mut config = scan_config(ui);
    config.two_char = false;
    let ai_on = ui.get_image_use_ai() && ui.get_ollama_connected();
    let text_model = ai_on.then(|| default_text_model(ui, state));
    let has_vision = text_model.as_ref().is_some_and(|m| {
        state.borrow().vision_models.iter().any(|v| ai_engine::same_model(v, m))
    });
    let second_pass = ui.get_image_second_pass() && has_vision;
    let scene_body_shape = effective_scene_body_shape(ui);
    ReadCtx {
        model: text_model,
        vision_encoder: ai_on.then(|| vision_model(ui, state)).flatten(),
        host: ui.get_ollama_host().to_string(),
        mode: AiMode::from_index(ui.get_ai_mode_index()),
        threshold: ui.get_tag_threshold(),
        config,
        second_pass,
        scene_body_shape,
        enable_dual_pass: ui.get_enable_dual_pass(),
        use_llm_synthesis: ui.get_use_llm_synthesis(),
    }
}

/// One character's reference image -> its tags (before Identity Only).
pub(crate) fn read_char_blocking(path: &std::path::Path, ctx: &ReadCtx, progress: &(dyn Fn(String) + Sync)) -> anyhow::Result<(String, &'static str, Vec<String>)> {
    if ctx.enable_dual_pass {
        if let (Some(enc), Some(model)) = (&ctx.vision_encoder, &ctx.model) {
            let tags = twoimg::tag_one_fast(path, progress)?;
            let raw_wd14 = tagger::build_prompt(&tags, prompt_condenser::WD14_GENERAL_THRESHOLD).prompt;
            let mode = if ctx.scene_body_shape {
                prompt_condenser::Mode::SceneShape
            } else {
                prompt_condenser::Mode::ScenePose
            };
            let opts = prompt_condenser::PipelineOptions {
                mode,
                use_llm_synthesis: ctx.use_llm_synthesis,
                gemma_vision_pass: ctx.second_pass,
            };
            let dual = ai_engine::run_dual_pass_pipeline(&ctx.host, ctx.mode, enc, model, path, &raw_wd14, opts);
            let combined = match (dual.character_prompt.is_empty(), dual.scene_prompt.is_empty()) {
                (false, false) => format!("{}, {}", dual.character_prompt, dual.scene_prompt),
                (false, true) => dual.character_prompt,
                (true, false) => dual.scene_prompt,
                (true, true) => raw_wd14,
            };
            return Ok((combined, "Dual-Pass AI", dual.warnings));
        }
    }
    let tags = twoimg::tag_one_fast(path, progress)?;
    let hints = tagger::build_prompt(&tags, ctx.threshold).prompt;
    let Some(model) = ctx.model.clone() else { return Ok((hints, "tagger", Vec::new())) };
    let req = ai_engine::ImageAiRequest {
        host: ctx.host.clone(),
        model,
        vision_encoder: ctx.vision_encoder.clone(),
        mode: ctx.mode,
        image_path: path.to_path_buf(),
        config: ctx.config.clone(),
        tag_hints: hints.clone(),
        strong_tags: ai_engine::strong_tags(&tags),
        tagger_scores: ai_engine::tag_scores(&tags),
        second_pass_vision: ctx.second_pass,
        scene_body_shape: ctx.scene_body_shape,
    };
    Ok(match ai_engine::describe_image(&req) {
        Ok(r) => (r.prompt, "AI", Vec::new()),
        Err(_) => (hints, "tagger, the AI failed", Vec::new()),
    })
}

/// A Scene / Pose reference image -> who does what (`sceneref`), for `n` characters.
/// One character: the pose-and-scene reading. Several: the person detector finds them
/// left to right and each one's pose goes to that character (like the image split).
pub(crate) fn read_scene_blocking(
    path: &std::path::Path,
    ctx: &ReadCtx,
    n: usize,
    progress: &(dyn Fn(String) + Sync),
) -> anyhow::Result<(Vec<sceneref::Action>, String, String, Vec<String>)> {
    if n >= 2 {
        let (actions, how, raw) = read_scene_split(path, ctx, n, progress)?;
        return Ok((actions, how, raw, Vec::new()));
    }
    if ctx.enable_dual_pass {
        if let (Some(enc), Some(model)) = (&ctx.vision_encoder, &ctx.model) {
            let tags = twoimg::tag_one_fast(path, progress)?;
            let raw_wd14 = tagger::build_prompt(&tags, prompt_condenser::WD14_GENERAL_THRESHOLD).prompt;
            let mode = if ctx.scene_body_shape {
                prompt_condenser::Mode::SceneShape
            } else {
                prompt_condenser::Mode::ScenePose
            };
            let opts = prompt_condenser::PipelineOptions {
                mode,
                use_llm_synthesis: ctx.use_llm_synthesis,
                gemma_vision_pass: ctx.second_pass,
            };
            let dual = ai_engine::run_dual_pass_pipeline(&ctx.host, ctx.mode, enc, model, path, &raw_wd14, opts);
            let combined = match (dual.scene_prompt.is_empty(), dual.character_prompt.is_empty()) {
                (false, false) => format!("{}, {}", dual.scene_prompt, dual.character_prompt),
                (false, true) => dual.scene_prompt,
                (true, false) => dual.character_prompt,
                (true, true) => raw_wd14,
            };
            let actions = sceneref::from_single_with_shape(&combined, false, ctx.scene_body_shape);
            return Ok((actions, "Dual-Pass AI".to_string(), combined, dual.warnings));
        }
    }
    progress("Tagging scene and character pose…".into());
    let tags = twoimg::tag_one_fast(path, progress)?;
    let raw_prompt = tagger::build_prompt(&tags, ctx.threshold).prompt;
    let hints = scene::filter_scene_and_pose_only_with_shape(&raw_prompt, false, ctx.scene_body_shape);
    let Some(model) = ctx.model.clone() else {
        return Ok((
            sceneref::from_single_with_shape(&raw_prompt, false, ctx.scene_body_shape),
            "tagger".into(),
            raw_prompt,
            Vec::new(),
        ));
    };
    let status_name = if let Some(ref enc) = ctx.vision_encoder {
        if ai_engine::same_model(enc, &model) {
            enc.clone()
        } else if ctx.second_pass {
            format!("{enc} + {model} (2nd Pass Vision)")
        } else {
            format!("{enc} → {model}")
        }
    } else {
        model.clone()
    };
    progress(format!("AI vision analyzing pose with {status_name}…"));
    let req = ai_engine::ImageAiRequest {
        host: ctx.host.clone(),
        model,
        vision_encoder: ctx.vision_encoder.clone(),
        mode: ctx.mode,
        image_path: path.to_path_buf(),
        config: ctx.config.clone(),
        tag_hints: hints.clone(),
        strong_tags: Vec::new(),
        tagger_scores: Vec::new(),
        second_pass_vision: ctx.second_pass,
        scene_body_shape: ctx.scene_body_shape,
    };
    let (prompt, how) = match ai_engine::describe_scene_pose(&req) {
        Ok(r) => (r.prompt, "AI".to_string()),
        Err(_) => (raw_prompt.clone(), "tagger, the AI failed".to_string()),
    };
    let actions = sceneref::from_single_with_shape(&prompt, false, ctx.scene_body_shape);
    Ok((actions, how, prompt, Vec::new()))
}

pub(crate) fn read_scene_split(
    path: &std::path::Path,
    ctx: &ReadCtx,
    n: usize,
    progress: &(dyn Fn(String) + Sync),
) -> anyhow::Result<(Vec<sceneref::Action>, String, String)> {
    if !detector::is_installed() {
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::scope(|s| {
            s.spawn(|| rx.into_iter().for_each(progress));
            let r = detector::download(&tx);
            drop(tx);
            r
        })?;
    }
    progress(format!("Finding the {n} characters in the scene…"));
    let tags = twoimg::read(path, n, false)?;
    let hints = twoimg::rule_split(&tags, ctx.threshold);
    let (split, mut how) = match ctx.model.clone() {
        Some(model) => {
            let status_name = match &ctx.vision_encoder {
                Some(enc) if !ai_engine::same_model(enc, &model) => {
                    if ctx.second_pass {
                        format!("{enc} + {model} (2nd Pass Vision)")
                    } else {
                        format!("{enc} → {model}")
                    }
                }
                _ => model.clone(),
            };
            progress(format!("AI is reading who does what with {status_name}…"));
            match ai_engine::split_image(&ai_engine::TwoImageRequest {
                host: ctx.host.clone(),
                model,
                vision_encoder: ctx.vision_encoder.clone(),
                mode: ctx.mode,
                hints: hints.clone(),
                split: tags.clone(),
                second_pass_vision: ctx.second_pass,
                scene_body_shape: ctx.scene_body_shape,
            }) {
                Ok((s, _)) => (twoimg::settle(s, &tags, ctx.threshold), "AI".to_string()),
                Err(_) => (hints, "tagger, the AI failed".to_string()),
            }
        }
        None => (hints, "tagger".to_string()),
    };
    if let Some(note) = twoimg::not_found_note(&tags) {
        how.push_str(&format!("; {note}, so character 1 got every pose"));
    }
    let raw_tags = format!("{} || {}", split.chars.join(" | "), split.scene);
    Ok((sceneref::from_split_with_shape(&split.chars, &split.scene, false, ctx.scene_body_shape), how, raw_tags))
}

/// Shows the Read button while reference images wait to be read.
pub(crate) fn update_pending(ui: &AppWindow, state: &State) {
    let st = state.borrow();
    let pending = st.scene_ref_path.is_some() && st.scene_ref.is_none()
        || (0..4).any(|k| st.char_paths[k].is_some() && st.char_images[k].is_none());
    ui.set_images_pending(pending);
}

/// Auto Scene button: scans the scene & poses immediately in fast mode (local tagger + rule split, skipping slow AI).
pub(crate) fn run_scene_fast_scan(ui: &AppWindow, state: &State) {
    if !tagger::is_installed() {
        ui.set_image_status("Download the Danbooru tagger first (the button under From Image).".into());
        return;
    }
    let scene_path = {
        let st = state.borrow();
        st.scene_ref_path.clone()
    };
    let Some(path) = scene_path else {
        ui.set_scene_ref_status("No scene reference image loaded yet.".into());
        return;
    };
    let mut ctx = read_ctx(ui, state);
    ctx.model = None;
    ctx.vision_encoder = None; // FORCE FAST SCAN (Skips slow AI LLM/VLM)
    let n = tab_chars(ui);
    let gen = (state.borrow().image_gens, ImageSlots::Scene);
    ai_engine::reset_cancel();
    ui.set_tagger_busy(true);
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let say = status_sender(&weak);
        say("Fast scanning scene & poses…".into());
        let scene_result = read_scene_blocking(&path, &ctx, n, &say);
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            if image_changed_since(&ui, gen) {
                return;
            }
            with_state(|state| {
                match scene_result {
                    Ok((actions, how, raw_tags, _)) => {
                        let found = actions.len();
                        set_scene_reading(&ui, state, actions, raw_tags);
                        ui.set_scene_ref_status(format!("⚡ Auto Scene ({how} • {secs:.1}s): {found} tags detected & assigned.").into());
                        ui.set_image_status(format!("Scene fast-scanned in {secs:.1}s").into());
                    }
                    Err(e) => {
                        ui.set_scene_ref_status(format!("Scene scan failed: {e:#}").into());
                    }
                }
                update_pending(&ui, state);
            });
        });
    });
}

/// The Read button with reference images: the Scene / Pose reference first (deep AI analysis
/// when AI is enabled), then characters 1..4 (who they are).
pub(crate) fn run_read_queue(ui: &AppWindow, state: &State) {
    if !tagger::is_installed() {
        ui.set_image_status("Download the Danbooru tagger first (the button under From Image).".into());
        return;
    }
    let (scene, chars) = {
        let st = state.borrow();
        let scene = st.scene_ref_path.clone();
        let chars: Vec<(usize, std::path::PathBuf)> = (0..tab_chars(ui))
            .filter_map(|k| st.char_paths[k].clone().map(|p| (k, p)))
            .collect();
        (scene, chars)
    };
    let ctx = read_ctx(ui, state);
    let n = tab_chars(ui);
    let gen = (state.borrow().image_gens, ImageSlots::Refs);
    ai_engine::reset_cancel();
    ui.set_tagger_busy(true);
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let say = status_sender(&weak);
        let scene_result = scene.map(|p| {
            say("Reading the scene & pose reference…".into());
            read_scene_blocking(&p, &ctx, n, &say)
        });
        let char_results: Vec<(usize, anyhow::Result<(String, &'static str, Vec<String>)>)> = chars
            .into_iter()
            .map(|(k, p)| {
                if ai_engine::cancelled() {
                    return (k, Err(anyhow::anyhow!("Cancelled")));
                }
                say(format!("Reading character {}…", k + 1));
                (k, read_char_blocking(&p, &ctx, &say))
            })
            .collect();
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            if image_changed_since(&ui, gen) {
                return;
            }
            with_state(|state| {
                let mut done = Vec::new();
                let mut all_warnings = Vec::new();
                // the scene goes in first, so the characters are read against it
                match scene_result {
                    Some(Ok((actions, how, raw_tags, warnings))) => {
                        all_warnings.extend(warnings);
                        let found = actions.len();
                        set_scene_reading(&ui, state, actions, raw_tags);
                        ui.set_scene_ref_status(format!("Auto Scene ({how}): {found} tags placed across characters & scene.").into());
                        done.push(format!("scene & pose ({how})"));
                    }
                    Some(Err(e)) => {
                        ui.set_scene_ref_status(format!("Reading the scene failed: {e:#}").into());
                        done.push(format!("scene & pose failed: {e:#}"));
                    }
                    None => {}
                }
                let mut chars_done = false;
                for (k, r) in char_results {
                    match r {
                        Ok((text, how, warnings)) => {
                            all_warnings.extend(warnings);
                            state.borrow_mut().char_images[k] = Some(text);
                            apply_char_image(&ui, state, k);
                            chars_done = true;
                            done.push(format!("character {} ({how})", k + 1));
                        }
                        Err(e) => done.push(format!("character {} failed: {e:#}", k + 1)),
                    }
                }
                if !all_warnings.is_empty() {
                    ui.set_pipeline_warnings(all_warnings.join(" | ").into());
                } else {
                    ui.set_pipeline_warnings("".into());
                }
                // a character image rewrote its box: put that character's scene actions back
                if chars_done {
                    apply_scene_ref(&ui, state);
                    sync_scene_panel(&ui, state);
                }
                update_pending(&ui, state);
                ui.set_image_status(format!("Read {} • {secs:.0} s", done.join(", ")).into());
            });
        });
    });
}

pub(crate) fn apply_char_image(ui: &AppWindow, state: &State, which: usize) {
    let Some(text) = state.borrow().char_images[which].clone() else { return };
    // one character: no quality tags, no "solo", no group counts
    // with a Scene / Pose reference, the character images only say who they are
    let drop_shape = state.borrow().scene_ref.is_some() && effective_scene_body_shape(ui);
    let text = scene::filter_identity_only_with_shape(&text, drop_shape);
    let text = twoimg::clean_list(&text, false);
    remember_boxes(ui, state);
    set_char_box(ui, which, &text);
    analyze(ui, state);
}

/// Lets a vision model write the prompt, with the tagger's tags as hints (off the UI thread).
pub(crate) fn run_ai_image(ui: &AppWindow, state: &State) {
    let Some(path) = state.borrow().image_path.clone() else { return };
    if !ui.get_ollama_connected() {
        ui.set_image_status("AI mode needs Ollama: connect first (or switch to Tagger).".into());
        return;
    }
    let Some(encoder) = vision_model(ui, state) else {
        ui.set_image_status(
            "AI mode needs a vision model (like Qwen2.5-VL): download it with `ollama run huihui_ai/qwen2.5-vl-abliterated:3b`.".into(),
        );
        return;
    };
    let text_model = default_text_model(ui, state);
    let cached_tags = state.borrow().last_tags.clone().filter(|_| !state.borrow().last_tags_single_pass);
    let has_vision = state.borrow().vision_models.iter().any(|v| ai_engine::same_model(v, &text_model));
    let second_pass = ui.get_image_second_pass() && has_vision;
    let req = ai_engine::ImageAiRequest {
        host: ui.get_ollama_host().to_string(),
        model: text_model.clone(),
        vision_encoder: Some(encoder.clone()),
        mode: AiMode::from_index(ui.get_ai_mode_index()),
        image_path: path.clone(),
        config: scan_config(ui),
        tag_hints: String::new(),
        strong_tags: Vec::new(),
        tagger_scores: Vec::new(),
        second_pass_vision: second_pass,
        scene_body_shape: ui.get_scene_body_shape(),
    };
    let threshold = ui.get_tag_threshold();
    let enable_dual_pass = ui.get_enable_dual_pass();
    let dual_pass_opts = if enable_dual_pass {
        let mode = if ui.get_dual_pass_mode_index() == 0 {
            prompt_condenser::Mode::SceneShape
        } else {
            prompt_condenser::Mode::ScenePose
        };
        Some(prompt_condenser::PipelineOptions {
            mode,
            use_llm_synthesis: ui.get_use_llm_synthesis(),
            // only a text model that can see gets the image (`second_pass` checks that); sending it
            // to E4B / Llama made Ollama reject the call and fall back to the mechanical merge
            gemma_vision_pass: second_pass && ui.get_use_llm_synthesis(),
        })
    } else {
        None
    };
    let gen = (state.borrow().image_gens, ImageSlots::Main);
    ai_engine::reset_cancel();
    ui.set_tagger_busy(true);
    let status_name = if enable_dual_pass {
        let mode_name = if ui.get_dual_pass_mode_index() == 0 { "Shape" } else { "Pose" };
        // the text model that writes the prompt (the dropdown's model, not always Gemma)
        let synth = if ui.get_use_llm_synthesis() {
            format!(" + {}{}", short_model_name(&text_model), if second_pass { " (sees the image)" } else { "" })
        } else {
            " (No LLM)".to_string()
        };
        format!("Dual-Pass [{mode_name}]: {encoder}{synth}")
    } else if ai_engine::same_model(&encoder, &text_model) {
        encoder.clone()
    } else if second_pass {
        format!("{encoder} + {text_model} (2nd Pass Vision)")
    } else {
        format!("{encoder} → {text_model}")
    };
    ui.set_image_status(format!("Combined reading: Tagger + {status_name}…").into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        if let Some(opts) = dual_pass_opts {
            let tags = if cached_tags.is_some() {
                cached_tags
            } else if tagger::is_installed() {
                twoimg::tag_one_fast(&path, &status_sender(&weak)).ok()
            } else {
                None
            };
            let wd14_tags = if let Some(ref t) = tags {
                tagger::build_prompt(t, prompt_condenser::WD14_GENERAL_THRESHOLD).prompt
            } else {
                String::new()
            };
            let strong = tags.as_deref().map(ai_engine::strong_tags).unwrap_or_default();
            let dual_result = ai_engine::run_dual_pass_pipeline(
                &req.host,
                req.mode,
                &encoder,
                &req.model,
                &path,
                &wd14_tags,
                opts,
            );
            let secs = started.elapsed().as_secs_f32();
            let _ = weak.upgrade_in_event_loop(move |ui| {
                ui.set_tagger_busy(false);
                if image_changed_since(&ui, gen) {
                    return;
                }
                with_state(|state| {
                    if tags.is_some() {
                        state.borrow_mut().last_tags = tags;
                        state.borrow_mut().last_tags_single_pass = false;
                    }
                    if !dual_result.warnings.is_empty() {
                        ui.set_pipeline_warnings(dual_result.warnings.join(" | ").into());
                    } else {
                        ui.set_pipeline_warnings("".into());
                    }
                    let raw_combined = match (dual_result.character_prompt.is_empty(), dual_result.scene_prompt.is_empty()) {
                        (false, false) => format!("{}, {}", dual_result.character_prompt, dual_result.scene_prompt),
                        (false, true) => dual_result.character_prompt.clone(),
                        (true, false) => dual_result.scene_prompt.clone(),
                        (true, true) => wd14_tags.clone(),
                    };
                    let deduped_prompt = parser::AntigravityTagFilter::new().resolve_and_strip_flags(&raw_combined);
                    // the tagger's sure skin / species / count tags can't be dropped or contradicted
                    let deduped_prompt = ai_engine::enforce_strong_tags(&deduped_prompt, &strong);
                    let mode_str = match opts.mode {
                        prompt_condenser::Mode::SceneShape => "Shape",
                        prompt_condenser::Mode::ScenePose => "Pose",
                    };
                    let res = ai_engine::ImageAiResult {
                        description: format!("Dual-Pass [{mode_str}]: Scene + Character merged."),
                        prompt: deduped_prompt,
                    };
                    state.borrow_mut().image_ai = Some(res.clone());
                    if ui.get_image_use_ai() {
                        set_prompt_from_image(&ui, state, &identity(&ui, &res.prompt));
                        let timing = format!(" • {secs:.1} s");
                        ui.set_image_status(format!("Dual-Pass [{mode_str}]{timing}: Scene + Character merged.").into());
                    }
                    if std::mem::take(&mut state.borrow_mut().scan_after_image) {
                        state.borrow_mut().scan_image_done = true; // the scan must not read it again
                        ui.invoke_run_ai_scan_requested();
                    }
                });
            });
            return;
        }

        // In Combined mode the tagger runs first: its sure skin / species / count tags are handed to the
        // vision model to check (it used to guess them, running at the same time), and the final prompt
        // may not drop them.
        if cached_tags.is_none() {
            status_sender(&weak)("Step 1/3 • tagging the image…".to_string());
        }
        let tags = cached_tags.or_else(|| {
            tagger::is_installed().then(|| twoimg::tag_one_fast(&path, &status_sender(&weak)).ok()).flatten()
        });
        let mut req = req;
        if let Some(t) = &tags {
            req.tag_hints = ai_engine::color_checked_hints(&tagger::build_prompt(t, threshold).prompt, &path);
            req.strong_tags = ai_engine::strong_tags(t);
            req.tagger_scores = ai_engine::tag_scores(t);
        }
        let say = status_sender(&weak);
        let vision_desc_res = (!ai_engine::same_model(&encoder, &text_model)).then(|| {
            say(format!("Step 2/3 • reading the image with {}…", short_model_name(&encoder)));
            ai_engine::check_cancel()?;
            ai_engine::observe_image(&path, &req.host, req.mode, &encoder, &req.tag_hints, &req.strong_tags)
        });
        say(format!("Step 3/3 • writing the prompt with {}…", short_model_name(&text_model)));
        let result = match vision_desc_res {
            Some(Ok(desc)) => ai_engine::describe_image_with_desc(&req, &desc, &encoder),
            Some(Err(e)) => Err(e),
            None => ai_engine::describe_image(&req),
        };
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            if image_changed_since(&ui, gen) {
                return;
            }
            with_state(|state| {
                if tags.is_some() {
                    state.borrow_mut().last_tags = tags;
                    state.borrow_mut().last_tags_single_pass = false;
                }
                match result {
                    Ok(res) => {
                        state.borrow_mut().image_ai = Some(res.clone());
                        // only show it if the switch is still on AI
                        if ui.get_image_use_ai() {
                            apply_ai_image(&ui, state, &res, Some(secs));
                        }
                        if std::mem::take(&mut state.borrow_mut().scan_after_image) {
                            state.borrow_mut().scan_image_done = true; // the scan must not read it again
                            ui.invoke_run_ai_scan_requested();
                        }
                    }
                    Err(e) => {
                        state.borrow_mut().scan_after_image = false;
                        ui.set_image_status(format!("AI image reading failed: {e:#}").into());
                    }
                }
            });
        });
    });
}

/// A thread-safe way to show progress in the image status line and scene status.
pub(crate) fn status_sender(weak: &slint::Weak<AppWindow>) -> impl Fn(String) + Sync {
    let weak = std::sync::Mutex::new(weak.clone());
    move |msg: String| {
        if let Ok(w) = weak.lock() {
            let _ = w.upgrade_in_event_loop(move |ui| {
                ui.set_image_status(msg.clone().into());
                ui.set_scene_ref_status(msg.into());
            });
        }
    }
}

/// "Identity Only": an image only supplies who the character is and what they wear.
pub(crate) fn identity(ui: &AppWindow, prompt: &str) -> String {
    if ui.get_image_identity_only() {
        scene::filter_identity_only(prompt)
    } else {
        parser::AntigravityTagFilter::new().resolve_and_strip_flags(prompt)
    }
}

pub(crate) fn apply_ai_image(ui: &AppWindow, state: &State, res: &ai_engine::ImageAiResult, secs: Option<f32>) {
    set_prompt_from_image(ui, state, &identity(ui, &res.prompt));
    let timing = secs.map(|s| format!(" • {s:.0} s")).unwrap_or_default();
    ui.set_image_status(format!("Combined{timing}: {}", res.description.trim()).into());
}

/// Runs the Danbooru tagger on the loaded image (off the UI thread).
pub(crate) fn run_tagger(ui: &AppWindow, state: &State) {
    let Some(path) = state.borrow().image_path.clone() else { return };
    let gen = (state.borrow().image_gens, ImageSlots::Main);
    ai_engine::reset_cancel();
    ui.set_tagger_busy(true);
    ui.set_image_status("Extracting Danbooru tags… (the first run loads the model)".into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        // One tagger pass plus the cleanup rules scored like the full AI pipeline on the test images
        // (78%, ~5 s instead of ~32 s); the extra crop passes cost ~20 s and added ~1 point.
        let result = twoimg::tag_one_passes(&path, &status_sender(&weak), false, false).map(|scores| {
            // picture checks once, here (they run the detector and more tagger passes): on the widest list the
            // slider can produce, which tags they remove
            let wide = ai_engine::clean_tagger_rules(&tagger::build_prompt(&scores, 0.15).prompt, &path);
            let drops: std::collections::HashSet<String> = match twoimg::figure_checks(&path, &wide) {
                Some(kept) => {
                    let kept: std::collections::HashSet<String> = kept.split(',').map(danbooru::key).collect();
                    wide.split(',').map(danbooru::key).filter(|k| !k.is_empty() && !kept.contains(k)).collect()
                }
                None => Default::default(),
            };
            (scores, drops)
        });
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            if image_changed_since(&ui, gen) {
                return;
            }
            with_state(|state| match result {
                Ok((scores, drops)) => {
                    {
                        let mut st = state.borrow_mut();
                        st.last_tags = Some(scores);
                        st.last_tags_single_pass = true;
                        st.tagger_picture_drops = drops;
                    }
                    apply_tags(&ui, state, Some(secs));
                }
                Err(e) => ui.set_image_status(format!("Tagging failed: {e:#}").into()),
            });
        });
    });
}

/// The tag threshold slider's default (`tag_threshold` in app.slint). The slider also sets the AI path's
/// tagger hints; in Tagger mode it is an offset from this value (`twoimg::build_fast_prompt`).
pub(crate) const TAG_SLIDER_DEFAULT: f32 = 0.35;

/// Turns the stored tag scores into the Before prompt at the current threshold.
/// `secs` is set right after tagging (first fill, with Undo); `None` = slider re-filter.
pub(crate) fn apply_tags(ui: &AppWindow, state: &State, secs: Option<f32>) {
    let (built, cleaned) = {
        let st = state.borrow();
        let Some(scores) = st.last_tags.as_ref() else { return };
        // the slider moves the per-kind Fast Tagger thresholds from their measured values (at its default)
        let built = twoimg::build_fast_prompt(scores, ui.get_tag_threshold() - TAG_SLIDER_DEFAULT);
        // the same cleanup rules as the AI path; the picture checks' result is reused, not re-run
        let cleaned = match st.image_path.as_ref() {
            Some(path) => ai_engine::clean_tagger_rules(&built.prompt, path)
                .split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty() && !st.tagger_picture_drops.contains(&danbooru::key(t)))
                .collect::<Vec<_>>()
                .join(", "),
            None => built.prompt.clone(),
        };
        (built, cleaned)
    };
    let prompt = identity(ui, &cleaned);
    if secs.is_some() {
        set_prompt_from_image(ui, state, &prompt);
    } else {
        ui.set_raw_prompt(prompt.into());
        analyze(ui, state);
    }
    let timing = secs.filter(|s| *s > 0.0).map(|s| format!(" • {s:.1} s")).unwrap_or_default();
    ui.set_image_status(
        format!(
            "{} Danbooru tags ({} character) • rating: {}{timing}",
            built.characters + built.general,
            built.characters,
            built.rating
        )
        .into(),
    );
}

/// Per-slot image counters (see `AppState::image_gens`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ImageGens {
    pub(crate) main: u64,
    pub(crate) scene: u64,
    pub(crate) chars: [u64; 4],
}

impl ImageGens {
    pub(crate) fn bump_all(&mut self) {
        self.main += 1;
        self.scene += 1;
        for c in &mut self.chars {
            *c += 1;
        }
    }
    /// Did any slot this read uses change between `self` (when it started) and `now`?
    pub(crate) fn changed(&self, now: &ImageGens, slots: ImageSlots) -> bool {
        match slots {
            ImageSlots::Main => self.main != now.main,
            ImageSlots::Scene => self.scene != now.scene,
            ImageSlots::Refs => self.scene != now.scene || self.chars != now.chars,
        }
    }
}

/// Which images a read uses: the main image, the scene reference, or the reference images
/// (scene reference plus character images, read together by the Read queue).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ImageSlots {
    Main,
    Scene,
    Refs,
}

/// For an image read finishing on the UI thread: true (and the result is to be dropped) when an
/// image it read changed since it started, which also cancels a Deep Scan waiting for it.
pub(crate) fn image_changed_since(ui: &AppWindow, started: (ImageGens, ImageSlots)) -> bool {
    let mut changed = false;
    let mut scan_waiting = false;
    with_state(|state| {
        if started.0.changed(&state.borrow().image_gens, started.1) {
            changed = true;
            scan_waiting = std::mem::take(&mut state.borrow_mut().scan_after_image);
        }
    });
    if changed {
        ui.set_image_status("The image changed while it was being read; that result was discarded.".into());
        // a Deep Scan pressed for the new image was waiting on this (old) read: start it now, which
        // reads the current image first (it used to be cancelled together with the old result)
        if scan_waiting {
            ui.invoke_run_ai_scan_requested();
        }
    }
    changed
}

#[cfg(test)]
mod image_gen_tests {
    use super::{short_model_name, ImageGens, ImageSlots};

    #[test]
    fn status_shows_the_real_text_model() {
        assert_eq!(short_model_name("huihui_ai/gemma-4-abliterated:12b"), "gemma-4-abliterated:12b");
        assert_eq!(short_model_name("prompt-condenser-gemma4-e4b:latest"), "prompt-condenser-gemma4-e4b");
        assert_eq!(short_model_name("richardyoung/llama-3.1-8b-instruct-abliterated"), "llama-3.1-8b-instruct-abliterated");
    }

    #[test]
    fn a_read_only_goes_stale_when_its_own_image_changes() {
        let start = ImageGens::default();
        let mut now = start;
        now.chars[1] += 1; // a character image was loaded during the read
        assert!(!start.changed(&now, ImageSlots::Main), "the main image read stays valid");
        assert!(!start.changed(&now, ImageSlots::Scene));
        assert!(start.changed(&now, ImageSlots::Refs), "the reference queue read it");
        now.main += 1;
        assert!(start.changed(&now, ImageSlots::Main));
        let mut all = start;
        all.bump_all();
        for slots in [ImageSlots::Main, ImageSlots::Scene, ImageSlots::Refs] {
            assert!(start.changed(&all, slots), "Redo Vision makes every read stale");
        }
    }
}

/// Wires image loading / reading and the Prompt from PNG button.
pub(crate) fn wire_image_read(ui: &AppWindow, state: &State) {
    // --- Image -> Danbooru tags ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_load_image_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(path) = rfd::FileDialog::new()
                .set_title("Pick an image to extract Danbooru tags from")
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_file()
            else {
                return;
            };
            if let Ok(img) = slint::Image::load_from_path(&path) {
                ui.set_image_thumb(img);
            }
            ui.set_image_loaded(true);
            {
                let mut st = state.borrow_mut();
                st.image_path = Some(path);
                st.image_gens.main += 1;
                st.last_tags = None;
                st.image_ai = None;
                st.two_tags = None;
                st.two_ai = None;
                st.two_before_image = None;
                st.char_images = [None, None, None, None];
                st.char_paths = [None, None, None, None];
                st.prompt_before_image = None; // Undo goes back to the prompt from before this image
            }
            ui.set_image_can_undo(false);
            // nothing is read until Read Image is pressed
            ui.set_image_status(
                if tagger::is_installed() {
                    "Image loaded. Press Read Image (pick Tagger or AI first)."
                } else {
                    "Image loaded. Download the Danbooru tagger once (1.2 GB) to read it."
                }
                .into(),
            );
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_read_image_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // reference images first (scene, then characters); otherwise the loaded image
            if ui.get_images_pending() || ui.get_scene_ref_loaded() {
                state.borrow_mut().char_images = [None, None, None, None];
                run_read_queue(&ui, &state);
            } else {
                image_extract(&ui, &state, true);
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_redo_image_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_image_use_ai(true);
            refresh_vision_encoder(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_remove_image_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            {
                let mut st = state.borrow_mut();
                st.image_path = None;
                st.image_gens.main += 1;
                st.last_tags = None;
                st.image_ai = None;
                st.two_tags = None;
                st.two_ai = None;
            }
            ui.set_image_loaded(false);
            ui.set_image_thumb(slint::Image::default());
            ui.set_image_status("Extract Danbooru tags from any image (WD EVA02)".into());
            ui.set_status_message("Image removed. Your prompt is unchanged (Undo still restores the one from before).".into());
        });
    }

    // --- Prompt from PNG: the generation prompt saved in a PNG's metadata into Before (Undo restores) ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_png_prompt_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(path) = rfd::FileDialog::new()
                .set_title("Pick a PNG to read its generation prompt from")
                .add_filter("PNG images", &["png"])
                .pick_file()
            else {
                return;
            };
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let msg = match png_meta::read_png_prompt(&path) {
                Ok(Some(p)) => {
                    {
                        let mut st = state.borrow_mut();
                        st.two_before_image = None;
                        st.prompt_before_image = Some(ui.get_raw_prompt().to_string());
                        st.card_choices.clear();
                    }
                    ui.set_image_can_undo(true);
                    ui.set_raw_prompt(p.positive.clone().into());
                    analyze(&ui, &state);
                    let negative = p.negative.as_deref().map(|n| format!(" Its negative: {}", n.chars().take(120).collect::<String>())).unwrap_or_default();
                    format!(
                        "Prompt from {name} ({}{}) put into Before; Undo restores yours.{negative}",
                        p.source,
                        if p.partial {
                            ", partial: no sampler link, took the longest text"
                        } else if p.first_pass_chosen {
                            ", the passes have different prompts: took the first pass"
                        } else {
                            ""
                        }
                    )
                }
                Ok(None) => format!("{name} has no saved generation prompt; Before is unchanged."),
                Err(e) => format!("Couldn't read {name} as a PNG: {e}"),
            };
            ui.set_status_message(msg.into());
        });
    }
}
