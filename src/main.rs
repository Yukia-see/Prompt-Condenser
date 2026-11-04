mod ai_engine;
mod analyzer;
mod condenser;
mod dynamic;
mod parser;
mod profiles;
mod scene;
mod tagger;

use ai_engine::{AiScanRequest, GpuInfo, OllamaModel};
use analyzer::{ChangeAction, PromptChange, ScanConfig};
use parser::PromptToken;
use profiles::{AiMode, AiModelTier, CheckpointProfile, TargetView};
use slint::{Model, ModelRc, SharedString, VecModel};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

slint::include_modules!();

/// What an AI scan was run against; its cards are dropped once any of these change.
type AiKey = (String, i32, i32, i32, i32); // (prompt, checkpoint, view, camera angle, pose)

/// UI-thread-only state behind the Slint properties.
#[derive(Default)]
struct AppState {
    /// Tokens after rule analysis (mode 0 of every change already applied).
    base_tokens: Vec<PromptToken>,
    /// Rule cards.
    changes: Vec<PromptChange>,
    /// AI cards from the last deep scan (opt-in, start as "Keep Original").
    ai_changes: Vec<PromptChange>,
    /// Scene analysis + full rewrite from the last deep scan (same lifetime as `ai_changes`).
    ai_scene: Option<SceneResult>,
    ai_key: Option<AiKey>,
    /// One model for the card list, updated row by row so unchanged cards
    /// aren't rebuilt on every keystroke.
    cards_model: Rc<VecModel<ChangeCardItem>>,
    /// Entries behind the AI model dropdown, same order as `model_labels`.
    model_choices: Vec<ModelChoice>,
    /// The image last loaded with "From Image…".
    image_path: Option<std::path::PathBuf>,
    /// Before-box text from just before an image replaced it (for Undo).
    prompt_before_image: Option<String>,
    /// Tag scores for the loaded image, so the threshold slider re-filters instantly.
    last_tags: Option<Vec<tagger::ScoredTag>>,
    /// The AI's reading of the loaded image (cached, so switching modes is instant).
    image_ai: Option<ai_engine::ImageAiResult>,
    /// Installed Ollama models that can read images.
    vision_models: Vec<String>,
    /// Deep Scan was pressed while the AI was still reading the image: scan once it's done.
    scan_after_image: bool,
    /// Kept alive for the whole session: on Linux the copied text disappears
    /// when the `Clipboard` that owns it is dropped.
    clipboard: Option<arboard::Clipboard>,
}

type State = Rc<RefCell<AppState>>;

struct SceneResult {
    summary: String,
    details: String,
    issues: String,
    improved_prompt: String,
    recommended_negative: String,
    /// improved_prompt with {a|b|c} variation groups (when dynamic prompts were on).
    dynamic_prompt: String,
}

impl SceneResult {
    fn from_ai(out: &ai_engine::AiJsonOutput, checkpoint: CheckpointProfile) -> Option<Self> {
        let b = &out.scene_breakdown;
        let details = [
            ("Subject", &b.subject),
            ("Pose", &b.pose),
            ("Framing", &b.framing),
            ("Setting", &b.setting),
            ("Lighting", &b.lighting),
            ("Mood", &b.mood),
        ]
        .iter()
        .filter(|(_, v)| !v.trim().is_empty())
        .map(|(k, v)| format!("{k}: {}", v.trim()))
        .collect::<Vec<_>>()
        .join("\n");
        let issues = out
            .issues
            .iter()
            .filter(|i| !i.trim().is_empty())
            .map(|i| format!("• {}", i.trim()))
            .collect::<Vec<_>>()
            .join("\n");
        let result = Self {
            summary: out.scene_summary.trim().to_string(),
            details,
            issues,
            improved_prompt: condenser::tidy_ai_prompt(&out.improved_prompt, checkpoint),
            recommended_negative: condenser::tidy_ai_prompt(&out.recommended_negative, checkpoint),
            dynamic_prompt: condenser::tidy_ai_prompt(&out.dynamic_prompt, checkpoint),
        };
        (!result.summary.is_empty() || !result.improved_prompt.is_empty()).then_some(result)
    }
}

/// One entry of the AI model dropdown.
#[derive(Clone)]
struct ModelChoice {
    /// Ollama model name.
    name: String,
    /// Set for the pre-packaged tiers; they can be downloaded when missing.
    tier: Option<AiModelTier>,
    installed: bool,
}

impl ModelChoice {
    fn label(&self) -> String {
        match (self.tier, self.installed) {
            (Some(t), true) => t.label().to_string(),
            (Some(t), false) => format!("{} (not installed)", t.label()),
            (None, _) => self.name.clone(),
        }
    }
}

fn scan_config(ui: &AppWindow) -> ScanConfig {
    ScanConfig {
        checkpoint: CheckpointProfile::from_index(ui.get_checkpoint_index()),
        target_view: TargetView::from_index(ui.get_target_view_index()),
        pinned_emphasis_raw: ui.get_pinned_emphasis().to_string(),
        emphasis_weight: (ui.get_emphasis_weight() * 100.0).round() / 100.0,
        cleanup_enabled: ui.get_cleanup_enabled(),
        resolve_conflicts_enabled: ui.get_resolve_conflicts_enabled(),
        clamp_weights_enabled: ui.get_clamp_weights_enabled(),
        camera_angle: ui.get_angle_index().max(0) as usize,
        pose: ui.get_pose_index().max(0) as usize,
        dynamic: ui.get_dynamic_enabled(),
        detail: crate::profiles::DetailLevel::from_index(ui.get_detail_index()),
    }
}

fn ai_key(ui: &AppWindow) -> AiKey {
    (
        ui.get_raw_prompt().to_string(),
        ui.get_checkpoint_index(),
        ui.get_target_view_index(),
        ui.get_angle_index(),
        ui.get_pose_index(),
    )
}

/// Re-runs the rule engine. Resets rule-card choices; AI cards survive unless the
/// prompt, checkpoint or view they were made for has changed.
fn analyze(ui: &AppWindow, state: &State) {
    let config = scan_config(ui);
    let tokens = parser::parse_prompt(&ui.get_raw_prompt());
    let result = analyzer::analyze_with_rules(&tokens, &config);
    let key = ai_key(ui);
    {
        let mut st = state.borrow_mut();
        st.base_tokens = result.processed_tokens;
        st.changes = result.changes;
        if st.ai_key.as_ref() != Some(&key) {
            st.ai_changes.clear();
            st.ai_scene = None;
            st.ai_key = None;
        }
    }
    render(ui, state);
}

fn clear_ai_cards(state: &State) {
    let mut st = state.borrow_mut();
    st.ai_changes.clear();
    st.ai_scene = None;
    st.ai_key = None;
}

/// Rebuilds the After prompt, stats and cards from the current state.
fn render(ui: &AppWindow, state: &State) {
    let config = scan_config(ui);
    let raw = ui.get_raw_prompt();
    let st = state.borrow();
    let all_changes: Vec<PromptChange> = st.changes.iter().chain(&st.ai_changes).cloned().collect();
    let out = condenser::build_condensed_output(&raw, &st.base_tokens, &all_changes, &config);

    // Scene analysis card + optional AI Improved view
    let improved = st.ai_scene.as_ref().map(|sc| sc.improved_prompt.clone()).filter(|p| !p.is_empty());
    ui.set_ai_improved_available(improved.is_some());
    if (improved.is_none() && ui.get_after_view() == 1) || (!config.dynamic && ui.get_after_view() == 2) {
        ui.set_after_view(0);
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
                let groups = p.matches('{').count();
                (p, format!("AI variations: {groups} group(s). Needs the Dynamic Prompts extension / node in ComfyUI."))
            }
            None => {
                let (p, kinds) = dynamic::make_dynamic(&out.final_prompt);
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
    } else { match improved.filter(|_| ui.get_after_view() == 1) {
        Some(ai_text) => {
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
    render_negative(ui, &config, &text, st.ai_scene.as_ref().filter(|_| ui.get_after_view() == 1));
    render_live_scene(ui, &text);
    ui.set_condensed_prompt(text.into());
    // The After box is plain selectable text, so the changed tags are listed under it.
    ui.set_substituted_list(condenser::marked_tags(&markdown, condenser::Highlight::Substituted).join(", ").into());
    ui.set_added_list(condenser::marked_tags(&markdown, condenser::Highlight::Added).join(", ").into());
    ui.set_category_breakdown(out.category_summary.into());
    ui.set_before_stats(format_stats(out.before_tokens, out.before_chunks).into());
    ui.set_after_stats(format_stats(after_tokens, after_chunks).into());

    let cards: Vec<ChangeCardItem> = all_changes.iter().map(to_card).collect();
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

/// Fills the Recommended Negative panel for the prompt currently shown in After.
/// On the AI Improved view it uses the AI's tailored negative (new entries in green).
fn render_negative(ui: &AppWindow, config: &ScanConfig, positive: &str, ai: Option<&SceneResult>) {
    if config.checkpoint.default_negative().is_empty() {
        ui.set_negative_text("".into());
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
    ui.set_negative_text(text.into());
    ui.set_negative_note(note.into());
}

/// Fills the live scene card from the prompt currently shown in After.
fn render_live_scene(ui: &AppWindow, prompt: &str) {
    let d = scene::describe(&parser::parse_prompt(prompt));
    let details = d.details.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join("\n");
    ui.set_live_headline(d.headline.into());
    ui.set_live_details(details.into());
    ui.set_live_missing(d.missing.join(", ").into());
}

fn format_stats(tokens: usize, chunks: usize) -> String {
    format!(
        "{} tokens / {} chunk{}",
        tokens,
        chunks,
        if chunks == 1 { "" } else { "s" }
    )
}

fn to_card(ch: &PromptChange) -> ChangeCardItem {
    let badge_kind = match ch.action {
        ChangeAction::FramingSafeSwap | ChangeAction::PositioningConflict => 0,
        ChangeAction::AiRewrite => 3,
        ChangeAction::KeptEmphasized | ChangeAction::AngleChoice | ChangeAction::PoseChoice => 1,
        ChangeAction::RemovedBloat | ChangeAction::MergedDuplicate | ChangeAction::WeightClamped => 2,
    };
    ChangeCardItem {
        id: ch.id as i32,
        badge: ch.action.badge_text().into(),
        badge_kind,
        original: ch.original.clone().into(),
        reason: ch.reason.clone().into(),
        suggested_alternative: ch.suggested_alternative.clone().into(),
        framing_alternative: ch.framing_alternative.clone().into(),
        has_framing_alt: !ch.framing_alternative.is_empty(),
        mode: ch.mode as i32,
    }
}

fn format_gpu(gpu: &GpuInfo) -> String {
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
fn refresh_hardware_badge(ui: &AppWindow) {
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let badge = format_gpu(&ai_engine::probe_gpu());
        let _ = weak.upgrade_in_event_loop(move |ui| ui.set_hardware_badge(badge.into()));
    });
}

/// The pre-packaged tiers first (installed or not), then everything else Ollama has.
fn build_model_choices(installed: &[OllamaModel]) -> Vec<ModelChoice> {
    let installed: Vec<String> = installed
        .iter()
        .map(|m| m.name.clone())
        .collect();
    let mut choices: Vec<ModelChoice> = AiModelTier::ALL
        .iter()
        .map(|t| ModelChoice {
            name: t.ollama_name().to_string(),
            tier: Some(*t),
            installed: installed.iter().any(|m| ai_engine::same_model(m, t.ollama_name())),
        })
        .collect();
    for m in &installed {
        if !AiModelTier::ALL.iter().any(|t| ai_engine::same_model(m, t.ollama_name())) {
            choices.push(ModelChoice { name: m.clone(), tier: None, installed: true });
        }
    }
    choices
}

/// Replaces the model dropdown, keeping the current selection when it still exists.
fn set_model_choices(ui: &AppWindow, state: &State, choices: Vec<ModelChoice>) {
    // Keep the user's pick only if it was a real, installed model (not the
    // placeholder shown before Ollama answered).
    let previous = selected_choice(ui, state).filter(|c| c.installed).map(|c| c.name);
    let index = previous
        .and_then(|p| choices.iter().position(|c| ai_engine::same_model(&c.name, &p)))
        .or_else(|| choices.iter().rposition(|c| c.installed && c.tier.is_some()))
        .or_else(|| choices.iter().position(|c| c.installed))
        .unwrap_or(0);
    let labels: Vec<SharedString> = choices.iter().map(|c| c.label().into()).collect();
    state.borrow_mut().model_choices = choices;
    ui.set_model_labels(ModelRc::new(VecModel::from(labels)));
    ui.set_model_index(index as i32);
    update_download_flag(ui, state);
}

fn selected_choice(ui: &AppWindow, state: &State) -> Option<ModelChoice> {
    state.borrow().model_choices.get(ui.get_model_index().max(0) as usize).cloned()
}

fn update_download_flag(ui: &AppWindow, state: &State) {
    let needs = selected_choice(ui, state).map(|c| !c.installed).unwrap_or(false);
    ui.set_model_needs_download(needs);
}

/// Asks Ollama for its models (off the UI thread) and refreshes the dropdown.
/// `quiet` is for background retries: no "Connecting..." flicker.
fn connect_ollama(ui: &AppWindow, quiet: bool) {
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

/// Puts an image's prompt into the Before box, remembering what was there for Undo.
fn set_prompt_from_image(ui: &AppWindow, state: &State, prompt: &str) {
    let current = ui.get_raw_prompt().to_string();
    // remember the prompt from before this image only once, so flipping Tagger / AI
    // doesn't overwrite what Undo goes back to
    let already = state.borrow().prompt_before_image.is_some();
    if !already && !current.trim().is_empty() && current.trim() != prompt.trim() {
        state.borrow_mut().prompt_before_image = Some(current);
        ui.set_image_can_undo(true);
    }
    ui.set_raw_prompt(prompt.into());
    analyze(ui, state);
}

/// Reads the loaded image with the method the Tagger / AI switch selects, reusing a
/// cached result when there is one.
fn image_extract(ui: &AppWindow, state: &State) {
    if state.borrow().image_path.is_none() {
        return;
    }
    if ui.get_image_use_ai() {
        let cached = state.borrow().image_ai.clone();
        match cached {
            Some(res) => apply_ai_image(ui, state, &res, None),
            None => run_ai_image(ui, state),
        }
    } else if state.borrow().last_tags.is_some() {
        apply_tags(ui, state, Some(0.0));
    } else if tagger::is_installed() {
        run_tagger(ui, state);
    } else {
        ui.set_image_status("Download the Danbooru tagger once (1.2 GB) to extract tags.".into());
    }
}

/// The model for reading images: the selected one if it can see, else any installed one that can.
fn vision_model(ui: &AppWindow, state: &State) -> Option<String> {
    let st = state.borrow();
    let selected = st.model_choices.get(ui.get_model_index().max(0) as usize).map(|c| c.name.clone());
    selected
        .filter(|s| st.vision_models.iter().any(|v| ai_engine::same_model(v, s)))
        .or_else(|| st.vision_models.first().cloned())
}

/// Lets a vision model write the prompt, with the tagger's tags as hints (off the UI thread).
fn run_ai_image(ui: &AppWindow, state: &State) {
    let Some(path) = state.borrow().image_path.clone() else { return };
    if !ui.get_ollama_connected() {
        ui.set_image_status("AI mode needs Ollama: connect first (or switch to Tagger).".into());
        return;
    }
    let Some(model) = vision_model(ui, state) else {
        ui.set_image_status(
            "AI mode needs a model that can see images: download Maximum Fast (Gemma 4 12B) in the AI model list.".into(),
        );
        return;
    };
    let cached_tags = state.borrow().last_tags.clone();
    let req = ai_engine::ImageAiRequest {
        host: ui.get_ollama_host().to_string(),
        model: model.clone(),
        mode: AiMode::from_index(ui.get_ai_mode_index()),
        image_path: path.clone(),
        config: scan_config(ui),
        tag_hints: String::new(),
    };
    let threshold = ui.get_tag_threshold();
    ui.set_tagger_busy(true);
    ui.set_image_status(format!("AI is reading the image with {model}…").into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        // the tagger's tags make the AI's reading more reliable
        let tags = cached_tags.or_else(|| tagger::is_installed().then(|| tagger::tag_image(&path).ok()).flatten());
        let mut req = req;
        if let Some(t) = &tags {
            req.tag_hints = tagger::build_prompt(t, threshold).prompt;
        }
        let result = ai_engine::describe_image(&req);
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            with_state(|state| {
                if tags.is_some() {
                    state.borrow_mut().last_tags = tags;
                }
                match result {
                    Ok(res) => {
                        state.borrow_mut().image_ai = Some(res.clone());
                        // only show it if the switch is still on AI
                        if ui.get_image_use_ai() {
                            apply_ai_image(&ui, state, &res, Some(secs));
                        }
                        if std::mem::take(&mut state.borrow_mut().scan_after_image) {
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

fn apply_ai_image(ui: &AppWindow, state: &State, res: &ai_engine::ImageAiResult, secs: Option<f32>) {
    set_prompt_from_image(ui, state, &res.prompt);
    let timing = secs.map(|s| format!(" • {s:.0} s")).unwrap_or_default();
    ui.set_image_status(format!("AI{timing}: {}", res.description.trim()).into());
}

/// Runs the Danbooru tagger on the loaded image (off the UI thread).
fn run_tagger(ui: &AppWindow, state: &State) {
    let Some(path) = state.borrow().image_path.clone() else { return };
    ui.set_tagger_busy(true);
    ui.set_image_status("Extracting Danbooru tags… (the first run loads the model)".into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let result = tagger::tag_image(&path);
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            with_state(|state| match result {
                Ok(scores) => {
                    state.borrow_mut().last_tags = Some(scores);
                    apply_tags(&ui, state, Some(secs));
                }
                Err(e) => ui.set_image_status(format!("Tagging failed: {e:#}").into()),
            });
        });
    });
}

/// Turns the stored tag scores into the Before prompt at the current threshold.
/// `secs` is set right after tagging (first fill, with Undo); `None` = slider re-filter.
fn apply_tags(ui: &AppWindow, state: &State, secs: Option<f32>) {
    let built = {
        let st = state.borrow();
        let Some(scores) = st.last_tags.as_ref() else { return };
        tagger::build_prompt(scores, ui.get_tag_threshold())
    };
    if secs.is_some() {
        set_prompt_from_image(ui, state, &built.prompt);
    } else {
        ui.set_raw_prompt(built.prompt.clone().into());
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

fn with_state(f: impl FnOnce(&State)) {
    STATE.with(|cell| {
        if let Some(state) = cell.borrow().as_ref() {
            f(state);
        }
    });
}

/// Window layout and connection settings remembered between sessions.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    before_ratio: f32,
    side_width: f32,
    cards_strip_height: f32,
    ollama_host: String,
    /// Whole-UI zoom on top of the monitor's scale (1.0 = Slint's normal size).
    ui_zoom: f32,
    /// AI detail level (0 Faithful, 1 Balanced, 2 Rich).
    detail_index: i32,
    /// Image reading: false = Danbooru tagger, true = AI.
    image_use_ai: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            before_ratio: 0.4,
            side_width: 320.0,
            cards_strip_height: 250.0,
            ui_zoom: 0.75,
            detail_index: 1,
            image_use_ai: false,
            ollama_host: "localhost:11434".into(),
        }
    }
}

fn settings_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))?;
    Some(base.join("prompt-condenser").join("settings.json"))
}

fn load_settings(ui: &AppWindow) {
    let st: Settings = settings_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    ui.set_before_ratio(st.before_ratio.clamp(0.15, 0.8));
    ui.set_side_width(st.side_width.max(160.0));
    ui.set_cards_strip_height(st.cards_strip_height.max(140.0));
    ui.set_ui_zoom(st.ui_zoom.clamp(ZOOM_MIN, ZOOM_MAX));
    ui.set_detail_index(st.detail_index.clamp(0, 2));
    ui.set_image_use_ai(st.image_use_ai);
    if !st.ollama_host.trim().is_empty() {
        ui.set_ollama_host(st.ollama_host.into());
    }
}

fn save_settings(ui: &AppWindow) {
    let st = Settings {
        before_ratio: ui.get_before_ratio(),
        side_width: ui.get_side_width(),
        cards_strip_height: ui.get_cards_strip_height(),
        ui_zoom: ui.get_ui_zoom(),
        detail_index: ui.get_detail_index(),
        image_use_ai: ui.get_image_use_ai(),
        ollama_host: ui.get_ollama_host().to_string(),
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
const ZOOM_BASE: f32 = 0.75;
/// 70% … 200% as shown, in 10% steps.
const ZOOM_MIN: f32 = ZOOM_BASE * 0.7;
const ZOOM_MAX: f32 = ZOOM_BASE * 2.0;

/// Scales the whole UI: Slint's Material widgets have fixed minimum sizes, so a
/// global scale is the only way to make everything smaller together.
/// The monitor's own scale (`base`) is tracked so moving to another screen still works.
struct Zoom {
    base: std::cell::Cell<f32>,
    applied: std::cell::Cell<f32>,
}

impl Zoom {
    fn apply(&self, ui: &AppWindow) {
        let target = self.base.get() * ui.get_ui_zoom();
        self.applied.set(target);
        ui.window().dispatch_event(slint::platform::WindowEvent::ScaleFactorChanged { scale_factor: target });
        ui.set_ui_zoom_percent((ui.get_ui_zoom() / ZOOM_BASE * 100.0).round() as i32);
    }

    /// Called periodically: if the platform reset the scale (new monitor, settings
    /// change), take that as the new base and re-apply the zoom on top of it.
    fn check(&self, ui: &AppWindow) {
        let current = ui.window().scale_factor();
        if (current - self.applied.get()).abs() > 0.001 {
            self.base.set(current);
            self.apply(ui);
        }
    }
}

fn main() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    load_settings(&ui);
    ui.set_tagger_installed(tagger::is_installed());
    ui.set_angle_labels(ModelRc::new(VecModel::from(
        profiles::CAMERA_ANGLES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    ui.set_pose_labels(ModelRc::new(VecModel::from(
        profiles::POSES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    let state: State = Rc::new(RefCell::new(AppState::default()));

    ui.set_change_cards(ModelRc::from(state.borrow().cards_model.clone()));

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

    // Changing the AI mode or model refreshes the page: AI cards came from the
    // previous model, so they are dropped and the rules re-run.
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_ai_settings_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
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

    // --- Card toggles ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_card_mode_changed(move |card_id, mode| {
            let Some(ui) = weak.upgrade() else { return };
            {
                let mut st = state.borrow_mut();
                let st = &mut *st;
                if let Some(ch) = st
                    .changes
                    .iter_mut()
                    .chain(st.ai_changes.iter_mut())
                    .find(|c| c.id as i32 == card_id)
                {
                    ch.mode = mode.clamp(0, 2) as u8;
                }
            }
            render(&ui, &state);
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

    // --- Clipboard ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_copy_output_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let text = ui.get_condensed_prompt().to_string();
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let msg = match st.clipboard.as_mut().map(|c| c.set_text(text)) {
                Some(Ok(())) => "Condensed prompt copied to clipboard.".to_string(),
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
            let text = ui.get_negative_text().to_string();
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let ok = st.clipboard.as_mut().map(|c| c.set_text(text).is_ok()).unwrap_or(false);
            ui.set_status_message(if ok { "Recommended negative prompt copied." } else { "Copy failed." }.into());
        });
    }


    // --- Ollama connection ---
    {
        let weak = ui.as_weak();
        ui.on_connect_ollama_requested(move || {
            if let Some(ui) = weak.upgrade() {
                connect_ollama(&ui, false);
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
                    connect_ollama(&ui, false);
                });
            });
        });
    }

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
                st.last_tags = None;
                st.image_ai = None;
                st.prompt_before_image = None; // Undo goes back to the prompt from before this image
            }
            ui.set_image_can_undo(false);
            image_extract(&ui, &state);
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
                st.last_tags = None;
                st.image_ai = None;
            }
            ui.set_image_loaded(false);
            ui.set_image_thumb(slint::Image::default());
            ui.set_image_status("Extract Danbooru tags from any image (WD EVA02)".into());
            ui.set_status_message("Image removed. Your prompt is unchanged (Undo still restores the one from before).".into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_image_mode_changed(move || {
            if let Some(ui) = weak.upgrade() {
                image_extract(&ui, &state);
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_add_image_tags_to_keep(move || {
            let Some(ui) = weak.upgrade() else { return };
            // the image's tags from the active mode, minus quality / rating filler
            let text = {
                let st = state.borrow();
                if ui.get_image_use_ai() {
                    st.image_ai.as_ref().map(|r| r.prompt.clone())
                } else {
                    st.last_tags.as_ref().map(|t| tagger::build_prompt(t, ui.get_tag_threshold()).prompt)
                }
            };
            let Some(text) = text else {
                ui.set_status_message("Nothing extracted from the image yet.".into());
                return;
            };
            let current = ui.get_pinned_emphasis().to_string();
            let mut seen: Vec<String> = parser::parse_prompt(&current).into_iter().map(|t| t.normalized).collect();
            let mut parts: Vec<String> = current.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
            let mut added = 0;
            for t in parser::parse_prompt(&text) {
                if t.category == parser::TokenCategory::QualityMeta || seen.contains(&t.normalized) {
                    continue;
                }
                seen.push(t.normalized.clone());
                parts.push(t.text.clone());
                added += 1;
            }
            ui.set_pinned_emphasis(parts.join(", ").into());
            analyze(&ui, &state);
            ui.set_status_message(format!("Added {added} tag(s) from the image to Keep & Emphasize.").into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_tag_threshold_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            let has_tags = state.borrow().last_tags.is_some();
            if has_tags && !ui.get_image_use_ai() {
                apply_tags(&ui, &state, None);
            }
        });
    }
    {
        let weak = ui.as_weak();
        ui.on_download_tagger_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_tagger_busy(true);
            let (tx, rx) = mpsc::channel::<String>();
            let progress_weak = weak.clone();
            std::thread::spawn(move || {
                for msg in rx {
                    let _ = progress_weak.upgrade_in_event_loop(move |ui| ui.set_image_status(msg.into()));
                }
            });
            let done_weak = weak.clone();
            std::thread::spawn(move || {
                let result = tagger::download(tx);
                let _ = done_weak.upgrade_in_event_loop(move |ui| {
                    ui.set_tagger_busy(false);
                    ui.set_tagger_installed(tagger::is_installed());
                    match result {
                        Ok(()) => with_state(|state| {
                            if state.borrow().image_path.is_some() {
                                run_tagger(&ui, state);
                            } else {
                                ui.set_image_status("Danbooru tagger ready. Click From Image… to tag a picture.".into());
                            }
                        }),
                        Err(e) => ui.set_image_status(format!("Tagger download failed: {e:#}").into()),
                    }
                });
            });
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_undo_image_prompt_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let previous = state.borrow_mut().prompt_before_image.take();
            if let Some(text) = previous {
                ui.set_raw_prompt(text.into());
                ui.set_image_can_undo(false);
                analyze(&ui, &state);
                ui.set_status_message("Restored your previous prompt.".into());
            }
        });
    }

    // --- Deep AI scan ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_run_ai_scan_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // With an image loaded and the switch on AI, read the image first, then scan.
            if ui.get_image_use_ai() && ui.get_image_loaded() {
                if ui.get_tagger_busy() {
                    state.borrow_mut().scan_after_image = true;
                    ui.set_status_message("Deep scan starts as soon as the AI has read the image…".into());
                    return;
                }
                let unread = state.borrow().image_ai.is_none();
                if unread && vision_model(&ui, &state).is_some() {
                    state.borrow_mut().scan_after_image = true;
                    run_ai_image(&ui, &state);
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
            let req = AiScanRequest {
                raw_prompt: raw.clone(),
                config: scan_config(&ui),
                mode: AiMode::from_index(ui.get_ai_mode_index()),
                host: ui.get_ollama_host().to_string(),
                model: choice.name.clone(),
                keep_alive: choice.tier.map(|t| t.keep_alive()).unwrap_or("30m").to_string(),
                think: false,
            };

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
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_ai_busy(false);
                    // `state` is Rc (UI thread only), so fetch it via a thread-local handle.
                    with_state(|state| {
                        match result {
                            Ok((ai_out, placement)) => {
                                ui.set_run_badge(format!("Last scan: {placement}").into());
                                let where_ran = format!("ran on {placement}");
                                if ui.get_raw_prompt().as_str() != raw {
                                    ui.set_status_message("Prompt changed during the AI scan — result discarded. Run it again.".into());
                                    return;
                                }
                                analyze(&ui, state);
                                let found = {
                                    let mut st = state.borrow_mut();
                                    let tokens = parser::parse_prompt(&raw);
                                    st.ai_changes = ai_engine::convert_ai_changes(&ai_out.changes, &tokens, &st.changes);
                                    st.ai_scene = SceneResult::from_ai(&ai_out, scan_config(&ui).checkpoint);
                                    st.ai_key = Some(ai_key(&ui));
                                    st.ai_changes.len()
                                };
                                let has_rewrite = state.borrow().ai_scene.as_ref().map(|s| !s.improved_prompt.is_empty()).unwrap_or(false);
                                if has_rewrite {
                                    ui.set_after_view(1);
                                }
                                render(&ui, state);
                                let msg = match (has_rewrite, found) {
                                    (true, _) => format!(
                                        "AI scan done — showing the AI Improved prompt; {found} individual suggestion(s) in the cards • {where_ran}"
                                    ),
                                    (false, 0) => format!("AI scan done — nothing beyond the rule cards • {where_ran}"),
                                    (false, n) => format!("AI scan done — {n} AI card(s) added; click to apply • {where_ran}"),
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

    STATE.with(|cell| *cell.borrow_mut() = Some(state.clone()));

    set_model_choices(&ui, &state, build_model_choices(&[]));
    // `prompt-condenser "1girl, ..."` opens with that prompt in the Before box.
    if let Some(prompt) = std::env::args().nth(1).filter(|a| !a.starts_with('-')) {
        ui.set_raw_prompt(prompt.into());
    }
    analyze(&ui, &state);
    refresh_hardware_badge(&ui);
    connect_ollama(&ui, false);

    // Keep retrying while disconnected, so starting Ollama after the app just works.
    let reconnect = slint::Timer::default();
    {
        let weak = ui.as_weak();
        reconnect.start(slint::TimerMode::Repeated, std::time::Duration::from_secs(5), move || {
            if let Some(ui) = weak.upgrade() {
                if !ui.get_ollama_connected() && !ui.get_ollama_busy() {
                    connect_ollama(&ui, true);
                }
            }
        });
    }
    // --- UI size ---
    let zoom = Rc::new(Zoom { base: std::cell::Cell::new(0.0), applied: std::cell::Cell::new(-1.0) });
    // each click is 10% of the shown size
    for (dir, step) in [(-1.0f32, 0.1f32), (1.0, 0.1)] {
        let weak = ui.as_weak();
        let zoom = zoom.clone();
        let handler = move || {
            let Some(ui) = weak.upgrade() else { return };
            let shown = ((ui.get_ui_zoom() / ZOOM_BASE + dir * step) * 10.0).round() / 10.0;
            ui.set_ui_zoom((shown * ZOOM_BASE).clamp(ZOOM_MIN, ZOOM_MAX));
            if zoom.base.get() > 0.0 {
                zoom.apply(&ui);
            }
        };
        if dir < 0.0 {
            ui.on_zoom_out_requested(handler);
        } else {
            ui.on_zoom_in_requested(handler);
        }
    }
    let zoom_watch = slint::Timer::default();
    {
        let weak = ui.as_weak();
        let zoom = zoom.clone();
        zoom_watch.start(slint::TimerMode::Repeated, std::time::Duration::from_millis(250), move || {
            if let Some(ui) = weak.upgrade() {
                zoom.check(&ui);
            }
        });
    }

    let result = ui.run();
    save_settings(&ui);
    result
}

thread_local! {
    /// The UI-thread `State`, reachable from `upgrade_in_event_loop` closures
    /// (those must be `Send`, so they can't capture the `Rc` directly).
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}
