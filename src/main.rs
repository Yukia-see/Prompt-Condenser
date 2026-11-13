mod ai_engine;
mod analyzer;
mod condenser;
mod caption;
mod check;
mod ideas;
mod prompt_cli;
mod danbooru;
mod detector;
mod dynamic;
mod lora_meta;
mod png_meta;
mod eval;
mod parser;
mod profiles;
mod scene;
mod sceneref;
mod tagger;
mod two;
mod twoimg;
pub mod prompt_condenser;
#[cfg(test)]
mod prompt_condenser_parity;
#[cfg(test)]
mod window_smoke;
mod comfyui;
mod dataset;
mod history;
mod live_bridge;
mod settings;
mod ollama_ui;
mod render;
mod image_read;
mod scene_panel;
mod vrp;
mod deep_scan;
mod translate_ui;
use settings::*;
use ollama_ui::*;
use render::*;
use image_read::*;
use scene_panel::*;
use vrp::*;
use deep_scan::*;
use translate_ui::*;

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
type AiKey = (String, i32, i32, i32, i32, i32, i32); // (prompt, checkpoint, view, camera angle, pose, bust, frame)

/// UI-thread-only state behind the Slint properties.
#[derive(Default)]
struct AppState {
    /// Prompt history (`history.rs`): recent and starred snapshots of the boxes.
    history: history::History,
    /// The user's rule-card choices, kept across edits (`card_key` → mode), and the checkpoint they
    /// were made for (a checkpoint change clears them).
    card_choices: std::collections::HashMap<CardKey, u8>,
    card_choices_checkpoint: i32,
    /// Trigger-word suggestions per character's LoRA box (`lora_meta`), before hiding the ones
    /// already in the box, and the folder + LoRA names they were read for.
    lora_suggest: [Vec<String>; 4],
    lora_suggest_key: String,
    /// Tokens after rule analysis (mode 0 of every change already applied).
    base_tokens: Vec<PromptToken>,
    /// Rule cards.
    changes: Vec<PromptChange>,
    /// AI cards from the last deep scan (opt-in, start as "Keep Original").
    ai_changes: Vec<PromptChange>,
    /// 2 Characters tab: character 2's tokens, rule cards and AI cards.
    /// Characters 2..N on the multi-character tabs: their tokens, rule and AI cards.
    others: Vec<CharCards>,
    /// Scene analysis + full rewrite from the last deep scan (same lifetime as `ai_changes`).
    ai_scene: Option<SceneResult>,
    ai_key: Option<AiKey>,
    /// One model for the card list, updated row by row so unchanged cards
    /// aren't rebuilt on every keystroke.
    cards_model: Rc<VecModel<ChangeCardItem>>,
    /// The region editor's boxes, updated row by row (replacing the model would rebuild the box being dragged).
    vrp_model: Rc<VecModel<VrpRegion>>,
    /// Entries behind the AI model dropdown, same order as `model_labels`.
    model_choices: Vec<ModelChoice>,
    /// The image last loaded with "From Image…".
    image_path: Option<std::path::PathBuf>,
    /// Before-box text from just before an image replaced it (for Undo).
    prompt_before_image: Option<String>,
    /// 🔍 Check Result and 💡 Ideas cards (opt-in), kept while `extra_key` matches the prompt (like AI cards).
    extra_cards: Vec<PromptChange>,
    extra_key: Option<AiKey>,
    /// Check Result cards whose "Add It" puts a tag in the negative: card id → tag.
    check_neg: std::collections::HashMap<usize, String>,
    /// Tags switched off on a card with tag chips (by card key); the others are added when it's accepted.
    chip_off: std::collections::HashMap<render::CardKey, std::collections::HashSet<String>>,
    /// The folded "N small fixes" card is opened (not saved).
    fixes_expanded: bool,
    /// Each 💡 Ideas card's chips that start off (all options but the first), so Reset can restore them.
    idea_chip_defaults: std::collections::HashMap<render::CardKey, std::collections::HashSet<String>>,
    /// The prompt the last image read put into Before (1 Character tab): a Deep Scan of exactly this prompt
    /// gets the read's tags it dropped put back (`deep_scan.rs`).
    image_prompt: Option<String>,
    /// Tag scores for the loaded image, so the threshold slider re-filters instantly.
    last_tags: Option<Vec<tagger::ScoredTag>>,
    /// `last_tags` came from Tagger mode's single pass (not the multi-pass tagging the AI path uses), so the
    /// AI path tags again instead of reusing them.
    last_tags_single_pass: bool,
    /// Tags Tagger mode's picture checks removed (footwear a cut-off picture can't show, inset diagram
    /// tags); the threshold slider re-applies them without running the checks again.
    tagger_picture_drops: std::collections::HashSet<String>,
    /// The AI's reading of the loaded image (cached, so switching modes is instant).
    image_ai: Option<ai_engine::ImageAiResult>,
    /// Installed Ollama models that can read images.
    vision_models: Vec<String>,
    /// Deep Scan was pressed while the AI was still reading the image: scan once it's done.
    scan_after_image: bool,
    /// Set by an image read's hand-off to the Deep Scan: the image was just read, don't read it again.
    scan_image_done: bool,
    /// One counter per image slot, bumped when that image is loaded or removed (Redo Vision bumps all), so a
    /// read that finishes afterwards knows its result is for an old image. Separate per slot:
    /// loading a character image must not throw away a read of the main image.
    image_gens: ImageGens,
    /// The model picked last time (settings.json): selected once Ollama lists it.
    preferred_model: Option<String>,
    /// 2 Characters tab: the detector + tagger reading of the image, and the AI's split.
    two_tags: Option<twoimg::TagSplit>,
    two_ai: Option<twoimg::SplitPrompt>,
    /// The three boxes from just before an image filled them (for Undo).
    two_before_image: Option<(Vec<String>, String)>,
    /// Char 1 / Char 2 reference images: what was read from each (before Identity Only).
    char_images: [Option<String>; 4],
    /// The Scene / Pose reference image and what it put into a prompt box.
    scene_ref: Option<SceneRef>,
    /// Reference images picked but not read yet (Read reads them: scene first, then 1..4).
    scene_ref_path: Option<std::path::PathBuf>,
    char_paths: [Option<std::path::PathBuf>; 4],
    /// Thumbnails of the character reference images (Scene panel and its bar).
    char_thumbs: [Option<slint::Image>; 4],
    /// Region boxes edited in the region editor (fractions of the canvas, one per character); `None` = use the
    /// detector boxes / even strips. Cleared when a new picture is split.
    vrp_regions: Option<Vec<[f64; 4]>>,
    /// Kept alive for the whole session: on Linux the copied text disappears
    /// when the `Clipboard` that owns it is dropped.
    clipboard: Option<arboard::Clipboard>,
}

type State = Rc<RefCell<AppState>>;

/// A Scene / Pose reference image: what it showed each character doing and the scene,
/// who the user gave each tag to, and what went into which box (so it can come out).
struct SceneRef {
    raw_tags: String,
    actions: Vec<sceneref::Action>,
    boxes: Vec<BoxSnap>,
}

/// One prompt box a Scene / Pose reference wrote into.
#[derive(Clone)]
struct BoxSnap {
    /// 0..3 = character 1..4's box (0 = Before), 4 = the shared Scene box.
    target: usize,
    /// The box before the tags went in, and right after (to tell if you edited it since).
    original: String,
    injected: String,
    /// The tags it added (not the ones the box already had).
    added: Vec<String>,
}

/// One extra character's rule-engine result and AI cards (characters 2..N).
#[derive(Default, Clone)]
struct CharCards {
    base_tokens: Vec<PromptToken>,
    changes: Vec<PromptChange>,
    ai_changes: Vec<PromptChange>,
}

/// A multi-line LoRA box as one comma-separated list ("<lora:a:0.8>\nfox girl" ->
/// "<lora:a:0.8>, fox girl").
fn lora_lines(text: &str) -> String {
    text.lines()
        .map(|l| l.trim().trim_matches(',').trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

/// How many characters the current tab has (tab 0 = 1 Character … tab 3 = 4 Characters).
fn tab_chars(ui: &AppWindow) -> usize {
    (ui.get_app_tab().clamp(0, 3) + 1) as usize
}

/// Character `i`'s box (0-based): Before, then Character 2, 3, 4.
fn char_box(ui: &AppWindow, i: usize) -> String {
    match i {
        0 => ui.get_raw_prompt(),
        1 => ui.get_char2_prompt(),
        2 => ui.get_char3_prompt(),
        _ => ui.get_char4_prompt(),
    }
    .to_string()
}

fn set_char_box(ui: &AppWindow, i: usize, text: &str) {
    match i {
        0 => ui.set_raw_prompt(text.into()),
        1 => ui.set_char2_prompt(text.into()),
        2 => ui.set_char3_prompt(text.into()),
        _ => ui.set_char4_prompt(text.into()),
    }
}

/// Characters 3 and 4 (prompt, LoRA box) on the 3 / 4 Characters tabs.
fn more_chars(ui: &AppWindow) -> Vec<(String, String)> {
    let all = [
        (ui.get_char3_prompt().to_string(), lora_lines(&ui.get_char3_lora_tags())),
        (ui.get_char4_prompt().to_string(), lora_lines(&ui.get_char4_lora_tags())),
    ];
    all.into_iter().take(tab_chars(ui).saturating_sub(2)).collect()
}

struct SceneResult {
    summary: String,
    details: String,
    issues: String,
    improved_prompt: String,
    recommended_negative: String,
    /// improved_prompt with {a|b|c} variation groups (when dynamic prompts were on).
    dynamic_prompt: String,
    /// The AI's natural-language sentence (when the toggle was on).
    natural_sentence: String,
}

impl SceneResult {
    fn from_ai(out: &ai_engine::AiJsonOutput, config: &analyzer::ScanConfig) -> Option<Self> {
        let checkpoint = config.checkpoint;
        // tidy_ai_prompt caps weights at the checkpoint's safe maximum (1.35 on Anima); the body rules run after it,
        // so Exaggerated's emphasis (bust :2.5…) survives, as in the rule view
        let tidy = |t: &str| ai_engine::body_choices_in(&condenser::tidy_ai_prompt(t, checkpoint), config);
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
            improved_prompt: tidy(&out.improved_prompt),
            recommended_negative: condenser::tidy_ai_prompt(&out.recommended_negative, checkpoint),
            dynamic_prompt: tidy(&out.dynamic_prompt),
            natural_sentence: out.natural_sentence.trim().replace('\n', " "),
        };
        (!result.summary.is_empty() || !result.improved_prompt.is_empty()).then_some(result)
    }
}

fn with_state(f: impl FnOnce(&State)) {
    STATE.with(|cell| {
        if let Some(state) = cell.borrow().as_ref() {
            f(state);
        }
    });
}

fn main() -> Result<(), slint::PlatformError> {
    let args: Vec<String> = std::env::args().collect();
    // batch captioning for training datasets: no window, no Ollama
    if args.len() > 1 && (args[1] == "--caption" || args[1] == "caption") {
        if let Err(e) = caption::run_caption_cli(&args[2..]) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // prompt cleaning for scripts / the ComfyUI node: JSON out, no window
    if args.len() > 1 && args[1] == "--prompt" {
        if let Err(e) = prompt_cli::run_prompt_cli(&args[2..]) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // tidies existing LoRA training captions (no tags added or swapped)
    if args.len() > 1 && args[1] == "--dataset" {
        if let Err(e) = dataset::run_dataset_cli(&args[2..]) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // installs the ComfyUI add-on into <ComfyUI>/custom_nodes
    if args.len() > 1 && args[1] == "--install-comfyui" {
        if let Err(e) = comfyui::run_install_cli(&args[2..]) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if args.len() > 1 && (args[1] == "eval" || args[1] == "--eval") {
        if let Err(e) = eval::run_eval_cli(&args[2..]) {
            eprintln!("Evaluation error: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let ui = AppWindow::new()?;
    load_settings(&ui);
    ui.set_app_version(env!("CARGO_PKG_VERSION").into());
    ui.set_tagger_installed(tagger::is_installed());
    ui.set_angle_labels(ModelRc::new(VecModel::from(
        profiles::CAMERA_ANGLES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    ui.set_pose_labels(ModelRc::new(VecModel::from(
        profiles::POSES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    ui.set_bust_labels(ModelRc::new(VecModel::from(
        profiles::BUSTS.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    ui.set_frame_labels(ModelRc::new(VecModel::from(
        profiles::FRAMES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    let mut scene_poses = vec![
        SharedString::from("+ Add Pose ▾"),
        SharedString::from("⚡ Auto (from Scene Ref)"),
        SharedString::from("🎲 Dynamic / Random Pose"),
    ];
    for (lbl, _) in profiles::POSES.iter().skip(1) {
        if *lbl != "Random pose (dynamic prompt)" {
            scene_poses.push(SharedString::from(*lbl));
        }
    }
    ui.set_scene_pose_labels(ModelRc::new(VecModel::from(scene_poses)));
    // Scene Panel View / Angle: "+ Add …", "⚡ Auto (from Scene Ref)", then the list from index 1
    for (list, add, set) in [
        (profiles::VIEWS, "+ Add View ▾", AppWindow::set_scene_view_labels as fn(&AppWindow, ModelRc<SharedString>)),
        (profiles::CAMERA_ANGLES, "+ Add Angle ▾", AppWindow::set_scene_angle_labels),
    ] {
        let mut labels = vec![SharedString::from(add), SharedString::from("⚡ Auto (from Scene Ref)")];
        labels.extend(list.iter().skip(1).map(|(l, _)| SharedString::from(*l)));
        set(&ui, ModelRc::new(VecModel::from(labels)));
    }
    let state: State = Rc::new(RefCell::new(AppState::default()));
    state.borrow_mut().history = history::History::load();
    history::push_rows(&ui, &state.borrow().history);
    state.borrow_mut().preferred_model = saved_ai_model();

    ui.set_change_cards(ModelRc::from(state.borrow().cards_model.clone()));
    ui.set_vrp_regions(ModelRc::from(state.borrow().vrp_model.clone()));

    wire_cards(&ui, &state);
    wire_settings_ui(&ui, &state);
    wire_ollama(&ui, &state);
    wire_vrp(&ui, &state);
    wire_scene_panel(&ui, &state);
    wire_image_read(&ui, &state);
    wire_translate(&ui, &state);
    wire_deep_scan(&ui, &state);
    live_bridge::start_probe(&ui);
    live_bridge::wire_queue(&ui);

    STATE.with(|cell| *cell.borrow_mut() = Some(state.clone()));

    set_model_choices(&ui, &state, build_model_choices(&[]));
    // `prompt-condenser "1girl, ..."` opens with that prompt in the Before box.
    if let Some(prompt) = std::env::args().nth(1).filter(|a| !a.starts_with('-')) {
        ui.set_raw_prompt(prompt.into());
    }
    analyze(&ui, &state);
    refresh_hardware_badge(&ui);
    connect_ollama(&ui, false, false);

    // Keep retrying while disconnected, so starting Ollama after the app just works.
    let reconnect = slint::Timer::default();
    {
        let weak = ui.as_weak();
        reconnect.start(slint::TimerMode::Repeated, std::time::Duration::from_secs(5), move || {
            if let Some(ui) = weak.upgrade() {
                if !ui.get_ollama_connected() && !ui.get_ollama_busy() {
                    connect_ollama(&ui, true, false);
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
                update_narrow(&ui);
            }
        });
    }

    let result = ui.run();
    save_settings(&ui);
    state.borrow().history.save();
    result
}

thread_local! {
    /// The UI-thread `State`, reachable from `upgrade_in_event_loop` closures
    /// (those must be `Send`, so they can't capture the `Rc` directly).
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

#[cfg(test)]
mod scene_panel_tests {
    use super::*;

    #[test]
    fn kept_card_choices_survive_an_unrelated_edit() {
        let config = analyzer::ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: TargetView::CloseUpPortrait,
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
            lora_tags_raw: String::new(),
            arranged: false,
            natural: false,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: false,
            scene_prompt_raw: String::new(),
            more_chars: Vec::new(),
            detail: profiles::DetailLevel::Balanced,
        };
        let run = |p: &str| analyzer::analyze_with_rules(&parser::parse_prompt(p), &config).changes;
        let first = run("1girl, thighhighs, red hair");
        let card = first.iter().find(|c| c.original == "thighhighs").expect("a close-up conflict card for thighhighs");
        assert_eq!(card.mode, 0);
        // the user keeps the tag (mode 2), then adds "smile"
        let mut choices = std::collections::HashMap::new();
        choices.insert(card_key(0, card), 2u8);
        let mut second = run("1girl, thighhighs, red hair, smile");
        apply_card_choices(&mut second, 0, &choices);
        let kept = second.iter().find(|c| c.original == "thighhighs").unwrap();
        assert_eq!(kept.mode, 2);
        // another character's card with the same tag isn't touched
        let mut other = run("1girl, thighhighs");
        apply_card_choices(&mut other, 1, &choices);
        assert_eq!(other.iter().find(|c| c.original == "thighhighs").unwrap().mode, 0);
    }

    #[test]
    fn view_and_angle_swap_only_their_own_tags() {
        let p = "1girl, red hair, upper body, from above, high angle, standing";
        let v = swap_group(p, "full body", profiles::VIEWS);
        assert_eq!(v, "1girl, red hair, from above, high angle, standing, full body");
        let a = swap_group(&v, "dutch angle, tilted frame", profiles::CAMERA_ANGLES);
        assert_eq!(a, "1girl, red hair, standing, full body, dutch angle, tilted frame");
        assert_eq!(list_index_in(&a, profiles::VIEWS), 4);
        assert_eq!(list_index_in(&a, profiles::CAMERA_ANGLES), 6);
        // a dynamic angle group is replaced by a fixed angle
        let d = swap_group("1girl, {from above|from below}", "from side, profile", profiles::CAMERA_ANGLES);
        assert_eq!(d, "1girl, from side, profile");
    }

    #[test]
    fn single_tag_view_and_angle_detection() {
        // "close-up" alone matches Close-up / portrait (index 1)
        assert_eq!(list_index_in("1girl, close-up, smile", profiles::VIEWS), 1);
        // "dutch angle" alone matches Dutch angle (index 6)
        assert_eq!(list_index_in("1girl, dutch angle, sword", profiles::CAMERA_ANGLES), 6);
        // "peace sign" alone matches Peace sign (index 8)
        let peace_idx = profiles::POSES.iter().position(|(l, _)| l.starts_with("Peace sign")).unwrap();
        assert_eq!(list_index_in("1girl, peace sign, school uniform", profiles::POSES), peace_idx);
    }

    #[test]
    fn pose_injection_preserves_other_dynamic_groups() {
        let p = "1girl, {red dress|blue dress}, standing";
        let new_p = inject_pose_into_prompt(p, "sitting");
        assert!(new_p.contains("{red dress|blue dress}"), "Dynamic outfit was erased! {new_p}");
        assert!(new_p.contains("sitting"));
        assert!(!new_p.contains("standing"));
    }

    #[test]
    fn extract_char_actions_only_returns_actions_and_poses() {
        let p = "1girl, mature female, long black hair, grey eyes, thick lips, freckles, sunglasses on head, light blue bikini, sitting, leaning forward, holding garment, looking at viewer, on bed, full body, bedroom, bed, soft lighting, anime style";
        let acts = extract_char_actions(p);
        assert_eq!(acts, "sitting, leaning forward, holding garment, looking at viewer");
        assert!(!acts.contains("black hair"));
        assert!(!acts.contains("grey eyes"));
        assert!(!acts.contains("bikini"));
        assert!(!acts.contains("bedroom"));
    }
}
