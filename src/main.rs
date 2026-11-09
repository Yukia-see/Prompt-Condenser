mod ai_engine;
mod analyzer;
mod condenser;
mod danbooru;
mod detector;
mod dynamic;
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
type AiKey = (String, i32, i32, i32, i32, i32); // (prompt, checkpoint, view, camera angle, pose, proportion)

/// UI-thread-only state behind the Slint properties.
#[derive(Default)]
struct AppState {
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
    /// Set by an image read's hand-off to the Deep Scan: the image was just read, don't read it again.
    scan_image_done: bool,
    /// One counter per image slot, bumped when that image is loaded, removed or re-read, so a
    /// read that finishes afterwards knows its result is for an old image. Separate per slot:
    /// loading a character image must not throw away a read of the main image.
    image_gens: ImageGens,
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
            natural_sentence: out.natural_sentence.trim().replace('\n', " "),
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
        proportion: ui.get_proportion_index().max(0) as usize,
        dynamic: ui.get_dynamic_enabled(),
        lora_tags_raw: lora_lines(&ui.get_lora_tags()), arranged: ui.get_arranged_enabled(), natural: ui.get_natural_enabled(), two_char: ui.get_app_tab() >= 1, char2_prompt_raw: ui.get_char2_prompt().to_string(), char2_lora_tags_raw: lora_lines(&ui.get_char2_lora_tags()), char_break: ui.get_char_break_enabled(), scene_prompt_raw: ui.get_scene_prompt().to_string(), more_chars: more_chars(ui),
        detail: crate::profiles::DetailLevel::from_index(ui.get_detail_index()),
    }
}

fn ai_key(ui: &AppWindow) -> AiKey {
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
        ui.get_proportion_index(),
    )
}

/// Re-runs the rule engine. Resets rule-card choices; AI cards survive unless the
/// prompt, checkpoint or view they were made for has changed.
fn analyze(ui: &AppWindow, state: &State) {
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
        label_two_char_cards(&mut result, 1);
    }
    let key = ai_key(ui);
    {
        let mut st = state.borrow_mut();
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
    }
    render(ui, state);
}

/// The prompts to work on, one per character: Before alone on the 1 Character tab, or
/// Before plus the other characters' boxes (a pasted "… BREAK …" prompt is split when
/// they're empty).
fn char_prompts(ui: &AppWindow, config: &ScanConfig) -> Vec<String> {
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
fn with_scene(c1: &str, config: &ScanConfig) -> String {
    let scene = config.scene_prompt_raw.trim().trim_matches(',').trim();
    if scene.is_empty() {
        return c1.to_string();
    }
    let c1 = c1.trim().trim_end_matches(',').trim();
    if c1.is_empty() { scene.to_string() } else { format!("{c1}, {scene}") }
}

/// Two characters: "solo" goes (with a card), and every card says whose it is.
fn label_two_char_cards(r: &mut analyzer::RuleAnalysisResult, n: usize) {
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

fn clear_ai_cards(state: &State) {
    let mut st = state.borrow_mut();
    st.ai_changes.clear();
    for o in &mut st.others {
        o.ai_changes.clear();
    }
    st.ai_scene = None;
    st.ai_key = None;
}

/// Rebuilds the After prompt, stats and cards from the current state.
fn render(ui: &AppWindow, state: &State) {
    let config = scan_config(ui);
    let two_char = config.two_char;
    let mut prompts = char_prompts(ui, &config);
    prompts[0] = with_scene(&prompts[0], &config);
    let c1_raw = prompts[0].clone();
    // what the AI views are compared against: every character on the multi-character tabs
    let raw = prompts.join(", ");
    let st = state.borrow();
    let changes1: Vec<PromptChange> = st.changes.iter().chain(&st.ai_changes).cloned().collect();
    let other_changes: Vec<Vec<PromptChange>> =
        st.others.iter().map(|o| o.changes.iter().chain(&o.ai_changes).cloned().collect()).collect();
    let all_changes: Vec<PromptChange> = changes1.iter().chain(other_changes.iter().flatten()).cloned().collect();
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
            format_stats(t, c)
        })
    };
    ui.set_char2_stats(stats_of(1).unwrap_or_default().into());
    ui.set_char3_stats(stats_of(2).unwrap_or_default().into());
    ui.set_char4_stats(stats_of(3).unwrap_or_default().into());
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
        let blocks_text = if two_char {
            two::combine_section_blocks(&rule_texts, &config, true, ai_sentence.as_deref())
        } else {
            two::combine_section_blocks(std::slice::from_ref(&rule_prompt), &config, true, ai_sentence.as_deref())
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
    render_negative(ui, &config, &text, st.ai_scene.as_ref().filter(|_| ui.get_after_view() == 1 || ui.get_after_view() == 4));
    match shown_parts.filter(|_| two_char) {
        Some(c) => set_live_scene(ui, scene::describe_many(&c.chars, &c.shared)),
        None => render_live_scene(ui, &text),
    }
    ui.set_condensed_prompt(text.into());
    // The After box is plain selectable text, so the changed tags are listed under it.
    ui.set_substituted_list(condenser::marked_tags(&markdown, condenser::Highlight::Substituted).join(", ").into());
    ui.set_added_list(condenser::marked_tags(&markdown, condenser::Highlight::Added).join(", ").into());
    ui.set_structure_rows(ModelRc::new(VecModel::from(structure_rows(&out.category_summary))));
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

/// The Organized Structure text ("[Subject] 1girl, solo" per line) as colored rows.
fn structure_rows(summary: &str) -> Vec<StructureRow> {
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
    set_live_scene(ui, scene::describe(&parser::parse_prompt(prompt)));
}

fn set_live_scene(ui: &AppWindow, d: scene::SceneDescription) {
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
        | ChangeAction::StyleConflict => 2,
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

/// Redoes the vision encoder by invalidating all cached image and reference analysis
/// and re-running the vision model against loaded images.
fn refresh_vision_encoder(ui: &AppWindow, state: &State) {
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
fn connect_ollama(ui: &AppWindow, quiet: bool, redo_vision: bool) {
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
/// Shows what the loaded image gives in the current mode. With `run`, reads it when
/// that hasn't happened yet (the Read Image button); without, only reuses what was
/// already read, so switching modes never starts a slow run by itself.
fn image_extract(ui: &AppWindow, state: &State, run: bool) {
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
fn two_image_extract(ui: &AppWindow, state: &State, run: bool) {
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

fn two_status(t: &twoimg::TagSplit, what: &str) -> String {
    match twoimg::not_found_note(t) {
        Some(note) => format!("{note}: {what} needs AI mode to separate them"),
        None if t.overlap > 0.25 => format!("{} characters found (some overlap) • {what}", t.wanted),
        None => format!("{} characters found • {what}", t.wanted),
    }
}

/// Detector + tagger (+ the vision model in AI mode), off the UI thread.
fn run_two_image(ui: &AppWindow, state: &State) {
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
            with_state(|state| match result {
                Ok((tags, ai_split)) => {
                    state.borrow_mut().two_tags = Some(tags.clone());
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
fn remember_boxes(ui: &AppWindow, state: &State) {
    let boxes: Vec<String> = (0..4).map(|i| char_box(ui, i)).collect();
    let scene = ui.get_scene_prompt().to_string();
    let had_text = boxes.iter().any(|b| !b.trim().is_empty()) || !scene.trim().is_empty();
    let mut st = state.borrow_mut();
    if st.two_before_image.is_none() && had_text {
        st.two_before_image = Some((boxes, scene));
        ui.set_image_can_undo(true);
    }
}

fn apply_two(ui: &AppWindow, state: &State, split: &twoimg::SplitPrompt, status: String) {
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
struct ReadCtx {
    /// The target text model (from dropdown / default tier).
    model: Option<String>,
    /// The vision encoder (defaults to Moondream).
    vision_encoder: Option<String>,
    host: String,
    mode: AiMode,
    threshold: f32,
    config: ScanConfig,
    second_pass: bool,
    scene_body_shape: bool,
    enable_dual_pass: bool,
    use_llm_synthesis: bool,
}

fn effective_scene_body_shape(ui: &AppWindow) -> bool {
    if ui.get_enable_dual_pass() {
        ui.get_dual_pass_mode_index() == 0
    } else {
        ui.get_scene_body_shape()
    }
}

fn read_ctx(ui: &AppWindow, state: &State) -> ReadCtx {
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
fn read_char_blocking(path: &std::path::Path, ctx: &ReadCtx, progress: &(dyn Fn(String) + Sync)) -> anyhow::Result<(String, &'static str, Vec<String>)> {
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
fn read_scene_blocking(
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

fn read_scene_split(
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
fn update_pending(ui: &AppWindow, state: &State) {
    let st = state.borrow();
    let pending = st.scene_ref_path.is_some() && st.scene_ref.is_none()
        || (0..4).any(|k| st.char_paths[k].is_some() && st.char_images[k].is_none());
    ui.set_images_pending(pending);
}

/// Auto Scene button: scans the scene & poses immediately in fast mode (local tagger + rule split, skipping slow AI).
fn run_scene_fast_scan(ui: &AppWindow, state: &State) {
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
fn run_read_queue(ui: &AppWindow, state: &State) {
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

fn apply_char_image(ui: &AppWindow, state: &State, which: usize) {
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

fn scene_overrides(ui: &AppWindow) -> sceneref::Overrides {
    sceneref::Overrides {
        scene: ui.get_scene_override_setting().to_string(),
        exclude: ui.get_scene_override_exclude().to_string(),
        chars: [
            ui.get_scene_override_c1().to_string(),
            ui.get_scene_override_c2().to_string(),
            ui.get_scene_override_c3().to_string(),
            ui.get_scene_override_c4().to_string(),
        ],
    }
}

/// Stores a new reading and applies its plan across the boxes.
fn set_scene_reading(ui: &AppWindow, state: &State, actions: Vec<sceneref::Action>, raw_tags: String) {
    let mut st = state.borrow_mut();
    // take out the old reading first, if any
    let old_boxes = st.scene_ref.take().map(|r| r.boxes).unwrap_or_default();
    drop(st);
    restore_boxes(ui, &old_boxes);

    ui.set_scene_override_setting("".into());
    ui.set_scene_override_c1("".into());
    ui.set_scene_override_c2("".into());
    ui.set_scene_override_c3("".into());
    ui.set_scene_override_c4("".into());
    ui.set_scene_prompt("".into());

    let mut st = state.borrow_mut();
    st.scene_ref = Some(SceneRef { raw_tags, actions, boxes: Vec::new() });
    drop(st);
    apply_scene_ref(ui, state);
    sync_scene_panel(ui, state);
}

/// Computes the plan from the current actions and overrides, and injects into the boxes.
fn apply_scene_ref(ui: &AppWindow, state: &State) {
    let n = tab_chars(ui);
    let (actions, old_boxes) = {
        let st = state.borrow();
        let Some(sr) = st.scene_ref.as_ref() else { return };
        (sr.actions.clone(), sr.boxes.clone())
    };
    let ov = scene_overrides(ui);
    let plan = sceneref::plan(&actions, &ov, n);

    // Box index: 0 = Before / Char 1, 1 = Char 2, 2 = Char 3, 3 = Char 4, 4 = Scene box
    // (box, tags, which predicate: 0 = actions, 1 = scene, 2 = both)
    let targets: Vec<(usize, Vec<String>, u8)> = {
        let mut list = Vec::new();
        for i in 0..n {
            let tags = plan.chars.get(i).cloned().unwrap_or_default();
            list.push((i, tags, 0));
        }
        list.push((4, plan.scene.clone(), 1));
        list
    };

    let mut new_snaps = Vec::new();
    let other_prompts: Vec<String> = (0..n).map(|i| char_box(ui, i)).collect();
    for (target, refs, kind) in targets {
        let current = if target == 4 { ui.get_scene_prompt().to_string() } else { char_box(ui, target) };
        let original = match old_boxes.iter().find(|b| b.target == target && b.injected == current) {
            Some(b) => b.original.clone(),
            None => current.clone(),
        };
        let injected = if target == 4 {
            // When doing references or scene: clear shared prompt; if text already exists that is different from character prompts, add to it
            sceneref::update_shared_scene_prompt(&original, &refs, &other_prompts)
        } else {
            let body_shape_on = effective_scene_body_shape(ui);
            sceneref::inject(&original, &refs, &|t| match kind {
                0 => sceneref::is_action(t) || (body_shape_on && crate::scene::is_body_shape_tag(&t.normalized)),
                1 => scene::is_scene_or_pose(t),
                _ => sceneref::is_pose_or_scene(t),
            })
        };
        if target == 4 {
            ui.set_scene_prompt(injected.clone().into());
        } else {
            set_char_box(ui, target, &injected);
        }
        new_snaps.push(BoxSnap { target, original, injected, added: refs });
    }

    if let Some(sr) = state.borrow_mut().scene_ref.as_mut() {
        sr.boxes = new_snaps;
    }
    analyze(ui, state);
}

/// Restores boxes from a snapshot: exact match restores original, modified only strips added tags.
fn restore_boxes(ui: &AppWindow, snaps: &[BoxSnap]) {
    for b in snaps {
        let current = if b.target == 4 { ui.get_scene_prompt().to_string() } else { char_box(ui, b.target) };
        let restored = if current == b.injected {
            b.original.clone()
        } else {
            parser::parse_prompt(&current)
                .into_iter()
                .filter(|t| !b.added.iter().any(|x| danbooru::key(x) == t.normalized))
                .map(|t| t.raw.trim().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        if b.target == 4 {
            ui.set_scene_prompt(restored.into());
        } else {
            set_char_box(ui, b.target, &restored);
        }
    }
}

/// Syncs the horizontal Scene Panel cards in Slint from the current plan.
/// Extracts only action and pose tags describing what a character is doing.
fn extract_char_actions(prompt: &str) -> String {
    let tokens = parser::parse_prompt(prompt);
    tokens
        .into_iter()
        .filter(sceneref::is_action)
        .map(|t| t.raw.trim().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn sync_scene_panel(ui: &AppWindow, state: &State) {
    let n = tab_chars(ui);
    let st = state.borrow();
    let ov = scene_overrides(ui);
    let plan = st.scene_ref.as_ref().map(|sr| sceneref::plan(&sr.actions, &ov, n));

    let cur_scene = ui.get_scene_prompt().to_string();
    if !cur_scene.trim().is_empty() && ui.get_scene_override_setting().trim().is_empty() {
        ui.set_scene_override_setting(cur_scene.into());
    } else if let Some(ref p) = plan {
        if ui.get_scene_override_setting().trim().is_empty() && !p.scene.is_empty() {
            ui.set_scene_override_setting(p.scene.join(", ").into());
        }
    }

    let mut items = Vec::new();
    for i in 0..n {
        let title = if n == 1 { "Character 1".to_string() } else { format!("Character {}", i + 1) };
        let cbox = char_box(ui, i);
        let actions = if let Some(ref p) = plan {
            let acts = p.chars.get(i).map(|v| v.join(", ")).unwrap_or_default();
            if !acts.trim().is_empty() {
                acts
            } else {
                extract_char_actions(&cbox)
            }
        } else {
            extract_char_actions(&cbox)
        };
        let has_actions = !actions.trim().is_empty();

        let pose_group = profiles::group_tags(profiles::POSES);
        let cbox_tokens = parser::parse_prompt(&cbox);
        let has_dynamic_pose = cbox_tokens.iter().any(|t| {
            let n = t.normalized.as_str();
            n.starts_with('{') && n.ends_with('}') && {
                let opts: Vec<String> = n.trim_matches(|c| c == '{' || c == '}').split('|').map(|o| o.trim().to_string()).collect();
                opts.iter().any(|o| pose_group.contains(o))
            }
        });
        let pose_index = if has_dynamic_pose {
            2 // Dynamic / Random Pose
        } else {
            let idx = list_index_in(&cbox, profiles::POSES);
            if idx > 0 && profiles::POSES[idx].0 != "Random pose (dynamic prompt)" {
                (idx + 2) as i32
            } else {
                0
            }
        };

        let view_index = match list_index_in(&cbox, profiles::VIEWS) {
            0 => 0,
            i => i + 1,
        } as i32;
        let angle_index = match list_index_in(&cbox, profiles::CAMERA_ANGLES) {
            0 if cbox.contains(profiles::CAMERA_ANGLES[profiles::CAMERA_ANGLES.len() - 1].1) => profiles::CAMERA_ANGLES.len() as i32,
            0 => 0,
            i => (i + 1) as i32,
        };
        items.push(ScenePanelChar {
            title: title.into(),
            actions: actions.into(),
            has_actions,
            view_index,
            angle_index,
            pose_index,
        });
    }
    ui.set_scene_panel_chars(Rc::new(VecModel::from(items)).into());
}

/// Puts `new_tags` from one dropdown list (VIEWS / CAMERA_ANGLES) into a character's
/// prompt, taking out that list's other tags first (and its {a|b} groups), and keeping
/// everything else, including other dropdowns' tags.
fn swap_group(current: &str, new_tags: &str, list: &[(&str, &str)]) -> String {
    let group = profiles::group_tags(list);
    let chosen: Vec<String> = parser::parse_prompt(new_tags).into_iter().map(|t| t.normalized).collect();
    let of_group = |t: &parser::PromptToken| {
        let n = t.normalized.as_str();
        if n.starts_with('{') && n.ends_with('}') {
            // a dynamic group of this list's options
            let opts: Vec<String> = n.trim_matches(|c| c == '{' || c == '}').split('|').map(|o| o.trim().to_string()).collect();
            return !opts.is_empty() && opts.iter().any(|o| group.contains(o));
        }
        group.iter().any(|g| g == n) && !chosen.iter().any(|c| c == n)
    };
    let mut kept: Vec<String> = parser::parse_prompt(current)
        .into_iter()
        .filter(|t| !of_group(t))
        .map(|t| t.raw.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for t in parser::parse_prompt(new_tags) {
        if !kept.iter().any(|k| danbooru::key(k) == t.normalized) {
            kept.push(t.raw.trim().to_string());
        }
    }
    kept.join(", ")
}

/// Which entry of a dropdown list (index into `list`) a prompt already uses: the first
/// whose tags are all in it, or the entry with the most matching tags. 0 = none.
fn list_index_in(prompt: &str, list: &[(&str, &str)]) -> usize {
    let toks: Vec<String> = parser::parse_prompt(prompt).into_iter().map(|t| t.normalized).collect();
    // First, look for an entry where all tags match (most specific first)
    let mut best_all: Option<(usize, usize)> = None;
    for (i, (_, tags)) in list.iter().enumerate().skip(1) {
        if tags.starts_with('{') {
            if toks.iter().any(|tok| tok == *tags) {
                return i;
            }
            continue;
        }
        let want: Vec<String> = parser::parse_prompt(tags).into_iter().map(|t| t.normalized).collect();
        if !want.is_empty()
            && want.iter().all(|w| toks.contains(w))
            && best_all.map(|(_, count)| want.len() > count).unwrap_or(true)
        {
            best_all = Some((i, want.len()));
        }
    }
    if let Some((i, _)) = best_all {
        return i;
    }
    // Otherwise, find the entry with the most matching tags (at least one)
    let mut best_any: Option<(usize, usize)> = None;
    for (i, (_, tags)) in list.iter().enumerate().skip(1) {
        if tags.starts_with('{') {
            continue;
        }
        let want: Vec<String> = parser::parse_prompt(tags).into_iter().map(|t| t.normalized).collect();
        let matches = want.iter().filter(|w| toks.contains(w)).count();
        if matches > 0 && best_any.map(|(_, count)| matches > count).unwrap_or(true) {
            best_any = Some((i, matches));
        }
    }
    best_any.map(|(i, _)| i).unwrap_or(0)
}

/// A Scene Panel View / Angle dropdown pick for character `idx`. `auto` takes that
/// list's tags from the Scene / Pose reference.
fn scene_list_selected(ui: &AppWindow, state: &State, idx: usize, val: &str, list: &[(&str, &str)], what: &str) {
    if val.starts_with('+') || val.is_empty() {
        return;
    }
    let current = char_box(ui, idx);
    let tags = if val.starts_with('⚡') {
        let group = profiles::group_tags(list);
        let found: Vec<String> = state
            .borrow()
            .scene_ref
            .as_ref()
            .map(|r| {
                r.actions
                    .iter()
                    .filter(|a| matches!(a.owner, sceneref::Owner::Scene | sceneref::Owner::All) || a.owner == sceneref::Owner::Char(idx))
                    .filter(|a| group.contains(&danbooru::key(&a.tag)))
                    .map(|a| a.tag.clone())
                    .collect()
            })
            .unwrap_or_default();
        if found.is_empty() {
            ui.set_scene_ref_status(format!("No {what} found in the Scene reference for Character {}.", idx + 1).into());
            return;
        }
        found.join(", ")
    } else {
        match list.iter().find(|(l, _)| *l == val) {
            Some((_, t)) if !t.is_empty() => t.to_string(),
            _ => return,
        }
    };
    set_char_box(ui, idx, &swap_group(&current, &tags, list));
    sync_scene_panel(ui, state);
    analyze(ui, state);
    ui.set_scene_ref_status(format!("Character {} {what}: {val}", idx + 1).into());
}

/// Injects or replaces a pose for a character prompt, replacing existing pose tags from the standard poses list.
fn inject_pose_into_prompt(current: &str, new_pose: &str) -> String {
    swap_group(current, new_pose, profiles::POSES)
}

/// Removes the reference image completely.
fn clear_scene_ref(ui: &AppWindow, state: &State) {
    state.borrow_mut().scene_ref_path = None;
    state.borrow_mut().image_gens.scene += 1;
    let old = state.borrow_mut().scene_ref.take();
    if let Some(r) = old {
        restore_boxes(ui, &r.boxes);
    }
    ui.set_scene_ref_loaded(false);
    ui.set_scene_ref_thumb(slint::Image::default());
    ui.set_scene_ref_status("".into());
    ui.set_scene_panel_open(false);
    sync_scene_panel(ui, state);
    update_pending(ui, state);
    analyze(ui, state);
}

/// The default target text model for writing prompts (from dropdown or default tier).
/// A model name short enough for the status line: `huihui_ai/gemma-4-abliterated:12b` -> `gemma-4-abliterated:12b`.
fn short_model_name(name: &str) -> &str {
    let n = name.rsplit('/').next().unwrap_or(name);
    n.strip_suffix(":latest").unwrap_or(n)
}

fn default_text_model(ui: &AppWindow, state: &State) -> String {
    let st = state.borrow();
    let selected = st.model_choices.get(ui.get_model_index().max(0) as usize).filter(|c| c.installed);
    // With the 2nd Pass Visual Check on, the text model has to see the image: keep the
    // dropdown's model when it can (the 12B).
    let selected_sees = selected.is_some_and(|c| st.vision_models.iter().any(|v| ai_engine::same_model(v, &c.name)));
    if !(ui.get_image_second_pass() && selected_sees) {
        // Otherwise image reads use E4B (Good): it wrote the best image prompts in the tests and
        // keeps the 12B out of image reads, so it doesn't share the memory with the tagger and the
        // vision model (that made the system swap). The Deep Scan still uses the dropdown's model.
        if let Some(e4b) = st.model_choices.iter().find(|c| c.installed && c.tier == Some(AiModelTier::Good)) {
            return e4b.name.clone();
        }
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
fn vision_model(ui: &AppWindow, state: &State) -> Option<String> {
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

/// Lets a vision model write the prompt, with the tagger's tags as hints (off the UI thread).
fn run_ai_image(ui: &AppWindow, state: &State) {
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
    let cached_tags = state.borrow().last_tags.clone();
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
        let tags = cached_tags.or_else(|| {
            tagger::is_installed().then(|| twoimg::tag_one_fast(&path, &status_sender(&weak)).ok()).flatten()
        });
        let mut req = req;
        if let Some(t) = &tags {
            req.tag_hints = tagger::build_prompt(t, threshold).prompt;
            req.strong_tags = ai_engine::strong_tags(t);
        }
        let vision_desc_res = (!ai_engine::same_model(&encoder, &text_model)).then(|| {
            ai_engine::observe_image(&path, &req.host, req.mode, &encoder, &req.tag_hints, &req.strong_tags)
        });
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
fn status_sender(weak: &slint::Weak<AppWindow>) -> impl Fn(String) + Sync {
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
fn identity(ui: &AppWindow, prompt: &str) -> String {
    if ui.get_image_identity_only() {
        scene::filter_identity_only(prompt)
    } else {
        parser::AntigravityTagFilter::new().resolve_and_strip_flags(prompt)
    }
}

fn apply_ai_image(ui: &AppWindow, state: &State, res: &ai_engine::ImageAiResult, secs: Option<f32>) {
    set_prompt_from_image(ui, state, &identity(ui, &res.prompt));
    let timing = secs.map(|s| format!(" • {s:.0} s")).unwrap_or_default();
    ui.set_image_status(format!("Combined{timing}: {}", res.description.trim()).into());
}

/// Runs the Danbooru tagger on the loaded image (off the UI thread).
fn run_tagger(ui: &AppWindow, state: &State) {
    let Some(path) = state.borrow().image_path.clone() else { return };
    let gen = (state.borrow().image_gens, ImageSlots::Main);
    ui.set_tagger_busy(true);
    ui.set_image_status("Extracting Danbooru tags… (the first run loads the model)".into());
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let result = twoimg::tag_one_fast(&path, &status_sender(&weak));
        let secs = started.elapsed().as_secs_f32();
        let _ = weak.upgrade_in_event_loop(move |ui| {
            ui.set_tagger_busy(false);
            if image_changed_since(&ui, gen) {
                return;
            }
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
    let prompt = identity(ui, &built.prompt);
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
struct ImageGens {
    main: u64,
    scene: u64,
    chars: [u64; 4],
}

impl ImageGens {
    fn bump_all(&mut self) {
        self.main += 1;
        self.scene += 1;
        for c in &mut self.chars {
            *c += 1;
        }
    }
    /// Did any slot this read uses change between `self` (when it started) and `now`?
    fn changed(&self, now: &ImageGens, slots: ImageSlots) -> bool {
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
enum ImageSlots {
    Main,
    Scene,
    Refs,
}

/// For an image read finishing on the UI thread: true (and the result is to be dropped) when an
/// image it read changed since it started, which also cancels a Deep Scan waiting for it.
fn image_changed_since(ui: &AppWindow, started: (ImageGens, ImageSlots)) -> bool {
    let mut changed = false;
    with_state(|state| {
        if started.0.changed(&state.borrow().image_gens, started.1) {
            changed = true;
            state.borrow_mut().scan_after_image = false;
        }
    });
    if changed {
        ui.set_image_status("The image changed while it was being read; that result was discarded.".into());
    }
    changed
}

fn with_state(f: impl FnOnce(&State)) {
    STATE.with(|cell| {
        if let Some(state) = cell.borrow().as_ref() {
            f(state);
        }
    });
}

/// Window layout, connection settings and the prompts, remembered between sessions.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    before_ratio: f32,
    side_width: f32,
    cards_strip_height: f32,
    ollama_host: String,
    /// Whole-UI zoom on top of the monitor's scale (1.0 = Slint's normal size).
    ui_zoom: f32,
    /// Target checkpoint (0 Anima, 1 Pony, 2 Illustrious, 3 SDXL, 4 SD1.5, 5 Flux).
    checkpoint_index: i32,
    /// AI execution mode (0 GPU, 1 CPU, 2 AI Off).
    ai_mode_index: i32,
    /// Target camera view (0 None, 1 360, 2 Upper Body, etc.).
    target_view_index: i32,
    /// Camera angle (0 None, 1 Straight-on, etc.).
    angle_index: i32,
    /// Pose (0 None, 1 Standing, etc.).
    pose_index: i32,
    /// Body proportion (0 None, 1 Flat chest, etc.).
    proportion_index: i32,
    /// AI detail level (0 Faithful, 1 Balanced, 2 Rich).
    detail_index: i32,
    /// Image reading: false = Danbooru tagger, true = AI.
    image_use_ai: bool,
    /// The Before prompt and the Keep & Emphasize list from the last session.
    raw_prompt: String,
    pinned_emphasis: String,
    lora_tags: String,
    arranged: bool,
    natural: bool,
    /// 0 = 1 Character, 1 = 2 Characters.
    app_tab: i32,
    char2_prompt: String,
    char2_lora_tags: String,
    char3_prompt: String,
    char3_lora_tags: String,
    char4_prompt: String,
    char4_lora_tags: String,
    char_break: bool,
    scene_prompt: String,
    image_identity_only: bool,
    image_second_pass: bool,
    scene_body_shape: bool,
    enable_dual_pass: bool,
    dual_pass_mode_index: i32,
    use_llm_synthesis: bool,
    /// THEME dropdown: 0 OLED Pitch Black, 1 M3 Deep Violet, 2 OLED Cyber Emerald,
    /// 3 M3 Ocean Blue, 4 M3 Crimson Ember, 5 M3 Mocha Gold.
    theme_index: i32,
    /// Prompt vs LoRA box split inside each character's card.
    lora_split_ratios: [f32; 4],
    /// The Options drawer open or closed.
    options_expanded: bool,
    /// Output view: 0 Rule-based, 1 AI Improved, 2 Dynamic, 3 Section Blocks.
    after_view: i32,
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
            detail_index: 1,
            image_use_ai: false,
            raw_prompt: String::new(),
            pinned_emphasis: String::new(),
            lora_tags: String::new(),
            arranged: false,
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
            image_second_pass: true,
            scene_body_shape: false,
            enable_dual_pass: true,
            dual_pass_mode_index: 0,
            use_llm_synthesis: true,
            theme_index: 1,
            lora_split_ratios: [0.7; 4],
            options_expanded: true,
            after_view: 3,
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
    ui.set_checkpoint_index(st.checkpoint_index.clamp(0, 5));
    ui.set_ai_mode_index(st.ai_mode_index.clamp(0, 2));
    ui.set_target_view_index(st.target_view_index.clamp(0, 7));
    ui.set_angle_index(st.angle_index.max(0));
    ui.set_pose_index(st.pose_index.max(0));
    ui.set_proportion_index(st.proportion_index.max(0));
    ui.set_detail_index(st.detail_index.clamp(0, 2));
    ui.set_image_use_ai(st.image_use_ai);
    ui.set_raw_prompt(st.raw_prompt.into());
    ui.set_pinned_emphasis(st.pinned_emphasis.into());
    ui.set_lora_tags(st.lora_tags.into());
    ui.set_arranged_enabled(st.arranged);
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
    ui.set_image_second_pass(st.image_second_pass);
    ui.set_scene_body_shape(st.scene_body_shape);
    ui.set_enable_dual_pass(st.enable_dual_pass);
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
}

fn save_settings(ui: &AppWindow) {
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
        proportion_index: ui.get_proportion_index(),
        detail_index: ui.get_detail_index(),
        image_use_ai: ui.get_image_use_ai(),
        raw_prompt: ui.get_raw_prompt().to_string(),
        pinned_emphasis: ui.get_pinned_emphasis().to_string(),
        lora_tags: ui.get_lora_tags().to_string(),
        arranged: ui.get_arranged_enabled(),
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
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && (args[1] == "eval" || args[1] == "--eval") {
        if let Err(e) = eval::run_eval_cli(&args[2..]) {
            eprintln!("Evaluation error: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let ui = AppWindow::new()?;
    load_settings(&ui);
    ui.set_tagger_installed(tagger::is_installed());
    ui.set_angle_labels(ModelRc::new(VecModel::from(
        profiles::CAMERA_ANGLES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    ui.set_pose_labels(ModelRc::new(VecModel::from(
        profiles::POSES.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
    )));
    ui.set_proportion_labels(ModelRc::new(VecModel::from(
        profiles::PROPORTIONS.iter().map(|(l, _)| SharedString::from(*l)).collect::<Vec<_>>(),
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
                    .chain(st.others.iter_mut().flat_map(|o| o.changes.iter_mut().chain(o.ai_changes.iter_mut())))
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
            let mut st = state.borrow_mut();
            if st.clipboard.is_none() {
                st.clipboard = arboard::Clipboard::new().ok();
            }
            let msg = match st.clipboard.as_mut().map(|c| c.set_text(text)) {
                Some(Ok(())) => if is_ai { "AI Section blocks prompt copied to clipboard.".to_string() } else { "Section blocks prompt copied to clipboard.".to_string() },
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
    // --- Theme: saved as soon as it changes ---
    {
        let weak = ui.as_weak();
        ui.on_theme_changed(move || {
            if let Some(ui) = weak.upgrade() {
                save_settings(&ui);
            }
        });
    }

    // --- Scene / Pose reference image ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_load_scene_ref_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(path) = rfd::FileDialog::new()
                .set_title("Pick a reference image for the pose and the scene")
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_file()
            else {
                return;
            };
            if let Ok(img) = slint::Image::load_from_path(&path) {
                ui.set_scene_ref_thumb(img);
            }
            ui.set_scene_ref_loaded(true);
            // a new reference replaces the old one: its tags come out now, the new
            // one's go in on Read
            let old_boxes = {
                let mut st = state.borrow_mut();
                st.scene_ref_path = Some(path);
                st.image_gens.scene += 1;
                st.scene_ref.take().map(|r| r.boxes).unwrap_or_default()
            };
            restore_boxes(&ui, &old_boxes);
            ui.set_scene_prompt("".into());
            analyze(&ui, &state);
            ui.set_scene_ref_status("Scene image loaded. Click '⚡ Auto Scene (Fast)' or '🧠 Read (AI)'.".into());
            update_pending(&ui, &state);
            sync_scene_panel(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_remove_scene_ref_requested(move || {
            if let Some(ui) = weak.upgrade() {
                clear_scene_ref(&ui, &state);
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_char_image_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let which = (which.max(0) as usize).min(tab_chars(&ui) - 1);
            let Some(path) = rfd::FileDialog::new()
                .set_title(format!("Pick a reference image for character {}", which + 1))
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_file()
            else {
                return;
            };
            if let Ok(img) = slint::Image::load_from_path(&path) {
                match which {
                    0 => { ui.set_char1_image_thumb(img.clone()); ui.set_char1_image_loaded(true); }
                    1 => { ui.set_char2_image_thumb(img.clone()); ui.set_char2_image_loaded(true); }
                    2 => { ui.set_char3_image_thumb(img.clone()); ui.set_char3_image_loaded(true); }
                    _ => { ui.set_char4_image_thumb(img.clone()); ui.set_char4_image_loaded(true); }
                }
                state.borrow_mut().char_thumbs[which] = Some(img);
            }
            {
                let mut st = state.borrow_mut();
                st.char_paths[which] = Some(path);
                st.image_gens.chars[which] += 1;
                st.char_images[which] = None;
            }
            update_pending(&ui, &state);
            ui.set_image_status(format!("Character {} image loaded. Click Read when ready.", which + 1).into());
        });
    }
    // --- Scene Panel callbacks: Auto Scene, Add Action, Overrides ---
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_auto_scene_requested(move || {
            let Some(ui) = weak.upgrade() else { return };
            // no scene image yet: pick one, then scan it right away (one click, not two)
            if state.borrow().scene_ref_path.is_none() {
                ui.invoke_load_scene_ref_requested();
                if state.borrow().scene_ref_path.is_none() {
                    return; // the picker was cancelled
                }
            }
            run_scene_fast_scan(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_overrides_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            let setting = ui.get_scene_override_setting().to_string();
            let cleaned = twoimg::clean_list(&setting, true);
            ui.set_scene_prompt(cleaned.into());
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_view_selected({
            let weak = weak.clone();
            let state = state.clone();
            move |which, val| {
                if let Some(ui) = weak.upgrade() {
                    scene_list_selected(&ui, &state, which.max(0) as usize, &val, profiles::VIEWS, "view");
                }
            }
        });
        ui.on_scene_angle_selected({
            let weak = weak.clone();
            let state = state.clone();
            move |which, val| {
                if let Some(ui) = weak.upgrade() {
                    scene_list_selected(&ui, &state, which.max(0) as usize, &val, profiles::CAMERA_ANGLES, "camera angle");
                }
            }
        });
        ui.on_scene_pose_selected(move |which, val| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let val_str = val.to_string();

            if val_str == "+ Add Pose ▾" || val_str.is_empty() {
                return;
            }

            let current = char_box(&ui, idx);

            let new_prompt = if val_str == "⚡ Auto (from Scene Ref)" {
                let group = profiles::group_tags(profiles::POSES);
                let detected: Vec<String> = {
                    let st = state.borrow();
                    if let Some(scene) = st.scene_ref.as_ref() {
                        scene.actions.iter()
                            .filter(|a| matches!(a.owner, sceneref::Owner::Char(c) if c == idx) || matches!(a.owner, sceneref::Owner::All))
                            .filter(|a| group.contains(&danbooru::key(&a.tag)))
                            .map(|a| a.tag.clone())
                            .collect()
                    } else {
                        Vec::new()
                    }
                };
                if detected.is_empty() {
                    ui.set_scene_ref_status(format!("No pose detected in Scene reference for Character {}.", idx + 1).into());
                    return;
                }
                inject_pose_into_prompt(&current, &detected.join(", "))
            } else if val_str == "🎲 Dynamic / Random Pose" {
                inject_pose_into_prompt(&current, "{standing|sitting|leaning against wall|dynamic pose|looking back|arms up}")
            } else {
                let pose_tag = profiles::POSES.iter()
                    .find(|(lbl, _)| *lbl == val_str)
                    .map(|(_, tags)| *tags)
                    .unwrap_or(&val_str);
                if pose_tag.is_empty() {
                    return;
                }
                inject_pose_into_prompt(&current, pose_tag)
            };

            set_char_box(&ui, idx, &new_prompt);
            sync_scene_panel(&ui, &state);
            analyze(&ui, &state);
            ui.set_scene_ref_status(format!("Character {} pose: {}", idx + 1, val_str).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_action_add_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let current = char_box(&ui, idx);

            // 1. Check if a scene reference detected poses for this character
            let group = profiles::group_tags(profiles::POSES);
            let detected: Vec<String> = {
                let st = state.borrow();
                if let Some(scene) = st.scene_ref.as_ref() {
                    scene.actions.iter()
                        .filter(|a| matches!(a.owner, sceneref::Owner::Char(c) if c == idx) || matches!(a.owner, sceneref::Owner::All))
                        .filter(|a| group.contains(&danbooru::key(&a.tag)))
                        .map(|a| a.tag.clone())
                        .collect()
                } else {
                    Vec::new()
                }
            };

            let new_prompt = if !detected.is_empty() {
                inject_pose_into_prompt(&current, &detected.join(", "))
            } else {
                inject_pose_into_prompt(&current, "standing")
            };

            set_char_box(&ui, idx, &new_prompt);
            sync_scene_panel(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_char_clear_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let current = char_box(&ui, idx);
            let new_prompt = sceneref::replace(&current, &[], &|t| sceneref::is_action(t));
            set_char_box(&ui, idx, &new_prompt);
            match idx {
                0 => ui.set_scene_override_c1("".into()),
                1 => ui.set_scene_override_c2("".into()),
                2 => ui.set_scene_override_c3("".into()),
                _ => ui.set_scene_override_c4("".into()),
            }
            let changed = {
                let mut st = state.borrow_mut();
                if let Some(r) = st.scene_ref.as_mut() {
                    sceneref::clear_char_actions(&mut r.actions, idx);
                    true
                } else {
                    false
                }
            };
            if changed {
                apply_scene_ref(&ui, &state);
            }
            sync_scene_panel(&ui, &state);
            analyze(&ui, &state);
            ui.set_scene_ref_status(format!("Character {} actions cleared.", idx + 1).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_char_text_edited(move |which, text| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = which.max(0) as usize;
            let current = char_box(&ui, idx);
            let action_tags: Vec<String> = parser::parse_prompt(&text)
                .into_iter()
                .map(|t| t.raw.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            // exactly the tags in the card: removing one removes it, emptying the card removes all
            let new_prompt = sceneref::replace(&current, &action_tags, &sceneref::is_action);
            set_char_box(&ui, idx, &new_prompt);
            if action_tags.is_empty() {
                // an empty card means "none", not "use the image's": drop them from the reading too
                if let Some(r) = state.borrow_mut().scene_ref.as_mut() {
                    sceneref::clear_char_actions(&mut r.actions, idx);
                }
            }
            match idx {
                0 => ui.set_scene_override_c1(text.clone()),
                1 => ui.set_scene_override_c2(text.clone()),
                2 => ui.set_scene_override_c3(text.clone()),
                _ => ui.set_scene_override_c4(text.clone()),
            }
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_action_move_requested(move |from, to| {
            let Some(ui) = weak.upgrade() else { return };
            let from = from.max(0) as usize;
            let to = to.max(0) as usize;
            let changed = {
                let mut st = state.borrow_mut();
                if let Some(r) = st.scene_ref.as_mut() {
                    sceneref::move_char_actions(&mut r.actions, from, to);
                    true
                } else {
                    false
                }
            };
            if changed {
                apply_scene_ref(&ui, &state);
                sync_scene_panel(&ui, &state);
                ui.set_scene_ref_status(format!("Moved pose from Character {} to Character {}.", from + 1, to + 1).into());
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_remove_char_image_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = (which.max(0) as usize).min(3);
            {
                let mut st = state.borrow_mut();
                st.char_paths[idx] = None;
                st.image_gens.chars[idx] += 1;
                st.char_images[idx] = None;
                st.char_thumbs[idx] = None;
            }
            match idx {
                0 => { ui.set_char1_image_loaded(false); ui.set_char1_image_thumb(slint::Image::default()); }
                1 => { ui.set_char2_image_loaded(false); ui.set_char2_image_thumb(slint::Image::default()); }
                2 => { ui.set_char3_image_loaded(false); ui.set_char3_image_thumb(slint::Image::default()); }
                _ => { ui.set_char4_image_loaded(false); ui.set_char4_image_thumb(slint::Image::default()); }
            }
            update_pending(&ui, &state);
            ui.set_status_message(format!("Character {} reference image removed.", idx + 1).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_clear_char_requested(move |which| {
            let Some(ui) = weak.upgrade() else { return };
            let idx = (which.max(0) as usize).min(3);
            match idx {
                0 => { ui.set_raw_prompt("".into()); ui.set_lora_tags("".into()); }
                1 => { ui.set_char2_prompt("".into()); ui.set_char2_lora_tags("".into()); }
                2 => { ui.set_char3_prompt("".into()); ui.set_char3_lora_tags("".into()); }
                _ => { ui.set_char4_prompt("".into()); ui.set_char4_lora_tags("".into()); }
            }
            analyze(&ui, &state);
            ui.set_status_message(format!("Character {} cleared.", idx + 1).into());
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_identity_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            // re-filter what was already read, no models run again
            image_extract(&ui, &state, false);
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
        });
    }
    {
        let weak = ui.as_weak();
        let _state = state.clone();
        ui.on_second_pass_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = ui.get_image_second_pass();
            let msg = if on {
                "2nd pass visual check enabled (vision models directly rescan image pixels)."
            } else {
                "2nd pass visual check disabled (pure text synthesis from pass 1 description)."
            };
            ui.set_image_status(msg.into());
            with_state(|state| {
                state.borrow_mut().image_ai = None;
                state.borrow_mut().two_ai = None;
            });
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_scene_body_shape_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = effective_scene_body_shape(&ui);
            let msg = if on {
                "Scene Body Shape enabled: Scene supplies body proportions & shape; Character keeps outfit, stature & face."
            } else {
                "Scene Body Shape disabled: Character supplies body proportions & shape."
            };
            ui.set_scene_ref_status(msg.into());
            {
                let mut st = state.borrow_mut();
                if let Some(ref mut sr) = st.scene_ref {
                    if !sr.raw_tags.is_empty() {
                        if sr.raw_tags.contains(" || ") {
                            let parts: Vec<&str> = sr.raw_tags.split(" || ").collect();
                            let char_slices: Vec<String> = parts[0].split(" | ").map(|s| s.to_string()).collect();
                            sr.actions = sceneref::from_split_with_shape(&char_slices, parts.get(1).unwrap_or(&""), false, on);
                        } else {
                            sr.actions = sceneref::from_single_with_shape(&sr.raw_tags, false, on);
                        }
                    }
                }
            }
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_dual_pass_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = ui.get_enable_dual_pass();
            let msg = if on {
                "Dual-Pass Vision enabled (Moondream 2-pass + WD14 reconciliation)."
            } else {
                "Dual-Pass Vision disabled (single-pass reading)."
            };
            ui.set_image_status(msg.into());
            let shape_on = effective_scene_body_shape(&ui);
            {
                let mut st = state.borrow_mut();
                st.image_ai = None;
                st.two_ai = None;
                if let Some(ref mut sr) = st.scene_ref {
                    if !sr.raw_tags.is_empty() {
                        if sr.raw_tags.contains(" || ") {
                            let parts: Vec<&str> = sr.raw_tags.split(" || ").collect();
                            let char_slices: Vec<String> = parts[0].split(" | ").map(|s| s.to_string()).collect();
                            sr.actions = sceneref::from_split_with_shape(&char_slices, parts.get(1).unwrap_or(&""), false, shape_on);
                        } else {
                            sr.actions = sceneref::from_single_with_shape(&sr.raw_tags, false, shape_on);
                        }
                    }
                }
            }
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_dual_pass_mode_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let mode_idx = ui.get_dual_pass_mode_index();
            let msg = if mode_idx == 0 {
                "Dual-Pass Mode: Scene + Body Shape (Mode A)"
            } else {
                "Dual-Pass Mode: Scene + Skeletal Pose (Mode B)"
            };
            ui.set_image_status(msg.into());
            let shape_on = effective_scene_body_shape(&ui);
            {
                let mut st = state.borrow_mut();
                st.image_ai = None;
                st.two_ai = None;
                if let Some(ref mut sr) = st.scene_ref {
                    if !sr.raw_tags.is_empty() {
                        if sr.raw_tags.contains(" || ") {
                            let parts: Vec<&str> = sr.raw_tags.split(" || ").collect();
                            let char_slices: Vec<String> = parts[0].split(" | ").map(|s| s.to_string()).collect();
                            sr.actions = sceneref::from_split_with_shape(&char_slices, parts.get(1).unwrap_or(&""), false, shape_on);
                        } else {
                            sr.actions = sceneref::from_single_with_shape(&sr.raw_tags, false, shape_on);
                        }
                    }
                }
            }
            for which in 0..tab_chars(&ui) {
                if state.borrow().char_images[which].is_some() {
                    apply_char_image(&ui, &state, which);
                }
            }
            apply_scene_ref(&ui, &state);
            analyze(&ui, &state);
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_llm_synthesis_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            save_settings(&ui);
            let on = ui.get_use_llm_synthesis();
            let msg = if on {
                "LLM Tag Synthesis enabled (Gemma 4 12B in Ollama)."
            } else {
                "LLM Tag Synthesis disabled (direct mechanical tag merge)."
            };
            ui.set_image_status(msg.into());
            let mut st = state.borrow_mut();
            st.image_ai = None;
            st.two_ai = None;
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_image_mode_changed(move || {
            if let Some(ui) = weak.upgrade() {
                image_extract(&ui, &state, false);
            }
        });
    }
    {
        let weak = ui.as_weak();
        let state = state.clone();
        ui.on_tag_threshold_changed(move || {
            let Some(ui) = weak.upgrade() else { return };
            if ui.get_app_tab() >= 1 {
                if !ui.get_image_use_ai() && state.borrow().two_tags.is_some() {
                    two_image_extract(&ui, &state, false);
                }
                return;
            }
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
            let two = state.borrow_mut().two_before_image.take();
            if let Some((boxes, scene)) = two {
                for (i, b) in boxes.iter().enumerate() {
                    set_char_box(&ui, i, b);
                }
                ui.set_scene_prompt(scene.into());
                ui.set_image_can_undo(false);
                analyze(&ui, &state);
                ui.set_status_message("Restored your previous prompts.".into());
                return;
            }
            let previous = state.borrow_mut().prompt_before_image.take();
            if let Some(text) = previous {
                ui.set_raw_prompt(text.into());
                ui.set_image_can_undo(false);
                analyze(&ui, &state);
                ui.set_status_message("Restored your previous prompt.".into());
            }
        });
    }

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
            ui.set_raw_prompt(prompt.into());
            ui.set_translate_open(false);
            analyze(&ui, &state);
            ui.set_status_message(
                if replace { "Before prompt replaced with the Danbooru tags." } else { "Danbooru tags added to the Before prompt." }.into(),
            );
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
                            Ok((ai_out, placement)) => {
                                ui.set_run_badge(format!("Last scan: {placement}").into());
                                let where_ran = format!("ran on {placement}");
                                if ai_key(&ui) != scanned_key {
                                    ui.set_status_message("Prompt changed during the AI scan — result discarded. Run it again.".into());
                                    return;
                                }
                                analyze(&ui, state);
                                let found = {
                                    let mut st = state.borrow_mut();
                                    let tokens = ai_engine::request_tokens(&req);
                                    let rules: Vec<PromptChange> =
                                        st.changes.iter().chain(st.others.iter().flat_map(|o| o.changes.iter())).cloned().collect();
                                    let cards = ai_engine::convert_ai_changes(&ai_out.changes, &tokens, &rules);
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
                                    st.ai_scene = SceneResult::from_ai(&ai_out, scan_config(&ui).checkpoint);
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

#[cfg(test)]
mod scene_panel_tests {
    use super::*;

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
