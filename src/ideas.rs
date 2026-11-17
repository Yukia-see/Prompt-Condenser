//! 💡 Ideas: context-aware suggestions for the parts a prompt leaves out or can enrich
//! (camera, setting, lighting, expression, pose, multi-character interaction, props & atmosphere).
//! Ranks real Danbooru tags by scene theme, generation stage, aspect ratio, and target view,
//! filters out conflicting or user-negated tags, asks the model for 3 distinct creative directions
//! with reasoning and protective negative guards, and falls back to curated offline ideas when AI is off.

use crate::analyzer::{self, ChangeAction, PromptChange, ScanConfig};
use crate::parser;
use crate::profiles::{AiMode, CheckpointProfile, GenerationStage, TargetView};
use crate::scene::{self, Slot};
use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Idea card ids start here (rule cards: 0 / `two::card_base`, AI cards: `ai_engine::AI_CARD_ID_BASE`).
pub const IDEA_CARD_ID_BASE: usize = 60_000;

/// A part the prompt leaves out (or can polish), with a context-ranked menu of real Danbooru tags.
#[derive(Debug, Clone)]
pub struct Gap {
    pub name: &'static str,
    pub menu: Vec<String>,
}

/// High-level visual theme detected from the prompt to rank Idea menus and guide generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneTheme {
    CyberpunkSciFi,
    FantasyMagic,
    TraditionalShrine,
    BeachSummer,
    CozyIndoor,
    SchoolSliceOfLife,
    DarkGothic,
    StudioPortrait,
    SceneryLandscape,
    General,
}

impl SceneTheme {
    pub fn label(&self) -> &'static str {
        match self {
            Self::CyberpunkSciFi => "cyberpunk / urban night",
            Self::FantasyMagic => "fantasy / adventure",
            Self::TraditionalShrine => "traditional / shrine",
            Self::BeachSummer => "beach / summer",
            Self::CozyIndoor => "cozy / indoor",
            Self::SchoolSliceOfLife => "school / slice of life",
            Self::DarkGothic => "dark / moody",
            Self::StudioPortrait => "portrait / studio",
            Self::SceneryLandscape => "scenery / landscape",
            Self::General => "general",
        }
    }
}

/// One creative option inside an Idea card (1-2 tags + why it fits + optional negative guards).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdeaOption {
    pub tags: Vec<String>,
    pub why: String,
    pub negative_guard: Vec<String>,
}

/// Rich idea suggestion for one prompt slot/category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdeaSuggestion {
    pub part: String,
    pub options: Vec<IdeaOption>,
}

const PARTS: &[(Slot, &str)] = &[
    (Slot::Camera, "camera framing"),
    (Slot::Setting, "setting / background"),
    (Slot::Light, "lighting / atmosphere"),
    (Slot::Expression, "expression"),
    (Slot::Pose, "pose"),
];

/// Verified real Danbooru tags used as fallback when `selected_tags.csv` is not on disk,
/// and as priority anchors when ranking menus.
const FALLBACK_CAMERA: &[&str] = &[
    "upper body", "cowboy shot", "full body", "portrait", "close-up", "wide shot",
    "from above", "from below", "from side", "from behind", "dutch angle",
    "straight-on", "foreshortening", "depth of field", "bokeh", "blurry background",
    "blurry foreground", "fisheye", "pov",
];

const FALLBACK_SETTING: &[&str] = &[
    "indoors", "outdoors", "bedroom", "classroom", "office", "library", "cafe",
    "kitchen", "hallway", "street", "city", "cityscape", "alley", "rooftop",
    "park", "garden", "forest", "bamboo forest", "beach", "ocean", "shrine",
    "ruins", "castle", "mountain", "river", "lake", "snow", "desert",
    "simple background", "white background", "gradient background", "dark background",
];

const FALLBACK_LIGHT: &[&str] = &[
    "sunlight", "dappled sunlight", "sunbeam", "backlighting", "moonlight",
    "sunset", "sunrise", "night", "twilight", "evening", "morning",
    "neon lights", "city lights", "light rays", "light particles", "lens flare",
    "glowing", "high contrast", "shadow", "bloom", "fog", "overcast",
];

const FALLBACK_EXPRESSION: &[&str] = &[
    "smile", "light smile", "grin", "smirk", "blush", "light blush",
    "parted lips", "open mouth", "closed mouth", "closed eyes", "one eye closed",
    "serious", "expressionless", "surprised", "sad", "tears", "embarrassed",
    "sleepy", "happy", "pout",
];

const FALLBACK_POSE: &[&str] = &[
    "standing", "sitting", "kneeling", "squatting", "lying",
    "leaning forward", "leaning back", "crossed legs", "crossed arms",
    "arms up", "arms behind back", "hand on own hip", "hands on own hips",
    "head tilt", "looking at viewer", "looking back", "looking to the side",
    "looking up", "looking down", "walking", "running", "contrapposto",
    "knee up", "outstretched arm",
];

const INTERACTION_MENU: &[&str] = &[
    "holding hands", "facing another", "back-to-back", "hug", "hug from behind",
    "looking at another", "eye contact", "cheek-to-cheek", "interlocked fingers",
    "group picture", "couple",
];

const MICRO_DETAIL_MENU: &[&str] = &[
    "parted lips", "eyelashes", "long eyelashes", "blush", "light blush",
    "sweat", "freckles", "sidelocks", "floating hair", "backlighting",
    "glowing eyes", "collarbone", "tears",
];

/// Detects the dominant visual theme of `prompt` to tailor Idea menus and LLM context.
pub fn detect_theme(prompt: &str) -> SceneTheme {
    let tokens = parser::parse_prompt(prompt);
    let norms: Vec<&str> = tokens.iter().map(|t| t.normalized.as_str()).collect();
    let has_any = |words: &[&str]| {
        norms.iter().any(|n| words.iter().any(|w| *n == *w || scene::has_word(n, &[w])))
    };
    let has_subject = norms.iter().any(|n| {
        *n == "solo"
            || crate::two::count_tag(n).is_some()
            || ["a girl", "a boy", "a woman", "a man", "two girls", "two boys", "two people", "three girls", "four girls"]
                .iter()
                .any(|p| parser::has_phrase(n, p))
    });

    if (has_any(&["no humans", "scenery"]) && !has_subject)
        || (has_any(&["mountain", "waterfall", "landscape"]) && !has_subject)
    {
        return SceneTheme::SceneryLandscape;
    }
    if has_any(&["cyberpunk", "neon", "neon lights", "city lights", "hologram", "sci-fi", "mecha", "android", "cyborg"]) {
        return SceneTheme::CyberpunkSciFi;
    }
    if has_any(&["shrine", "torii", "kimono", "yukata", "hakama", "miko", "katana", "samurai", "bamboo forest", "lantern"]) {
        return SceneTheme::TraditionalShrine;
    }
    if has_any(&["beach", "ocean", "bikini", "swimsuit", "pool", "palm tree", "surfboard"]) {
        return SceneTheme::BeachSummer;
    }
    if has_any(&["sword", "magic", "magic circle", "witch", "wizard", "elf", "knight", "armor", "wings", "dragon", "castle", "staff", "fantasy"]) {
        return SceneTheme::FantasyMagic;
    }
    if has_any(&["school uniform", "serafuku", "classroom", "school", "desk", "chalkboard", "blazer", "uwabaki"]) {
        return SceneTheme::SchoolSliceOfLife;
    }
    if has_any(&["bedroom", "bed", "cafe", "coffee", "tea", "teacup", "library", "couch", "pillow", "futon", "kitchen", "living room"]) {
        return SceneTheme::CozyIndoor;
    }
    if has_any(&["gothic", "vampire", "demon", "bat wings", "ruins", "graveyard", "skull", "blood"]) {
        return SceneTheme::DarkGothic;
    }
    if has_any(&["simple background", "white background", "grey background", "gradient background", "portrait", "close-up"]) {
        return SceneTheme::StudioPortrait;
    }
    SceneTheme::General
}

fn theme_priority_tags(theme: SceneTheme, slot_name: &str) -> &'static [&'static str] {
    match (theme, slot_name) {
        (SceneTheme::CyberpunkSciFi, "setting / background") => &[
            "city", "cityscape", "alley", "rooftop", "street", "dark background", "indoors",
        ],
        (SceneTheme::CyberpunkSciFi, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "neon lights", "city lights", "night", "backlighting", "chromatic aberration", "glowing", "high contrast", "fog", "rain",
        ],
        (SceneTheme::CyberpunkSciFi, "props & environmental storytelling") => &[
            "rain", "puddle", "lamppost", "chromatic aberration", "glowing", "fog", "smoke",
        ],
        (SceneTheme::FantasyMagic, "setting / background") => &[
            "forest", "ruins", "castle", "mountain", "outdoors", "shrine", "lake",
        ],
        (SceneTheme::FantasyMagic, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "light particles", "glowing", "sunbeam", "backlighting", "moonlight", "fog", "sunset", "bloom",
        ],
        (SceneTheme::FantasyMagic, "props & environmental storytelling") => &[
            "magic circle", "glowing", "light particles", "sparkle", "fog", "wind", "falling petals",
        ],
        (SceneTheme::TraditionalShrine, "setting / background") => &[
            "shrine", "bamboo forest", "garden", "forest", "outdoors", "park",
        ],
        (SceneTheme::TraditionalShrine, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "dappled sunlight", "sunset", "moonlight", "twilight", "backlighting", "sunbeam", "light rays",
        ],
        (SceneTheme::TraditionalShrine, "props & environmental storytelling") => &[
            "cherry blossoms", "falling petals", "petals", "lantern", "fog", "wind",
        ],
        (SceneTheme::BeachSummer, "setting / background") => &[
            "beach", "ocean", "outdoors", "sky",
        ],
        (SceneTheme::BeachSummer, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "sunlight", "sunset", "lens flare", "backlighting", "bloom", "bright",
        ],
        (SceneTheme::BeachSummer, "props & environmental storytelling") => &[
            "wet", "water drop", "wind", "sparkle", "sunlight",
        ],
        (SceneTheme::SchoolSliceOfLife, "setting / background") => &[
            "classroom", "hallway", "rooftop", "street", "park", "indoors",
        ],
        (SceneTheme::SchoolSliceOfLife, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "sunlight", "sunbeam", "light rays", "sunset", "dappled sunlight", "afternoon", "morning",
        ],
        (SceneTheme::CozyIndoor, "setting / background") => &[
            "bedroom", "cafe", "library", "living room", "indoors", "kitchen",
        ],
        (SceneTheme::CozyIndoor, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "sunlight", "light rays", "evening", "backlighting", "depth of field", "bokeh",
        ],
        (SceneTheme::CozyIndoor, "props & environmental storytelling") => &[
            "teacup", "coffee cup", "book", "steam", "dust", "lantern",
        ],
        (SceneTheme::DarkGothic, "setting / background") => &[
            "ruins", "castle", "dark background", "forest", "alley",
        ],
        (SceneTheme::DarkGothic, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "moonlight", "night", "high contrast", "shadow", "backlighting", "fog", "glowing",
        ],
        (SceneTheme::StudioPortrait, "setting / background") => &[
            "simple background", "gradient background", "white background", "dark background",
        ],
        (SceneTheme::StudioPortrait, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "backlighting", "high contrast", "shadow", "depth of field", "bokeh", "bloom",
        ],
        (SceneTheme::SceneryLandscape, "setting / background") => &[
            "mountain", "forest", "river", "lake", "ocean", "ruins", "shrine", "cityscape", "outdoors",
        ],
        (SceneTheme::SceneryLandscape, "lighting / atmosphere" | "atmosphere & lighting accents") => &[
            "sunset", "sunrise", "sunbeam", "light rays", "dappled sunlight", "fog", "overcast", "moonlight", "twilight",
        ],
        _ => &[],
    }
}

fn norm_key(t: &str) -> String {
    let trimmed = t.trim();
    let unwrapped = if trimmed.starts_with('(') && trimmed.ends_with(')') && trimmed.contains(':') {
        let inner = &trimmed[1..trimmed.len() - 1];
        inner.rsplit_once(':').and_then(|(b, w)| w.trim().parse::<f32>().ok().map(|_| b)).unwrap_or(trimmed)
    } else {
        trimmed
    };
    unwrapped
        .trim()
        .to_lowercase()
        .replace('_', " ")
        .replace("\\(", "(")
        .replace("\\)", ")")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn custom_neg_set(config: &ScanConfig) -> HashSet<String> {
    parser::parse_prompt(&config.custom_negative_raw)
        .into_iter()
        .map(|t| norm_key(&t.normalized))
        .collect()
}

fn is_sketch_or_mono(norms: &[&str]) -> bool {
    norms.iter().any(|n| matches!(*n, "monochrome" | "greyscale" | "grayscale" | "sketch" | "lineart"))
}

fn is_simple_bg(norms: &[&str]) -> bool {
    norms.iter().any(|n| matches!(*n, "simple background" | "white background" | "grey background" | "solid background" | "gradient background"))
}

/// Checks if proposing `candidate` would collide with the prompt, target view, stage, or custom negatives.
fn candidate_compatible(
    candidate: &str,
    in_prompt: &[String],
    config: &ScanConfig,
    custom_neg: &HashSet<String>,
    sketch: bool,
    simple_bg: bool,
) -> bool {
    let c = norm_key(candidate);
    if c.is_empty() || in_prompt.iter().any(|p| *p == c) || custom_neg.contains(&c) {
        return false;
    }
    if sketch && matches!(c.as_str(), "bokeh" | "lens flare" | "chromatic aberration" | "bloom" | "neon lights" | "blue sky") {
        return false;
    }
    if simple_bg && matches!(c.as_str(), "bokeh" | "blurry background" | "blurry foreground" | "depth of field" | "outdoors" | "forest" | "city") {
        return false;
    }
    // Stage restrictions
    match config.stage {
        GenerationStage::Detailer => {
            if matches!(
                c.as_str(),
                "full body" | "cowboy shot" | "wide shot" | "dutch angle" | "fisheye"
                    | "standing" | "walking" | "running" | "squatting" | "kneeling"
                    | "outdoors" | "indoors" | "beach" | "forest" | "city" | "classroom" | "bedroom"
            ) {
                return false;
            }
        }
        GenerationStage::Inpaint => {
            if matches!(c.as_str(), "full body" | "cowboy shot" | "upper body" | "wide shot") {
                return false;
            }
        }
        _ => {}
    }
    // Target view restrictions
    match config.target_view {
        TargetView::Turnaround360 => {
            if matches!(c.as_str(), "close-up" | "portrait" | "upper body" | "cowboy shot" | "dutch angle" | "depth of field" | "bokeh" | "blurry background" | "sitting") {
                return false;
            }
        }
        TargetView::UpperBody => {
            if matches!(c.as_str(), "full body" | "wide shot" | "walking" | "running" | "squatting" | "kneeling") {
                return false;
            }
        }
        TargetView::CloseUpPortrait => {
            if matches!(c.as_str(), "full body" | "cowboy shot" | "wide shot" | "standing" | "walking" | "running" | "sitting" | "kneeling" | "squatting" | "contrapposto") {
                return false;
            }
        }
        TargetView::FrontView => {
            if matches!(c.as_str(), "from behind" | "back view" | "looking back" | "from side") {
                return false;
            }
        }
        TargetView::FromBehind => {
            if matches!(c.as_str(), "front view" | "straight-on" | "facing viewer") {
                return false;
            }
        }
        TargetView::SideProfile => {
            if matches!(c.as_str(), "front view" | "from behind" | "straight-on") {
                return false;
            }
        }
        TargetView::FullBody => {
            if matches!(c.as_str(), "close-up" | "portrait" | "upper body") {
                return false;
            }
        }
        TargetView::None => {}
    }
    // Existing prompt tag conflicts
    for p in in_prompt {
        if analyzer::tags_conflict(p, &c) || analyzer::size_conflict(p, &c) {
            return false;
        }
        if analyzer::tags_posture_conflict(p, &c) {
            return false;
        }
        if p == "closed eyes" && matches!(c.as_str(), "looking at viewer" | "looking at another" | "glowing eyes" | "sparkling eyes") {
            return false;
        }
        if matches!(p.as_str(), "indoors" | "bedroom" | "classroom" | "office" | "library" | "kitchen") && matches!(c.as_str(), "outdoors" | "beach" | "forest" | "mountain") {
            return false;
        }
        if matches!(p.as_str(), "outdoors" | "beach" | "forest" | "mountain") && matches!(c.as_str(), "indoors" | "bedroom" | "classroom" | "office") {
            return false;
        }
        if matches!(p.as_str(), "night" | "midnight" | "moonlight" | "starry sky") && matches!(c.as_str(), "sunlight" | "morning" | "afternoon" | "sunrise") {
            return false;
        }
        if matches!(p.as_str(), "day" | "sunlight" | "morning" | "afternoon") && matches!(c.as_str(), "night" | "moonlight" | "starry sky") {
            return false;
        }
    }
    true
}

fn validate_against_danbooru(raw: &str) -> Option<String> {
    let w = norm_key(raw);
    if w.is_empty() {
        return None;
    }
    if let Some(list) = crate::danbooru::local() {
        list.get(&w)
            .filter(|t| !t.character && t.count >= 1_000)
            .map(|t| t.name.clone())
    } else {
        Some(w)
    }
}

fn build_ranked_menu(
    slot_name: &str,
    candidates: &[&str],
    theme: SceneTheme,
    in_prompt: &[String],
    config: &ScanConfig,
    custom_neg: &HashSet<String>,
    sketch: bool,
    simple_bg: bool,
    cap: usize,
) -> Vec<String> {
    let mut menu: Vec<String> = Vec::new();
    let priority = theme_priority_tags(theme, slot_name);
    let mut push_if_ok = |raw: &str| {
        if menu.len() >= cap {
            return;
        }
        let Some(tag) = validate_against_danbooru(raw) else { return };
        if candidate_compatible(&tag, in_prompt, config, custom_neg, sketch, simple_bg) && !menu.contains(&tag) {
            menu.push(tag);
        }
    };
    for &p in priority {
        push_if_ok(p);
    }
    // Aspect-ratio priority boosts for camera framing / composition
    if matches!(slot_name, "camera framing" | "camera & composition polish") {
        if config.aspect_ratio.is_widescreen() {
            for &t in &["wide shot", "depth of field", "blurry background", "blurry foreground"] {
                push_if_ok(t);
            }
        } else if config.aspect_ratio.is_portrait() {
            for &t in &["full body", "cowboy shot", "from below"] {
                push_if_ok(t);
            }
        }
    }
    for &w in candidates {
        push_if_ok(w);
    }
    menu
}

/// The parts the prompt has no tag for, each with its menu of real Danbooru tags (needs the tag list on disk;
/// without it the menus would be unchecked, so nothing is suggested).
pub fn gaps(prompt: &str) -> Vec<Gap> {
    let Some(list) = crate::danbooru::local() else { return Vec::new() };
    let tokens = parser::parse_prompt(prompt);
    let present: Vec<Slot> = tokens.iter().map(scene::slot).collect();
    // a scenery prompt has no expression or pose to fill: only a character count (1girl, 2boys, solo) says
    // someone is in it ("scenery" itself sorts as a subject word)
    let has_subject = tokens.iter().any(|t| t.normalized == "solo" || crate::two::count_tag(&t.normalized).is_some());
    PARTS
        .iter()
        .filter(|(slot, _)| !present.contains(slot))
        .filter(|(slot, _)| has_subject || !matches!(slot, Slot::Expression | Slot::Pose))
        .filter_map(|(slot, name)| {
            let mut menu: Vec<String> = Vec::new();
            for w in scene::slot_words(*slot) {
                let w = w.trim().replace("\\(", "(").replace("\\)", ")");
                if let Some(t) = list.get(&w).filter(|t| !t.character && t.count >= 1_000) {
                    if !menu.contains(&t.name) {
                        menu.push(t.name.clone());
                    }
                }
            }
            (!menu.is_empty()).then_some(Gap { name, menu })
        })
        .collect()
}

/// Context-aware gap detection: adapts menus to scene theme, generation stage, aspect ratio,
/// target view, multi-character interaction needs, and custom negatives. Also works offline
/// with curated fallback vocabularies when `selected_tags.csv` is not on disk.
pub fn contextual_gaps(prompt: &str, config: &ScanConfig) -> Vec<Gap> {
    let tokens = parser::parse_prompt(prompt);
    let norms: Vec<&str> = tokens.iter().map(|t| t.normalized.as_str()).collect();
    let in_prompt: Vec<String> = norms.iter().map(|s| norm_key(s)).collect();
    let present: Vec<Slot> = tokens.iter().map(scene::slot).collect();
    let theme = detect_theme(prompt);
    let sketch = is_sketch_or_mono(&norms);
    let simple_bg = is_simple_bg(&norms);
    let custom_neg = custom_neg_set(config);

    let has_subject = norms.iter().any(|n| {
        *n == "solo"
            || crate::two::count_tag(n).is_some()
            || ["a girl", "a boy", "a woman", "a man", "two girls", "two boys", "two people", "three girls", "four girls"]
                .iter()
                .any(|p| parser::has_phrase(n, p))
    });
    let is_multi = config.two_char
        || norms.iter().any(|n| crate::two::count_tag(n).is_some_and(|(c, _)| c >= 2))
        || norms.iter().any(|n| matches!(*n, "multiple girls" | "multiple boys" | "multiple others"));
    let has_interaction = norms.iter().any(|n| crate::two::is_interaction(n));

    let mut out = Vec::new();
    for (slot, name) in PARTS {
        if present.contains(slot) {
            continue;
        }
        if !has_subject && matches!(slot, Slot::Expression | Slot::Pose) {
            continue;
        }
        if config.stage == GenerationStage::Detailer && matches!(slot, Slot::Setting | Slot::Camera | Slot::Pose) {
            continue;
        }
        if config.stage == GenerationStage::Inpaint && *slot == Slot::Camera {
            continue;
        }
        if *slot == Slot::Camera && config.target_view != TargetView::None {
            continue;
        }

        let fallback: &[&str] = match slot {
            Slot::Camera => FALLBACK_CAMERA,
            Slot::Setting => FALLBACK_SETTING,
            Slot::Light => FALLBACK_LIGHT,
            Slot::Expression => FALLBACK_EXPRESSION,
            Slot::Pose => FALLBACK_POSE,
            _ => &[],
        };
        let slot_words = scene::slot_words(*slot);
        let mut combined: Vec<&str> = fallback.to_vec();
        for w in slot_words {
            if !combined.contains(&w) {
                combined.push(w);
            }
        }
        let menu = build_ranked_menu(name, &combined, theme, &in_prompt, config, &custom_neg, sketch, simple_bg, 28);
        if !menu.is_empty() {
            out.push(Gap { name, menu });
        }
    }

    // Multi-character interaction gap when several people lack an interaction tag
    if is_multi && !has_interaction && config.stage == GenerationStage::Base {
        let menu = build_ranked_menu(
            "character interaction",
            INTERACTION_MENU,
            theme,
            &in_prompt,
            config,
            &custom_neg,
            sketch,
            simple_bg,
            12,
        );
        if !menu.is_empty() {
            out.push(Gap { name: "character interaction", menu });
        }
    }

    // Detailer micro-detail gap when detailer stage skips broad slots
    if config.stage == GenerationStage::Detailer && has_subject {
        let menu = build_ranked_menu(
            "micro-detail & texture",
            MICRO_DETAIL_MENU,
            theme,
            &in_prompt,
            config,
            &custom_neg,
            sketch,
            simple_bg,
            14,
        );
        if !menu.is_empty() {
            out.push(Gap { name: "micro-detail & texture", menu });
        }
    }

    out
}

/// Secondary refinement menus when all 5 basic slots already have at least one tag,
/// offering atmosphere, composition, and detail accents not yet in `prompt`.
#[allow(dead_code)]
pub fn refinement_gaps(prompt: &str) -> Vec<Gap> {
    let Some(list) = crate::danbooru::local() else { return Vec::new() };
    let tokens = parser::parse_prompt(prompt);
    let in_prompt: Vec<String> = tokens.iter().map(|t| t.normalized.as_str().to_string()).collect();
    let has_subject = tokens.iter().any(|t| t.normalized == "solo" || crate::two::count_tag(&t.normalized).is_some());

    let mut categories: Vec<(&str, &[&str])> = vec![
        (
            "atmosphere & lighting accents",
            &[
                "rim lighting", "backlighting", "volumetric lighting", "sunbeam", "lens flare",
                "depth of field", "bokeh", "chromatic aberration", "bloom", "floating particles",
                "dust", "sparkles", "mist", "warm lighting", "dramatic shadows",
            ],
        ),
        (
            "camera & composition polish",
            &[
                "dutch angle", "from below", "from above", "close-up", "cowboy shot",
                "wide shot", "foreshortening", "dynamic angle", "rule of thirds", "blurry background",
                "blurry foreground", "cinematic composition",
            ],
        ),
    ];
    if has_subject {
        categories.push((
            "character & motion accents",
            &[
                "wind", "floating hair", "flowing clothes", "looking back", "head tilt",
                "hand on hip", "parted lips", "soft smile", "blush", "hair blowing",
                "collarbone", "wet", "sweat",
            ],
        ));
    }

    // this fallback only runs when the prompt already has camera framing: another shot or angle would fight it
    // (`full body` + `close-up`), so the composition menu keeps only effects
    const SHOTS: &[&str] = &["close-up", "cowboy shot", "wide shot", "from below", "from above", "upper body", "full body", "portrait", "dutch angle"];
    categories
        .into_iter()
        .filter_map(|(name, words)| {
            let mut menu: Vec<String> = Vec::new();
            for &w in words {
                if in_prompt.iter().any(|p| p == w) || SHOTS.contains(&w) {
                    continue;
                }
                if let Some(t) = list.get(w).filter(|t| !t.character && t.count >= 1_000) {
                    if !menu.contains(&t.name) {
                        menu.push(t.name.clone());
                    }
                }
            }
            (!menu.is_empty()).then_some(Gap { name, menu })
        })
        .collect()
}

/// Config-aware secondary refinement menus (atmosphere, composition polish, props & storytelling,
/// character & motion accents) when the basic slots are already filled.
pub fn contextual_refinement_gaps(prompt: &str, config: &ScanConfig) -> Vec<Gap> {
    let tokens = parser::parse_prompt(prompt);
    let norms: Vec<&str> = tokens.iter().map(|t| t.normalized.as_str()).collect();
    let in_prompt: Vec<String> = norms.iter().map(|s| norm_key(s)).collect();
    let theme = detect_theme(prompt);
    let sketch = is_sketch_or_mono(&norms);
    let simple_bg = is_simple_bg(&norms);
    let custom_neg = custom_neg_set(config);
    let has_subject = norms.iter().any(|n| {
        *n == "solo"
            || crate::two::count_tag(n).is_some()
            || ["a girl", "a boy", "a woman", "a man", "two girls", "two boys", "two people", "three girls", "four girls"]
                .iter()
                .any(|p| parser::has_phrase(n, p))
    });

    const SHOTS: &[&str] = &["close-up", "cowboy shot", "wide shot", "from below", "from above", "upper body", "full body", "portrait"];
    let has_shot = in_prompt.iter().any(|p| SHOTS.contains(&p.as_str())) || config.target_view != TargetView::None;
    let has_angle = in_prompt.iter().any(|p| matches!(p.as_str(), "dutch angle" | "from below" | "from above" | "from side" | "from behind" | "foreshortening"))
        || config.camera_angle > 0;

    let mut out = Vec::new();

    // 1. Atmosphere & lighting accents
    let atmo_candidates: &[&str] = if sketch {
        &["high contrast", "backlighting", "shadow", "hatching (texture)", "traditional media", "monochrome"]
    } else {
        &[
            "backlighting", "sunbeam", "dappled sunlight", "light rays", "light particles",
            "lens flare", "depth of field", "bokeh", "chromatic aberration", "bloom",
            "film grain", "high contrast", "fog", "glowing", "neon lights",
        ]
    };
    let atmo_menu = build_ranked_menu(
        "atmosphere & lighting accents",
        atmo_candidates,
        theme,
        &in_prompt,
        config,
        &custom_neg,
        sketch,
        simple_bg,
        14,
    );
    if !atmo_menu.is_empty() {
        out.push(Gap { name: "atmosphere & lighting accents", menu: atmo_menu });
    }

    // 2. Camera & composition polish (keeps angles when no angle yet, avoids conflicting shot distances)
    if config.stage != GenerationStage::Detailer && config.target_view != TargetView::Turnaround360 {
        let mut comp_candidates: Vec<&str> = vec!["depth of field", "blurry background", "blurry foreground", "bokeh", "foreshortening", "fisheye"];
        if !has_angle {
            comp_candidates.extend(["dutch angle", "from below", "from above"]);
        }
        if !has_shot {
            comp_candidates.extend(["cowboy shot", "upper body", "wide shot", "full body"]);
        }
        let comp_menu = build_ranked_menu(
            "camera & composition polish",
            &comp_candidates,
            theme,
            &in_prompt,
            config,
            &custom_neg,
            sketch,
            simple_bg,
            12,
        );
        if !comp_menu.is_empty() {
            out.push(Gap { name: "camera & composition polish", menu: comp_menu });
        }
    }

    // 3. Props & environmental storytelling
    if config.stage != GenerationStage::Detailer && !simple_bg {
        let prop_candidates: &[&str] = &[
            "falling petals", "petals", "cherry blossoms", "light particles", "dust", "sparkle",
            "fog", "rain", "puddle", "snow", "steam", "smoke", "lantern", "book",
            "teacup", "coffee cup", "umbrella", "magic circle", "wind", "lamppost",
        ];
        let prop_menu = build_ranked_menu(
            "props & environmental storytelling",
            prop_candidates,
            theme,
            &in_prompt,
            config,
            &custom_neg,
            sketch,
            simple_bg,
            14,
        );
        if !prop_menu.is_empty() {
            out.push(Gap { name: "props & environmental storytelling", menu: prop_menu });
        }
    }

    // 4. Character & motion accents
    if has_subject {
        let char_candidates: &[&str] = if config.stage == GenerationStage::Detailer {
            MICRO_DETAIL_MENU
        } else {
            &[
                "wind", "wind lift", "floating hair", "floating clothes", "looking back",
                "head tilt", "hand on own hip", "parted lips", "light smile", "light blush",
                "collarbone", "wet", "sweat", "eyelashes",
            ]
        };
        let char_menu = build_ranked_menu(
            "character & motion accents",
            char_candidates,
            theme,
            &in_prompt,
            config,
            &custom_neg,
            sketch,
            simple_bg,
            14,
        );
        if !char_menu.is_empty() {
            out.push(Gap { name: "character & motion accents", menu: char_menu });
        }
    }

    out
}

/// Asks the model for up to 3 options (1-2 tags each) per gap, from the menus only.
pub fn suggest(host: &str, model: &str, mode: AiMode, prompt: &str, gaps: &[Gap]) -> Result<Vec<(String, Vec<Vec<String>>)>> {
    let rich = suggest_rich(host, model, mode, prompt, gaps, &ScanConfig::default())?;
    Ok(rich
        .into_iter()
        .map(|s| (s.part, s.options.into_iter().map(|o| o.tags).collect()))
        .collect())
}

/// Rich, context-aware Idea generation: guides the LLM with checkpoint, theme, stage, aspect ratio,
/// and target view, requesting 3 distinct creative directions with concise reasoning and optional
/// negative-guard pairings. Falls back to curated offline ideas when `mode == AiMode::Disabled`.
pub fn suggest_rich(
    host: &str,
    model: &str,
    mode: AiMode,
    prompt: &str,
    gaps: &[Gap],
    config: &ScanConfig,
) -> Result<Vec<IdeaSuggestion>> {
    if gaps.is_empty() {
        return Ok(Vec::new());
    }
    if mode == AiMode::Disabled {
        return Ok(offline_suggest(prompt, gaps, config));
    }
    let base = crate::ai_engine::base_url(host)?;
    let theme = detect_theme(prompt);
    let menus: String = gaps.iter().map(|g| format!("- {}: {}\n", g.name, g.menu.join(", "))).collect();

    let mut ctx_notes = vec![
        format!("Checkpoint: {}", config.checkpoint.display_name()),
        format!("Detected theme: {}", theme.label()),
        format!("Stage: {}", config.stage.display_name()),
        format!("Aspect: {}", config.aspect_ratio.display_name()),
    ];
    if config.target_view != TargetView::None {
        ctx_notes.push(format!("Target framing: {}", config.target_view.display_name()));
    }
    if config.two_char {
        ctx_notes.push(format!("Multi-character scene ({} characters)", config.char_count()));
    }
    if !config.custom_negative_raw.trim().is_empty() {
        ctx_notes.push(format!("Avoid user's negative tags: {}", config.custom_negative_raw.trim()));
    }

    let system = "You are an expert AI image-prompt art director. For each missing or polish part of the prompt, \
        propose 3 genuinely distinct creative directions that elevate the scene's mood, storytelling, and visual \
        clarity (e.g., Option 1 = natural/classic fit, Option 2 = dramatic/cinematic contrast, Option 3 = atmospheric/expressive). \
        Each option uses 1 or 2 tags chosen ONLY from that part's menu, spelled exactly as in the menu. \
        Include a concise 'why' (5-12 words explaining the visual synergy) and an optional 'negative_guard' \
        (0-2 concrete Danbooru tags naming what to keep out, never 'no X', never contradicting the prompt or chosen tags). \
        Never repeat or contradict tags already in the prompt. Reply with JSON only.";

    let user = format!(
        "Context: {}\n\nPrompt: {prompt}\n\nParts to suggest and their allowed menus:\n{menus}",
        ctx_notes.join(" | ")
    );

    let schema = json!({
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "part": { "type": "string" },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tags": { "type": "array", "items": { "type": "string" } },
                                    "why": { "type": "string" },
                                    "negative_guard": { "type": "array", "items": { "type": "string" } }
                                },
                                "required": ["tags", "why"]
                            }
                        }
                    },
                    "required": ["part", "options"]
                }
            }
        },
        "required": ["suggestions"]
    });

    let body = json!({
        "model": model,
        "system": system,
        "prompt": user,
        "format": schema,
        "stream": false,
        "think": false,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.65,
            "num_predict": 900,
            "num_gpu": if mode == AiMode::CpuOnly { Some(0) } else { None::<i32> }
        },
    });
    let http = crate::ai_engine::client(Some(Duration::from_secs(180)))?;
    let mut resp = http
        .post(format!("{base}/api/generate"))
        .json(&body)
        .send()
        .map_err(|_| anyhow!("Couldn't reach Ollama at {base}"))?;
    if resp.status().as_u16() == 400 {
        let mut b = body.clone();
        b.as_object_mut().expect("object").remove("think");
        resp = http
            .post(format!("{base}/api/generate"))
            .json(&b)
            .send()
            .map_err(|_| anyhow!("Couldn't reach Ollama at {base}"))?;
    }
    if !resp.status().is_success() {
        return Err(anyhow!("Ollama answered {} for the idea request", resp.status()));
    }
    let v: serde_json::Value = resp.json()?;
    let raw = v["response"].as_str().unwrap_or("");
    let reply: serde_json::Value = parse_ideas_json(raw)
        .ok_or_else(|| anyhow!("The model's reply was unreadable. Try again."))?;
    Ok(validate_rich(&reply, prompt, gaps, config))
}

/// Parses the LLM's Ideas JSON, salvaging complete suggestion objects if output truncated at `num_predict`.
fn parse_ideas_json(raw: &str) -> Option<serde_json::Value> {
    let start = raw.find('{')?;
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw[start..]) {
        return Some(v);
    }
    let body = &raw[start..];
    let mut items: Vec<serde_json::Value> = Vec::new();
    let (mut depth, mut in_str, mut escaped, mut obj_start) = (0i32, false, false, None);
    for (i, c) in body.char_indices() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => {
                depth += 1;
                if depth == 2 {
                    obj_start = Some(i);
                }
            }
            '}' => {
                if depth == 2 {
                    if let Some(st) = obj_start.take() {
                        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&body[st..=i]) {
                            if obj.get("part").is_some() && obj.get("options").is_some() {
                                items.push(obj);
                            }
                        }
                    }
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    (!items.is_empty()).then(|| json!({ "suggestions": items }))
}

/// Deterministic, context-ranked offline Ideas when AI mode is turned off.
pub fn offline_suggest(prompt: &str, gaps: &[Gap], config: &ScanConfig) -> Vec<IdeaSuggestion> {
    let theme = detect_theme(prompt);
    let mut out = Vec::new();
    for gap in gaps {
        if gap.menu.is_empty() {
            continue;
        }
        let mut options = Vec::new();
        // Build up to 3 distinct 1-2 tag options from the already context-ranked menu
        let mut idx = 0usize;
        while idx < gap.menu.len() && options.len() < 3 {
            let first = gap.menu[idx].clone();
            let mut tags = vec![first.clone()];
            // Pair complementary non-conflicting tag if available (never pair mutually exclusive shots/poses)
            let is_camera_gap = gap.name == "camera framing";
            if let Some(second) = gap.menu.get(idx + 1) {
                if !is_camera_gap && !analyzer::recommendation_pair_conflict(&first, second) {
                    tags.push(second.clone());
                    idx += 2;
                } else {
                    idx += 1;
                }
            } else {
                idx += 1;
            }
            let neg = default_negative_guard_for_tags(&tags, config);
            options.push(IdeaOption {
                why: format!("Curated {} fit for {}", gap.name, theme.label()),
                tags,
                negative_guard: neg,
            });
        }
        if !options.is_empty() {
            out.push(IdeaSuggestion { part: gap.name.to_string(), options });
        }
    }
    out
}

fn default_negative_guard_for_tags(tags: &[String], config: &ScanConfig) -> Vec<String> {
    if config.checkpoint == CheckpointProfile::Flux {
        return Vec::new();
    }
    let mut out = Vec::new();
    for t in tags {
        let g: &[&str] = match t.as_str() {
            "backlighting" | "rim lighting" => &["flat lighting"],
            "night" | "moonlight" | "neon lights" => &["daytime", "sunlight"],
            "sunlight" | "dappled sunlight" => &["night"],
            "simple background" | "white background" => &["cluttered background"],
            "upper body" | "cowboy shot" => &["full body"],
            "full body" => &["cropped", "out of frame"],
            "close-up" | "portrait" => &["wide shot", "full body"],
            _ => &[],
        };
        for &n in g {
            if !out.iter().any(|x| x == n) {
                out.push(n.to_string());
            }
        }
    }
    out.truncate(2);
    out
}

/// Resolves a model-returned tag to its exact menu spelling if possible (exact match,
/// plural/dash fix via `danbooru::local()`, or unique whole-word match in `menu`).
fn resolve_menu_tag(raw: &str, menu: &[String]) -> Option<String> {
    let k = norm_key(raw);
    if k.is_empty() {
        return None;
    }
    if let Some(exact) = menu.iter().find(|m| norm_key(m) == k) {
        return Some(exact.clone());
    }
    if let Some(list) = crate::danbooru::local() {
        if let Some(fixed) = list.fix(&k) {
            if let Some(m) = menu.iter().find(|m| norm_key(m) == fixed.name) {
                return Some(m.clone());
            }
        }
    }
    None
}

/// Keeps only menu tags that aren't in the prompt yet, at most 3 options per gap, in the gaps' order.
pub fn validate(reply: &serde_json::Value, prompt: &str, gaps: &[Gap]) -> Vec<(String, Vec<Vec<String>>)> {
    validate_rich_inner(reply, prompt, gaps, &ScanConfig::default(), false)
        .into_iter()
        .map(|s| (s.part, s.options.into_iter().map(|o| o.tags).collect()))
        .collect()
}

/// Rich validation: handles both structured `{tags, why, negative_guard}` options and legacy string
/// options, normalizes near-misses onto the allowed menu, rejects conflicting/negated tags, and
/// cleans negative guards.
pub fn validate_rich(
    reply: &serde_json::Value,
    prompt: &str,
    gaps: &[Gap],
    config: &ScanConfig,
) -> Vec<IdeaSuggestion> {
    validate_rich_inner(reply, prompt, gaps, config, true)
}

fn validate_rich_inner(
    reply: &serde_json::Value,
    prompt: &str,
    gaps: &[Gap],
    config: &ScanConfig,
    check_conflicts: bool,
) -> Vec<IdeaSuggestion> {
    let tokens = parser::parse_prompt(prompt);
    let in_prompt: Vec<String> = tokens.iter().map(|t| norm_key(&t.normalized)).collect();
    let custom_neg = custom_neg_set(config);
    let sketch = is_sketch_or_mono(&in_prompt.iter().map(String::as_str).collect::<Vec<_>>());
    let simple_bg = is_simple_bg(&in_prompt.iter().map(String::as_str).collect::<Vec<_>>());
    let items: Vec<&serde_json::Value> = reply["suggestions"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    let mut out = Vec::new();

    for (gi, gap) in gaps.iter().enumerate() {
        let item = items
            .iter()
            .find(|it| {
                it["part"]
                    .as_str()
                    .is_some_and(|p| norm_key(p) == norm_key(gap.name) || gap.name.starts_with(&norm_key(p)))
            })
            .or_else(|| items.get(gi))
            .copied();
        let Some(item) = item else { continue };
        let mut options: Vec<IdeaOption> = Vec::new();
        let mut used: Vec<String> = Vec::new();

        for opt in item["options"].as_array().into_iter().flatten() {
            let (raw_tags, why, raw_negs): (Vec<String>, String, Vec<String>) = if let Some(s) = opt.as_str() {
                (s.split(',').map(|x| x.to_string()).collect(), String::new(), Vec::new())
            } else if opt.is_object() {
                let tags = opt["tags"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                let why = opt["why"].as_str().unwrap_or("").trim().to_string();
                let negs = opt["negative_guard"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                (tags, why, negs)
            } else {
                continue;
            };

            let mut tags: Vec<String> = Vec::new();
            for rt in raw_tags {
                for piece in rt.split(',') {
                    let Some(resolved) = resolve_menu_tag(piece, &gap.menu) else { continue };
                    let rk = norm_key(&resolved);
                    if in_prompt.contains(&rk)
                        || used.contains(&rk)
                        || tags.iter().any(|x| norm_key(x) == rk)
                        || (check_conflicts && !candidate_compatible(&rk, &in_prompt, config, &custom_neg, sketch, simple_bg))
                        || tags.iter().any(|prev| analyzer::tags_conflict(prev, &rk) || analyzer::tags_posture_conflict(prev, &rk))
                    {
                        continue;
                    }
                    tags.push(resolved);
                    if tags.len() == 2 {
                        break;
                    }
                }
                if tags.len() == 2 {
                    break;
                }
            }
            if tags.is_empty() {
                continue;
            }
            for t in &tags {
                used.push(norm_key(t));
            }
            let neg_clean: Vec<String> = raw_negs
                .into_iter()
                .map(|n| crate::ai_engine::clean_negative(&n))
                .flat_map(|n| n.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
                .filter(|n| {
                    let nk = norm_key(n);
                    !nk.is_empty()
                        && !in_prompt.contains(&nk)
                        && !tags.iter().any(|t| norm_key(t) == nk)
                })
                .take(2)
                .collect();

            options.push(IdeaOption { tags, why, negative_guard: neg_clean });
            if options.len() == 3 {
                break;
            }
        }
        if !options.is_empty() {
            out.push(IdeaSuggestion { part: gap.name.to_string(), options });
        }
    }
    out
}

/// One opt-in card per gap: every option's tags as chips (the caller switches off all but the first option).
pub fn cards(suggestions: &[(String, Vec<Vec<String>>)]) -> Vec<PromptChange> {
    let rich: Vec<IdeaSuggestion> = suggestions
        .iter()
        .map(|(part, opts)| IdeaSuggestion {
            part: part.clone(),
            options: opts
                .iter()
                .map(|t| IdeaOption { tags: t.clone(), why: String::new(), negative_guard: Vec::new() })
                .collect(),
        })
        .collect();
    rich_cards(&rich).0
}

/// Returns per-option `(option_tags, negative_guard)` pairs keyed by Idea card id so the UI
/// only activates negative guards for options whose chips remain switched ON.
pub fn rich_card_option_negatives(suggestions: &[IdeaSuggestion]) -> HashMap<usize, Vec<(Vec<String>, Vec<String>)>> {
    suggestions
        .iter()
        .enumerate()
        .filter_map(|(i, sug)| {
            let pairs: Vec<(Vec<String>, Vec<String>)> = sug
                .options
                .iter()
                .filter(|o| !o.negative_guard.is_empty())
                .map(|o| (o.tags.clone(), o.negative_guard.clone()))
                .collect();
            (!pairs.is_empty()).then_some((IDEA_CARD_ID_BASE + i, pairs))
        })
        .collect()
}

/// Converts rich `IdeaSuggestion`s into opt-in `ChangeAction::Idea` cards plus any associated
/// negative-guard tags (`card_id -> comma-separated negative tags`) applied when the card is accepted.
pub fn rich_cards(suggestions: &[IdeaSuggestion]) -> (Vec<PromptChange>, HashMap<usize, String>) {
    let mut cards = Vec::new();
    let mut negatives = HashMap::new();
    for (i, sug) in suggestions.iter().enumerate() {
        let id = IDEA_CARD_ID_BASE + i;
        let opts_summary: Vec<String> = sug
            .options
            .iter()
            .map(|o| {
                if o.why.is_empty() {
                    o.tags.join(" + ")
                } else {
                    format!("{} ({})", o.tags.join(" + "), o.why)
                }
            })
            .collect();
        let all_negs: Vec<String> = sug
            .options
            .iter()
            .flat_map(|o| o.negative_guard.iter().cloned())
            .collect::<Vec<_>>();
        let mut uniq_negs = Vec::new();
        for n in all_negs {
            if !uniq_negs.contains(&n) {
                uniq_negs.push(n);
            }
        }
        let neg_note = if uniq_negs.is_empty() {
            String::new()
        } else {
            let joined = uniq_negs.join(", ");
            negatives.insert(id, joined.clone());
            format!("\n• Negative guards on Add It: {joined}")
        };
        cards.push(PromptChange {
            id,
            original: String::new(),
            action: ChangeAction::Idea,
            reason: format!(
                "Idea for {}: {}{}\nPick the tags you want, then Add It.",
                sug.part,
                opts_summary.join(" · "),
                neg_note
            ),
            suggested_alternative: sug
                .options
                .iter()
                .flat_map(|o| o.tags.iter().cloned())
                .collect::<Vec<_>>()
                .join(", "),
            framing_alternative: String::new(),
            mode: 2,
        });
    }
    (cards, negatives)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replies_keep_only_menu_tags_not_in_the_prompt() {
        let gaps = vec![
            Gap { name: "lighting / atmosphere", menu: vec!["sunset".into(), "backlighting".into(), "moonlight".into()] },
            Gap { name: "pose", menu: vec!["sitting".into(), "standing".into()] },
        ];
        let reply = json!({ "suggestions": [
            { "part": "lighting / atmosphere", "options": ["sunset, backlighting", "golden hour", "moonlight"] },
            { "part": "pose", "options": ["sitting", "standing", "leaning"] },
        ]});
        let out = validate(&reply, "1girl, standing", &gaps);
        assert_eq!(out[0].1, vec![vec!["sunset".to_string(), "backlighting".to_string()], vec!["moonlight".to_string()]]);
        // "standing" is already in the prompt, "leaning" isn't on the menu
        assert_eq!(out[1].1, vec![vec!["sitting".to_string()]]);
        let c = cards(&out);
        assert_eq!(c[0].suggested_alternative, "sunset, backlighting, moonlight");
        assert!(c.iter().all(|c| c.mode == 2 && c.action == ChangeAction::Idea));
    }

    #[test]
    fn rich_replies_validate_reasons_and_negative_guards_and_reject_conflicts() {
        let gaps = vec![
            Gap {
                name: "lighting / atmosphere",
                menu: vec!["neon lights".into(), "backlighting".into(), "moonlight".into(), "sunlight".into()],
            },
            Gap {
                name: "pose",
                menu: vec!["standing".into(), "sitting".into(), "looking at viewer".into()],
            },
        ];
        let reply = json!({ "suggestions": [
            {
                "part": "lighting / atmosphere",
                "options": [
                    { "tags": ["neon lights", "backlighting"], "why": "Crisp cyberpunk rim glow", "negative_guard": ["daytime", "no shadows", "night"] },
                    { "tags": ["sunlight"], "why": "Should be rejected by night prompt", "negative_guard": [] }
                ]
            },
            {
                "part": "pose",
                "options": [
                    { "tags": ["looking at viewer"], "why": "Should be rejected by closed eyes", "negative_guard": [] },
                    { "tags": ["standing"], "why": "Should be rejected by sitting", "negative_guard": [] },
                    { "tags": ["sitting"], "why": "Already in prompt", "negative_guard": [] }
                ]
            }
        ]});
        let mut cfg = ScanConfig::default();
        cfg.custom_negative_raw = "moonlight".into();
        let out = validate_rich(&reply, "1girl, night, sitting, closed eyes", &gaps, &cfg);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].options[0].tags, vec!["neon lights", "backlighting"]);
        assert_eq!(out[0].options[0].why, "Crisp cyberpunk rim glow");
        // "no shadows" cleaned out, "night" already in prompt -> only "daytime" survives
        assert_eq!(out[0].options[0].negative_guard, vec!["daytime"]);
        let (cards, negs) = rich_cards(&out);
        assert_eq!(negs.get(&cards[0].id).map(String::as_str), Some("daytime"));
        assert!(cards[0].reason.contains("Crisp cyberpunk rim glow"));
    }

    #[test]
    fn contextual_gaps_adapt_to_theme_stage_view_and_multi_char() {
        let mut cfg = ScanConfig::default();
        let gaps = contextual_gaps("1girl, cyberpunk, cityscape, raincoat", &cfg);
        let light = gaps.iter().find(|g| g.name == "lighting / atmosphere").expect("light gap");
        assert!(light.menu.iter().take(4).any(|t| t == "neon lights" || t == "city lights" || t == "night"), "{:?}", light.menu);

        // Detailer skips setting and wide camera framing, offers micro-detail
        cfg.stage = GenerationStage::Detailer;
        let det_gaps = contextual_gaps("1girl, smile", &cfg);
        assert!(!det_gaps.iter().any(|g| g.name == "setting / background" || g.name == "camera framing"));
        assert!(det_gaps.iter().any(|g| g.name == "micro-detail & texture"));

        // Multi-character without interaction offers interaction gap
        cfg.stage = GenerationStage::Base;
        cfg.two_char = true;
        let multi_gaps = contextual_gaps("2girls, red hair, blue hair", &cfg);
        assert!(multi_gaps.iter().any(|g| g.name == "character interaction"));

        // Offline fallback generates curated suggestions even when AI is disabled
        let off = offline_suggest("1girl, beach, bikini", &contextual_gaps("1girl, beach, bikini", &ScanConfig::default()), &ScanConfig::default());
        assert!(!off.is_empty());
        assert!(off.iter().all(|s| !s.options.is_empty()));
    }

    #[test]
    fn gaps_follow_what_the_prompt_already_has() {
        if crate::danbooru::local().is_none() {
            return; // needs the tag list on disk
        }
        let names: Vec<&str> = gaps("1girl, smile, standing, upper body").iter().map(|g| g.name).collect();
        assert_eq!(names, vec!["setting / background", "lighting / atmosphere"]);
        // scenery: no expression or pose
        let names: Vec<&str> = gaps("scenery, mountain").iter().map(|g| g.name).collect();
        assert!(!names.contains(&"pose") && !names.contains(&"expression"), "{names:?}");
        for g in gaps("1girl") {
            assert!(!g.menu.is_empty() && g.menu.iter().all(|t| crate::danbooru::local().unwrap().get(t).is_some()), "{g:?}");
        }
    }

    #[test]
    fn offline_camera_never_pairs_conflicting_shots_and_truncated_json_salvages() {
        let cfg = ScanConfig::default();
        let gaps = contextual_gaps("1girl, solo", &cfg);
        let off = offline_suggest("1girl, solo", &gaps, &cfg);
        if let Some(cam) = off.iter().find(|s| s.part == "camera framing") {
            for opt in &cam.options {
                for i in 0..opt.tags.len() {
                    for j in (i + 1)..opt.tags.len() {
                        assert!(!analyzer::recommendation_pair_conflict(&opt.tags[i], &opt.tags[j]), "{:?}", opt.tags);
                    }
                }
            }
        }
        // Flux prose subjects ("a girl") unlock expression & pose gaps
        let mut flux_cfg = ScanConfig::default();
        flux_cfg.checkpoint = CheckpointProfile::Flux;
        let flux_gaps = contextual_gaps("A girl with red hair, wearing a white dress", &flux_cfg);
        assert!(flux_gaps.iter().any(|g| g.name == "expression"));
        assert!(flux_gaps.iter().any(|g| g.name == "pose"));

        // Truncated JSON salvage keeps completed suggestion blocks
        let truncated = r#"{"suggestions":[{"part":"lighting / atmosphere","options":[{"tags":["backlighting"],"why":"Crisp rim glow"}]},{"part":"pose","options":[{"tags":["sit"#;
        let salvaged = parse_ideas_json(truncated).expect("should salvage first completed suggestion");
        assert_eq!(salvaged["suggestions"].as_array().unwrap().len(), 1);
    }
}
