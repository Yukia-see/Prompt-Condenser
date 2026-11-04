//! Live, rule-based description of the scene a prompt describes (no AI).
//! Sorts each tag into who / looks / outfit / expression / pose / camera /
//! setting / light / style and turns that into a plain-English summary.

use crate::parser::{PromptToken, TokenCategory};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneDescription {
    /// One or two plain sentences.
    pub headline: String,
    /// "Label: tags" lines for the parts that are present.
    pub details: Vec<(String, String)>,
    /// Parts the prompt leaves unspecified (e.g. lighting), as a hint.
    pub missing: Vec<&'static str>,
}

#[derive(Default)]
struct Parts {
    subject: Vec<String>,
    traits: Vec<String>,
    hair: Vec<String>,
    eyes: Vec<String>,
    body: Vec<String>,
    outfit: Vec<String>,
    expression: Vec<String>,
    pose: Vec<String>,
    camera: Vec<String>,
    setting: Vec<String>,
    light: Vec<String>,
    style: Vec<String>,
    other: Vec<String>,
    loras: usize,
}

const SUBJECT: &[&str] = &[
    "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "1boy", "2boys", "3boys", "6+boys", "1other",
    "solo", "multiple girls", "multiple boys", "no humans", "scenery", "female", "male", "woman", "man", "girl",
    "boy", "couple", "group",
];
const TRAITS: &[&str] = &[
    "ears", "tail", "kitsune", "fox girl", "cat girl", "catgirl", "elf", "demon", "angel", "horns", "wings", "halo",
    "vampire", "succubus", "android", "robot", "furry", "kemonomimi", "pointy ears", "monster girl", "mermaid",
];
const EXPRESSION: &[&str] = &[
    "smile", "smirk", "grin", "blush", "open mouth", "closed mouth", "crying", "tears", "angry", "frown", "pout",
    "expressionless", "surprised", "happy", "sad", "laughing", "tongue out", "closed eyes", "one eye closed", "wink",
    "seductive", "embarrassed", "serious", "sleepy", "nervous", "ahegao", "parted lips", ":d", ":o", ":3",
];
const CAMERA: &[&str] = &[
    "upper body", "cowboy shot", "full body", "portrait", "close-up", "close up", "headshot", "lower body",
    "from above", "from below", "from side", "from behind", "back view", "front view", "side view", "dutch angle",
    "wide shot", "pov", "360 view", "character turnaround", "multiple views", "straight-on", "fisheye",
    "foreshortening", "profile", "three-quarter view", "depth of field", "bokeh", "blurry background",
];
const LIGHT: &[&str] = &[
    "lighting", "light", "sunlight", "moonlight", "sunset", "sunrise", "golden hour", "night", "day", "evening",
    "morning", "dusk", "dawn", "shadow", "glow", "neon", "backlighting", "rim light", "dark", "dim", "bright",
    "volumetric", "sun", "moon", "stars", "starry", "candlelight", "lens flare",
];
const SETTING: &[&str] = &[
    "background", "outdoors", "indoors", "room", "bedroom", "classroom", "kitchen", "office", "street", "city",
    "forest", "beach", "ocean", "sea", "sky", "shrine", "temple", "castle", "park", "garden", "field", "mountain",
    "river", "lake", "rooftop", "cafe", "library", "bathroom", "pool", "snow", "rain", "desert", "space", "cherry blossoms",
    "school", "train", "car", "bed", "window", "stage", "ruins", "village", "alley", "water", "flower", "tree",
];
const OUTFIT: &[&str] = &[
    "dress", "kimono", "yukata", "uniform", "shirt", "skirt", "bikini", "swimsuit", "armor", "jacket", "coat",
    "hoodie", "sweater", "hat", "gloves", "boots", "shoes", "thighhighs", "stockings", "pantyhose", "socks", "shorts",
    "pants", "jeans", "leotard", "bodysuit", "apron", "maid", "lingerie", "underwear", "panties", "bra", "nude",
    "naked", "topless", "cape", "scarf", "ribbon", "bow", "necklace", "earrings", "glasses", "mask", "crown",
    "headband", "hair ornament", "choker", "belt", "corset", "robe", "suit", "tie", "sleeves", "off shoulder",
    "collar", "veil", "tiara", "jewelry", "outfit", "clothes", "costume", "hairclip",
];
const STYLE: &[&str] = &[
    "anime", "realistic", "photorealistic", "photo", "watercolor", "sketch", "3d", "pixel art", "oil painting",
    "monochrome", "greyscale", "lineart", "chibi", "cel shading", "illustration", "painting", "comic", "manga",
    "cinematic", "2.5d", "concept art", "flat color", "retro", "vaporwave", "art style",
];
const POSE: &[&str] = &[
    "standing", "sitting", "lying", "kneeling", "squatting", "walking", "running", "jumping", "dancing", "holding",
    "reading", "eating", "drinking", "sleeping", "looking at viewer", "looking back", "looking away", "looking up",
    "looking down", "arms up", "arms behind", "hand on", "hands on", "crossed arms", "leaning", "stretching",
    "arched back", "spread", "on back", "on stomach", "fighting", "pointing", "waving", "peace sign", "hugging",
    "kissing", "pose", "posing", "bent over", "all fours", "wariza", "seiza",
];

fn has(text: &str, words: &[&str]) -> bool {
    words.iter().any(|w| text == *w || text.contains(w))
}

/// Matches whole words only (so "day" doesn't match "birthday").
fn has_word(text: &str, words: &[&str]) -> bool {
    words.iter().any(|w| {
        text == *w
            || text.split(|c: char| !c.is_alphanumeric() && c != '-').any(|part| part == *w)
            || (w.contains(' ') && text.contains(w))
    })
}

/// Where a tag belongs. Declaration order is the arranged-layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    Quality,
    Lora,
    Subject,
    Other,
    Traits,
    Hair,
    Eyes,
    Body,
    Outfit,
    Expression,
    Pose,
    Camera,
    Setting,
    Light,
    Style,
    Clause,
}

fn slot(t: &PromptToken) -> Slot {
    let n = t.normalized.as_str();
    match t.category {
        TokenCategory::QualityMeta => return Slot::Quality,
        TokenCategory::LoraOrSpecial => return Slot::Lora,
        _ => {}
    }
    if SUBJECT.contains(&n) {
        Slot::Subject
    } else if n.contains("hair") || has_word(n, &["ponytail", "twintails", "bangs", "braid", "bun", "ahoge", "sidelocks"]) {
        Slot::Hair
    } else if n.contains("eyes") && !has_word(n, &["closed eyes", "one eye closed"]) || n.contains("pupils") || n.contains("heterochromia") {
        Slot::Eyes
    } else if has(n, EXPRESSION) {
        Slot::Expression
    } else if has(n, CAMERA) {
        Slot::Camera
    } else if has(n, POSE) {
        Slot::Pose
    } else if has(n, OUTFIT) {
        Slot::Outfit
    } else if has(n, TRAITS) {
        Slot::Traits
    } else if has_word(n, LIGHT) {
        Slot::Light
    } else if has(n, SETTING) {
        Slot::Setting
    } else if has(n, STYLE) {
        Slot::Style
    } else if matches!(t.category, TokenCategory::CharacterBody | TokenCategory::PinnedEmphasis) {
        Slot::Body
    } else if t.category == TokenCategory::NaturalLanguageClause {
        Slot::Clause
    } else {
        Slot::Other
    }
}

fn classify(parts: &mut Parts, t: &PromptToken) {
    let text = t.text.trim().to_string();
    match slot(t) {
        Slot::Quality => {}
        Slot::Lora => {
            if t.normalized.starts_with("<lora") {
                parts.loras += 1;
            }
        }
        Slot::Subject => parts.subject.push(text),
        Slot::Hair => parts.hair.push(text),
        Slot::Eyes => parts.eyes.push(text),
        Slot::Expression => parts.expression.push(text),
        Slot::Camera => parts.camera.push(text),
        Slot::Pose => parts.pose.push(text),
        Slot::Outfit => parts.outfit.push(text),
        Slot::Traits => parts.traits.push(text),
        Slot::Light => parts.light.push(text),
        Slot::Setting => parts.setting.push(text),
        Slot::Style => parts.style.push(text),
        Slot::Body => parts.body.push(text),
        Slot::Clause | Slot::Other => parts.other.push(text),
    }
}

/// The arranged layout: one line per group, most important first (quality, LoRA,
/// who, looks, outfit, expression and pose, camera, setting and light, style,
/// sentences). Line breaks are plain whitespace to ComfyUI; only commas separate tags.
pub fn arrange(tokens: &[PromptToken], lora_block: &str, sentence: Option<&str>) -> String {
    let groups: [&[Slot]; 10] = [
        &[Slot::Quality],
        &[Slot::Lora],
        &[Slot::Subject, Slot::Other],
        &[Slot::Traits, Slot::Hair, Slot::Eyes, Slot::Body],
        &[Slot::Outfit],
        &[Slot::Expression, Slot::Pose],
        &[Slot::Camera],
        &[Slot::Setting, Slot::Light],
        &[Slot::Style],
        &[Slot::Clause],
    ];
    let slots: Vec<Slot> = tokens.iter().map(slot).collect();
    let mut lines: Vec<String> = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let mut items: Vec<String> = Vec::new();
        if i == 1 && !lora_block.trim().is_empty() {
            items.push(lora_block.trim().trim_matches(',').trim().to_string());
        }
        // within a group, keep the group's slot order, then the prompt's order
        for s in group.iter() {
            items.extend(tokens.iter().zip(&slots).filter(|(_, ts)| *ts == s).map(|(t, _)| t.raw.trim().to_string()));
        }
        if i == 9 {
            if let Some(sen) = sentence.map(str::trim).filter(|s| !s.is_empty()) {
                items.push(sen.to_string());
            }
        }
        if !items.is_empty() {
            lines.push(items.join(", "));
        }
    }
    lines.join(",\n")
}

/// A plain-English sentence about the prompt for natural-language checkpoints
/// (the rule-based stand-in when no Deep Scan has written one).
pub fn prompt_sentence(tokens: &[PromptToken]) -> String {
    let d = describe(tokens);
    // "(+2 more)" belongs in the scene card, not in a prompt
    let mut s = d.headline;
    while let (Some(a), Some(b)) = (s.find(" (+"), s.find(" more)")) {
        if b < a {
            break;
        }
        s.replace_range(a..b + " more)".len(), "");
    }
    s
}

fn subject_phrase(subject: &[String]) -> String {
    let lower: Vec<String> = subject.iter().map(|s| s.to_lowercase()).collect();
    let has = |s: &str| lower.iter().any(|x| x == s);
    let count = |suffix: &str| -> Option<String> {
        lower.iter().find(|x| x.ends_with(suffix) && x.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)).cloned()
    };
    if has("no humans") || (has("scenery") && count("girl").is_none() && count("boy").is_none()) {
        return "A scenery shot with no people".to_string();
    }
    let word = |n: &str, one: &str, many: &str| match n {
        "1" => format!("a {one}"),
        "2" => format!("two {many}"),
        "3" => format!("three {many}"),
        "4" => format!("four {many}"),
        "5" => format!("five {many}"),
        _ => format!("several {many}"),
    };
    let mut who = Vec::new();
    if let Some(g) = count("girls").or_else(|| count("girl")) {
        who.push(word(&g[..g.find(|c: char| !c.is_ascii_digit() && c != '+').unwrap_or(1)], "girl", "girls"));
    }
    if let Some(b) = count("boys").or_else(|| count("boy")) {
        who.push(word(&b[..b.find(|c: char| !c.is_ascii_digit() && c != '+').unwrap_or(1)], "boy", "boys"));
    }
    if who.is_empty() {
        if has("woman") || has("female") || has("girl") {
            who.push("a woman".into());
        } else if has("man") || has("male") || has("boy") {
            who.push("a man".into());
        }
    }
    if who.is_empty() {
        return "A scene".to_string();
    }
    let mut s = who.join(" and ");
    if has("solo") && who.len() == 1 {
        s = s.replacen("a ", "a lone ", 1);
    }
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

fn list(items: &[String], max: usize) -> String {
    let shown: Vec<&str> = items.iter().take(max).map(String::as_str).collect();
    let more = items.len().saturating_sub(max);
    let mut s = match shown.len() {
        0 => String::new(),
        1 => shown[0].to_string(),
        _ => format!("{} and {}", shown[..shown.len() - 1].join(", "), shown[shown.len() - 1]),
    };
    if more > 0 {
        s.push_str(&format!(" (+{more} more)"));
    }
    s
}

/// Describes the scene of an already-parsed prompt.
pub fn describe(tokens: &[PromptToken]) -> SceneDescription {
    let mut p = Parts::default();
    for t in tokens {
        classify(&mut p, t);
    }
    if tokens.iter().all(|t| t.category == TokenCategory::QualityMeta || t.category == TokenCategory::LoraOrSpecial) {
        return SceneDescription::default();
    }

    let scenery_only = p.subject.iter().any(|s| s.eq_ignore_ascii_case("no humans"));
    let mut sentence = subject_phrase(&p.subject);
    let looks: Vec<String> = p.traits.iter().chain(&p.hair).chain(&p.eyes).cloned().collect();
    if !looks.is_empty() && !scenery_only {
        sentence.push_str(&format!(" with {}", list(&looks, 4)));
    }
    if !p.outfit.is_empty() {
        sentence.push_str(&format!(", wearing {}", list(&p.outfit, 3)));
    }
    if !p.expression.is_empty() {
        sentence.push_str(&format!(", {}", list(&p.expression, 2)));
    }
    if !p.pose.is_empty() {
        sentence.push_str(&format!(", {}", list(&p.pose, 2)));
    }
    sentence.push('.');
    let mut second = Vec::new();
    if !p.camera.is_empty() {
        second.push(format!("Framed as {}", list(&p.camera, 2)));
    }
    if !p.setting.is_empty() {
        second.push(format!("at {}", list(&p.setting, 3)));
    }
    if !p.light.is_empty() {
        second.push(format!("with {}", list(&p.light, 2)));
    }
    if !p.style.is_empty() {
        second.push(format!("in a {} style", list(&p.style, 2)));
    }
    let headline = if second.is_empty() {
        sentence
    } else {
        let mut s = second.join(", ");
        if let Some(f) = s.get(..1) {
            s = f.to_uppercase() + &s[1..];
        }
        format!("{sentence} {s}.")
    };

    let mut details = Vec::new();
    let mut add = |label: &str, items: &Vec<String>| {
        if !items.is_empty() {
            details.push((label.to_string(), items.join(", ")));
        }
    };
    add("Who", &p.subject);
    add("Traits", &p.traits);
    add("Hair", &p.hair);
    add("Eyes", &p.eyes);
    add("Body", &p.body);
    add("Outfit", &p.outfit);
    add("Expression", &p.expression);
    add("Pose", &p.pose);
    add("Camera", &p.camera);
    add("Setting", &p.setting);
    add("Light / time", &p.light);
    add("Style", &p.style);
    add("Other", &p.other);
    if p.loras > 0 {
        details.push(("LoRAs".into(), p.loras.to_string()));
    }

    let mut missing = Vec::new();
    if p.subject.is_empty() {
        missing.push("who is in it");
    }
    if p.camera.is_empty() {
        missing.push("camera framing");
    }
    if p.setting.is_empty() {
        missing.push("setting / background");
    }
    if p.light.is_empty() {
        missing.push("lighting / time of day");
    }
    if p.expression.is_empty() && !scenery_only && !p.subject.is_empty() {
        missing.push("expression");
    }

    SceneDescription { headline, details, missing }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_prompt;

    #[test]
    fn arranges_tags_into_lines() {
        let t = parse_prompt(
            "moonlight, 1girl, (red eyes:1.2), masterpiece, kimono, smile, upper body, shrine, long white hair, solo, best quality, sitting, night",
        );
        let out = arrange(&t, "<lora:fox:0.8>, foxgirl", Some("A fox girl resting at a shrine."));
        assert_eq!(
            out,
            "masterpiece, best quality,\n<lora:fox:0.8>, foxgirl,\n1girl, solo,\nlong white hair, (red eyes:1.2),\nkimono,\n\
             smile, sitting,\nupper body,\nshrine, moonlight, night,\nA fox girl resting at a shrine."
        );
        // the layout is still the same prompt to the parser
        assert_eq!(parse_prompt(&out).len(), t.len() + 3);
        let s = prompt_sentence(&t);
        assert!(s.starts_with("A lone girl with") && !s.contains("more)"), "{s}");
    }

    #[test]
    fn describes_a_character_scene() {
        let d = describe(&parse_prompt(
            "masterpiece, 1girl, solo, kitsune, fox ears, long white hair, red eyes, kimono, smile, sitting, upper body, shrine, cherry blossoms, night, moonlight",
        ));
        assert_eq!(
            d.headline,
            "A lone girl with kitsune, fox ears, long white hair and red eyes, wearing kimono, smile, sitting. \
             Framed as upper body, at shrine and cherry blossoms, with night and moonlight."
        );
        assert!(d.missing.is_empty(), "{:?}", d.missing);
        assert!(d.details.iter().any(|(k, v)| k == "Outfit" && v == "kimono"));
    }

    #[test]
    fn scenery_and_gaps() {
        let d = describe(&parse_prompt("no humans, scenery, forest, river"));
        assert!(d.headline.starts_with("A scenery shot with no people."));
        assert!(d.missing.contains(&"lighting / time of day"));
        assert!(!d.missing.contains(&"expression"));
        assert_eq!(describe(&parse_prompt("masterpiece, best quality")), SceneDescription::default());
    }
}
