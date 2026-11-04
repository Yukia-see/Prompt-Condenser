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
// Matched as whole words (`has_word`): "elf" isn't in "bookshelf", "cape" isn't in
// "landscape", "bra" isn't in "library", "bow" isn't in "rainbow".
const TRAITS: &[&str] = &[
    "ears", "ear", "tail", "tails", "kitsune", "fox girl", "cat girl", "catgirl", "elf", "demon", "angel", "horns",
    "horn", "wings", "wing", "halo", "vampire", "succubus", "android", "robot", "furry", "kemonomimi", "pointy ears",
    "monster girl", "mermaid", "fang", "fangs", "freckles", "mole", "tattoo", "scar",
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
    // any "… shot" or "… angle" (medium shot, low angle)
    " shot", " angle",
];
const LIGHT: &[&str] = &[
    "lighting", "light", "sunlight", "moonlight", "sunset", "sunrise", "golden hour", "night", "day", "evening",
    "morning", "dusk", "dawn", "shadow", "glow", "neon", "backlighting", "rim light", "dark", "dim", "bright",
    "volumetric", "sun", "moon", "stars", "starry", "candlelight", "lens flare", "shadows", "sunbeam", "sunbeams",
    "twilight", "afternoon", "midnight", "lamplight", "firelight", "backlight", "spotlight", "warm lighting",
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
    "collar", "veil", "tiara", "jewelry", "outfit", "clothes", "costume", "hairclip", "obi", "sash", "cardigan",
    "vest", "blouse", "kneehighs", "heels", "sandals", "bracelet", "scrunchie", "boot", "shoe", "glove", "sock",
    "earring", "sleeve", "sleeveless", "necktie", "bowtie", "serafuku", "sailor collar", "beret", "headwear",
    "footwear", "mary janes", "loafers", "uwabaki", "miniskirt", "blazer", "camisole", "leggings", "pantsu",
    "swimwear", "neckerchief", "hairband", "hairpin", "hair bow", "hair ribbon", "piercing", "garter", "garter belt",
    "fishnets", "overalls", "tank top", "crop top", "turtleneck", "gown", "uniform", "qipao", "cheongsam", "hakama",
];
/// What a character *is* (species, role): the noun of `character_clause`.
const IDENTITY: &[&str] = &[
    "kitsune", "fox girl", "cat girl", "catgirl", "wolf girl", "dog girl", "bunny girl", "dragon girl", "elf",
    "dark elf", "demon", "demon girl", "angel", "vampire", "succubus", "android", "robot", "monster girl", "mermaid",
    "witch", "maid", "knight", "nun", "miko", "ninja", "idol", "princess", "queen", "warrior", "policewoman",
    "nurse", "schoolgirl", "gyaru", "kogal", "magical girl", "pirate", "samurai", "cyborg", "ghost", "oni",
];
const STYLE: &[&str] = &[
    "anime", "realistic", "photorealistic", "photo", "watercolor", "sketch", "3d", "pixel art", "oil painting",
    "monochrome", "greyscale", "lineart", "chibi", "cel shading", "illustration", "painting", "comic", "manga",
    "cinematic", "2.5d", "concept art", "flat color", "retro", "vaporwave", "art style", " style", "fantasy",
    // mood
    "atmosphere", "romantic", "melancholic", "peaceful", "dramatic", "serene", "cozy", "mysterious", "ethereal",
    "whimsical", "nostalgic", "eerie", "epic", "moody",
];
const POSE: &[&str] = &[
    "standing", "sitting", "lying", "kneeling", "squatting", "walking", "running", "jumping", "dancing", "holding",
    "reading", "eating", "drinking", "sleeping", "looking at viewer", "looking back", "looking away", "looking up",
    "looking down", "arms up", "arms behind", "hand on", "hands on", "crossed arms", "leaning", "stretching",
    "arched back", "spread", "on back", "on stomach", "fighting", "pointing", "waving", "peace sign", "hugging",
    "kissing", "pose", "posing", "bent over", "all fours", "wariza", "seiza",
    // Hand gestures, arm poses, and bodily actions
    "gesture", "gesturing", "hand gesture", "hand gestures", "thumbs up", "thumbs-up", "heart hands", "finger heart",
    "v sign", "salute", "saluting", "shushing", "finger to mouth", "hand to mouth", "hand to face", "hand on chest",
    "hand on hip", "hands on hips", "head tilt", "tilted head", "crouching", "reaching", "arms crossed",
    "crossed legs", "legs crossed", "hands to face", "hands on own face", "hand on own face", "covering face",
    "hand up", "hands up", "arm up", "outstretched arm", "clenched hand", "clenched hands",
];

/// A hair tag ("long hair", "hairband"), but not furniture ("chair", "armchair").
fn is_hair(n: &str) -> bool {
    n.contains("hair") && !n.contains("chair")
}

/// Matches whole words only (so "day" doesn't match "birthday").
fn has_word(text: &str, words: &[&str]) -> bool {
    words.iter().any(|w| {
        text == *w
            || text.split(|c: char| !c.is_alphanumeric() && c != '-').any(|part| part == *w)
            || (w.contains(' ') && text.contains(w))
    })
}

/// Checks whether a tag describes character identity, body traits, or clothing
/// rather than scene environment or camera/style.
pub fn is_character_like(n: &str) -> bool {
    if CAMERA.iter().any(|c| *c == n) || n.ends_with(" shot") || n.ends_with(" angle") || n.ends_with(" view") {
        return false;
    }
    (SUBJECT.contains(&n) && n != "no humans" && n != "scenery")
        || has_word(n, IDENTITY)
        || has_word(n, OUTFIT)
        || has_word(n, TRAITS)
        || has_word(n, &["girl", "boy", "guy", "woman", "man", "female", "male", "person", "human", "character"])
        || is_hair(n)
        || n.contains("eyes")
        || n.contains("skin")
        || n.contains("breast")
        || n.contains("cleavage")
        || n.contains("nipple")
        || n.contains("areola")
        || n.contains("navel")
        || n.contains("belly")
        || n.contains("midriff")
        || n.contains("stomach")
        || n.contains("waist")
        || has_word(n, &["abs", "six-pack", "six pack", "chest"])
        || n.contains("pussy")
        || n.contains("vagina")
        || n.contains("penis")
        || n.contains("erection")
        || n.contains("testicle")
        || n.contains("crotch")
        || n.contains("pubic")
        || n.contains("genital")
        || n.contains("anus")
        || has_word(n, &["butt", "butts", "ass", "asses", "booty", "buttocks", "gluteal"])
        || n.contains("thigh")
        || has_word(n, &["leg", "legs"])
        || has_word(n, &["foot", "feet", "toe", "toes", "sole", "soles"])
        || has_word(n, &["arm", "arms", "shoulder", "shoulders", "collarbone", "neck", "armpit", "armpits"])
        || n.contains("nude")
        || n.contains("naked")
        || n.contains("topless")
        || n.contains("bottomless")
        || has_word(n, &["body", "figure", "anatomy"])
        || has_word(n, &["horn", "horns", "tail", "tails", "wing", "wings", "ear", "ears"])
        || n.contains("dress")
        || n.contains("shirt")
        || n.contains("skirt")
        || n.contains("pants")
        || n.contains("shoe")
        || n.contains("sock")
        || n.contains("glove")
        || has_word(n, &["hat", "hats", "suit", "suits"])
}

/// Where a tag belongs. Declaration order is the arranged-layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Slot {
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

pub(crate) fn slot(t: &PromptToken) -> Slot {
    let n = t.normalized.as_str();
    match t.category {
        TokenCategory::QualityMeta => return Slot::Quality,
        TokenCategory::LoraOrSpecial => return Slot::Lora,
        _ => {}
    }
    if SUBJECT.contains(&n) {
        Slot::Subject
    } else if is_hair(n) || has_word(n, &["ponytail", "twintails", "bangs", "braid", "bun", "ahoge", "sidelocks"]) {
        Slot::Hair
    } else if n.contains("eyes") && !has_word(n, &["closed eyes", "one eye closed"])
        || n.contains("pupils")
        || n.contains("heterochromia")
        || n.contains("eyebrow")
        || n.contains("eyelash")
    {
        Slot::Eyes
    } else if has_word(n, EXPRESSION) {
        Slot::Expression
    } else if has_word(n, CAMERA) {
        Slot::Camera
    } else if has_word(n, POSE) || n == "v" || n == "v sign" {
        Slot::Pose
    } else if has_word(n, OUTFIT) {
        Slot::Outfit
    } else if has_word(n, TRAITS) || has_word(n, IDENTITY) {
        Slot::Traits
    } else if has_word(n, LIGHT) {
        Slot::Light
    } else if has_word(n, SETTING) {
        Slot::Setting
    } else if has_word(n, STYLE) {
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

fn and_list(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        n => format!("{} and {}", items[..n - 1].join(", "), items[n - 1]),
    }
}

/// One character as a self-contained clause for Anima, keeping every tag (weights too):
/// "the girl on the left is a kitsune with long white hair and red eyes, wearing a
/// red kimono, smile, sitting". `role` is e.g. "the girl on the left".
pub fn character_clause(tokens: &[PromptToken], role: &str) -> String {
    let (mut names, mut looks, mut outfit, mut rest) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for t in tokens {
        let n = t.normalized.as_str();
        if crate::two::count_tag(n).is_some() || n == "solo" {
            continue;
        }
        let raw = t.raw.trim().to_string();
        if IDENTITY.contains(&n) || crate::danbooru::is_character(n) || slot(t) == Slot::Subject {
            names.push(raw);
            continue;
        }
        match slot(t) {
            Slot::Traits | Slot::Hair | Slot::Eyes | Slot::Body => looks.push(raw),
            Slot::Outfit => outfit.push(raw),
            // words the lists don't know ("pocky") come after the outfit, as plain tags
            _ => rest.push(raw),
        }
    }
    let mut s = role.to_string();
    if names.is_empty() {
        if !looks.is_empty() {
            s.push_str(&format!(" has {}", and_list(&looks)));
        }
    } else {
        // a character name stands alone ("is hatsune miku"), anything else gets "a"
        let first = names[0].to_lowercase();
        let article = if crate::danbooru::is_character(&first) {
            ""
        } else if first.starts_with(['a', 'e', 'i', 'o', 'u']) {
            "an "
        } else {
            "a "
        };
        s.push_str(&format!(" is {article}{}", names.join(", ")));
        if !looks.is_empty() {
            s.push_str(&format!(" with {}", and_list(&looks)));
        }
    }
    if !outfit.is_empty() {
        s.push_str(&format!(", wearing {}", and_list(&outfit)));
    }
    if !rest.is_empty() {
        s.push_str(&format!(", {}", rest.join(", ")));
    }
    s
}

/// "Identity Only": keeps who the character is and what they wear (count, name,
/// traits, hair, eyes, body, outfit, LoRA tags) and drops expression, pose, gaze,
/// camera, setting and lighting, so a reference image doesn't dictate the scene.
pub fn filter_identity_only(prompt: &str) -> String {
    crate::parser::parse_prompt(prompt)
        .into_iter()
        .filter(|t| {
            let n = t.normalized.as_str();
            let gaze = n.starts_with("looking ") || n.ends_with(" gaze") || n == "eye contact";
            !gaze
                && matches!(
                    slot(t),
                    Slot::Subject | Slot::Other | Slot::Traits | Slot::Hair | Slot::Eyes | Slot::Body | Slot::Outfit | Slot::Lora
                )
        })
        .map(|t| t.raw.trim().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// "Scene / Pose reference": keeps what the picture *does* (pose, action, holding,
/// gaze and facing, interactions, camera and framing, setting and props, lighting,
/// style) and drops who is in it (count, species, hair, eyes, body, outfit), so a
/// reference image can pose your own characters. `keep_unknown` keeps words the
/// keyword lists don't know (the AI's props); the tagger's are dropped, to be safe.
pub fn filter_scene_and_pose_only(prompt: &str, keep_unknown: bool) -> String {
    crate::parser::parse_prompt(prompt)
        .into_iter()
        .filter(|t| {
            let n = t.normalized.as_str();
            if crate::two::count_tag(n).is_some() || n == "solo" || n == "multiple girls" || n == "multiple boys" || is_character_like(n) {
                return false;
            }
            let gaze = n.starts_with("looking ") || n.starts_with("facing ") || n.ends_with(" gaze") || n == "eye contact";
            if gaze || crate::two::is_interaction(n) {
                return true;
            }
            match slot(t) {
                Slot::Pose | Slot::Camera | Slot::Setting | Slot::Light | Slot::Style => true,
                Slot::Other => keep_unknown && !is_character_like(n),
                _ => false,
            }
        })
        .map(|t| t.raw.trim().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Is this tag something a Scene / Pose reference replaces (pose, gaze, camera,
/// setting, light, style)?
pub fn is_scene_or_pose(t: &PromptToken) -> bool {
    let n = t.normalized.as_str();
    n.starts_with("looking ")
        || n.starts_with("facing ")
        || matches!(slot(t), Slot::Pose | Slot::Camera | Slot::Setting | Slot::Light | Slot::Style)
}

/// One character's part of a two-character description:
/// "a kitsune with long white hair and red eyes, wearing a kimono, smile, sitting".
fn character_phrase(tokens: &[PromptToken]) -> String {
    let mut p = Parts::default();
    for t in tokens {
        classify(&mut p, t);
    }
    let names: Vec<String> = p.subject.iter().chain(&p.other).cloned().collect();
    let looks: Vec<String> = p.traits.iter().chain(&p.hair).chain(&p.eyes).chain(&p.body).cloned().collect();
    let mut s = if names.is_empty() { String::new() } else { list(&names, 3) };
    if !looks.is_empty() {
        s.push_str(&format!("{}with {}", if s.is_empty() { "" } else { " " }, list(&looks, 5)));
    }
    if !p.outfit.is_empty() {
        s.push_str(&format!(", wearing {}", list(&p.outfit, 3)));
    }
    for extra in [&p.expression, &p.pose] {
        if !extra.is_empty() {
            s.push_str(&format!(", {}", list(extra, 2)));
        }
    }
    s.trim_start_matches(", ").to_string()
}

/// The scene of a two-character prompt: who each character is, then the shared
/// camera, setting and light. `shared` holds the count, interaction and scene tags.
pub fn describe_many(chars: &[Vec<PromptToken>], shared: &[PromptToken]) -> SceneDescription {
    let mut sp = Parts::default();
    for t in shared {
        classify(&mut sp, t);
    }
    let together: Vec<String> = sp.pose.iter().chain(&sp.expression).chain(&sp.other).cloned().collect();
    let mut headline = subject_phrase(&sp.subject);
    if headline == "A scene" {
        headline = format!("{} characters", ["Two", "Three", "Four"][chars.len().clamp(2, 4) - 2]);
    }
    let mut parts = Vec::new();
    for (i, c) in chars.iter().enumerate() {
        let d = character_phrase(c);
        if !d.is_empty() {
            parts.push(format!("the {} {d}", ["first", "second", "third", "fourth"][i.min(3)]));
        }
    }
    if !parts.is_empty() {
        headline.push_str(&format!(": {}", parts.join("; ")));
    }
    if !together.is_empty() {
        headline.push_str(&format!(", together {}", list(&together, 3)));
    }
    headline.push('.');
    let mut second = Vec::new();
    if !sp.camera.is_empty() {
        second.push(format!("Framed as {}", list(&sp.camera, 2)));
    }
    if !sp.setting.is_empty() {
        second.push(format!("at {}", list(&sp.setting, 3)));
    }
    if !sp.light.is_empty() {
        second.push(format!("with {}", list(&sp.light, 2)));
    }
    if !sp.style.is_empty() {
        second.push(format!("in a {} style", list(&sp.style, 2)));
    }
    if !second.is_empty() {
        let mut s = second.join(", ");
        if let Some(f) = s.get(..1) {
            s = f.to_uppercase() + &s[1..];
        }
        headline.push_str(&format!(" {s}."));
    }

    let join = |v: &[PromptToken]| v.iter().map(|t| t.text.trim().to_string()).collect::<Vec<_>>().join(", ");
    let mut details = Vec::new();
    let mut rows = vec![("Who".to_string(), sp.subject.join(", "))];
    rows.extend(chars.iter().enumerate().map(|(i, c)| (format!("Character {}", i + 1), join(c))));
    rows.extend([
        ("Together".to_string(), together.join(", ")),
        ("Camera".to_string(), sp.camera.join(", ")),
        ("Setting".to_string(), sp.setting.join(", ")),
        ("Light / time".to_string(), sp.light.join(", ")),
        ("Style".to_string(), sp.style.join(", ")),
    ]);
    for (label, items) in rows {
        if !items.is_empty() {
            details.push((label, items));
        }
    }
    let mut missing = Vec::new();
    if chars.iter().any(|c| c.is_empty()) {
        missing.push("one of the characters");
    }
    if sp.camera.is_empty() {
        missing.push("camera framing");
    }
    if sp.setting.is_empty() {
        missing.push("setting / background");
    }
    if sp.light.is_empty() {
        missing.push("lighting / time of day");
    }
    SceneDescription { headline, details, missing }
}

/// `describe_many`'s headline, ready to go into a prompt.
pub fn prompt_sentence_many(chars: &[Vec<PromptToken>], shared: &[PromptToken]) -> String {
    strip_more(describe_many(chars, shared).headline)
}

fn strip_more(mut s: String) -> String {
    while let (Some(a), Some(b)) = (s.find(" (+"), s.find(" more)")) {
        if b < a {
            break;
        }
        s.replace_range(a..b + " more)".len(), "");
    }
    s
}

/// A plain-English sentence about the prompt for natural-language checkpoints
/// (the rule-based stand-in when no Deep Scan has written one).
pub fn prompt_sentence(tokens: &[PromptToken]) -> String {
    // "(+2 more)" belongs in the scene card, not in a prompt
    strip_more(describe(tokens).headline)
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
    fn scene_words_are_not_mistaken_for_looks_or_poses() {
        // "chair" is furniture, not hair; "flying" isn't "lying"; "composition" isn't a pose
        for (tag, not) in [("chair", Slot::Hair), ("armchair", Slot::Hair), ("flying", Slot::Pose), ("composition", Slot::Pose),
            ("spreadsheet", Slot::Pose), ("heating", Slot::Pose), ("epicenter", Slot::Style)]
        {
            assert_ne!(slot(&parse_prompt(tag)[0]), not, "{tag}");
        }
        for (tag, want) in [("long hair", Slot::Hair), ("hairband", Slot::Hair), ("lying on back", Slot::Pose),
            ("eating", Slot::Pose), ("close-up", Slot::Camera), ("slight smile", Slot::Expression), ("anime style", Slot::Style)]
        {
            assert_eq!(slot(&parse_prompt(tag)[0]), want, "{tag}");
        }
    }

    #[test]
    fn character_words_match_whole_words() {
        for scenery in ["bookshelf", "landscape", "rainbow", "library", "wardrobe", "escape", "that", "detailed"] {
            let t = &parse_prompt(scenery)[0];
            assert!(!matches!(slot(t), Slot::Outfit | Slot::Traits), "{scenery} -> {:?}", slot(t));
            assert!(!is_character_like(&t.normalized), "{scenery}");
        }
        for (tag, want) in [("animal ear fluff", Slot::Traits), ("black necktie", Slot::Outfit), ("serafuku", Slot::Outfit),
            ("mary janes", Slot::Outfit), ("white wings", Slot::Traits), ("red bow", Slot::Outfit), ("fox tail", Slot::Traits)]
        {
            assert_eq!(slot(&parse_prompt(tag)[0]), want, "{tag}");
        }
    }

    #[test]
    fn scene_and_pose_only_drops_who_is_in_it() {
        let p = "2girls, solo, kitsune, dragon girl, long white hair, (red eyes:1.2), large breasts, red kimono, fox ears, \
                 sitting, holding umbrella, looking at viewer, facing away, holding hands, from below, upper body, \
                 shrine, night, moonlight, anime style, petals";
        assert_eq!(
            filter_scene_and_pose_only(p, false),
            "sitting, holding umbrella, looking at viewer, facing away, holding hands, from below, upper body, shrine, night, \
             moonlight, anime style"
        );
        // the AI's unknown words (props) survive when asked to, but dragon girl and kitsune are never kept
        let with_unknown = filter_scene_and_pose_only(p, true);
        assert!(!with_unknown.contains("dragon girl"));
        assert!(!with_unknown.contains("kitsune"));
        assert!(with_unknown.ends_with("anime style, petals"));
    }

    #[test]
    fn identity_only_keeps_looks_and_outfit() {
        let out = filter_identity_only(
            "1girl, long white hair, (red eyes:1.2), fox ears, large breasts, red kimono, smile, sitting, looking at viewer, \
             upper body, from side, shrine, night, moonlight, <lora:fox:0.8>",
        );
        assert_eq!(out, "1girl, long white hair, (red eyes:1.2), fox ears, large breasts, red kimono, <lora:fox:0.8>");
    }

    #[test]
    fn character_clause_keeps_every_tag() {
        let t = parse_prompt("1girl, kitsune, long white hair, (red eyes:1.2), red kimono, obi, smile, sitting");
        assert_eq!(
            character_clause(&t, "the girl on the left"),
            "the girl on the left is a kitsune with long white hair and (red eyes:1.2), wearing red kimono and obi, smile, sitting"
        );
        let t = parse_prompt("1boy, black hair, armor");
        assert_eq!(character_clause(&t, "the boy on the right"), "the boy on the right has black hair, wearing armor");
    }

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
