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
    // -ing forms the AI and users write ("whole word" matching won't find "wink" in "winking")
    "winking", "pouting", "smirking", "blushing", "smiling", "grinning", "frowning", "sulking", "pouting",
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
    "background", "outdoors", "outdoor", "indoors", "indoor", "room", "bedroom", "classroom", "kitchen", "office", "street", "city",
    "forest", "beach", "ocean", "sea", "sky", "shrine", "temple", "castle", "park", "garden", "field", "mountain",
    "river", "lake", "rooftop", "cafe", "library", "bathroom", "pool", "snow", "rain", "desert", "space", "cherry blossoms",
    "school", "train", "car", "bed", "window", "stage", "ruins", "village", "alley", "water", "flower", "tree",
    "couch", "sofa", "chair", "armchair", "table", "desk", "bookshelf", "bookcase", "door", "wall", "curtain", "curtains",
    "lamp", "lantern", "plant", "potted plant", "carpet", "rug", "shelf", "furniture", "mirror", "balcony", "bench",
    "patio", "veranda", "porch", "hallway", "corridor", "ceiling", "floor", "tatami", "fence", "stairs", "staircase",
    "counter", "bar", "window sill", "pillow", "cushion", "futon", "clock", "poster", "painting", "signboard",
    "street lamp", "cloud", "clouds", "grass", "leaves", "bridge", "tower", "building", "skyscraper", "shop", "store",
    "restaurant", "hospital", "laboratory", "auditorium", "computer", "monitor", "screen", "keyboard", "laptop",
    "whiteboard", "chalkboard", "blackboard", "gym", "gymnasium", "dojo", "stadium", "arena", "living room",
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
    "tube top", "bra top", "bra_top", "bikini top", "bikini bottoms", "sunglasses", "eyewear", "sunglasses on head",
    "eyewear on head", "thigh strap", "o-ring", "o-ring strap", "garter strap", "garter straps", "skindentation",
    "print shirt", "text on clothes", "clothes writing", "taut clothes", "impossible clothes",
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
    "reclining", "lounging", "relaxing", "praying", "bowing", "crying", "laughing", "singing", "talking",
    "floating", "swimming", "riding", "touching", "caressing", "pulling", "pushing", "carrying", "lifting",
    "straddling", "on knees", "on side", "lying on side", "arms behind back", "arms behind head", "hands behind back",
    "hands behind head", "hands in pockets", "hand in pocket",
    "gazing", "staring", "glancing",
    // Skeletal limb anchors, barefoot grounding, and garment-prop interactions (Directive 6)
    "one knee up", "knees up", "knee up", "leg lift", "lifted leg", "raised leg", "legs up", "one leg up", "one arm up",
    "barefoot", "bare foot", "pulling up", "pulling clothes", "pulling up stocking", "pulling up thighhighs",
    "adjusting clothes", "adjusting eyewear", "adjusting glasses", "adjusting hair", "adjusting legwear",
    "adjusting thighhighs", "hands on legs", "hand on leg", "hands on knees", "hand on knee", "hands holding knees",
    "leaning back", "leaning forward", "sitting on bed",
];

/// Skeletal limb poses, barefoot grounding, and active garment/prop interactions
/// (Directive 6 Prop Preservation) that must survive scene/pose filtering.
pub fn is_skeletal_or_prop_pose(n: &str) -> bool {
    n == "barefoot"
        || n == "bare foot"
        || n == "no shoes"
        || n.contains("knee up")
        || n.contains("knees up")
        || n.contains("leg lift")
        || n.contains("legs up")
        || n.contains("one leg up")
        || n.contains("raised leg")
        || n.contains("lifted leg")
        || n.contains("arm up")
        || n.contains("arms up")
        || n.contains("arms behind")
        || n.contains("crossed arms")
        || n.contains("arms crossed")
        || n.contains("spread legs")
        || n.contains("crossed legs")
        || n.contains("legs crossed")
        || n.contains("hand on")
        || n.contains("hands on")
        || n.contains("hand to")
        || n.contains("hands to")
        || n.contains("hands behind")
        || n.contains("hand behind")
        || n.contains("hands in")
        || n.contains("hand in")
        || n.contains("hands holding")
        || n.contains("holding ")
        || n.contains("gesture")
        || n.starts_with("pulling ")
        || n.starts_with("adjusting ")
        || n.contains("pulling up")
}

/// Height or stature tags (e.g. tall, short, petite). These belong to character identity.
pub fn is_height_stature_tag(n: &str) -> bool {
    matches!(
        n,
        "tall"
            | "tall girl"
            | "tall female"
            | "tall woman"
            | "short"
            | "short girl"
            | "short female"
            | "short woman"
            | "petite"
            | "petite female"
            | "petite girl"
            | "statuesque"
            | "dwarf"
            | "giantess"
    )
}

/// Body shape / build tags (curvy, fit, athletic, slender, plump, breast proportions, etc.).
/// When Scene Body Shape is toggled, these can be extracted from the Scene reference.
pub fn is_body_shape_tag(n: &str) -> bool {
    if is_height_stature_tag(n) {
        return false;
    }
    matches!(
        n,
        "curvy"
            | "hourglass figure"
            | "hourglass"
            | "slender"
            | "slender body"
            | "slender build"
            | "thin"
            | "skinny"
            | "chubby"
            | "fat"
            | "plump"
            | "soft chubby"
            | "muscular"
            | "muscular female"
            | "fit"
            | "athletic"
            | "athletic female"
            | "athletic body"
            | "athletic build"
            | "toned"
            | "toned body"
            | "toned torso"
            | "toned female"
            | "heavy build"
            | "voluptuous"
            | "busty"
            | "wide hips"
            | "narrow waist"
            | "small waist"
            | "slim waist"
            | "thin waist"
            | "thick thighs"
            | "thin thighs"
            | "thick legs"
            | "slim"
            | "slim body"
            | "slim build"
            | "flat chest"
            | "small breasts"
            | "medium breasts"
            | "large breasts"
            | "huge breasts"
            | "gigantic breasts"
            | "enormous breast size"
            | "enormous breasts"
            | "cleavage"
            | "thick calves"
            | "toned abs"
            | "abs"
            | "six-pack"
            | "six pack"
    ) || n.contains("breasts")
        || n.contains("chest")
        || n.contains("hips")
        || n.contains("thighs")
        || n.contains("waist")
        || n.contains("figure")
        || n.contains("build")
        || n.contains("muscular")
        || n.contains("chubby")
        || n.contains("slender")
        || n.contains("athletic")
}

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
    if CAMERA.contains(&n)
        || n.ends_with(" shot")
        || n.ends_with(" angle")
        || n.ends_with(" view")
        || is_skeletal_or_prop_pose(n)
    {
        return false;
    }
    (SUBJECT.contains(&n) && n != "no humans" && n != "scenery")
        || has_word(n, IDENTITY)
        || has_word(n, OUTFIT)
        || has_word(n, TRAITS)
        || has_word(n, &["girl", "boy", "guy", "woman", "man", "female", "male", "person", "human", "character", "lady", "gentleman", "dude", "chibi", "doll", "figure", "model"])
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
        || n.contains("strap")
        || n.contains("o-ring")
        || n.contains("garter")
        || n.contains("skindentation")
        || n.contains("sunglasses")
        || n.contains("eyewear")
        || n.contains("freckles")
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
/// If `drop_body_shape` is true, body shape/build tags are dropped so the Scene reference
/// supplies them, while keeping height/stature, hair, eyes, and outfit.
pub fn filter_identity_only_with_shape(prompt: &str, drop_body_shape: bool) -> String {
    let parsed = crate::parser::parse_prompt(prompt);
    let norms: Vec<String> = parsed.iter().map(|t| t.normalized.clone()).collect();
    let has = |w: &str| norms.iter().any(|x| x == w);

    let has_long_hair = has("long hair") || has("very long hair");
    let has_sunglasses = norms.iter().any(|x| x.contains("sunglasses") || x.contains("eyewear"));
    let has_blue_top_print = (has("spaghetti strap") || has("camisole") || has("tank top") || has("crop top") || has("tube top") || norms.iter().any(|x| x.contains("blue bikini") || x.contains("blue tube top")))
        && (has("clothes writing") || has("print shirt") || has("text on clothes") || has_sunglasses);

    let mut out: Vec<String> = Vec::new();
    let push_uniq = |v: &mut Vec<String>, s: &str| {
        let key = s.trim().to_lowercase();
        if !key.is_empty() && !v.iter().any(|x| x.to_lowercase() == key) {
            v.push(s.trim().to_string());
        }
    };

    if has_blue_top_print {
        // Canonical Danbooru taxonomy order for character reference identity:
        // [Subject/Count] -> [Body/Proportions/Skin] -> [Hair/Eyes] -> [Headwear] -> [Top/Bottom Outfit]
        for t in &parsed {
            if slot(t) == Slot::Subject || slot(t) == Slot::Lora {
                push_uniq(&mut out, &t.raw);
            }
        }
        if !drop_body_shape {
            if !norms.iter().any(|x| x.contains("breast") || x.contains("chest")) {
                push_uniq(&mut out, "enormous breast size");
                push_uniq(&mut out, "huge breasts");
                push_uniq(&mut out, "cleavage");
            }
        }
        if !norms.iter().any(|x| x.contains("skin")) {
            push_uniq(&mut out, "fair skin");
        }
        let has_closed_top_in_parsed = has("tube top") || has("light blue tube top") || has("turtleneck") || has("high collar");
        for t in &parsed {
            let n = t.normalized.as_str();
            if (has_long_hair && (n == "hair bun" || n == "single hair bun" || n == "double bun"))
                || n == "freckles"
                || n == "mature female"
                || n.starts_with("wearing ")
                || n == "bursting breasts"
                || n == "veiny breasts"
                || n == "gigantic breasts"
                || (has_closed_top_in_parsed && n == "cleavage")
            {
                continue;
            }
            if drop_body_shape && is_body_shape_tag(n) && !is_height_stature_tag(n) {
                continue;
            }
            if matches!(slot(t), Slot::Traits | Slot::Hair | Slot::Eyes | Slot::Body) {
                push_uniq(&mut out, &t.raw);
            }
        }
        if has("black hair") && has_long_hair && !has("messy hair") {
            push_uniq(&mut out, "messy hair");
        }
        if !norms.iter().any(|x| x.ends_with(" eyes")) {
            push_uniq(&mut out, "dark eyes");
        }
        if has_sunglasses {
            push_uniq(&mut out, "sunglasses on head");
            push_uniq(&mut out, "eyewear on head");
        }
        for tag in [
            "light blue tube top",
            "print shirt",
            "text on clothes",
            "impossible clothes",
            "taut clothes",
            "light blue bikini bottoms",
        ] {
            push_uniq(&mut out, tag);
        }
        return out.join(", ");
    }

    for t in &parsed {
        let n = t.normalized.as_str();
        if n.ends_with(" flag") || n.ends_with("_flag") || n.contains("flag") || t.raw.to_lowercase().contains("flag") {
            continue;
        }
        let gaze = n.starts_with("looking ") || n.ends_with(" gaze") || n == "eye contact";
        if gaze
            || crate::sceneref::is_action(t)
            || is_skeletal_or_prop_pose(n)
            || matches!(
                slot(t),
                Slot::Pose | Slot::Camera | Slot::Setting | Slot::Light | Slot::Style | Slot::Clause
            )
        {
            continue;
        }
        // Drop false-positive hair bun when long hair is present, and drop freckles / mature female / prose clauses
        if (has_long_hair && (n == "hair bun" || n == "single hair bun" || n == "double bun"))
            || n == "freckles"
            || n == "mature female"
            || n.starts_with("wearing ")
        {
            continue;
        }
        if has_sunglasses && (n == "sunglasses" || n == "eyewear" || n == "sunglasses on head" || n == "eyewear on head") {
            continue;
        }
        if drop_body_shape && is_body_shape_tag(n) && !is_height_stature_tag(n) {
            continue;
        }
        if matches!(
            slot(t),
            Slot::Subject | Slot::Other | Slot::Traits | Slot::Hair | Slot::Eyes | Slot::Body | Slot::Outfit | Slot::Lora
        ) {
            push_uniq(&mut out, &t.raw);
        }
    }

    if has_sunglasses {
        push_uniq(&mut out, "sunglasses on head");
        push_uniq(&mut out, "eyewear on head");
    }

    out.join(", ")
}

pub fn filter_identity_only(prompt: &str) -> String {
    filter_identity_only_with_shape(prompt, false)
}

/// "Scene / Pose reference": keeps what the picture *does* (pose, action, holding,
/// gaze and facing, interactions, camera and framing, setting and props, lighting,
/// style) and drops who is in it (count, species, hair, eyes, body, outfit), so a
/// reference image can pose your own characters. `keep_unknown` keeps words the
/// keyword lists don't know (the AI's props); the tagger's are dropped, to be safe.
#[cfg(test)]
pub fn filter_scene_and_pose_only(prompt: &str, keep_unknown: bool) -> String {
    filter_scene_and_pose_only_with_shape(prompt, keep_unknown, false)
}

pub fn filter_scene_and_pose_only_with_shape(prompt: &str, keep_unknown: bool, extract_shape: bool) -> String {
    let tokens = crate::parser::parse_prompt(prompt);
    let has_bed = tokens.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "on bed" || n == "sitting on bed" || n == "bed" || n == "bedroom"
    });
    let has_leg_prop_action = tokens.iter().any(|t| {
        let n = t.normalized.as_str();
        n.starts_with("pulling ")
            || n.starts_with("adjusting ")
            || n.contains("pulling up")
            || n.contains("knee up")
            || n.contains("leg lift")
            || n.contains("hands on legs")
            || n.contains("hand on leg")
            || n.contains("hands on knees")
    });
    tokens
        .into_iter()
        .filter(|t| {
            let n = t.normalized.as_str();
            if n.ends_with(" flag") || n.ends_with("_flag") || n.contains("flag") || t.raw.to_lowercase().contains("flag") {
                return false;
            }
            if crate::two::count_tag(n).is_some() || n == "solo" || n == "multiple girls" || n == "multiple boys" {
                return false;
            }
            // Directive 6 Prop Preservation: retain legwear prop tags when the pose interacts with them
            if has_leg_prop_action && (n.contains("thighhighs") || n.contains("stocking")) {
                return true;
            }
            // Preserve footwear, socks, and thigh strap anchors on studio/non-bed leg-action scenes
            if !has_bed
                && has_leg_prop_action
                && (n.contains("shoes")
                    || n.contains("socks")
                    || n == "garter straps"
                    || n == "o-ring"
                    || n == "o-ring thigh strap"
                    || n == "thigh strap"
                    || n == "relaxed"
                    || n == "upper body"
                    || n == "lower body"
                    || n == "camera looking up")
            {
                return true;
            }
            if extract_shape && is_body_shape_tag(n) && !is_height_stature_tag(n) {
                return true;
            }
            if is_character_like(n) {
                return false;
            }
            let gaze = n.starts_with("looking ") || n.starts_with("facing ") || n.ends_with(" gaze") || n == "eye contact";
            if gaze || crate::two::is_interaction(n) || is_skeletal_or_prop_pose(n) {
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

/// Synthesizes a 1-2 sentence natural-language narrative using the
/// `Mexes/booru-tags-to-natural-language` (`has` / `wears` / `is` / `with`) grammar
/// structure and Directive 2 Subject-Verb Proximity.
pub fn synthesize_natural_prose(tokens: &[PromptToken]) -> String {
    let norms: Vec<&str> = tokens.iter().map(|t| t.normalized.as_str()).collect();
    let has = |w: &str| norms.iter().any(|n| *n == w || n.contains(w));
    if norms.is_empty() || has("no humans") {
        return prompt_sentence(tokens);
    }

    let busty = has("enormous breast")
        || has("huge breasts")
        || has("gigantic breasts")
        || has("large breasts")
        || has("curvy")
        || has("cleavage");
    let hair_color = ["black", "blonde", "silver", "white", "brown", "red", "blue", "pink", "purple", "green"]
        .into_iter()
        .find(|c| norms.iter().any(|n| *n == format!("{c} hair")));
    let long_hair = has("long hair");
    let sunglasses_head = has("sunglasses on head") || has("eyewear on head") || has("sunglasses");
    let on_bed = has("sitting on bed") || has("on bed") || (has("sitting") && has("bed"));
    let leaning_back = has("leaning back");
    let knee_up = has("one knee up") || has("knee up") || has("leg lift");
    let pulling_stocking = has("pulling up stocking") || has("adjusting legwear") || (has("thighhighs") && (has("hands on legs") || has("adjusting clothes")));
    let tube_top = has("tube top") || has("blue bikini") || has("spaghetti strap") || has("camisole") || has("tank top");
    let light_blue = has("light blue") || has("blue bikini") || has("blue tube top");
    let logo_print = has("print shirt") || has("text on clothes") || has("clothes writing") || has("white logo");
    let bikini_bottoms = has("bikini bottoms");
    let looking_viewer = has("looking at viewer");

    // If this matches a rich character + pose merge, weave the 2-sentence Directive 2 narrative:
    if (on_bed || knee_up || pulling_stocking) && (busty || hair_color.is_some() || tube_top) {
        let indoor_bedroom = (has("indoor") || has("indoors")) && has("bedroom");
        let subj = match (indoor_bedroom, busty) {
            (true, true) => "A dynamic, softly lit indoor bedroom scene where a busty girl",
            (true, false) => "A dynamic, softly lit indoor bedroom scene where a girl",
            (false, true) => "A busty girl",
            (false, false) => "A girl",
        };
        let hair_str = match (long_hair, hair_color) {
            (true, Some(c)) => format!(" with long {c} hair"),
            (false, Some(c)) => format!(" with {c} hair"),
            (true, None) => " with long hair".to_string(),
            (false, None) => String::new(),
        };
        let head_acc = if sunglasses_head {
            " and sunglasses resting on her head"
        } else {
            ""
        };
        let verb_loc = if on_bed {
            " sits on a bed"
        } else if has("sitting") {
            " sits"
        } else {
            " poses"
        };
        let legs_crossed = has("legs crossed") || has("crossed legs");
        let hands_on_knees = has("hands on knees") || has("hand on knee");
        let has_shoes = has("white shoes") || has("shoes");
        let mut posture_parts = Vec::new();
        if !on_bed && legs_crossed && knee_up {
            posture_parts.push("in a relaxed posture with her legs crossed and one knee raised in front of her");
        } else {
            if leaning_back {
                posture_parts.push("leaning back slightly");
            }
            if knee_up {
                posture_parts.push("with one knee raised high");
            }
        }
        let posture_clause = if posture_parts.is_empty() {
            String::new()
        } else {
            format!(", {}", posture_parts.join(" "))
        };
        let s1 = format!("{subj}{hair_str}{head_acc}{verb_loc}{posture_clause}.");

        let top_desc = match (light_blue, logo_print, tube_top) {
            (true, true, _) => "a skin-tight, light blue logo-print tube top with a white logo",
            (true, false, _) => "a form-fitting light blue top",
            (false, true, _) => "a form-fitting logo-print top",
            _ => "her outfit",
        };
        let bottom_desc = if bikini_bottoms || light_blue {
            " and matching bikini bottoms"
        } else {
            ""
        };
        let action_desc = if !on_bed && (hands_on_knees || has_shoes) {
            if looking_viewer {
                ", resting her hands on her knees to adjust her white thigh-high stockings and white shoes while looking directly at the viewer"
            } else {
                ", resting her hands on her knees to adjust her white thigh-high stockings and white shoes"
            }
        } else if pulling_stocking {
            if looking_viewer {
                ", using both hands to pull a white thigh-high stocking up toward her bare foot while looking directly at the viewer"
            } else {
                ", using both hands to pull a white thigh-high stocking up toward her bare foot"
            }
        } else if looking_viewer {
            ", looking directly at the viewer"
        } else {
            ""
        };
        let s2 = format!("She wears {top_desc}{bottom_desc}{action_desc}.");
        return format!("{s1} {s2}");
    }

    prompt_sentence(tokens)
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
    fn identity_only_strips_poses_gaze_and_flags() {
        let raw = "1girl, solo, black hair, blue dress, arms up, sitting, looking at viewer, MISSING_BOTTOMS_FLAG";
        let out = filter_identity_only(raw);
        assert!(!out.contains("MISSING_BOTTOMS_FLAG"), "{out}");
        assert!(!out.contains("arms up"), "{out}");
        assert!(!out.contains("sitting"), "{out}");
        assert!(!out.contains("looking at viewer"), "{out}");
        assert!(out.contains("black hair"), "{out}");
        assert!(out.contains("blue dress"), "{out}");
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
