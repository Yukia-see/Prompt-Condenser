//! Danbooru tag list (the WD tagger's `selected_tags.csv`, ~10k common tags with
//! usage counts). Used by the Danbooru translator to offer real tag names to the AI
//! and to check the tags it writes.

use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct TagInfo {
    /// Prompt form: spaces instead of underscores ("holding umbrella").
    pub name: String,
    pub count: u64,
    pub character: bool,
}

pub struct TagList {
    tags: Vec<TagInfo>,
    by_name: HashMap<String, usize>,
}

static LIST: OnceLock<TagList> = OnceLock::new();
static PROMPT_CONDENSER_VOCAB: OnceLock<HashSet<String>> = OnceLock::new();

fn csv_path() -> std::path::PathBuf {
    crate::tagger::tagger_dir().join("selected_tags.csv")
}

pub fn get_prompt_condenser_vocab() -> Option<&'static HashSet<String>> {
    let set = PROMPT_CONDENSER_VOCAB.get_or_init(|| {
        let p = csv_path();
        if let Ok(content) = std::fs::read_to_string(&p) {
            crate::prompt_condenser::load_vocab_from_csv(&content)
        } else {
            HashSet::new()
        }
    });
    if set.is_empty() {
        None
    } else {
        Some(set)
    }
}

/// Lowercase, underscores to spaces, escaped parentheses unescaped.
pub fn key(tag: &str) -> String {
    tag.trim()
        .to_lowercase()
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse(text: &str) -> TagList {
    let mut tags = Vec::new();
    // tag_id,name,category,count (names may contain commas)
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let category = f[f.len() - 2];
        if category == "9" {
            continue; // rating tags
        }
        tags.push(TagInfo {
            name: key(&f[1..f.len() - 2].join(",")),
            count: f[f.len() - 1].trim().parse().unwrap_or(0),
            character: category == "4",
        });
    }
    let by_name = tags.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
    TagList { tags, by_name }
}

/// The tag list, downloading the small CSV (~300 KB) the first time if the tagger
/// isn't installed.
pub fn list() -> Result<&'static TagList> {
    if let Some(l) = LIST.get() {
        return Ok(l);
    }
    let path = csv_path();
    if !path.is_file() {
        std::fs::create_dir_all(path.parent().expect("has a parent"))?;
        let body = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?
            .get(format!("{}/selected_tags.csv", crate::tagger::REPO))
            .header("User-Agent", "prompt-condenser")
            .send()
            .context("Couldn't download the Danbooru tag list")?;
        if !body.status().is_success() {
            return Err(anyhow!("Downloading the Danbooru tag list failed: HTTP {}", body.status()));
        }
        std::fs::write(&path, body.bytes()?)?;
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(LIST.get_or_init(|| parse(&text)))
}

/// The tag list if it's already loaded or on disk (never downloads; safe on the UI thread).
pub fn local() -> Option<&'static TagList> {
    if let Some(l) = LIST.get() {
        return Some(l);
    }
    let text = std::fs::read_to_string(csv_path()).ok()?;
    Some(LIST.get_or_init(|| parse(&text)))
}

/// True for a Danbooru character name ("hatsune miku", "rouge the bat").
pub fn is_character(tag: &str) -> bool {
    local().and_then(|l| l.get(tag)).map(|t| t.character).unwrap_or(false)
}

const STOPWORDS: &[&str] = &[
    "the", "and", "with", "her", "his", "their", "she", "him", "they", "that", "this", "from", "into", "onto",
    "over", "under", "while", "who", "which", "very", "some", "for", "are", "was", "is", "its", "has", "have",
    "being", "wearing", "looking", "a", "an", "of", "in", "on", "at", "to", "by", "as", "or", "but", "out",
];

impl TagList {
    pub fn get(&self, tag: &str) -> Option<&TagInfo> {
        self.by_name.get(&key(tag)).map(|&i| &self.tags[i])
    }

    /// Real tag for a near miss: plural/singular or a dash/space difference.
    pub fn fix(&self, tag: &str) -> Option<&TagInfo> {
        let k = key(tag);
        let tries = [
            format!("{k}s"),
            k.strip_suffix('s').unwrap_or(&k).to_string(),
            k.replace('-', " "),
            k.replace(' ', "-"),
        ];
        tries.iter().find_map(|t| self.get(t))
    }

    /// Real tags close to an unknown one, ranked by its distinctive words: words shared
    /// with hundreds of tags ("eyes", "hair") don't count, so "downcast eyes" doesn't
    /// suggest "blue eyes".
    pub fn similar(&self, tag: &str, n: usize) -> Vec<&TagInfo> {
        let words = |t: &str| -> Vec<String> {
            t.split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
                .map(|w| w.strip_suffix('s').filter(|s| s.len() >= 3).unwrap_or(w).to_string())
                .collect()
        };
        let want = words(&key(tag));
        let freq = |w: &str| self.tags.iter().filter(|t| !t.character && words(&t.name).iter().any(|x| x == w)).count();
        let weights: Vec<(String, f32)> = want
            .iter()
            .filter_map(|w| {
                let f = freq(w);
                (f > 0 && f <= 40).then(|| (w.clone(), 1.0 / f as f32))
            })
            .collect();
        if weights.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f32, &TagInfo)> = self
            .tags
            .iter()
            .filter(|t| !t.character)
            .filter_map(|t| {
                let tw = words(&t.name);
                let score: f32 = weights.iter().filter(|(w, _)| tw.contains(w)).map(|(_, x)| x).sum();
                (score > 0.0).then_some((score, t))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.count.cmp(&a.1.count)));
        scored.into_iter().take(n).map(|(_, t)| t).collect()
    }

    /// Common tags that contain the description's words, most used first.
    pub fn candidates(&self, text: &str, per_word: usize, total: usize) -> Vec<&TagInfo> {
        let words: Vec<String> = text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
            .map(str::to_string)
            .collect();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for w in &words {
            let stem = w.strip_suffix("es").filter(|s| s.len() >= 3).or_else(|| w.strip_suffix('s')).unwrap_or(w);
            let mut hits: Vec<&TagInfo> = self
                .tags
                .iter()
                .filter(|t| !t.character)
                .filter(|t| t.name.split(|c: char| !c.is_alphanumeric()).any(|tw| tw == w || tw == stem || tw.strip_suffix('s') == Some(stem)))
                .collect();
            hits.sort_by_key(|a| std::cmp::Reverse(a.count));
            for t in hits.into_iter().take(per_word) {
                if seen.insert(t.name.clone()) {
                    out.push(t);
                }
            }
            if out.len() >= total {
                break;
            }
        }
        out.truncate(total);
        out
    }
}

/// Sentence continuation kinds from `Mexes/booru-tags-to-natural-language` (`phrases.csv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // not wired to the UI yet
pub enum PhraseKind {
    /// "She is ___" (posture, active verbs, expressions, gaze)
    Is,
    /// "..., with ___" (secondary limb state, props, body details)
    With,
    /// "She has ___" (hair, eyes, skin, body proportions)
    Has,
    /// "She wears ___" (top, bottom, legwear, footwear, headwear, eyewear)
    Wears,
    /// "The image is ___" (camera framing and angle)
    Framing,
    /// Noun phrase for setting, lighting, or atmosphere
    Scene,
}

/// Converts a Danbooru tag into its `Mexes/booru-tags-to-natural-language` `(PhraseKind, String)`
/// with `{pos}` expanded to `pos` (e.g. `"her"`, `"his"`, `"their"`).
#[allow(dead_code)] // not wired to the UI yet
pub fn tag_to_phrase(tag: &str, pos: &str) -> Option<(PhraseKind, String)> {
    let k = key(tag);
    if k.is_empty() {
        return None;
    }
    let (kind, tpl): (PhraseKind, &str) = match k.as_str() {
        // Pose & active limb/prop phrases (`is` / `with`)
        "sitting on bed" | "on bed" => (PhraseKind::Is, "sits on a bed"),
        "sitting" => (PhraseKind::Is, "sits"),
        "standing" => (PhraseKind::Is, "stands"),
        "kneeling" => (PhraseKind::Is, "kneels"),
        "wariza" => (PhraseKind::Is, "sitting with {pos} legs splayed to the sides"),
        "seiza" => (PhraseKind::Is, "kneeling formally on {pos} heels"),
        "leaning back" => (PhraseKind::With, "leaning back slightly"),
        "leaning forward" => (PhraseKind::With, "leaning forward"),
        "one knee up" | "knee up" | "leg lift" => (PhraseKind::With, "with one knee raised high"),
        "pulling up stocking" | "adjusting legwear" | "pulling up thighhighs" => {
            (PhraseKind::Is, "using both hands to pull a white thigh-high stocking up {pos} raised leg")
        }
        "hands on legs" | "hand on leg" | "adjusting clothes" => {
            (PhraseKind::With, "both hands on {pos} raised leg")
        }
        "looking at viewer" => (PhraseKind::Is, "looking at the viewer"),
        "looking back" => (PhraseKind::Is, "looking back over {pos} shoulder"),
        "bare foot" | "barefoot" | "no shoes" => (PhraseKind::With, "one bare foot visible"),
        "hand on own hip" | "hand on hip" => (PhraseKind::With, "one hand on {pos} hip"),
        "hands on hips" => (PhraseKind::With, "both hands on {pos} hips"),
        "arms behind back" | "hands behind back" => (PhraseKind::With, "both hands behind {pos} back"),
        "arms behind head" | "hands behind head" => (PhraseKind::With, "both arms raised behind {pos} head"),
        // Headwear & eyewear (`wears`)
        "sunglasses on head" | "eyewear on head" | "sunglasses" => {
            (PhraseKind::Wears, "sunglasses resting on {pos} head")
        }
        // Clothing & tension (`wears`)
        "light blue tube top" | "blue bikini top" | "wearing a blue bikini" => {
            (PhraseKind::Wears, "a skin-tight, light blue logo-print tube top")
        }
        "light blue bikini bottoms" | "bikini bottoms" => {
            (PhraseKind::Wears, "matching bikini bottoms")
        }
        "white thighhighs" | "thighhighs" => (PhraseKind::Wears, "white thigh-high stockings"),
        // Appearance & proportions (`has`)
        "enormous breast size" | "huge breasts" | "gigantic breasts" | "large breasts" | "curvy" => {
            (PhraseKind::Has, "busty")
        }
        "long hair" => (PhraseKind::Has, "long"),
        "black hair" => (PhraseKind::Has, "black hair"),
        "messy hair" => (PhraseKind::Has, "messy hair"),
        "dark eyes" | "brown eyes" => (PhraseKind::Has, "dark eyes"),
        "grey eyes" => (PhraseKind::Has, "grey eyes"),
        "fair skin" => (PhraseKind::Has, "fair skin"),
        // Framing & scene
        "full body" => (PhraseKind::Framing, "a full-body shot"),
        "cowboy shot" => (PhraseKind::Framing, "a cowboy shot from the thighs up"),
        "dynamic angle" => (PhraseKind::Framing, "captured from a dynamic angle"),
        "depth of field" => (PhraseKind::Framing, "with shallow depth of field"),
        "soft lighting" => (PhraseKind::Scene, "soft ambient lighting"),
        "bedroom" | "indoor" | "indoors" => (PhraseKind::Scene, "an indoor bedroom"),
        "polka dot background" | "blue background" => (PhraseKind::Scene, "a light blue polka-dot backdrop"),
        _ => return None,
    };
    Some((kind, tpl.replace("{pos}", pos)))
}

/// Reverse-matches natural-language descriptions (from Moondream 1.8B or Gemma 4)
/// against the `Mexes/booru-tags-to-natural-language` phrase patterns and
/// `danbooru_tags_post_count.csv` canonical vocabulary to recover missing Danbooru tags.
/// A clause that says something is absent: it contains a negating word ("no", "not", "without",
/// "never", "none", "neither", "nor", "isn't"...).
fn clause_is_negated(clause: &str) -> bool {
    const NEGATIONS: &[&str] = &[
        "no", "not", "without", "never", "none", "neither", "nor", "isn't", "aren't", "doesn't", "don't", "wasn't",
        "weren't", "lacks", "lacking", "absent",
    ];
    clause
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '\''))
        .any(|w| NEGATIONS.contains(&w))
}

/// `prose_has` for other modules (whole words, common endings allowed; `text` lowercase).
pub fn prose_has_pub(text: &str, phrase: &str) -> bool {
    prose_has(text, phrase)
}

/// `phrase` appears in `text` as whole words: the character before it is not a letter or digit,
/// and after it comes either nothing or a common ending (s, es, ed, d, ing, plus the doubled-consonant
/// forms like "sitting"), then a non-letter. So "sit" finds "sits" and "sitting" but not "visible",
/// and "fit" doesn't find "outfit". `text` must already be lowercase.
fn prose_has(text: &str, phrase: &str) -> bool {
    const ENDINGS: &[&str] = &["", "s", "es", "ed", "d", "ing"];
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(pos) = text[from..].find(phrase) {
        let i = from + pos;
        let j = i + phrase.len();
        let starts_word = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        let tail: String = text[j..].chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        // "sitting", "dropped": a doubled last consonant before -ing / -ed
        let doubled = phrase.chars().last().is_some_and(|c| tail.starts_with(c) && ENDINGS.contains(&&tail[c.len_utf8()..]));
        if starts_word && (ENDINGS.contains(&tail.as_str()) || doubled) {
            return true;
        }
        from = i + 1;
        while from < text.len() && !text.is_char_boundary(from) {
            from += 1;
        }
    }
    false
}

pub fn extract_tags_from_prose(prose: &str) -> Vec<&'static str> {
    let raw = prose.to_lowercase();
    if raw.trim().is_empty() {
        return Vec::new();
    }
    // Clauses that negate ("no keyhole cutout is visible", "she is not wearing garter straps") are left
    // out: vision models answer checklists item by item, and every "no X" used to become the tag X.
    // "a red dress with no shoes": the negation starts a clause of its own (keeping its word), so only
    // "no shoes" is left out, not the red dress
    let marked = raw
        .replace(" with no ", ". no ")
        .replace(" but no ", ". no ")
        .replace(" without ", ". without ")
        .replace(" lacking ", ". lacking ");
    let clauses: Vec<&str> = marked
        .split(|c| matches!(c, '.' | ';' | ',' | '\n' | '!' | '?' | ':'))
        .flat_map(|c| c.split(" but "))
        .filter(|c| !clause_is_negated(c))
        .collect();
    let lower = clauses.join(". ");
    // the few rules that are about an absence read the full text
    let raw_has = |phrase: &str| prose_has(&raw, phrase);
    // Whole words only (with common endings): plain substring matching turned "reference sheet"
    // into a bedroom, "outfit" into athletic, "clamp" into a lamp and "visible" into sitting.
    let has = |phrase: &str| prose_has(&lower, phrase);
    let mut out: Vec<&'static str> = Vec::new();
    let mut push = |t: &'static str| {
        if !out.contains(&t) {
            out.push(t);
        }
    };

    // Scene & surface grounding
    if has("bedroom") || has("bed") || has("mattress") || has("bed sheet") || has("bedsheet") {
        push("indoor");
        push("bedroom");
        if has("sit") || has("seated") || has("sits") {
            push("sitting on bed");
        }
    }
    if has("soft light") || has("soft ambient") || has("warm light") {
        push("soft lighting");
    }
    if has("depth of field") || has("shallow focus") || has("bokeh") {
        push("depth of field");
    }
    if has("dynamic angle") || has("low angle") || has("foreshorten") {
        push("dynamic angle");
    }

    // Skeletal posture & active prop/stocking interactions
    if has("lean") && has("back") {
        push("leaning back");
    }
    if (has("knee") && (has("raised") || has("up") || has("high") || has("bent")))
        || has("raised leg")
        || has("leg lift")
    {
        push("one knee up");
        push("leg lift");
    }
    if (has("pull") || has("adjust") || has("tug"))
        && (has("stocking") || has("thigh-high") || has("thighhigh") || has("legwear") || has("sock"))
    {
        push("white thighhighs");
        push("pulling up stocking");
        push("hands on legs");
        push("adjusting clothes");
    }
    if has("bare foot") || has("barefoot") || has("bare feet") || raw_has("no shoes") || has("unshod") {
        push("bare foot");
    }
    if has("looking at the viewer") || has("looking at viewer") || has("looks at the camera") || has("direct gaze") {
        push("looking at viewer");
    }

    // Body shape, build, and proportions
    let has_closed_top = has("tube top")
        || has("turtleneck")
        || has("high collar")
        || has("closed neckline")
        || has("covered chest")
        || raw_has("no cleavage");
    let has_confirmed_cleavage = (has("cleavage")
        || has("plunging neckline")
        || has("deep neckline")
        || has("open neckline")
        || has("exposed cleavage"))
        && !has_closed_top;

    if has("curvy") || has("hourglass") || has("voluptuous") {
        push("curvy");
        push("hourglass figure");
    }
    if has("athletic") || has("toned") || has("muscular") {
        push("athletic");
        push("fit");
        push("toned");
    }
    if has("slender") || has("slim") || has("skinny") {
        push("slender");
        push("narrow waist");
    }
    if has("chubby") || has("plump") {
        push("chubby");
        push("soft chubby");
    }
    if has("wide hip") || has("broad hip") {
        push("wide hips");
    }
    if has("thick thigh") || has("thick leg") {
        push("thick thighs");
    }
    if has("narrow waist") || has("small waist") || has("tiny waist") {
        push("narrow waist");
    }
    if has("flat chest") || has("flat-chested") {
        push("flat chest");
    } else if has("small breast") || has("small chest") || has("petite breast") {
        push("small breasts");
    } else if has("medium breast") || has("moderate breast") {
        push("medium breasts");
    } else if (has("enormous") || has("gigantic") || has("massive")) && (has("breast") || has("chest") || has("bust")) {
        push("enormous breast size");
        push("huge breasts");
        if has_confirmed_cleavage {
            push("cleavage");
        }
    } else if has("huge breast") || has("large breast") || has("big breast") || has("busty") {
        push("large breasts");
        if has_confirmed_cleavage {
            push("cleavage");
        }
    }
    if has_confirmed_cleavage {
        push("cleavage");
    }
    if has("sideboob") {
        push("sideboob");
    }
    if has("underboob") {
        push("underboob");
    }
    if has("keyhole") {
        push("keyhole");
    }
    if has("off shoulder") || has("bare shoulders") {
        push("off shoulder");
        push("bare shoulders");
    }

    // Character hair, skin, headwear, hairpins, and wardrobe
    if has("fair skin") || has("pale skin") || has("light skin") {
        push("fair skin");
    }
    if has("black hair") || has("dark hair") {
        push("black hair");
    }
    if has("long hair") || has("long black hair") || has("long dark hair") {
        push("long hair");
    }
    if has("messy hair") || has("tousled") || has("loose strands") {
        push("messy hair");
    }
    if has("dark eyes") || has("brown eyes") {
        push("dark eyes");
    }
    // only when they are worn up there, not merely mentioned next to the word "hair" or "head"
    if has("sunglasses")
        && (has("on her head") || has("on his head") || has("on head") || has("on top of her head") || has("in her hair")
            || has("perched") || has("resting on") || has("pushed up"))
    {
        push("sunglasses on head");
        push("eyewear on head");
    }

    // Specific hairpins, ornaments, and accessories
    if has("flower hairpin") || has("flower hair ornament") || has("hair flower") {
        push("flower hair ornament");
    } else if has("butterfly hair ornament") || has("butterfly hairpin") {
        push("butterfly hair ornament");
    } else if has("star hair ornament") {
        push("star hair ornament");
    } else if has("heart hair ornament") {
        push("heart hair ornament");
    } else if has("x hair ornament") {
        push("x hair ornament");
    } else if has("hairclip") || has("hair clip") {
        push("hairclip");
    } else if has("hair ribbon") || has("hair bow") {
        push("hair ribbon");
        push("hair bow");
    } else if has("hairpin") {
        push("hairpin");
    } else if has("hairband") || has("headband") {
        push("hairband");
    } else if has("scrunchie") {
        push("scrunchie");
    } else if has("tiara") {
        push("tiara");
    } else if has("crown") {
        push("crown");
    } else if has("maid headdress") || has("headdress") {
        push("maid headdress");
    }
    if has("earring") || has("earrings") {
        push("earrings");
    }
    if has("choker") {
        push("choker");
    }
    if has("necklace") {
        push("necklace");
    }

    // Held items and hand actions
    if has("holding phone") || has("holding cellphone") || has("holding smartphone") || has("holding mobile") {
        push("holding phone");
    } else if has("holding cup") || has("holding mug") || has("holding teacup") {
        push("holding cup");
    } else if has("holding drink") || has("holding glass") || has("holding bottle") {
        push("holding drink");
    } else if has("holding book") {
        push("holding book");
    } else if has("holding umbrella") || has("holding parasol") {
        push("holding umbrella");
    } else if has("holding bag") || has("holding handbag") {
        push("holding bag");
    } else if has("holding sword") || has("holding blade") || has("holding katana") {
        push("holding sword");
    } else if has("holding weapon") || has("holding gun") {
        push("holding weapon");
    } else if has("holding staff") || has("holding wand") {
        push("holding staff");
    } else if has("holding fan") {
        push("holding fan");
    } else if has("holding flower") || has("holding bouquet") {
        push("holding flower");
    } else if has("holding food") {
        push("holding food");
    } else if has("holding plushie") || has("holding doll") {
        push("holding plushie");
    } else if has("holding camera") {
        push("holding camera");
    } else if has("holding microphone") {
        push("holding microphone");
    }

    // Hand gestures and pose anchors
    if has("hand on hip") || has("hands on hips") {
        push("hand on hip");
    }
    if has("hands on knees") || has("hand on knee") || has("hands on legs") {
        push("hands on knees");
    }
    if has("arms up") {
        push("arms up");
    }
    if has("crossed arms") || has("arms crossed") {
        push("crossed arms");
    }
    if has("peace sign") || has("v sign") {
        push("peace sign");
        push("v");
    }

    // Footwear, legwear, and straps
    if has("o-ring") && (has("strap") || has("thigh")) {
        push("o-ring thigh strap");
    } else if has("thigh strap") {
        push("thigh strap");
    }
    if has("garter strap") || has("garter belt") {
        push("garter straps");
    }
    if has("white shoe") || has("white sneakers") {
        push("white shoes");
    } else if has("shoe") || has("sneaker") {
        push("shoes");
    } else if has("boot") || has("boots") {
        push("boots");
    } else if has("heel") || has("high heels") {
        push("high heels");
    } else if has("sandal") || has("sandals") {
        push("sandals");
    }
    if has("white sock") || has("white socks") {
        push("white socks");
    }

    // Environmental plants and concrete objects
    if has("potted plant") || has("houseplant") {
        push("potted plant");
    } else if has("plant") && !has("plantation") {
        push("plant");
    }
    if has("flower") && !has("flower hair") && !has("holding flower") {
        push("flower");
    }
    if has("vase") {
        push("vase");
    }
    if has("window") {
        push("window");
    }
    if has("curtain") || has("curtains") {
        push("curtains");
    }
    if has("lamp") || has("lantern") {
        push("lamp");
    }
    if has("chair") || has("armchair") {
        push("chair");
    }
    if has("couch") || has("sofa") {
        push("couch");
    }
    if has("table") || has("desk") {
        push("table");
    }
    if has("bookshelf") || has("bookcase") {
        push("bookshelf");
    }
    if has("mirror") {
        push("mirror");
    }
    if has("clock") {
        push("clock");
    }
    // a light blue top only when one clause says so ("a light blue crop top"), not any top plus
    // "blue" anywhere (a blue jacket made every orange crop top light blue)
    let light_blue_top = clauses.iter().any(|c| {
        ["tube top", "crop top", "camisole", "tank top", "bikini top"].iter().any(|t| prose_has(c, t))
            && ["light blue", "pale blue", "baby blue"].iter().any(|b| prose_has(c, b))
    });
    if light_blue_top {
        push("light blue tube top");
    }
    if has("logo") || has("print") || has("text on") || has("clothes writing") || has("patterned") {
        push("print shirt");
        push("text on clothes");
    }
    if has("skin-tight") || has("skintight") || has("form-fitting") || has("taut") || has("tight") {
        push("impossible clothes");
        push("taut clothes");
    }
    if has("bikini bottom") || has("matching bottom") || has("matching bikini") {
        push("light blue bikini bottoms");
    }

    out
}

/// Sanitizes scanner tags, resolves camera framing paradoxes, purges archetype vs. uniform collisions,
/// and infers implicit full-body framing when footwear/legwear is detected.
pub fn antigravity_tag_filter(raw_tag_string: &str) -> String {
    const WIDE_SHOTS: &[&str] = &["full body", "cowboy shot", "wide shot"];
    const CLOSE_SHOTS: &[&str] = &["close-up", "portrait", "upper body", "face focus"];
    const TOPS: &[&str] = &[
        "bra_top", "shirt", "jacket", "tank top", "camisole", "bikini top", "tube top", "crop top", "sweater", "hoodie",
    ];
    const BOTTOMS: &[&str] = &[
        "pants", "skirt", "shorts", "bikini bottoms", "panties", "underwear", "dress", "jeans", "leggings",
    ];
    const STATIC_POSES: &[&str] = &["standing", "sitting", "kneeling"];
    const LIMB_ANCHORS: &[&str] = &[
        "hand", "hands", "arm", "arms", "holding", "reaching", "pulling", "adjusting", "crossed", "lap", "hips",
    ];
    const ARCHETYPE_OVERRIDES: &[&str] = &["bunny girl", "cat girl", "fox girl"];
    const UNIFORMS: &[&str] = &["maid", "maid outfit", "plaid dress", "school uniform", "serafuku", "suit"];

    let mut tags: Vec<String> = Vec::new();
    for raw in raw_tag_string.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let clean = raw
            .replace('(', "")
            .replace(')', "")
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if !clean.is_empty() && !tags.contains(&clean) {
            tags.push(clean);
        }
    }

    // 2. Implicit Full-Body Inference (Footwear/Legwear Trigger)
    let has_footwear = tags.iter().any(|t| {
        ["thighhighs", "mary janes", "boots", "heels", "barefoot", "sandals"]
            .iter()
            .any(|fw| t.contains(fw))
    });
    let has_cam = tags.iter().any(|t| {
        WIDE_SHOTS.contains(&t.as_str()) || CLOSE_SHOTS.contains(&t.as_str())
    });
    if has_footwear && !has_cam {
        tags.insert(0, "full body".to_string());
    }

    // 3. Camera Paradox Resolution (Wide overrides Close)
    let has_wide = tags.iter().any(|t| WIDE_SHOTS.contains(&t.as_str()));
    if has_wide {
        tags.retain(|t| !CLOSE_SHOTS.contains(&t.as_str()));
    }

    // 4. Archetype vs. Uniform Collision Purge
    let has_uniform = tags.iter().any(|t| UNIFORMS.iter().any(|u| t.contains(u)));
    if has_uniform {
        tags.retain(|t| !ARCHETYPE_OVERRIDES.contains(&t.as_str()));
    }

    // 5. Bottomless Hazard Flagging
    let has_top = tags.iter().any(|t| TOPS.iter().any(|top| t.contains(top)));
    let has_bottom = tags.iter().any(|t| BOTTOMS.iter().any(|bot| t.contains(bot)));
    if has_wide && has_top && !has_bottom {
        tags.push("MISSING_BOTTOMS_FLAG".to_string());
    }

    // 6. Unrouted Limb Flagging
    let has_static = tags.iter().any(|t| STATIC_POSES.contains(&t.as_str()));
    let has_limb_anchor = tags.iter().any(|t| LIMB_ANCHORS.iter().any(|l| t.contains(l)));
    if has_static && !has_limb_anchor {
        tags.push("UNANCHORED_LIMBS_FLAG".to_string());
    }

    tags.join(", ")
}

/// Strips redundant uncolored parent garment tags (`shirt`, `skirt`, `thighhighs`, `shoes`, `socks`, `bikini`, `bow`, `thigh strap`)
/// whenever a specific colored or modified child version (`white shirt`, `blue skirt`, `white thighhighs`, `o-ring thigh strap`) exists.
pub fn dedup_color_parents(prompt: &str) -> String {
    const PARENTS: &[&str] = &[
        "hair",
        "eyes",
        "dress",
        "china dress",
        "shirt",
        "jacket",
        "skirt",
        "shorts",
        "pants",
        "thighhighs",
        "kneehighs",
        "socks",
        "gloves",
        "boots",
        "shoes",
        "bikini",
        "bra",
        "panties",
        "bow",
        "ribbon",
        "tail",
        "ears",
        "thigh strap",
        "breasts",
        "underwear",
        "food",
    ];
    let items: Vec<&str> = prompt.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let norms: Vec<String> = items.iter().map(|s| key(s)).collect();
    items
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| {
            let n = &norms[*idx];
            if !PARENTS.contains(&n.as_str()) {
                return true;
            }
            // Drop bare parent if another token ends with ` <parent>` (e.g. `white thighhighs` -> drop `thighhighs`)
            let suffix = format!(" {n}");
            !norms.iter().enumerate().any(|(j, other)| j != *idx && other.ends_with(&suffix))
        })
        .map(|(_, raw)| raw)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Orders character tags into the 8 canonical Danbooru buckets from `master_synergy_rules.json`:
/// 1. Count/Quality -> 2. Framing -> 3. Character/Species -> 4. Proportions -> 5. Hair/Face -> 6. Outfit/Synergy -> 7. Pose/Action -> 8. Other
pub fn sort_canonical_buckets(prompt: &str) -> String {
    let tokens = crate::parser::parse_prompt(prompt);
    if tokens.is_empty() {
        return String::new();
    }
    let bucket_of = |t: &crate::parser::PromptToken| -> u8 {
        let n = t.normalized.as_str();
        if crate::two::count_tag(n).is_some() || n == "solo" || n == "multiple girls" || n == "multiple boys" {
            return 1;
        }
        if matches!(n, "full body" | "cowboy shot" | "upper body" | "lower body" | "portrait" | "close-up") {
            return 2;
        }
        if is_character(n) || n.ends_with(" ears") || n.ends_with(" tail") || n.ends_with(" wings") || n.ends_with(" horns") {
            return 3;
        }
        if crate::scene::is_body_shape_tag(n) || crate::scene::is_height_stature_tag(n) || n == "fair skin" || n == "dark skin" || n == "shiny skin" {
            return 4;
        }
        if n.contains("hair") || n.contains("bangs") || n.contains("twintails") || n.contains("ponytail") || n.contains("bob cut") || n.contains("ahoge")
            || n.contains("eyes") || n.contains("eyelashes") || n == "eyepatch" || n == "one eye covered" || n == "blush" || n.contains("smile") || n == "smiling"
            || n.contains("mouth") || n.contains("lips") || n == "teeth" || n == "oval face"
        {
            return 5;
        }
        if crate::sceneref::is_action(t) || crate::scene::is_skeletal_or_prop_pose(n) || n == "relaxed" || n == "relaxed pose" || n == "head tilt" {
            return 7;
        }
        6
    };
    let mut indexed: Vec<(u8, usize, String)> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| (bucket_of(&t), i, t.raw.trim().to_string()))
        .collect();
    indexed.sort_by_key(|(b, i, _)| (*b, *i));
    indexed.into_iter().map(|(_, _, raw)| raw).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_and_fixes_tags() {
        let l = parse(
            "tag_id,name,category,count\n1,holding_umbrella,0,50000\n2,umbrella,0,90000\n3,rain,0,80000\n\
             4,hatsune_miku,4,100000\n5,general,9,1\n6,ribbon,0,70000\n",
        );
        assert_eq!(l.get("Holding_Umbrella").unwrap().count, 50000);
        assert!(l.get("general").is_none());
        assert_eq!(l.fix("ribbons").unwrap().name, "ribbon");
        let c: Vec<&str> = l.candidates("a girl sheltering under umbrellas in the rain", 5, 20).iter().map(|t| t.name.as_str()).collect();
        assert_eq!(c, vec!["umbrella", "holding umbrella", "rain"]);
        let s: Vec<&str> = l.similar("holding parasol umbrella", 2).iter().map(|t| t.name.as_str()).collect();
        assert_eq!(s, vec!["holding umbrella", "umbrella"]);
        assert_eq!(
            tag_to_phrase("sunglasses on head", "her"),
            Some((PhraseKind::Wears, "sunglasses resting on her head".to_string()))
        );
        let extracted = extract_tags_from_prose(
            "A dynamic, softly lit indoor bedroom scene where a girl leans back on a bed with one leg raised in the air, using both hands to pull a white thigh-high stocking up toward her bare foot while looking directly at the viewer.",
        );
        assert!(extracted.contains(&"pulling up stocking"));
        assert!(extracted.contains(&"white thighhighs"));
        assert!(extracted.contains(&"bare foot"));
    }

    #[test]
    fn dual_reference_scene_and_character_merge_produces_recommended_output() {
        use crate::analyzer::ScanConfig;
        use crate::profiles::{CheckpointProfile, DetailLevel, TargetView};

        // Simulated raw tagger output from scene.jpg and character.jpg
        let raw_scene_tags = "1girl, solo, cowboy shot, side view, looking at viewer, blush, smile, brown hair, green eyes, hair between eyes, bare shoulders, white thighhighs, garter straps, thigh strap, o-ring, sitting, indoor, knee up, on bed, sitting on bed, no shoes, adjusting legwear";
        let raw_char_tags = "1girl, solo, black hair, long hair, hair bun, blush, smile, closed mouth, Looking at viewer, upper body, sunglasses, eyewear on head, sleeveless, spaghetti strap, camisole, blue theme, indoors, freckles, selfie, clothes writing, tank top, mature female, wearing a blue bikini";

        let actions = crate::sceneref::from_single(raw_scene_tags, false);
        let plan = crate::sceneref::plan(&actions, &crate::sceneref::Overrides::default(), 1);
        let char_identity = crate::scene::filter_identity_only(raw_char_tags);
        let merged_char = crate::sceneref::inject(&char_identity, &plan.chars[0], &crate::sceneref::is_action);

        let config = ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            camera_angle: 0,
            pose: 0,
            proportion: 0,
            dynamic: false,
            lora_tags_raw: String::new(),
            arranged: false,
            natural: true,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: true,
            scene_prompt_raw: plan.scene.join(", "),
            more_chars: Vec::new(),
            detail: DetailLevel::Balanced,
        };

        let result = crate::two::combine_section_blocks(&[merged_char], &config, true, None);
        println!("\n=== DUAL REFERENCE MERGE RESULT ===\n{result}\n===================================\n");
        assert!(result.contains("[Scene / Environment & Shared Framing]:"));
        assert!(result.contains("[Character 1]:"));
        assert!(result.contains("masterpiece, best quality, 1girl, solo"));
        assert!(result.contains("indoor, bedroom, sitting on bed, dynamic angle, depth of field, soft lighting"));
        assert!(result.contains("pulling up stocking"));
        assert!(result.contains("white thighhighs"));
        assert!(result.contains("bare foot"));
        assert!(result.contains("sunglasses on head, eyewear on head"));
        assert!(result.contains("light blue tube top, print shirt, text on clothes, impossible clothes, taut clothes, light blue bikini bottoms"));
        assert!(!result.contains("thigh strap"));
        assert!(!result.contains("garter"));
        assert!(!result.contains("freckles"));
    }
    #[test]
    fn negation_drops_only_the_negated_part() {
        let tags = extract_tags_from_prose("A girl with long hair with no shoes. She wears garter straps without boots.");
        for want in ["long hair", "bare foot", "garter straps"] {
            assert!(tags.contains(&want), "{want} missing: {tags:?}");
        }
        assert!(!tags.contains(&"shoes") && !tags.contains(&"boots"), "a negated item became a tag: {tags:?}");
    }

    #[test]
    fn prose_keywords_match_whole_words_only() {
        // what the image tests got wrong: "reference sheet" -> bedroom, "outfit" -> athletic,
        // "clamp" -> lamp, "massive tail" -> huge breasts, "headband" -> sunglasses on head
        let wrong = extract_tags_from_prose(
            "A character design reference sheet. Her outfit is visible from the back view, with a massive bushy tail, \
             a hair clamp, a headband and sunglasses in her bag. An outfitted adventurer in a plantation.",
        );
        for bad in ["bedroom", "sitting on bed", "athletic", "fit", "lamp", "huge breasts", "sunglasses on head", "plant", "leaning back"] {
            assert!(!wrong.contains(&bad), "invented {bad:?} from {wrong:?}");
        }
        // real matches still work, word endings included
        let right = extract_tags_from_prose(
            "She sits on the bed, leaning back, sunglasses resting on her head. Toned build, gigantic breasts, \
             black boots, earrings, a potted plant by the window and a lamp on the table.",
        );
        for good in ["bedroom", "sitting on bed", "leaning back", "sunglasses on head", "athletic", "huge breasts", "boots", "earrings", "potted plant", "window", "lamp", "table"] {
            assert!(right.contains(&good), "missed {good:?} in {right:?}");
        }
        assert!(prose_has("she is sitting down", "sit") && prose_has("two lamps", "lamp"));
        assert!(!prose_has("clearly visible", "sit") && !prose_has("an outfit", "fit") && !prose_has("a clamp", "lamp"));
    }

    #[test]
    fn negated_checklist_answers_add_no_tags() {
        // how the vision model answers the zone checklist, from the raccoon-girl test image
        let tags = extract_tags_from_prose(
            "Orange crop top with buttons. No keyhole cutout is visible, and there is no sideboob. \
             She is not wearing garter straps. A blue jacket hangs off her shoulders.",
        );
        for bad in ["keyhole", "sideboob", "garter straps", "light blue tube top"] {
            assert!(!tags.contains(&bad), "{bad:?} from a negation or a stray 'blue': {tags:?}");
        }
        // what really is there still counts, and the rules that are about an absence still work
        assert!(extract_tags_from_prose("She wears a light blue crop top.").contains(&"light blue tube top"));
        assert!(extract_tags_from_prose("She stands with no shoes on.").contains(&"bare foot"));
        assert!(extract_tags_from_prose("A clear keyhole cutout on her top.").contains(&"keyhole"));
    }

}
