//! Part of `ai_engine`; see mod.rs.
use super::*;

pub(crate) const COLOR_WORDS: &[&str] = &[
    "white", "black", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "grey", "gray", "silver",
    "gold", "golden", "light", "dark", "pale", "navy", "beige", "cream", "teal", "aqua", "violet", "magenta", "crimson",
    "turquoise", "lavender", "maroon", "burgundy", "coral", "indigo", "cyan", "amber", "olive", "charcoal", "emerald",
    "ruby", "bronze", "copper", "blonde", "hazel",
];

/// Materials and patterns: a phrase like "sheer black dress" stays one descriptive tag.
pub(crate) const MATERIAL_WORDS: &[&str] = &[
    "sheer", "glossy", "lace", "latex", "leather", "silk", "satin", "velvet", "denim", "metallic", "striped", "plaid",
    "checkered", "frilled", "ribbed",
];

/// Real hues (not light / dark / pale, and not black / white / grey, which greyscale can show).
const HUE_WORDS: &[&str] = &[
    "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "blonde", "navy", "teal", "aqua", "violet",
    "magenta", "crimson", "turquoise", "lavender", "maroon", "burgundy", "coral", "indigo", "cyan", "amber", "olive",
    "emerald", "ruby", "hazel", "gold", "golden", "silver", "beige", "cream", "bronze", "copper",
];
const NEUTRAL_WORDS: &[&str] = &["black", "white", "grey", "gray"];

/// The tags the checklist form states reliably: framing (how much of the body, from which side),
/// background and gaze. Deterministic, so they can't be forgotten by the writer (it kept dropping
/// `cowboy shot` and the background color). The pose line is left to the writer: the form gets it wrong
/// too often.
pub fn tags_from_form(form: &str) -> Vec<String> {
    // "LABEL: answer", where the answer may also continue on the following lines (bullets) up to
    // the next "LABEL:" line
    let lines: Vec<String> = form.lines().map(|l| l.trim().trim_start_matches(['*', '-', ' ']).replace("**", "")).collect();
    let is_label = |l: &str| l.split_once(':').is_some_and(|(k, _)| !k.is_empty() && k.chars().all(|c| c.is_ascii_uppercase() || c == ' '));
    let field = |name: &str| -> String {
        let Some(i) = lines.iter().position(|l| l.to_uppercase().starts_with(&format!("{name}:"))) else { return String::new() };
        let mut text = lines[i][name.len() + 1..].trim().to_string();
        for l in lines.iter().skip(i + 1).take_while(|l| !is_label(l)) {
            text.push(' ');
            text.push_str(l);
        }
        text.trim().to_lowercase()
    };
    let has = |text: &str, w: &str| crate::danbooru::prose_has_pub(text, w);
    let mut out: Vec<String> = Vec::new();
    let framing = field("FRAMING");
    if !framing.is_empty() {
        let to = framing.rsplit(" to ").next().unwrap_or(&framing).to_string();
        if has(&to, "feet") || has(&to, "toe") || has(&to, "shoe") || has(&framing, "full body") || has(&framing, "entire body") {
            out.push("full body".into());
        } else if has(&to, "thigh") || has(&to, "knee") || has(&to, "mid-thigh") {
            out.push("cowboy shot".into());
        } else if has(&to, "waist") || has(&to, "hip") || has(&to, "chest") {
            out.push("upper body".into());
        }
        if has(&framing, "behind") || has(&framing, "back view") || has(&framing, "from the back") {
            out.push("from behind".into());
        } else if has(&framing, "side") || has(&framing, "profile") {
            out.push("from side".into());
        } else if has(&framing, "above") {
            out.push("from above".into());
        } else if has(&framing, "below") {
            out.push("from below".into());
        }
    }
    let background = field("BACKGROUND");
    if !background.is_empty() {
        if ["plain", "solid", "simple", "blank", "no specific", "nothing", "empty"].iter().any(|w| background.contains(w)) {
            out.push("simple background".into());
        }
        if let Some(c) = ["grey", "gray", "white", "black", "blue", "pink", "yellow", "green", "red", "purple", "brown", "orange"]
            .iter()
            .find(|c| has(&background, c))
        {
            let c = if *c == "gray" { "grey" } else { c };
            out.push(format!("{c} background"));
        }
    }
    let gaze = field("GAZE AND EXPRESSION");
    if has(&gaze, "over her shoulder") || has(&gaze, "over his shoulder") || has(&gaze, "over shoulder") || has(&gaze, "looking back") {
        out.push("looking back".into());
    }
    if has(&gaze, "viewer") || has(&gaze, "camera") || has(&gaze, "looking forward") {
        out.push("looking at viewer".into());
    }
    if has(&gaze, "smile") || has(&gaze, "smiling") {
        out.push("smile".into());
    }
    out
}

/// Phrases models write that are no tag, mapped to the tag they mean ("" = drop it).
pub(crate) fn phrase_to_tag(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "standing sideways" | "side view" | "side profile" | "facing sideways" | "facing left" | "facing right" => "from side",
        "back view" | "facing away" | "facing away from viewer" | "standing away from viewer" => "from behind",
        "looking over shoulder" | "looking over her shoulder" => "looking back",
        "curvy hips" | "broad hips" => "wide hips",
        "medium length hair" | "medium-length hair" | "shoulder-length hair" | "shoulder length hair" => "medium hair",
        "smiling" | "gentle smile" | "slight smile" => "smile",
        "hands behind back" | "hands behind her back" => "arms behind back",
        "plain background" => "simple background",
        "off the shoulders" | "off the shoulder" | "off-shoulder" | "off shoulders" => "off shoulder",
        "plain grey background" | "plain gray background" => "grey background",
        "soft shadows" | "even lighting" | "athletic build" | "slim build" | "blue trim" | "visible tail" | "tail visible" => "",
        _ => return None,
    })
}

/// Checks each tag against the tagger's Danbooru list (`danbooru::local`, ~10,000 common tags): a known
/// tag stays, a near miss is corrected (`fix`), a color / material / style word in front of a known tag
/// stays (`black leather shorts`), a known phrase is mapped (`phrase_to_tag`), anything else multi-word
/// is dropped. Without the list everything is kept.
pub fn validate_tags(tags: Vec<String>) -> Vec<String> {
    let Some(list) = crate::danbooru::local() else { return tags };
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let l = t.to_lowercase();
        // weights "(tag:1.2)", LoRAs and dynamic groups pass untouched; a "Name: tags" label does not count
        if l.contains('<') || l.contains('{') || (l.contains('(') && l.contains(')')) {
            out.push(t);
            continue;
        }
        if let Some((label, rest)) = t.split_once(':') {
            // "Nagi: black hair ponytail" -> "black hair ponytail" (the name is a label, not a tag)
            if !label.contains(' ') && !rest.trim().is_empty() {
                out.extend(validate_tags(vec![rest.trim().to_string()]));
                continue;
            }
        }
        if let Some(m) = phrase_to_tag(&l) {
            if !m.is_empty() {
                out.push(m.to_string());
            }
            continue;
        }
        if list.get(&l).is_some() || matches!(l.as_str(), "masterpiece" | "best quality" | "score 9" | "score 8 up" | "score 7 up") {
            out.push(t);
            continue;
        }
        if let Some(fixed) = list.fix(&l) {
            out.push(fixed.name.clone());
            continue;
        }
        // An unknown phrase made of known tags plus filler ("from behind looking back at viewer",
        // "tight white shirt", "parted lips showing upper teeth") becomes those tags. Kept whole when the
        // only extra words are colors ("teal bikini top"): that is a description worth keeping.
        if let Some(pieces) = known_pieces(&l, list) {
            out.extend(pieces);
            continue;
        }
        // Writer filler about the drawing rather than the picture ("anime style", "soft shading", "japanese text
        // decoration", "clenched hand in foreground"): the meta words go; what's left stays only if it's a known
        // tag ("japanese text", "clenched hand"). Known tags with these words (`backlighting`) never get here.
        if let Some(rest) = strip_meta_words(&l) {
            if list.get(&rest).is_some() || list.fix(&rest).is_some() {
                out.push(list.get(&rest).map(|t| t.name.clone()).or_else(|| list.fix(&rest).map(|t| t.name.clone())).unwrap_or(rest));
            }
            continue;
        }
        // Unknown multi-word tags are kept: the tag list is only the ~10,000 most common tags, and
        // descriptive ones ("white teal bikini top") are what Anima reads best. Dropping them emptied a
        // whole prompt in testing. Only phrases about nothing visible go.
        const NOT_VISUAL: &[&str] = &["visible", "build", "shadows", "trim", "tone", "tones", "accents", "expression", "vibe", "aesthetic"];
        if !l.split_whitespace().any(|w| NOT_VISUAL.contains(&w)) {
            out.push(t);
        }
    }
    out
}

/// Words that describe the drawing, not what's in it.
const META_WORDS: &[&str] = &[
    "style", "styled", "shading", "shaded", "decoration", "decorations", "decorative", "foreground", "rendering",
    "rendered", "drawn", "artstyle",
];

/// `Some(the phrase without its meta words and the little words left dangling)` when the phrase has a meta
/// word; `None` otherwise.
fn strip_meta_words(phrase: &str) -> Option<String> {
    let words: Vec<&str> = phrase.split_whitespace().collect();
    if !words.iter().any(|w| META_WORDS.contains(w)) {
        return None;
    }
    let mut kept: Vec<&str> = words.into_iter().filter(|w| !META_WORDS.contains(w)).collect();
    const DANGLING: &[&str] = &["in", "on", "the", "a", "with", "of", "and"];
    while kept.last().is_some_and(|w| DANGLING.contains(w)) {
        kept.pop();
    }
    while kept.first().is_some_and(|w| DANGLING.contains(w)) {
        kept.remove(0);
    }
    Some(kept.join(" "))
}

/// Words of a named uniform tag that say nothing about which school or army (`tomoeda` and `ooarai` are names).
const GENERIC_UNIFORM_WORDS: &[&str] = &[
    "school", "elementary", "middle", "high", "military", "sailor", "gym", "maid", "nurse", "police", "waitress",
    "chef", "baseball", "soccer", "track", "sports", "office", "japanese", "naval", "navy", "army", "kindergarten",
    "cheerleader", "basketball", "volleyball", "tennis", "band", "marching", "summer", "winter", "uniform", "shrine",
];

/// Named uniforms (`tomoeda elementary school uniform`, `ooarai military uniform`) are dropped when the vision
/// model's description doesn't name that school / army: the tagger is very sure of them on look-alike
/// outfits (0.96 and 0.89 on a magical-girl costume). Without a description (`form` empty) nothing changes.
pub fn drop_unbacked_named_uniforms(tags: Vec<String>, form: &str) -> Vec<String> {
    if form.trim().is_empty() {
        return tags;
    }
    let desc = form.to_lowercase();
    tags.into_iter()
        .filter(|t| {
            let k = crate::danbooru::key(t);
            if !k.ends_with(" uniform") {
                return true;
            }
            let names: Vec<&str> = k.split_whitespace().filter(|w| !GENERIC_UNIFORM_WORDS.contains(w)).collect();
            names.is_empty() || names.iter().any(|n| crate::danbooru::prose_has_pub(&desc, n))
        })
        .collect()
}

/// The known tags inside an unknown phrase, longest match first, left to right. `None` when there are
/// none, or when the words left over are only colors (then the phrase stays as a description).
fn known_pieces(phrase: &str, list: &crate::danbooru::TagList) -> Option<Vec<String>> {
    let words: Vec<&str> = phrase.split_whitespace().collect();
    if words.len() < 2 {
        return None;
    }
    let mut pieces = Vec::new();
    let mut leftover = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let found = (i + 1..=words.len()).rev().find(|&j| {
            let cand = words[i..j].join(" ");
            // single words only when they are tags on their own and not a bare color or size
            (j - i > 1 || !(COLOR_WORDS.contains(&words[i]) || matches!(words[i], "large" | "small" | "long" | "short" | "medium")))
                && list.get(&cand).is_some()
        });
        match found {
            Some(j) => {
                pieces.push(words[i..j].join(" "));
                i = j;
            }
            None => {
                leftover.push(words[i]);
                i += 1;
            }
        }
    }
    // "looking back at viewer": the gaze tag is split by the matcher, so it is put back
    if words.contains(&"looking") && phrase.contains("at viewer") && !pieces.iter().any(|p| p == "looking at viewer") {
        pieces.push("looking at viewer".to_string());
        leftover.retain(|w| !matches!(*w, "at" | "viewer"));
    }
    // Kept whole when only colors are left over, or when the last word (the thing described, as in
    // "teal bikini top") is not part of a known tag.
    let head_covered = !leftover.last().is_some_and(|w| Some(w) == words.last());
    let only_colors_left =
        !leftover.is_empty() && leftover.iter().all(|w| COLOR_WORDS.contains(w) || MATERIAL_WORDS.contains(w));
    if pieces.is_empty() || only_colors_left || !head_covered {
        return None;
    }
    Some(pieces)
}

/// Splits a tag the writer ran together without commas ("blue jacket open jacket off the shoulders")
/// into the known tags it is made of ("blue jacket", "open jacket", "off shoulder"). Only when the whole
/// phrase is not a tag itself and every word is covered by known tags (or `phrase_to_tag` phrases), using
/// as few pieces as possible, so descriptive tags ("very short black leather shorts") stay whole.
pub fn split_merged_tags(tag: String) -> Vec<String> {
    let Some(list) = crate::danbooru::local() else { return vec![tag] };
    let l = tag.to_lowercase();
    let words: Vec<&str> = l.split_whitespace().collect();
    if words.len() < 4 || list.get(&l).is_some() || l.contains(['(', ':', '<', '{']) {
        return vec![tag];
    }
    let piece = |i: usize, j: usize| -> Option<String> {
        let p = words[i..j].join(" ");
        if let Some(m) = phrase_to_tag(&p) {
            return (!m.is_empty()).then(|| m.to_string());
        }
        list.get(&p).map(|t| t.name.clone())
    };
    // fewest pieces covering every word (pieces of 1-5 words)
    let n = words.len();
    let mut best: Vec<Option<Vec<String>>> = vec![None; n + 1];
    best[0] = Some(Vec::new());
    for j in 1..=n {
        for i in j.saturating_sub(5)..j {
            if let (Some(prev), Some(p)) = (best[i].clone(), piece(i, j)) {
                let mut cand = prev;
                cand.push(p);
                if best[j].as_ref().is_none_or(|b| cand.len() < b.len()) {
                    best[j] = Some(cand);
                }
            }
        }
    }
    match best[n].take() {
        Some(parts) if parts.len() >= 2 => parts,
        _ => vec![tag],
    }
}

/// `barefoot` / `bare feet` next to footwear can't both be true. Legwear (thighhighs, socks, pantyhose) can
/// go with bare feet on Danbooru, so only shoes count.
pub fn drop_contradicted_feet(tags: Vec<String>) -> Vec<String> {
    let shod = tags.iter().any(|t| {
        let l = t.to_lowercase();
        ["shoes", "sneakers", "boots", "sandals", "heels", "loafers", "mary janes", "uwabaki", "footwear", "slippers", "geta"]
            .iter()
            .any(|f| l == *f || l.ends_with(&format!(" {f}")))
    });
    if !shod {
        return tags;
    }
    tags.into_iter().filter(|t| !matches!(t.to_lowercase().as_str(), "barefoot" | "bare feet" | "bare foot")).collect()
}

/// Hair tags that contradict a "short hair with long locks" description (twintails, long hair): the
/// tagger reads the long locks as twintails or long hair, and the writer copies both.
pub fn drop_contradicted_hair(tags: Vec<String>) -> Vec<String> {
    let lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    let short_with_locks = lower.iter().any(|t| t == "short hair with long locks")
        || (lower.iter().any(|t| t == "short hair") && lower.iter().any(|t| t.contains("sidelocks")));
    if !short_with_locks {
        return tags;
    }
    tags.into_iter()
        .filter(|t| !matches!(t.to_lowercase().as_str(), "twintails" | "low twintails" | "long hair" | "very long hair"))
        .collect()
}

/// One tag per body part where two can't both be true: eye color (unless heterochromia), skin tone,
/// hair length (unless the tags say long locks / sidelocks), boot height. Of two, a tag from the
/// Danbooru list beats an unknown one, then the first written wins.
pub fn one_per_part(tags: Vec<String>) -> Vec<String> {
    let list = crate::danbooru::local();
    let known = |t: &str| list.is_some_and(|l| l.get(t).is_some());
    let lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    // Two or more characters can each have their own eyes, hair, skin and size (the eye rule had dropped
    // Aoba's purple eyes next to Lillie's green ones), so nothing is reduced then.
    let group = lower.iter().any(|t| {
        crate::two::count_tag(t).is_some_and(|(n, _)| n >= 2) || matches!(t.as_str(), "multiple girls" | "multiple boys")
    });
    if group {
        return tags;
    }
    let hetero = lower.iter().any(|t| t == "heterochromia");
    let colorish_word = |w: &str| COLOR_WORDS.contains(&w) || w.contains('-') || w.ends_with("ish");
    let part = |t: &str| -> Option<&'static str> {
        let first = t.split_whitespace().next().unwrap_or("");
        let colorish = colorish_word(first);
        if t.ends_with(" eyes") && colorish && !hetero {
            return Some("eye color");
        }
        // "pink hair", "light brown hair", "pinkish-orange hair"; not multicolored / two-tone / streaked
        if let Some(color) = t.strip_suffix(" hair") {
            if !color.is_empty() && color.split_whitespace().all(|w| colorish_word(w)) {
                return Some("hair color");
            }
        }
        if t.ends_with(" skin") && t != "shiny skin" && t != "oily skin" {
            return Some("skin tone");
        }
        if matches!(t, "very short hair" | "short hair" | "medium hair" | "long hair" | "very long hair" | "absurdly long hair") {
            return Some("hair length");
        }
        if matches!(t, "thigh boots" | "knee boots" | "ankle boots" | "knee high boots" | "knee-high boots" | "thigh-high boots" | "thigh high boots") {
            return Some("boot height");
        }
        if BREAST_SIZES.contains(&t) {
            return Some("breast size");
        }
        None
    };
    let mut chosen: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    for (i, t) in lower.iter().enumerate() {
        if let Some(p) = part(t) {
            match chosen.get(p) {
                None => {
                    chosen.insert(p, i);
                }
                Some(&j) if !known(&lower[j]) && known(t) => {
                    chosen.insert(p, i);
                }
                _ => {}
            }
        }
    }
    tags.into_iter()
        .enumerate()
        .filter(|(i, t)| part(&t.to_lowercase()).is_none_or(|p| chosen.get(p) == Some(i)))
        .map(|(_, t)| t)
        .collect()
}

/// The last pass over an image prompt written from a checklist form: the form's framing / background /
/// gaze tags are added, every tag is checked against the Danbooru list, one tag per body part is kept,
/// duplicates go, and the tagger's sure tags are enforced last.
/// `polish_image_prompt`, plus: a Danbooru character name stays only if the tagger's tags (`tagger_tags`)
/// name that character too. Vision models invent identities (ToriiGate called an OC "Musashi from Azur
/// Lane"); the tagger only reports characters it recognizes.
pub fn polish_image_prompt_with(
    prompt: &str,
    form: &str,
    strong: &[String],
    tagger_tags: &str,
    tagger_scores: &[(String, f32)],
) -> String {
    let tagger: std::collections::HashSet<String> = tagger_tags.split(',').map(crate::danbooru::key).collect();
    let tags: Vec<String> = prompt
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .filter(|t| !crate::danbooru::is_character(t) || tagger.contains(&crate::danbooru::key(t)))
        .collect();
    let mut tags: Vec<String> = tags.into_iter().flat_map(split_merged_tags).collect();
    for t in tags_from_form(form) {
        if !tags.iter().any(|x| crate::danbooru::key(x) == crate::danbooru::key(&t)) {
            tags.push(t);
        }
    }
    // The tagger's tags come back before the one-per-part checks, so they get the same checks as the
    // writer's, and the writer's (listed first) wins a conflict.
    // PC_NO_RECONCILE=1 skips it, to measure what it adds (testing only)
    let tags = if std::env::var_os("PC_NO_RECONCILE").is_some() {
        validate_tags(tags)
    } else {
        restore_tagger_tags(validate_tags(tags), tagger_tags, form, tagger_scores)
    };
    let tags = drop_unbacked_named_uniforms(drop_contradicted_feet(drop_contradicted_hair(one_per_part(tags))), form);
    let mut seen = std::collections::HashSet::new();
    let tags: Vec<String> = tags.into_iter().filter(|t| seen.insert(crate::danbooru::key(t))).collect();
    let tags = crate::danbooru::dedup_color_parents(&tags.join(", "));
    enforce_strong_tags(&tags, strong)
}

/// Defaults a small writer adds without looking: kept only when the tagger or the description backs them.
const UNBACKED_DEFAULTS: &[&str] =
    &["looking at viewer", "full body", "thigh gap", "kemonomimi", "taut clothes", "impossible clothes"];
/// How many of the tagger's tags may come back after the writer (measured offline: ~18-20 is the best F1).
const MAX_RECOVERED: usize = 18;
/// With scores known, every tag the tagger is sure of (>= `RECOVERY_MIN`) comes back; this only guards against
/// a runaway list. On the 12 test images the cap of 18 cut correct sure tags on busy pictures (loose socks,
/// cardigan, armor, blazer); without it recall rose 8-10 points. `PC_RECOVERY_MAX` overrides it (testing only).
const MAX_RECOVERED_SURE: usize = 60;

fn recovery_max(scored: bool) -> usize {
    if !scored {
        return MAX_RECOVERED;
    }
    std::env::var("PC_RECOVERY_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_RECOVERED_SURE)
}
/// The lowest tagger score a recovered tag needs (when scores are known). `PC_RECOVERY_MIN` overrides it
/// for measuring (testing only).
const RECOVERY_MIN: f32 = 0.5;

fn recovery_min() -> f32 {
    std::env::var("PC_RECOVERY_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(RECOVERY_MIN)
}
/// Tagger tags that say nothing on their own (a specific tag says it better) or aren't visible content.
const RECOVERY_SKIP: &[&str] = &[
    "breasts", "shirt", "skirt", "socks", "shoes", "jewelry", "underwear", "tail", "animal ears", "multiple girls",
    "multiple boys", "thighs", "legs", "ribbon", "bag", "food", "plaid", "clothes", "bow", "gloves", "wings", "jacket",
    "belt", "boots", "dress", "pants", "shorts", "sleeves", "virtual youtuber", "alternate breast size",
    "character name", "copyright name", "artist name", "twitter username", "signature", "watermark",
    "parody", "style parody", "retro artstyle", "alternate costume", "adapted costume",
];

/// Breast sizes, smallest first (for `one_per_part` and the tagger's size authority).
pub(crate) const BREAST_SIZES: &[&str] =
    &["flat chest", "small breasts", "medium breasts", "large breasts", "huge breasts", "gigantic breasts"];

/// Reconciles the writer's tags with the tagger's (the writer dropped 202 of 314 missed answer tags that the
/// tagger had right): unbacked defaults go; a breast size the tagger is >= 0.5 sure of replaces the writer's;
/// then the tagger's sure tags (all of them when scores are known, else up to `MAX_RECOVERED`), in its order, are appended (not ones already there,
/// quality / count / solo, or `RECOVERY_SKIP`). Conflicts are left to `one_per_part`, which keeps the first.
pub fn restore_tagger_tags(mut tags: Vec<String>, tagger_tags: &str, form: &str, tagger_scores: &[(String, f32)]) -> Vec<String> {
    let key = crate::danbooru::key;
    let tagger: Vec<String> = tagger_tags.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let tagger_keys: std::collections::HashSet<String> = tagger.iter().map(|t| key(t)).collect();
    let form_lower = form.to_lowercase();
    tags.retain(|t| {
        let k = key(t);
        !UNBACKED_DEFAULTS.contains(&k.as_str())
            || tagger_keys.contains(&k)
            || crate::danbooru::prose_has_pub(&form_lower, &k)
    });
    // size authority (breasts only: the tagger's hair length is often wrong, e.g. "long hair" for short
    // hair with long locks)
    if let Some((size, _)) = tagger_scores
        .iter()
        .filter(|(t, s)| *s >= 0.5 && BREAST_SIZES.contains(&key(t).as_str()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
    {
        if let Some(at) = tags.iter().position(|t| BREAST_SIZES.contains(&key(t).as_str())) {
            tags.retain(|t| !BREAST_SIZES.contains(&key(t).as_str()));
            tags.insert(at.min(tags.len()), size.clone());
        }
    }
    let mut present: std::collections::HashSet<String> = tags.iter().map(|t| key(t)).collect();
    let scores: std::collections::HashMap<String, f32> = tagger_scores.iter().map(|(t, s)| (key(t), *s)).collect();
    let min = recovery_min();
    let max = recovery_max(!scores.is_empty());
    let mut added = 0;
    for t in &tagger {
        if added >= max {
            break;
        }
        let k = key(t);
        if present.contains(&k)
            || is_quality_tag_any(&k)
            || k == "solo"
            || crate::two::count_tag(&k).is_some()
            || RECOVERY_SKIP.contains(&k.as_str())
            // only tags the tagger is sure enough of (unknown score: no scores were passed, keep the old behavior)
            || scores.get(&k).is_some_and(|s| *s < min)
        {
            continue;
        }
        present.insert(k);
        tags.push(t.clone());
        added += 1;
    }
    tags
}

/// The tagger's style tags that mean "no colors in this picture".
const MONOCHROME_TAGS: &[&str] = &["monochrome", "greyscale", "sketch", "lineart"];
/// What the tagger (and then the writer) see in a sketch's construction lines and joints.
const SKETCH_ARTIFACTS: &[&str] =
    &["robot", "humanoid robot", "android", "joints", "doll joints", "robot joints", "mechanical parts", "no humans", "mannequin"];

/// True when at least 1.5% of the pixels are clearly colored (saturation > 0.35, neither near black
/// nor near white). Measured on the test images: pencil sketches on warm paper 0.0%, a peach lineart
/// chart 0.4%, every color picture 2.8% or more. (A mean color difference fails here: warm photographed
/// paper scores like a color picture.)
pub fn is_colorful(img: &image::DynamicImage) -> bool {
    let small = img.resize(400, 400, image::imageops::FilterType::Triangle).to_rgb8();
    let total = (small.width() * small.height()).max(1) as f32;
    let strong = small
        .pixels()
        .filter(|p| {
            let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            max > 40.0 && max < 250.0 && (max - min) / max > 0.35
        })
        .count() as f32;
    strong / total >= 0.015
}

/// The tagger's tags with `monochrome` / `greyscale` removed when the picture is clearly colorful: the
/// tagger says monochrome for a color painting with small grey insets, ToriiGate then repeats it, and
/// the sketch cleanup strips every color. Unchanged when the image can't be read.
pub fn color_checked_hints(tag_hints: &str, path: &std::path::Path) -> String {
    let colorful = image::open(path).is_ok_and(|img| is_colorful(&img));
    if !colorful {
        return tag_hints.to_string();
    }
    tag_hints
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty() && !matches!(crate::danbooru::key(t).as_str(), "monochrome" | "greyscale"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// True when the tagger says the picture is a monochrome sketch / line drawing.
pub fn is_monochrome(tagger_tags: &str) -> bool {
    tagger_tags.split(',').map(crate::danbooru::key).any(|t| MONOCHROME_TAGS.contains(&t.as_str()))
}

/// Finds full-height vertical divider lines (a dark line splitting the picture into panels, like a
/// two-panel comparison) and returns the number of panels: 1 when there is none. Only lines in the
/// middle half count, so a frame border isn't one.
pub fn panel_count(img: &image::DynamicImage) -> usize {
    let small = img.resize(400, 400, image::imageops::FilterType::Triangle).to_luma8();
    let (w, h) = small.dimensions();
    if w < 40 || h < 40 {
        return 1;
    }
    let dark_share = |x: u32| (0..h).filter(|&y| small.get_pixel(x, y)[0] < 70).count() as f32 / h as f32;
    let shares: Vec<f32> = (0..w).map(dark_share).collect();
    let mut dividers = 0;
    let mut x = w / 4;
    while x < w * 3 / 4 {
        if shares[x as usize] >= 0.92 {
            // a run of dark columns is one line; it must be thin and stand out from both sides
            let start = x;
            while x < w && shares[x as usize] >= 0.92 {
                x += 1;
            }
            let thin = x - start <= w / 25 + 1;
            let left = shares[start.saturating_sub(6) as usize];
            let right = shares[(x + 5).min(w - 1) as usize];
            if thin && left < 0.5 && right < 0.5 {
                dividers += 1;
            }
        }
        x += 1;
    }
    dividers + 1
}

/// The writer's extra instructions for a sketch or a picture split into panels.
pub fn image_style_note(tagger_tags: &str, panels: usize) -> String {
    let mut note = String::new();
    if is_monochrome(tagger_tags) {
        note.push_str(
            "\n\n## Style: monochrome sketch\nThis picture is a monochrome sketch / line drawing. It has no colors, so write \
             no color tags (hair, eyes, clothes). Construction lines, circles and joints are drawing guides, not robot or doll \
             parts. Keep the style tags (monochrome, sketch, lineart, traditional media).",
        );
    }
    if panels > 1 {
        note.push_str(&format!(
            "\n\n## Panels\nThe picture is split into {panels} panels that show the same character in different poses or \
             outfits. Write `multiple views`, count the character once (1girl, not 2girls), and tag what each panel shows: \
             every pose, outfit and held item."
        ));
    }
    note
}

/// Last fixes on an image prompt: a sketch loses color tags and robot / joint tags and keeps its style
/// tags; a picture with panels gets `multiple views`; character names the tagger recognized (>= 0.85)
/// are put back if the writer dropped them.
pub fn apply_image_style(prompt: &str, tagger_tags: &str, panels: usize) -> String {
    let tagger: Vec<String> = tagger_tags.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let mut tags: Vec<String> = prompt.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let has = |tags: &[String], t: &str| tags.iter().any(|x| crate::danbooru::key(x) == crate::danbooru::key(t));
    if is_monochrome(tagger_tags) {
        let line_drawing = tagger.iter().any(|t| matches!(crate::danbooru::key(t).as_str(), "sketch" | "lineart"));
        tags.retain(|t| {
            let k = crate::danbooru::key(t);
            if SKETCH_ARTIFACTS.contains(&k.as_str()) {
                return false;
            }
            // Skin tone (screentone / shading shows it), backgrounds ("white background" on paper), and
            // light / dark words that aren't colors ("light blush", "dark aura") stay. A hue on hair, eyes or
            // clothes goes; black / white / grey ones only on an uncolored line drawing, since filled
            // greyscale manga can show black hair.
            if k.ends_with(" skin") || k.contains("-skinned") || k == "tan" || k.ends_with("background") {
                return true;
            }
            let words: Vec<&str> = k.split_whitespace().collect();
            let colored = words.iter().any(|w| HUE_WORDS.contains(w) || (line_drawing && NEUTRAL_WORDS.contains(w)));
            let looks = crate::parser::parse_prompt(t).into_iter().next().is_some_and(|tok| {
                matches!(crate::scene::slot(&tok), crate::scene::Slot::Hair | crate::scene::Slot::Eyes | crate::scene::Slot::Outfit)
            });
            !(colored && looks)
        });
        for style in ["monochrome", "greyscale", "sketch", "lineart", "traditional media"] {
            if tagger.iter().any(|t| crate::danbooru::key(t) == style) && !has(&tags, style) {
                tags.push(style.to_string());
            }
        }
    }
    // The tagger saw no people (a logo, scenery): the writer may not invent a girl. Not for sketches,
    // where "no humans" comes from the construction lines.
    if !is_monochrome(tagger_tags) && tagger.iter().any(|t| crate::danbooru::key(t) == "no humans") {
        tags.retain(|t| {
            let k = crate::danbooru::key(t);
            crate::two::count_tag(&k).is_none() && !matches!(k.as_str(), "solo" | "multiple girls" | "multiple boys")
        });
        if !has(&tags, "no humans") {
            let at = tags.iter().position(|t| !is_quality_tag_any(&crate::danbooru::key(t))).unwrap_or(tags.len());
            tags.insert(at, "no humans".to_string());
        }
    }
    if panels > 1 && !has(&tags, "multiple views") {
        tags.push("multiple views".to_string());
    }
    // recognized characters go right after the quality / count tags
    let names: Vec<String> =
        tagger.iter().filter(|t| crate::danbooru::is_character(t) && !has(&tags, t)).cloned().collect();
    if !names.is_empty() {
        let at = tags
            .iter()
            .position(|t| {
                let k = crate::danbooru::key(t);
                !(is_quality_tag_any(&k) || crate::two::count_tag(&k).is_some() || k == "solo")
            })
            .unwrap_or(tags.len());
        for (i, n) in names.into_iter().enumerate() {
            tags.insert(at + i, n);
        }
    }
    tags.join(", ")
}

