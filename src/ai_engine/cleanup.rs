//! Part of `ai_engine`; see mod.rs.
use super::*;

/// Rules a small model sometimes ignores, applied to its reply: Flux has no negative prompt,
/// and the checkpoint's quality tags lead the rewrite (AI text only; the rule-based output
/// never gets tags the user didn't write).
pub(crate) fn enforce_checkpoint_rules(out: &mut AiJsonOutput, checkpoint: CheckpointProfile) {
    if checkpoint.default_negative().is_empty() {
        out.recommended_negative.clear();
    }
    out.improved_prompt = strip_placeholders(&out.improved_prompt);
    out.dynamic_prompt = strip_placeholders(&out.dynamic_prompt);
    out.recommended_negative = strip_placeholders(&out.recommended_negative);
    out.improved_prompt = ensure_quality_tags(&out.improved_prompt, checkpoint);
    out.dynamic_prompt = ensure_quality_tags(&out.dynamic_prompt, checkpoint);
}

/// Removes what a model copies out of its instructions instead of writing a tag:
/// `[UNANCHORED_LIMBS_FLAG resolved: hands in lap]` becomes `hands in lap`, a bare `*_FLAG`
/// goes, and so do placeholder tags such as `[framing]`, `{none}` or
/// `{missing framing tag placeholder}`. Real tags, weights `(tag:1.2)` and dynamic groups
/// `{a|b}` are kept; the separators and line breaks between the kept tags stay as written.
pub fn strip_placeholders(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    // Only words that are never a tag of their own. Real tags in brackets are A1111 / ComfyUI
    // de-emphasis (`[background]`, `[lighting]`, `[missing tooth]`) and must stay.
    const PLACEHOLDER_WORDS: &[&str] = &["none", "n/a", "null", "nil", "empty", "tbd", "todo", "...", "framing", "tag", "tags"];
    let placeholder = |inner: &str| {
        let l = inner.trim().to_lowercase();
        l.is_empty() || PLACEHOLDER_WORDS.contains(&l.as_str()) || l.contains("placeholder") || l.contains("_flag")
    };
    // "[UNANCHORED_LIMBS_FLAG resolved: hands in lap, legs crossed]" spans commas: resolve whole
    // brackets first (ASCII lowercasing keeps byte offsets valid in `text`)
    let mut text = text.to_string();
    while let Some(at) = text.to_ascii_lowercase().find("_flag resolved:") {
        let open = text[..at].rfind('[');
        let close = text[at..].find(']').map(|c| at + c);
        match (open, close) {
            (Some(o), Some(c)) => {
                let inner = text[at + "_flag resolved:".len()..c].trim().to_string();
                text.replace_range(o..=c, &inner);
            }
            _ => break, // no brackets around it: the per-tag rule below handles it
        }
    }
    let text = text.as_str();
    let mut kept: Vec<String> = Vec::new();
    for seg in text.split(',') {
        let lead: String = seg.chars().take_while(|c| c.is_whitespace()).collect();
        let tag = seg.trim();
        // "[X_FLAG resolved: hands in lap]" -> "hands in lap"
        // ASCII lowercasing keeps every byte where it was, so `i` is valid in `tag` (full Unicode
        // lowercasing can change lengths and made this slice panic on some characters)
        if let Some(i) = tag.to_ascii_lowercase().find("resolved:") {
            if tag.contains("_FLAG") || tag.contains("_flag") {
                let fix = tag[i + "resolved:".len()..].trim().trim_end_matches([']', ')', '}']).trim();
                if !fix.is_empty() {
                    kept.push(format!("{lead}{fix}"));
                }
                continue;
            }
        }
        // a bare flag in any spelling: "MISSING_BOTTOMS_FLAG", "missing_bottoms_flag", "(MISSING_BOTTOMS_FLAG:1.1)"
        let bare = tag.trim_matches(['(', ')', '[', ']']).split(':').next().unwrap_or("").trim().to_ascii_lowercase();
        if bare.ends_with("_flag") && !bare.contains(' ') {
            continue;
        }
        let bracketed = (tag.starts_with('[') && tag.ends_with(']')) || (tag.starts_with('{') && tag.ends_with('}'));
        if bracketed && !tag.contains('|') && placeholder(&tag[1..tag.len() - 1]) {
            continue;
        }
        kept.push(seg.to_string());
    }
    let joined = kept.join(",");
    // a dropped first tag leaves the next one's leading space at the very start
    if text.starts_with(char::is_whitespace) { joined } else { joined.trim_start().to_string() }
}

/// Turns a prompt a model wrote in a section layout back into one flat tag list: `[Scene …]:` and
/// `[Character 1]:` labels go, a `[tag, tag]:` header and `(tag, tag)` groups are unwrapped (weights
/// such as `(tag:1.2)` stay), prose sentences of 8+ words go (the app writes its own sentence), and
/// repeated tags are kept once.
pub fn flatten_section_layout(prompt: &str) -> String {
    if !prompt.contains("]:") && !prompt.contains("[Scene") && !prompt.contains("[Character") {
        return prompt.to_string();
    }
    let mut text = prompt.to_string();
    // "[...]:" headers: a label is dropped, a tag list inside is kept
    while let Some(open) = text.find('[') {
        let Some(rel) = text[open..].find("]:") else { break };
        let close = open + rel;
        let inner = text[open + 1..close].to_string();
        let keep = if inner.contains(',') { format!("{inner},") } else { String::new() };
        text.replace_range(open..close + 2, &keep);
    }
    // "(tag, tag, …)" groups without a weight
    let mut out = String::new();
    let mut depth_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        match c {
            '(' if depth_start.is_none() => depth_start = Some(i),
            ')' if depth_start.is_some() => {
                let start = depth_start.take().unwrap_or(i);
                let inner = &text[start + 1..i];
                if inner.contains(',') && !inner.contains(':') {
                    // commas around it, so the next words (often a sentence) don't fuse with the last tag
                    out.push_str(", ");
                    out.push_str(inner);
                    out.push_str(", ");
                } else {
                    out.push_str(&text[start..=i]);
                }
            }
            _ if depth_start.is_none() => out.push(c),
            _ => {}
        }
    }
    if let Some(start) = depth_start {
        out.push_str(&text[start..]);
    }
    let mut seen = std::collections::HashSet::new();
    let tags: Vec<String> = out
        .split([',', '\n'])
        .map(|t| t.trim().trim_end_matches('.').trim().to_string())
        .filter(|t| !t.is_empty() && t.split_whitespace().count() < 8)
        .filter(|t| seen.insert(t.to_lowercase()))
        .collect();
    tags.join(", ")
}

/// A tag the tagger can be trusted on when it is very sure: skin tone, animal species parts
/// (`raccoon ears`, `fox tail`, `cat girl`) and person counts (`1girl`, `2boys`).
pub(crate) fn is_guarded_tag(name: &str) -> bool {
    let t = name.to_lowercase();
    let count = t.chars().next().is_some_and(|c| c.is_ascii_digit())
        && ["girl", "girls", "boy", "boys", "other", "others"].iter().any(|s| t.ends_with(s));
    let skin = t.ends_with(" skin") || t.contains("-skinned") || t == "tan" || t == "tanlines";
    // only an animal's part or an animal person: "raccoon ears", "fox girl", not "striped tail" or "magical girl"
    let first = t.split_whitespace().next().unwrap_or("");
    let species = ["ears", "tail", "horns", "wings", "girl", "boy"].iter().any(|p| t.ends_with(&format!(" {p}")))
        && is_animal_word(first)
        && !count;
    count || skin || species || species_word_animal(&t).is_some()
}

/// Animals whose name starts a species tag (`raccoon ears`, `fox girl`). `bunny girl` counts as a rabbit, not
/// the costume: next to a sure `cat girl` it is a conflict far more often.
pub(crate) const ANIMAL_WORDS: &[&str] = &[
    "cat", "dog", "fox", "wolf", "raccoon", "rabbit", "bunny", "mouse", "bear", "cow", "horse", "tiger", "lion",
    "sheep", "goat", "squirrel", "bat", "bird", "dragon", "deer", "monkey", "pig", "snake", "shark", "lizard",
    "tanuki", "ferret", "weasel", "otter", "panda", "hamster", "koala",
];

/// True for an animal name, or a one-word species tag (`kitsune`).
pub(crate) fn is_animal_word(word: &str) -> bool {
    ANIMAL_WORDS.contains(&word) || species_word_animal(word).is_some()
}

/// The animal a tag is about: `raccoon tail` → raccoon, `kitsune` → fox, `bunny girl` → rabbit.
fn tag_animal(tag: &str) -> Option<&'static str> {
    if let Some(a) = species_word_animal(tag) {
        return Some(a);
    }
    let first = tag.split_whitespace().next()?;
    let a = ANIMAL_WORDS.iter().find(|w| **w == first).copied()?;
    Some(if a == "bunny" { "rabbit" } else { a })
}

/// The animal a one-word species tag stands for (`kitsune` → fox, `nekomimi` → cat …).
pub(crate) fn species_word_animal(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "kitsune" | "kyuubi" => "fox",
        "nekomimi" | "catgirl" | "nekomata" => "cat",
        "inumimi" | "doggirl" => "dog",
        "usagimimi" | "bunnygirl" => "rabbit",
        "okamimimi" | "wolfgirl" => "wolf",
        "tanuki" => "tanuki",
        "kemonomimi mode" => return None,
        _ => return None,
    })
}

/// The tagger's skin / species / count tags it is at least 80% sure of (prompt spelling).
pub fn strong_tags(tags: &[crate::tagger::ScoredTag]) -> Vec<String> {
    tags.iter()
        .filter(|t| t.score >= 0.8 && is_guarded_tag(&t.name.replace('_', " ")))
        .map(|t| t.name.replace('_', " "))
        .collect()
}

/// Every tagger tag with its score (prompt spelling): `restore_tagger_tags` only brings back sure ones,
/// and a sure breast size beats the writer's.
pub fn tag_scores(tags: &[crate::tagger::ScoredTag]) -> Vec<(String, f32)> {
    tags.iter().map(|t| (t.name.replace('_', " "), t.score)).collect()
}

/// Puts the tagger's sure tags back when the writer dropped or contradicted them: a different animal on
/// the same part (`cat ears` vs. a sure `raccoon ears`) is replaced, an opposite skin tone removed, a
/// different count of the same kind replaced; missing ones are added after the leading quality tags.
pub fn enforce_strong_tags(prompt: &str, strong: &[String]) -> String {
    if strong.is_empty() || prompt.trim().is_empty() {
        return prompt.to_string();
    }
    let mut tags: Vec<String> = prompt.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let key = |t: &str| crate::danbooru::key(t);
    // sure tags never push each other out ("striped tail" is a pattern, not a rival to "raccoon tail")
    let sure_keys: std::collections::HashSet<String> = strong.iter().map(|t| key(t)).collect();
    for sure in strong {
        let s = sure.to_lowercase();
        let part = ["ears", "tail", "horns", "wings", "girl", "boy"].iter().find(|p| s.ends_with(&format!(" {p}"))).copied();
        let is_count = s.chars().next().is_some_and(|c| c.is_ascii_digit());
        let is_skin = s.ends_with(" skin") || s.contains("-skinned");
        tags.retain(|t| {
            let l = t.to_lowercase();
            if key(&l) == key(&s) || sure_keys.contains(&key(&l)) {
                return true;
            }
            if is_count {
                let kind = if s.contains("girl") { "girl" } else if s.contains("boy") { "boy" } else { "other" };
                return !(l.chars().next().is_some_and(|c| c.is_ascii_digit()) && l.contains(kind));
            }
            if is_skin {
                let dark = s.contains("dark") || s.contains("tan") || s.contains("brown");
                let opposite = if dark { ["pale skin", "fair skin", "white skin", "light skin"] } else { ["dark skin", "tan skin", "brown skin", "dark-skinned female"] };
                return !opposite.contains(&l.as_str());
            }
            if let (Some(p), Some(sure_animal)) = (part, tag_animal(&s)) {
                // Another animal on the same part ("cat ears" vs "raccoon ears"), or a species tag for another
                // animal ("kitsune" / "fox girl" next to a sure "raccoon girl"). Only tags that name an animal:
                // "striped tail", "fluffy tail", "magical girl", "school girl" and "animal ears" stay.
                let animal = tag_animal(&l);
                let same_part = l.ends_with(&format!(" {p}")) && l.split_whitespace().count() == 2;
                let person = l.ends_with(" girl") || l.ends_with(" boy") || species_word_animal(&l).is_some();
                let rival = animal.is_some_and(|a| a != sure_animal) && (same_part || person);
                return !rival;
            }
            true
        });
        if !tags.iter().any(|t| key(t) == key(&s)) {
            let quality = ["masterpiece", "best quality", "score 9", "score_9", "score 8 up", "score_8_up", "score 7 up", "score_7_up", "absurdres", "highres"];
            let at = tags.iter().take_while(|t| quality.contains(&t.to_lowercase().as_str())).count();
            tags.insert(at, sure.clone());
        }
    }
    tags.join(", ")
}

/// Removes instruction text a model copied into its prompt: tag-count notes ("25-45 tags total",
/// "tag count: 40") and explanatory parentheses ("raccoon tail (inferred from tagger/context)" becomes
/// "raccoon tail"). Weights such as "(red hair:1.2)" and real parenthesised tags are kept.
pub fn strip_instruction_leaks(prompt: &str) -> String {
    const NOTE_WORDS: &[&str] = &[
        "inferred", "tagger", "context", "note", "assumed", "assume", "guess", "likely", "possibly", "maybe",
        "if visible", "optional", "approx", "uncertain", "presumably",
    ];
    let mut kept: Vec<String> = Vec::new();
    for seg in prompt.split(',') {
        let lead: String = seg.chars().take_while(|c| c.is_whitespace()).collect();
        let mut tag = seg.trim().to_string();
        let l = tag.to_lowercase();
        let count_note = l.contains("tags total")
            || l.contains("tag count")
            || l.contains("total tags")
            || (l.ends_with(" tags") && l.chars().next().is_some_and(|c| c.is_ascii_digit()));
        if count_note {
            continue;
        }
        // "(...)" with no weight inside and a note word: an explanation, not part of the tag
        if let (Some(open), Some(close)) = (tag.find('('), tag.rfind(')')) {
            if close > open {
                let inner = tag[open + 1..close].to_lowercase();
                if !inner.contains(':') && NOTE_WORDS.iter().any(|w| inner.contains(w)) {
                    tag = format!("{}{}", tag[..open].trim_end(), &tag[close + 1..]).trim().to_string();
                }
            }
        }
        if !tag.is_empty() {
            kept.push(format!("{lead}{tag}"));
        }
    }
    let joined = kept.join(",");
    if prompt.starts_with(char::is_whitespace) { joined } else { joined.trim_start().to_string() }
}

/// Puts the checkpoint's missing quality tags at the front when the prompt lacks its first one.
/// An empty prompt stays empty.
pub fn ensure_quality_tags(prompt: &str, checkpoint: CheckpointProfile) -> String {
    let quality = checkpoint.canonical_quality_tags();
    let Some(first) = quality.first() else { return prompt.to_string() };
    if prompt.trim().is_empty() {
        return prompt.to_string();
    }
    // "score_9", "score 9", "[score_9]" and "(score_9:1.1)" are the same tag
    let norm = |t: &str| {
        t.trim().trim_matches(['(', ')', '[', ']']).split(':').next().unwrap_or("").trim().to_lowercase().replace('_', " ")
    };
    let present: Vec<String> = prompt.split(|c| c == ',' || c == '\n').map(norm).collect();
    if present.iter().any(|t| *t == norm(first)) {
        return prompt.to_string();
    }
    let missing: Vec<&str> = quality.iter().copied().filter(|q| !present.iter().any(|t| *t == norm(q))).collect();
    format!("{}, {}", missing.join(", "), prompt.trim_start())
}
