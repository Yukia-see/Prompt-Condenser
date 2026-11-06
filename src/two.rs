//! The 2 Characters tab. Each character's prompt goes through the normal rule engine
//! on its own (so duplicates, LoRA conflicts and framing swaps stay per character),
//! then `combine` builds one prompt that keeps each character's tags together:
//!
//! quality, count (2girls / 1girl, 1boy), interaction | character 1 | BREAK |
//! character 2 | camera, setting, light, style, sentence
//!
//! Character 1's box also holds the scene; scene tags from either box are shared.

use crate::analyzer::{strip_lora_tokens, ScanConfig};
use crate::parser::{parse_prompt, PromptToken};
use crate::profiles::CheckpointProfile;
use crate::scene::{slot, Slot};

/// Rule cards for character 2 are numbered from here (character 1's start at 0,
/// AI cards at `ai_engine::AI_CARD_ID_BASE`); see `card_base`.
pub const CHAR2_CARD_ID_BASE: usize = 5_000;

/// Where character `i`'s (0-based) rule card ids start.
pub fn card_base(i: usize) -> usize {
    if i == 0 { 0 } else { CHAR2_CARD_ID_BASE + (i - 1) * 1_000 }
}

/// Tags about the characters together; they go in the shared header.
const INTERACTION: &[&str] = &[
    "holding hands", "hand holding", "facing each other", "side by side", "back-to-back", "couple", "hug",
    "hugging", "kiss", "kissing", "looking at another", "eye contact", "arm in arm", "arms around neck",
    "carrying", "princess carry", "piggyback", "hug from behind", "head on another's shoulder", "yuri", "yaoi",
    "hetero", "sharing", "tickling", "cheek-to-cheek", "face-to-face", "symmetrical docking", "group hug",
    "multiple girls", "multiple boys", "interlocked fingers", "group picture", "lineup",
];

/// `1girl` -> (1, "girl"), `2boys` -> (2, "boy"), `6+girls` -> (6, "girl").
pub(crate) fn count_tag(n: &str) -> Option<(u32, &'static str)> {
    let digits: String = n.chars().take_while(|c| c.is_ascii_digit()).collect();
    let num: u32 = digits.parse().ok()?;
    let rest = n[digits.len()..].trim_start_matches('+');
    let kind = match rest {
        "girl" | "girls" => "girl",
        "boy" | "boys" => "boy",
        "other" | "others" => "other",
        _ => return None,
    };
    Some((num, kind))
}

/// Where a shared tag sorts: tags the keyword lists don't know (Scene-box words like
/// "petals") count as setting.
fn footer_slot(t: &PromptToken) -> Slot {
    match slot(t) {
        s @ (Slot::Camera | Slot::Setting | Slot::Light | Slot::Style | Slot::Clause | Slot::Quality) => s,
        _ => Slot::Setting,
    }
}

pub(crate) fn is_interaction(n: &str) -> bool {
    INTERACTION.contains(&n) || n.starts_with("holding another's") || n.ends_with(" each other")
}

/// BREAK only means something to CLIP text encoders (and to UIs that split on it).
pub fn break_applies(config: &ScanConfig) -> bool {
    config.char_break && config.checkpoint != CheckpointProfile::Flux
}

/// Splits "a … BREAK b … BREAK c" at every standalone BREAK.
pub fn split_breaks(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let (mut parts, mut start, mut i) = (Vec::new(), 0, 0);
    while let Some(off) = text[i..].find("BREAK") {
        let at = i + off;
        let before = at == 0 || !b[at - 1].is_ascii_alphanumeric();
        let after = at + 5 >= b.len() || !b[at + 5].is_ascii_alphanumeric();
        if before && after {
            parts.push(text[start..at].to_string());
            start = at + 5;
        }
        i = at + 5;
    }
    parts.push(text[start..].to_string());
    parts.into_iter().map(|p| p.trim().trim_matches(',').trim().to_string()).filter(|p| !p.is_empty()).collect()
}

/// Extracts section blocks if the text contains bracketed headers like `[Scene...]:` or `[Character...]:`.
/// Returns `Some((char_prompts, scene_prompt))` if at least one character or scene block was found.
pub fn parse_section_blocks(text: &str) -> Option<(Vec<String>, String)> {
    if !text.contains('[') || (!text.contains("]:") && !text.contains(']')) {
        return None;
    }
    let mut scene = String::new();
    let mut chars = Vec::new();
    let mut current_type: Option<bool> = None; // true = scene, false = char
    let mut current_lines: Vec<String> = Vec::new();

    let flush = |ctype: Option<bool>, clines: &mut Vec<String>, sc: &mut String, ch: &mut Vec<String>| {
        let content = clines.join(", ").trim().trim_matches(',').trim().to_string();
        clines.clear();
        if content.is_empty() {
            return;
        }
        match ctype {
            Some(true) => {
                if sc.is_empty() {
                    *sc = content;
                } else {
                    sc.push_str(", ");
                    sc.push_str(&content);
                }
            }
            Some(false) => {
                ch.push(content);
            }
            None => {}
        }
    };

    for line in text.lines() {
        let trimmed = line.trim();
        let is_header = trimmed.starts_with('[') && (trimmed.contains("]:") || trimmed.ends_with(']'));
        if is_header {
            flush(current_type, &mut current_lines, &mut scene, &mut chars);
            let header_lower = trimmed.to_lowercase();
            if header_lower.contains("scene") || header_lower.contains("environment") || header_lower.contains("framing") {
                current_type = Some(true);
            } else {
                current_type = Some(false);
            }
            let rest = if let Some((_, r)) = trimmed.split_once("]:") {
                r.trim()
            } else if let Some((_, r)) = trimmed.split_once(']') {
                r.trim()
            } else {
                ""
            };
            if !rest.is_empty() {
                current_lines.push(rest.to_string());
            }
        } else if !trimmed.is_empty() {
            current_lines.push(trimmed.to_string());
        }
    }
    flush(current_type, &mut current_lines, &mut scene, &mut chars);

    if chars.is_empty() && scene.is_empty() {
        None
    } else {
        Some((chars, scene))
    }
}

/// The prompts to analyze, one per character: the boxes as typed, or, with the other
/// boxes empty, a pasted "… BREAK … BREAK …" prompt or Section Blocks prompt split up.
pub fn prompts(char1: &str, others: &[String]) -> Vec<String> {
    let n = others.len() + 1;
    if others.iter().all(|o| o.trim().is_empty()) {
        if let Some((chars, scene)) = parse_section_blocks(char1) {
            if !chars.is_empty() {
                let mut out: Vec<String> = chars.into_iter().take(n).collect();
                if !scene.is_empty() && !out.is_empty() {
                    out[0] = if out[0].is_empty() {
                        scene
                    } else {
                        format!("{}, {}", out[0].trim_end_matches(',').trim(), scene)
                    };
                }
                out.resize(n, String::new());
                return out;
            }
        }
        let parts = split_breaks(char1);
        if parts.len() >= 2 {
            let has_chars = parts.iter().any(|p| describes_a_character(p));
            if has_chars {
                let (mut char_parts, shared_parts): (Vec<String>, Vec<String>) =
                    parts.into_iter().partition(|p| describes_a_character(p));
                let mut out: Vec<String> = if char_parts.len() > n {
                    let mut head: Vec<String> = char_parts.drain(..n - 1).collect();
                    head.push(char_parts.join(", "));
                    head
                } else {
                    char_parts
                };
                out.resize(n, String::new());
                let shared_str = shared_parts.join(", ");
                if !shared_str.is_empty() && !out.is_empty() {
                    out[0] = if out[0].is_empty() {
                        shared_str
                    } else {
                        format!("{}, {}", out[0].trim_end_matches(',').trim(), shared_str)
                    };
                }
                return out;
            } else {
                let mut out: Vec<String> = parts.iter().take(n).cloned().collect();
                if parts.len() > n {
                    out[n - 1] = parts[n - 1..].join(", ");
                }
                out.resize(n, String::new());
                return out;
            }
        }
    }
    std::iter::once(char1.to_string()).chain(others.iter().cloned()).collect()
}

/// Does this BREAK part describe a character (hair, eyes, body, outfit, traits), or only
/// shared things (quality, count, interaction, camera, setting, light)?
fn describes_a_character(part: &str) -> bool {
    character_signal(part) > 0
}

/// How strongly a BREAK part looks like one character: a 1girl-style count tag counts a
/// lot; otherwise it takes at least two look / outfit tags, so one stray word in the
/// shared scene can't make it a "character".
fn character_signal(part: &str) -> usize {
    let toks = parse_prompt(part);
    let count = toks.iter().any(|t| matches!(count_tag(&t.normalized), Some((1, _))));
    let looks = toks.iter().filter(|t| matches!(slot(t), Slot::Traits | Slot::Hair | Slot::Eyes | Slot::Outfit)).count();
    if count {
        100 + looks
    } else if looks >= 2 {
        looks
    } else {
        0
    }
}

/// An AI-written combined prompt, split back into one part per character, plus the
/// shared tags. The AI is asked to put BREAK between the characters, and often also
/// fences off the opening tags and the shared scene with BREAKs: parts that describe no
/// character are shared. The character parts are matched to the boxes they resemble
/// most (the model may change the order). Without a clean split, each tag goes to the
/// box it came from (character 1 for new ones).
pub fn split_ai(text: &str, boxes: &[String]) -> (Vec<String>, String) {
    let n = boxes.len();
    let set = |s: &str| parse_prompt(s).into_iter().map(|t| t.normalized).collect::<Vec<_>>();
    let box_sets: Vec<Vec<String>> = boxes.iter().map(|b| set(b)).collect();
    let all_parts = split_breaks(text);
    let (mut parts, mut shared): (Vec<String>, Vec<String>) = all_parts.iter().cloned().partition(|p| describes_a_character(p));
    // more character-looking parts than characters: the weakest ones are shared after all
    if parts.len() > n && n > 0 {
        let mut ranked: Vec<(usize, usize)> = parts.iter().enumerate().map(|(i, p)| (character_signal(p), i)).collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let keep: Vec<usize> = ranked.iter().take(n).map(|x| x.1).collect();
        type IndexedPart = (usize, String);
        let (kept, extra): (Vec<IndexedPart>, Vec<IndexedPart>) =
            parts.into_iter().enumerate().partition(|(i, _)| keep.contains(i));
        parts = kept.into_iter().map(|x| x.1).collect();
        shared.extend(extra.into_iter().map(|x| x.1));
    }
    if parts.len() == n {
        let part_sets: Vec<Vec<String>> = parts.iter().map(|p| set(p)).collect();
        let hits = |x: &[String], y: &[String]| x.iter().filter(|t| y.contains(t)).count();
        let identity: Vec<usize> = (0..n).collect();
        let score = |perm: &[usize]| perm.iter().enumerate().map(|(i, &p)| hits(&part_sets[p], &box_sets[i])).sum::<usize>();
        let mut best = identity.clone();
        for perm in permutations(n) {
            if score(&perm) > score(&best) {
                best = perm;
            }
        }
        let shared = shared.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(", ");
        return (best.iter().map(|&p| parts[p].clone()).collect(), shared);
    }
    let mut out: Vec<Vec<String>> = vec![Vec::new(); n];
    for t in parse_prompt(text) {
        if t.normalized == "break" {
            continue;
        }
        let owner = (1..n).find(|&i| box_sets[i].contains(&t.normalized) && !box_sets[0].contains(&t.normalized)).unwrap_or(0);
        out[owner].push(t.raw);
    }
    (out.into_iter().map(|v| v.join(", ")).collect(), String::new())
}

/// `combine` for an AI-written prompt: the shared part goes in as Scene-box tags, so it
/// stays shared whatever the keyword lists think of its words.
pub fn combine_ai(text: &str, boxes: &[String], config: &ScanConfig, sentence: Option<&str>) -> Combined {
    let (mut parts, shared) = split_ai(text, boxes);
    if shared.is_empty() {
        return combine(&parts, config, true, sentence);
    }
    let mut cfg = config.clone();
    cfg.scene_prompt_raw = [cfg.scene_prompt_raw.trim(), shared.as_str()].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", ");
    if let Some(first) = parts.first_mut() {
        *first = [shared.as_str(), first.as_str()].join(", ");
    }
    combine(&parts, &cfg, true, sentence)
}

/// Every ordering of 0..n (n is at most 4).
pub(crate) fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n <= 1 {
        return vec![(0..n).collect()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for pos in 0..=rest.len() {
            let mut p = rest.clone();
            p.insert(pos, n - 1);
            out.push(p);
        }
    }
    out
}

/// Where character `i` of `n` stands, for Anima's clauses.
pub fn position(i: usize, n: usize) -> &'static str {
    match (n, i) {
        (2, 0) | (3, 0) => "on the left",
        (2, _) | (3, 2) => "on the right",
        (3, _) => "in the middle",
        (_, 0) => "on the far left",
        (_, 1) => "second from the left",
        (_, 2) => "second from the right",
        _ => "on the far right",
    }
}

/// Position tags that describe spatial character placement across the canvas.
pub fn is_spatial_position(n: &str) -> bool {
    matches!(
        n,
        "on the left"
            | "on the right"
            | "in the middle"
            | "on the far left"
            | "on the far right"
            | "second from the left"
            | "second from the right"
            | "far left"
            | "far right"
            | "center"
            | "middle"
    )
}

const ORDINALS: [&str; 4] = ["first", "second", "third", "fourth"];

/// A combined prompt plus its parts (for the live scene description).
pub struct Combined {
    pub text: String,
    /// Each character's own tags.
    pub chars: Vec<Vec<PromptToken>>,
    /// Count, interaction and scene tags.
    pub shared: Vec<PromptToken>,
}

/// Builds the combined prompt from the characters' finished prompts (one per character).
/// `strip_conflicts` also drops tags that contradict a LoRA box (used on AI text).
pub fn combine(texts: &[String], config: &ScanConfig, strip_conflicts: bool, sentence: Option<&str>) -> Combined {
    let n = texts.len();
    let tokens = |text: &str, lora: &str| -> Vec<PromptToken> {
        crate::analyzer::fix_body_type(strip_lora_tokens(parse_prompt(text), lora, strip_conflicts))
            .into_iter()
            .filter(|t| t.normalized != "break")
            // "solo" contradicts several characters (the rule cards remove it; AI text too)
            .filter(|t| !(strip_conflicts && t.normalized == "solo"))
            .collect()
    };
    let chars: Vec<Vec<PromptToken>> = texts.iter().enumerate().map(|(i, t)| tokens(t, config.lora_of(i))).collect();
    let scene_box: Vec<String> = parse_prompt(&config.scene_prompt_raw).into_iter().map(|t| t.normalized).collect();

    let mut quality: Vec<PromptToken> = Vec::new();
    let mut group_counts: Vec<PromptToken> = Vec::new(); // "2girls" written by the user
    let mut kinds: Vec<&'static str> = Vec::new(); // each character's "girl" / "boy" / "other"
    let mut kind_of: Vec<Option<&'static str>> = vec![None; n];
    let mut interaction: Vec<PromptToken> = Vec::new();
    let mut footer: Vec<PromptToken> = Vec::new();
    let mut blocks: Vec<Vec<PromptToken>> = vec![Vec::new(); n];
    let push_unique = |v: &mut Vec<PromptToken>, t: &PromptToken| {
        if !v.iter().any(|x| x.normalized == t.normalized) {
            v.push(t.clone());
        }
    };
    for (i, toks) in chars.iter().enumerate() {
        let mut kind = None;
        for t in toks {
            let n = t.normalized.as_str();
            if let Some((num, k)) = count_tag(n) {
                if num >= 2 {
                    push_unique(&mut group_counts, t);
                } else {
                    kind = kind.or(Some(k));
                }
                continue;
            }
            if is_interaction(n) {
                push_unique(&mut interaction, t);
                continue;
            }
            // anything from the Scene box is shared, whatever it is
            if scene_box.iter().any(|x| x == n) {
                if slot(t) == Slot::Quality {
                    push_unique(&mut quality, t);
                } else {
                    push_unique(&mut footer, t);
                }
                continue;
            }
            match slot(t) {
                Slot::Quality => push_unique(&mut quality, t),
                Slot::Camera | Slot::Setting | Slot::Light | Slot::Style | Slot::Clause => push_unique(&mut footer, t),
                _ => {
                    if !is_spatial_position(n) {
                        blocks[i].push(t.clone());
                    }
                }
            }
        }
        kinds.extend(kind);
        kind_of[i] = kind;
    }

    // the combined count: what the user wrote for the group, else from each character's tag
    let count_text = if !group_counts.is_empty() {
        group_counts.iter().map(|t| t.raw.trim().to_string()).collect::<Vec<_>>().join(", ")
    } else {
        let count = |k: &str| kinds.iter().filter(|x| **x == k).count();
        let mut parts = Vec::new();
        for k in ["girl", "boy", "other"] {
            match count(k) {
                0 => {}
                1 => parts.push(format!("1{k}")),
                c => parts.push(format!("{c}{k}s")),
            }
        }
        parts.join(", ")
    };
    let count_tokens = parse_prompt(&count_text);

    if config.cleanup_enabled || config.arranged {
        footer.sort_by_key(footer_slot); // camera, setting, light, style, sentences
    }
    let natural = config.natural && config.checkpoint.supports_natural_language();
    let sentence = sentence.map(str::trim).filter(|s| natural && !s.is_empty());
    let raw = |v: &[PromptToken]| v.iter().map(|t| t.raw.trim().to_string()).collect::<Vec<String>>();
    let lora = |s: &str| s.trim().trim_matches(',').trim().to_string();
    let block = |i: usize| {
        let mut b = blocks[i].clone();
        if config.arranged {
            b.sort_by_key(slot);
        }
        let mut items = Vec::new();
        if !lora(config.lora_of(i)).is_empty() {
            items.push(lora(config.lora_of(i)));
        }
        items.extend(raw(&b));
        items.join(", ")
    };

    let text = if config.checkpoint == CheckpointProfile::Anima {
        if config.char_break {
            // Anima Pencil-XL / SDXL default: BREAK chunk fencing between scene header and characters
            let sep = "BREAK";
            let noun = |i: usize| match kind_of[i] {
                Some("girl") => "girl",
                Some("boy") => "boy",
                _ => "character",
            };
            let header = [raw(&quality), raw(&count_tokens), raw(&interaction), raw(&footer)].concat().join(", ");
            let mut groups = vec![header];
            for i in 0..n {
                let pos = position(i, n);
                let b = block(i);
                // For one line include BREAK, but for line separation (arranged) don't include BREAK
                if !config.arranged {
                    groups.push(sep.to_string());
                }
                if b.is_empty() {
                    groups.push(format!("1{}, {pos}", noun(i)));
                } else {
                    groups.push(format!("1{}, {pos}, {b}", noun(i)));
                }
            }
            if let Some(sen) = sentence {
                groups.push(sen.to_string());
            }
            let groups: Vec<String> = groups.into_iter().filter(|g| !g.trim().is_empty()).collect();
            groups.join(if config.arranged { ",\n" } else { ", " })
        } else {
            // Relational clauses fallback when char_break is disabled
            let clause = |i: usize| {
                let noun = match kind_of[i] {
                    Some("girl") => "girl",
                    Some("boy") => "boy",
                    _ => "character",
                };
                let mut b = blocks[i].clone();
                b.sort_by_key(slot);
                let l = lora(config.lora_of(i));
                if b.is_empty() && l.is_empty() {
                    return String::new();
                }
                let c = crate::scene::character_clause(&b, &format!("the {noun} {}", position(i, n)));
                if l.is_empty() { c } else { format!("{l}, {c}") }
            };
            let header = [raw(&quality), raw(&count_tokens), raw(&interaction)].concat().join(", ");
            let mut groups = vec![header, raw(&footer).join(", ")];
            groups.extend((0..n).map(clause));
            groups.push(sentence.unwrap_or("").to_string());
            let groups: Vec<String> = groups.into_iter().filter(|g| !g.trim().is_empty()).collect();
            groups.join(if config.arranged { ",\n" } else { ", " })
        }
    } else if config.checkpoint == CheckpointProfile::Flux {
        // prose: say which description belongs to which character
        let mut parts = vec![count_text.clone()];
        parts.extend(raw(&interaction));
        let mut s = parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(", ");
        for i in 0..n {
            let b = block(i);
            if !b.is_empty() {
                s.push_str(&format!(". The {} character: {b}", ORDINALS[i.min(3)]));
            }
        }
        let scene = raw(&footer).join(", ");
        if !scene.is_empty() {
            s.push_str(&format!(". {scene}"));
        }
        if let Some(sen) = sentence {
            s.push_str(&format!(". {sen}"));
        }
        s.trim_start_matches(". ").to_string()
    } else {
        let sep = if break_applies(config) && !config.arranged { "BREAK" } else { "" };
        let header = [raw(&count_tokens), raw(&interaction)].concat().join(", ");
        let mut groups: Vec<String> = vec![raw(&quality).join(", "), header];
        for i in 0..n {
            if i > 0 {
                groups.push(sep.to_string());
            }
            groups.push(block(i));
        }
        if config.arranged {
            let of = |s: &[Slot]| raw(&footer.iter().filter(|t| s.contains(&footer_slot(t))).cloned().collect::<Vec<_>>()).join(", ");
            groups.push(of(&[Slot::Camera]));
            groups.push(of(&[Slot::Setting, Slot::Light]));
            groups.push(of(&[Slot::Style]));
            groups.push([of(&[Slot::Clause]), sentence.unwrap_or("").to_string()].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(", "));
        } else {
            groups.push(raw(&footer).join(", "));
            groups.push(sentence.unwrap_or("").to_string());
        }
        let groups: Vec<String> = groups.into_iter().filter(|g| !g.trim().is_empty()).collect();
        groups.join(if config.arranged { ",\n" } else { ", " })
    };

    let mut shared = count_tokens;
    shared.extend(interaction);
    shared.extend(footer);
    let chars = chars
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.into_iter().filter(|t| blocks[i].iter().any(|b| b.normalized == t.normalized)).collect())
        .collect();
    Combined { text, chars, shared }
}

/// Builds the Section-by-Section Blocks (Modular Node Layout — NO BREAK Needed) format.
/// Formats the prompt into distinct labeled modular sections:
/// - [Scene / Environment & Shared Framing]
/// - [Left Character (Character 1 Identity)]
/// - [Right Character (Character 2 Identity)]
///   etc.
pub fn combine_section_blocks(texts: &[String], config: &ScanConfig, strip_conflicts: bool, sentence: Option<&str>) -> String {
    let n = texts.len();
    if n == 0 {
        return String::new();
    }
    let tokens = |text: &str, lora: &str| -> Vec<PromptToken> {
        crate::analyzer::fix_body_type(strip_lora_tokens(parse_prompt(text), lora, strip_conflicts))
            .into_iter()
            .filter(|t| t.normalized != "break")
            .filter(|t| !(strip_conflicts && t.normalized == "solo"))
            .collect()
    };
    let chars: Vec<Vec<PromptToken>> = texts.iter().enumerate().map(|(i, t)| tokens(t, config.lora_of(i))).collect();
    let scene_box: Vec<String> = parse_prompt(&config.scene_prompt_raw).into_iter().map(|t| t.normalized).collect();

    let mut quality: Vec<PromptToken> = Vec::new();
    let mut group_counts: Vec<PromptToken> = Vec::new();
    let mut kinds: Vec<&'static str> = Vec::new();
    let mut kind_of: Vec<Option<&'static str>> = vec![None; n];
    let mut interaction: Vec<PromptToken> = Vec::new();
    let mut footer: Vec<PromptToken> = Vec::new();
    let mut blocks: Vec<Vec<PromptToken>> = vec![Vec::new(); n];
    let push_unique = |v: &mut Vec<PromptToken>, t: &PromptToken| {
        if !v.iter().any(|x| x.normalized == t.normalized) {
            v.push(t.clone());
        }
    };
    for (i, toks) in chars.iter().enumerate() {
        let mut kind = None;
        for t in toks {
            let n = t.normalized.as_str();
            if let Some((num, k)) = count_tag(n) {
                if num >= 2 {
                    push_unique(&mut group_counts, t);
                } else {
                    kind = kind.or(Some(k));
                }
                continue;
            }
            if is_interaction(n) {
                push_unique(&mut interaction, t);
                continue;
            }
            if scene_box.iter().any(|x| x == n) {
                if slot(t) == Slot::Quality {
                    push_unique(&mut quality, t);
                } else {
                    push_unique(&mut footer, t);
                }
                continue;
            }
            match slot(t) {
                Slot::Quality => push_unique(&mut quality, t),
                Slot::Camera | Slot::Setting | Slot::Light | Slot::Style | Slot::Clause => push_unique(&mut footer, t),
                _ => {
                    if !is_spatial_position(n) {
                        blocks[i].push(t.clone());
                    }
                }
            }
        }
        kinds.extend(kind);
        kind_of[i] = kind;
    }

    for t in parse_prompt(&config.scene_prompt_raw) {
        if slot(&t) == Slot::Quality {
            push_unique(&mut quality, &t);
        } else {
            push_unique(&mut footer, &t);
        }
    }

    let count_text = if !group_counts.is_empty() {
        group_counts.iter().map(|t| t.raw.trim().to_string()).collect::<Vec<_>>().join(", ")
    } else {
        let count = |k: &str| kinds.iter().filter(|x| **x == k).count();
        let mut parts = Vec::new();
        for k in ["girl", "boy", "other"] {
            match count(k) {
                0 => {}
                1 => parts.push(format!("1{k}")),
                c => parts.push(format!("{c}{k}s")),
            }
        }
        parts.join(", ")
    };
    let count_tokens = parse_prompt(&count_text);

    if config.cleanup_enabled || config.arranged {
        footer.sort_by_key(footer_slot);
    }

    let raw = |v: &[PromptToken]| v.iter().map(|t| t.raw.trim().to_string()).collect::<Vec<String>>();
    let lora = |s: &str| s.trim().trim_matches(',').trim().to_string();
    let block = |i: usize| {
        let mut b = blocks[i].clone();
        if config.arranged {
            b.sort_by_key(slot);
        }
        let mut items = Vec::new();
        if !lora(config.lora_of(i)).is_empty() {
            items.push(lora(config.lora_of(i)));
        }
        items.extend(raw(&b));
        items.join(", ")
    };

    let mut sections = Vec::new();

    // Section 1: Scene / Environment & Shared Framing
    let mut header_items = [raw(&quality), raw(&count_tokens), raw(&interaction), raw(&footer)].concat();
    if let Some(sen) = sentence.map(str::trim).filter(|s| !s.is_empty()) {
        header_items.push(sen.to_string());
    }
    let header_text = header_items.into_iter().filter(|s| !s.trim().is_empty()).collect::<Vec<_>>().join(", ");
    if !header_text.is_empty() {
        let title = if n > 1 {
            "[Scene / Environment & Shared Framing]:"
        } else {
            "[Scene / Environment & Framing]:"
        };
        sections.push(format!("{title}\n{header_text}"));
    }

    // Section 2..N+1: Individual Character Blocks
    let noun = |i: usize| match kind_of[i] {
        Some("girl") => "girl",
        Some("boy") => "boy",
        _ => "character",
    };

    for i in 0..n {
        let label = match (n, i) {
            (1, _) => "[Character 1]:",
            (2, 0) => "[Left Character (Character 1 Identity)]:",
            (2, 1) => "[Right Character (Character 2 Identity)]:",
            (3, 0) => "[Left Character (Character 1 Identity)]:",
            (3, 1) => "[Center Character (Character 2 Identity)]:",
            (3, 2) => "[Right Character (Character 3 Identity)]:",
            (4, 0) => "[Far Left Character (Character 1 Identity)]:",
            (4, 1) => "[Center Left Character (Character 2 Identity)]:",
            (4, 2) => "[Center Right Character (Character 3 Identity)]:",
            (4, 3) => "[Far Right Character (Character 4 Identity)]:",
            _ => "[Character]:",
        };
        let b = block(i);
        let char_content = if n > 1 {
            let pos = position(i, n);
            let mut remaining = Vec::new();
            for tok in parse_prompt(&b) {
                let norm = tok.normalized.as_str();
                if norm == pos || is_spatial_position(norm) || count_tag(norm).is_some() {
                    continue;
                }
                remaining.push(tok.raw);
            }
            let b_clean = remaining.join(", ");
            if b_clean.is_empty() {
                format!("1{}, {pos}", noun(i))
            } else {
                format!("1{}, {pos}, {b_clean}", noun(i))
            }
        } else {
            let mut remaining = Vec::new();
            for tok in parse_prompt(&b) {
                let norm = tok.normalized.as_str();
                if count_tag(norm).is_some() {
                    continue;
                }
                remaining.push(tok.raw);
            }
            let b_clean = remaining.join(", ");
            if b_clean.is_empty() {
                format!("1{}", noun(i))
            } else {
                format!("1{}, {b_clean}", noun(i))
            }
        };
        if !char_content.trim().is_empty() {
            sections.push(format!("{label}\n{char_content}"));
        }
    }

    sections.join("\n\n")
}

/// Builds the Section-by-Section Blocks (Modular Node Layout) format from an AI-written prompt.
/// Splits the AI text into character parts and shared scene tags, then formats them into
/// cleanly labeled modular blocks without BREAKs.
pub fn combine_section_blocks_ai(text: &str, boxes: &[String], config: &ScanConfig, sentence: Option<&str>) -> String {
    let (parts, shared) = split_ai(text, boxes);
    let mut cfg = config.clone();
    if !shared.is_empty() {
        cfg.scene_prompt_raw = [cfg.scene_prompt_raw.trim(), shared.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
    }
    combine_section_blocks(&parts, &cfg, true, sentence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::TargetView;

    fn cfg() -> ScanConfig {
        ScanConfig {
            checkpoint: CheckpointProfile::PonyV6,
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
            natural: false,
            two_char: true,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: true, scene_prompt_raw: String::new(), more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
        }
    }

    #[test]
    fn keeps_each_character_together() {
        let mut c = cfg();
        c.lora_tags_raw = "<lora:a:0.8>".into();
        c.char2_lora_tags_raw = "<lora:b:0.7>".into();
        let out = combine(&["score_9, 1girl, blonde hair, red dress, smile, holding hands, upper body, park, sunset".to_string(), "1girl, black hair, school uniform, smile, holding hands, park".to_string()],
            &c,
            false,
            None,
        );
        assert_eq!(
            out.text,
            "score_9, 2girls, holding hands, <lora:a:0.8>, blonde hair, red dress, smile, BREAK, <lora:b:0.7>, \
             black hair, school uniform, smile, upper body, park, sunset"
        );
        // Anima: no BREAK when char_break is false; a girl and a boy: "1girl, 1boy"; arranged on lines
        c.checkpoint = CheckpointProfile::Anima;
        c.char_break = false;
        c.arranged = true;
        let out = combine(&["masterpiece, 1girl, silver hair, kimono, night".to_string(), "1boy, black hair, armor".to_string()], &c, false, None);
        assert_eq!(
            out.text,
            "masterpiece, 1girl, 1boy,\nnight,\n<lora:a:0.8>, the girl on the left has silver hair, wearing kimono,\n\
             <lora:b:0.7>, the boy on the right has black hair, wearing armor"
        );
        // Anima: BREAK chunk fencing when char_break is true (default) on single line
        c.char_break = true;
        c.arranged = false;
        let out_break = combine(&["masterpiece, 1girl, silver hair, kimono, night".to_string(), "1boy, black hair, armor".to_string()], &c, false, None);
        assert_eq!(
            out_break.text,
            "masterpiece, 1girl, 1boy, night, BREAK, 1girl, on the left, <lora:a:0.8>, silver hair, kimono, BREAK, 1boy, on the right, <lora:b:0.7>, black hair, armor"
        );
        // Anima: line separation (arranged) does NOT include BREAK; newlines separate the chunks cleanly
        c.arranged = true;
        let out_arranged = combine(&["masterpiece, 1girl, silver hair, kimono, night".to_string(), "1boy, black hair, armor".to_string()], &c, false, None);
        assert_eq!(
            out_arranged.text,
            "masterpiece, 1girl, 1boy, night,\n1girl, on the left, <lora:a:0.8>, silver hair, kimono,\n1boy, on the right, <lora:b:0.7>, black hair, armor"
        );
    }

    /// The whole rule path the app takes: each character analyzed on its own, condensed,
    /// then combined.
    fn run_both(c1: &str, c2: &str, c: &ScanConfig) -> String {
        use crate::analyzer::analyze_with_rules;
        use crate::condenser::build_condensed_output;
        let r1 = analyze_with_rules(&parse_prompt(c1), c);
        let mut c2cfg = c.clone();
        c2cfg.lora_tags_raw = c.char2_lora_tags_raw.clone();
        let mut r2 = analyze_with_rules(&parse_prompt(c2), &c2cfg);
        r2.processed_tokens.retain(|t| t.normalized != "solo");
        let o1 = build_condensed_output(c1, &r1.processed_tokens, &r1.changes, c);
        let o2 = build_condensed_output(c2, &r2.processed_tokens, &r2.changes, c);
        combine(&[o1.final_prompt, o2.final_prompt], c, false, None).text
    }

    #[test]
    fn rules_stay_per_character() {
        let mut c = cfg();
        c.lora_tags_raw = "blonde hair, large breasts".into();
        let out = run_both(
            "score_9, 1girl, blonde hair, long hair, smile, smile, beach, sunset",
            "1girl, solo, black hair, medium breasts, long hair, smile",
            &c,
        );
        // shared tags stay with both characters, each character's duplicates go,
        // character 1's LoRA never touches character 2, "solo" is gone
        assert_eq!(out.matches("long hair").count(), 2, "{out}");
        assert_eq!(out.matches("smile").count(), 2, "{out}");
        assert!(out.contains("BREAK, black hair, medium breasts"), "{out}");
        assert!(!out.contains("solo"), "{out}");
        assert!(out.starts_with("score_9, 2girls, blonde hair, large breasts"), "{out}");
        assert!(out.ends_with("beach, sunset"), "{out}");
        // scene tags the AI invents go to the shared footer, not into a character
        let ai = combine(&["1girl, red kimono, medium shot".to_string(), "1girl, armor, long shadows, fantasy style".to_string()], &c, true, None).text;
        assert!(ai.ends_with("medium shot, long shadows, fantasy style"), "{ai}");
    }

    /// Deep Scan with three characters on a real Ollama, showing what the AI wrote and how
    /// it was split: `cargo test live_three_char -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_three_char() {
        use crate::ai_engine::{build_ai_system_and_user_prompt_for, run_ai_scan, AiScanRequest, ModelFamily};
        let ckpt = match std::env::var("THREE_CKPT").as_deref() { Ok("pony") => CheckpointProfile::PonyV6, _ => CheckpointProfile::Anima };
        let mut c = cfg();
        c.checkpoint = ckpt;
        c.natural = true;
        c.scene_prompt_raw = "standing, side by side, park, sunset, full body".into();
        c.char2_prompt_raw = "1girl, blue hair, short hair, school uniform, glasses, shy".into();
        c.more_chars = vec![("1girl, green hair, twintails, maid outfit, smile, peace sign".into(), String::new())];
        let raw = "1girl, red hair, long hair, ponytail, red dress, high heels, confident";
        if std::env::var("SHOW_PROMPT").is_ok() {
            let (sys, user) = build_ai_system_and_user_prompt_for(raw, &c, ModelFamily::Gemma);
            println!("{sys}\n----\n{user}");
        }
        let req = AiScanRequest {
            raw_prompt: raw.into(), config: c.clone(), mode: crate::profiles::AiMode::Gpu,
            host: "localhost:11434".into(), model: "richardyoung/llama-3.1-8b-instruct-abliterated".into(),
            keep_alive: "5m".into(), think: false,
        };
        let (out, place) = run_ai_scan(&req).unwrap();
        let boxes = vec![raw.to_string(), c.char2_prompt_raw.clone(), c.more_chars[0].0.clone()];
        let (parts, shared) = split_ai(&out.improved_prompt, &boxes);
        println!("\n== {} ({place})\nAI IMPROVED: {}\nSENTENCE: {}", ckpt.display_name(), out.improved_prompt, out.natural_sentence);
        for (i, p) in parts.iter().enumerate() { println!("PART {}: {p}", i + 1); }
        println!("SHARED: {shared}");
        println!("FINAL: {}", combine_ai(&out.improved_prompt, &boxes, &c, Some(&out.natural_sentence)).text);
    }

    /// The audit's instruction changes against the real model: Deep Scans for each checkpoint
    /// and 1-3 characters, each result checked. `cargo test live_audit -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_audit() {
        use crate::ai_engine::{run_ai_scan, scan_problems, AiScanRequest};
        struct Case { name: &'static str, ckpt: CheckpointProfile, chars: Vec<&'static str>, scene: &'static str }
        let cases = vec![
            Case { name: "Anima 1", ckpt: CheckpointProfile::Anima, chars: vec!["1girl, kitsune, fox ears, long white hair, red eyes, kimono, sitting, shrine, night"], scene: "" },
            Case { name: "Pony 1", ckpt: CheckpointProfile::PonyV6, chars: vec!["1girl, knight, silver armor, red cape, holding sword, standing, castle, sunset"], scene: "" },
            Case { name: "Flux 1", ckpt: CheckpointProfile::Flux, chars: vec!["a woman in a red coat walking through a snowy street at dusk"], scene: "" },
            Case { name: "Anima 2", ckpt: CheckpointProfile::Anima, chars: vec!["1girl, red hair, long hair, red dress, smile", "1girl, blue hair, short hair, school uniform, glasses, shy"], scene: "park, sunset, standing side by side" },
            Case { name: "Pony 2", ckpt: CheckpointProfile::PonyV6, chars: vec!["1girl, red hair, red dress, smile", "1boy, black hair, armor, serious"], scene: "castle courtyard, night" },
            Case { name: "Pony 3", ckpt: CheckpointProfile::PonyV6, chars: vec!["1girl, red hair, ponytail, red dress", "1girl, blue hair, school uniform, glasses", "1girl, green hair, twintails, maid outfit"], scene: "park, sunset, full body" },
        ];
        let colors = ["red hair", "blue hair", "green hair"];
        let mut failures: Vec<String> = Vec::new();
        for case in cases {
            let n = case.chars.len();
            let mut c = cfg();
            c.checkpoint = case.ckpt;
            c.two_char = n >= 2;
            c.natural = true;
            c.scene_prompt_raw = case.scene.into();
            if n >= 2 { c.char2_prompt_raw = case.chars[1].into(); }
            c.more_chars = case.chars.iter().skip(2).map(|p| (p.to_string(), String::new())).collect();
            let req = AiScanRequest {
                raw_prompt: case.chars[0].into(), config: c.clone(), mode: crate::profiles::AiMode::Gpu,
                host: "localhost:11434".into(), model: "richardyoung/llama-3.1-8b-instruct-abliterated".into(),
                keep_alive: "10m".into(), think: false,
            };
            let t = std::time::Instant::now();
            let (out, place) = run_ai_scan(&req).unwrap();
            let mut checks: Vec<(String, bool)> = Vec::new();
            let low = out.improved_prompt.to_lowercase();
            let problems = scan_problems(&out, &req);
            checks.push((format!("complete scan {:?}", problems), problems.is_empty()));
            let neg = out.recommended_negative.to_lowercase();
            if case.ckpt == CheckpointProfile::Flux {
                checks.push(("Flux: no negative".into(), neg.trim().is_empty()));
                checks.push(("Flux: no (tag:1.x) weights".into(), !low.contains(":1.")));
            } else {
                let banned = ["wrong framing", "unwanted extra", "wrong camera", "lack of", ", no "];
                checks.push((format!("negative has no instruction phrases ({neg})"), !banned.iter().any(|b| neg.contains(b))));
                checks.push(("negative present".into(), neg.split(',').count() >= 5));
            }
            let q = case.ckpt.canonical_quality_tags();
            if let Some(first) = q.first() {
                checks.push((format!("quality tag '{first}'"), low.contains(&first.to_lowercase())));
            }
            checks.push(("no dynamic_prompt (off)".into(), out.dynamic_prompt.trim().is_empty()));
            if n >= 2 {
                checks.push(("no 'solo'".into(), !low.split(',').any(|t| t.trim() == "solo")));
                let boxes: Vec<String> = case.chars.iter().map(|s| s.to_string()).collect();
                let (parts, shared) = split_ai(&out.improved_prompt, &boxes);
                checks.push((format!("{n} character parts"), parts.len() == n && parts.iter().all(|p| !p.is_empty())));
                checks.push(("shared scene kept apart".into(), !shared.is_empty()));
                for (i, p) in parts.iter().enumerate() {
                    let pl = p.to_lowercase();
                    let own = case.chars[i].to_lowercase();
                    if let Some(col) = colors.iter().find(|c| own.contains(**c)) {
                        checks.push((format!("character {} keeps '{col}'", i + 1), pl.contains(col)));
                    }
                    for (j, other) in colors.iter().enumerate() {
                        if j != i && j < n && case.chars[i].to_lowercase().contains(other) == false && pl.contains(other) && case.chars[j].to_lowercase().contains(other) {
                            checks.push((format!("character {} has no '{other}' from character {}", i + 1, j + 1), false));
                        }
                    }
                }
            }
            println!("\n== {} ({:.0}s, {place})", case.name, t.elapsed().as_secs_f32());
            println!("   IMPROVED: {}", out.improved_prompt);
            println!("   NEGATIVE: {}", out.recommended_negative);
            for (name, ok) in &checks {
                println!("   {} {}", if *ok { "PASS" } else { "FAIL" }, name);
                if !*ok { failures.push(format!("{}: {}", case.name, name)); }
            }
        }
        println!("\nAUDIT: {} failure(s) {:?}", failures.len(), failures);
    }

    /// Deep Scan on the prompts saved in settings.json (the user's real boxes):
    /// `cargo test live_user_scan -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_user_scan() {
        use crate::ai_engine::{run_ai_scan, AiScanRequest};
        let home = std::env::var("HOME").unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{home}/.config/prompt-condenser/settings.json")).unwrap()).unwrap();
        let g = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
        let n: usize = std::env::var("USER_N").ok().and_then(|x| x.parse().ok()).unwrap_or(3);
        let mut c = cfg();
        c.checkpoint = CheckpointProfile::Anima;
        c.natural = v.get("natural").and_then(|x| x.as_bool()).unwrap_or(false);
        c.arranged = v.get("arranged").and_then(|x| x.as_bool()).unwrap_or(false);
        c.detail = crate::profiles::DetailLevel::from_index(v.get("detail_index").and_then(|x| x.as_i64()).unwrap_or(1) as i32);
        c.scene_prompt_raw = g("scene_prompt");
        c.char2_prompt_raw = g("char2_prompt");
        c.more_chars = (3..=n).map(|k| (g(&format!("char{k}_prompt")), String::new())).collect();
        let raw = g("raw_prompt");
        let req = AiScanRequest {
            raw_prompt: raw.clone(), config: c.clone(), mode: crate::profiles::AiMode::Gpu,
            host: "localhost:11434".into(), model: "richardyoung/llama-3.1-8b-instruct-abliterated".into(),
            keep_alive: "5m".into(), think: false,
        };
        let (out, place) = run_ai_scan(&req).unwrap();
        let mut boxes = vec![raw];
        boxes.extend((1..n).map(|i| c.prompt_of(i).to_string()));
        let (parts, shared) = split_ai(&out.improved_prompt, &boxes);
        println!("\n== ({place})\nAI IMPROVED: {}\nSENTENCE: {}", out.improved_prompt, out.natural_sentence);
        for (i, p) in parts.iter().enumerate() { println!("PART {}: {p}", i + 1); }
        println!("SHARED: {shared}");
        println!("FINAL: {}", combine_ai(&out.improved_prompt, &boxes, &c, Some(&out.natural_sentence)).text);
    }

    /// Deep Scan with two characters on a real Ollama:
    /// `cargo test live_two_char -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_two_char() {
        use crate::ai_engine::{run_ai_scan, AiScanRequest};
        for ckpt in [CheckpointProfile::PonyV6, CheckpointProfile::Anima] {
            let mut c = cfg();
            c.checkpoint = ckpt;
            c.natural = true;
            c.char2_prompt_raw = "1girl, black hair, short hair, knight, silver armor, serious, holding sword".into();
            c.lora_tags_raw = "<lora:fox:0.8>".into();
            let raw = "1girl, solo, blonde hair, long hair, fox ears, red kimono, smile, holding hands, castle, sunset";
            let req = AiScanRequest {
                raw_prompt: raw.into(), config: c.clone(), mode: crate::profiles::AiMode::Gpu,
                host: "localhost:11434".into(), model: "richardyoung/llama-3.1-8b-instruct-abliterated".into(),
                keep_alive: "30m".into(), think: false,
            };
            let t = std::time::Instant::now();
            let (out, place) = run_ai_scan(&req).unwrap();
            let final_text = combine_ai(&out.improved_prompt, &[raw.to_string(), c.char2_prompt_raw.clone()], &c, Some(&out.natural_sentence)).text;
            println!(
                "\n== {} ({:.0}s, {place})\nAI      : {}\nSENTENCE: {}\nFINAL   : {}\nCARDS   : {}",
                ckpt.display_name(), t.elapsed().as_secs_f32(), out.improved_prompt, out.natural_sentence, final_text,
                out.changes.len()
            );
        }
    }

    #[test]
    fn three_and_four_characters() {
        let mut c = cfg();
        c.more_chars = vec![("".into(), "<lora:c:0.6>".into())];
        let t = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        let out = combine(&t(&["1girl, red hair, park", "1girl, blue hair", "1boy, black hair, smile"]), &c, false, None).text;
        assert_eq!(out, "2girls, 1boy, red hair, BREAK, blue hair, BREAK, <lora:c:0.6>, black hair, smile, park");
        c.checkpoint = CheckpointProfile::Anima;
        c.char_break = false;
        c.more_chars.push(("".into(), "".into()));
        let out = combine(&t(&["1girl, red hair", "1girl, blue hair", "1girl, green hair", "1girl, pink hair"]), &c, false, None).text;
        assert_eq!(
            out,
            "4girls, the girl on the far left has red hair, the girl second from the left has blue hair, \
             <lora:c:0.6>, the girl second from the right has green hair, the girl on the far right has pink hair"
        );
        assert_eq!(permutations(3).len(), 6);
    }

    #[test]
    fn anima_clauses_keep_scene_words_out() {
        let mut c = cfg();
        c.checkpoint = CheckpointProfile::Anima;
        c.char_break = false;
        let out = combine(&["1girl, blonde hair, red kimono".to_string(), "1girl, knight, black hair, warm backlight, romantic, silver armor".to_string()], &c, true, None).text;
        assert_eq!(
            out,
            "2girls, warm backlight, romantic, the girl on the left has blonde hair, wearing red kimono, \
             the girl on the right is a knight with black hair, wearing silver armor"
        );
    }

    #[test]
    fn scene_box_tags_are_always_shared() {
        let mut c = cfg();
        c.scene_prompt_raw = "rooftop, petals, wind, upper body".into();
        // "petals" and "wind" aren't recognized as scene words, but they came from the Scene box
        let out = combine(&["1girl, red hair, rooftop, petals".to_string(), "1girl, blue hair, wind".to_string()], &c, false, None).text;
        assert_eq!(out, "2girls, red hair, BREAK, blue hair, rooftop, petals, wind");
    }

    #[test]
    fn splits_break_prompts_and_ai_text() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        assert_eq!(prompts("1girl, red hair BREAK 1boy, armor", &s(&[""])), s(&["1girl, red hair", "1boy, armor"]));
        assert_eq!(prompts("1girl, BREAKDANCE", &s(&[""]))[1], "");
        assert_eq!(prompts("a BREAK b BREAK c", &s(&["", ""])), s(&["a", "b", "c"]));
        assert_eq!(
            split_ai("2girls, blonde hair, smile, black hair, armor", &s(&["blonde hair", "black hair, armor"])).0,
            s(&["2girls, blonde hair, smile", "black hair, armor"])
        );
        // written in another order: matched back to the boxes
        assert_eq!(
            split_ai("black hair, armor, BREAK, red hair, BREAK, blonde hair, smile", &s(&["blonde hair, smile", "black hair, armor", "red hair"])).0,
            s(&["blonde hair, smile", "black hair, armor", "red hair"])
        );
        // the way Gemma actually writes it: BREAKs around the opening tags and the shared scene too
        let ai = "masterpiece, 3girls, park, sunset, full body, BREAK, 1girl, red hair, red dress, BREAK, 1girl, blue hair, \
                  school uniform, shy expression, looking down, BREAK, 1girl, green hair, maid outfit, BREAK, lush green grass, warm lighting";
        let (parts, shared) = split_ai(ai, &s(&["red hair", "blue hair, school uniform", "green hair"]));
        assert_eq!(parts, s(&["1girl, red hair, red dress", "1girl, blue hair, school uniform, shy expression, looking down", "1girl, green hair, maid outfit"]));
        assert!(shared.contains("3girls") && shared.contains("lush green grass"), "{shared}");
        let mut c = cfg();
        c.checkpoint = CheckpointProfile::Anima;
        c.char_break = false;
        let out = combine_ai(ai, &s(&["red hair", "blue hair, school uniform", "green hair"]), &c, None).text;
        assert!(out.contains("the girl in the middle has blue hair, wearing school uniform, shy expression, looking down"), "{out}");
        assert!(!out.contains("has red hair and lush green grass") && out.contains("lush green grass"), "{out}");
        c.char_break = true;
        let out_b = combine_ai(ai, &s(&["red hair", "blue hair, school uniform", "green hair"]), &c, None).text;
        assert!(out_b.contains("BREAK, 1girl, in the middle, blue hair, school uniform, shy expression, looking down"), "{out_b}");
        // the shared scene part has a word that once looked like a trait ("bookshelf" / "elf")
        let ai = "masterpiece, 3girls, BREAK, 1girl, red hair, red dress, BREAK, 1girl, blue hair, school uniform, BREAK, \
                  1girl, green hair, maid outfit, BREAK, indoor, bookshelf, sofa, rainbow, landscape, upper body";
        let (parts, shared) = split_ai(ai, &s(&["red hair", "blue hair", "green hair"]));
        assert_eq!(parts.len(), 3);
        assert!(parts[1].starts_with("1girl, blue hair") && shared.contains("bookshelf"), "{parts:?} / {shared}");
    }

    #[test]
    fn ai_text_drops_solo_and_lora_conflicts_per_character() {
        let mut c = cfg();
        c.lora_tags_raw = "large breasts".into();
        let out = combine(&["1girl, solo, gigantic breasts".to_string(), "1girl, gigantic breasts, flat chest".to_string()], &c, true, None);
        // character 1's LoRA tag replaces its own conflict, never character 2's
        assert_eq!(out.text, "2girls, large breasts, BREAK, gigantic breasts, flat chest");
    }

    #[test]
    fn section_blocks_modular_formatting() {
        let mut c = cfg();
        c.scene_prompt_raw = "indoors, gym, gymnasium, polished floor".to_string();
        let c1 = "1girl, red hair, ponytail, athletic tank top, red gym shorts, standing".to_string();
        let c2 = "1girl, blue hair, twin-tails, track jacket, running shorts, looking at viewer".to_string();
        let blocks = combine_section_blocks(&[c1, c2], &c, true, None);
        assert!(blocks.contains("[Scene / Environment & Shared Framing]:"), "{blocks}");
        assert!(blocks.contains("gym, gymnasium"), "{blocks}");
        assert!(blocks.contains("[Left Character (Character 1 Identity)]:"), "{blocks}");
        assert!(blocks.contains("1girl, on the left"), "{blocks}");
        assert!(blocks.contains("[Right Character (Character 2 Identity)]:"), "{blocks}");
        assert!(blocks.contains("1girl, on the right"), "{blocks}");
        assert!(!blocks.contains("BREAK"), "Section blocks must not contain BREAK: {blocks}");
    }

    #[test]
    fn combine_section_blocks_ai_formatting() {
        let c = cfg();
        let ai = "masterpiece, best quality, 2girls, indoors, gym, basketball court, bright lighting, BREAK, \
                  1girl, on the left, ruby red hair, ponytail, red gym shorts, athletic, BREAK, \
                  1girl, on the right, navy blue hair, track jacket, running shoes";
        let boxes = vec!["red hair".to_string(), "blue hair".to_string()];
        let blocks = combine_section_blocks_ai(ai, &boxes, &c, Some("both girls stretch in the gym"));
        assert!(blocks.contains("[Scene / Environment & Shared Framing]:"), "{blocks}");
        assert!(blocks.contains("gym, basketball court"), "{blocks}");
        assert!(blocks.contains("both girls stretch in the gym"), "{blocks}");
        assert!(blocks.contains("[Left Character (Character 1 Identity)]:"), "{blocks}");
        assert!(blocks.contains("1girl, on the left, ruby red hair"), "{blocks}");
        assert!(blocks.contains("[Right Character (Character 2 Identity)]:"), "{blocks}");
        assert!(blocks.contains("1girl, on the right, navy blue hair"), "{blocks}");
        assert!(!blocks.contains("BREAK"), "AI section blocks must not contain BREAK: {blocks}");
        assert!(!blocks.contains("on the left, on the left"), "Must not duplicate position tags: {blocks}");
    }

    #[test]
    fn parse_section_blocks_unpacks_properly() {
        let sample = "\
[Scene / Environment & Shared Framing]:
masterpiece, 2girls, indoors, gym, polished floor

[Left Character (Character 1 Identity)]:
1girl, on the left, ruby red hair, ponytail, gym shorts

[Right Character (Character 2 Identity)]:
1girl, on the right, midnight blue hair, track jacket
";
        let parsed = parse_section_blocks(sample);
        assert!(parsed.is_some());
        let (chars, scene) = parsed.unwrap();
        assert_eq!(chars.len(), 2);
        assert!(chars[0].contains("ruby red hair"));
        assert!(chars[1].contains("midnight blue hair"));
        assert!(scene.contains("gym, polished floor"));

        let prompts = prompts(sample, &["".to_string()]);
        assert_eq!(prompts.len(), 2);
        assert!(prompts[0].contains("ruby red hair"));
        assert!(prompts[0].contains("gym, polished floor"));
        assert!(prompts[1].contains("midnight blue hair"));

        // Header without colon
        let sample2 = "[Scene / Environment]\nindoors, beach\n[Character 1]\n1girl, swimsuit\n[Character 2]\n1girl, bikini";
        let parsed2 = parse_section_blocks(sample2);
        assert!(parsed2.is_some());
        let (chars2, scene2) = parsed2.unwrap();
        assert_eq!(chars2.len(), 2);
        assert!(scene2.contains("indoors, beach"));
    }

    #[test]
    fn break_prompt_with_scene_prefix_separates_cleanly() {
        let raw = "masterpiece, 2girls, indoors, gym BREAK 1girl, ruby red hair, ponytail, gym shorts BREAK 1girl, midnight blue hair, track jacket";
        let res = prompts(raw, &["".to_string()]);
        assert_eq!(res.len(), 2);
        assert!(res[0].contains("ruby red hair"), "Char 1 got: {}", res[0]);
        assert!(res[0].contains("gym"), "Char 1 scene got: {}", res[0]);
        assert!(res[1].contains("midnight blue hair"), "Char 2 got: {}", res[1]);
        assert!(!res[1].contains("ruby red hair"), "Char 2 contaminated: {}", res[1]);
    }
}
