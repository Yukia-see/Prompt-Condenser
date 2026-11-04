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
    config.char_break && !matches!(config.checkpoint, CheckpointProfile::Anima | CheckpointProfile::Flux)
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
    parts.into_iter().map(|p| p.trim().trim_matches(',').trim().to_string()).collect()
}

/// The prompts to analyze, one per character: the boxes as typed, or, with the other
/// boxes empty, a pasted "… BREAK … BREAK …" prompt split up.
pub fn prompts(char1: &str, others: &[String]) -> Vec<String> {
    let n = others.len() + 1;
    if others.iter().all(|o| o.trim().is_empty()) {
        let parts = split_breaks(char1);
        if parts.len() >= 2 {
            let mut out: Vec<String> = parts.iter().take(n).cloned().collect();
            if parts.len() > n {
                out[n - 1] = parts[n - 1..].join(", ");
            }
            out.resize(n, String::new());
            return out;
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
        let (kept, extra): (Vec<(usize, String)>, Vec<(usize, String)>) =
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
fn position(i: usize, n: usize) -> &'static str {
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
                _ => blocks[i].push(t.clone()),
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
        // Anima reads prompts with a language model and ignores BREAK, so each character
        // becomes a clause that says where it is and what it has (the format the Anima Ref
        // Adapter uses): header, shared scene, then the characters left to right
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
        let sep = if break_applies(config) { "BREAK" } else { "" };
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
        // Anima: no BREAK; a girl and a boy: "1girl, 1boy"; arranged on lines
        c.checkpoint = CheckpointProfile::Anima;
        c.arranged = true;
        let out = combine(&["masterpiece, 1girl, silver hair, kimono, night".to_string(), "1boy, black hair, armor".to_string()], &c, false, None);
        assert_eq!(
            out.text,
            "masterpiece, 1girl, 1boy,\nnight,\n<lora:a:0.8>, the girl on the left has silver hair, wearing kimono,\n\
             <lora:b:0.7>, the boy on the right has black hair, wearing armor"
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
            host: "localhost:11434".into(), model: "prompt-condenser-gemma4-12b".into(),
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
                host: "localhost:11434".into(), model: "prompt-condenser-gemma4-12b".into(),
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
            host: "localhost:11434".into(), model: "prompt-condenser-gemma4-12b".into(),
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
                host: "localhost:11434".into(), model: "prompt-condenser-gemma4-12b".into(),
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
        let out = combine_ai(ai, &s(&["red hair", "blue hair, school uniform", "green hair"]), &c, None).text;
        assert!(out.contains("the girl in the middle has blue hair, wearing school uniform, shy expression, looking down"), "{out}");
        assert!(!out.contains("has red hair and lush green grass") && out.contains("lush green grass"), "{out}");
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
}
