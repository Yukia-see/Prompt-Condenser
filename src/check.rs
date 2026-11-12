//! "Check result": the tagger reads a picture generated from the prompt and compares the two.
//! Prompt tags the tagger can't find get a weight-boost card (it can read them, so it should have seen them);
//! tags it clearly sees that contradict the prompt (another hair color, a second girl, extra arms, a
//! watermark) are offered for the negative. Nothing new goes into the positive prompt.

use crate::parser::{self, TokenCategory};
use crate::tagger::{ScoredTag, TagKind};

/// Below this score a prompt tag counts as missing from the picture.
pub const MISSING_BELOW: f32 = 0.12;
/// From this score a contradicting tag counts as clearly in the picture.
pub const UNWANTED_FROM: f32 = 0.6;
/// The boost a missing tag gets, and the weight it never goes past.
pub const BOOST: f32 = 0.2;
pub const MAX_WEIGHT: f32 = 1.5;

/// Generation errors worth a negative whenever the tagger sees them (unless the prompt asks for them).
const ERROR_TAGS: &[&str] = &[
    "extra arms", "extra legs", "extra ears", "extra digits", "extra mouth", "extra eyes", "multiple views",
    "monochrome", "greyscale", "blurry", "jpeg artifacts", "text", "english text", "speech bubble",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Missing {
    /// The tag as written in the prompt (its raw segment, weight syntax included).
    pub raw: String,
    pub tag: String,
    pub score: f32,
    /// The weight it had and the one suggested (equal when it was already at `MAX_WEIGHT`).
    pub weight: f32,
    pub new_weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unwanted {
    pub tag: String,
    pub score: f32,
    /// The prompt tag it contradicts, or "" for a generation error / printed text.
    pub against: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub missing: Vec<Missing>,
    pub unwanted: Vec<Unwanted>,
    /// Prompt tags the tagger knows (could be checked) and how many of those it found.
    pub checked: usize,
    pub found: usize,
}

fn key(name: &str) -> String {
    name.replace('_', " ").to_lowercase()
}

/// How many people a count tag says: `Some((n, exact))`; "multiple girls" is (2, false) = at least two.
fn people(t: &str) -> Option<(u32, bool)> {
    let t = t.trim();
    match t {
        "solo" => return Some((1, true)),
        "multiple girls" | "multiple boys" | "multiple others" => return Some((2, false)),
        _ => {}
    }
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    let rest = &t[digits.len()..];
    (!digits.is_empty() && matches!(rest, "girl" | "girls" | "boy" | "boys" | "other" | "others"))
        .then(|| (digits.parse().unwrap_or(1), true))
}

fn count_tag(t: &str) -> bool {
    people(t).is_some()
}

/// The prompt's people count: the sum of its 1girl / 2boys… tags, else 1 for `solo`.
fn prompt_people(tags: &[String]) -> Option<u32> {
    let sum: u32 = tags
        .iter()
        .filter(|t| t.as_str() != "solo" && !t.starts_with("multiple "))
        .filter_map(|t| people(t))
        .map(|(n, _)| n)
        .sum();
    if sum > 0 {
        Some(sum)
    } else {
        tags.iter().any(|t| t == "solo").then_some(1)
    }
}

/// Compares a prompt with the tagger's scores for the picture made from it.
pub fn compare(prompt: &str, scores: &[ScoredTag]) -> Report {
    let general: Vec<(String, f32)> =
        scores.iter().filter(|s| s.kind == TagKind::General).map(|s| (key(&s.name), s.score)).collect();
    let score_of = |t: &str| general.iter().find(|(n, _)| n == t).map(|(_, s)| *s);
    let tokens: Vec<_> = parser::parse_prompt(prompt)
        .into_iter()
        .filter(|t| {
            !matches!(
                t.category,
                TokenCategory::QualityMeta | TokenCategory::LoraOrSpecial | TokenCategory::NaturalLanguageClause
            ) && t.normalized != "break"
                && !t.normalized.contains('|')
        })
        .collect();
    let prompt_tags: Vec<String> = tokens.iter().map(|t| t.normalized.replace('\\', "")).collect();

    let mut report = Report::default();
    for (t, tag) in tokens.iter().zip(&prompt_tags) {
        // only tags the tagger can read are checked; negative weights ask for less of it anyway
        let Some(score) = score_of(tag) else { continue };
        if t.weight <= 0.0 {
            continue;
        }
        report.checked += 1;
        if score >= MISSING_BELOW {
            report.found += 1;
            continue;
        }
        let new_weight = ((t.weight + BOOST).min(MAX_WEIGHT.max(t.weight)) * 100.0).round() / 100.0;
        report.missing.push(Missing { raw: t.raw.trim().to_string(), tag: tag.clone(), score, weight: t.weight, new_weight });
    }

    let in_prompt = |t: &str| prompt_tags.iter().any(|p| p == t);
    for (tag, score) in &general {
        if *score < UNWANTED_FROM || in_prompt(tag) {
            continue;
        }
        if ERROR_TAGS.contains(&tag.as_str()) || crate::tagger::is_image_text_tag(tag) {
            report.unwanted.push(Unwanted { tag: tag.clone(), score: *score, against: String::new() });
            continue;
        }
        // more (or fewer) people than the prompt asks for
        if let Some((n, exact)) = people(tag) {
            let Some(want) = prompt_people(&prompt_tags) else { continue };
            let wrong = if tag == "solo" { want >= 2 } else if exact { tag.starts_with(|c: char| c.is_ascii_digit()) && n > want } else { want == 1 };
            if wrong {
                let against = prompt_tags.iter().find(|p| count_tag(p)).cloned().unwrap_or_default();
                report.unwanted.push(Unwanted { tag: tag.clone(), score: *score, against });
            }
            continue;
        }
        // the same body part with another color / size / length
        let against = prompt_tags.iter().find(|p| !count_tag(p) && crate::analyzer::tags_conflict(p, tag));
        if let Some(p) = against {
            report.unwanted.push(Unwanted { tag: tag.clone(), score: *score, against: p.clone() });
        }
    }
    report.unwanted.sort_by(|a, b| b.score.total_cmp(&a.score));
    report
}

/// Check Result card ids start here (`ideas::IDEA_CARD_ID_BASE` is 60_000).
pub const CHECK_CARD_ID_BASE: usize = 50_000;

/// The report as opt-in cards: a missing tag → a card that raises its weight; a contradiction → a card whose
/// "Add It" puts the tag in the negative (returned id → tag, since the card itself changes no positive tag).
pub fn cards(report: &Report) -> (Vec<crate::analyzer::PromptChange>, std::collections::HashMap<usize, String>) {
    use crate::analyzer::{ChangeAction, PromptChange};
    let mut out = Vec::new();
    let mut negatives = std::collections::HashMap::new();
    for m in report.missing.iter().filter(|m| m.new_weight != m.weight) {
        let text = parser::parse_prompt(&m.raw).first().map(|t| t.text.clone()).unwrap_or_else(|| m.tag.clone());
        out.push(PromptChange {
            id: CHECK_CARD_ID_BASE + out.len(),
            original: m.raw.clone(),
            action: ChangeAction::CheckResult,
            reason: format!(
                "'{}' isn't in the generated picture (the tagger can read it, score {:.2}): a stronger weight helps it show up.",
                m.tag, m.score
            ),
            suggested_alternative: format!("({text}:{})", m.new_weight),
            framing_alternative: String::new(),
            mode: 2,
        });
    }
    for u in &report.unwanted {
        let id = CHECK_CARD_ID_BASE + out.len();
        negatives.insert(id, u.tag.clone());
        out.push(PromptChange {
            id,
            original: String::new(),
            action: ChangeAction::CheckResult,
            reason: if u.against.is_empty() {
                format!("The picture shows '{}' (score {:.2}), which the prompt doesn't ask for. Add It puts it in the negative.", u.tag, u.score)
            } else {
                format!("The picture shows '{}' (score {:.2}) against your '{}'. Add It puts it in the negative.", u.tag, u.score, u.against)
            },
            suggested_alternative: String::new(),
            framing_alternative: String::new(),
            mode: 2,
        });
    }
    (out, negatives)
}

/// A one-paragraph summary for the status line.
pub fn summary(report: &Report) -> String {
    if report.checked == 0 {
        return "The tagger can't read any of this prompt's tags, so there's nothing to compare.".to_string();
    }
    let mut parts = vec![format!("{} of {} checkable tags are in the picture", report.found, report.checked)];
    if !report.missing.is_empty() {
        let list: Vec<String> = report
            .missing
            .iter()
            .map(|m| if m.new_weight != m.weight { format!("{} → {}", m.tag, m.new_weight) } else { format!("{} (already {})", m.tag, m.weight) })
            .collect();
        parts.push(format!("missing (a card raises each weight): {}", list.join(", ")));
    }
    if !report.unwanted.is_empty() {
        let list: Vec<String> = report
            .unwanted
            .iter()
            .map(|u| if u.against.is_empty() { u.tag.clone() } else { format!("{} (vs {})", u.tag, u.against) })
            .collect();
        parts.push(format!("seen but not asked for (a card adds each to the negative): {}", list.join(", ")));
    }
    parts.join("; ") + "."
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(name: &str, score: f32) -> ScoredTag {
        ScoredTag { name: name.to_string(), kind: TagKind::General, score }
    }

    #[test]
    fn missing_tags_get_boosted_and_contradictions_go_negative() {
        let scores = vec![
            s("1girl", 0.99), s("solo", 0.95), s("red_hair", 0.05), s("blue_hair", 0.9), s("white_dress", 0.8),
            s("thighhighs", 0.02), s("2girls", 0.1), s("watermark", 0.7), s("smile", 0.3), s("extra_arms", 0.65),
        ];
        let prompt = "masterpiece, 1girl, solo, red hair, (thighhighs:1.4), white dress,\nsmile, <lora:x:0.8>";
        let r = compare(prompt, &scores);
        assert_eq!(r.checked, 6, "{r:?}");
        let missing: Vec<(&str, f32)> = r.missing.iter().map(|m| (m.tag.as_str(), m.new_weight)).collect();
        assert_eq!(missing, vec![("red hair", 1.2), ("thighhighs", 1.5)]);
        let unwanted: Vec<&str> = r.unwanted.iter().map(|u| u.tag.as_str()).collect();
        assert!(unwanted.contains(&"blue hair") && unwanted.contains(&"watermark") && unwanted.contains(&"extra arms"), "{unwanted:?}");
        assert!(!unwanted.contains(&"2girls"));
        // as cards: two weight boosts (opt-in), and one negative card per contradiction
        let (cards, negatives) = cards(&r);
        assert!(cards.iter().all(|c| c.mode == 2));
        assert_eq!(cards[0].original, "red hair");
        assert_eq!(cards[0].suggested_alternative, "(red hair:1.2)");
        assert_eq!(cards[1].suggested_alternative, "(thighhighs:1.5)");
        assert!(negatives.values().any(|t| t == "blue hair") && negatives.len() == r.unwanted.len());
    }

    #[test]
    fn a_second_girl_contradicts_solo() {
        let scores = vec![s("1girl", 0.5), s("solo", 0.05), s("2girls", 0.8), s("multiple_girls", 0.85)];
        let r = compare("1girl, solo", &scores);
        let unwanted: Vec<&str> = r.unwanted.iter().map(|u| u.tag.as_str()).collect();
        assert!(unwanted.contains(&"2girls") && unwanted.contains(&"multiple girls"), "{unwanted:?}");
    }
}
