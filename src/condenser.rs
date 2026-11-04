use crate::analyzer::{ChangeAction, PromptChange, ScanConfig};
use crate::parser::{estimate_clip_tokens, parse_prompt, PromptToken, TokenCategory};
use crate::profiles::CheckpointProfile;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct CondensedOutput {
    pub final_prompt: String,
    /// Same prompt as Slint markdown, with substituted tags colored.
    pub highlighted_markdown: String,
    pub category_summary: String,
    pub before_tokens: usize,
    pub before_chunks: usize,
    pub after_tokens: usize,
    pub after_chunks: usize,
}

/// Rebuilds the final prompt from the original tokens + the current interactive state (`mode`) of each `PromptChange`.
/// - `mode == 0`: Apply primary suggestion (`suggested_alternative` or remove if empty)
/// - `mode == 1`: Apply secondary framing alternative (`framing_alternative` — e.g. keep `ass` and swap `upper body` -> `cowboy shot, from behind`)
/// - `mode == 2`: Restore original (`original` as-is)
///
/// AI cards start at mode 2 and only change the prompt once accepted.
pub fn build_condensed_output(
    raw_prompt: &str,
    base_tokens: &[PromptToken],
    changes: &[PromptChange],
    config: &ScanConfig,
) -> CondensedOutput {
    let (before_tokens, before_chunks) = estimate_clip_tokens(raw_prompt);

    let mut result_tokens: Vec<PromptToken> = base_tokens.to_vec();

    // Apply any user overrides where `mode != 0` (AI cards are handled below)
    for ch in changes.iter().filter(|c| c.action != ChangeAction::AiRewrite) {
        match ch.mode {
            1 => {
                // Mode 1: User chose "Adjust Camera Framing Instead" on a FramingSafeSwap
                // Restore the original body part (e.g. `ass`) at emphasis weight and replace framing with `framing_alternative`
                if ch.action == ChangeAction::FramingSafeSwap && !ch.framing_alternative.is_empty() {
                    let body_alts: HashSet<String> = ch
                        .suggested_alternative
                        .split(',')
                        .map(|s| s.trim().to_lowercase())
                        .collect();
                    result_tokens.retain(|t| {
                        !body_alts.contains(&t.normalized)
                            && t.normalized != "upper body"
                            && t.normalized != "portrait"
                            && t.normalized != "close-up"
                            && t.normalized != "front view"
                    });

                    result_tokens.push(PromptToken {
                        raw: ch.original.clone(),
                        text: ch
                            .original
                            .trim_matches(|c| c == '(' || c == ')')
                            .split(':')
                            .next()
                            .unwrap_or(&ch.original)
                            .to_string(),
                        normalized: ch.original.to_lowercase(),
                        weight: config.emphasis_weight,
                        had_runaway_weight: false,
                        replaced: false,
                        added: false,
                        original_index: 50,
                        category: TokenCategory::PinnedEmphasis,
                    });

                    for f_part in ch
                        .framing_alternative
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                    {
                        result_tokens.push(PromptToken {
                            raw: f_part.to_string(),
                            text: f_part.to_string(),
                            normalized: f_part.to_lowercase(),
                            weight: 1.0,
                            had_runaway_weight: false,
                            replaced: true,
                            added: false,
                            original_index: 60,
                            category: TokenCategory::FramingPose,
                        });
                    }
                }
            }
            2 => {
                // Mode 2: User chose "Restore Original" on this specific change
                if !ch.suggested_alternative.is_empty() {
                    let alts: HashSet<String> = ch
                        .suggested_alternative
                        .split(',')
                        .map(|s| s.trim().to_lowercase())
                        .collect();
                    result_tokens.retain(|t| !alts.contains(&t.normalized));
                }
                let orig_clean = ch.original.trim();
                if !orig_clean.is_empty()
                    && !result_tokens.iter().any(|t| t.raw == orig_clean || t.text == orig_clean)
                {
                    result_tokens.push(PromptToken {
                        raw: orig_clean.to_string(),
                        text: orig_clean.to_string(),
                        normalized: orig_clean.to_lowercase(),
                        weight: 1.0,
                        had_runaway_weight: false,
                        replaced: false,
                        added: false,
                        original_index: 500,
                        category: TokenCategory::Attire,
                    });
                }
            }
            _ => {}
        }
    }

    for ch in changes.iter().filter(|c| c.action == ChangeAction::AiRewrite) {
        apply_ai_card(&mut result_tokens, ch);
    }

    // Order tokens:
    // - If `cleanup_enabled` is TRUE: group cleanly by Checkpoint category hierarchy
    // - If `cleanup_enabled` is FALSE: preserve the user's exact `original_index` order!
    if config.cleanup_enabled {
        result_tokens.sort_by_key(|t| (t.category, t.original_index));
    } else {
        result_tokens.sort_by_key(|t| t.original_index);
    }

    let use_weights = config.checkpoint.uses_explicit_weights();
    let plain = |t: &PromptToken| t.format_for_output(use_weights);
    let marked = |t: &PromptToken| highlight_md(&t.format_for_output(use_weights), Highlight::of(t));
    let (final_prompt, highlighted_markdown) =
        if config.checkpoint == CheckpointProfile::Flux && config.cleanup_enabled {
            (
                format_as_flux_natural_language(&result_tokens, plain),
                format_as_flux_natural_language(&result_tokens, marked),
            )
        } else {
            (
                result_tokens.iter().map(plain).collect::<Vec<_>>().join(", "),
                result_tokens.iter().map(marked).collect::<Vec<_>>().join(", "),
            )
        };

    let category_summary = build_category_breakdown(&result_tokens, use_weights);
    let (after_tokens, after_chunks) = estimate_clip_tokens(&final_prompt);

    CondensedOutput {
        final_prompt,
        highlighted_markdown,
        category_summary,
        before_tokens,
        before_chunks,
        after_tokens,
        after_chunks,
    }
}

fn format_as_flux_natural_language(tokens: &[PromptToken], fmt: impl Fn(&PromptToken) -> String) -> String {
    let mut subject_parts = Vec::new();
    let mut emphasis_parts = Vec::new();
    let mut framing_parts = Vec::new();
    let mut body_parts = Vec::new();
    let mut attire_parts = Vec::new();
    let mut scene_parts = Vec::new();

    for t in tokens {
        match t.category {
            TokenCategory::QualityMeta => {}
            TokenCategory::SubjectCount => subject_parts.push(fmt(t)),
            TokenCategory::PinnedEmphasis => emphasis_parts.push(fmt(t)),
            TokenCategory::FramingPose => framing_parts.push(fmt(t)),
            TokenCategory::CharacterBody => body_parts.push(fmt(t)),
            TokenCategory::Attire | TokenCategory::NaturalLanguageClause => {
                attire_parts.push(fmt(t))
            }
            TokenCategory::EnvironmentLighting => scene_parts.push(fmt(t)),
            TokenCategory::LoraOrSpecial => scene_parts.push(fmt(t)),
        }
    }

    let mut clauses = Vec::new();
    if !framing_parts.is_empty() || !subject_parts.is_empty() {
        let framing = if framing_parts.is_empty() {
            "A detailed depiction".to_string()
        } else {
            format!("A {} shot", framing_parts.join(", "))
        };
        let subj = if subject_parts.is_empty() {
            String::new()
        } else {
            format!(" featuring {}", subject_parts.join(", "))
        };
        clauses.push(format!("{}{}", framing, subj));
    }

    if !emphasis_parts.is_empty() {
        clauses.push(format!(
            "with prominent focus on {}",
            emphasis_parts.join(", ")
        ));
    }

    if !body_parts.is_empty() {
        clauses.push(format!("characterized by {}", body_parts.join(", ")));
    }

    if !attire_parts.is_empty() {
        clauses.push(attire_parts.join(", "));
    }

    if !scene_parts.is_empty() {
        clauses.push(format!("set in {}", scene_parts.join(", ")));
    }

    clauses.join(", ")
}

fn build_category_breakdown(tokens: &[PromptToken], use_weights: bool) -> String {
    let categories = [
        TokenCategory::QualityMeta,
        TokenCategory::SubjectCount,
        TokenCategory::PinnedEmphasis,
        TokenCategory::FramingPose,
        TokenCategory::CharacterBody,
        TokenCategory::Attire,
        TokenCategory::NaturalLanguageClause,
        TokenCategory::EnvironmentLighting,
        TokenCategory::LoraOrSpecial,
    ];

    let mut lines = Vec::new();
    for cat in categories {
        let items: Vec<String> = tokens
            .iter()
            .filter(|t| t.category == cat)
            .map(|t| t.format_for_output(use_weights))
            .collect();
        if !items.is_empty() {
            lines.push(format!("[{}] {}", cat.label(), items.join(", ")));
        }
    }
    if lines.is_empty() {
        "No tokens parsed yet.".to_string()
    } else {
        lines.join("\n")
    }
}

/// How a tag is marked in the After view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Highlight {
    None,
    /// Put in place of one of the user's tags (amber).
    Substituted,
    /// New, not in the user's prompt (green).
    Added,
}

impl Highlight {
    pub fn of(t: &PromptToken) -> Self {
        if t.added {
            Self::Added
        } else if t.replaced {
            Self::Substituted
        } else {
            Self::None
        }
    }
}

/// Escapes text for Slint's markdown parser so prompt syntax like `(tag:1.2)`,
/// `score_9` or `<lora:x>` shows literally, then optionally colors it.
pub fn highlight_md(text: &str, highlight: Highlight) -> String {
    let mut escaped = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        match c {
            // Slint's interpolation placeholder; never valid in user text
            '\u{e541}' => {}
            '\n' | '\r' => escaped.push(' '),
            c if c.is_ascii_punctuation() => {
                escaped.push('\\');
                escaped.push(c);
            }
            c => escaped.push(c),
        }
    }
    match highlight {
        Highlight::None => escaped,
        Highlight::Substituted => format!("<font color=\"#F5C77E\"><u>{escaped}</u></font>"),
        Highlight::Added => format!("<font color=\"#8CD9A8\"><u>{escaped}</u></font>"),
    }
}

/// Highlights an AI-written prompt against the user's original: tags the user
/// didn't have are marked as added. Sentences (Flux) are compared per comma clause.
/// Returns the markdown and how many tags are new.
pub fn highlight_against_original(raw_prompt: &str, text: &str) -> (String, usize) {
    let original: HashSet<String> = parse_prompt(raw_prompt).into_iter().map(|t| t.normalized).collect();
    let mut new_count = 0;
    let md = parse_prompt(text)
        .iter()
        .map(|t| {
            let is_new = !original.contains(&t.normalized);
            new_count += is_new as usize;
            highlight_md(&t.raw, if is_new { Highlight::Added } else { Highlight::None })
        })
        .collect::<Vec<_>>()
        .join(", ");
    (md, new_count)
}

/// Applies one accepted AI card in place of the tag(s) it names.
/// mode 0 = replace with the suggestion (empty suggestion = remove),
/// mode 1 = keep the tag and use the framing alternative, mode 2 = no change.
fn apply_ai_card(tokens: &mut Vec<PromptToken>, ch: &PromptChange) {
    let targets: HashSet<String> = parse_prompt(&ch.original).into_iter().map(|t| t.normalized).collect();
    let is_add = targets.is_empty();
    let anchor = if is_add {
        800 // new details go after the user's own tags in their category
    } else {
        match tokens.iter().find(|t| targets.contains(&t.normalized)) {
            Some(first) => first.original_index,
            None => return,
        }
    };

    let insert = |tokens: &mut Vec<PromptToken>, text: &str| {
        for mut t in parse_prompt(text) {
            if tokens.iter().any(|e| e.normalized == t.normalized) {
                continue;
            }
            t.replaced = !is_add;
            t.added = is_add;
            t.original_index = anchor;
            tokens.push(t);
        }
    };

    match ch.mode {
        0 => {
            tokens.retain(|t| !targets.contains(&t.normalized));
            insert(tokens, &ch.suggested_alternative);
        }
        1 => insert(tokens, &ch.framing_alternative),
        _ => {}
    }
}

/// Cleans up a model-written prompt: balances brackets, unwraps a prompt wrapped
/// in one big bracket group, fixes bare `tag:1.0` weights, clamps weights to what
/// the checkpoint tolerates and drops duplicates. Prose (Flux) is left as is.
pub fn tidy_ai_prompt(text: &str, checkpoint: CheckpointProfile) -> String {
    let text = text.trim();
    if !checkpoint.uses_explicit_weights() {
        return text.to_string();
    }
    let balanced = balance_brackets(text);

    // Flatten groups that contain several tags, e.g. "(a, b, c)" -> a, b, c
    let mut tokens: Vec<PromptToken> = Vec::new();
    let mut queue = parse_prompt(&balanced);
    queue.reverse();
    while let Some(t) = queue.pop() {
        // never split {a, b|c} variation groups
        if t.text.contains(',') && t.category != TokenCategory::LoraOrSpecial && !t.text.starts_with('{') {
            let mut inner = parse_prompt(&balance_brackets(&t.text));
            inner.reverse();
            queue.extend(inner);
            continue;
        }
        tokens.push(t);
    }

    let max_w = checkpoint.max_safe_weight();
    let mut seen = HashSet::new();
    tokens
        .into_iter()
        .filter(|t| seen.insert(t.normalized.clone()))
        .map(|mut t| {
            if t.category == TokenCategory::QualityMeta {
                t.weight = 1.0; // quality tags don't need weights
            }
            t.weight = t.weight.min(max_w);
            t.format_for_output(true)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Drops `)`/`]` with no opener and `(`/`[` that are never closed.
fn balance_brackets(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut keep = vec![true; chars.len()];
    let mut open: Vec<usize> = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '(' | '[' => open.push(i),
            ')' | ']' => {
                let want = if c == ')' { '(' } else { '[' };
                match open.last() {
                    Some(&j) if chars[j] == want => {
                        open.pop();
                    }
                    _ => keep[i] = false,
                }
            }
            _ => {}
        }
    }
    for j in open {
        keep[j] = false;
    }
    chars.iter().zip(keep).filter(|(_, k)| *k).map(|(c, _)| c).collect()
}

/// The user's tags that an AI rewrite left out (filler the rules would drop anyway is ignored).
pub fn dropped_tags(raw_prompt: &str, ai_text: &str, checkpoint: CheckpointProfile) -> Vec<String> {
    let kept: HashSet<String> = parse_prompt(ai_text).into_iter().map(|t| t.normalized).collect();
    // An enriched version keeps the tag: every word of it appears in one AI tag, in any
    // position ("red cape" -> "flowing red velvet cape", "long white hair" -> "long flowing white hair").
    let words = |t: &str| -> Vec<String> {
        t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect()
    };
    let enriched = |tag: &str| {
        let want = words(tag);
        !want.is_empty() && kept.iter().any(|k| {
            let have = words(k);
            want.iter().all(|w| have.contains(w))
        })
    };
    let mut seen = HashSet::new();
    parse_prompt(raw_prompt)
        .into_iter()
        .filter(|t| !kept.contains(&t.normalized) && !enriched(&t.normalized))
        .filter(|t| checkpoint.check_bloat_tag(&t.normalized).is_none())
        .filter(|t| seen.insert(t.normalized.clone()))
        .map(|t| t.text)
        .collect()
}

/// Recommended negative prompt: the checkpoint's standard negatives plus tags that
/// keep the camera on the target view, minus anything the positive prompt asks for.
pub fn recommended_negative(config: &ScanConfig, positive: &str) -> String {
    if config.checkpoint.default_negative().is_empty() {
        return String::new();
    }
    let base = format!("{}, {}", config.checkpoint.default_negative(), config.target_view.negative_tags());
    merge_negative(&base, positive)
}

/// Cleans a negative prompt: one tag per entry, no duplicates, and nothing that
/// the positive prompt contains (a tag in both cancels itself out).
pub fn merge_negative(negative: &str, positive: &str) -> String {
    let wanted: HashSet<String> = parse_prompt(positive).into_iter().map(|t| t.normalized).collect();
    let mut seen = HashSet::new();
    parse_prompt(negative)
        .into_iter()
        .filter(|t| !wanted.contains(&t.normalized) && !contradicts_positive(&t.normalized, &wanted))
        .filter(|t| seen.insert(t.normalized.clone()))
        .map(|t| t.raw)
        .collect::<Vec<_>>()
        .join(", ")
}

/// A negative like "indoors" fights a positive "classroom" even though the words differ.
/// Each rule: (words that mark the negative tag, words in the positive that it would fight).
const NEGATIVE_CONFLICTS: &[(&[&str], &[&str])] = &[
    (
        &["indoor", "interior"],
        &["indoors", "room", "bedroom", "classroom", "kitchen", "office", "library", "bathroom", "interior", "house",
          "cafe", "restaurant", "desk", "hallway", "shop", "train interior", "car interior"],
    ),
    (
        &["outdoor", "exterior"],
        &["outdoors", "forest", "beach", "street", "city", "sky", "park", "field", "garden", "mountain", "shrine",
          "ocean", "river", "lake", "rooftop", "snow", "desert"],
    ),
    (&["daytime", "day time", "sunlight", "sunny"], &["day", "daytime", "sunlight", "afternoon", "morning", "noon", "sunny", "blue sky"]),
    (&["night", "nighttime", "dark"], &["night", "night sky", "moon", "moonlight", "stars", "starry sky", "dark"]),
];

fn contradicts_positive(negative_tag: &str, positive: &HashSet<String>) -> bool {
    // "people"/"person" would suppress the prompt's own character
    if matches!(negative_tag, "people" | "person" | "human" | "humans" | "woman" | "girl")
        && positive.iter().any(|p| p.ends_with("girl") || p.ends_with("boy") || p.ends_with("girls") || p.ends_with("boys") || p == "solo")
    {
        return true;
    }
    NEGATIVE_CONFLICTS.iter().any(|(marks, fights)| {
        marks.iter().any(|m| negative_tag.contains(m))
            && positive.iter().any(|p| fights.iter().any(|f| p == f || p.contains(f)))
    })
}

/// The tags highlighted with `kind` in markup made by `highlight_md`, as plain text.
pub fn marked_tags(markdown: &str, kind: Highlight) -> Vec<String> {
    let color = match kind {
        Highlight::Substituted => "#F5C77E",
        Highlight::Added => "#8CD9A8",
        Highlight::None => return Vec::new(),
    };
    let open = format!("<font color=\"{color}\"><u>");
    let mut out = Vec::new();
    let mut rest = markdown;
    while let Some(i) = rest.find(&open) {
        rest = &rest[i + open.len()..];
        let Some(end) = rest.find("</u></font>") else { break };
        // undo the markdown escaping ("\(" -> "(")
        let mut tag = String::new();
        let mut chars = rest[..end].chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(n) = chars.next() {
                    tag.push(n);
                }
            } else {
                tag.push(c);
            }
        }
        out.push(tag);
        rest = &rest[end..];
    }
    out
}
