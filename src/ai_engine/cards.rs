//! Part of `ai_engine`; see mod.rs.
use super::*;

/// Parses the model's JSON reply. If the reply was cut off (output limit reached),
/// the change entries that did arrive complete are still returned.
pub fn extract_json_from_llm(raw_text: &str) -> Option<AiJsonOutput> {
    if let (Some(start), Some(end)) = (raw_text.find('{'), raw_text.rfind('}')) {
        if end > start {
            if let Ok(out) = serde_json::from_str::<AiJsonOutput>(&raw_text[start..=end]) {
                return Some(out);
            }
        }
    }
    if let Some(out) = close_truncated_json(raw_text) {
        return Some(out);
    }
    let changes = salvage_change_objects(raw_text);
    if changes.is_empty() {
        None
    } else {
        Some(AiJsonOutput { changes, ..Default::default() })
    }
}

/// Repairs a reply that stopped mid-way: cuts it after the last complete value
/// and closes the brackets still open there, so everything before the cut
/// (scene analysis, improved prompt, finished changes) is kept.
pub(crate) fn close_truncated_json(text: &str) -> Option<AiJsonOutput> {
    let start = text.find('{')?;
    let body = &text[start..];
    let (mut stack, mut in_str, mut escaped) = (Vec::<char>::new(), false, false);
    // (byte index just after a complete value, closers needed at that point)
    let mut cuts: Vec<(usize, String)> = Vec::new();
    let closers = |stack: &[char]| stack.iter().rev().map(|c| if *c == '{' { '}' } else { ']' }).collect::<String>();
    for (i, c) in body.char_indices() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => {
                    in_str = false;
                    // a finished string value inside an array, e.g. one entry of "issues"
                    if stack.last() == Some(&'[') {
                        cuts.push((i + 1, closers(&stack)));
                    }
                }
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' | '[' => stack.push(c),
            '}' | ']' => {
                stack.pop();
                if stack.is_empty() {
                    break;
                }
                cuts.push((i + 1, closers(&stack)));
            }
            ',' => cuts.push((i, closers(&stack))),
            _ => {}
        }
    }
    cuts.iter().rev().take(40).find_map(|(at, close)| {
        let candidate = format!("{}{}", &body[..*at], close);
        let mut out = serde_json::from_str::<AiJsonOutput>(&candidate).ok()?;
        // Only keep changes the model actually finished; the last one may be half-written.
        out.changes = salvage_change_objects(body);
        Some(out)
    })
}

/// Picks every complete `{...}` object nested one level inside the outer object
/// (the entries of `"changes": [...]`), ignoring braces inside strings.
pub(crate) fn salvage_change_objects(text: &str) -> Vec<AiChangeItem> {
    let mut out = Vec::new();
    let (mut depth, mut in_str, mut escaped, mut start) = (0i32, false, false, None);
    for (i, c) in text.char_indices() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => {
                depth += 1;
                if depth == 2 {
                    start = Some(i);
                }
            }
            '}' => {
                if depth == 2 {
                    if let Some(st) = start.take() {
                        // `scene_breakdown` is also a nested object; skip anything that isn't a change
                        if let Ok(item) = serde_json::from_str::<AiChangeItem>(&text[st..=i]) {
                            if !item.action.is_empty() {
                                out.push(item);
                            }
                        }
                    }
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    out
}

/// First id used for AI cards, so they never collide with rule-card ids.
pub const AI_CARD_ID_BASE: usize = 10_000;

/// Turns AI findings into cards the user can accept one by one (they start as
/// "Keep Original", so nothing changes until a card is accepted).
/// Findings are dropped when their `original` isn't in the prompt, when a rule
/// card already covers that tag, or when they suggest no actual change.
/// A tag the app's own layout writes, never a content change: the checkpoint's quality tags,
/// the BREAK separator, a group count (`4girls`, `2boys`, `multiple girls`) or a position (`far left`,
/// `second from the right`, `in the middle`…).
pub fn is_layout_tag(normalized: &str, checkpoint: CheckpointProfile) -> bool {
    let t = normalized.trim().to_lowercase();
    if t == "break" {
        return true; // the chunk separator between characters
    }
    // the parser normalizes `score_9` to `score 9`: compare both spellings
    if checkpoint.canonical_quality_tags().iter().any(|q| *q == t || q.replace('_', " ") == t.replace('_', " ")) {
        return true;
    }
    let count = |s: &str| {
        let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
        digits > 0
            && digits < s.len()
            && matches!(&s[digits..], "girls" | "boys" | "others" | "girl" | "boy" | "other" | "people")
            && !matches!(s, "1girl" | "1boy" | "1other") // a character's own count tag is content
    };
    if count(&t) || matches!(t.as_str(), "multiple girls" | "multiple boys" | "multiple others") {
        return true;
    }
    let pos = t.trim_start_matches("on the ").trim_start_matches("the ").trim_start_matches("on ");
    matches!(
        pos,
        "far left" | "far right" | "left" | "right" | "middle" | "center" | "in the middle" | "in the center"
            | "second from left" | "second from right" | "second from the left" | "second from the right"
    )
}

#[cfg(test)]
pub fn convert_ai_changes(
    ai_items: &[AiChangeItem],
    prompt_tokens: &[crate::parser::PromptToken],
    rule_changes: &[PromptChange],
) -> Vec<PromptChange> {
    convert_ai_changes_for(ai_items, prompt_tokens, rule_changes, false)
}

/// A quality tag of any checkpoint (`ensure_quality_tags` writes the right ones, so an AI card
/// offering one is never needed).
pub(crate) fn is_quality_tag_any(normalized: &str) -> bool {
    use CheckpointProfile::*;
    [Anima, PonyV6, Illustrious, Sdxl, Sd15].iter().any(|c| {
        let t = normalized.trim().to_lowercase().replace('_', " ");
        c.canonical_quality_tags().iter().any(|q| q.replace('_', " ") == t)
    })
}

/// `convert_ai_changes`; `multi` = a 2-4 Characters tab, where group counts, positions and BREAK are
/// written by the app's own layout. On the 1 Character tab they are content (`2girls` there is a real
/// change the user should see as a card).
pub fn convert_ai_changes_for(
    ai_items: &[AiChangeItem],
    prompt_tokens: &[crate::parser::PromptToken],
    rule_changes: &[PromptChange],
    multi: bool,
) -> Vec<PromptChange> {
    use crate::parser::parse_prompt;
    use std::collections::HashSet;

    let in_prompt: HashSet<&str> = prompt_tokens.iter().map(|t| t.normalized.as_str()).collect();
    let covered: HashSet<String> = rule_changes
        .iter()
        .flat_map(|c| parse_prompt(&c.original))
        .map(|t| t.normalized)
        .collect();
    let mut seen: HashSet<String> = HashSet::new();

    let mut out = Vec::new();
    for item in ai_items {
        let action = item.action.to_lowercase();
        let suggestion: Vec<String> =
            parse_prompt(&item.suggested_alternative).into_iter().map(|t| t.normalized).collect();
        let named: Vec<String> = parse_prompt(&item.original).into_iter().map(|t| t.normalized).collect();
        // An "add" names no tag of the prompt. Models often put the new tag in `original` too
        // (original == suggested, or a tag the prompt doesn't have): that is still an addition.
        let misfiled_add = action == "add"
            && !named.is_empty()
            && (named == suggestion || !named.iter().all(|t| in_prompt.contains(t.as_str())));
        let targets: Vec<String> = if action == "add"
            && (misfiled_add || matches!(item.original.trim().to_lowercase().as_str(), "none" | "n/a" | "null"))
        {
            Vec::new()
        } else {
            named
        };
        // Group count and position tags are written by the multi-character layout: not a card.
        let layout = |t: &String| is_quality_tag_any(t) || (multi && is_layout_tag(t, CheckpointProfile::Flux));
        if targets.is_empty() && !suggestion.is_empty() && suggestion.iter().all(layout) {
            continue;
        }

        if targets.is_empty() {
            // A new detail: keep it only if it adds something not already in the prompt.
            if suggestion.is_empty() || suggestion.iter().all(|t| in_prompt.contains(t.as_str())) {
                continue;
            }
        } else {
            if !targets.iter().all(|t| in_prompt.contains(t.as_str())) {
                continue; // names a tag that isn't in the prompt
            }
            if targets.iter().any(|t| covered.contains(t)) {
                continue; // a rule card already handles it
            }
            // Same text back is only a change if the weight differs (reweight)
            if item.suggested_alternative.trim() == item.original.trim() && item.framing_alternative.trim().is_empty() {
                continue;
            }
            if suggestion == targets && !item.suggested_alternative.contains(':') && !item.original.contains(':') {
                continue; // same tag back, no weight change
            }
        }
        // For additions, show only what is actually new
        let added_text = if targets.is_empty() {
            parse_prompt(&item.suggested_alternative)
                .into_iter()
                .filter(|t| !in_prompt.contains(t.normalized.as_str()))
                .map(|t| t.raw)
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            String::new()
        };
        let key = if targets.is_empty() { suggestion.join(",") } else { targets.join(",") };
        if !seen.insert(key) {
            continue;
        }
        // "remove" with a replacement text is really a replace
        let is_remove = action == "remove" && item.suggested_alternative.trim().is_empty();
        let kind = match action.as_str() {
            "framing_safe_swap" => "framing-safe swap",
            "remove" if is_remove => "remove",
            "remove" => "replace",
            "add" => "add detail",
            "reweight" => "reweight",
            _ => "replace",
        };
        let reason = if item.reason.trim().is_empty() {
            format!("[AI · {kind}]")
        } else {
            format!("[AI · {kind}] {}", item.reason.trim())
        };
        out.push(PromptChange {
            id: AI_CARD_ID_BASE + out.len(),
            // an addition names no tag of the prompt; a misfiled one ("seashells" -> "seashells") would
            // otherwise be applied as a replace of a tag that isn't there and do nothing
            original: if targets.is_empty() { String::new() } else { item.original.trim().to_string() },
            action: ChangeAction::AiRewrite,
            reason,
            suggested_alternative: if is_remove {
                String::new()
            } else if targets.is_empty() {
                added_text
            } else {
                item.suggested_alternative.trim().to_string()
            },
            // A camera alternative only makes sense for framing swaps; models fill it with noise otherwise.
            framing_alternative: if action == "framing_safe_swap"
                && item.framing_alternative.trim() != item.suggested_alternative.trim()
            {
                item.framing_alternative.trim().to_string()
            } else {
                String::new()
            },
            mode: 2,
        });
    }
    out
}

// ============================================================================
// Image -> prompt with a vision model (Gemma 4 12B), guided by the tagger's tags
// ============================================================================
