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

    let is_active_mode1_framing = |c: &PromptChange| {
        !c.action.is_opt_in()
            && c.mode == 1
            && c.action == ChangeAction::FramingSafeSwap
            && parse_non_empty_tags(&c.framing_alternative).is_some()
    };

    let mut result_tokens: Vec<PromptToken> = base_tokens.to_vec();
    let needs_orig_parsed = changes.iter().any(|c| {
        !c.action.is_opt_in()
            && parse_non_empty_tags(&c.original).is_some()
            && (c.mode == 2 || is_active_mode1_framing(c))
    });
    let orig_parsed: Vec<PromptToken> = if needs_orig_parsed {
        parse_prompt(raw_prompt)
    } else {
        Vec::new()
    };
    let mut used_orig: Vec<bool> = vec![false; orig_parsed.len()];

    // Suggestions still active in mode 0 (or mode 1 without an active framing override) must not be
    // deleted by sibling mode 1 / mode 2 overrides.
    let active_mode0_alts: HashSet<String> = changes
        .iter()
        .filter(|c| !c.action.is_opt_in() && (c.mode == 0 || (c.mode == 1 && !is_active_mode1_framing(c))))
        .flat_map(|c| normalized_tag_set(&c.suggested_alternative))
        .collect();
    // Tokens kept (original) or inserted (framing_alternative) by active mode 1 framing overrides
    // so neither sibling mode 1 cards nor mode 2 restorations clobber them regardless of card order.
    let protected_mode1: HashSet<String> = changes
        .iter()
        .filter(|c| is_active_mode1_framing(c))
        .flat_map(|c| {
            normalized_tag_set(&c.original)
                .into_iter()
                .chain(normalized_tag_set(&c.framing_alternative))
        })
        .collect();

    // Apply any user overrides where `mode != 0` (AI and Recommendation cards are handled below)
    for ch in changes
        .iter()
        .filter(|c| !c.action.is_opt_in())
    {
        match ch.mode {
            1 => {
                // Mode 1: User chose "Adjust Camera Framing Instead" on a FramingSafeSwap
                // Restore the original body part (e.g. `ass`) at emphasis weight and replace framing with `framing_alternative`
                if ch.action == ChangeAction::FramingSafeSwap {
                    if let Some(new_framing_tokens) = parse_non_empty_tags(&ch.framing_alternative) {
                        let body_alts = normalized_tag_set(&ch.suggested_alternative);
                        let new_framing: HashSet<String> =
                            new_framing_tokens.iter().map(|t| t.normalized.clone()).collect();

                        result_tokens.retain(|t| {
                            let drop_body_alt = t.replaced
                                && body_alts.contains(&t.normalized)
                                && !active_mode0_alts.contains(&t.normalized)
                                && !protected_mode1.contains(&t.normalized);
                            let drop_framing = is_framing_override_target(&t.normalized)
                                && !new_framing.contains(&t.normalized)
                                && !protected_mode1.contains(&t.normalized);
                            !drop_body_alt && !drop_framing
                        });

                        for (mut orig_tok, matched_in_orig) in
                            resolve_original_tokens_with_match(&ch.original, &orig_parsed, &mut used_orig, 50)
                        {
                            orig_tok.weight = config.emphasis_weight;
                            orig_tok.had_runaway_weight = false;
                            orig_tok.replaced = false;
                            orig_tok.added = false;
                            orig_tok.category = TokenCategory::PinnedEmphasis;
                            push_resolved_token(&mut result_tokens, orig_tok, matched_in_orig);
                        }

                        for mut f_tok in new_framing_tokens {
                            if result_tokens.iter().any(|e| e.normalized == f_tok.normalized) {
                                continue;
                            }
                            normalize_token_inline_whitespace(&mut f_tok);
                            f_tok.weight = 1.0;
                            f_tok.had_runaway_weight = false;
                            f_tok.replaced = true;
                            f_tok.added = false;
                            f_tok.original_index = 60;
                            f_tok.category = TokenCategory::FramingPose;
                            result_tokens.push(f_tok);
                        }
                    }
                }
            }
            2 => {
                // Mode 2: User chose "Restore Original" on this specific change
                let alts = normalized_tag_set(&ch.suggested_alternative);
                if !alts.is_empty() {
                    let in_place_action = matches!(
                        ch.action,
                        ChangeAction::KeptEmphasized | ChangeAction::WeightClamped
                    );
                    let orig_raw_trimmed = ch.original.trim();
                    let has_exact_raw = in_place_action
                        && !orig_raw_trimmed.is_empty()
                        && result_tokens.iter().any(|t| t.raw.trim() == orig_raw_trimmed);
                    result_tokens.retain(|t| {
                        let is_alt_token = match ch.action {
                            ChangeAction::KeptEmphasized | ChangeAction::WeightClamped => {
                                if has_exact_raw {
                                    t.raw.trim() == orig_raw_trimmed
                                } else {
                                    alts.contains(&t.normalized)
                                }
                            }
                            ChangeAction::AngleChoice
                            | ChangeAction::PoseChoice
                            | ChangeAction::ProportionChoice => {
                                (t.added || t.replaced) && alts.contains(&t.normalized)
                            }
                            _ => t.replaced && alts.contains(&t.normalized),
                        };
                        let drop = is_alt_token
                            && !active_mode0_alts.contains(&t.normalized)
                            && !protected_mode1.contains(&t.normalized);
                        !drop
                    });
                }
                for (restored, matched_in_orig) in
                    resolve_original_tokens_with_match(&ch.original, &orig_parsed, &mut used_orig, 500)
                {
                    push_resolved_token(&mut result_tokens, restored, matched_in_orig);
                }
            }
            _ => {}
        }
    }

    for ch in changes
        .iter()
        .filter(|c| c.action.is_opt_in())
    {
        apply_ai_card(&mut result_tokens, ch);
    }

    // Composition / Aspect-Ratio guard (16:9 / 21:9 widescreen & 9:16 portrait single-character):
    // Only on Base / HiresFix (never inject global framing/solo clauses into Detailer/Inpaint crops).
    let stage_allows_composition_guard = matches!(
        config.stage,
        crate::profiles::GenerationStage::Base | crate::profiles::GenerationStage::HiresFix
    );
    if stage_allows_composition_guard
        && !config.two_char
        && (config.aspect_ratio.is_widescreen() || config.aspect_ratio.is_portrait())
        && crate::analyzer::is_single_character_tokens(&result_tokens)
    {
        if config.checkpoint != CheckpointProfile::Flux
            && !result_tokens.iter().any(|t| t.normalized == "solo")
        {
            result_tokens.push(PromptToken {
                raw: "solo".to_string(),
                text: "solo".to_string(),
                normalized: "solo".to_string(),
                weight: 1.0,
                had_runaway_weight: false,
                replaced: false,
                added: true,
                original_index: 5,
                category: TokenCategory::SubjectCount,
            });
        }
        if config.aspect_ratio.is_widescreen() && config.checkpoint.supports_natural_language() {
            for (idx, phrase) in [(6usize, "single character only"), (7usize, "one person alone in frame")] {
                if !result_tokens.iter().any(|t| t.normalized == phrase) {
                    result_tokens.push(PromptToken {
                        raw: phrase.to_string(),
                        text: phrase.to_string(),
                        normalized: phrase.to_string(),
                        weight: 1.0,
                        had_runaway_weight: false,
                        replaced: false,
                        added: true,
                        original_index: idx,
                        category: TokenCategory::SubjectCount,
                    });
                }
            }
        }
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

/// Parses a tag list when it contains at least one non-empty token.
pub(crate) fn parse_non_empty_tags(input: &str) -> Option<Vec<PromptToken>> {
    if input.trim().is_empty() {
        return None;
    }
    let tokens = parse_prompt(input);
    (!tokens.is_empty()).then_some(tokens)
}

/// Parses a tag list into normalized forms, ignoring empty segments and weight wrappers.
fn normalized_tag_set(input: &str) -> HashSet<String> {
    parse_non_empty_tags(input)
        .map(|tokens| tokens.into_iter().map(|t| t.normalized).collect())
        .unwrap_or_default()
}

/// Collapses runs of inline whitespace in `text` and `raw` on synthesized framing tokens.
fn normalize_token_inline_whitespace(tok: &mut PromptToken) {
    if tok.text.split_whitespace().count() > 0 {
        let cleaned = tok.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned != tok.text {
            tok.text = cleaned.clone();
            tok.raw = cleaned;
        }
    }
}

/// Inserts a resolved original token unless that occurrence is already present.
fn push_resolved_token(result_tokens: &mut Vec<PromptToken>, tok: PromptToken, matched_in_orig: bool) {
    let already_present = if matched_in_orig {
        result_tokens
            .iter()
            .any(|t| t.original_index == tok.original_index && t.normalized == tok.normalized)
    } else {
        result_tokens.iter().any(|t| t.normalized == tok.normalized)
    };
    if !already_present {
        result_tokens.push(tok);
    }
}

/// Resolves original tags against `orig_parsed` to preserve weights, categories, and original indices.
/// Ranks matches as: exact raw > exact text + weight > exact text > normalized + weight > normalized,
/// resolving more-specific queries first and preferring unclaimed occurrences to disambiguate
/// multiple tags sharing the same text.
#[cfg(test)]
fn resolve_original_tokens(
    orig_raw: &str,
    orig_parsed: &[PromptToken],
    used_orig: &mut [bool],
    fallback_index: usize,
) -> Vec<PromptToken> {
    resolve_original_tokens_with_match(orig_raw, orig_parsed, used_orig, fallback_index)
        .into_iter()
        .map(|(tok, _)| tok)
        .collect()
}

/// Tiered matching predicate for `resolve_original_tokens_with_match` (`0` = strongest).
fn matches_resolution_tier(parsed: &PromptToken, cand: &PromptToken, tier: usize) -> bool {
    match tier {
        0 => cand.raw.trim() == parsed.raw.trim(),
        1 => cand.text.trim() == parsed.text.trim() && (cand.weight - parsed.weight).abs() < 0.01,
        2 => cand.text.trim() == parsed.text.trim(),
        3 => cand.normalized == parsed.normalized && (cand.weight - parsed.weight).abs() < 0.01,
        4 => cand.normalized == parsed.normalized,
        _ => false,
    }
}

/// Same as `resolve_original_tokens`, also returning whether each token matched an entry in `orig_parsed`.
fn resolve_original_tokens_with_match(
    orig_raw: &str,
    orig_parsed: &[PromptToken],
    used_orig: &mut [bool],
    fallback_index: usize,
) -> Vec<(PromptToken, bool)> {
    let Some(parsed_tokens) = parse_non_empty_tags(orig_raw) else {
        return Vec::new();
    };

    // Resolve higher-specificity queries first so a bare text tag cannot claim an occurrence
    // that a sibling exact-raw or weighted query in the same list specifically names.
    let mut order: Vec<(usize, usize)> = parsed_tokens
        .iter()
        .enumerate()
        .map(|(pos, parsed)| {
            let best_tier = (0..5)
                .find(|&tier| orig_parsed.iter().any(|t| matches_resolution_tier(parsed, t, tier)))
                .unwrap_or(5);
            (best_tier, pos)
        })
        .collect();
    order.sort_unstable();

    let mut resolved: Vec<Option<(PromptToken, bool)>> = vec![None; parsed_tokens.len()];
    let mut parsed_opts: Vec<Option<PromptToken>> = parsed_tokens.into_iter().map(Some).collect();
    for (_, pos) in order {
        let mut parsed = parsed_opts[pos].take().expect("index visited once");
        let find_unused = |tier: usize| -> Option<usize> {
            orig_parsed
                .iter()
                .enumerate()
                .position(|(i, t)| !used_orig.get(i).copied().unwrap_or(false) && matches_resolution_tier(&parsed, t, tier))
        };
        let find_any = |tier: usize| -> Option<usize> {
            orig_parsed
                .iter()
                .position(|t| matches_resolution_tier(&parsed, t, tier))
        };

        // Exact-identity tiers (raw, text+weight) may reuse an already-claimed exact slot rather
        // than falling back to a wrong-weight token; weak tiers check all unused tiers first.
        let matched = find_unused(0)
            .or_else(|| find_any(0))
            .or_else(|| find_unused(1))
            .or_else(|| find_any(1))
            .or_else(|| find_unused(2))
            .or_else(|| find_unused(3))
            .or_else(|| find_unused(4))
            .or_else(|| find_any(2))
            .or_else(|| find_any(3))
            .or_else(|| find_any(4));

        resolved[pos] = Some(if let Some(idx) = matched {
            if let Some(slot) = used_orig.get_mut(idx) {
                *slot = true;
            }
            (orig_parsed[idx].clone(), true)
        } else {
            parsed.original_index = fallback_index;
            (parsed, false)
        });
    }

    resolved.into_iter().map(|entry| entry.expect("all slots filled")).collect()
}

/// Returns true for framing/view tags cleared when a `FramingSafeSwap` switches to mode 1.
fn is_framing_override_target(normalized: &str) -> bool {
    matches!(
        normalized,
        "upper body"
            | "cowboy shot"
            | "full body"
            | "lower body"
            | "wide shot"
            | "portrait"
            | "close-up"
            | "close up"
            | "headshot"
            | "front view"
            | "straight-on"
            | "straight on"
            | "facing viewer"
            | "from behind"
            | "back view"
            | "rear view"
            | "from side"
            | "side view"
            | "side profile"
            | "360 view"
            | "character turnaround"
            | "multiple views"
    )
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
            TokenCategory::SubjectCount => {
                if matches!(t.normalized.as_str(), "single character only" | "one person alone in frame") {
                    framing_parts.push(fmt(t));
                } else {
                    subject_parts.push(fmt(t));
                }
            }
            TokenCategory::PinnedEmphasis => emphasis_parts.push(fmt(t)),
            TokenCategory::FramingPose => framing_parts.push(fmt(t)),
            TokenCategory::CharacterBody => body_parts.push(fmt(t)),
            // the parser only knows a few setting words: scene.rs's lists catch cliff, sunset, classroom…
            TokenCategory::NaturalLanguageClause => attire_parts.push(fmt(t)),
            TokenCategory::Attire => match crate::scene::slot(t) {
                crate::scene::Slot::Setting | crate::scene::Slot::Light => scene_parts.push(fmt(t)),
                crate::scene::Slot::Camera => framing_parts.push(fmt(t)),
                crate::scene::Slot::Expression | crate::scene::Slot::Pose => body_parts.push(fmt(t)),
                _ => attire_parts.push(fmt(t)),
            },
            TokenCategory::EnvironmentLighting => scene_parts.push(fmt(t)),
            TokenCategory::LoraOrSpecial => scene_parts.push(fmt(t)),
        }
    }

    // Flux reads plain English: "A girl with red hair, wearing a white dress. Upper body, from side.
    // Setting: cliff, sunset." The user's own sentences stay sentences.
    let (sentences, attire_parts): (Vec<String>, Vec<String>) =
        attire_parts.into_iter().partition(|p| p.split_whitespace().count() >= 5 && p.chars().next().is_some_and(char::is_uppercase));
    let subject: Vec<String> = subject_parts.iter().filter_map(|s| flux_subject(s)).collect();
    let mut clauses = Vec::new();
    let mut first = if subject.is_empty() { String::new() } else { capitalize(&subject.join(" and ")) };
    if !body_parts.is_empty() {
        first = if first.is_empty() { capitalize(&body_parts.join(", ")) } else { format!("{first} with {}", body_parts.join(", ")) };
    }
    if !attire_parts.is_empty() {
        first = if first.is_empty() { format!("Wearing {}", attire_parts.join(", ")) } else { format!("{first}, wearing {}", attire_parts.join(", ")) };
    }
    if !emphasis_parts.is_empty() {
        first = if first.is_empty() {
            format!("Focus on {}", emphasis_parts.join(", "))
        } else {
            format!("{first}, with prominent focus on {}", emphasis_parts.join(", "))
        };
    }
    if !first.is_empty() {
        clauses.push(first);
    }
    if !framing_parts.is_empty() {
        clauses.push(capitalize(&framing_parts.join(", ")));
    }
    clauses.extend(sentences.into_iter().map(|s| s.trim().trim_end_matches('.').to_string()));
    if !scene_parts.is_empty() {
        clauses.push(format!("Setting: {}", scene_parts.join(", ")));
    }
    if clauses.is_empty() {
        return String::new();
    }
    format!("{}.", clauses.join(". "))
}

/// A count tag as plain words for Flux ("1girl" → "a girl", "2girls" → "two girls"); `solo` says nothing new.
fn flux_subject(tag: &str) -> Option<String> {
    let t = tag.trim();
    if t.eq_ignore_ascii_case("solo") {
        return None;
    }
    let numbers = ["", "a", "two", "three", "four", "five", "six"];
    for (noun, plural) in [("girl", "girls"), ("boy", "boys"), ("other", "others")] {
        for (n, word) in numbers.iter().enumerate().skip(1) {
            if t == format!("{n}{noun}") || t == format!("{n}{plural}") {
                let noun = if noun == "other" { "person" } else { noun };
                let plural = if plural == "others" { "people" } else { plural };
                return Some(if n == 1 { format!("a {noun}") } else { format!("{word} {plural}") });
            }
        }
    }
    Some(t.to_string())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
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
    let targets = normalized_tag_set(&ch.original);
    let is_add = targets.is_empty();
    let anchor = if is_add {
        800 // new details go after the user's own tags in their category
    } else {
        match tokens.iter().find(|t| targets.contains(&t.normalized)) {
            Some(first) => first.original_index,
            None => return,
        }
    };

    let insert_tokens = |tokens: &mut Vec<PromptToken>, parsed: Vec<PromptToken>, clean_ws: bool| {
        for mut t in parsed {
            if tokens.iter().any(|e| e.normalized == t.normalized) {
                continue;
            }
            if clean_ws {
                normalize_token_inline_whitespace(&mut t);
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
            insert_tokens(tokens, parse_prompt(&ch.suggested_alternative), false);
        }
        1 => {
            if let Some(framing_tokens) = parse_non_empty_tags(&ch.framing_alternative) {
                let new_framing: HashSet<String> =
                    framing_tokens.iter().map(|t| t.normalized.clone()).collect();
                tokens.retain(|t| {
                    new_framing.contains(&t.normalized)
                        || !matches!(
                            t.normalized.as_str(),
                            "upper body" | "portrait" | "close-up" | "close up" | "front view" | "from behind" | "back view"
                        )
                });
                insert_tokens(tokens, framing_tokens, true);
            }
        }
        _ => {}
    }
}

/// Cleans up a model-written prompt: balances brackets, unwraps a prompt wrapped
/// in one big bracket group, fixes bare `tag:1.0` weights, clamps weights to what
/// the checkpoint tolerates and drops duplicates. Prose (Flux) is left as is.
pub fn tidy_ai_prompt(text: &str, checkpoint: CheckpointProfile) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }
    let balanced = balance_brackets(text);
    if !checkpoint.uses_explicit_weights() {
        // Prose checkpoints (Flux): keep colons/times intact without weight unwrapping,
        // just balance brackets and drop exact duplicate comma clauses.
        let mut seen = HashSet::new();
        return balanced
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|s| seen.insert(s.to_lowercase()))
            .collect::<Vec<_>>()
            .join(", ");
    }

    // Flatten groups that contain several tags, e.g. "(a, b, c)" -> a, b, c
    // while preserving the outer group's weight on inner tags that lack their own.
    let mut tokens: Vec<PromptToken> = Vec::new();
    let mut queue = parse_prompt(&balanced);
    queue.reverse();
    while let Some(t) = queue.pop() {
        // never split {a, b|c} variation groups
        if t.text.contains(',') && t.category != TokenCategory::LoraOrSpecial && !t.text.starts_with('{') {
            let trimmed = t.text.trim();
            // If wrapped in brackets, unwrap outer layer to split inner comma items
            let candidate = if (trimmed.starts_with('(') && trimmed.ends_with(')'))
                || (trimmed.starts_with('[') && trimmed.ends_with(']'))
            {
                &trimmed[1..trimmed.len() - 1]
            } else {
                trimmed
            };
            let mut inner = parse_prompt(&balance_brackets(candidate));
            if inner.len() == 1 && inner[0].text == t.text {
                tokens.push(t);
                continue;
            }
            // Propagate explicit group weights like "(red hair, blue eyes:1.2)" to inner tags,
            // but unwrap whole-prompt accidental bracket wrappers "(masterpiece, 1girl, ...)" without multiplying weights.
            let raw_unwrapped = t.raw.trim().trim_end_matches([')', ']']);
            let had_explicit_outer_weight = raw_unwrapped
                .rsplit_once(':')
                .is_some_and(|(left, w)| !left.ends_with('(') && !left.contains('(') && w.trim().parse::<f32>().is_ok())
                || (raw_unwrapped.starts_with('(')
                    && raw_unwrapped.rfind(':').is_some_and(|c| {
                        let before = &raw_unwrapped[1..c];
                        !before.contains('(') && !before.contains(')') && raw_unwrapped[c + 1..].trim().parse::<f32>().is_ok()
                    }));
            if had_explicit_outer_weight && inner.len() <= 4 && (t.weight - 1.0).abs() >= 0.01 {
                for it in &mut inner {
                    if (it.weight - 1.0).abs() < 0.01 {
                        it.weight = t.weight;
                    }
                }
            }
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
        .filter(|t| {
            // with two characters, each side of BREAK may repeat a tag ("smile")
            if t.normalized == "break" {
                seen.clear();
                return true;
            }
            seen.insert(t.normalized.clone())
        })
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
        let escaped = {
            let mut slashes = 0usize;
            let mut k = i;
            while k > 0 && chars[k - 1] == '\\' {
                slashes += 1;
                k -= 1;
            }
            slashes % 2 == 1
        };
        if escaped {
            continue;
        }
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

/// Structured negative prompt output with active protection layer diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NegativeSummary {
    pub text: String,
    pub tokens: usize,
    pub active_layers: Vec<&'static str>,
}

/// Recommended negative prompt: the checkpoint's standard negatives plus tags that
/// keep the camera on the target view, enforce generation stage/aspect/strictness guards,
/// lock out conflicting body proportions, and prevent style/scene bleed, minus anything
/// the positive prompt asks for.
pub fn recommended_negative(config: &ScanConfig, positive: &str) -> String {
    analyze_negative(config, positive).text
}

/// Builds the full recommended negative prompt along with diagnostics on which
/// protective layers are active for this prompt and configuration.
pub fn analyze_negative(config: &ScanConfig, positive: &str) -> NegativeSummary {
    if config.checkpoint.default_negative().is_empty() {
        return NegativeSummary::default();
    }
    let mut layers: Vec<&'static str> = vec!["Checkpoint defaults"];
    let pos_tokens = parse_prompt(positive);
    let view_negs = if config.target_view != crate::profiles::TargetView::None {
        config.target_view.negative_tags()
    } else {
        let eff = crate::analyzer::detect_effective_view(&pos_tokens, crate::profiles::TargetView::None);
        eff.negative_tags()
    };
    if !view_negs.is_empty() {
        layers.push("View lock");
    }
    let prop_tiers = crate::analyzer::detect_all_proportion_tiers(&pos_tokens);
    let mut auto_negs: Vec<String> = Vec::new();
    // Multi-character isolation: gather tags from all other active character and scene prompts
    // so we never put something into the negative prompt that another character in the scene asks for!
    let mut other_char_tags: HashSet<String> = HashSet::new();
    if config.two_char {
        let other_prompts = [&config.char2_prompt_raw, &config.scene_prompt_raw];
        for p in other_prompts {
            for t in parse_prompt(p) {
                other_char_tags.insert(t.normalized);
            }
        }
        for (extra_prompt, extra_lora) in &config.more_chars {
            for t in parse_prompt(extra_prompt) {
                other_char_tags.insert(t.normalized);
            }
            for t in parse_prompt(extra_lora) {
                other_char_tags.insert(t.normalized);
            }
        }
    }
    let conflicts_with_other_char = |tag: &str| -> bool {
        if other_char_tags.is_empty() {
            return false;
        }
        let norm = tag.trim().to_lowercase().replace('_', " ");
        other_char_tags.iter().any(|o| {
            o == &norm
                || crate::parser::has_phrase(o, &norm)
                || crate::parser::has_phrase(&norm, o)
                || crate::analyzer::tags_conflict(o, &norm)
        })
    };
    let push_neg = |list: &mut Vec<String>, tag: &str| {
        if !conflicts_with_other_char(tag) && !list.iter().any(|a| a == tag) {
            list.push(tag.to_string());
        }
    };

    let legs_apart = pos_tokens.iter().any(|t| crate::analyzer::legs_apart(&t.normalized));
    // a slender build stretches a shortstack / petite figure (removed from the positive too)
    if matches!(frame_label(config.frame), "shortstack" | "petite")
        || pos_tokens.iter().any(|t| matches!(t.normalized.as_str(), "shortstack" | "petite"))
    {
        push_neg(&mut auto_negs, "slender build");
    }
    // `breasts apart` with clothes on: keep the outfit on (the tag pulls toward topless / torn tops)
    if pos_tokens.iter().any(|t| t.normalized == "breasts apart")
        && !crate::analyzer::chest_is_bare(&pos_tokens)
        && pos_tokens.iter().any(|t| crate::analyzer::covers_chest(&t.normalized))
    {
        for t in ["topless", "nude", "torn clothes"] {
            push_neg(&mut auto_negs, t);
        }
    }
    let is_shortstack = pos_tokens.iter().any(|t| t.normalized == "shortstack")
        || config.frame == 2;
    for (tier, _) in prop_tiers {
        let rec = crate::analyzer::get_proportion_recommendations(tier);
        for tag in rec.auto_negative_lock {
            let clean = tag.replace('_', " ");
            if is_shortstack && clean == "petite" {
                continue;
            }
            // spread legs separate the thighs: "no thigh gap" would fight the pose
            if clean == "thigh gap" && legs_apart {
                continue;
            }
            if pos_tokens.iter().any(|t| t.normalized == clean) {
                continue;
            }
            push_neg(&mut auto_negs, &clean);
        }
    }

    // Posture collapse lock: prevent seiza/kneeling when pulling up stockings or lifting a knee
    if pos_tokens.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "pulling up stocking" || n == "adjusting legwear" || n == "one knee up"
    }) {
        for tag in ["kneeling", "seiza", "sitting on heels"] {
            push_neg(&mut auto_negs, tag);
        }
    }

    let mut default_neg = config.checkpoint.default_negative().to_string();
    if config.exaggerated_proportions && (is_shortstack || config.bust >= 5 || config.frame > 0) {
        // Strip negative tags that pull exaggerated anatomy/proportions back toward realistic scale
        default_neg = filter_negative_tags(&default_neg, |n| {
            matches!(n, "bad anatomy" | "bad proportions" | "deformed")
        });
    }
    if config.exaggerated_proportions && is_shortstack {
        // Add negative tags that prevent realistic human body scaling
        for tag in ["realistic proportions", "realistic anatomy", "proportionate"] {
            if !pos_tokens.iter().any(|t| t.normalized == tag) {
                push_neg(&mut auto_negs, tag);
            }
        }
    }

    // "Exaggerated" Bust / Frame picks: push the smaller sizes and a tall figure away, so the picked size
    // isn't settled at a "normal big" (merge_negative below drops anything the positive prompt has)
    if config.exaggerated_proportions {
        for (_, tags) in crate::profiles::BUSTS.iter().take(config.bust).skip(1) {
            for t in tags.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                push_neg(&mut auto_negs, t);
            }
        }
        // each figure's opposite silhouette (non-tag words are fine in a negative: they still steer)
        let frame_negs: &[&str] = match frame_label(config.frame) {
            // shortstack / petite are adult bodies: the adult guard keeps the compact height from reading as a child
            "petite" => &["tall", "long legs", "long torso", "broad shoulders", "giantess", "child", "loli", "aged down"],
            "shortstack" => &["tall", "long legs", "long torso", "long neck", "lanky", "broad shoulders", "child", "loli", "aged down"],
            "thick" => &["bowlegs", "knock-knees"],
            "pear" => &["broad shoulders", "wide shoulders", "muscular shoulders", "V-taper", "top-heavy"],
            "hourglass" => &["narrow shoulders", "boxy waist", "wide waist"],
            "muscular" => &["skinny", "frail", "chubby", "plump", "soft body", "flabby", "noodle arms"],
            _ => &[],
        };
        // a flat chest stays flat under a heavy weight
        let bust_negs: &[&str] = if config.bust == 1 {
            &["cleavage", "breast press", "sideboob", "underboob", "small breasts", "medium breasts", "large breasts"]
        } else {
            &[]
        };
        // the errors heavy weights cause, named without naming proportions (shorter for CLIP's 75-token chunks)
        let guard: &[&str] = if config.bust == 0 && config.frame == 0 {
            &[]
        } else if config.checkpoint.uses_clip_chunks() {
            EXAGGERATED_ERROR_GUARD_CLIP
        } else {
            EXAGGERATED_ERROR_GUARD
        };
        // realism pulls the body back toward real proportions (removed from the positive too)
        let realism: &[&str] = if config.bust > 0 || config.frame > 0 { &["realistic"] } else { &[] };
        for t in frame_negs.iter().chain(bust_negs).chain(guard).chain(realism) {
            push_neg(&mut auto_negs, t);
        }
    }
    if !auto_negs.is_empty() {
        layers.push("Proportion locks");
    }

    // Aspect-ratio composition negative guards (Base / HiresFix only)
    let stage_allows_composition_guard = matches!(
        config.stage,
        crate::profiles::GenerationStage::Base | crate::profiles::GenerationStage::HiresFix
    );
    if stage_allows_composition_guard && !config.two_char {
        if config.aspect_ratio.is_widescreen()
            && crate::analyzer::is_single_character_tokens(&pos_tokens)
        {
            layers.push("Aspect guard");
            for tag in WIDESCREEN_SOLO_NEGATIVES {
                push_neg(&mut auto_negs, tag);
            }
            if config.aspect_ratio == crate::profiles::AspectRatio::Ultrawide21x9 {
                for tag in ["mirror image", "two-shot"] {
                    push_neg(&mut auto_negs, tag);
                }
            }
        } else if config.aspect_ratio.is_portrait() {
            layers.push("Aspect guard");
            for tag in ["split screen", "multiple views", "4koma", "comic", "letterboxed"] {
                push_neg(&mut auto_negs, tag);
            }
        }
    }

    // Multi-stage generation negative adjustments
    match config.stage {
        crate::profiles::GenerationStage::HiresFix => {
            layers.push("Stage guard");
            for tag in ["multiple views", "split screen", "cloned", "duplicate"] {
                push_neg(&mut auto_negs, tag);
            }
        }
        crate::profiles::GenerationStage::Detailer => {
            layers.push("Stage guard");
            for tag in [
                "multiple views",
                "split screen",
                "extra people",
                "cloned",
                "full body",
                "wide shot",
                "wide-angle",
                "panoramic",
                "distant",
                "out of focus",
            ] {
                push_neg(&mut auto_negs, tag);
            }
        }
        crate::profiles::GenerationStage::Refiner => {
            layers.push("Stage guard");
            for tag in ["over-sharpened", "artifacts", "noisy", "color banding"] {
                push_neg(&mut auto_negs, tag);
            }
        }
        crate::profiles::GenerationStage::Inpaint => {
            layers.push("Stage guard");
            // Do not suppress cropped/out-of-frame during partial-region inpainting
            default_neg = filter_negative_tags(&default_neg, |l| {
                matches!(
                    l,
                    "cropped"
                        | "out of frame"
                        | "head out of frame"
                        | "feet out of frame"
                        | "cut off"
                        | "partially out of frame"
                )
            });
            for tag in ["seam", "color mismatch", "disjointed"] {
                push_neg(&mut auto_negs, tag);
            }
        }
        crate::profiles::GenerationStage::Base => {}
    }

    // Negative strictness & context-aware scene/pose/style bleed protection
    match config.negative_strictness {
        crate::profiles::NegativeStrictness::Light => {
            // Keep only the first 4 core quality tags from default_neg so modern models aren't over-constrained
            default_neg = expand_negative_tokens(&default_neg)
                .into_iter()
                .take(4)
                .map(|t| t.raw)
                .collect::<Vec<_>>()
                .join(", ");
        }
        crate::profiles::NegativeStrictness::Balanced => {}
        crate::profiles::NegativeStrictness::Strict => {
            layers.push("Strict anatomy & artifact guard");
            let strict_anatomy = if config.exaggerated_proportions && is_shortstack {
                // Keep hand/finger & lighting fixes without forcing realistic body proportions on shortstack
                &[
                    "mutated hands",
                    "extra fingers",
                    "fused fingers",
                    "chromatic aberration",
                    "overexposed",
                    "washed out",
                    "low contrast",
                ][..]
            } else {
                &[
                    "mutated hands",
                    "extra limbs",
                    "fused fingers",
                    "bad proportions",
                    "asymmetrical eyes",
                    "chromatic aberration",
                    "overexposed",
                    "washed out",
                    "low contrast",
                ][..]
            };
            for tag in strict_anatomy {
                push_neg(&mut auto_negs, tag);
            }
            if !config.two_char
                && config.stage != crate::profiles::GenerationStage::Detailer
                && crate::analyzer::is_single_character_tokens(&pos_tokens)
            {
                for tag in ["clone", "duplicate", "extra people", "multiple views"] {
                    push_neg(&mut auto_negs, tag);
                }
            }
            if pos_tokens.iter().any(|t| {
                t.normalized.starts_with("holding ")
                    || matches!(t.normalized.as_str(), "sword" | "katana" | "staff" | "umbrella" | "book" | "cup")
            }) {
                push_neg(&mut auto_negs, "floating object");
            }
        }
    }

    // Context-aware style, scene, and single-character posture bleed protection (Balanced & Strict)
    if config.negative_strictness != crate::profiles::NegativeStrictness::Light {
        let has_any_pos = |tags: &[&str]| {
            pos_tokens.iter().any(|t| {
                let n = t.normalized.as_str();
                tags.iter().any(|tag| n == *tag || crate::parser::has_phrase(n, tag))
            })
        };

        // 1. Style-aware 2D/3D bleed guard
        let wants_2d_tag = has_any_pos(&["anime", "2d", "cel shading", "flat color", "illustration", "manga", "anime coloring"]);
        let wants_3d = has_any_pos(&["3d", "realistic", "photorealistic", "photo", "raw photo", "cgi", "render"]);
        if (wants_2d_tag || config.checkpoint.is_2d_anime_checkpoint()) && !wants_3d {
            layers.push("Style bleed guard");
            for tag in ["3d", "cgi", "render", "photorealistic"] {
                push_neg(&mut auto_negs, tag);
            }
        } else if wants_3d && !wants_2d_tag {
            layers.push("Style bleed guard");
            for tag in ["anime", "2d", "cel shading", "cartoon"] {
                push_neg(&mut auto_negs, tag);
            }
        }

        // 2. Scene / lighting opposite locks (Base & HiresFix only)
        if stage_allows_composition_guard {
            let is_night = has_any_pos(&["night", "midnight", "night sky", "starry sky", "moonlight"]);
            let is_day = has_any_pos(&["day", "daytime", "sunlight", "morning", "afternoon", "noon", "blue sky"]);
            let is_indoors = has_any_pos(&["indoors", "bedroom", "classroom", "office", "library", "kitchen", "bathroom", "hallway"]);
            let is_outdoors = has_any_pos(&["outdoors", "beach", "forest", "mountain", "ocean", "street", "park", "rooftop", "garden"]);
            let is_simple_bg = has_any_pos(&["simple background", "white background", "grey background", "solid background", "gradient background"]);

            let mut added_scene_lock = false;
            if is_night && !is_day {
                for tag in ["daytime", "sunlight"] {
                    push_neg(&mut auto_negs, tag);
                }
                added_scene_lock = true;
            } else if is_day && !is_night && config.negative_strictness == crate::profiles::NegativeStrictness::Strict {
                push_neg(&mut auto_negs, "night");
                added_scene_lock = true;
            }
            if is_indoors && !is_outdoors {
                push_neg(&mut auto_negs, "outdoors");
                added_scene_lock = true;
            } else if is_outdoors && !is_indoors {
                push_neg(&mut auto_negs, "indoors");
                added_scene_lock = true;
            }
            if is_simple_bg && !is_outdoors {
                for tag in ["cluttered background", "busy background"] {
                    push_neg(&mut auto_negs, tag);
                }
                added_scene_lock = true;
            }
            if added_scene_lock {
                layers.push("Scene & lighting locks");
            }
        }

        // 3. Single-character posture lock (Base stage, single-char only)
        if config.stage == crate::profiles::GenerationStage::Base
            && !config.two_char
            && crate::analyzer::is_single_character_tokens(&pos_tokens)
        {
            let is_sitting = has_any_pos(&["sitting", "seated", "sitting on bed", "seiza", "wariza"]);
            let is_standing = has_any_pos(&["standing", "standing upright", "contrapposto"]);
            let is_lying = has_any_pos(&["lying", "lying down", "on back", "on stomach", "on side"]);
            let mut added_pose_lock = false;
            if is_sitting && !is_standing && !is_lying {
                push_neg(&mut auto_negs, "standing");
                added_pose_lock = true;
            } else if is_standing && !is_sitting && !is_lying {
                for tag in ["sitting", "lying"] {
                    push_neg(&mut auto_negs, tag);
                }
                added_pose_lock = true;
            } else if is_lying && !is_standing && !is_sitting {
                push_neg(&mut auto_negs, "standing");
                added_pose_lock = true;
            }
            if added_pose_lock {
                layers.push("Pose locks");
            }
        }
    }

    let mut parts = Vec::new();
    if !default_neg.trim().is_empty() {
        parts.push(default_neg);
    }
    if !view_negs.is_empty() {
        parts.push(view_negs.to_string());
    }
    if !auto_negs.is_empty() {
        parts.push(auto_negs.join(", "));
    }
    let custom_neg = config.custom_negative_raw.trim();
    if !custom_neg.is_empty() {
        layers.push("Custom tags");
        parts.push(custom_neg.to_string());
    }
    let base = parts.join(", ");
    let text = merge_negative(&base, positive);
    let (tokens, _) = estimate_clip_tokens(&text);
    NegativeSummary { text, tokens, active_layers: layers }
}

/// Negative tags that prevent extra people, clones, or split-screen in 16:9 / 21:9 widescreen
/// single-character generations.
pub const WIDESCREEN_SOLO_NEGATIVES: &[&str] = &[
    "2girls", "2boys", "multiple girls", "multiple boys", "multiple views", "split screen",
    "clone", "cloned", "duplicate", "twin", "twins", "doppelganger", "extra people",
    "crowd", "background characters", "two people",
];

/// Standalone widescreen single-character guard for post-processed or live-streamed prompts.
/// Fixes multi-count detection (`1girl, 1boy` is NOT single-character) and Flux trailing punctuation.
pub fn apply_widescreen_single_char_guard(
    positive: &str,
    negative: &str,
    checkpoint: CheckpointProfile,
) -> (String, String) {
    let toks = parse_prompt(positive);
    if !crate::analyzer::is_single_character_tokens(&toks) {
        return (positive.to_string(), negative.to_string());
    }
    let has_norm = |tag: &str| toks.iter().any(|t| t.normalized == tag);

    let mut pos_add: Vec<&str> = Vec::new();
    if !has_norm("solo") && checkpoint != CheckpointProfile::Flux {
        pos_add.push("solo");
    }
    if checkpoint.supports_natural_language() {
        if !has_norm("single character only") {
            pos_add.push("single character only");
        }
        if !has_norm("one person alone in frame") {
            pos_add.push("one person alone in frame");
        }
    }

    let new_pos = if pos_add.is_empty() {
        positive.to_string()
    } else if positive.trim().is_empty() {
        pos_add.join(", ")
    } else if checkpoint == CheckpointProfile::Flux {
        let trimmed = positive.trim().trim_end_matches('.');
        format!("{trimmed}. {}.", pos_add.join(", "))
    } else {
        format!("{}, {}", positive.trim().trim_end_matches(','), pos_add.join(", "))
    };

    if checkpoint.default_negative().is_empty() {
        return (new_pos, negative.to_string());
    }
    let wide = WIDESCREEN_SOLO_NEGATIVES.join(", ");
    let combined_neg = if negative.trim().is_empty() {
        wide
    } else {
        format!("{}, {}", negative.trim().trim_end_matches(','), wide)
    };
    let new_neg = merge_negative(&combined_neg, &new_pos);
    (new_pos, new_neg)
}

/// "Exaggerated" error guard for the negative: the structural errors heavy weights cause, named without naming
/// body proportions (so they don't pull the exaggerated shape back toward normal).
pub const EXAGGERATED_ERROR_GUARD: &[&str] = &[
    "extra limbs", "extra arms", "extra legs", "extra fingers", "fused fingers", "missing fingers", "bad hands",
    "malformed hands", "missing arms", "missing legs", "fused limbs", "fused legs", "disconnected limbs",
    "anatomical nonsense", "double navel", "extra breasts", "fused breasts",
];
/// The hand / limb part of `EXAGGERATED_ERROR_GUARD`, for CLIP checkpoints (a long negative is chunked).
pub const EXAGGERATED_ERROR_GUARD_CLIP: &[&str] = &[
    "extra limbs", "extra arms", "extra legs", "extra fingers", "fused fingers", "missing fingers", "bad hands",
    "malformed hands",
];

/// A short name for a Frame menu index: "petite", "shortstack", "thick", "pear", "hourglass", "muscular"
/// ("" for none or out of range); follows the order of `profiles::FRAMES`.
pub fn frame_label(frame: usize) -> &'static str {
    ["", "petite", "shortstack", "thick", "pear", "hourglass", "muscular"].get(frame).copied().unwrap_or("")
}

/// Unwraps multi-tag weighted negative groups like `(worst quality, low quality:1.4)` into
/// individual weighted tokens while preserving single-tag weights and plain `score_4` spellings.
pub(crate) fn expand_negative_tokens(negative: &str) -> Vec<PromptToken> {
    let mut out = Vec::new();
    for t in parse_prompt(negative) {
        if t.text.contains(',') && !t.text.starts_with('{') && t.category != TokenCategory::LoraOrSpecial {
            let has_weight = (t.weight - 1.0).abs() >= 0.04;
            for mut inner in parse_prompt(&t.text) {
                if has_weight && (inner.weight - 1.0).abs() < 0.04 {
                    inner.weight = t.weight;
                    inner.raw = inner.format_for_output(true);
                }
                out.push(inner);
            }
        } else {
            out.push(t);
        }
    }
    out
}

/// Filters negative tags using `parse_prompt` (safe around grouped weights like `(bad anatomy:1.2)`).
pub fn filter_negative_tags(negative: &str, drop: impl Fn(&str) -> bool) -> String {
    expand_negative_tokens(negative)
        .into_iter()
        .filter(|t| !drop(t.normalized.as_str()))
        .map(|t| t.raw)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Cleans a negative prompt: one tag per entry, no duplicates, no counterproductive
/// phrasing (`no X`, `wrong X`, `without X`), and nothing that the positive prompt
/// contains or implies (a tag in both cancels itself out).
pub fn merge_negative(negative: &str, positive: &str) -> String {
    let wanted: HashSet<String> = parse_prompt(positive).into_iter().map(|t| t.normalized).collect();
    let mut seen = HashSet::new();
    expand_negative_tokens(negative)
        .into_iter()
        .filter_map(|mut t| {
            let l = t.normalized.as_str();
            if l.is_empty() {
                return None;
            }
            if let Some(rest) = t.raw.trim().strip_prefix("unwanted ") {
                t.raw = rest.trim().to_string();
                t.normalized = t.normalized.strip_prefix("unwanted ").unwrap_or(&t.normalized).trim().to_string();
            }
            let norm = t.normalized.as_str();
            if norm != "no humans"
                && (norm.starts_with("no ")
                    || norm.starts_with("wrong ")
                    || norm.starts_with("non-")
                    || norm.starts_with("non ")
                    || norm.starts_with("lack of ")
                    || norm.starts_with("not ")
                    || norm.starts_with("without "))
            {
                return None;
            }
            Some(t)
        })
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
        &["indoor", "indoors", "interior"],
        &["indoors", "room", "bedroom", "classroom", "kitchen", "office", "library", "bathroom", "interior", "house",
          "cafe", "restaurant", "desk", "hallway", "shop", "train interior", "car interior"],
    ),
    (
        &["outdoor", "outdoors", "exterior"],
        &["outdoors", "forest", "beach", "street", "city", "sky", "park", "field", "garden", "mountain", "shrine",
          "ocean", "river", "lake", "rooftop", "snow", "desert"],
    ),
    (&["daytime", "day time", "sunlight", "sunny"], &["day", "daytime", "sunlight", "afternoon", "morning", "noon", "sunny", "blue sky"]),
    (&["night", "nighttime", "dark background", "dark lighting"], &["night", "night sky", "moon", "moonlight", "stars", "starry sky", "dark background", "dark lighting"]),
    (&["standing"], &["standing", "standing upright", "contrapposto", "walking", "running"]),
    (&["sitting", "seated"], &["sitting", "seated", "sitting on bed", "seiza", "wariza"]),
    (&["lying"], &["lying", "lying down", "on back", "on stomach", "on side"]),
    (&["busy background", "cluttered background", "complex background"], &["forest", "city", "street", "crowd", "classroom", "library"]),
    (&["simple background", "white background", "plain background"], &["simple background", "white background", "grey background", "gradient background"]),
];

fn contradicts_positive(negative_tag: &str, positive: &HashSet<String>) -> bool {
    // "people"/"person"/"man"/"woman" would suppress the prompt's own character
    if matches!(negative_tag, "people" | "person" | "human" | "humans" | "woman" | "girl" | "man" | "boy")
        && positive.iter().any(|p| p.ends_with("girl") || p.ends_with("boy") || p.ends_with("girls") || p.ends_with("boys") || p == "solo")
    {
        return true;
    }
    // Multi-character prompts should never have anti-group/anti-multi-person tags in negative
    if matches!(
        negative_tag,
        "multiple girls" | "multiple boys" | "multiple others" | "extra people" | "two people" | "crowd" | "background characters"
    ) {
        let count_sum: usize = positive
            .iter()
            .filter_map(|p| crate::two::count_tag(p).map(|(c, _)| c as usize))
            .sum();
        let has_multi_word = positive.iter().any(|p| {
            matches!(
                p.as_str(),
                "multiple girls" | "multiple boys" | "multiple others" | "crowd" | "group" | "couple" | "twins"
            )
        });
        if count_sum >= 2 || has_multi_word {
            return true;
        }
    }
    // Style / medium / lighting contradiction protection
    if matches!(negative_tag, "3d" | "cgi" | "render" | "realistic" | "photorealistic")
        && positive.iter().any(|p| {
            matches!(
                p.as_str(),
                "3d" | "cgi" | "render" | "realistic" | "photorealistic" | "photo" | "raw photo"
            )
        })
    {
        return true;
    }
    if matches!(negative_tag, "anime" | "2d" | "cel shading" | "cartoon")
        && positive.iter().any(|p| {
            matches!(
                p.as_str(),
                "anime" | "2d" | "cel shading" | "flat color" | "illustration" | "manga" | "anime coloring"
            )
        })
    {
        return true;
    }
    if matches!(negative_tag, "sketch" | "lineart" | "monochrome" | "greyscale" | "rough" | "unfinished")
        && positive.iter().any(|p| {
            matches!(
                p.as_str(),
                "sketch" | "lineart" | "monochrome" | "greyscale" | "grayscale" | "ink"
            )
        })
    {
        return true;
    }
    if matches!(negative_tag, "overexposed" | "lens flare" | "chromatic aberration" | "glow")
        && positive.iter().any(|p| {
            matches!(
                p.as_str(),
                "lens flare" | "chromatic aberration" | "glowing" | "neon" | "neon lights" | "bloom"
            )
        })
    {
        return true;
    }
    // Explicit / bare-skin protection: never suppress requested nudity/exposure
    if matches!(negative_tag, "nsfw" | "explicit")
        && positive.iter().any(|p| {
            matches!(
                p.as_str(),
                "nude" | "naked" | "topless" | "bottomless" | "nipples" | "pussy" | "penis" | "explicit" | "nsfw" | "sex"
            )
        })
    {
        return true;
    }
    if matches!(negative_tag, "chibi" | "bad proportions")
        && positive.iter().any(|p| matches!(p.as_str(), "chibi" | "shortstack"))
    {
        return true;
    }
    // Hair / body build / trait hierarchy protection
    if negative_tag == "long hair"
        && positive.iter().any(|p| matches!(p.as_str(), "very long hair" | "absurdly long hair"))
    {
        return true;
    }
    if negative_tag == "medium hair"
        && positive.iter().any(|p| matches!(p.as_str(), "long hair" | "very long hair" | "absurdly long hair" | "short hair"))
    {
        return true;
    }
    if negative_tag == "short hair"
        && positive.iter().any(|p| p == "very short hair")
    {
        return true;
    }
    if matches!(negative_tag, "breasts" | "chest")
        && positive.iter().any(|p| crate::analyzer::detect_breast_tier(p).is_some())
    {
        return true;
    }
    if matches!(negative_tag, "muscular" | "abs" | "toned")
        && positive.iter().any(|p| matches!(p.as_str(), "muscular" | "muscular female" | "abs" | "six-pack" | "toned"))
    {
        return true;
    }
    if matches!(negative_tag, "wings" | "horns" | "tail")
        && positive.iter().any(|p| p.ends_with(&format!(" {negative_tag}")) || p == negative_tag)
    {
        return true;
    }
    NEGATIVE_CONFLICTS.iter().any(|(marks, fights)| {
        marks.iter().any(|m| crate::parser::has_phrase(negative_tag, m))
            && positive.iter().any(|p| fights.iter().any(|f| p == f || crate::parser::has_phrase(p, f)))
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

/// The prompt syntax a copy is written in (the format dropdown next to Copy, `--export`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// `(tag:1.2)`, as the app writes it.
    ComfyUi,
    /// A1111 / Forge: the same syntax as ComfyUI.
    WebUi,
    /// `{tag}` = ×1.05 and `[tag]` = ÷1.05 per pair; parentheses are plain text.
    NovelAi,
    /// No weights, brackets or LoRA tags; BREAK and `{a|b}` groups stay.
    Plain,
}

impl ExportFormat {
    /// Dropdown order: ComfyUI, WebUI, NovelAI, Plain.
    pub fn from_index(i: i32) -> Self {
        match i {
            1 => Self::WebUi,
            2 => Self::NovelAi,
            3 => Self::Plain,
            _ => Self::ComfyUi,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "comfyui" | "comfy" => Some(Self::ComfyUi),
            "webui" | "a1111" | "forge" => Some(Self::WebUi),
            "novelai" | "nai" => Some(Self::NovelAi),
            "plain" => Some(Self::Plain),
            _ => None,
        }
    }
}

/// A note for the status line when the format can't express part of the prompt: NovelAI reads
/// `{…}` as emphasis, so dynamic-prompt groups `{a|b}` would change meaning there.
pub fn export_warning(text: &str, format: ExportFormat) -> Option<&'static str> {
    (format == ExportFormat::NovelAi && has_dynamic_group(text))
        .then_some("Note: NovelAI reads {…} as emphasis; the {a|b} dynamic groups were left as they are.")
}

fn has_dynamic_group(text: &str) -> bool {
    let mut depth = 0i32;
    for c in text.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = (depth - 1).max(0),
            '|' if depth > 0 => return true,
            _ => {}
        }
    }
    false
}

/// The prompt in another tool's syntax. ComfyUI and WebUI share the app's syntax (unchanged). NovelAI:
/// `(tag:W)` becomes N = round(ln W / ln 1.05) pairs of `{}` (W > 1) or `[]` (W < 1), a bare `(tag)`
/// counts as 1.1, weight 1.0 is bare, and `\(` / `\)` become plain parentheses. Plain: weights and
/// brackets go, `\(` / `\)` are unescaped and `<lora:…>` tags are dropped (they mean nothing outside
/// a LoRA loader); BREAK and `{a|b}` dynamic groups are left alone in every format.
pub fn format_prompt(text: &str, format: ExportFormat) -> String {
    if matches!(format, ExportFormat::ComfyUi | ExportFormat::WebUi) {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let out = convert_span(&chars, format);
    if format == ExportFormat::Plain {
        out.trim().trim_end_matches(',').trim_end().to_string()
    } else {
        out
    }
}

fn convert_span(chars: &[char], format: ExportFormat) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' if matches!(chars.get(i + 1), Some('(') | Some(')')) => {
                out.push(chars[i + 1]);
                i += 2;
            }
            '{' => {
                // a dynamic group (or the user's own braces): copied as it is
                let end = matching(chars, i, '{', '}').unwrap_or(chars.len() - 1);
                out.extend(&chars[i..=end]);
                i = end + 1;
            }
            '<' if format == ExportFormat::Plain && starts_with_ci(&chars[i..], "<lora:") => {
                let end = chars[i..].iter().position(|&c| c == '>').map_or(chars.len(), |e| i + e + 1);
                i = end;
                // the separator after it goes too
                while i < chars.len() && chars[i] == ' ' {
                    i += 1;
                }
                if i < chars.len() && chars[i] == ',' {
                    i += 1;
                    while i < chars.len() && chars[i] == ' ' {
                        i += 1;
                    }
                }
            }
            '(' => match matching(chars, i, '(', ')') {
                Some(end) => {
                    let inner = &chars[i + 1..end];
                    let (content, weight) = split_weight(inner);
                    let content = convert_span(content, format);
                    if format == ExportFormat::NovelAi {
                        let n = (weight.ln() / 1.05f32.ln()).round() as i32;
                        let (open, close) = if n > 0 { ('{', '}') } else { ('[', ']') };
                        let n = n.unsigned_abs() as usize;
                        out.extend(std::iter::repeat(open).take(n));
                        out.push_str(&content);
                        out.extend(std::iter::repeat(close).take(n));
                    } else {
                        out.push_str(&content);
                    }
                    i = end + 1;
                }
                None => {
                    out.push(c);
                    i += 1;
                }
            },
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

fn starts_with_ci(chars: &[char], prefix: &str) -> bool {
    chars.len() >= prefix.len() && chars.iter().zip(prefix.chars()).all(|(a, b)| a.to_ascii_lowercase() == b)
}

/// The index of the bracket closing the one at `start` (escaped brackets don't count).
fn matching(chars: &[char], start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0;
    let mut i = start;
    while i < chars.len() {
        if chars[i] == '\\' {
            i += 2;
            continue;
        }
        if chars[i] == open {
            depth += 1;
        } else if chars[i] == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// `tag:1.2` → (`tag`, 1.2); without a top-level `:number` the A1111 default 1.1.
fn split_weight(inner: &[char]) -> (&[char], f32) {
    let mut depth = 0;
    let mut colon = None;
    let mut i = 0;
    while i < inner.len() {
        match inner[i] {
            '\\' => i += 1,
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => depth -= 1,
            ':' if depth == 0 => colon = Some(i),
            _ => {}
        }
        i += 1;
    }
    if let Some(c) = colon {
        let w: String = inner[c + 1..].iter().collect();
        if let Ok(w) = w.trim().parse::<f32>() {
            if w > 0.0 {
                return (&inner[..c], w);
            }
        }
    }
    (inner, 1.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_written_in_other_formats() {
        use ExportFormat::*;
        let p = "masterpiece, (red hair:1.15), (smile:0.9), (glasses:1.5), (blue eyes:1.0), kay \\(girls und panzer\\), {sitting|standing}";
        assert_eq!(format_prompt(p, ComfyUi), p);
        assert_eq!(format_prompt(p, WebUi), p);
        assert_eq!(
            format_prompt(p, NovelAi),
            "masterpiece, {{{red hair}}}, [[smile]], {{{{{{{{glasses}}}}}}}}, blue eyes, kay (girls und panzer), {sitting|standing}"
        );
        assert!(export_warning(p, NovelAi).is_some());
        assert!(export_warning("(red hair:1.2)", NovelAi).is_none() && export_warning(p, Plain).is_none());
        // plain: no weights, brackets or LoRA tags; BREAK and dynamic groups stay
        assert_eq!(
            format_prompt("<lora:kay:0.8>, 1girl, ((red hair)), (smile:1.2), kay \\(gup\\), BREAK, 1boy, {a|b}, <LORA:x:1>", Plain),
            "1girl, red hair, smile, kay (gup), BREAK, 1boy, {a|b}"
        );
        // a bare (tag) is 1.1: two braces in NovelAI; arranged line breaks survive
        assert_eq!(format_prompt("(smile),\n(night:1.3)", NovelAi), "{{smile}},\n{{{{{night}}}}}");
        assert_eq!(ExportFormat::from_name("NAI"), Some(NovelAi));
        assert_eq!(ExportFormat::from_index(3), Plain);
    }

    #[test]
    fn front_view_negative_detection_and_rear_suppression() {
        use crate::analyzer::ScanConfig;
        use crate::profiles::{CheckpointProfile, TargetView};
        let cfg = ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            fit_suggestions: true,
            exaggerated_proportions: false,
            camera_angle: 0,
            pose: 0,
            bust: 0,
            frame: 0,
            dynamic: false,
            lora_tags_raw: String::new(),
            arranged: false,
            natural: false,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: true,
            scene_prompt_raw: String::new(),
            more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
            ..Default::default()
        };
        let prompt = "masterpiece, 1girl, solo, front view, sitting on bed";
        let neg = recommended_negative(&cfg, prompt);
        assert!(neg.contains("from behind") || neg.contains("back view") || neg.contains("rear view"), "{neg}");
        assert!(neg.contains("over the shoulder"), "{neg}");
    }

    #[test]
    fn shortstack_suppresses_petite_in_recommended_negative() {
        use crate::analyzer::ScanConfig;
        use crate::profiles::{CheckpointProfile, TargetView};
        let cfg = ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            fit_suggestions: true,
            exaggerated_proportions: false,
            camera_angle: 0,
            pose: 0,
            bust: 0,
            frame: 0,
            dynamic: false,
            lora_tags_raw: String::new(),
            arranged: false,
            natural: false,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: true,
            scene_prompt_raw: String::new(),
            more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
            ..Default::default()
        };
        let prompt = "1girl, solo, gigantic breasts, shortstack";
        let neg = recommended_negative(&cfg, prompt);
        assert!(!neg.contains("petite"), "petite must not be in negative when shortstack is present: {neg}");
        assert!(neg.contains("flat chest"), "{neg}");
    }

    #[test]
    fn exaggerated_picks_push_smaller_sizes_into_the_negative() {
        let mut c = ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: crate::profiles::TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            fit_suggestions: true,
            exaggerated_proportions: false,
            camera_angle: 0,
            pose: 0,
            bust: 0,
            frame: 0,
            dynamic: false,
            lora_tags_raw: String::new(),
            arranged: false,
            natural: false,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: true,
            scene_prompt_raw: String::new(),
            more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
            ..Default::default()
        };
        c.checkpoint = CheckpointProfile::Anima;
        c.bust = 6;
        c.frame = 2;
        c.exaggerated_proportions = true;
        let neg = recommended_negative(&c, "1girl, gigantic breasts, shortstack");
        for t in ["small breasts", "medium breasts", "large breasts", "huge breasts", "tall", "long legs"] {
            assert!(neg.contains(t), "{t} missing: {neg}");
        }
        assert!(!neg.contains("gigantic breasts"), "{neg}");
        c.exaggerated_proportions = false;
        assert!(!recommended_negative(&c, "1girl, gigantic breasts, shortstack").contains("long legs"));
    }

    #[test]
    fn shortstack_exaggerated_proportions_modifies_negative_prompt() {
        use crate::analyzer::ScanConfig;
        use crate::profiles::{CheckpointProfile, TargetView};
        let cfg = ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            fit_suggestions: true,
            exaggerated_proportions: true,
            camera_angle: 0,
            pose: 0,
            bust: 0,
            frame: 0,
            dynamic: false,
            lora_tags_raw: String::new(),
            arranged: false,
            natural: false,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: true,
            scene_prompt_raw: String::new(),
            more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
            ..Default::default()
        };
        let prompt = "1girl, solo, gigantic breasts, shortstack";
        let neg = recommended_negative(&cfg, prompt);
        // Stripped: bad anatomy, bad proportions, deformed
        assert!(!neg.contains("bad anatomy"), "bad anatomy must be stripped: {neg}");
        assert!(!neg.contains("bad proportions"), "bad proportions must be stripped: {neg}");
        assert!(!neg.contains("deformed"), "deformed must be stripped: {neg}");

        // Added: realistic proportions, realistic anatomy, proportionate
        assert!(neg.contains("realistic proportions"), "realistic proportions must be added: {neg}");
        assert!(neg.contains("realistic anatomy"), "realistic anatomy must be added: {neg}");
        assert!(neg.contains("proportionate"), "proportionate must be added: {neg}");
    }

    #[test]
    fn robust_negative_system_layers_and_contradiction_guards() {
        use crate::analyzer::ScanConfig;
        use crate::profiles::{CheckpointProfile, GenerationStage, NegativeStrictness};

        let mut cfg = ScanConfig::default();
        cfg.checkpoint = CheckpointProfile::Anima;
        cfg.negative_strictness = NegativeStrictness::Balanced;

        // 1. Night + sitting + indoors -> locks out daytime, sunlight, standing, outdoors, plus 2D->3D bleed
        let sum = analyze_negative(&cfg, "1girl, solo, sitting, classroom, night");
        assert!(sum.text.contains("daytime") && sum.text.contains("sunlight"), "{}", sum.text);
        assert!(sum.text.contains("standing") && sum.text.contains("outdoors"), "{}", sum.text);
        assert!(sum.text.contains("3d") && sum.text.contains("photorealistic"), "{}", sum.text);
        assert!(sum.active_layers.contains(&"Scene & lighting locks"));
        assert!(sum.active_layers.contains(&"Pose locks"));
        assert!(sum.active_layers.contains(&"Style bleed guard"));

        // 2. Photorealistic positive protects realistic/3d and guards against anime/2d bleed
        let photo = analyze_negative(&cfg, "1girl, solo, photorealistic, photo, standing, beach");
        assert!(!photo.text.contains("photorealistic") && !photo.text.contains("3d"), "{}", photo.text);
        assert!(photo.text.contains("anime") && photo.text.contains("cel shading"), "{}", photo.text);
        assert!(photo.text.contains("indoors") && photo.text.contains("sitting"), "{}", photo.text);

        // 3. Strict mode adds clone + prop guards for single-character holding sword
        cfg.negative_strictness = NegativeStrictness::Strict;
        let strict = analyze_negative(&cfg, "1girl, solo, holding sword, standing");
        assert!(strict.text.contains("clone") && strict.text.contains("floating object"), "{}", strict.text);
        assert!(strict.active_layers.contains(&"Strict anatomy & artifact guard"));

        // 4. Detailer stage does not inject posture locks or scene opposites
        cfg.stage = GenerationStage::Detailer;
        let det = analyze_negative(&cfg, "1girl, solo, sitting, classroom, night");
        assert!(!det.text.contains("standing") && !det.text.contains("outdoors"), "{}", det.text);
        assert!(det.text.contains("out of focus") && det.text.contains("wide shot"), "{}", det.text);

        // 5. merge_negative sanitizes "no X"/"wrong X" and respects hair/explicit hierarchy
        let merged = merge_negative("no smile, wrong angle, nsfw, long hair, blurry", "1girl, nude, very long hair");
        assert_eq!(merged, "blurry");

        // 6. NEGATIVE_CONFLICTS uses whole-word matching: "dark skin" doesn't drop "night",
        // "mushroom" doesn't drop "indoors", and grouped weights "(bad anatomy, blurry:1.4)" filter cleanly
        let dark_skin = merge_negative("night, indoors, (worst quality, bad anatomy:1.4)", "1girl, dark skin, mushroom");
        assert!(dark_skin.contains("night") && dark_skin.contains("indoors"), "{dark_skin}");
        let filtered_group = filter_negative_tags("(worst quality, bad anatomy:1.4), blurry", |l| l == "bad anatomy");
        assert_eq!(filtered_group, "(worst quality:1.4), blurry");

        // 7. tidy_ai_prompt preserves group weights "(red hair, blue eyes:1.2)" and cleans Flux duplicates
        assert_eq!(
            tidy_ai_prompt("(red hair, blue eyes:1.2), smile", CheckpointProfile::Anima),
            "(red hair:1.2), (blue eyes:1.2), smile"
        );
        assert_eq!(
            tidy_ai_prompt("A girl at 10:30, A girl at 10:30, (smiling", CheckpointProfile::Flux),
            "A girl at 10:30, smiling"
        );

        // 8. Mode 2 (Restore Original) preserves weight and category
        let raw = "(masterpiece:1.2), 1girl, (ass:1.3), upper body";
        let toks = parse_prompt(raw);
        let mut res = crate::analyzer::analyze_with_rules(&toks, &ScanConfig {
            target_view: crate::profiles::TargetView::UpperBody,
            ..Default::default()
        });
        for ch in &mut res.changes {
            if ch.original.contains("ass") {
                ch.mode = 2;
            }
        }
        let restored = build_condensed_output(raw, &res.processed_tokens, &res.changes, &ScanConfig {
            target_view: crate::profiles::TargetView::UpperBody,
            ..Default::default()
        });
        assert!(restored.final_prompt.contains("(ass:1.3)"), "{}", restored.final_prompt);
    }

    #[test]
    fn mode1_and_mode2_handle_brackets_user_tags_and_multiple_swaps() {
        use crate::analyzer::{analyze_with_rules, ChangeAction, PromptChange, ScanConfig};
        use crate::profiles::TargetView;

        // 1. Mode 1 strips brackets/underscores via orig_parsed and preserves user-authored tags
        let raw = "1girl, upper body, athletic build, [big_ass]";
        let cfg = ScanConfig {
            target_view: TargetView::UpperBody,
            emphasis_weight: 1.25,
            ..Default::default()
        };
        let mut res = analyze_with_rules(&parse_prompt(raw), &cfg);
        // Synthetic swap matching a weighted/bracketed original whose suggested_alternative overlaps user's `athletic build`
        res.processed_tokens.push(PromptToken {
            raw: "hourglass figure".into(),
            text: "hourglass figure".into(),
            normalized: "hourglass figure".into(),
            weight: 1.15,
            had_runaway_weight: false,
            original_index: 3,
            category: TokenCategory::CharacterBody,
            replaced: true,
            added: false,
        });
        let swap = PromptChange {
            id: 900,
            original: "[big_ass]".into(),
            action: ChangeAction::FramingSafeSwap,
            reason: "test".into(),
            suggested_alternative: "athletic build, (hourglass figure:1.15), ,".into(),
            framing_alternative: "cowboy shot, from behind".into(),
            mode: 1,
        };
        let out = build_condensed_output(raw, &res.processed_tokens, &[swap], &cfg);
        assert!(out.final_prompt.contains("(big_ass:1.25)"), "{}", out.final_prompt);
        assert!(!out.final_prompt.contains("[big_ass]"), "{}", out.final_prompt);
        assert!(out.final_prompt.contains("athletic build"), "user tag must be kept: {}", out.final_prompt);
        assert!(!out.final_prompt.contains("hourglass figure"), "replaced alt must be dropped: {}", out.final_prompt);
        assert!(out.final_prompt.contains("cowboy shot") && out.final_prompt.contains("from behind"), "{}", out.final_prompt);
        assert!(!out.final_prompt.contains("upper body"), "{}", out.final_prompt);

        // 2. Multiple FramingSafeSwap cards: mode 0 keeps shared tag, multiple mode 1 cards don't clobber each other
        let raw_multi = "1girl, upper body, ass, thighs";
        let mut res_multi = analyze_with_rules(&parse_prompt(raw_multi), &cfg);
        let mut swaps: Vec<PromptChange> = res_multi
            .changes
            .iter()
            .filter(|c| c.action == ChangeAction::FramingSafeSwap)
            .cloned()
            .collect();
        assert!(swaps.len() >= 2, "expected 2 framing swaps: {swaps:?}");
        swaps[0].mode = 1;
        swaps[1].mode = 1;
        let out_multi = build_condensed_output(raw_multi, &res_multi.processed_tokens, &swaps, &cfg);
        assert!(out_multi.final_prompt.contains("cowboy shot"), "first card framing kept: {}", out_multi.final_prompt);
        assert!(out_multi.final_prompt.contains("full body"), "second card framing kept: {}", out_multi.final_prompt);
        assert!(out_multi.final_prompt.contains("ass") && out_multi.final_prompt.contains("thighs"), "{}", out_multi.final_prompt);

        // One mode 0 + one mode 1 sharing a replacement tag: active mode 0 tag survives
        let base_shared = vec![
            PromptToken {
                raw: "1girl".into(),
                text: "1girl".into(),
                normalized: "1girl".into(),
                weight: 1.0,
                had_runaway_weight: false,
                original_index: 0,
                category: TokenCategory::SubjectCount,
                replaced: false,
                added: false,
            },
            PromptToken {
                raw: "narrow waist".into(),
                text: "narrow waist".into(),
                normalized: "narrow waist".into(),
                weight: 1.15,
                had_runaway_weight: false,
                original_index: 1,
                category: TokenCategory::CharacterBody,
                replaced: true,
                added: false,
            },
        ];
        let shared_changes = vec![
            PromptChange {
                id: 1,
                original: "ass".into(),
                action: ChangeAction::FramingSafeSwap,
                reason: "keep".into(),
                suggested_alternative: "narrow waist".into(),
                framing_alternative: "cowboy shot".into(),
                mode: 0,
            },
            PromptChange {
                id: 2,
                original: "hips".into(),
                action: ChangeAction::FramingSafeSwap,
                reason: "override".into(),
                suggested_alternative: "narrow waist".into(),
                framing_alternative: "full body".into(),
                mode: 1,
            },
        ];
        let out_shared = build_condensed_output("1girl, ass, hips", &base_shared, &shared_changes, &cfg);
        assert!(out_shared.final_prompt.contains("narrow waist"), "mode 0 tag must survive: {}", out_shared.final_prompt);

        // 3. Mode 2 restores multi-tag original (e.g. AngleChoice)
        res_multi.changes.clear();
        let angle_restore = PromptChange {
            id: 901,
            original: "low angle, from below".into(),
            action: ChangeAction::AngleChoice,
            reason: "test".into(),
            suggested_alternative: "high angle".into(),
            framing_alternative: String::new(),
            mode: 2,
        };
        let out_angle = build_condensed_output(
            "1girl, low angle, from below",
            &[PromptToken {
                raw: "high angle".into(),
                text: "high angle".into(),
                normalized: "high angle".into(),
                weight: 1.0,
                had_runaway_weight: false,
                original_index: 940,
                category: TokenCategory::FramingPose,
                replaced: false,
                added: true,
            }],
            &[angle_restore],
            &cfg,
        );
        assert!(out_angle.final_prompt.contains("low angle"), "{}", out_angle.final_prompt);
        assert!(out_angle.final_prompt.contains("from below"), "{}", out_angle.final_prompt);
        assert!(!out_angle.final_prompt.contains("high angle"), "{}", out_angle.final_prompt);
    }

    #[test]
    fn whitespace_framing_alt_sibling_deletion_and_token_resolution_edge_cases() {
        use crate::analyzer::{ChangeAction, PromptChange, ScanConfig};
        use crate::profiles::TargetView;

        let cfg = ScanConfig {
            target_view: TargetView::UpperBody,
            emphasis_weight: 1.25,
            cleanup_enabled: false,
            ..Default::default()
        };

        // 1. Whitespace-only or delimiter-only framing_alternative in mode 1 is a no-op:
        //    existing framing & body alternatives are untouched, and sibling mode 2 won't delete them.
        let base_ws = parse_prompt("1girl, upper body, narrow waist");
        let mut base_ws_replaced = base_ws.clone();
        base_ws_replaced[2].replaced = true;
        for blank_alt in ["   \t ", " , , "] {
            let changes = vec![
                PromptChange {
                    id: 1,
                    original: "ass".into(),
                    action: ChangeAction::FramingSafeSwap,
                    reason: "blank framing".into(),
                    suggested_alternative: "narrow waist".into(),
                    framing_alternative: blank_alt.into(),
                    mode: 1,
                },
                PromptChange {
                    id: 2,
                    original: "hips".into(),
                    action: ChangeAction::FramingSafeSwap,
                    reason: "restore".into(),
                    suggested_alternative: "narrow waist".into(),
                    framing_alternative: "full body".into(),
                    mode: 2,
                },
            ];
            let out = build_condensed_output("1girl, upper body, ass, hips", &base_ws_replaced, &changes, &cfg);
            assert!(out.final_prompt.contains("upper body"), "framing must stay for {blank_alt:?}: {}", out.final_prompt);
            assert!(out.final_prompt.contains("narrow waist"), "inactive mode 1 alt must survive: {}", out.final_prompt);
            assert!(!out.final_prompt.contains("ass"), "{}", out.final_prompt);
        }

        // AI card with whitespace-only framing_alternative in mode 1 also leaves framing intact
        let ai_blank = PromptChange {
            id: 3,
            original: "smile".into(),
            action: ChangeAction::AiRewrite,
            reason: "blank".into(),
            suggested_alternative: "grin".into(),
            framing_alternative: "   ".into(),
            mode: 1,
        };
        let out_ai = build_condensed_output("1girl, upper body, smile", &parse_prompt("1girl, upper body, smile"), &[ai_blank], &cfg);
        assert!(out_ai.final_prompt.contains("upper body"), "{}", out_ai.final_prompt);

        // 2. Mode 2 before Mode 1 must not clobber Mode 1 tokens, and Mode 2 must preserve user-authored tags
        let base_order = vec![
            PromptToken {
                raw: "1girl".into(),
                text: "1girl".into(),
                normalized: "1girl".into(),
                weight: 1.0,
                had_runaway_weight: false,
                original_index: 0,
                category: TokenCategory::SubjectCount,
                replaced: false,
                added: false,
            },
            PromptToken {
                raw: "upper body".into(),
                text: "upper body".into(),
                normalized: "upper body".into(),
                weight: 1.0,
                had_runaway_weight: false,
                original_index: 1,
                category: TokenCategory::FramingPose,
                replaced: false,
                added: false,
            },
            PromptToken {
                raw: "slender".into(),
                text: "slender".into(),
                normalized: "slender".into(),
                weight: 1.0,
                had_runaway_weight: false,
                original_index: 2,
                category: TokenCategory::CharacterBody,
                replaced: false,
                added: false,
            },
            PromptToken {
                raw: "curvy".into(),
                text: "curvy".into(),
                normalized: "curvy".into(),
                weight: 1.15,
                had_runaway_weight: false,
                original_index: 3,
                category: TokenCategory::CharacterBody,
                replaced: true,
                added: false,
            },
        ];
        let order_changes = vec![
            PromptChange {
                id: 10,
                original: "hips".into(),
                action: ChangeAction::FramingSafeSwap,
                reason: "mode 2 before mode 1".into(),
                suggested_alternative: "slender, curvy, cowboy shot".into(),
                framing_alternative: String::new(),
                mode: 2,
            },
            PromptChange {
                id: 11,
                original: "ass".into(),
                action: ChangeAction::FramingSafeSwap,
                reason: "mode 1".into(),
                suggested_alternative: "curvy".into(),
                framing_alternative: "cowboy shot".into(),
                mode: 1,
            },
        ];
        let out_order = build_condensed_output("1girl, upper body, slender, ass, hips", &base_order, &order_changes, &cfg);
        assert!(out_order.final_prompt.contains("slender"), "user tag must survive mode 2: {}", out_order.final_prompt);
        assert!(!out_order.final_prompt.contains("curvy"), "replaced alt must drop: {}", out_order.final_prompt);
        assert!(out_order.final_prompt.contains("cowboy shot"), "mode 1 framing must survive earlier mode 2: {}", out_order.final_prompt);
        assert!(out_order.final_prompt.contains("ass") && out_order.final_prompt.contains("hips"), "{}", out_order.final_prompt);

        // 3. Disambiguation in resolve_original_tokens when multiple tags share the same text
        let raw_dups = "1girl, BIG_ASS, (big ass:1.2), (big ass:1.4)";
        let orig_parsed = parse_prompt(raw_dups);
        let mut used = vec![false; orig_parsed.len()];

        // Exact text match beats normalized-only BIG_ASS, and weight match picks 1.4 over 1.2
        let resolved_14 = resolve_original_tokens("1.4::big ass::", &orig_parsed, &mut used, 500);
        assert_eq!(resolved_14.len(), 1);
        assert!((resolved_14[0].weight - 1.4).abs() < 0.01, "expected 1.4, got {:?}", resolved_14[0]);
        assert_eq!(resolved_14[0].original_index, 3);

        // Weak text-only match first does not hijack a later exact-raw match
        let mut used_order = vec![false; orig_parsed.len()];
        let first_text = resolve_original_tokens("big ass", &orig_parsed, &mut used_order, 500);
        assert_eq!(first_text[0].original_index, 2); // (big ass:1.2)
        let exact_raw_later = resolve_original_tokens("(big ass:1.2)", &orig_parsed, &mut used_order, 500);
        assert_eq!(exact_raw_later[0].original_index, 2);

        // Multiple identical queries resolve to distinct occurrences sharing the same text
        let mut used_multi = vec![false; orig_parsed.len()];
        let pair = resolve_original_tokens("big ass, big ass", &orig_parsed, &mut used_multi, 500);
        assert_eq!(pair.len(), 2);
        assert_eq!(pair[0].original_index, 2);
        assert_eq!(pair[1].original_index, 3);

        // Mixed-specificity query resolves exact raw first so a leading bare tag does not hijack it
        let mut used_mixed = vec![false; orig_parsed.len()];
        let mixed = resolve_original_tokens("big ass, (big ass:1.2)", &orig_parsed, &mut used_mixed, 500);
        assert_eq!(mixed.len(), 2);
        assert_eq!(mixed[0].original_index, 3);
        assert_eq!(mixed[1].original_index, 2);

        // Weak text-only fallback checks unused normalized tiers before reusing an already-claimed token
        let orig_one_text = parse_prompt("1girl, BIG_ASS, (big ass:1.2)");
        let mut used_one_text = vec![false; orig_one_text.len()];
        let pair_diff_case = resolve_original_tokens("big ass, big ass", &orig_one_text, &mut used_one_text, 500);
        assert_eq!(pair_diff_case.len(), 2);
        assert_eq!(pair_diff_case[0].original_index, 2);
        assert_eq!(pair_diff_case[1].original_index, 1);

        // 4. Inline whitespace runs in framing_alternative are normalized cleanly in mode 1 & AI mode 1
        let ws_swap = PromptChange {
            id: 20,
            original: "ass".into(),
            action: ChangeAction::FramingSafeSwap,
            reason: "ws".into(),
            suggested_alternative: "curvy".into(),
            framing_alternative: "  cowboy\t  shot , \n from   behind  ".into(),
            mode: 1,
        };
        let out_ws_swap = build_condensed_output("1girl, upper body, ass", &parse_prompt("1girl, upper body, curvy"), &[ws_swap], &cfg);
        assert!(out_ws_swap.final_prompt.contains("cowboy shot, from behind"), "{}", out_ws_swap.final_prompt);
        assert!(!out_ws_swap.final_prompt.contains('\t') && !out_ws_swap.final_prompt.contains("  "), "{}", out_ws_swap.final_prompt);

        let ws_ai = PromptChange {
            id: 21,
            original: "smile".into(),
            action: ChangeAction::AiRewrite,
            reason: "ws".into(),
            suggested_alternative: "grin".into(),
            framing_alternative: "  cowboy\t  shot  ".into(),
            mode: 1,
        };
        let out_ws_ai = build_condensed_output("1girl, upper body, smile", &parse_prompt("1girl, upper body, smile"), &[ws_ai], &cfg);
        assert!(out_ws_ai.final_prompt.contains("cowboy shot"), "{}", out_ws_ai.final_prompt);
        assert!(!out_ws_ai.final_prompt.contains('\t') && !out_ws_ai.final_prompt.contains("  "), "{}", out_ws_ai.final_prompt);

        // 5. Mode 2 WeightClamped restores exact occurrence even when another tag shares normalized text
        let raw_clamp = "1girl, big ass, (big ass:1.5)";
        let mut base_clamp = parse_prompt(raw_clamp);
        base_clamp[2].weight = 1.25;
        let clamp_change = PromptChange {
            id: 22,
            original: "(big ass:1.5)".into(),
            action: ChangeAction::WeightClamped,
            reason: "clamp".into(),
            suggested_alternative: "(big ass:1.25)".into(),
            framing_alternative: String::new(),
            mode: 2,
        };
        let out_clamp = build_condensed_output(raw_clamp, &base_clamp, &[clamp_change], &cfg);
        assert!(out_clamp.final_prompt.contains("big ass, (big ass:1.5)"), "restored clamped tag missing: {}", out_clamp.final_prompt);
    }
}
