//! Token-budget mode (`ScanConfig.token_budget`, Options "Budget", `--budget N`): when the rule-based After
//! prompt is estimated above N tokens (`parser::estimate_clip_tokens`, the LoRA box included), the least
//! important tags are dropped one at a time until it fits, each with a `ChangeAction::TokenBudget` card
//! (Keep Original puts the tag back, over budget).
//!
//! Never dropped: quality tags, LoRAs / special tokens, count and subject tags (`1girl`, `solo`), Keep &
//! Emphasize pins, tags a dropdown or a rule put in (`added` / `replaced`), tags the user weighted above
//! 1.0, BREAK and Danbooru character names.
//!
//! Drop order: tags the user weighted below 1.0 first (they said those matter less); then every tag that
//! isn't the first of its kind (`scene::slot`), so one setting, one light and one style tag survive as long
//! as possible; within that by kind (style, light, setting, sentences, unknown words, expression, pose,
//! camera, outfit, body, eyes, hair, traits), then the one written last. Then the same order for the lone
//! tags of each kind.

use crate::analyzer::{ChangeAction, PromptChange};
use crate::parser::{estimate_clip_tokens, PromptToken, TokenCategory};
use crate::scene::{self, Slot};

/// The budget choices offered in the window (Options "Budget" ComboBox; index 0 = off).
pub const BUDGET_CHOICES: [usize; 5] = [0, 75, 150, 225, 300];

/// Estimated tokens of the tags as the After view joins them, plus the LoRA box.
pub fn estimate(tokens: &[PromptToken], use_weights: bool, lora_raw: &str) -> usize {
    let mut text: Vec<String> = tokens.iter().map(|t| t.format_for_output(use_weights)).collect();
    let lora = lora_raw.trim();
    if !lora.is_empty() {
        text.push(lora.to_string());
    }
    estimate_clip_tokens(&text.join(", ")).0
}

fn is_protected(t: &PromptToken) -> bool {
    matches!(
        t.category,
        TokenCategory::QualityMeta | TokenCategory::SubjectCount | TokenCategory::PinnedEmphasis | TokenCategory::LoraOrSpecial
    ) || t.added
        || t.replaced
        || t.weight > 1.04
        || t.normalized == "break"
        || t.normalized.contains("<lora:")
        || matches!(scene::slot(t), Slot::Quality | Slot::Lora | Slot::Subject)
        || crate::danbooru::is_character(&t.normalized)
}

/// Lower = dropped sooner.
fn kind_rank(slot: Slot) -> u8 {
    match slot {
        Slot::Style => 0,
        Slot::Light => 1,
        Slot::Setting => 2,
        Slot::Clause => 3,
        Slot::Other => 4,
        Slot::Expression => 5,
        Slot::Pose => 6,
        Slot::Camera => 7,
        Slot::Outfit => 8,
        Slot::Body => 9,
        Slot::Eyes => 10,
        Slot::Hair => 11,
        Slot::Traits => 12,
        Slot::Quality | Slot::Lora | Slot::Subject => 255,
    }
}

/// Drops tags from `tokens` until `estimate` fits `budget` (0 = off). Returns one card per dropped tag,
/// ids from `next_id`. If even the protected tags are over budget, it drops what it may and stops; the
/// last card says so.
pub fn fit_to_budget(
    tokens: &mut Vec<PromptToken>,
    budget: usize,
    use_weights: bool,
    lora_raw: &str,
    next_id: &mut usize,
) -> Vec<PromptChange> {
    let mut cards = Vec::new();
    if budget == 0 || estimate(tokens, use_weights, lora_raw) <= budget {
        return cards;
    }

    // the first-written tag of each kind is "lone": it goes only after every extra of every kind
    let mut seen: Vec<Slot> = Vec::new();
    let mut order: Vec<(bool, bool, u8, std::cmp::Reverse<usize>, String)> = Vec::new();
    let mut by_index: Vec<&PromptToken> = tokens.iter().collect();
    by_index.sort_by_key(|t| t.original_index);
    for t in by_index {
        let slot = scene::slot(t);
        let lone = !seen.contains(&slot);
        if lone {
            seen.push(slot);
        }
        if is_protected(t) {
            continue;
        }
        let full_weight = t.weight >= 0.96;
        order.push((full_weight, lone, kind_rank(slot), std::cmp::Reverse(t.original_index), t.normalized.clone()));
    }
    order.sort();

    for (_, _, _, _, norm) in order {
        let now = estimate(tokens, use_weights, lora_raw);
        if now <= budget {
            break;
        }
        let Some(pos) = tokens.iter().position(|t| t.normalized == norm) else { continue };
        let tok = tokens.remove(pos);
        let after = estimate(tokens, use_weights, lora_raw);
        cards.push(PromptChange {
            id: *next_id,
            original: tok.raw.trim().to_string(),
            action: ChangeAction::TokenBudget,
            reason: format!(
                "Token budget {budget}: '{}' was dropped as one of the least important tags (~{now} → ~{after} tokens). Keep Original puts it back.",
                tok.text
            ),
            suggested_alternative: String::new(),
            framing_alternative: String::new(),
            mode: 0,
        });
        *next_id += 1;
    }
    let end = estimate(tokens, use_weights, lora_raw);
    if end > budget {
        if let Some(last) = cards.last_mut() {
            last.reason.push_str(&format!(
                " Still ~{end} tokens: the rest are protected (quality, count, LoRA, pinned, weighted, dropdown picks)."
            ));
        }
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{analyze_with_rules, ScanConfig};
    use crate::condenser::build_condensed_output;
    use crate::parser::parse_prompt;
    use crate::profiles::CheckpointProfile;

    const LONG: &str = "masterpiece, best quality, 1girl, solo, long hair, red hair, blue eyes, smile, \
        school uniform, pleated skirt, white thighhighs, standing, outdoors, city street, night, neon lights, \
        moonlight, rain, wet, film grain, depth of field, chromatic aberration, bokeh, light particles";

    fn cfg(budget: usize) -> ScanConfig {
        ScanConfig { checkpoint: CheckpointProfile::Illustrious, token_budget: budget, fit_suggestions: false, ..Default::default() }
    }

    fn run(prompt: &str, config: &ScanConfig) -> (String, Vec<PromptChange>) {
        let res = analyze_with_rules(&parse_prompt(prompt), config);
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, config);
        (out.final_prompt, res.changes)
    }

    fn budget_cards(changes: &[PromptChange]) -> Vec<&PromptChange> {
        changes.iter().filter(|c| c.action == ChangeAction::TokenBudget).collect()
    }

    #[test]
    fn budget_off_or_already_fitting_changes_nothing() {
        let (off, c_off) = run(LONG, &cfg(0));
        assert!(budget_cards(&c_off).is_empty());
        let (big, c_big) = run(LONG, &cfg(300));
        assert!(budget_cards(&c_big).is_empty());
        assert_eq!(off, big);
    }

    #[test]
    fn budget_drops_least_important_tags_with_a_card_each() {
        let (full, _) = run(LONG, &cfg(0));
        let budget = estimate_clip_tokens(&full).0 - 12;
        let (out, changes) = run(LONG, &cfg(budget));
        let dropped = budget_cards(&changes);
        assert!(!dropped.is_empty());
        assert!(estimate_clip_tokens(&out).0 <= budget, "{out}");
        let out_tags: Vec<&str> = out.split(',').map(str::trim).collect();
        for c in &dropped {
            assert!(!out_tags.contains(&c.original.as_str()), "{} still in {out}", c.original);
            assert_eq!(c.mode, 0);
        }
        for keep in ["masterpiece", "1girl", "solo", "red hair", "blue eyes", "school uniform"] {
            assert!(out_tags.contains(&keep), "{keep} missing from {out}");
        }
        // extras go before the first tag of a kind: the first setting tag survives
        assert!(out_tags.contains(&"outdoors"), "{out}");
        assert!(dropped.iter().any(|c| c.original == "light particles" || c.original == "bokeh"), "{dropped:?}");
    }

    #[test]
    fn keep_original_puts_a_dropped_tag_back() {
        let (full, _) = run(LONG, &cfg(0));
        let config = cfg(estimate_clip_tokens(&full).0 - 6);
        let res = analyze_with_rules(&parse_prompt(LONG), &config);
        let mut changes = res.changes.clone();
        let card = changes.iter_mut().find(|c| c.action == ChangeAction::TokenBudget).expect("a budget card");
        let tag = card.original.clone();
        card.mode = 2;
        let out = build_condensed_output(LONG, &res.processed_tokens, &changes, &config).final_prompt;
        assert!(out.split(',').any(|t| t.trim() == tag), "{tag} not restored in {out}");
    }

    #[test]
    fn protected_tags_are_never_dropped() {
        let prompt = "score_9, 1girl, (red hair:1.3), smile, park, sunset, watercolor \\(medium\\)";
        let config = ScanConfig { checkpoint: CheckpointProfile::PonyV6, token_budget: 3, fit_suggestions: false, ..Default::default() };
        let (out, changes) = run(prompt, &config);
        assert!(out.contains("1girl") && out.contains("red hair") && out.contains("score_9"), "{out}");
        let last = *budget_cards(&changes).last().expect("cards");
        assert!(last.reason.contains("protected"), "{}", last.reason);
    }

    #[test]
    fn lighter_weighted_tags_go_first() {
        let prompt = "1girl, red hair, park, (sunset:0.7), night, rain";
        let (full, _) = run(prompt, &cfg(0));
        let (_, changes) = run(prompt, &cfg(estimate_clip_tokens(&full).0 - 1));
        let first = budget_cards(&changes)[0];
        assert!(first.original.contains("sunset"), "{first:?}");
    }

    #[test]
    fn lora_box_counts_toward_the_budget() {
        let prompt = "1girl, red hair, smile, park, sunset, night, rain";
        let mut config = cfg(0);
        let (full, _) = run(prompt, &config);
        config.token_budget = estimate_clip_tokens(&full).0;
        assert!(budget_cards(&run(prompt, &config).1).is_empty());
        config.lora_tags_raw = "<lora:x:0.8>, trigger word, other trigger".into();
        assert!(!budget_cards(&run(prompt, &config).1).is_empty());
    }
}
