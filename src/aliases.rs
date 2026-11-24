//! A small Danbooru tag-alias table: spellings Danbooru itself redirects to another tag (`yellow hair` →
//! `blonde hair`, `smiling` → `smile`). The tagger's `selected_tags.csv` only lists real tags with their
//! counts, so it can't say which spelling a user's word stands for; this table can. Every entry is a real
//! Danbooru alias: the old spelling isn't in the CSV and the new one is (`aliases_match_the_danbooru_tag_list`
//! checks it when the CSV is on disk). Keep it to clear aliases; anything a user would read as a different
//! meaning stays out (Danbooru aliases `silver hair` to `grey hair`, but people pick silver on purpose).
//!
//! Used by `analyzer::analyze_with_rules` (Clean Bloat on, not Flux): a tag written in an alias spelling
//! becomes the Danbooru tag with a `ChangeAction::AliasTag` card, and through `canonical_synonym_key` two
//! spellings of one tag merge as duplicates.

/// (alias as the parser normalizes it, the Danbooru tag), sorted by alias.
pub const ALIASES: &[(&str, &str)] = &[
    ("animal ear", "animal ears"),
    ("arms crossed", "crossed arms"),
    ("big breasts", "large breasts"),
    ("blond hair", "blonde hair"),
    ("blushing", "blush"),
    ("boobs", "breasts"),
    ("brunette", "brown hair"),
    ("butt", "ass"),
    ("cat ear", "cat ears"),
    ("closeup", "close-up"),
    ("eyeglasses", "glasses"),
    ("eyes closed", "closed eyes"),
    ("gray eyes", "grey eyes"),
    ("gray hair", "grey hair"),
    ("hair band", "hairband"),
    ("hair ribbons", "hair ribbon"),
    ("headphone", "headphones"),
    ("inside", "indoors"),
    ("kitsunemimi", "fox ears"),
    ("knee highs", "kneehighs"),
    ("legs crossed", "crossed legs"),
    ("looking at camera", "looking at viewer"),
    ("mouth open", "open mouth"),
    ("naked", "nude"),
    ("nekomimi", "cat ears"),
    ("outside", "outdoors"),
    ("pony tail", "ponytail"),
    ("red eye", "red eyes"),
    ("redhead", "red hair"),
    ("sea", "ocean"),
    ("smiling", "smile"),
    ("sun glasses", "sunglasses"),
    ("thigh highs", "thighhighs"),
    ("thigh-highs", "thighhighs"),
    ("thighhigh", "thighhighs"),
    ("tights", "pantyhose"),
    ("twin tails", "twintails"),
    ("yellow hair", "blonde hair"),
];

/// The Danbooru tag `norm` (a parser-normalized tag: lowercase, spaces) is an alias of, if any.
pub fn canonical(norm: &str) -> Option<&'static str> {
    ALIASES.binary_search_by(|(a, _)| (*a).cmp(norm)).ok().map(|i| ALIASES[i].1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_unique_and_never_chains() {
        for w in ALIASES.windows(2) {
            assert!(w[0].0 < w[1].0, "not sorted / duplicate: {:?} {:?}", w[0], w[1]);
        }
        for (a, t) in ALIASES {
            assert_ne!(a, t);
            assert!(canonical(t).is_none(), "{t} is itself an alias (chain)");
            assert_eq!(canonical(a), Some(*t));
            assert_eq!(*a, a.to_lowercase().trim());
        }
        assert_eq!(canonical("blonde hair"), None);
        assert_eq!(canonical("hair"), None);
    }

    /// With the tag list on disk: every alias is missing from it and every target is a real tag.
    #[test]
    fn aliases_match_the_danbooru_tag_list() {
        let Some(list) = crate::danbooru::local() else {
            eprintln!("selected_tags.csv not on disk; skipped");
            return;
        };
        for (a, t) in ALIASES {
            assert!(list.get(a).is_none(), "'{a}' is a real tag, not an alias");
            assert!(list.get(t).is_some(), "'{t}' is not a Danbooru tag");
        }
    }

    fn run(prompt: &str, config: &crate::analyzer::ScanConfig) -> (String, Vec<crate::analyzer::PromptChange>) {
        let res = crate::analyzer::analyze_with_rules(&crate::parser::parse_prompt(prompt), config);
        let out = crate::condenser::build_condensed_output(prompt, &res.processed_tokens, &res.changes, config).final_prompt;
        (out, res.changes)
    }

    fn tags(out: &str) -> Vec<String> {
        out.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect()
    }

    #[test]
    fn alias_spellings_become_the_danbooru_tag_with_a_card() {
        use crate::analyzer::{ChangeAction, ScanConfig};
        use crate::profiles::CheckpointProfile;
        let config = ScanConfig { checkpoint: CheckpointProfile::Illustrious, fit_suggestions: false, ..Default::default() };
        let (out, changes) = run("1girl, yellow hair, smiling, (thigh highs:1.2), park", &config);
        let t = tags(&out);
        assert!(t.contains(&"blonde hair".into()) && t.contains(&"smile".into()), "{out}");
        assert!(t.iter().any(|x| x.starts_with("(thighhighs:1.2")), "weight kept: {out}");
        assert!(!out.contains("yellow hair") && !out.contains("smiling"), "{out}");
        let cards: Vec<_> = changes.iter().filter(|c| c.action == ChangeAction::AliasTag).collect();
        assert_eq!(cards.len(), 3, "{cards:?}");
        assert!(cards.iter().any(|c| c.original == "yellow hair" && c.suggested_alternative == "blonde hair"));

        // Keep Original puts the user's spelling back
        let res = crate::analyzer::analyze_with_rules(&crate::parser::parse_prompt("1girl, yellow hair"), &config);
        let mut ch = res.changes.clone();
        ch.iter_mut().find(|c| c.action == ChangeAction::AliasTag).unwrap().mode = 2;
        let kept = crate::condenser::build_condensed_output("1girl, yellow hair", &res.processed_tokens, &ch, &config).final_prompt;
        assert!(tags(&kept).contains(&"yellow hair".into()) && !kept.contains("blonde"), "{kept}");
    }

    #[test]
    fn alias_next_to_its_tag_merges_as_a_duplicate() {
        use crate::analyzer::{ChangeAction, ScanConfig};
        use crate::profiles::CheckpointProfile;
        let config = ScanConfig { checkpoint: CheckpointProfile::Illustrious, fit_suggestions: false, ..Default::default() };
        for prompt in ["1girl, blonde hair, yellow hair", "1girl, yellow hair, blonde hair"] {
            let (out, changes) = run(prompt, &config);
            assert_eq!(tags(&out).iter().filter(|t| *t == "blonde hair").count(), 1, "{prompt} -> {out}");
            assert!(!out.contains("yellow"), "{out}");
            assert!(changes.iter().any(|c| c.action == ChangeAction::MergedDuplicate), "{changes:?}");
        }
    }

    #[test]
    fn aliases_are_left_alone_on_flux_without_cleanup_and_when_pinned() {
        use crate::analyzer::{ChangeAction, ScanConfig};
        use crate::profiles::CheckpointProfile;
        let flux = ScanConfig { checkpoint: CheckpointProfile::Flux, fit_suggestions: false, ..Default::default() };
        assert!(!run("1girl, redhead", &flux).1.iter().any(|c| c.action == ChangeAction::AliasTag));
        let off = ScanConfig { checkpoint: CheckpointProfile::Illustrious, cleanup_enabled: false, fit_suggestions: false, ..Default::default() };
        assert!(tags(&run("1girl, redhead", &off).0).contains(&"redhead".into()));
        let pinned = ScanConfig { checkpoint: CheckpointProfile::Illustrious, pinned_emphasis_raw: "redhead".into(), fit_suggestions: false, ..Default::default() };
        assert!(!run("1girl, redhead", &pinned).1.iter().any(|c| c.action == ChangeAction::AliasTag));
    }
}
