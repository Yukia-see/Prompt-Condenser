//! Dynamic Prompts ({a|b|c}) variations without AI: the prompt's pose, camera angle,
//! expression, lighting and time-of-day tags become variation groups, so every image
//! in a batch differs while the character, outfit and setting stay the same.

use crate::parser::parse_prompt;

/// Interchangeable options per kind of tag. A tag found in one of these lists becomes
/// `{tag|a|b|c}` using the next options of the same list.
const VARIATION_GROUPS: &[(&str, &[&str])] = &[
    ("camera angle", &["from above", "from below", "dutch angle", "three-quarter view", "from side", "straight-on", "high angle", "low angle"]),
    ("pose", &["standing", "sitting", "leaning forward", "leaning against wall", "arms up", "hand on hip", "crossed arms", "looking back", "dynamic pose", "kneeling", "squatting", "walking"]),
    ("expression", &["smile", "grin", "smirk", "blush", "serious", "surprised", "laughing", "light smile", "pout", "expressionless"]),
    ("lighting", &["soft lighting", "rim lighting", "dramatic lighting", "backlighting", "cinematic lighting", "volumetric lighting", "neon lighting", "golden hour", "moonlight", "sunlight"]),
    ("time of day", &["day", "sunset", "night", "dawn", "evening", "dusk", "morning"]),
];

/// How many options each group gets (the original plus this many - 1).
const OPTIONS_PER_GROUP: usize = 4;

/// Returns the prompt with up to one variation group per kind, and the kinds varied.
pub fn make_dynamic(prompt: &str) -> (String, Vec<&'static str>) {
    let mut used: Vec<&'static str> = Vec::new();
    let parts: Vec<String> = parse_prompt(prompt)
        .into_iter()
        .map(|t| {
            if t.raw.starts_with('{') || t.raw.starts_with('<') {
                return t.raw;
            }
            for (kind, options) in VARIATION_GROUPS {
                if used.contains(kind) {
                    continue;
                }
                let Some(pos) = options.iter().position(|o| *o == t.normalized) else { continue };
                used.push(kind);
                let mut opts = vec![t.text.clone()];
                opts.extend(
                    (1..options.len())
                        .map(|i| options[(pos + i) % options.len()].to_string())
                        .take(OPTIONS_PER_GROUP - 1),
                );
                return format!("{{{}}}", opts.join("|"));
            }
            t.raw
        })
        .collect();
    (parts.join(", "), used)
}

/// A fresh seed for `roll` (std's per-process random hash keys; no RNG dependency).
pub fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish()
}

/// xorshift64: enough for picking options.
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        XorShift(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % n as u64) as usize
    }
}

/// Resolves every `{a|b|c}` group to one option, picked uniformly with `seed` (nested groups from the
/// inside out), the way the Dynamic Prompts node does for one image. Everything else is kept as written:
/// `<lora:…>`, `(tag:1.2)`, escaped `\{` / `\}` / `\|`, a `|` outside a group, an unclosed `{`.
pub fn roll(prompt: &str, seed: u64) -> String {
    let chars: Vec<char> = prompt.chars().collect();
    let mut rng = XorShift::new(seed);
    let mut i = 0;
    let mut out = String::new();
    while i < chars.len() {
        let (s, _) = roll_seq(&chars, &mut i, &mut rng);
        out.push_str(&s);
        if i < chars.len() {
            // a `|` or `}` at the top level is plain text
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Reads text up to the end or this level's `|` / `}`, which is returned unread.
fn roll_seq(chars: &[char], i: &mut usize, rng: &mut XorShift) -> (String, Option<char>) {
    let mut out = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        match c {
            '\\' => {
                out.push(c);
                if let Some(&n) = chars.get(*i + 1) {
                    out.push(n);
                }
                *i += 2;
            }
            '|' | '}' => return (out, Some(c)),
            '{' => {
                let start = *i;
                *i += 1;
                let mut options = Vec::new();
                let closed = loop {
                    let (s, end) = roll_seq(chars, i, rng);
                    options.push(s);
                    match end {
                        Some('|') => *i += 1,
                        Some(_) => {
                            *i += 1;
                            break true;
                        }
                        None => break false,
                    }
                };
                if closed {
                    out.push_str(&options[rng.below(options.len())]);
                } else {
                    // unclosed: keep the rest as written
                    out.extend(&chars[start..]);
                    *i = chars.len();
                }
            }
            _ => {
                out.push(c);
                *i += 1;
            }
        }
    }
    (out, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_picks_one_option_per_group() {
        let p = "1girl, {smile|grin|smirk}, {standing|sitting}";
        // the same seed gives the same pick
        assert_eq!(roll(p, 7), roll(p, 7));
        let out = roll(p, 7);
        assert!(!out.contains('{') && !out.contains('|'), "{out}");
        assert!(out.starts_with("1girl, "));
        // every option comes up across seeds
        let picks: std::collections::HashSet<String> = (1..200).map(|s| roll("{a|b|c}", s)).collect();
        assert_eq!(picks.len(), 3, "{picks:?}");
        assert_ne!(random_seed(), random_seed());
    }

    #[test]
    fn roll_resolves_nested_groups_from_the_inside_out() {
        let seen: std::collections::HashSet<String> = (1..400).map(|s| roll("{red {hair|eyes}|blue}", s)).collect();
        let want: std::collections::HashSet<String> = ["red hair", "red eyes", "blue"].iter().map(|s| s.to_string()).collect();
        assert_eq!(seen, want);
    }

    #[test]
    fn roll_leaves_loras_weights_and_escapes_alone() {
        let p = "<lora:style_v2:0.8:0.6>, (masterpiece:1.2), \\{literal\\|x\\}, kay \\(girls und panzer\\), {smile|smile}, a|b, {open";
        assert_eq!(
            roll(p, 3),
            "<lora:style_v2:0.8:0.6>, (masterpiece:1.2), \\{literal\\|x\\}, kay \\(girls und panzer\\), smile, a|b, {open"
        );
        assert_eq!(roll("no groups here", 1), "no groups here");
        assert!(roll("{a|}", 1).len() <= 1);
    }

    #[test]
    fn varies_pose_expression_and_light_but_not_identity() {
        let (out, kinds) = make_dynamic("1girl, solo, silver hair, kimono, sitting, smile, moonlight, shrine");
        assert_eq!(out, "1girl, solo, silver hair, kimono, {sitting|leaning forward|leaning against wall|arms up}, {smile|grin|smirk|blush}, {moonlight|sunlight|soft lighting|rim lighting}, shrine");
        assert_eq!(kinds, vec!["pose", "expression", "lighting"]);
    }

    #[test]
    fn leaves_existing_groups_and_loras_alone() {
        let (out, kinds) = make_dynamic("<lora:x:0.8>, 1girl, {red|blue} hair");
        assert_eq!(out, "<lora:x:0.8>, 1girl, {red|blue} hair");
        assert!(kinds.is_empty());
    }
}
