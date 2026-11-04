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

#[cfg(test)]
mod tests {
    use super::*;

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
