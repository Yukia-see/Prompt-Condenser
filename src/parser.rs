use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TokenCategory {
    QualityMeta,
    SubjectCount,
    PinnedEmphasis,
    CharacterBody,
    FramingPose,
    Attire,
    EnvironmentLighting,
    NaturalLanguageClause,
    LoraOrSpecial,
}

impl TokenCategory {
    pub fn label(&self) -> &'static str {
        match self {
            Self::QualityMeta => "Quality / Meta",
            Self::SubjectCount => "Subject",
            Self::PinnedEmphasis => "Pinned Emphasis",
            Self::CharacterBody => "Body / Anatomy",
            Self::FramingPose => "Framing / Pose",
            Self::Attire => "Attire / Details",
            Self::EnvironmentLighting => "Scene / Lighting",
            Self::NaturalLanguageClause => "Natural Language (Anima/Flux)",
            Self::LoraOrSpecial => "LoRA / Special",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptToken {
    /// Original raw text as pasted by the user (e.g. `"((((masterpiece))))"` or `"(ass:1.4)"`)
    pub raw: String,
    /// Inner cleaned text without outer weight brackets (e.g. `"masterpiece"` or `"ass"`)
    pub text: String,
    /// Lowercase normalized form for matching
    pub normalized: String,
    /// Effective CLIP attention weight (1.0 = normal)
    pub weight: f32,
    /// Whether the user had runaway nested parentheses `(((...)))` or extreme weight
    pub had_runaway_weight: bool,
    /// Original index in the prompt (so we can preserve exact order when Cleanup Mode is OFF)
    pub original_index: usize,
    /// Detected semantic category
    pub category: TokenCategory,
    /// True when this token was put in place of something the user wrote
    /// (framing-safe swap, view-conflict fix, etc.). Shown highlighted in the output.
    #[serde(default)]
    pub replaced: bool,
    /// True when an accepted AI card added this tag (not in the user's prompt at all).
    #[serde(default)]
    pub added: bool,
}

impl PromptToken {
    pub fn format_for_output(&self, use_explicit_weights: bool) -> String {
        if self.category == TokenCategory::LoraOrSpecial || self.normalized == "break" {
            return self.text.clone();
        }
        if !use_explicit_weights || (self.weight - 1.0).abs() < 0.04 {
            self.text.clone()
        } else {
            format!("({}:{:.2})", self.text, self.weight)
                .replace(".00)", ")")
                .replace("0)", ")")
        }
    }
}

/// Splits a ComfyUI prompt into top-level segments while respecting parentheses `(...)`, `<...>`, and `BREAK`.
/// NovelAI V4 numeric emphasis: `1.5::red hair, blue eyes::` gives every tag inside the weight
/// (`(red hair:1.5), (blue eyes:1.5)`); a group with no closing `::` runs to the end of the prompt.
/// Text without a `<number>::` opener is returned unchanged.
pub fn convert_novelai_numeric(input: &str) -> String {
    if !input.contains("::") {
        return input.to_string();
    }
    let mut out = String::new();
    let mut rest = input;
    while let Some(pos) = rest.find("::") {
        // the weight is the number right before "::"
        let before = &rest[..pos];
        let num_start = before
            .char_indices()
            .rev()
            .take_while(|(_, c)| c.is_ascii_digit() || *c == '.' || *c == '-')
            .last()
            .map(|(i, _)| i);
        let weight = num_start.and_then(|i| before[i..].parse::<f32>().ok().map(|w| (i, w)));
        let Some((i, w)) = weight.filter(|(i, _)| before[..*i].trim_end().is_empty() || before[..*i].trim_end().ends_with(',') || before[..*i].ends_with(char::is_whitespace)) else {
            out.push_str(&rest[..pos + 2]);
            rest = &rest[pos + 2..];
            continue;
        };
        out.push_str(&before[..i]);
        let body_and_more = &rest[pos + 2..];
        let (body, after) = match body_and_more.find("::") {
            Some(end) => (&body_and_more[..end], &body_and_more[end + 2..]),
            None => (body_and_more, ""),
        };
        let w = (w * 100.0).round() / 100.0;
        let tags: Vec<String> = body
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| if (w - 1.0).abs() < f32::EPSILON { t.to_string() } else { format!("({t}:{w})") })
            .collect();
        out.push_str(&tags.join(", "));
        rest = after;
    }
    out.push_str(rest);
    out
}

pub fn parse_prompt(input: &str) -> Vec<PromptToken> {
    let mut raw_segments = Vec::new();
    let mut current = String::new();
    let mut paren_depth: i32 = 0;
    let mut angle_depth: i32 = 0;

    // Normalize standalone BREAK keywords so they become clean segments
    let normalized_input = convert_novelai_numeric(&input.replace("\r\n", "\n"));

    for ch in normalized_input.chars() {
        match ch {
            // {a|b} is Dynamic Prompts variation syntax: keep it as one segment
            '(' | '[' | '{' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                paren_depth = (paren_depth - 1).max(0);
                current.push(ch);
            }
            '<' => {
                angle_depth += 1;
                current.push(ch);
                // If `<lora:...>` was preceded by text without comma, keep going until `>`
            }
            '>' => {
                angle_depth = (angle_depth - 1).max(0);
                current.push(ch);
            }
            ',' | '\n' if paren_depth == 0 && angle_depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    raw_segments.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        raw_segments.push(tail);
    }

    let mut tokens = Vec::new();
    let mut idx = 0;

    for seg in raw_segments {
        // Extract inline <lora:...> blocks if glued to tags
        let (lora_tags, remainder) = extract_inline_loras(&seg);
        for lora in lora_tags {
            tokens.push(PromptToken {
                raw: lora.clone(),
                text: lora.clone(),
                normalized: lora.to_lowercase(),
                weight: 1.0,
                had_runaway_weight: false,
                replaced: false,
                added: false,
                original_index: idx,
                category: TokenCategory::LoraOrSpecial,
            });
            idx += 1;
        }

        let rem = remainder.trim();
        if rem.is_empty() {
            continue;
        }

        let (inner_text, weight, runaway) = unwrap_weight_syntax(rem);
        let clean_text = inner_text.trim().to_string();
        if clean_text.is_empty() {
            continue;
        }

        let normalized = clean_text
            .to_lowercase()
            .replace('_', " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        let category = categorize_token(&clean_text, &normalized);

        tokens.push(PromptToken {
            raw: rem.to_string(),
            text: clean_text,
            normalized,
            weight,
            had_runaway_weight: runaway,
            replaced: false,
            added: false,
            original_index: idx,
            category,
        });
        idx += 1;
    }

    tokens
}

fn extract_inline_loras(segment: &str) -> (Vec<String>, String) {
    let mut loras = Vec::new();
    let mut remainder = String::new();
    let mut chars = segment.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '<' {
            let mut tag = String::from("<");
            for nc in chars.by_ref() {
                tag.push(nc);
                if nc == '>' {
                    break;
                }
            }
            if tag.to_lowercase().starts_with("<lora:") {
                loras.push(tag);
            } else {
                remainder.push_str(&tag);
            }
        } else {
            remainder.push(c);
        }
    }

    (loras, remainder)
}

/// Unwraps ComfyUI / A1111 weight syntax:
/// - `((((masterpiece))))` -> `("masterpiece", 1.464, true)`
/// - `(cybernetic spine:1.3)` -> `("cybernetic spine", 1.3, false)`
/// - `[[blurry]]` -> `("blurry", 0.81, false)`
fn unwrap_weight_syntax(raw: &str) -> (String, f32, bool) {
    let mut s = raw.trim();
    let mut paren_layers = 0i32;
    let mut bracket_layers = 0i32;
    let mut brace_layers = 0i32;

    loop {
        if s.len() >= 2 && s.starts_with('(') && s.ends_with(')') && is_outer_pair_matching(s, '(', ')') {
            s = s[1..s.len() - 1].trim();
            paren_layers += 1;
        } else if s.len() >= 2 && s.starts_with('[') && s.ends_with(']') && is_outer_pair_matching(s, '[', ']') {
            s = s[1..s.len() - 1].trim();
            bracket_layers += 1;
        } else if s.len() >= 2 && s.starts_with('{') && s.ends_with('}') && !s.contains('|') && is_outer_pair_matching(s, '{', '}') {
            s = s[1..s.len() - 1].trim();
            brace_layers += 1;
        } else {
            break;
        }
    }

    // Check for explicit `:weight` at the end of `s`
    if let Some(colon_pos) = s.rfind(':') {
        let candidate_num = s[colon_pos + 1..].trim();
        if let Ok(explicit_w) = candidate_num.parse::<f32>() {
            let text_part = s[..colon_pos].trim().to_string();
            if !text_part.is_empty() {
                let extra_layers = (paren_layers - 1).max(0);
                let final_w = explicit_w * 1.1_f32.powi(extra_layers) * 1.05_f32.powi(brace_layers);
                let runaway = paren_layers >= 3 || brace_layers >= 6 || final_w > 1.38;
                return (text_part, (final_w * 100.0).round() / 100.0, runaway);
            }
        }
    }

    let computed_w = if paren_layers > 0 {
        1.1_f32.powi(paren_layers)
    } else if brace_layers > 0 {
        1.05_f32.powi(brace_layers)
    } else if bracket_layers > 0 {
        0.9_f32.powi(bracket_layers)
    } else {
        1.0
    };

    let runaway = paren_layers >= 3 || brace_layers >= 6 || computed_w > 1.35;
    (s.to_string(), (computed_w * 100.0).round() / 100.0, runaway)
}

fn is_outer_pair_matching(s: &str, open: char, close: char) -> bool {
    let mut depth = 0i32;
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 && i < chars.len() - 1 {
                return false;
            }
        }
    }
    depth == 0
}

/// Does `text` contain `phrase` as whole words? ("night" is in "night sky" and "at night",
/// not in "knight"; "close-up" matches "close up".)
pub fn has_phrase(text: &str, phrase: &str) -> bool {
    let words = |s: &str| s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect::<Vec<_>>();
    let (hay, needle) = (words(text), words(phrase));
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle.as_slice())
}

pub fn categorize_token(raw_text: &str, normalized: &str) -> TokenCategory {
    if normalized == "break" || raw_text.starts_with('<') {
        return TokenCategory::LoraOrSpecial;
    }

    // Natural language detection (for Anima / Flux): > 6 words or contains verbs/articles
    let word_count = normalized.split_whitespace().count();
    if word_count >= 6 {
        return TokenCategory::NaturalLanguageClause;
    }

    let quality_keywords = [
        "masterpiece",
        "best quality",
        "absurdres",
        "highres",
        "score 9",
        "score 8 up",
        "score 7 up",
        "score 6 up",
        "source anime",
        "rating safe",
        "rating questionable",
        "rating explicit",
        "newest",
        "ultra detailed",
        "extremely detailed",
        "hyperdetailed",
        "8k",
        "4k",
        "uhd",
        "hdr",
        "intricate",
        "intricate details",
        "highly detailed",
        "unreal engine",
        "octane render",
    ];
    if quality_keywords.iter().any(|k| normalized == *k || normalized.starts_with("score ")) {
        return TokenCategory::QualityMeta;
    }

    let subject_keywords = [
        "1girl", "2girls", "1boy", "2boys", "solo", "multiple girls", "1other", "female", "male",
        "woman", "man", "girl", "boy",
    ];
    if subject_keywords.contains(&normalized) {
        return TokenCategory::SubjectCount;
    }

    let framing_keywords = [
        "upper body",
        "cowboy shot",
        "full body",
        "portrait",
        "close-up",
        "close up",
        "extreme close-up",
        "front view",
        "from behind",
        "back view",
        "from side",
        "side view",
        "side profile",
        "360 view",
        "character turnaround",
        "multiple views",
        "reference sheet",
        "looking at viewer",
        "looking back",
        "looking away",
        "standing",
        "sitting",
        "kneeling",
        "lying",
        "arched back",
        "over shoulder",
        "dutch angle",
        "foreshortening",
        "from above",
        "from below",
        "low angle",
        "high angle",
        "wide shot",
        "cropped",
    ];
    if framing_keywords.iter().any(|k| has_phrase(normalized, k)) {
        return TokenCategory::FramingPose;
    }

    let body_keywords = [
        "ass",
        "butt",
        "breasts",
        "cleavage",
        "sideboob",
        "underboob",
        "thighs",
        "thick thighs",
        "hips",
        "wide hips",
        "waist",
        "narrow waist",
        "hourglass figure",
        "curvy",
        "voluptuous",
        "abs",
        "muscular",
        "toned",
        "back",
        "spine",
        "shoulder",
        "collarbone",
        "legs",
        "feet",
        "hands",
        "eyes",
        "hair",
        "face",
        "ears",
        "tail",
        "wings",
        "horns",
        "physique",
        "silhouette",
    ];
    // whole words only: "ass" isn't in "glasses" or "grass", "back" isn't "background"
    if body_keywords.iter().any(|k| has_phrase(normalized, k)) {
        return TokenCategory::CharacterBody;
    }

    let env_keywords = [
        "background",
        "simple background",
        "white background",
        "blurry background",
        "depth of field",
        "bokeh",
        "lighting",
        "cinematic lighting",
        "dramatic lighting",
        "sunlight",
        "night",
        "indoors",
        "outdoors",
        "sky",
        "clouds",
        "forest",
        "city",
        "bedroom",
        "street",
        "shadows",
        "rim lighting",
    ];
    // whole words: "night" isn't in "knight" or "nightgown", "street" isn't "streetwear"
    if env_keywords.iter().any(|k| has_phrase(normalized, k)) {
        return TokenCategory::EnvironmentLighting;
    }

    TokenCategory::Attire
}

/// Estimates CLIP token count and 75-token chunk count for a prompt string.
pub fn estimate_clip_tokens(prompt: &str) -> (usize, usize) {
    if prompt.trim().is_empty() {
        return (0, 0);
    }
    let words = prompt
        .split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')' || c == ':')
        .filter(|s| !s.is_empty())
        .count();
    let punctuation = prompt.chars().filter(|&c| c == ',' || c == '(' || c == ')').count();
    let estimated_tokens = words + (punctuation / 2);
    let chunks = ((estimated_tokens.max(1) - 1) / 75) + 1;
    (estimated_tokens, chunks)
}

/// Sanitizes scanner/raw tags and injects structural audit flags before the LLM pass.
#[derive(Debug, Clone)]
pub struct AntigravityTagFilter {
    pub wide_shots: Vec<&'static str>,
    pub close_shots: Vec<&'static str>,
    pub tops: Vec<&'static str>,
    pub bottoms: Vec<&'static str>,
    pub static_poses: Vec<&'static str>,
    pub limb_anchors: Vec<&'static str>,
    pub archetype_overrides: Vec<&'static str>,
    pub uniforms: Vec<&'static str>,
}

impl Default for AntigravityTagFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AntigravityTagFilter {
    pub fn new() -> Self {
        Self {
            wide_shots: vec!["full body", "cowboy shot", "wide shot"],
            close_shots: vec!["close-up", "portrait", "upper body", "face focus"],
            tops: vec![
                "bra_top",
                "bra top",
                "shirt",
                "jacket",
                "tank top",
                "camisole",
                "bikini top",
                "blue bikini",
                "wearing a blue bikini",
                "spaghetti strap",
                "clothes writing",
                "text on clothes",
                "print shirt",
                "tube top",
                "crop top",
                "halterneck",
                "bandeau",
                "sweater",
                "hoodie",
            ],
            bottoms: vec![
                "pants",
                "skirt",
                "shorts",
                "bikini bottoms",
                "panties",
                "underwear",
                "dress",
                "jeans",
                "leggings",
            ],
            static_poses: vec!["standing", "sitting", "kneeling"],
            limb_anchors: vec![
                "hand",
                "hands",
                "arm",
                "arms",
                "holding",
                "reaching",
                "pulling",
                "adjusting",
                "crossed",
                "lap",
                "hips",
            ],
            archetype_overrides: vec!["bunny girl", "cat girl", "fox girl"],
            uniforms: vec![
                "maid",
                "maid outfit",
                "plaid dress",
                "school uniform",
                "serafuku",
                "suit",
            ],
        }
    }

    /// Sanitizes scanner tags and injects structural audit flags for the LLM pass.
    pub fn process_tags(&self, raw_tag_string: &str) -> String {
        if raw_tag_string.trim().is_empty() {
            return String::new();
        }
        // 1. Normalize and strip inline weights/parentheses
        let raw_list: Vec<String> = raw_tag_string
            .split(',')
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        let mut tags: Vec<String> = Vec::new();
        for tag in raw_list {
            let clean = if tag.starts_with('<') && tag.ends_with('>') {
                tag
            } else {
                tag.replace('(', "")
                    .replace(')', "")
                    .split(':')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            };
            if !clean.is_empty() && !tags.contains(&clean) {
                tags.push(clean);
            }
        }

        // 2. Implicit Full-Body Inference (Footwear/Legwear Trigger)
        let footwear_triggers = [
            "thighhighs",
            "stocking",
            "stockings",
            "kneehighs",
            "mary janes",
            "boots",
            "heels",
            "barefoot",
            "bare foot",
            "sandals",
        ];
        if tags
            .iter()
            .any(|t| footwear_triggers.iter().any(|fw| t.contains(fw)))
        {
            let has_framing = tags.iter().any(|t| {
                self.wide_shots.contains(&t.as_str()) || self.close_shots.contains(&t.as_str())
            });
            if !has_framing {
                tags.insert(0, "full body".to_string());
            }
        }

        // 3. Camera Paradox Resolution (Wide overrides Close)
        let has_wide = tags.iter().any(|t| self.wide_shots.contains(&t.as_str()));
        if has_wide {
            tags.retain(|t| !self.close_shots.contains(&t.as_str()));
        }
        let has_close = tags.iter().any(|t| self.close_shots.contains(&t.as_str()));

        // 4. Archetype vs. Uniform Collision Purge
        if tags
            .iter()
            .any(|t| self.uniforms.iter().any(|u| t.contains(u)))
        {
            tags.retain(|t| !self.archetype_overrides.contains(&t.as_str()));
        }

        // 5. Bottomless Hazard Flagging
        let has_static = tags
            .iter()
            .any(|t| self.static_poses.contains(&t.as_str()) || t.starts_with("sitting "));
        let has_top = tags
            .iter()
            .any(|t| self.tops.iter().any(|top| t.contains(top)));
        let has_bottom = tags
            .iter()
            .any(|t| self.bottoms.iter().any(|bot| t.contains(bot)));
        if (has_wide || (has_static && !has_close)) && has_top && !has_bottom {
            tags.push("MISSING_BOTTOMS_FLAG".to_string());
        }

        // 6. Unrouted Limb Flagging
        let has_limb_anchor = tags
            .iter()
            .any(|t| self.limb_anchors.iter().any(|limb| t.contains(limb)));
        if has_static && !has_limb_anchor {
            tags.push("UNANCHORED_LIMBS_FLAG".to_string());
        }

        tags.join(", ")
    }

    /// Resolves internal flags according to Directive 3 and strips all internal `*_FLAG` tokens.
    pub fn resolve_and_strip_flags(&self, prompt: &str) -> String {
        let mut tokens: Vec<String> = prompt
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        let has_missing_bottoms = tokens.iter().any(|t| t == "MISSING_BOTTOMS_FLAG");

        tokens.retain(|t| {
            let l = t.to_lowercase();
            !l.ends_with("_flag") && !l.ends_with(" flag") && !l.contains("flag")
        });

        if has_missing_bottoms {
            let has_bot = tokens.iter().any(|t| {
                let l = t.to_lowercase();
                self.bottoms.iter().any(|b| l.contains(b))
            });
            if !has_bot {
                let is_blue = tokens.iter().any(|t| {
                    let l = t.to_lowercase();
                    l.contains("blue") || l.contains("tube top")
                });
                if is_blue {
                    tokens.push("light blue bikini bottoms".to_string());
                } else {
                    tokens.push("denim shorts".to_string());
                }
            }
            let has_feet = tokens.iter().any(|t| {
                let l = t.to_lowercase();
                l.contains("barefoot")
                    || l.contains("bare foot")
                    || l.contains("shoes")
                    || l.contains("sandals")
                    || l.contains("boots")
            });
            if !has_feet {
                tokens.push("bare foot".to_string());
            }
        }

        tokens.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_and_scene_words_match_whole_words() {
        for tag in ["knight", "nightgown", "streetwear", "flying", "electricity", "broom"] {
            let c = categorize_token(tag, tag);
            assert!(c != TokenCategory::EnvironmentLighting && c != TokenCategory::FramingPose, "{tag} -> {c:?}");
        }
        for (tag, want) in [("night", TokenCategory::EnvironmentLighting), ("night sky", TokenCategory::EnvironmentLighting),
            ("sitting on chair", TokenCategory::FramingPose), ("close-up", TokenCategory::FramingPose),
            ("cinematic lighting", TokenCategory::EnvironmentLighting)]
        {
            assert_eq!(categorize_token(tag, tag), want, "{tag}");
        }
        assert!(has_phrase("close-up shot", "close up"));
    }

    #[test]
    fn body_words_match_whole_words() {
        for tag in ["glasses", "grass", "classroom", "background", "backlighting", "detailed", "tears"] {
            assert_ne!(categorize_token(tag, tag), TokenCategory::CharacterBody, "{tag}");
        }
        for tag in ["ass", "large breasts", "thick thighs", "long hair", "fox tail", "toned back"] {
            assert_eq!(categorize_token(tag, tag), TokenCategory::CharacterBody, "{tag}");
        }
    }

    #[test]
    fn parses_weights_loras_and_nesting() {
        let toks = parse_prompt("((((masterpiece)))), (cybernetic spine:1.3), [[blurry]], <lora:foo:0.8> 1girl, upper_body");
        assert_eq!(toks[0].text, "masterpiece");
        assert!(toks[0].had_runaway_weight);
        assert!((toks[0].weight - 1.46).abs() < 0.01);

        assert_eq!(toks[1].text, "cybernetic spine");
        assert!((toks[1].weight - 1.3).abs() < 0.001);

        assert_eq!(toks[2].text, "blurry");
        assert!((toks[2].weight - 0.81).abs() < 0.01);

        assert_eq!(toks[3].category, TokenCategory::LoraOrSpecial);
        assert_eq!(toks[4].normalized, "1girl");
        assert_eq!(toks[5].normalized, "upper body");
        assert_eq!(toks[5].category, TokenCategory::FramingPose);

        // NovelAI V4 numeric groups
        assert_eq!(convert_novelai_numeric("1girl, 1.5::red hair, blue eyes::, smile"), "1girl, (red hair:1.5), (blue eyes:1.5), smile");
        assert_eq!(convert_novelai_numeric("-1::hat::, solo"), "(hat:-1), solo");
        assert_eq!(convert_novelai_numeric("1.2::open run"), "(open run:1.2)");
        assert_eq!(convert_novelai_numeric("artist::name, smile"), "artist::name, smile");
        let nai = parse_prompt("1girl, 1.5::red hair::");
        assert!(nai.iter().any(|t| t.normalized == "red hair" && (t.weight - 1.5).abs() < 0.01), "{nai:?}");
        // NovelAI curly braces: {tag} is 1.05x, {{tag}} is 1.1025x
        let n_toks = parse_prompt("{{masterpiece}}, {blue eyes}, {red|blue} hair");
        assert_eq!(n_toks[0].text, "masterpiece");
        assert!((n_toks[0].weight - 1.10).abs() < 0.01);
        assert_eq!(n_toks[1].text, "blue eyes");
        assert!((n_toks[1].weight - 1.05).abs() < 0.01);
        // Dynamic prompt variation {red|blue} is preserved without weight strip:
        assert!(n_toks[2].text.contains("{red|blue}"));
    }

    #[test]
    fn commas_inside_parentheses_do_not_split() {
        let toks = parse_prompt("(red dress, frills:1.2), smile");
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].text, "red dress, frills");
    }

    #[test]
    fn clip_chunk_estimate() {
        assert_eq!(estimate_clip_tokens(""), (0, 0));
        let long = vec!["tag"; 100].join(", ");
        assert_eq!(estimate_clip_tokens(&long).1, 2);
    }

    #[test]
    fn antigravity_tag_filter_rules() {
        let filter = AntigravityTagFilter::new();
        // Footwear triggers full body, wide overrides close, uniform purges bunny girl,
        // wide + top without bottom flags MISSING_BOTTOMS_FLAG, static without limb flags UNANCHORED_LIMBS_FLAG
        let out = filter.process_tags("(1girl:1.2), bunny girl, maid, white shirt, thighhighs, standing");
        assert_eq!(
            out,
            "full body, 1girl, maid, white shirt, thighhighs, standing, MISSING_BOTTOMS_FLAG, UNANCHORED_LIMBS_FLAG"
        );
        // Wide overrides close
        let out2 = filter.process_tags("full body, upper body, portrait, 1girl, shirt, skirt, sitting, hands in lap");
        assert_eq!(out2, "full body, 1girl, shirt, skirt, sitting, hands in lap");
    }
}
