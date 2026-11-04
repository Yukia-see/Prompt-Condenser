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
pub fn parse_prompt(input: &str) -> Vec<PromptToken> {
    let mut raw_segments = Vec::new();
    let mut current = String::new();
    let mut paren_depth: i32 = 0;
    let mut angle_depth: i32 = 0;

    // Normalize standalone BREAK keywords so they become clean segments
    let normalized_input = input.replace("\r\n", "\n");

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

    loop {
        if s.len() >= 2 && s.starts_with('(') && s.ends_with(')') && is_outer_pair_matching(s, '(', ')') {
            s = s[1..s.len() - 1].trim();
            paren_layers += 1;
        } else if s.len() >= 2 && s.starts_with('[') && s.ends_with(']') && is_outer_pair_matching(s, '[', ']') {
            s = s[1..s.len() - 1].trim();
            bracket_layers += 1;
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
                let final_w = explicit_w * 1.1_f32.powi(extra_layers);
                let runaway = paren_layers >= 3 || final_w > 1.38;
                return (text_part, (final_w * 100.0).round() / 100.0, runaway);
            }
        }
    }

    let computed_w = if paren_layers > 0 {
        1.1_f32.powi(paren_layers)
    } else if bracket_layers > 0 {
        0.9_f32.powi(bracket_layers)
    } else {
        1.0
    };

    let runaway = paren_layers >= 3 || computed_w > 1.35;
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
    if framing_keywords.iter().any(|k| normalized.contains(k)) {
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
    if body_keywords.iter().any(|k| normalized.contains(k)) {
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
    if env_keywords.iter().any(|k| normalized.contains(k)) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
