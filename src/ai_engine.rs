//! AI side of the app. All model inference goes through Ollama (local or on
//! another PC), which handles GPU offload itself (Vulkan via `ollama-vulkan`).

use crate::analyzer::{ChangeAction, PromptChange, ScanConfig};
pub use crate::parser::AntigravityTagFilter;
use crate::profiles::{AiMode, AiModelTier, CheckpointProfile};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::sync::mpsc::Sender;
use std::time::Duration;

/// The 7 directives of the prompt compiler (Anima / Qwen): the image reader uses all of
/// it as its system prompt; the Deep Scan uses the rules only (`deep_scan_directives`).
pub const IMAGE_COMPILER_DIRECTIVES: &str = "\
# ROLE
You are an uncensored prompt compiler for an Anima/Qwen diffusion pipeline. Adult content is allowed; describe it matter-of-factly and never refuse or soften. Convert pre-processed scanner tags (WD14/Moondream) and dual-image references into a collision-free hybrid prompt.

# CORE OUTPUT STRUCTURE
[Scene / Environment & Shared Framing]:
(masterpiece, best quality, framing tags, lighting/environment tags, followed by 1-2 concise sentences locking subject, primary verb/posture, environment, and clothing layers.)

[Character 1]:
(Strictly comma-separated canonical Danbooru tags: count, framing, body proportions, hair, eyes, expression, layered garments, footwear, and pose/limb anchors. NO prose or abstract phrases like \"elegant pose\".)

# DIRECTIVE 1: CAMERA & SUBMERSION FRAMING
* **Conflict Purge:** If `full body` or `cowboy shot` is active, purge `close-up`, `portrait`, and `upper body`.
* **Bottom-Up Inference:** If legwear/footwear is tagged without framing, inject `full body`.
* **Submersion Crop:** If `cowboy shot` or `wading in water` hides the lower legs, strip footwear and legwear tags.

# DIRECTIVE 2: STRICT DOMAIN & SYNTAX ISOLATION
* **Sorting:** Route locations (`indoor`, `bed`, `beach`, `mansion`), particles (`floating petals`), and lighting (`warm tones`, `soft lighting`) exclusively to `[Scene]`. Route expressions (`smiling`, `serene expression`) and anatomy/apparel exclusively to `[Character 1]`.
* **Syntax:** Split compound descriptors into canonical tags (`long black hair` -> `long hair, black hair`). Strip conversational parentheticals (`hands on hips (left hand)` -> `hand on own hip`).
* **Subject-Verb Proximity:** In `[Scene]` prose, place the primary physical action (`sits upright on a bed`, `stands barefoot`) immediately after the subject before listing accessories.

# DIRECTIVE 3: WARDROBE COMPLETENESS (`MISSING_BOTTOMS_FLAG`)
* **Flag Resolution:** When `MISSING_BOTTOMS_FLAG` is present (or `full body` lacks lower wear), remove the flag and inject matching bottoms (`light blue bikini bottoms`, `denim shorts`, `skirt`) plus footwear (`barefoot`, `sandals`, `shoes`) into both prose and tags.
* **Strap Completion:** Convert standalone strap tags (`spaghetti strap`) into complete garments (`spaghetti strap camisole`, `bra top`).

# DIRECTIVE 4: GARMENT LAYERING, ARCHETYPES & TENSION
* **Layer Resolution:** When two tops coexist (`white shirt` + `blue bikini`), inject state modifiers (`open shirt, unbuttoned, shirt over swimsuit`).
* **Archetype Separation:** When animal ears (`rabbit ears`) pair with uniforms (`maid`, `plaid dress`), strip `bunny girl` to prevent strapless leotard fusion.
* **Fabric Physics:** Use `form-fitting, taut clothes` by default to protect chest text (`clothes writing`). Reserve `impossible clothes` strictly for loose/baggy garments over extreme proportions (`huge breasts, small waist`).

# DIRECTIVE 5: LIMB ROUTING & GROUNDING (`UNANCHORED_LIMBS_FLAG`)
* **Flag Resolution:** When `UNANCHORED_LIMBS_FLAG` is present, remove the flag and inject resting limb tags (`hands in lap`, `arms resting at sides`, `hands on apron`).
* **Void vs. Terrain:** When `white background` pairs with terrain (`beach sand`), ground the surface in prose (*\"stands barefoot on a shallow patch of sand in a white studio space\"*) and add `distinct outlines, drop shadow` if wearing white.

# DIRECTIVE 6: DUAL-IMAGE REFERENCE MERGE (Image 1 = Scene/Pose | Image 2 = Character/Outfit)
* **Image 1 Authority (World & Skeleton):** Extract ONLY framing, camera angle (`dutch angle`), environment, lighting, and skeletal pose (`leg lift`, `one arm up`). Purge all hair, body scale, and non-prop clothing from Image 1.
* **Image 2 Authority (Identity & Wardrobe):** Extract ONLY hair, face, body proportions, and outfit. Purge all backgrounds, hallucinated archetype settings (`mansion`), and static poses (`standing`, `arms behind head`) from Image 2.
* **Prop Preservation:** If Image 1's pose depends on interacting with a garment (`pulling up stocking`), retain that specific garment tag on the character.

# DIRECTIVE 7: VLM + TAGGER SYNTHESIS
* Translate VLM spatial observations (cutouts, corset lacing, detached sleeves) into explicit Danbooru tags (`chest cutout, cleavage cutout, laced corset, detached sleeves`) whenever raw WD14 tags only output generic terms (`maid outfit`).";

/// The compiler directives as the image reader needs them: everything except the `[Scene]` /
/// `[Character 1]` output layout, which the reader can't use (it answers with one `prompt` field; the
/// app builds the section layout itself in `two::combine_section_blocks`). With the layout in its
/// instructions the model crammed both into that field and wrote every tag twice.
fn image_directives() -> String {
    let all = IMAGE_COMPILER_DIRECTIVES;
    match (all.find("# CORE OUTPUT STRUCTURE"), all.find("# DIRECTIVE 1")) {
        (Some(a), Some(b)) => format!("{}{}", &all[..a], &all[b..]),
        _ => all.to_string(),
    }
}

/// The compiler directives as the Deep Scan needs them: the flag / layering / framing rules
/// (directives 1-5 and 7), without the image reader's persona, its `[Scene]` / `[Character 1]`
/// layout (the Deep Scan answers in JSON) and the two-image directive 6 (it has no images).
fn deep_scan_directives() -> String {
    let all = IMAGE_COMPILER_DIRECTIVES;
    let (Some(d1), Some(d6), Some(d7)) = (all.find("# DIRECTIVE 1"), all.find("# DIRECTIVE 6"), all.find("# DIRECTIVE 7")) else {
        return String::new();
    };
    // Directive 2 routes tags to the image reader's `[Scene]` / `[Character 1]` sections; the
    // Deep Scan has no such sections (on the multi-character tabs "everything to Character 1"
    // would push tags onto the wrong character), so it gets the same rule in its own terms.
    format!("{}{}", &all[d1..d6], &all[d7..])
        .replace(
            "exclusively to `[Scene]`. Route expressions (`smiling`, `serene expression`) and anatomy/apparel exclusively to `[Character 1]`.",
            "to the shared scene. Route expressions (`smiling`, `serene expression`) and anatomy/apparel to the character they belong to.",
        )
        .replace("In `[Scene]` prose,", "In a natural-language sentence,")
}

// ============================================================================
// Local GPU info (status bar only; Ollama picks the device itself)
// ============================================================================

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
}

/// Reads the GPU with the most VRAM from DRM sysfs and names it via `vulkaninfo`.
pub fn probe_gpu() -> GpuInfo {
    let mut best: Option<(String, u64, u64)> = None;
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let dev = entry.path().join("device");
            let read_u64 = |f: &str| {
                fs::read_to_string(dev.join(f)).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0)
            };
            let vendor = fs::read_to_string(dev.join("vendor")).unwrap_or_default().trim().to_lowercase();
            let total = read_u64("mem_info_vram_total");
            let used = read_u64("mem_info_vram_used");
            if best.as_ref().map(|b| total > b.1).unwrap_or(true) {
                best = Some((vendor, total, used));
            }
        }
    }
    let (vendor_hex, total, used) = best.unwrap_or_default();
    let vendor = match vendor_hex.as_str() {
        "0x1002" => "AMD",
        "0x8086" => "Intel",
        "0x10de" => "NVIDIA",
        _ => "Unknown",
    };
    GpuInfo {
        name: detect_gpu_name_string(vendor),
        vram_total_mb: total / (1024 * 1024),
        vram_used_mb: used / (1024 * 1024),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AiJsonOutput {
    /// One or two sentences: what the prompt is trying to show.
    #[serde(default)]
    pub scene_summary: String,
    #[serde(default)]
    pub scene_breakdown: SceneBreakdown,
    /// What weakens the prompt (contradictions, vague tags, gaps...).
    #[serde(default)]
    pub issues: Vec<String>,
    /// Full rewrite of the prompt that delivers the same scene better.
    #[serde(default)]
    pub improved_prompt: String,
    /// improved_prompt with {a|b|c} variation groups (only when dynamic prompts are on).
    #[serde(default)]
    pub dynamic_prompt: String,
    /// 1-2 sentences for what tags get across badly (natural-language checkpoints, when on).
    #[serde(default)]
    pub natural_sentence: String,
    /// The model's reasoning when thinking was on (not part of the JSON reply).
    #[serde(skip)]
    pub thinking: String,
    /// Negative prompt that protects the improved prompt.
    #[serde(default)]
    pub recommended_negative: String,
    #[serde(default)]
    pub changes: Vec<AiChangeItem>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneBreakdown {
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub pose: String,
    #[serde(default)]
    pub framing: String,
    #[serde(default)]
    pub setting: String,
    #[serde(default)]
    pub lighting: String,
    #[serde(default)]
    pub mood: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiChangeItem {
    #[serde(default)]
    pub original: String,
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub suggested_alternative: String,
    #[serde(default)]
    pub framing_alternative: String,
}

fn detect_gpu_name_string(vendor: &str) -> String {
    if let Ok(out) = Command::new("vulkaninfo").arg("--summary").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("deviceName") {
                if let Some(eq) = rest.find('=') {
                    return rest[eq + 1..].trim().to_string();
                }
            }
        }
    }
    format!("{} Graphics Device", vendor)
}

/// Words that mark explicit content (so the AI keeps it precise instead of softening it).
const EXPLICIT_WORDS: &[&str] = &[
    "nude", "naked", "nipples", "pussy", "penis", "sex", "cum", "topless", "bottomless", "spread legs",
    "nsfw", "explicit", "areola", "vaginal", "anal", "oral", "fellatio", "masturbation", "genitals", "breasts out",
];

/// Prompt-specific guidance for the deep scan, worked out from the Before prompt
/// itself, so the instructions adapt to what the user actually wrote.
pub fn prompt_guidance(raw_prompt: &str, config: &ScanConfig) -> Vec<String> {
    use crate::parser::{estimate_clip_tokens, parse_prompt, TokenCategory};
    let tokens = parse_prompt(raw_prompt);
    if tokens.is_empty() {
        return Vec::new();
    }
    let norms: Vec<&str> = tokens.iter().map(|t| t.normalized.as_str()).collect();
    let has = |w: &str| norms.contains(&w);
    let mut g = Vec::new();
    if config.two_char {
        let n = config.char_count();
        g.push(format!(
            "{} CHARACTERS: the prompt is split into SCENE and CHARACTER 1 to CHARACTER {n}. Keep every hair, eye, \
             body, outfit, expression and pose tag with the character it belongs to; never move a feature from one to \
             another. SCENE and any camera, setting, lighting or style tags are shared. Rearrange freely: a scene tag \
             written under a character belongs with the shared tags, and a character's feature written under SCENE \
             belongs to that character.",
            ["", "", "TWO", "THREE", "FOUR"][n.min(4)]
        ),
        );
    }

    // Who is in it
    let person_counts: Vec<&str> = norms
        .iter()
        .copied()
        .filter(|n| {
            n.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
                && (n.ends_with("girl") || n.ends_with("girls") || n.ends_with("boy") || n.ends_with("boys") || n.ends_with("other") || n.ends_with("others"))
                && !n.contains(' ')
        })
        .collect();
    let scenery = has("no humans") || (has("scenery") && person_counts.is_empty());
    let multi = person_counts.len() > 1
        || person_counts.iter().any(|n| !n.starts_with('1'))
        || has("multiple girls")
        || has("multiple boys");
    if scenery && !config.two_char {
        g.push("This is a scenery / no-people image. Never add characters or body tags; the framing and body-part rules \
                don't apply. Improve composition, depth (foreground / background), atmosphere, weather, time of day and \
                lighting instead.".into());
    } else if multi && !config.two_char {
        g.push(format!(
            "There are several characters ({}). Keep the count exactly. Keep each character's traits attached to the \
             right person and don't merge or swap them; prefer interaction or placement tags (e.g. side by side, facing \
             each other, holding hands) over new people.",
            person_counts.join(", ")
        ));
    }

    // LoRAs and character names must survive untouched
    let loras: Vec<&str> = tokens.iter().filter(|t| t.normalized.starts_with("<lora:")).map(|t| t.text.as_str()).collect();
    if !loras.is_empty() {
        g.push(format!(
            "LoRA tags are present ({}). Keep them exactly as written, with their weights, and keep unusual words near \
             them: those are likely trigger words.",
            loras.join(", ")
        ));
    }
    let named: Vec<&str> = tokens
        .iter()
        .filter(|t| t.text.contains("\\(") || (t.text.contains('(') && t.text.contains(')') && !t.text.starts_with('(')))
        .map(|t| t.text.as_str())
        .collect();
    if !named.is_empty() {
        g.push(format!(
            "Character / series names are present ({}). Keep them exactly, including the escaped brackets.",
            named.join(", ")
        ));
    }

    // Prose vs tags
    let prose = tokens.iter().filter(|t| t.category == TokenCategory::NaturalLanguageClause).count();
    if prose > 0 && config.checkpoint.uses_explicit_weights() {
        g.push(format!(
            "The prompt mixes {prose} natural-language sentence(s) with tags. Keep those as clean sentences in \
             improved_prompt instead of chopping them into tags{}.",
            if config.checkpoint.supports_natural_language() { "" } else { " (or turn each into a few precise tags if this checkpoint reads tags only)" }
        ));
    }

    // Explicit content
    // whole words: "oral" isn't in "floral", "cum" isn't in "cumulus", "anal" isn't in "canal"
    if norms.iter().any(|n| EXPLICIT_WORDS.iter().any(|w| crate::parser::has_phrase(n, w))) {
        g.push("The scene is explicit. Keep the explicit tags precise and anatomically consistent with the pose and \
                camera angle; don't soften, censor or drop them.".into());
    }

    // Length: expand short prompts, condense long ones, as far as the detail level allows
    use crate::profiles::{CheckpointProfile, DetailLevel};
    let (clip_tokens, _) = estimate_clip_tokens(raw_prompt);
    // CLIP reads 75-token chunks, and BREAK gives each character its own chunk(s);
    // Anima (an LLM text encoder) and Flux (T5) have no such limit
    let max_clip_tokens = 75 * if config.two_char { config.char_count() } else { 1 };
    let clip = !matches!(config.checkpoint, CheckpointProfile::Anima | CheckpointProfile::Flux);
    if tokens.len() < 10 && config.detail == DetailLevel::Balanced && !config.two_char {
        g.push(format!(
            "The prompt is short ({} tags). Expand it into a complete scene (about 20-35 tags) with details that fit what's \
             there; add them as changes with action 'add'.",
            tokens.len()
        ));
    } else if clip && clip_tokens > max_clip_tokens && config.detail != DetailLevel::Rich {
        g.push(if config.two_char {
            format!(
                "The prompt is long (about {clip_tokens} CLIP tokens for {} characters). Condensing comes first: merge \
                 redundant tags and drop filler so each character fits in about 75 tokens if possible, before adding \
                 anything new.",
                config.char_count()
            )
        } else {
            format!(
                "The prompt is long (about {clip_tokens} CLIP tokens). Condensing comes first: merge redundant tags and \
                 drop filler so improved_prompt fits in about 75 tokens if possible, before adding anything new."
            )
        });
    }

    // What the scene already has vs what's missing (strip prompt tags that contradict LoRA first)
    // (only on the 1 Character tab: with several characters the text holds all of them, and
    // the first LoRA box says nothing about the others)
    let clean_tokens = if config.lora_tags_raw.trim().is_empty() || config.two_char {
        tokens.clone()
    } else {
        crate::analyzer::strip_lora_tokens(tokens.clone(), &config.lora_tags_raw, true)
    };
    let scene = crate::scene::describe(&clean_tokens);
    let present: Vec<&str> = scene
        .details
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| {
            matches!(*k, "Setting" | "Light / time" | "Expression" | "Camera" | "Style")
                && !(*k == "Camera" && (config.camera_angle > 0 || config.target_view != crate::profiles::TargetView::None))
        })
        .collect();
    if !present.is_empty() {
        g.push(format!(
            "Already specified: {}. Refine these rather than replacing them.",
            present.join(", ").to_lowercase()
        ));
    }
    let missing: Vec<&str> = scene
        .missing
        .into_iter()
        .filter(|&m| !(m == "camera framing" && (config.target_view != crate::profiles::TargetView::None || config.camera_angle > 0)))
        .collect();
    if !missing.is_empty() && !scenery {
        g.push(if config.detail == DetailLevel::Faithful {
            format!(
                "Not specified yet: {}. Point these out in issues; FAITHFUL allows filling only the most important one.",
                missing.join(", ")
            )
        } else {
            format!("Not specified yet: {}. Fill these in with details that match the rest of the scene.", missing.join(", "))
        });
    } else if scenery && missing.contains(&"lighting / time of day") {
        g.push("No lighting or time of day yet: add one that suits the place.".into());
    }

    // What the rule engine already found
    if config.resolve_conflicts_enabled {
        let rules = crate::analyzer::analyze_with_rules(&tokens, config);
        let mut conflicts: Vec<String> = Vec::new();
        for c in rules.changes {
            if matches!(
                c.action,
                crate::analyzer::ChangeAction::FramingSafeSwap
                    | crate::analyzer::ChangeAction::PositioningConflict
                    | crate::analyzer::ChangeAction::PoseConflict
                    | crate::analyzer::ChangeAction::AnatomyConflict
                    | crate::analyzer::ChangeAction::SizeConflict
                    | crate::analyzer::ChangeAction::StyleConflict
            ) && !conflicts.contains(&c.original)
            {
                conflicts.push(c.original);
            }
        }
        if !conflicts.is_empty() {
            g.push(format!(
                "The rule checker already found framing conflicts with: {}. improved_prompt must resolve these too.",
                conflicts.join(", ")
            ));
        }

        let prop_tiers = crate::analyzer::detect_all_proportion_tiers(&tokens);
        for (tier, tier_tok) in prop_tiers {
            let rec = crate::analyzer::get_proportion_recommendations(tier);
            g.push(format!(
                "Detected proportion tier '{}'. Reinforce it using compatible clothing ({}) and pose/physics ({}); strictly avoid conflicting volume tags ({}).",
                tier_tok.text,
                rec.clothing_interaction.join(", "),
                rec.pose_and_physics.join(", "),
                rec.auto_negative_lock.join(", ")
            ));
        }
    }
    g
}

/// Which instruction style a model gets: families follow different formats best.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Gemma 4: markdown sections plus a format example from an unrelated scene.
    Gemma,
    /// Qwen2.5 (ChatML): plain uppercase sections (the format these were tuned on).
    Qwen25,
    /// Qwen3 / Qwen3.5: same format as Qwen2.5; thinking is switched by the template.
    Qwen3,
    /// 1.5B / 3B models: a short version with fewer tasks and fewer items.
    Small,
    /// Anything else the user has in Ollama.
    Generic,
}

/// Parameter count in billions from a model name ("llama3.2:3b" -> 3.0, "qwen2.5:32b" -> 32.0).
/// 10B parameters or more (the 12B): kept in memory only briefly after a Deep Scan.
pub fn is_big_model(name: &str) -> bool {
    param_billions(name).is_some_and(|b| b >= 10.0)
}

/// Sets how long Ollama keeps an already loaded model ("2m", "0" = unload now). Only touches a
/// model that is loaded: a request for one that isn't would load it first.
pub fn set_keep_alive(host: &str, model: &str, keep_alive: &str) {
    let Ok(base) = base_url(host) else { return };
    let Ok(http) = client(Some(Duration::from_secs(10))) else { return };
    let loaded = http
        .get(format!("{base}/api/ps"))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|ps| ps["models"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .any(|m| m["name"].as_str().is_some_and(|n| same_model(n, model)));
    if loaded {
        let _ = http.post(format!("{base}/api/generate")).json(&json!({ "model": model, "keep_alive": keep_alive })).send();
    }
}

fn param_billions(name: &str) -> Option<f32> {
    let bytes = name.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if c != b'b' || i == 0 || !bytes[i - 1].is_ascii_digit() {
            continue;
        }
        // the 'b' must end the size (not part of a word like "13bit")
        if bytes.get(i + 1).map(|n| n.is_ascii_alphanumeric()).unwrap_or(false) {
            continue;
        }
        let mut start = i;
        while start > 0 && (bytes[start - 1].is_ascii_digit() || bytes[start - 1] == b'.') {
            start -= 1;
        }
        if let Ok(v) = name[start..i].parse::<f32>() {
            return Some(v);
        }
    }
    None
}

impl ModelFamily {
    pub fn detect(model: &str) -> Self {
        let m = model.to_lowercase();
        let m = m.strip_suffix(":latest").unwrap_or(&m);
        let small = param_billions(m).map(|b| b < 4.0).unwrap_or(false);
        if small {
            Self::Small
        } else if m.contains("gemma") {
            Self::Gemma
        } else if m.contains("qwen3") {
            Self::Qwen3
        } else if m.contains("qwen") {
            Self::Qwen25
        } else {
            Self::Generic
        }
    }

    /// Sampling temperature: Gemma writes richer details a little warmer; small models need it cold.
    pub fn temperature(&self) -> f32 {
        match self {
            Self::Gemma => 0.3,
            Self::Small => 0.1,
            _ => 0.15,
        }
    }

    fn max_issues_changes(&self) -> (usize, usize) {
        if *self == Self::Small {
            (4, 6)
        } else {
            (8, 12)
        }
    }
}

/// Builds the system prompt and user message for the deep scan, in the style that
/// works best for `family`. The content is shared; only the format and amount differ.
pub fn build_ai_system_and_user_prompt_for(raw_prompt: &str, config: &ScanConfig, family: ModelFamily) -> (String, String) {
    let small = family == ModelFamily::Small;
    let (max_issues, max_changes) = family.max_issues_changes();
    let cleanup = if config.cleanup_enabled {
        "ON: drop redundant filler, duplicates and synonym bloat, and order tags the way this checkpoint expects."
    } else {
        "OFF: keep the user's wording and tag order wherever possible; only change what actually hurts the image."
    };
    let pinned = if config.pinned_emphasis_raw.trim().is_empty() {
        "none".to_string()
    } else if config.checkpoint.uses_explicit_weights() {
        format!("{} (keep them, weighted {:.2})", config.pinned_emphasis_raw.trim(), config.emphasis_weight)
    } else {
        format!("{} (keep them prominent in the description)", config.pinned_emphasis_raw.trim())
    };
    let quality_tags = config.checkpoint.canonical_quality_tags();
    let quality_line = if quality_tags.is_empty() {
        "Quality tags for this checkpoint: none (natural language only). Drop quality filler (masterpiece, best quality, \
         8k, 4k, uhd, ultra detailed, hyperdetailed, highly detailed, best resolution, octane render...)."
            .to_string()
    } else {
        format!(
            "Quality tags for this checkpoint: {}. Use exactly these; drop other quality filler (8k, 4k, uhd, ultra \
             detailed, hyperdetailed, highly detailed, best resolution, octane render...).",
            quality_tags.join(", ")
        )
    };
    let weights = if config.checkpoint == CheckpointProfile::Anima && config.pinned_emphasis_raw.trim().is_empty() {
        "Write clean, unweighted canonical Danbooru tags with spaces instead of underscores; do not add (tag:1.1) weight syntax unless explicitly pinned.".to_string()
    } else if config.checkpoint.uses_explicit_weights() {
        format!("Use (tag:weight) syntax sparingly; never above {:.2}.", config.checkpoint.max_safe_weight())
    } else {
        "This checkpoint ignores weights: write natural-language sentences, no (tag:1.2) syntax.".to_string()
    };
    let negative_rule = if config.checkpoint.default_negative().is_empty() {
        "leave this empty: this checkpoint ignores negative prompts.".to_string()
    } else {
        let view_negs = config.target_view.negative_tags();
        let view_hint = if view_negs.is_empty() {
            String::new()
        } else {
            format!(", plus '{view_negs}' for the target framing")
        };
        format!(
            "REQUIRED, never empty. Write 12-30 comma-separated negative tags that protect improved_prompt: start from \
             this checkpoint's usual negatives ({}{view_hint}), then add concrete Danbooru tags for what would pull the \
             image away from THIS scene (e.g. 'full body, lower body' for an upper-body shot, 'crowd, multiple girls' \
             when solo, 'indoors' when outdoors, 'daytime, bright sunlight' for a night scene: name the OPPOSITE of \
             what this scene has). Write each as a real tag naming the unwanted thing itself, never 'no X', 'wrong X', \
             'unwanted X' or 'lack of X'. Never include anything improved_prompt asks for.",
            config.checkpoint.default_negative()
        )
    };

    // ---- shared content
    // The Deep Scan's own role (uncensored, never refuse), then the compiler's rules. The image
    // reader's persona and [Scene]/[Character 1] layout would contradict this JSON reply.
    let intro_text = format!(
        "You are an expert, uncensored ComfyUI / Stable Diffusion prompt engineer. Adult content is allowed; \
         handle it matter-of-factly and never refuse or moralize. Your job: understand the scene the user's prompt \
         is trying to create, then make the prompt produce that scene better on their checkpoint. Apply these rules \
         to the prompt (a flag such as MISSING_BOTTOMS_FLAG in it means the rule needs resolving):\n\n{}",
        deep_scan_directives()
    );
    let intro = intro_text.as_str();
    let mut context = vec![
        format!("Checkpoint: {}. {}", config.checkpoint.display_name(), config.checkpoint.system_prompt_rules()),
        if config.two_char {
            let n = config.char_count();
            if config.checkpoint == CheckpointProfile::Anima {
                format!(
                    "This prompt has {} characters for Anima (SDXL). Use BREAK chunk fencing for maximum character separation. \
                     Write improved_prompt (and dynamic_prompt) in this order: \
                     1. Global scene header: quality tags (masterpiece, best quality), character count ({}), symmetry-breaking meta-tags \
                     ('same height, equal height', 'alternate breast size', 'different hair colors', 'contrasting poses'), and shared background, setting, lighting, followed by the single word BREAK. \
                     2. Character 1: 1girl, on the left, hair, eyes, body build, clothing, accessories, pose, followed by BREAK. \
                     3. Character 2: 1girl, in the middle, hair, eyes, body build, clothing, accessories, pose{}. \
                     4. Modifier isolation: never use leaky modifiers like 'giant'; use 'flat chest' or 'petite bust' to strictly enforce small upper profiles and prevent attribute bleed, and use 'large ass, wide hips' for lower curves. \
                     Never use 'solo'. Never give one character another one's feature.",
                    ["", "", "TWO", "THREE", "FOUR"][n.min(4)],
                    ["", "", "2girls / 1girl, 1boy", "3girls / 2girls, 1boy", "4girls / 2girls, 2boys"][n.min(4)],
                    (3..=n).map(|k| format!("; then BREAK; then Character {k}: 1girl, on the right, hair, body, clothing, pose")).collect::<String>()
                )
            } else {
                format!(
                    "This prompt has {} characters. Write improved_prompt (and dynamic_prompt) in this order: quality tags, \
                     the character count (e.g. {}) and tags about them together (holding hands, facing each other...); then \
                     the single word BREAK; then ALL of character 1's own tags; then BREAK; then ALL of character 2's own \
                     tags{}; then BREAK; then the shared camera, setting, lighting and style. Never use 'solo'. Never give \
                     one character another one's feature.",
                    ["", "", "TWO", "THREE", "FOUR"][n.min(4)],
                    ["", "", "2girls / 1girl, 1boy", "3girls / 2girls, 1boy", "4girls / 2girls, 2boys"][n.min(4)],
                    (3..=n).map(|k| format!("; then BREAK; then ALL of character {k}'s own tags")).collect::<String>(),
                )
            }
        } else {
            String::new()
        },
        weights,
        quality_line,
        format!(
            "Target camera view: {}. {} (This is a menu label, not a tag: write normal tags like 'upper body'.)",
            config.target_view.display_name(),
            config.target_view.system_prompt_rules()
        ),
        format!("Must keep and emphasize: {pinned}"),
        if config.two_char {
            let n = config.char_count();
            let mut loras = Vec::new();
            if !config.lora_tags_raw.trim().is_empty() {
                loras.push(format!("Character 1's LoRA: {}", config.lora_tags_raw.trim().trim_matches(',')));
            }
            for i in 1..n {
                let l = config.lora_of(i).trim().trim_matches(',');
                if !l.is_empty() {
                    loras.push(format!("Character {}'s LoRA: {l}", i + 1));
                }
            }
            if loras.is_empty() {
                "LoRA tags: none".to_string()
            } else {
                format!(
                    "LoRA tags (HIGHEST PRIORITY — {}): the app puts each character's LoRA into the final prompt \
                     automatically. They define the character's fixed appearance (hair, eyes, skin, body type, features). \
                     Do not repeat them in improved_prompt, and NEVER contradict them: drop any conflicting hair, eye, skin, \
                     or body tags from improved_prompt. In scene_summary and scene_breakdown, ALWAYS describe each character \
                     matching their LoRA.",
                    loras.join("; ")
                )
            }
        } else if config.lora_tags_raw.trim().is_empty() {
            "LoRA tags: none".to_string()
        } else {
            format!(
                "LoRA tags (HIGHEST PRIORITY): {}. The app puts these into the final prompt exactly as written. \
                 They define the character's fixed appearance (hair, eyes, skin, body type, features). Do not \
                 repeat them in improved_prompt, and NEVER contradict them: drop any conflicting hair, eye, skin, \
                 or body tags from improved_prompt (e.g. if LoRA has 'white hair', drop 'black hair'; if LoRA has \
                 'amber eyes', drop 'grey eyes'; if LoRA has 'short hair', drop 'long hair'). In scene_summary and \
                 scene_breakdown, ALWAYS describe the character matching their LoRA.",
                config.lora_tags_raw.trim().trim_matches(',')
            )
        },
        format!("Cleanup: {cleanup}"),
    ];
    for (list, idx, what) in [
        (crate::profiles::CAMERA_ANGLES, config.camera_angle, "camera angle"),
        (crate::profiles::POSES, config.pose, "pose"),
    ] {
        if let Some((label, tags)) = list.get(idx).filter(|_| idx > 0) {
            context.push(format!(
                "The user picked this {what}: {label} ({tags}). Use these tags in improved_prompt instead of the prompt's \
                 own {what}, and make the rest of the scene agree with it."
            ));
        }
    }
    if small {
        // shorter for small models, but never at the cost of the user's own choices
        // (LoRA tags, Keep list, angle / pose picks): only generic lines go
        let generic_view = config.target_view == crate::profiles::TargetView::None;
        context.retain(|c| {
            !c.starts_with("Cleanup:")
                && c != "LoRA tags: none"
                && c != "Must keep and emphasize: none"
                && !(generic_view && c.starts_with("Target camera view:"))
        });
    }

    let mut steps = vec![
        if small {
            "scene_summary: one sentence about the intended image. scene_breakdown: a few words per field.".to_string()
        } else {
            "scene_summary / scene_breakdown: read every tag and describe the intended image: subject(s) and appearance, \
             pose/action, camera framing, setting, lighting, mood/style. Never leave a field empty: when the prompt doesn't \
             specify it, write 'not specified' and propose something specific that fits the scene."
                .to_string()
        },
        format!(
            "issues: up to {max_issues} things that weaken it: tags that contradict each other or the target view, body-part \
             tags that pull the camera away from the framing, filler, vague tags, wrong syntax or weights, and gaps that \
             make results generic (no lighting, no background, unclear pose or expression)."
        ),
        improved_prompt_step(config),
    ];
    if wants_sentence(config) {
        steps.push(
            "natural_sentence: 1-2 concise natural-language sentences for the [Scene / Environment & Shared Framing] block \
             locking the subject (body proportions, hair length and color, headwear such as sunglasses on head), followed \
             immediately by the primary physical verb/posture and environment (e.g. 'A busty girl with long black hair and \
             sunglasses resting on her head sits on a bed, leaning back slightly with one knee raised high.'), and a second \
             sentence locking the specific clothing layers, matching bottoms, and active limb/prop interaction (e.g. 'She wears \
             a skin-tight, light blue logo-print tube top and matching bikini bottoms, using both hands to pull a white \
             thigh-high stocking up her raised leg while looking at the viewer.'). With several characters, say clearly which \
             one has which features and who does what by position; never describe anyone in a way their tags contradict \
             (e.g. no 'petite' or 'slender' for a character with large/huge breasts), and leave quality words (masterpiece, \
             best quality) to the tags."
                .to_string(),
        );
    }
    if config.dynamic {
        steps.push(
            "dynamic_prompt: improved_prompt again, but with 3-5 Dynamic Prompts variation groups written as \
             {option1|option2|option3} for things that can change between images without changing the scene's identity: \
             pose, camera angle, expression, lighting or time of day, small background details. Keep the subject, \
             character, hair, outfit, setting and art style fixed (no groups for them). Every option must fit the scene."
                .to_string(),
        );
    }
    steps.push(format!("recommended_negative: {negative_rule}"));
    steps.push(format!(
        "changes: up to {max_changes} individual edits from improved_prompt, so the user can apply them one at a time. \
         'original' is one tag copied exactly from the user's prompt, or \"\" when adding something new. action is one of: \
         replace, remove, add, reweight, framing_safe_swap. 'suggested_alternative' is the new text (\"\" for remove). For \
         framing_safe_swap, 'framing_alternative' is a camera change that would let the user keep the original tag."
    ));

    let all_boxes;
    let guidance_text = if config.two_char {
        let mut parts = vec![raw_prompt.trim().to_string(), config.scene_prompt_raw.trim().to_string()];
        parts.extend((1..config.char_count()).map(|i| config.prompt_of(i).trim().to_string()));
        all_boxes = parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(", ");
        all_boxes.as_str()
    } else {
        raw_prompt
    };
    let mut guidance = prompt_guidance(guidance_text, config);
    if small {
        guidance.truncate(3);
    }

    let mut rules = vec![
        "Never change who the subject is, how many subjects there are, or the overall scenario.".to_string(),
        "Keep every concrete detail the user wrote (traits, clothing, objects, setting, expression, mood). Only remove a \
         tag if it is filler or truly conflicts. Never contradict the user: if they wrote 'smile', the character smiles."
            .to_string(),
        "improved_prompt is a plain comma-separated tag list (or prose for natural-language checkpoints) with balanced \
         brackets; write weights only as (tag:1.2) and never wrap the whole prompt in brackets."
            .to_string(),
        "Reply with JSON only.".to_string(),
    ];
    if !small {
        rules.insert(2, "Be specific and concrete rather than generic ('good lighting' is useless).".into());
        rules.insert(4, "A tag that is already in the prompt is not missing; never suggest adding it again.".into());
        rules.insert(
            5,
            format!("The most important differences between the original and improved_prompt appear in changes (at most {max_changes})."),
        );
        rules.insert(6, "Keep issues and reasons short (under 20 words each).".into());
    }

    context.retain(|c| !c.is_empty());
    // ---- format per family
    let bullets = |v: &[String]| v.iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n");
    let numbered = |v: &[String]| v.iter().enumerate().map(|(i, l)| format!("{}. {l}", i + 1)).collect::<Vec<_>>().join("\n");
    let system_prompt = match family {
        ModelFamily::Gemma => {
            // the example shows the layout the model has to follow (one character, or several with BREAK)
            let ex_chars = [
                ("1boy, astronaut, white spacesuit, standing", "an astronaut"),
                ("1girl, robot girl, silver armor, waving", "a robot girl"),
                ("1boy, alien boy, green skin, sitting", "an alien boy"),
                ("1girl, cat girl, orange fur, jumping", "a cat girl"),
            ];
            let q_prefix = if quality_tags.is_empty() {
                String::new()
            } else {
                format!("{}, ", quality_tags.join(", "))
            };
            let (ex_summary, ex_subject, ex_prompt, ex_sentence) = if config.two_char {
                let n = config.char_count();
                let count = ["", "", "1boy, 1girl", "2boys, 1girl", "2boys, 2girls"][n.min(4)];
                let who = ex_chars[..n].iter().map(|c| c.1).collect::<Vec<_>>();
                if config.checkpoint == CheckpointProfile::Anima {
                    (
                        format!("{} stand side by side on a red desert dune at dusk.", who.join(" and ")),
                        ex_chars[..n].iter().map(|c| c.0).collect::<Vec<_>>().join("; "),
                        format!(
                            "{q_prefix}{count}, side by side, same height, contrasting poses, red desert, sand dunes, dusk, rim lighting, BREAK, \
                             1boy, on the left, spacesuit, standing, BREAK, 1girl, on the right, silver armor, waving"
                        ),
                        format!("On the left {} stands while the robot girl on the right waves, all lit by the dying sun.", who[0]),
                    )
                } else {
                    (
                        format!("{} stand side by side on a red desert dune at dusk.", who.join(" and ")),
                        ex_chars[..n].iter().map(|c| c.0).collect::<Vec<_>>().join("; "),
                        format!(
                            "{q_prefix}{count}, side by side, BREAK, {}, BREAK, full body, from below, red desert, \
                             sand dunes, dusk, rim lighting",
                            ex_chars[..n].iter().map(|c| c.0).collect::<Vec<_>>().join(", BREAK, ")
                        ),
                        format!("On the left {} waves at the others while the last one looks away, all lit by the dying sun.", who[0]),
                    )
                }
            } else {
                (
                    "A lone astronaut stands on a red desert dune at dusk.".to_string(),
                    "1boy, astronaut, white spacesuit".to_string(),
                    format!(
                        "{q_prefix}1boy, solo, astronaut, white spacesuit, standing, full body, from below, \
                         red desert, sand dunes, dusk, rim lighting"
                    ),
                    "A lone astronaut gazes up at a sky he will never reach, dwarfed by endless dunes.".to_string(),
                )
            };
            let natural_example = if wants_sentence(config) {
                format!(",\n  \"natural_sentence\": \"{ex_sentence}\"")
            } else {
                String::new()
            };
            let dynamic_example = if config.dynamic {
                if config.two_char {
                    let n = config.char_count();
                    let count = ["", "", "1boy, 1girl", "2boys, 1girl", "2boys, 2girls"][n.min(4)];
                    let mut char_dyn = Vec::new();
                    for (i, c) in ex_chars[..n].iter().enumerate() {
                        let pose_opt = if i == 0 { "{standing|kneeling|walking}" } else { "{standing|sitting}" };
                        char_dyn.push(format!("{}, {pose_opt}", c.0.trim_end_matches(", standing")));
                    }
                    format!(
                        ",\n  \"dynamic_prompt\": \"{q_prefix}{count}, side by side, BREAK, {}, BREAK, full body, {{from below|from side}}, red desert, sand dunes, {{dusk|night|dawn}}, rim lighting\"",
                        char_dyn.join(", BREAK, ")
                    )
                } else {
                    format!(
                        ",\n  \"dynamic_prompt\": \"{q_prefix}1boy, solo, astronaut, white spacesuit, {{standing|kneeling|walking}}, full body, {{from below|from side}}, red desert, sand dunes, {{dusk|night|dawn}}, rim lighting\""
                    )
                }
            } else {
                String::new()
            };
            let ex_negative = if config.checkpoint.default_negative().is_empty() {
                ""
            } else {
                "worst quality, low quality, crowd, daytime"
            };
            format!(
                "{intro}\n\n## Context\n{}\n\n## Task (fill these JSON fields, in this order)\n{}\n\n{}## Rules\n{}\n\n\
                 ## Output format\nOne JSON object. Format example from an UNRELATED scene: copy its structure, never its content.\n\
                 ```json\n{{\n  \"scene_summary\": \"{ex_summary}\",\n  \
                 \"scene_breakdown\": {{\"subject\": \"{ex_subject}\", \"pose\": \"standing\", \"framing\": \
                 \"full body, from below\", \"setting\": \"red desert dunes\", \"lighting\": \"dusk, rim lighting\", \"mood\": \
                 \"lonely, epic\"}},\n  \"issues\": [\"no time of day\"],\n  \"improved_prompt\": \"{ex_prompt}\"{natural_example}{dynamic_example},\n  \"recommended_negative\": \"{ex_negative}\",\n  \
                 \"changes\": [{{\"original\": \"\", \"action\": \"add\", \"reason\": \"adds time of day\", \
                 \"suggested_alternative\": \"dusk, rim lighting\", \"framing_alternative\": \"\"}}]\n}}\n```",
                bullets(&context),
                numbered(&steps),
                if guidance.is_empty() { String::new() } else { format!("## About this prompt (detected from it; follow these)\n{}\n\n", bullets(&guidance)) },
                bullets(&rules),
            )
        }
        _ => format!(
            "{intro}\n\nCONTEXT\n{}\n\nSTEPS\n{}\n\n{}RULES\n{}",
            bullets(&context),
            numbered(&steps),
            if guidance.is_empty() { String::new() } else { format!("ABOUT THIS PROMPT (detected from it; follow these)\n{}\n\n", bullets(&guidance)) },
            bullets(&rules),
        ),
    };

    let tag_filter = AntigravityTagFilter::new();
    let filtered_single;
    let two;
    let raw_prompt = if config.two_char {
        let n = config.char_count();
        let place = |i: usize| match (n, i) {
            (2, 0) | (3, 0) => "on the left",
            (2, _) | (3, 2) => "on the right",
            (3, _) => "in the middle",
            (_, 0) => "far left",
            (_, 1) => "second from the left",
            (_, 2) => "second from the right",
            _ => "far right",
        };
        let mut sections = vec![format!(
            "SCENE (shared by all: setting, camera, lighting, mood, what they do together):\n{}",
            config.scene_prompt_raw.trim()
        )];
        sections.push(format!("CHARACTER 1 ({}):\n{}", place(0), tag_filter.process_tags(raw_prompt)));
        for i in 1..n {
            sections.push(format!(
                "CHARACTER {} ({}):\n{}",
                i + 1,
                place(i),
                tag_filter.process_tags(config.prompt_of(i))
            ));
        }
        two = sections.join("\n\n");
        two.as_str()
    } else {
        filtered_single = tag_filter.process_tags(raw_prompt);
        filtered_single.as_str()
    };
    let user_prompt = match family {
        ModelFamily::Gemma => format!(
            "Analyze and improve this ComfyUI prompt. Reply with the JSON object only.\n\nPROMPT:\n\"\"\"\n{}\n\"\"\"",
            raw_prompt.trim()
        ),
        _ => format!("Analyze and improve this ComfyUI prompt:\n\"\"\"\n{}\n\"\"\"", raw_prompt.trim()),
    };
    (system_prompt, user_prompt)
}

/// The Deep Scan writes a natural_sentence only when the toggle is on and the
/// checkpoint understands natural language.
fn wants_sentence(config: &ScanConfig) -> bool {
    config.natural && config.checkpoint.supports_natural_language()
}

/// The improved_prompt task, worded for the chosen detail level.
fn improved_prompt_step(config: &ScanConfig) -> String {
    use crate::profiles::DetailLevel;
    let base = "improved_prompt: rewrite the full prompt so it delivers the SAME scene more convincingly. Keep every \
                subject, character, LoRA tag (<lora:...>) and the user's core ideas. Fix every issue. Body-part tags that \
                break the framing become framing-safe equivalents (e.g. 'ass' with upper body -> 'curvy, hourglass \
                figure, narrow waist, arched back').";
    let prose = !config.checkpoint.uses_explicit_weights() && !config.two_char;
    match config.detail {
        DetailLevel::Faithful => format!(
            "{base} DETAIL LEVEL = FAITHFUL: keep the user's tags and wording; only fix what's wrong (conflicts, filler, \
             syntax, weights). {add_limit}, and only to fill an obvious gap such as missing lighting.",
            add_limit = if prose { "Add at most 1-2 descriptive details" } else { "Add at most 3 tags" }
        ),
        DetailLevel::Balanced => format!(
            "{base} Fill gaps: if there's no lighting, background/setting or expression, add specific ones that fit THIS \
             scene's setting, time of day and mood. Invent them from the scene; don't reuse stock phrases."
        ),
        DetailLevel::Rich => format!(
            "{base} DETAIL LEVEL = RICH: turn it into a detailed, vivid description of the same scene. Go through this \
             checklist and describe every item that fits the scene: \
             CHARACTER (hair length, style and color; hair accessories; eye color and shape; skin; build; expression; hands); \
             OUTFIT (each garment's type, material, fit, color and pattern; accessories, jewelry, footwear); \
             POSE (body position, what the hands do, head tilt, where the gaze goes); \
             SETTING (foreground, background, props, architecture or nature details); \
             LIGHT (light source, its direction and color, time of day, shadows, reflections); \
             CAMERA (shot type, angle, lens feel, depth of field); \
             ATMOSPHERE (mood, weather, particles like petals or dust, color palette, art style or medium). \
             Write descriptive pairs instead of bare tags ('flowing silver hair' rather than 'silver hair', 'red silk \
             kimono with gold embroidery' rather than 'kimono'); enriching a user's tag this way still counts as keeping it. \
             {length} Every added detail must fit the scene's setting, time of day and mood and must never contradict \
             the user's tags.",
            length = if prose {
                "Aim for 3-5 vivid sentences (subject, appearance, action, setting, light, style, in that order)."
            } else {
                "Aim for 40-60 tags, most important first, so it stays within about three 75-token chunks."
            }
        ),
    }
}

/// The generic-format instructions (tests).
#[cfg(test)]
pub fn build_ai_system_and_user_prompt(raw_prompt: &str, config: &ScanConfig) -> (String, String) {
    build_ai_system_and_user_prompt_for(raw_prompt, config, ModelFamily::Generic)
}

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
fn close_truncated_json(text: &str) -> Option<AiJsonOutput> {
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
fn salvage_change_objects(text: &str) -> Vec<AiChangeItem> {
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

/// `is_layout_tag` for any checkpoint (quality tags of every profile count): used where the
/// checkpoint isn't known, so an "add masterpiece" never becomes a card, since
/// `ensure_quality_tags` writes the right quality tags itself.
pub fn is_layout_tag_any(normalized: &str) -> bool {
    use CheckpointProfile::*;
    [Anima, PonyV6, Illustrious, Sdxl, Sd15].iter().any(|c| is_layout_tag(normalized, *c))
}

pub fn convert_ai_changes(
    ai_items: &[AiChangeItem],
    prompt_tokens: &[crate::parser::PromptToken],
    rule_changes: &[PromptChange],
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
        if targets.is_empty() && !suggestion.is_empty() && suggestion.iter().all(|t| is_layout_tag_any(t)) {
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
            original: item.original.trim().to_string(),
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

#[derive(Debug, Clone)]
pub struct ImageAiRequest {
    pub host: String,
    pub model: String,
    /// Vision encoder (e.g. `moondream:1.8b`). If set and different from `model`, it reads the
    /// image pixels first, and its visual description is piped to `model` for checkpoint formatting.
    pub vision_encoder: Option<String>,
    pub mode: AiMode,
    pub image_path: std::path::PathBuf,
    pub config: ScanConfig,
    /// Tags the Danbooru tagger read from the image (may be empty).
    pub tag_hints: String,
    /// The tagger's high-confidence skin / species / count tags (`strong_tags`): the vision model is
    /// asked to check them, and the final prompt may not drop or contradict them.
    pub strong_tags: Vec<String>,
    /// If true and the target text model has vision capability, directly attach image pixels for a 2nd pass visual check.
    pub second_pass_vision: bool,
    /// If true, extract body shape/build from the scene reference for the character.
    #[allow(dead_code)]
    pub scene_body_shape: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ImageAiResult {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub prompt: String,
}

/// What a vision model should capture about each character, so the prompt recreates
/// them 1-to-1.
const CHARACTER_CHECKLIST: &str = "Capture each character exactly: hair (length, color, bangs or parting, \
     ahoge, braids, ornaments); face (eye color, eyebrows such as thick eyebrows, glasses, earrings, makeup, marks); \
     body (stature such as tall or petite, build, chest size); every visible garment layer by layer with its color \
     and material, plus accessories; orientation (facing the viewer, facing to the side, from behind), gaze (looking \
     at viewer, looking at another, looking away) and placement (foreground, background).";

/// Vision models gain nothing from huge images but get much slower: anything over
/// 1024 px on its longest side is scaled down and sent as JPEG.
pub(crate) fn shrink_for_vision_custom(path: &std::path::Path, max_edge: u32) -> Result<Vec<u8>> {
    let original = std::fs::read(path).with_context(|| format!("Couldn't read {}", path.display()))?;
    let Ok(img) = image::load_from_memory(&original) else {
        return Ok(original);
    };
    if img.width().max(img.height()) <= max_edge && original.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok(original);
    }
    let small = if img.width().max(img.height()) > max_edge {
        img.resize(max_edge, max_edge, image::imageops::FilterType::Triangle).to_rgb8()
    } else {
        img.to_rgb8()
    };
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85).encode_image(&small)?;
    Ok(out)
}

pub(crate) fn shrink_for_vision(path: &std::path::Path) -> Result<Vec<u8>> {
    shrink_for_vision_custom(path, 1024)
}

fn image_system_prompt(config: &ScanConfig, tag_hints: &str) -> String {
    let style = if config.checkpoint.uses_explicit_weights() {
        let quality = config.checkpoint.canonical_quality_tags().join(", ");
        let quality_prefix = if quality.is_empty() {
            String::new()
        } else {
            format!("starting with the quality tags ({quality}), ")
        };
        format!(
            "Write 25-45 comma-separated Danbooru-style tags, most important first, {quality_prefix}\
             Cover who is in it (e.g. 1girl, solo; or 'no humans, scenery' when nobody is the subject), their concrete \
             looks (hair colour and style, eye colour, build), clothing, expression, pose, the shot and angle, the \
             place, the light and the art style. Every tag is a concrete visual fact like 'silver hair', never a bare \
             category word like 'hair' or 'lighting'. {}",
            CHARACTER_CHECKLIST
        )
    } else {
        format!(
            "Write 3-5 vivid natural-language sentences: subject and appearance, clothing, pose, framing, setting, light \
             and style, most important first. {CHARACTER_CHECKLIST}"
        )
    };
    let filtered_hints = AntigravityTagFilter::new().process_tags(tag_hints);
    let hints = if filtered_hints.trim().is_empty() {
        String::new()
    } else {
        format!(
            "## Tagger Detections (from Danbooru tagger, pre-processed by AntigravityTagFilter)\n{filtered_hints}\n\n\
             ## Combination & Synthesis Instructions\n\
             Combine the visual description from the vision encoder with the Danbooru tagger detections:\n\
             1. PRESERVE specific micro-features detected by the tagger that match what is visible: mouth shapes and expressions (e.g. ':3', 'smile', 'blush'), face decals and ornaments (e.g. 'blush stickers', 'hair ornament'), animal ears and tails (e.g. 'rabbit ears', 'rabbit tail'), and specific accessories.\n\
             2. USE the visual description to correct garment and clothing misidentifications: unify single-piece dresses (e.g. 'pink china dress', 'cheongsam') instead of fragmented underwear/skirt tags, correctly identify sleeves (e.g. 'detached sleeves' instead of gloves), resolve color conflicts (e.g. choose between red eyes vs pink eyes), and add archetype tags ('kemonomimi', 'rabbit girl') and scene framing/lighting.\n\
             3. DEDUPLICATE and clean up: keep the most descriptive tag (e.g. 'very long hair' over 'long hair'), and do not add body-fur tags to smooth human skin.\n\n"
        )
    };
    format!(
        "{}\n\n\
         ## Target checkpoint\n{}. {}\n\n## Prompt style\n{style}\n\n{hints}## Rules\n\
         - Only describe what is actually visible; don't invent characters or details.\n\
         - Count people carefully; the tagger's count (1girl, 2girls, multiple girls...) is usually right. 'solo' only \
         when exactly one person is in the image.\n\
         - The tagger's detections are more reliable than the visual description for species and animal features (ears, \
         tails), skin color (keep 'dark skin' / 'dark-skinned female' whenever the tagger lists it), hair and eye colors, \
         footwear, background and framing. Keep every tagger tag unless the description clearly and specifically \
         contradicts it, and use the description for what the tagger can't name (garment details, poses, held items).\n\
         - When multiple characters are present (e.g. 2girls, 3girls):\n\
           * NEVER assign them all the same generic pose or shared body size in the main tag list.\n\
           * Include contrast meta-tags in the scene header when they differ: 'alternate breast size', 'different hair colors', 'contrasting poses', 'same height, equal height'.\n\
           * Lock vertical stature: include 'same height, equal height' and 'standing upright' on upright figures to prevent leaning or proportion distortion from squishing characters vertically.\n\
           * Isolate characters using BREAK chunk fencing by default: [quality, count, scene header, contrast tags], BREAK, 1girl, on the left, [traits, outfit, pose], BREAK, 1girl, in the middle, [traits, outfit, pose], BREAK, 1girl, on the right, [traits, outfit, pose].\n\
           * Modifier isolation: never use leaky adjectives like 'giant' near chest areas; use 'flat chest' or 'petite bust' to strictly enforce small upper profiles and prevent attribute bleed, and use 'large ass, wide hips' for lower curves.\n\
         - Never write the checkpoint's name in the prompt.\n\
         - 'description' is 1-2 plain sentences about the image; 'prompt' is the prompt itself: ONE flat, \
         comma-separated list of tags, quality tags first. No section headers such as [Scene] or [Character 1], no \
         labels, no parentheses around groups of tags, no notes: the app builds its own layout from the list.\n\
         - Reply with JSON only.",
        image_directives(),
        config.checkpoint.display_name(),
        config.checkpoint.system_prompt_rules(),
    )
}

/// "Scene / Pose reference": the vision model describes only what the picture does
/// (poses, positions, interaction, camera, setting, lighting), plus garment props
/// required by the pose (Directive 6 Prop Preservation), never who is in it.
fn scene_pose_system_prompt(tag_hints: &str, scene_body_shape: bool) -> String {
    let filtered_hints = AntigravityTagFilter::new().process_tags(tag_hints);
    let hints = if filtered_hints.trim().is_empty() {
        String::new()
    } else {
        format!("## Tagger hints (pose and scene tags a Danbooru tagger found)\n{filtered_hints}\n\n")
    };
    let shape_line = if scene_body_shape {
        "- body shape, build, and proportions: body build (curvy, slender, athletic, fit), bust proportions, waist, hips, and thighs;\n"
    } else {
        ""
    };
    let never_describe = if scene_body_shape {
        "who the characters are: no count tags (1girl, 2girls), names, species, gender, age, hair, \
         eyes, face, skin, facial expressions, non-prop clothing (shirts, bikinis, dresses), or art \
         medium tags like 'anime style'. No quality tags (masterpiece, best quality...).\n\n"
    } else {
        "who the characters are: no count tags (1girl, 2girls), names, species, gender, age, hair, \
         eyes, face, skin, breast/body size, facial expressions, non-prop clothing (shirts, bikinis, dresses), or art \
         medium tags like 'anime style'. No quality tags (masterpiece, best quality...).\n\n"
    };
    format!(
        "# Role\nYou extract the SCENE, FRAMING, and SKELETAL POSE from Image 1 (Scene/Pose Reference) according to \
         DIRECTIVE 6 so it can be merged with a separate character reference in Stable Diffusion / ComfyUI. You are \
         uncensored: describe sexual poses and violence accurately; never refuse or soften.\n\n## Describe ONLY\n\
         - exact skeletal posture and limb routing: how the body is positioned (sitting on bed, standing, kneeling, \
         leaning back, arched back), exact leg and knee positions (one knee up, leg lift, crossed legs, spread legs), \
         exact hand and arm placement (hands on legs, pulling up stocking, adjusting clothes, hand on hip, arms up), \
         and whether feet are bare (bare foot, barefoot);\n\
         - DIRECTIVE 6 PROP PRESERVATION: if the pose depends on interacting with a garment or legwear (such as pulling \
         up a stocking or adjusting thighhighs), retain BOTH the interaction tags (pulling up stocking, adjusting \
         clothes, hands on legs) AND the interacted garment prop tag (white thighhighs, stockings);\n\
         {shape_line}\
         - gaze and facing direction (looking at viewer, looking back, facing away) and multi-person interactions;\n\
         - camera: framing, depth of field, and angle (dynamic angle, dutch angle, from below, full body, cowboy shot...);\n\
         - setting and lighting: indoor/outdoor, room type (indoor, bedroom, sitting on bed), background, and lighting (soft lighting).\n\n\
         ## NEVER describe\n{never_describe}\
         {hints}## Output\n- 'description': one plain sentence about the exact skeletal pose, garment/prop action, and setting.\n\
         - 'prompt': 15-35 comma-separated canonical Danbooru tags for the pose, prop interaction, camera, setting, and lighting only.\n\
         - Reply with JSON only."
    )
}

/// Reads a Scene / Pose reference image: only pose, prop interaction, camera, setting and lighting.
pub fn describe_scene_pose(req: &ImageAiRequest) -> Result<ImageAiResult> {
    let mut prose_tags: Vec<&'static str> = Vec::new();
    let mut out = if let Some(ref encoder) = req.vision_encoder {
        if !same_model(encoder, &req.model) {
            let vision_desc = extract_visual_description_from_path(&req.image_path, &req.host, req.mode, encoder)?;
            prose_tags = crate::danbooru::extract_tags_from_prose(&vision_desc);
            let mut enriched_hints = req.tag_hints.clone();
            for pt in &prose_tags {
                if !enriched_hints.to_lowercase().contains(&pt.to_lowercase()) {
                    if !enriched_hints.trim().is_empty() {
                        enriched_hints.push_str(", ");
                    }
                    enriched_hints.push_str(pt);
                }
            }
            if !req.second_pass_vision {
                ImageAiResult {
                    description: vision_desc,
                    prompt: enriched_hints,
                }
            } else {
                let system = scene_pose_system_prompt(&enriched_hints, req.scene_body_shape);
                let user_prompt = format!(
                    "## Visual description of the pose and scene from vision encoder ({encoder}):\n{vision_desc}\n\nExtract the skeletal pose, active garment-prop interactions (Directive 6), camera angle, setting, and lighting."
                );
                describe_with_text_model(req, &system, &user_prompt)?
            }
        } else {
            describe_image_with(req, &scene_pose_system_prompt(&req.tag_hints, req.scene_body_shape), "Describe only the pose and the scene.")?
        }
    } else {
        describe_image_with(req, &scene_pose_system_prompt(&req.tag_hints, req.scene_body_shape), "Describe only the pose and the scene.")?
    };
    for pt in prose_tags {
        if !out.prompt.to_lowercase().contains(&pt.to_lowercase()) {
            out.prompt.push_str(", ");
            out.prompt.push_str(pt);
        }
    }
    // Filter out character identity and medium bleed ("anime style") while keeping AI pose/prop tags and body shape tags
    let filtered = crate::scene::filter_scene_and_pose_only_with_shape(&out.prompt, true, true);
    out.prompt = filtered
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty() && !t.eq_ignore_ascii_case("anime style") && !t.eq_ignore_ascii_case("anime"))
        .collect::<Vec<_>>()
        .join(", ");
    if out.prompt.trim().is_empty() {
        return Err(anyhow!("The model found no pose or scene to reuse. Try again."));
    }
    Ok(out)
}

// ===========================================================================
// Dual-Pass Moondream + WD14 + Gemma 4 12B Pipeline
// ===========================================================================

/// Executes the full dual-pass vision pipeline according to ANTIGRAVITY_RUST_MISSION.md:
/// 1. Runs Moondream pass 1 and pass 2 with temperature 0.1 (+0.2 on retry)
/// 2. Converts prose descriptions to Danbooru tags using selected_tags.csv vocabulary
/// 3. Reconciles WD14 tags (0.22 general, 0.85 character) with Moondream (WD14 wins attributes, Moondream wins pose)
/// 4. Optionally performs LLM tag synthesis using Gemma 4 12B in Ollama (JSON mode, temp 0.35)
/// 5. Automatically falls back to mechanical merge and generates warnings if Ollama is unreachable.
pub fn run_dual_pass_pipeline(
    host: &str,
    mode_ai: AiMode,
    vision_model: &str,
    text_model: &str,
    image_path: &std::path::Path,
    raw_wd14_tags: &str,
    opts: crate::prompt_condenser::PipelineOptions,
) -> crate::prompt_condenser::PipelineResult {
    run_dual_pass_asymmetric_pipeline(
        host,
        mode_ai,
        vision_model,
        vision_model,
        text_model,
        image_path,
        raw_wd14_tags,
        opts,
    )
}

/// Executes dual-pass vision with potentially different models for Pass 1 (spatial/body) and Pass 2 (identity/wardrobe).
pub fn run_dual_pass_asymmetric_pipeline(
    host: &str,
    mode_ai: AiMode,
    pass1_vision_model: &str,
    pass2_vision_model: &str,
    text_model: &str,
    image_path: &std::path::Path,
    raw_wd14_tags: &str,
    opts: crate::prompt_condenser::PipelineOptions,
) -> crate::prompt_condenser::PipelineResult {
    use base64::Engine as _;
    let base = match base_url(host) {
        Ok(b) => b,
        Err(e) => {
            let mut res = crate::prompt_condenser::PipelineResult::default();
            res.warnings.push(format!("Invalid Ollama host: {e}"));
            return res;
        }
    };

    let img_bytes = match shrink_for_vision(image_path) {
        Ok(b) => b,
        Err(e) => {
            let mut res = crate::prompt_condenser::PipelineResult::default();
            res.warnings.push(format!("Failed to read image for vision: {e}"));
            return res;
        }
    };
    let b64_image = base64::engine::general_purpose::STANDARD.encode(&img_bytes);

    // The mode the user picked (Mode A: scene + body shape, Mode B: scene + skeletal pose)
    // decides the vision prompts and the synthesis instruction. It used to be forced to A.
    let effective_opts = opts;

    // the pipeline below runs with `Config::default()`, so that is the first-try temperature
    let first_try_temp = crate::prompt_condenser::Config::default().vlm_temperature;
    // the vision encoders are unloaded once, before the first text-model call (or at the end)
    let encoders_unloaded = std::cell::Cell::new(false);
    let moondream_fn = |prompt: &str, temp: f32| -> Result<String, String> {
        let vision_model = vision_model_for_prompt(effective_opts.mode, prompt, pass1_vision_model, pass2_vision_model);
        let (temperature, seed) = vlm_sampling(temp, first_try_temp);
        let body = serde_json::json!({
            "model": vision_model,
            "prompt": prompt,
            "images": [&b64_image],
            "stream": false,
            "think": false,
            "keep_alive": encoder_keep_alive(vision_model),
            "options": {
                "temperature": temperature,
                "top_p": 0.85,
                "seed": seed,
                "repeat_penalty": 1.15,
                "num_predict": 500,
                "num_ctx": 4096,
                "num_gpu": if mode_ai == AiMode::CpuOnly { Some(0) } else { None::<i32> },
            },
        });
        let c = client(Some(Duration::from_secs(120))).map_err(|e| e.to_string())?;
        let resp = c.post(format!("{base}/api/generate"))
            .json(&body)
            .send()
            .map_err(|e| if e.is_timeout() { "Moondream timed out after 2 minutes".to_string() } else { format!("Ollama unreachable: {e}") })?;
        let gen: GenerateResponse = resp.json().map_err(|e| format!("Failed to parse Moondream JSON: {e}"))?;
        let desc = gen.response.trim().to_string();
        if desc.is_empty() {
            Err("Vision encoder returned empty description".to_string())
        } else {
            Ok(desc)
        }
    };

    let gemma_fn: Option<Box<crate::prompt_condenser::GemmaFn<'_>>> = if opts.use_llm_synthesis {
        // Health check: GET /api/tags
        let check_model = (|| -> Result<bool, String> {
            let c = client(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
            let resp = c.get(format!("{base}/api/tags"))
                .send()
                .map_err(|e| format!("Failed to contact Ollama at {base}: {e}"))?;
            let tags_json = resp.text().map_err(|e| e.to_string())?;
            crate::prompt_condenser::ollama_model_pulled(&tags_json, text_model)
        })();

        match check_model {
            Ok(true) => {
                Some(Box::new(|req: &crate::prompt_condenser::GemmaRequest| -> Result<crate::prompt_condenser::GemmaReply, String> {
                    let img_opt = if req.with_image { Some(b64_image.as_str()) } else { None };
                    // the vision encoders are done: free their memory before the big model loads
                    if !encoders_unloaded.replace(true) {
                        unload_dedicated_encoders(&base, &[pass1_vision_model, pass2_vision_model], text_model);
                        // the image part is done: free the tagger's RAM for the text model too
                        crate::tagger::release();
                    }
                    let req_body = crate::prompt_condenser::ollama_chat_request_body_with(
                        text_model,
                        req.text,
                        0.10, // Calibrated deterministic synthesis temperature (prevents tag improvisation)
                        crate::prompt_condenser::GEMMA_CONTEXT_WINDOW,
                        req.max_tokens,
                        img_opt,
                        true, // json_mode
                        Some(false), // think: off
                        Some(1.1),   // repeat penalty: stops the endless tag lists the 12B writes on crowded images
                    );
                    let c = client(Some(Duration::from_secs(300))).map_err(|e| e.to_string())?;
                    let resp = c.post(format!("{base}/api/chat"))
                        .header("Content-Type", "application/json")
                        .body(req_body)
                        .send()
                        .map_err(|e| if e.is_timeout() { "Gemma synthesis timed out after 5 minutes".to_string() } else { format!("Gemma chat error: {e}") })?;
                    let resp_text = resp.text().map_err(|e| e.to_string())?;
                    if std::env::var_os("PC_DEBUG_GEMMA").is_some() {
                        eprintln!("[gemma raw reply, max_tokens {}]\n{resp_text}\n[/gemma]", req.max_tokens);
                    }
                    crate::prompt_condenser::parse_ollama_chat_response(&resp_text)
                }))
            }
            _ => None,
        }
    } else {
        None
    };

    let vocab = crate::danbooru::get_prompt_condenser_vocab();
    let cfg = crate::prompt_condenser::Config::default();
    let gemma_ref = gemma_fn.as_ref().map(|f| f.as_ref());
    let sanitized_wd14 = crate::danbooru::antigravity_tag_filter(raw_wd14_tags)
        .replace(", MISSING_BOTTOMS_FLAG", "")
        .replace(", UNANCHORED_LIMBS_FLAG", "");

    // Enrich with multi-zone/quadrant prose detections if vision model is available
    let mut multizone_warning: Option<String> = None;
    let multizone_tags = match extract_visual_description_from_path(image_path, host, mode_ai, pass2_vision_model) {
        Ok(desc) => crate::danbooru::extract_tags_from_prose(&desc),
        Err(e) => {
            multizone_warning = Some(format!("Multi-zone scan skipped: {e}"));
            Vec::new()
        }
    };
    let mut enriched_wd14 = sanitized_wd14;
    for mt in multizone_tags {
        if !enriched_wd14.to_lowercase().contains(&mt.to_lowercase()) {
            enriched_wd14.push_str(", ");
            enriched_wd14.push_str(mt);
        }
    }

    let mut result = crate::prompt_condenser::run_dual_pass(
        &enriched_wd14,
        &effective_opts,
        &cfg,
        vocab,
        &moondream_fn,
        gemma_ref,
    );
    if !encoders_unloaded.replace(true) {
        unload_dedicated_encoders(&base, &[pass1_vision_model, pass2_vision_model], text_model);
    }
    if let Some(w) = multizone_warning {
        result.warnings.push(w);
    }

    // Post-synthesis rule-engine upgrades (Deduplication, Layering, White-on-White, Gaze, and 8-Bucket Ordering)
    if !result.character_prompt.is_empty() {
        let mut char_tags = crate::danbooru::dedup_color_parents(&result.character_prompt);
        let lower_char = char_tags.to_lowercase();
        let lower_scene = result.scene_prompt.to_lowercase();

        // White-on-White Silhouette & Contrast Protection
        if lower_scene.contains("white background")
            && (lower_char.contains("white bikini")
                || lower_char.contains("white shirt")
                || lower_char.contains("white dress")
                || lower_char.contains("white hair"))
            && !lower_char.contains("bold outlines")
            && !lower_char.contains("distinct outlines")
        {
            char_tags.push_str(", bold outlines, drop shadow");
        }

        // Garment Layering Resolver (shirt + swimsuit)
        if (lower_char.contains("white shirt") || lower_char.contains("blue shirt") || lower_char.contains("sailor shirt"))
            && (lower_char.contains("school swimsuit") || lower_char.contains("one-piece swimsuit"))
            && !lower_char.contains("swimsuit under clothes")
            && !lower_char.contains("shirt over swimsuit")
        {
            char_tags.push_str(", swimsuit under clothes");
        }

        // Gaze vs. Eyelid Conflict Resolver
        if lower_char.contains("closed eyes")
            && !lower_char.contains("one eye closed")
            && !lower_char.contains("eyepatch")
            && !lower_char.contains("one eye covered")
        {
            char_tags = char_tags
                .split(',')
                .map(str::trim)
                .filter(|t| !t.eq_ignore_ascii_case("looking at viewer"))
                .collect::<Vec<_>>()
                .join(", ");
        }

        // Multi-Character Stature & Posture Lock (strip global standing from scene when sitting is active)
        if (lower_char.contains("2girls") || lower_char.contains("3girls") || lower_char.contains("multiple girls"))
            && lower_char.contains("sitting")
            && lower_scene.contains("standing")
        {
            result.scene_prompt = result
                .scene_prompt
                .split(',')
                .map(str::trim)
                .filter(|t| !t.eq_ignore_ascii_case("standing"))
                .collect::<Vec<_>>()
                .join(", ");
        }

        result.character_prompt = crate::danbooru::sort_canonical_buckets(&char_tags);
    }

    if opts.use_llm_synthesis && gemma_fn.is_none() {
        let msg = crate::prompt_condenser::ollama_not_pulled_message(text_model);
        if !result.warnings.iter().any(|w| w.contains(&msg)) {
            result.warnings.push(msg);
        }
    }

    result
}

/// Asks a vision model to write a prompt for the image. If a separate vision encoder is set
/// (such as moondream), it reads the image pixels first and pipes the description into `model`.
pub fn describe_image(req: &ImageAiRequest) -> Result<ImageAiResult> {
    if let Some(ref encoder) = req.vision_encoder {
        if !same_model(encoder, &req.model) {
            let vision_desc = observe_image(&req.image_path, &req.host, req.mode, encoder, &req.tag_hints, &req.strong_tags)?;
            return describe_image_with_desc(req, &vision_desc, encoder);
        }
    }
    describe_image_with(req, &image_system_prompt(&req.config, &req.tag_hints), "Write the prompt for this image.")
}

/// Synthesizes a visual description and tagger hints into a Danbooru prompt using the text model.
pub fn describe_image_with_desc(
    req: &ImageAiRequest,
    vision_desc: &str,
    encoder_name: &str,
) -> Result<ImageAiResult> {
    let prose_tags = crate::danbooru::extract_tags_from_prose(vision_desc);
    let mut enriched_hints = req.tag_hints.clone();
    for pt in &prose_tags {
        if !enriched_hints.to_lowercase().contains(&pt.to_lowercase()) {
            if !enriched_hints.trim().is_empty() {
                enriched_hints.push_str(", ");
            }
            enriched_hints.push_str(pt);
        }
    }
    // The plain tag merge (tagger + words from the description), used only if the text model fails.
    // It used to be returned whenever the 2nd Pass Visual Check was off, so the selected text model
    // never wrote the prompt in Combined mode. The check now only decides whether the image is
    // attached to the text model's request (see `describe_with_text_model`).
    let tag_merge = || {
        let tag_filter = AntigravityTagFilter::new();
        let tagged_prompt = tag_filter.process_tags(&enriched_hints);
        ImageAiResult { description: vision_desc.to_string(), prompt: tag_filter.resolve_and_strip_flags(&tagged_prompt) }
    };

    let candidates = crate::danbooru::list()
        .ok()
        .map(|l| {
            l.candidates(vision_desc, 4, 30)
                .iter()
                .map(|t| format!("{} ({})", t.name, t.count))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|c| !c.is_empty())
        .map(|c| format!("\n\n## Canonical Danbooru Candidates (from 50k post-count dictionary)\n{c}"))
        .unwrap_or_default();

    let system = image_system_prompt(&req.config, &enriched_hints);
    let second_pass_note = if !req.second_pass_vision { "" } else { "\n\n## 2nd Pass Visual Verification\nAn image is attached for direct visual verification. Cross-examine the visual description and tagger detections against the actual image to ensure exact hair, eyes, body build, bust size, clothing, and details are 100% accurate." };
    // The vision model's description is usually richer than what the writer kept (it dropped ToriiGate's
    // "two long locks", "wing-shaped hair ornaments", "jacket slipping off her shoulders", "white boots").
    let keep_details = "\n\n## Keep every detail\nTurn EVERY concrete visual detail in the description into tags: hair style \
         (locks, sidelocks, bangs, how it is tied), hair ornaments and their shape, how each garment is worn (open, off the \
         shoulders, unbuttoned), garment colors and materials, footwear type and color, where the arms and hands are, the \
         framing and the background. Do not summarize or drop details. Leave out character names, scores and notes.";
    let user_prompt = if enriched_hints.trim().is_empty() {
        format!(
            "## Visual description from vision encoder ({encoder_name}):\n{vision_desc}{candidates}{second_pass_note}{keep_details}\n\nWrite the prompt for this image."
        )
    } else {
        format!(
            "## Visual description from vision encoder ({encoder_name}):\n{vision_desc}{candidates}{second_pass_note}{keep_details}\n\n\
             ## Task\n\
             Synthesize the visual description and tagger detections into a complete, accurate, conflict-free Danbooru prompt according to the combination instructions."
        )
    };
    // The words found in the description are already in the hints the model saw; adding them again
    // after it answered overrode its choices (and brought back what it rightly left out).
    let mut res = match describe_with_text_model(req, &system, &user_prompt) {
        Ok(r) => r,
        Err(_) => tag_merge(),
    };
    // PC_NO_POLISH=1 turns the last pass off, to measure what it adds (testing only)
    if std::env::var_os("PC_NO_POLISH").is_none() {
        res.prompt = polish_image_prompt_with(&res.prompt, vision_desc, &req.strong_tags, &req.tag_hints);
    }
    Ok(res)
}

/// Encodes an in-memory DynamicImage to JPEG base64 string, scaling down if over 1024px.
fn encode_dynamic_image_b64(img: &image::DynamicImage) -> Result<String> {
    use base64::Engine as _;
    let small = if img.width().max(img.height()) > 1024 {
        img.resize(1024, 1024, image::imageops::FilterType::Triangle).to_rgb8()
    } else {
        img.to_rgb8()
    };
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85).encode_image(&small)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&out))
}

/// Extracts a visual description from an image file path using a vision encoder.
/// For single characters, performs an anatomical multi-zone scan (Scene + Upper + Middle + Lower zones)
/// so fine physical traits (ears, horns, expressions, belts, claws, paws, tails, footwear, chest/bust size) are enlarged and detected.
/// For multiple characters, performs a left-to-right positional scan so each character's position, features,
/// chest size, outfit, and pose are clearly delineated.
pub fn extract_visual_description_from_path(
    path: &std::path::Path,
    host: &str,
    mode: AiMode,
    encoder: &str,
) -> Result<String> {
    extract_visual_description_with_hints(path, host, mode, encoder, &[])
}

/// `extract_visual_description_from_path` with the tagger's high-confidence skin / species / count
/// tags (`strong_tags`), which the vision model is asked to confirm or correct instead of guessing.
pub fn extract_visual_description_with_hints(
    path: &std::path::Path,
    host: &str,
    mode: AiMode,
    encoder: &str,
    verify: &[String],
) -> Result<String> {
    let base = base_url(host)?;
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;

    // 1. Pass 1: Global scene & environment overview
    let scene_b64 = encode_dynamic_image_b64(&img)?;
    // Nothing used to ask for skin tone or what animal the ears and tail are, so the description never
    // said "dark skin" and called every pair of animal ears cat ears. The tagger's sure tags are given
    // to check, not to copy.
    let verify_note = if verify.is_empty() {
        String::new()
    } else {
        format!(
            " An image tagger detected these with high confidence: {}. Check each against the image and say whether it is right; correct it if not.",
            verify.join(", ")
        )
    };
    let scene_ask = format!(
        "Describe the image: the setting and background, the lighting, the camera framing (how much of the body is visible and from which side), \
         each character's skin tone, any animal features (ears, tail, horns: say which animal they belong to, judging by their shape and by the \
         tail's color and pattern), the hair (color, length, style), the clothing and the pose. Describe only what you can see.{verify_note}"
    );
    let scene_ask = scene_ask.as_str();
    let scene_desc = ask_vision_encoder(&base, mode, encoder, scene_b64, scene_ask)?;

    // The calls below run one after another: they get a total time budget and a cap, so a
    // crowded image can't turn one read into minutes of vision calls. The people come first
    // (they matter most); the quadrant decor pass only uses what time is left.
    let budget = ScanBudget::new(MULTIZONE_BUDGET);

    // 2. Per-person zone scan (Upper + Middle + Lower; one zone each for 3+ people)
    let boxes = crate::detector::detect(&img).unwrap_or_default();
    let mut char_sections = Vec::new();
    if !boxes.is_empty() {
        let count = boxes.len().clamp(1, 4);
        let chars = crate::detector::main_n(&boxes, count).unwrap_or_else(|| boxes.clone());

        let pos_labels: &[&str] = match count {
            1 => &["Character"],
            2 => &["Left Character", "Right Character"],
            3 => &["Left Character", "Center Character", "Right Character"],
            _ => &["Far-Left Character", "Center-Left Character", "Center-Right Character", "Far-Right Character"],
        };

        let upper_only = count >= 3;
        let mut zone_calls = 0usize;
        for (i, b) in chars.iter().enumerate() {
            let label = pos_labels.get(i).copied().unwrap_or("Character");
            let zones = crate::detector::three_zone_crops(&img, b);
            let mut zone_lines = Vec::new();

            for (z_name, z_img) in &zones {
                if upper_only && !z_name.contains("Upper") {
                    continue; // 3+ people: one call each (head, hair, upper outfit)
                }
                if zone_calls >= MULTIZONE_MAX_ZONE_CALLS || budget.spent() {
                    break;
                }
                zone_calls += 1;
                if let Ok(z_b64) = encode_dynamic_image_b64(z_img) {
                    let ask = if z_name.contains("Upper") {
                    format!(
                        "This crop shows the head of the {label}. Describe what you see: hair color, length and style, eye color, \
                         eyewear, ears or horns (which animal?), hair ornaments, skin tone and facial expression. Describe only what is \
                         actually there; don't mention things that are absent."
                    )
                } else if z_name.contains("Middle") {
                    format!(
                        "This crop shows the upper body of the {label}. Describe what you see: each garment (type, color, pattern, how it \
                         is worn), skin tone, arm positions and anything held. Describe only what is actually there; don't list things \
                         that are absent."
                    )
                } else {
                    format!(
                        "This crop shows the lower body of the {label}. Describe what you see: the clothing, legwear and footwear (type \
                         and color), a tail if there is one (its color and pattern), and the leg position. Describe only what is actually \
                         there; don't list things that are absent."
                    )
                };
                    let desc = ask_vision_encoder(&base, mode, encoder, z_b64, &ask).unwrap_or_default();
                    if !desc.trim().is_empty() {
                        zone_lines.push(format!("- **{}**: {}", z_name, desc.trim()));
                    }
                }
            }

            if !zone_lines.is_empty() {
                char_sections.push(format!("### Character: {}\n{}", label, zone_lines.join("\n")));
            }
        }
    }

    // 3. 4-Quadrant environmental objects & props (plants, furniture, lamps, windows, decor)
    let quads = crate::detector::quadrant_crops(&img);
    let mut quad_parts = Vec::new();
    for (name, q_img) in &quads {
        if budget.spent() {
            break; // out of time: the people above matter more than the decor
        }
        if let Ok(q_tags) = crate::tagger::tag_dynamic(q_img) {
            let triggers = crate::tagger::extract_zone_triggers(&q_tags, 0.40);
            let has_objects = triggers.iter().any(|t| crate::tagger::is_concrete_scene_object(t));
            if has_objects {
                if let Ok(q_b64) = encode_dynamic_image_b64(q_img) {
                    let q_ask = format!(
                        "In this {} crop, pinpoint and describe any visible background or environmental objects (specifically: plants, potted plants, flowers, trees, furniture, lamps, windows, mirrors, wall art, or props). Triggers detected: {}. Be concrete and concise.",
                        name,
                        triggers.join(", ")
                    );
                    if let Ok(desc) = ask_vision_encoder(&base, mode, encoder, q_b64, &q_ask) {
                        if !desc.trim().is_empty() {
                            quad_parts.push(format!("- **{} (Objects & Decor)**: {}", name, desc.trim()));
                        }
                    }
                }
            }
        }
    }
    let quad_section = if quad_parts.is_empty() {
        String::new()
    } else {
        format!("\n\n### Quadrant Environment & Objects (Plants, Decor & Props):\n{}", quad_parts.join("\n"))
    };


    if boxes.is_empty() {
        return Ok(format!(
            "### Scene & Environment (Spatial Overview):\n{}{}\n\n*Warning: No figure detected; processed full frame and scene quadrants only.*",
            scene_desc.trim(),
            quad_section
        ));
    }
    Ok(format!(
        "### Scene & Environment (Spatial Overview):\n{}{}\n\n{}",
        scene_desc.trim(),
        quad_section,
        char_sections.join("\n\n")
    ))
}

/// Sampling for one vision pass. The first try is near-deterministic (low temperature, seed 42);
/// `run_vlm`'s retry asks for a higher temperature, and gets it with a different seed, so the
/// retry is a real second attempt instead of the same request again.
fn vlm_sampling(temp: f32, first_try_temp: f32) -> (f32, u32) {
    let retry = temp > first_try_temp + 1e-3;
    if retry {
        (temp, 1042)
    } else {
        ((temp * 0.5).min(0.05), 42)
    }
}

/// Total time the multi-zone scan may spend on its extra (zone and quadrant) vision calls.
const MULTIZONE_BUDGET: Duration = Duration::from_secs(60);
/// After this long, a Deep Scan starts no more re-scans or follow-ups.
const DEEP_SCAN_BUDGET: Duration = Duration::from_secs(180);
/// Most per-person zone calls one multi-zone scan makes (2 people x 3 zones, or 1 each for 3-4).
const MULTIZONE_MAX_ZONE_CALLS: usize = 6;

/// A deadline for a run of model calls: `spent()` once the time is up.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScanBudget {
    deadline: std::time::Instant,
}

impl ScanBudget {
    pub(crate) fn new(limit: Duration) -> Self {
        Self { deadline: std::time::Instant::now() + limit }
    }
    pub(crate) fn spent(&self) -> bool {
        std::time::Instant::now() >= self.deadline
    }
}

/// One structured look at the whole image: a fixed checklist with one short answer per line, and the
/// clues stated before the conclusion for the error-prone items (animal, framing). Small vision models
/// guess less when they have to fill in a form than when asked to "describe" freely.
pub const OBSERVATION_CHECKLIST: &str = "Look at the image carefully and fill in this form. One short answer per line, \
only what you can actually see; write 'none' if something isn't there.\n\
PEOPLE: how many people\n\
SKIN: skin tone\n\
HAIR: color; length; style (bangs, locks, tied parts)\n\
EYES: eye color; eyewear\n\
EARS: describe their shape and color first, then say which animal they come from\n\
TAIL: describe its shape, color and pattern (stripes or rings?) first, then say which animal it comes from\n\
HEAD ACCESSORIES:\n\
TOP: garment type, color, sleeves, neckline\n\
OUTERWEAR: garment, color, and how it is worn (open, off the shoulders, tied...)\n\
BOTTOM: garment type, color, material\n\
LEGWEAR:\n\
FOOTWEAR: type, color, height\n\
OTHER ACCESSORIES:\n\
BODY: build and proportions\n\
POSE: stance; where the arms and hands are\n\
GAZE AND EXPRESSION:\n\
FRAMING: from where to where the body is visible (e.g. head to thighs), and from which side\n\
BACKGROUND: color and contents\n\
LIGHTING:";

/// True for ToriiGate (Minthy's anime captioning model), which needs its own prompt format.
pub fn is_toriigate(model: &str) -> bool {
    model.to_lowercase().replace(['-', '_'], "").contains("toriigate")
}

/// ToriiGate's official request (its `scripts/prompts.py`: the system prompt, the "min_structured_md"
/// captioning format, the image's booru tags, and "avoid to guess names"). Its card says other prompts are
/// "highly discouraged". On the raccoon-girl test image it read hair, ornaments, jacket, boots, pose and
/// framing correctly in 14 s, where the general models failed; its weakness is inventing character names.
pub fn observe_with_toriigate(path: &std::path::Path, host: &str, mode: AiMode, model: &str, booru_tags: &str) -> Result<String> {
    const SYSTEM: &str = "You are image captioning expert. Describe user's picture according to requested format and instructions.";
    const FORMAT: &str = "Your answer must contain 3 parts:\n<format>\n# 1. Thoughts about characters\n\
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.\n\
If no characters are listed in input - just write here \"No named characters\"\n# 2. Key details\n\
Here you need to write about the key details on image, prefere using regular text.\n# 3. Structured description\n## General\n\
Write about general composition, content of image, background and all things that are not related to characters directly.\n\
## Character name 1 (put here the name if any)\nWrite about datails and content related to specific character, including features, poses, look, used objects, interactions, and other things.\n\
## Character name 2 (put here the name if any)\nSame for each character.\n## Image effects\nMention image effect, style, camera angle\n</format>\nIn general stick to shorter descriptions.\n";
    let base = base_url(host)?;
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    let mut prompt = format!("# Captioning format:\n{FORMAT}\n");
    if !booru_tags.trim().is_empty() {
        prompt.push_str(&format!("# Booru tags for the image\n[{}]\n\n", booru_tags.trim()));
    }
    prompt.push_str("# Characters on picture:\nAvoid to guess names for characters.\n");
    let body = json!({
        "model": model,
        "system": SYSTEM,
        "prompt": prompt,
        "images": [encode_dynamic_image_b64(&img)?],
        "stream": false,
        "think": false,
        "keep_alive": "30s",
        "options": {
            "temperature": 0.2,
            "num_predict": 900,
            "num_ctx": 8192,
            "num_gpu": if mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });
    let resp = client(Some(Duration::from_secs(240)))?
        .post(format!("{base}/api/generate"))
        .json(&body)
        .send()
        .map_err(|e| anyhow!("Couldn't reach Ollama at {base}: {e}"))?;
    let gen: GenerateResponse = check(resp, "ToriiGate")?.json()?;
    Ok(gen.response.trim().to_string())
}

/// The vision step for Combined mode: ToriiGate's own format when it is the vision model, else the
/// `OBSERVATION_CHECKLIST` form.
pub fn observe_image(path: &std::path::Path, host: &str, mode: AiMode, encoder: &str, booru_tags: &str, verify: &[String]) -> Result<String> {
    if is_toriigate(encoder) {
        observe_with_toriigate(path, host, mode, encoder, booru_tags)
    } else {
        observe_with_checklist(path, host, mode, encoder, verify)
    }
}

/// Asks a vision model the `OBSERVATION_CHECKLIST` about a whole image, with the tagger's sure tags
/// (`strong_tags`) to confirm or correct. Used by Combined mode instead of the multi-zone scan: on the
/// test image it was more accurate (skin tone, striped tail → raccoon, boots, pose, framing, nothing
/// invented) and 3x faster (12 s vs 34 s) with the 3B.
pub fn observe_with_checklist(path: &std::path::Path, host: &str, mode: AiMode, encoder: &str, verify: &[String]) -> Result<String> {
    let base = base_url(host)?;
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    let ask = if verify.is_empty() {
        OBSERVATION_CHECKLIST.to_string()
    } else {
        format!(
            "{OBSERVATION_CHECKLIST}\nCHECK: an image tagger detected these with high confidence: {}. For each, say whether it is right.",
            verify.join(", ")
        )
    };
    ask_vision_encoder(&base, mode, encoder, encode_dynamic_image_b64(&img)?, &ask)
}

const COLOR_WORDS: &[&str] = &[
    "white", "black", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "grey", "gray", "silver",
    "gold", "golden", "light", "dark", "pale", "navy", "beige", "cream", "teal", "aqua", "violet", "magenta", "crimson",
];

/// The tags the checklist form states reliably: framing (how much of the body, from which side),
/// background and gaze. Deterministic, so they can't be forgotten by the writer (it kept dropping
/// `cowboy shot` and the background color). The pose line is left to the writer: the form gets it wrong
/// too often.
pub fn tags_from_form(form: &str) -> Vec<String> {
    // "LABEL: answer", where the answer may also continue on the following lines (bullets) up to
    // the next "LABEL:" line
    let lines: Vec<String> = form.lines().map(|l| l.trim().trim_start_matches(['*', '-', ' ']).replace("**", "")).collect();
    let is_label = |l: &str| l.split_once(':').is_some_and(|(k, _)| !k.is_empty() && k.chars().all(|c| c.is_ascii_uppercase() || c == ' '));
    let field = |name: &str| -> String {
        let Some(i) = lines.iter().position(|l| l.to_uppercase().starts_with(&format!("{name}:"))) else { return String::new() };
        let mut text = lines[i][name.len() + 1..].trim().to_string();
        for l in lines.iter().skip(i + 1).take_while(|l| !is_label(l)) {
            text.push(' ');
            text.push_str(l);
        }
        text.trim().to_lowercase()
    };
    let has = |text: &str, w: &str| crate::danbooru::prose_has_pub(text, w);
    let mut out: Vec<String> = Vec::new();
    let framing = field("FRAMING");
    if !framing.is_empty() {
        let to = framing.rsplit(" to ").next().unwrap_or(&framing).to_string();
        if has(&to, "feet") || has(&to, "toe") || has(&to, "shoe") || has(&framing, "full body") || has(&framing, "entire body") {
            out.push("full body".into());
        } else if has(&to, "thigh") || has(&to, "knee") || has(&to, "mid-thigh") {
            out.push("cowboy shot".into());
        } else if has(&to, "waist") || has(&to, "hip") || has(&to, "chest") {
            out.push("upper body".into());
        }
        if has(&framing, "behind") || has(&framing, "back view") || has(&framing, "from the back") {
            out.push("from behind".into());
        } else if has(&framing, "side") || has(&framing, "profile") {
            out.push("from side".into());
        } else if has(&framing, "above") {
            out.push("from above".into());
        } else if has(&framing, "below") {
            out.push("from below".into());
        }
    }
    let background = field("BACKGROUND");
    if !background.is_empty() {
        if ["plain", "solid", "simple", "blank", "no specific", "nothing", "empty"].iter().any(|w| background.contains(w)) {
            out.push("simple background".into());
        }
        if let Some(c) = ["grey", "gray", "white", "black", "blue", "pink", "yellow", "green", "red", "purple", "brown", "orange"]
            .iter()
            .find(|c| has(&background, c))
        {
            let c = if *c == "gray" { "grey" } else { c };
            out.push(format!("{c} background"));
        }
    }
    let gaze = field("GAZE AND EXPRESSION");
    if has(&gaze, "over her shoulder") || has(&gaze, "over his shoulder") || has(&gaze, "over shoulder") || has(&gaze, "looking back") {
        out.push("looking back".into());
    }
    if has(&gaze, "viewer") || has(&gaze, "camera") || has(&gaze, "looking forward") {
        out.push("looking at viewer".into());
    }
    if has(&gaze, "smile") || has(&gaze, "smiling") {
        out.push("smile".into());
    }
    out
}

/// Phrases models write that are no tag, mapped to the tag they mean ("" = drop it).
fn phrase_to_tag(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "standing sideways" | "side view" | "side profile" | "facing sideways" | "facing left" | "facing right" => "from side",
        "back view" | "facing away" | "facing away from viewer" | "standing away from viewer" => "from behind",
        "looking over shoulder" | "looking over her shoulder" => "looking back",
        "curvy hips" | "broad hips" => "wide hips",
        "medium length hair" | "medium-length hair" | "shoulder-length hair" | "shoulder length hair" => "medium hair",
        "smiling" | "gentle smile" | "slight smile" => "smile",
        "hands behind back" | "hands behind her back" => "arms behind back",
        "plain background" => "simple background",
        "off the shoulders" | "off the shoulder" | "off-shoulder" | "off shoulders" => "off shoulder",
        "plain grey background" | "plain gray background" => "grey background",
        "soft shadows" | "even lighting" | "athletic build" | "slim build" | "blue trim" | "visible tail" | "tail visible" => "",
        _ => return None,
    })
}

/// Checks each tag against the tagger's Danbooru list (`danbooru::local`, ~10,000 common tags): a known
/// tag stays, a near miss is corrected (`fix`), a color / material / style word in front of a known tag
/// stays (`black leather shorts`), a known phrase is mapped (`phrase_to_tag`), anything else multi-word
/// is dropped. Without the list everything is kept.
pub fn validate_tags(tags: Vec<String>) -> Vec<String> {
    let Some(list) = crate::danbooru::local() else { return tags };
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let l = t.to_lowercase();
        // weights "(tag:1.2)", LoRAs and dynamic groups pass untouched; a "Name: tags" label does not count
        if l.contains('<') || l.contains('{') || (l.contains('(') && l.contains(')')) {
            out.push(t);
            continue;
        }
        if let Some((label, rest)) = t.split_once(':') {
            // "Nagi: black hair ponytail" -> "black hair ponytail" (the name is a label, not a tag)
            if !label.contains(' ') && !rest.trim().is_empty() {
                out.extend(validate_tags(vec![rest.trim().to_string()]));
                continue;
            }
        }
        if let Some(m) = phrase_to_tag(&l) {
            if !m.is_empty() {
                out.push(m.to_string());
            }
            continue;
        }
        if list.get(&l).is_some() || matches!(l.as_str(), "masterpiece" | "best quality" | "score 9" | "score 8 up" | "score 7 up") {
            out.push(t);
            continue;
        }
        if let Some(fixed) = list.fix(&l) {
            out.push(fixed.name.clone());
            continue;
        }
        // Unknown multi-word tags are kept: the tag list is only the ~10,000 most common tags, and
        // descriptive ones ("white teal bikini top") are what Anima reads best. Dropping them emptied a
        // whole prompt in testing. Only phrases about nothing visible go.
        const NOT_VISUAL: &[&str] = &["visible", "build", "shadows", "trim", "tone", "tones", "accents", "expression", "vibe", "aesthetic"];
        if !l.split_whitespace().any(|w| NOT_VISUAL.contains(&w)) {
            out.push(t);
        }
    }
    out
}

/// Splits a tag the writer ran together without commas ("blue jacket open jacket off the shoulders")
/// into the known tags it is made of ("blue jacket", "open jacket", "off shoulder"). Only when the whole
/// phrase is not a tag itself and every word is covered by known tags (or `phrase_to_tag` phrases), using
/// as few pieces as possible, so descriptive tags ("very short black leather shorts") stay whole.
pub fn split_merged_tags(tag: String) -> Vec<String> {
    let Some(list) = crate::danbooru::local() else { return vec![tag] };
    let l = tag.to_lowercase();
    let words: Vec<&str> = l.split_whitespace().collect();
    if words.len() < 4 || list.get(&l).is_some() || l.contains(['(', ':', '<', '{']) {
        return vec![tag];
    }
    let piece = |i: usize, j: usize| -> Option<String> {
        let p = words[i..j].join(" ");
        if let Some(m) = phrase_to_tag(&p) {
            return (!m.is_empty()).then(|| m.to_string());
        }
        list.get(&p).map(|t| t.name.clone())
    };
    // fewest pieces covering every word (pieces of 1-5 words)
    let n = words.len();
    let mut best: Vec<Option<Vec<String>>> = vec![None; n + 1];
    best[0] = Some(Vec::new());
    for j in 1..=n {
        for i in j.saturating_sub(5)..j {
            if let (Some(prev), Some(p)) = (best[i].clone(), piece(i, j)) {
                let mut cand = prev;
                cand.push(p);
                if best[j].as_ref().is_none_or(|b| cand.len() < b.len()) {
                    best[j] = Some(cand);
                }
            }
        }
    }
    match best[n].take() {
        Some(parts) if parts.len() >= 2 => parts,
        _ => vec![tag],
    }
}

/// Hair tags that contradict a "short hair with long locks" description (twintails, long hair): the
/// tagger reads the long locks as twintails or long hair, and the writer copies both.
pub fn drop_contradicted_hair(tags: Vec<String>) -> Vec<String> {
    let lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    let short_with_locks = lower.iter().any(|t| t == "short hair with long locks")
        || (lower.iter().any(|t| t == "short hair") && lower.iter().any(|t| t.contains("sidelocks")));
    if !short_with_locks {
        return tags;
    }
    tags.into_iter()
        .filter(|t| !matches!(t.to_lowercase().as_str(), "twintails" | "low twintails" | "long hair" | "very long hair"))
        .collect()
}

/// One tag per body part where two can't both be true: eye color (unless heterochromia), skin tone,
/// hair length (unless the tags say long locks / sidelocks), boot height. Of two, a tag from the
/// Danbooru list beats an unknown one, then the first written wins.
pub fn one_per_part(tags: Vec<String>) -> Vec<String> {
    let list = crate::danbooru::local();
    let known = |t: &str| list.is_some_and(|l| l.get(t).is_some());
    let lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    let hetero = lower.iter().any(|t| t == "heterochromia");
    let part = |t: &str| -> Option<&'static str> {
        let first = t.split_whitespace().next().unwrap_or("");
        let colorish = COLOR_WORDS.contains(&first) || first.contains('-') || first.ends_with("ish");
        if t.ends_with(" eyes") && colorish && !hetero {
            return Some("eye color");
        }
        if t.ends_with(" skin") && t != "shiny skin" && t != "oily skin" {
            return Some("skin tone");
        }
        if matches!(t, "very short hair" | "short hair" | "medium hair" | "long hair" | "very long hair" | "absurdly long hair") {
            return Some("hair length");
        }
        if matches!(t, "thigh boots" | "knee boots" | "ankle boots" | "knee high boots" | "knee-high boots" | "thigh-high boots" | "thigh high boots") {
            return Some("boot height");
        }
        None
    };
    let mut chosen: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    for (i, t) in lower.iter().enumerate() {
        if let Some(p) = part(t) {
            match chosen.get(p) {
                None => {
                    chosen.insert(p, i);
                }
                Some(&j) if !known(&lower[j]) && known(t) => {
                    chosen.insert(p, i);
                }
                _ => {}
            }
        }
    }
    tags.into_iter()
        .enumerate()
        .filter(|(i, t)| part(&t.to_lowercase()).is_none_or(|p| chosen.get(p) == Some(i)))
        .map(|(_, t)| t)
        .collect()
}

/// The last pass over an image prompt written from a checklist form: the form's framing / background /
/// gaze tags are added, every tag is checked against the Danbooru list, one tag per body part is kept,
/// duplicates go, and the tagger's sure tags are enforced last.
/// `polish_image_prompt`, plus: a Danbooru character name stays only if the tagger's tags (`tagger_tags`)
/// name that character too. Vision models invent identities (ToriiGate called an OC "Musashi from Azur
/// Lane"); the tagger only reports characters it recognizes.
pub fn polish_image_prompt_with(prompt: &str, form: &str, strong: &[String], tagger_tags: &str) -> String {
    let tagger: std::collections::HashSet<String> = tagger_tags.split(',').map(crate::danbooru::key).collect();
    let mut tags: Vec<String> = prompt
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .filter(|t| !crate::danbooru::is_character(t) || tagger.contains(&crate::danbooru::key(t)))
        .collect();
    let mut tags: Vec<String> = tags.into_iter().flat_map(split_merged_tags).collect();
    for t in tags_from_form(form) {
        if !tags.iter().any(|x| crate::danbooru::key(x) == crate::danbooru::key(&t)) {
            tags.push(t);
        }
    }
    let tags = drop_contradicted_hair(one_per_part(validate_tags(tags)));
    let mut seen = std::collections::HashSet::new();
    let tags: Vec<String> = tags.into_iter().filter(|t| seen.insert(crate::danbooru::key(t))).collect();
    enforce_strong_tags(&tags.join(", "), strong)
}

/// Which vision model reads which pass: the mode's pass-1 prompt goes to the spatial /
/// body model, everything else to the identity / wardrobe model.
fn vision_model_for_prompt<'a>(
    mode: crate::prompt_condenser::Mode,
    prompt: &str,
    pass1_model: &'a str,
    pass2_model: &'a str,
) -> &'a str {
    if prompt == mode.pass1_prompt() { pass1_model } else { pass2_model }
}

/// A model that only reads images (Moondream, Qwen-VL, LLaVA…), as opposed to a text model
/// that happens to see (Gemma). Dedicated encoders are small and quick to reload.
pub fn is_dedicated_encoder(name: &str) -> bool {
    let l = name.to_lowercase();
    l.contains("moondream") || l.contains("llava") || l.contains("minicpm-v") || (l.contains("qwen") && l.contains("vl")) || is_toriigate(&l)
}

/// How long Ollama keeps an encoder loaded after a request: a dedicated encoder stays for
/// the burst of calls the pipeline makes in a row, then goes (it used to stay 30 minutes
/// next to the 12B text model, which is too much on a shared-memory GPU).
fn encoder_keep_alive(encoder: &str) -> &'static str {
    if is_dedicated_encoder(encoder) { "30s" } else { "30m" }
}

/// Unloads the dedicated vision encoders that are currently loaded (never the text model,
/// and never a model that isn't loaded: asking Ollama to unload one would load it first).
fn unload_dedicated_encoders(base: &str, encoders: &[&str], text_model: &str) {
    let Ok(http) = client(Some(Duration::from_secs(5))) else { return };
    let Ok(resp) = http.get(format!("{base}/api/ps")).send() else { return };
    let Ok(ps) = resp.json::<serde_json::Value>() else { return };
    let loaded: Vec<String> = ps["models"]
        .as_array()
        .map(|m| m.iter().filter_map(|x| x["name"].as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let mut done: Vec<&str> = Vec::new();
    for enc in encoders {
        if done.contains(enc) || !is_dedicated_encoder(enc) || same_model(enc, text_model) {
            continue;
        }
        done.push(enc);
        if loaded.iter().any(|l| same_model(l, enc)) {
            let _ = http.post(format!("{base}/api/generate")).json(&json!({ "model": enc, "keep_alive": 0 })).send();
        }
    }
}

/// Sends a single image to a vision encoder (like Moondream or Qwen 2.5-VL) to extract an objective description.
pub(crate) fn ask_vision_encoder(base: &str, mode: AiMode, encoder: &str, base64_image: String, ask: &str) -> Result<String> {
    let body = json!({
        "model": encoder,
        "prompt": ask,
        "images": [base64_image],
        "stream": false,
        "think": false,
        "keep_alive": encoder_keep_alive(encoder),
        "options": {
            "temperature": 0.05,
            "top_p": 0.85,
            "seed": 42,
            "repeat_penalty": 1.15,
            "num_predict": 500,
            "num_ctx": 4096,
            "num_gpu": if mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });
    let resp = client(Some(Duration::from_secs(120)))?
        .post(format!("{base}/api/generate"))
        .json(&body)
        .send()
        .map_err(|e| if e.is_timeout() { anyhow!("Vision encoder took over 2 minutes") } else { anyhow!("Couldn't reach Ollama at {base}") })?;
    let gen: GenerateResponse = check(resp, "Vision encoder reading image")?.json()?;
    let desc = gen.response.trim().to_string();
    if desc.is_empty() {
        return Err(anyhow!("The vision encoder returned an empty description."));
    }
    Ok(desc)
}

/// Formats a visual description into checkpoint-tailored prompt tags using a target text model.
fn describe_with_text_model(
    req: &ImageAiRequest,
    system: &str,
    user_prompt: &str,
) -> Result<ImageAiResult> {
    let base = base_url(&req.host)?;
    let text = json!({ "type": "string" });
    let mut body = json!({
        "model": req.model,
        "system": system,
        "prompt": user_prompt,
        "format": { "type": "object", "properties": { "description": text, "prompt": text }, "required": ["description", "prompt"] },
        "stream": false,
        "think": false,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.10,
            "top_p": 0.90,
            "repeat_penalty": 1.18,
            "num_predict": 900,
            "num_ctx": 16384,
            "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });
    if req.second_pass_vision {
        if let Ok(img_bytes) = shrink_for_vision(&req.image_path) {
            use base64::Engine as _;
            let b64 = base64::engine::general_purpose::STANDARD.encode(img_bytes);
            body["images"] = json!([b64]);
        }
    }
    let resp = client(Some(Duration::from_secs(300)))?
        .post(format!("{base}/api/generate"))
        .json(&body)
        .send()
        .map_err(|e| if e.is_timeout() { anyhow!("The model took over 5 minutes to write the prompt") } else { anyhow!("Couldn't reach Ollama at {base}") })?;
    let gen: GenerateResponse = check(resp, "Formatting the prompt")?.json()?;
    let mut out = parse_image_ai_result(gen.response.trim())?;
    clean_image_ai_prompt(&mut out, req.config.checkpoint);
    out.prompt = enforce_strong_tags(&out.prompt, &req.strong_tags);
    Ok(out)
}

/// Parses the model's reply into an ImageAiResult, handling code fences, surrounding text, or minor JSON flaws.
fn parse_image_ai_result(raw: &str) -> Result<ImageAiResult> {
    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        if end > start {
            if let Ok(r) = serde_json::from_str::<ImageAiResult>(&raw[start..=end]) {
                if !r.prompt.trim().is_empty() {
                    return Ok(r);
                }
            }
        }
    }
    // Fallback: extract "prompt" and "description" fields if JSON had trailing characters or formatting quirks
    let find_field = |field: &str| -> Option<String> {
        let pattern = format!("\"{field}\"");
        let idx = raw.find(&pattern)?;
        let rest = &raw[idx + pattern.len()..];
        let colon = rest.find(':')?;
        let after_colon = &rest[colon + 1..];
        let val_start = after_colon.find('"')? + 1;
        let body = &after_colon[val_start..];
        let mut escaped = false;
        let mut val_end = 0;
        for (i, c) in body.char_indices() {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                val_end = i;
                break;
            }
        }
        if val_end > 0 {
            Some(body[..val_end].to_string())
        } else {
            let trimmed = body.trim_end_matches(|c: char| c == ',' || c == '"' || c.is_whitespace());
            if !trimmed.is_empty() {
                Some(trimmed.to_string())
            } else {
                None
            }
        }
    };
    if let Some(prompt) = find_field("prompt") {
        if !prompt.trim().is_empty() {
            let description = find_field("description").unwrap_or_default();
            return Ok(ImageAiResult { description, prompt });
        }
    }
    Err(anyhow!("The model's reply was unreadable: {raw}"))
}

fn describe_image_with(req: &ImageAiRequest, system: &str, ask: &str) -> Result<ImageAiResult> {
    use base64::Engine as _;
    let base = base_url(&req.host)?;
    let image = base64::engine::general_purpose::STANDARD.encode(shrink_for_vision(&req.image_path)?);
    let text = json!({ "type": "string" });
    let body = json!({
        "model": req.model,
        "system": system,
        "prompt": ask,
        "images": [image],
        "format": { "type": "object", "properties": { "description": text, "prompt": text }, "required": ["description", "prompt"] },
        "stream": false,
        "think": false,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.10,
            "top_p": 0.90,
            "repeat_penalty": 1.18,
            "num_predict": 900,
            "num_ctx": 16384,
            "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });
    let resp = client(Some(Duration::from_secs(300)))?
        .post(format!("{base}/api/generate"))
        .json(&body)
        .send()
        .map_err(|e| if e.is_timeout() { anyhow!("The model took over 5 minutes to read the image") } else { anyhow!("Couldn't reach Ollama at {base}") })?;
    let gen: GenerateResponse = check(resp, "Reading the image")?.json()?;
    let mut out = parse_image_ai_result(gen.response.trim())?;
    clean_image_ai_prompt(&mut out, req.config.checkpoint);
    out.prompt = enforce_strong_tags(&out.prompt, &req.strong_tags);
    Ok(out)
}

fn clean_image_ai_prompt(out: &mut ImageAiResult, checkpoint: crate::profiles::CheckpointProfile) {
    out.prompt = strip_placeholders(&strip_instruction_leaks(&flatten_section_layout(&out.prompt)));
    out.prompt = AntigravityTagFilter::new().resolve_and_strip_flags(&out.prompt);
    // small cleanups: the checkpoint's name and bare category words aren't image content
    let name = checkpoint.display_name().to_lowercase();
    let junk = ["hair", "eyes", "lighting", "light", "pose", "expression", "clothing", "background", "style", "setting"];
    let cleaned: Vec<String> = crate::condenser::tidy_ai_prompt(&out.prompt, checkpoint)
        .split(", ")
        .map(str::to_string)
        .filter(|t| {
            let l = t.to_lowercase();
            !l.is_empty() && l != name && !junk.contains(&l.as_str()) && !l.ends_with("_flag") && !l.contains("_flag")
        })
        .collect();
    // "solo" and 1girl/1boy contradict a group
    let group = cleaned.iter().any(|t| {
        let l = t.to_lowercase();
        l.starts_with("multiple ") || (l.len() > 1 && l.as_bytes()[0].is_ascii_digit() && l.as_bytes()[0] > b'1' && (l.ends_with("girls") || l.ends_with("boys") || l.ends_with("others")))
    });
    let mut kept: Vec<String> = Vec::new();
    for t in cleaned.into_iter().filter(|t| {
        if group {
            let l = t.to_lowercase();
            if l == "solo" || l == "1girl" || l == "1boy" || l == "1other" {
                return false;
            }
        }
        true
    }) {
        let n = crate::danbooru::key(&t);
        // Only strip duplicate size attributes on single-character prompts; multi-character groups can have alternate breast/body sizes
        if !group && kept.iter().any(|k| crate::analyzer::size_conflict(&crate::danbooru::key(k), &n)) {
            continue;
        }
        kept.push(t);
    }
    out.prompt = kept.join(", ");
}

// ============================================================================
// Two-character image: the vision model checks and fixes the detector + tagger split
// ============================================================================

pub struct TwoImageRequest {
    pub host: String,
    pub model: String,
    pub vision_encoder: Option<String>,
    pub mode: AiMode,
    pub split: crate::twoimg::TagSplit,
    /// The tag-only split, given to the model as hints.
    pub hints: crate::twoimg::SplitPrompt,
    /// If true and the target text model has vision capability, directly attach image pixels for a 2nd pass visual check.
    pub second_pass_vision: bool,
    /// If true, extract body shape/build from the scene reference for the character.
    #[allow(dead_code)]
    pub scene_body_shape: bool,
}

pub fn split_image(req: &TwoImageRequest) -> Result<(crate::twoimg::SplitPrompt, String)> {
    if let Some(ref encoder) = req.vision_encoder {
        if !same_model(encoder, &req.model) {
            return split_image_pipelined(req, encoder);
        }
    }
    split_image_direct(req)
}

fn split_image_pipelined(req: &TwoImageRequest, encoder: &str) -> Result<(crate::twoimg::SplitPrompt, String)> {
    use crate::twoimg::BOX_COLORS;
    let n = req.split.wanted.clamp(2, 4);
    let words = ["", "", "two", "three", "four"];
    let base = base_url(&req.host)?;

    // 1. Describe each character crop and the overall scene with the vision encoder (one image per call)
    let mut vision_parts = Vec::new();
    if let Some(ref boxes) = req.split.boxes {
        for (i, b) in boxes.iter().take(n).enumerate() {
            let zones = crate::detector::three_zone_crops(&req.split.image, b);
            let mut zone_lines = Vec::new();

            for (z_name, z_img) in &zones {
                if let Ok(z_b64) = encode_dynamic_image_b64(z_img) {
                    let ask = if z_name.contains("Upper") {
                        format!("Describe head, hair color and style, specific hairpins/hair ornaments, eyewear, earrings, choker, and expression for character {}.", i + 1)
                    } else if z_name.contains("Middle") {
                        format!("Describe torso, neckline (verify cleavage vs closed top), chest proportions, clothing cut/color, arm pose, and any items held for character {}.", i + 1)
                    } else {
                        format!("Describe lower body, skirt/shorts/pants, leg posture, thigh straps, and footwear (shoes, socks, barefoot) for character {}.", i + 1)
                    };
                    let desc = ask_vision_encoder(&base, req.mode, encoder, z_b64, &ask).unwrap_or_default();
                    if !desc.trim().is_empty() {
                        zone_lines.push(format!("- **{}**: {}", z_name, desc.trim()));
                    }
                }
            }

            vision_parts.push(format!(
                "### Character {} ({} box):\n{}",
                i + 1, BOX_COLORS[i].1, zone_lines.join("\n")
            ));
        }
        let scene_b64 = crate::twoimg::encode(&req.split.image, 768)?;
        let ask_scene = "Describe the camera angle, framing, setting, background, lighting, and the exact spatial arrangement of characters from left to right (who is on the left, center, right) and how they interact. Do not describe character outfits or hair.";
        let scene_desc = ask_vision_encoder(&base, req.mode, encoder, scene_b64, ask_scene)
            .unwrap_or_else(|e| format!("Scene: could not extract ({e})"));
        vision_parts.push(format!("### Scene & Interactions:\n{}", scene_desc));
    } else if let Some(ref chars) = req.split.chars {
        for (i, (crop, _)) in chars.iter().take(n).enumerate() {
            let b64 = crate::twoimg::encode(crop, 768)?;
            let ask = format!(
                "Describe character {} in objective detail: gender, hair color and style, eyes, face, body build, chest/breast size (e.g. flat chest, small, medium, large, or huge breasts), clothing or nudity, pose, and expression.",
                i + 1
            );
            let desc = ask_vision_encoder(&base, req.mode, encoder, b64, &ask)
                .unwrap_or_else(|e| format!("Character {}: could not extract ({e})", i + 1));
            vision_parts.push(format!("### Character {} ({} box):\n{}", i + 1, BOX_COLORS[i].1, desc));
        }
        let scene_b64 = crate::twoimg::encode(&req.split.image, 768)?;
        let ask_scene = "Describe the camera angle, framing, setting, background, lighting, and the exact spatial arrangement of characters from left to right (who is on the left, center, right) and how they interact. Do not describe character outfits or hair.";
        let scene_desc = ask_vision_encoder(&base, req.mode, encoder, scene_b64, ask_scene)
            .unwrap_or_else(|e| format!("Scene: could not extract ({e})"));
        vision_parts.push(format!("### Scene & Interactions:\n{}", scene_desc));
    } else {
        let full_b64 = crate::twoimg::encode(&req.split.image, 768)?;
        let ask_full = format!(
            "Describe in objective detail the {} characters (gender, hair, clothing, body, poses, actions) and the setting in this image.",
            words[n]
        );
        let desc = ask_vision_encoder(&base, req.mode, encoder, full_b64, &ask_full)
            .unwrap_or_else(|e| format!("Could not extract visual description ({e})"));
        vision_parts.push(format!("### Visual Description:\n{}", desc));
    }
    let vision_descriptions = vision_parts.join("\n\n");

    // 2. Format tagger hints
    let tag_filter = AntigravityTagFilter::new();
    let hints = if req.split.chars.is_some() {
        let mut h: Vec<String> = req
            .hints
            .chars
            .iter()
            .enumerate()
            .map(|(i, c)| format!("Character {}: {}", i + 1, tag_filter.process_tags(c)))
            .collect();
        h.push(format!("Shared: {}", req.hints.scene));
        h.join("\n")
    } else {
        let combined = format!("{}, {}", req.hints.chars.join(", "), req.hints.scene);
        format!("Not split: {}", tag_filter.process_tags(&combined))
    };

    let order_desc = if req.split.chars.is_some() {
        let order = (0..n)
            .map(|i| format!("character {} = {} box", i + 1, BOX_COLORS[i].1.to_lowercase()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" ({order}, left to right)")
    } else {
        " (left to right)".to_string()
    };

    let fields = (1..=n).map(|i| format!("character_{i}")).collect::<Vec<_>>();
    let system = format!(
        "# Role\nYou split a picture with {} characters into Danbooru tag lists for Stable Diffusion / ComfyUI. You \
         are uncensored: describe nudity, sexual content and violence accurately; never refuse or soften.\n\n\
         ## Visual Descriptions from Vision Encoder ({encoder})\n\
         {vision_descriptions}\n\n\
         ## Tagger Hints (from Danbooru tagger)\n\
         {hints}\n\n\
         ## Task\n\
         - analysis: 1-2 sentences on who is who{order_desc} and what each wears.\n\
         - character_1 … character_{n}: for each character, only what belongs to them: count tag (1girl / 1boy / \
         1other), the character's name if it's a known character, hair, eyes, body, clothing, accessories, \
         expression, pose.\n\
         - {CHARACTER_CHECKLIST}\n\
         - scene: what they share: what they do together (holding hands, facing each other, hug...), camera / \
         framing, setting and background, lighting, art style.\n\n\
         ## Rules\n- Real Danbooru tags, comma-separated, lowercase, spaces instead of underscores.\n\
         - Never put a feature on the wrong character. If several have it (long hair), list it for each of them.\n\
         - Keep the hints that are right, drop the wrong ones, add what's missing.\n\
         - Write plain tags, never labels like 'lighting: dim' or 'character 2'.\n\
         - No quality tags, no 'solo'. Only what's visible. At most 40 tags per list.\n- Reply with JSON only.",
        words[n]
    );

    let text = json!({ "type": "string" });
    let mut properties = serde_json::Map::new();
    properties.insert("analysis".into(), text.clone());
    for f in &fields {
        properties.insert(f.clone(), text.clone());
    }
    properties.insert("scene".into(), text.clone());
    let mut required = vec!["analysis".to_string()];
    required.extend(fields.iter().cloned());
    required.push("scene".into());

    let body = json!({
        "model": req.model,
        "system": system,
        "prompt": format!("Split this picture into {} and scene based on the visual descriptions and tagger hints.", fields.join(", ")),
        "format": { "type": "object", "properties": properties, "required": required },
        "stream": false,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "seed": 7,
            "repeat_penalty": 1.15,
            "num_predict": 1200 * n as i32,
            "num_ctx": 16384,
            "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });

    let mut last_err = anyhow!("The model's reply was unreadable. Try again.");
    let mut body = body;
    if req.second_pass_vision {
        if let Ok(b64) = crate::twoimg::encode(&req.split.image, 1024) {
            body["images"] = json!([b64]);
        }
    }
    for attempt in 0..2u32 {
        body["options"]["seed"] = json!(7 + attempt);
        body["options"]["temperature"] = json!(if attempt == 0 { 0.0 } else { 0.3 });
        match split_once(&base, &body, n) {
            Ok(r) => return Ok(r),
            Err(e) if e.to_string().starts_with("Couldn't reach") => return Err(e),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

fn split_image_direct(req: &TwoImageRequest) -> Result<(crate::twoimg::SplitPrompt, String)> {
    use crate::twoimg::{encode, BOX_COLORS};
    let n = req.split.wanted.clamp(2, 4);
    let words = ["", "", "two", "three", "four"];
    let base = base_url(&req.host)?;
    let mut images = vec![encode(&crate::twoimg::marked(&req.split), 1024)?];
    let layout = match &req.split.chars {
        Some(chars) => {
            for (crop, _) in chars {
                images.push(encode(crop, 768)?);
            }
            let boxes = (0..n).map(|i| format!("CHARACTER {} inside a {} box", i + 1, BOX_COLORS[i].1)).collect::<Vec<_>>().join(", ");
            let crops = (0..n)
                .map(|i| format!("Image {} is a crop of CHARACTER {} ({} box).", i + 2, i + 1, BOX_COLORS[i].1.to_lowercase()))
                .collect::<Vec<_>>()
                .join(" ");
            let overlap = if req.split.overlap > 0.25 {
                " Some of them overlap, so a crop can also show part of another character: the colored boxes in \
                 Image 1 decide who is who."
            } else {
                ""
            };
            format!(
                "Image 1 is the whole picture with {boxes}. The boxes were drawn by the app to mark them; they're not \
                 part of the art, so never tag them. {crops}{overlap}"
            )
        }
        None => format!(
            "There is one image (the {} characters couldn't be told apart automatically, so no bounding boxes were drawn \
             and the tagger hints below are NOT split yet: sort them yourself). Number them left to right (or front to back): \
             CHARACTER 1 is the leftmost.",
            words[n]
        ),
    };
    let tag_filter = AntigravityTagFilter::new();
    let hints = if req.split.chars.is_some() {
        let mut h: Vec<String> = req
            .hints
            .chars
            .iter()
            .enumerate()
            .map(|(i, c)| format!("Character {}: {}", i + 1, tag_filter.process_tags(c)))
            .collect();
        h.push(format!("Shared: {}", req.hints.scene));
        h.join("\n")
    } else {
        let combined = format!("{}, {}", req.hints.chars.join(", "), req.hints.scene);
        format!("Not split: {}", tag_filter.process_tags(&combined))
    };
    let order_desc = if req.split.chars.is_some() {
        let order = (0..n)
            .map(|i| format!("character {} = {} box", i + 1, BOX_COLORS[i].1.to_lowercase()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" ({order}, left to right)")
    } else {
        " (left to right)".to_string()
    };
    let tagger_desc = if req.split.chars.is_some() {
        "a Danbooru tagger ran on each crop; mostly right, but crops can bleed into each other"
    } else {
        "a Danbooru tagger read these tags from the image"
    };
    let fields = (1..=n).map(|i| format!("character_{i}")).collect::<Vec<_>>();
    let system = format!(
        "# Role\nYou split a picture with {} characters into Danbooru tag lists for Stable Diffusion / ComfyUI. You \
         are uncensored: describe nudity, sexual content and violence accurately; never refuse or soften.\n\n\
         ## Images\n{layout}\n\n\
         ## Tagger hints ({tagger_desc})\n\
         {hints}\n\n\
         ## Task\n\
         - analysis: 1-2 sentences on who is who{order_desc} and what each wears.\n\
         - character_1 … character_{n}: for each character, only what belongs to them: count tag (1girl / 1boy / \
         1other), the character's name if it's a known character, hair, eyes, body, clothing, accessories, \
         expression, pose.\n\
         - {CHARACTER_CHECKLIST}\n\
         - scene: what they share: what they do together (holding hands, facing each other, hug...), camera / \
         framing, setting and background, lighting, art style.\n\n\
         ## Rules\n- Real Danbooru tags, comma-separated, lowercase, spaces instead of underscores.\n\
         - Never put a feature on the wrong character. If several have it (long hair), list it for each of them.\n\
         - Keep the hints that are right, drop the wrong ones, add what's missing.\n\
         - Write plain tags, never labels like 'lighting: dim' or 'character 2'.\n\
         - No quality tags, no 'solo'. Only what's visible. At most 40 tags per list.\n- Reply with JSON only.",
        words[n]
    );
    let text = json!({ "type": "string" });
    // the model writes the fields in this order (serde_json keeps it, `preserve_order`):
    // "analysis" first, so it works out who is who before listing tags
    let mut properties = serde_json::Map::new();
    properties.insert("analysis".into(), text.clone());
    for f in &fields {
        properties.insert(f.clone(), text.clone());
    }
    properties.insert("scene".into(), text.clone());
    let mut required = vec!["analysis".to_string()];
    required.extend(fields.iter().cloned());
    required.push("scene".into());
    let body = json!({
        "model": req.model,
        "system": system,
        "prompt": format!("Split this picture into {} and scene.", fields.join(", ")),
        "images": images,
        "format": { "type": "object", "properties": properties, "required": required },
        "stream": false,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "seed": 7,
            "repeat_penalty": 1.15,
            "num_predict": 1500 * n as i32,
            "num_ctx": 12288 + 2048 * n as i32,
            "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });
    // a model stuck repeating a tag gets stopped by Ollama; a second try usually works
    let mut last_err = anyhow!("The model's reply was unreadable. Try again.");
    let mut body = body;
    for attempt in 0..2u32 {
        body["options"]["seed"] = json!(7 + attempt);
        // a retry after a repetition loop needs a little randomness to get out of it
        body["options"]["temperature"] = json!(if attempt == 0 { 0.0 } else { 0.3 });
        match split_once(&base, &body, n) {
            Ok(r) => return Ok(r),
            Err(e) if e.to_string().starts_with("Couldn't reach") => return Err(e),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

fn split_once(base: &str, body: &serde_json::Value, n: usize) -> Result<(crate::twoimg::SplitPrompt, String)> {
    use crate::twoimg::clean_list;
    let resp = client(Some(Duration::from_secs(300)))?
        .post(format!("{base}/api/generate"))
        .json(body)
        .send()
        .map_err(|e| if e.is_timeout() { anyhow!("The model took over 5 minutes to read the image") } else { anyhow!("Couldn't reach Ollama at {base}") })?;
    let gen: GenerateResponse = check(resp, "Reading the image")?.json()?;
    let raw = gen.response.trim();
    let body = &raw[raw.find('{').unwrap_or(0)..];
    // a long reply can hit the output limit: keep what was fully written
    let v: serde_json::Value = serde_json::from_str(body)
        // flat object of strings: close the open string (dropping its half-written last tag) and the object
        .or_else(|_| serde_json::from_str(&format!("{}\"}}", body.rsplit_once(',').map(|(a, _)| a).unwrap_or(body))))
        .or_else(|_| serde_json::from_str(&format!("{body}\"}}")))
        .map_err(|_| anyhow!("The model's reply was unreadable. Try again."))?;
    let get = |k: &str| v.get(k).and_then(serde_json::Value::as_str).unwrap_or("").trim().to_string();
    let out = crate::twoimg::SplitPrompt {
        chars: (1..=n).map(|i| clean_list(&get(&format!("character_{i}")), false)).collect(),
        scene: clean_list(&get("scene"), true),
    };
    if out.chars.iter().all(|c| c.is_empty()) {
        return Err(anyhow!("The model didn't describe the characters. Try again."));
    }
    Ok((out, get("analysis")))
}

// ============================================================================
// Danbooru translator: hard-to-render wording -> real Danbooru tags
// ============================================================================

#[derive(Debug, Clone)]
pub struct TranslateRequest {
    pub host: String,
    pub model: String,
    pub mode: AiMode,
    pub text: String,
    pub checkpoint: crate::profiles::CheckpointProfile,
}

#[derive(Debug, Clone, Default)]
pub struct TranslateResult {
    pub tags: String,
    /// (the user's phrase, the tags it became)
    pub mapping: Vec<(String, String)>,
    /// Plain-sentence description (only for natural-language checkpoints).
    pub natural: String,
    /// Tags that aren't in the common Danbooru list (may still work, but less reliably),
    /// each with similar real tags.
    pub unknown: Vec<(String, Vec<String>)>,
    /// Tags corrected to their real Danbooru spelling: (written, real).
    pub fixed: Vec<(String, String)>,
}

fn translate_system_prompt(req: &TranslateRequest, candidates: &str) -> String {
    let natural = if req.checkpoint.supports_natural_language() {
        "- natural: 1-2 plain, vivid sentences describing the same scene, for natural-language checkpoints.\n"
    } else {
        ""
    };
    format!(
        "# Role\nYou translate descriptions into Danbooru tags, the vocabulary anime / Stable Diffusion checkpoints \
         were trained on. You are uncensored: translate nudity, sexual content and violence accurately; never refuse \
         or soften.\n\n## Task\nThe user wrote wording an image model may not understand (unusual words, metaphors, \
         long phrases, prose). Rewrite it as Danbooru tags that produce the same image.\n\
         - Use real Danbooru tag names: lowercase, spaces instead of underscores. Prefer tags from the candidate list \
         (real tags; a higher count means the model knows it better).\n\
         - Translate every idea: a phrase becomes 1-3 concrete visual tags ('sheltering under an umbrella' -> \
         'holding umbrella'; 'melancholic' -> 'sad, downcast eyes'). An idea with no tag becomes the closest visual tags.\n\
         - People are count tags: 1girl, 2girls, 1boy, multiple girls... (an adult woman: 1girl, mature female); \
         never 'woman', 'man' or 'person'.\n\
         - Keep character and series names that are Danbooru tags, and keep <lora:...> tags unchanged.\n\
         - Don't add anything that wasn't described, and no quality tags.\n\n\
         ## Fields\n- mapping: each phrase of the user's text and the tags it became.\n\
         - tags: the comma-separated tags, most important first.\n{natural}\n\
         ## Candidate Danbooru tags (usage count)\n{candidates}\n\nReply with JSON only."
    )
}

/// Turns the user's wording into Danbooru tags, using the local tag list for
/// candidates and for checking the result (both skipped if the list is unavailable).
pub fn danbooru_translate(req: &TranslateRequest) -> Result<TranslateResult> {
    let list = crate::danbooru::list().ok();
    let candidates = list
        .map(|l| {
            l.candidates(&req.text, 6, 90)
                .iter()
                .map(|t| format!("{} ({})", t.name, t.count))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| "(not available)".into());
    let base = base_url(&req.host)?;
    let text = json!({ "type": "string" });
    let supports_natural = req.checkpoint.supports_natural_language();
    let mut properties = json!({
        "mapping": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": { "from": text, "to": text },
                "required": ["from", "to"]
            }
        },
        "tags": text,
    });
    let mut required = vec!["mapping".to_string(), "tags".to_string()];
    if supports_natural {
        if let Some(obj) = properties.as_object_mut() {
            obj.insert("natural".into(), text.clone());
        }
        required.push("natural".into());
    }
    let body = json!({
        "model": req.model,
        "system": translate_system_prompt(req, &candidates),
        "prompt": req.text,
        "format": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        "stream": false,
        "think": false,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "num_predict": 1200,
            "num_ctx": 16384,
            "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> },
        },
    });
    let mut resp = client(Some(Duration::from_secs(300)))?
        .post(format!("{base}/api/generate"))
        .json(&body)
        .send()
        .map_err(|_| anyhow!("Couldn't reach Ollama at {base}"))?;
    if resp.status().as_u16() == 400 {
        // models that reject the think setting
        let mut body = body.clone();
        body.as_object_mut().expect("object").remove("think");
        resp = client(Some(Duration::from_secs(300)))?
            .post(format!("{base}/api/generate"))
            .json(&body)
            .send()
            .map_err(|_| anyhow!("Couldn't reach Ollama at {base}"))?;
    }
    let gen: GenerateResponse = check(resp, "Translating")?.json()?;
    let raw = gen.response.trim();
    let v: serde_json::Value = serde_json::from_str(&raw[raw.find('{').unwrap_or(0)..])
        .map_err(|_| anyhow!("The model's reply was unreadable. Try again."))?;
    let s = |k: &str| v.get(k).and_then(serde_json::Value::as_str).unwrap_or("").trim().to_string();
    let mut out = TranslateResult {
        natural: if req.checkpoint.supports_natural_language() { s("natural") } else { String::new() },
        mapping: v
            .get("mapping")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| Some((m.get("from")?.as_str()?.trim().to_string(), m.get("to")?.as_str()?.trim().to_string())))
                    .filter(|(f, t)| !f.is_empty() && !t.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        ..Default::default()
    };
    let mut tags: Vec<String> = Vec::new();
    for t in s("tags").split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let tag = match list {
            Some(l) if !t.starts_with('<') && l.get(t).is_none() => match l.fix(t) {
                Some(real) => {
                    out.fixed.push((t.to_string(), real.name.clone()));
                    real.name.clone()
                }
                None => {
                    let similar = l.similar(t, 3).iter().map(|c| c.name.clone()).collect();
                    out.unknown.push((t.to_string(), similar));
                    t.to_string()
                }
            },
            _ => t.to_string(),
        };
        if !tags.iter().any(|x| crate::danbooru::key(x) == crate::danbooru::key(&tag)) {
            tags.push(tag);
        }
    }
    if tags.is_empty() {
        return Err(anyhow!("The model returned no tags. Try again or reword it."));
    }
    // parentheses in tag names would read as weights in ComfyUI
    out.tags = tags.iter().map(|t| if t.starts_with('<') { t.clone() } else { t.replace('(', "\\(").replace(')', "\\)") }).collect::<Vec<_>>().join(", ");
    Ok(out)
}

// ============================================================================
// Ollama client
// ============================================================================

pub const OLLAMA_DEFAULT_PORT: u16 = 11434;

/// Turns whatever the user typed into a base URL like `http://192.168.1.50:11434`.
/// Accepts `192.168.1.50`, `192.168.1.50:11434`, `http://host:port/`, `host/api/tags`, etc.
pub fn normalize_ollama_url(input: &str) -> Option<String> {
    let mut s = input.trim().trim_end_matches('/').to_string();
    if s.is_empty() {
        return None;
    }
    let scheme = if let Some(rest) = s.strip_prefix("https://") {
        s = rest.to_string();
        "https"
    } else if let Some(rest) = s.strip_prefix("http://") {
        s = rest.to_string();
        "http"
    } else {
        "http"
    };
    // Drop any path (e.g. "/api/tags")
    let host_port = s.split('/').next().unwrap_or("").trim();
    if host_port.is_empty() || host_port.contains(char::is_whitespace) {
        return None;
    }
    // Bracketed IPv6 "[::1]:port" or "[::1]"
    let has_port = if host_port.starts_with('[') {
        host_port.contains("]:")
    } else {
        host_port.matches(':').count() == 1
    };
    if has_port {
        let port = host_port.rsplit(':').next().unwrap_or("");
        if port.parse::<u16>().is_err() {
            return None;
        }
        Some(format!("{scheme}://{host_port}"))
    } else {
        Some(format!("{scheme}://{host_port}:{OLLAMA_DEFAULT_PORT}"))
    }
}

/// The chat template a tier is registered with (None = keep the GGUF's own).
/// All tiers are Gemma 4, whose GGUFs carry a template Ollama understands.
fn tier_template(_tier: AiModelTier) -> Option<&'static str> {
    None
}

fn client(timeout: Option<Duration>) -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(timeout)
        .build()?)
}

fn base_url(host: &str) -> Result<String> {
    normalize_ollama_url(host)
        .ok_or_else(|| anyhow!("'{}' is not a valid Ollama host (try localhost:11434)", host))
}

/// True for hosts on this machine, where we can suggest starting the service.
pub fn is_local_host(host: &str) -> bool {
    let Some(base) = normalize_ollama_url(host) else { return false };
    let h = base.split("://").nth(1).unwrap_or("");
    h.starts_with("localhost:") || h.starts_with("127.") || h.starts_with("[::1]")
}

/// Ollama reports `name` as `model:latest` when no tag was given.
pub fn same_model(a: &str, b: &str) -> bool {
    let strip = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_string();
    strip(a) == strip(b)
}

/// Ollama answers errors as `{"error": "..."}`; surface that text.
fn check(resp: reqwest::blocking::Response, what: &str) -> Result<reqwest::blocking::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let body = resp.text().unwrap_or_default();
    let msg = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
        .unwrap_or(body);
    Err(anyhow!("{what} failed (HTTP {status}): {}", msg.trim()))
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelEntry {
    #[serde(default)]
    name: String,
    /// e.g. ["completion", "vision"]
    #[serde(default)]
    capabilities: Vec<String>,
}

/// An installed Ollama model.
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaModel {
    pub name: String,
    /// Can read images (the GGUF has a vision projector).
    pub vision: bool,
}

/// `GET /api/tags` with a 2-second timeout. Returns the base URL and installed models.
pub fn list_ollama_models(host: &str) -> Result<(String, Vec<OllamaModel>)> {
    let base = base_url(host)?;
    let resp = client(Some(Duration::from_secs(2)))?
        .get(format!("{base}/api/tags"))
        .send()
        .with_context(|| format!("No Ollama server answered at {base}"))?;
    let tags: OllamaTagsResponse = check(resp, "Listing models")?
        .json()
        .with_context(|| format!("{base} answered, but not like an Ollama server"))?;
    let mut models: Vec<OllamaModel> = tags
        .models
        .into_iter()
        .filter(|m| !m.name.is_empty())
        .map(|m| OllamaModel { vision: m.capabilities.iter().any(|c| c == "vision"), name: m.name })
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((base, models))
}

#[derive(Debug, Deserialize)]
struct PullProgress {
    #[serde(default)]
    status: String,
    #[serde(default)]
    total: u64,
    #[serde(default)]
    completed: u64,
    #[serde(default)]
    error: Option<String>,
}

/// Downloads a pre-packaged tier through Ollama (`/api/pull` of the Hugging Face GGUF),
/// then registers it under the app's short name with the ChatML template.
pub fn pull_tier_model(host: &str, tier: AiModelTier, progress: Sender<String>) -> Result<()> {
    pull_model(host, tier.hf_ref(), tier.ollama_name(), tier.label(), tier_template(tier), progress)
}

fn pull_model(
    host: &str,
    source: &str,
    name: &str,
    label: &str,
    template: Option<&str>,
    progress: Sender<String>,
) -> Result<()> {
    let base = base_url(host)?;
    let http = client(None)?;

    let resp = http
        .post(format!("{base}/api/pull"))
        .json(&json!({ "model": source, "stream": true }))
        .send()
        .with_context(|| format!("Lost connection to Ollama at {base}"))?;
    let resp = check(resp, "Download")?;

    let mut last_pct = u64::MAX;
    for line in BufReader::new(resp).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let p: PullProgress = serde_json::from_str(&line)?;
        if let Some(err) = p.error {
            return Err(anyhow!("Ollama: {err}"));
        }
        if let Some(pct) = (p.completed * 100).checked_div(p.total) {
            if pct != last_pct {
                last_pct = pct;
                let _ = progress.send(format!(
                    "Downloading {}: {} / {} MB ({pct}%)",
                    label,
                    p.completed / (1024 * 1024),
                    p.total / (1024 * 1024)
                ));
            }
        } else if !p.status.is_empty() {
            let _ = progress.send(format!("Ollama: {}", p.status));
        }
    }

    if !same_model(source, name) {
        let _ = progress.send(format!("Registering {name} in Ollama..."));
        let created = if let Some(template) = template {
            http.post(format!("{base}/api/create"))
            .json(&json!({
                "model": name,
                "from": source,
                "template": template,
                "parameters": { "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] },
                "stream": false,
            }))
            .send()
            .map_err(anyhow::Error::from)
            .and_then(|r| check(r, "Registering model"))
        } else {
            Err(anyhow!("plain copy"))
        };
        if created.is_err() {
            // Older servers: plain copy keeps the GGUF's own template.
            let r = http
                .post(format!("{base}/api/copy"))
                .json(&json!({ "source": source, "destination": name }))
                .send()?;
            check(r, "Registering model")?;
        }
        // The long hf.co name is now redundant; its data stays shared with the copy.
        let _ = http.delete(format!("{base}/api/delete")).json(&json!({ "model": source })).send();
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    system: &'a str,
    prompt: &'a str,
    /// JSON schema: Ollama constrains generation to exactly this shape.
    format: serde_json::Value,
    stream: bool,
    /// How long the model stays loaded after the scan.
    keep_alive: &'a str,
    /// `false` turns off "thinking" (Qwen3, Gemma 4) so the answer comes straight away.
    /// Left out for models that reject the setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    options: GenerateOptions,
}

#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_predict: i32,
    /// Room for the long system prompt plus a full analysis.
    num_ctx: i32,
    /// 0 = keep every layer on the CPU. Omitted = Ollama offloads to the GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    num_gpu: Option<i32>,
    /// Fixed per attempt, so the same prompt gives the same scan (a retry uses the next one).
    seed: u32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    #[serde(default)]
    response: String,
    /// The model's reasoning, when thinking was on and the model supports it.
    #[serde(default)]
    thinking: String,
    /// "stop", or "length" when the reply hit `num_predict`.
    #[serde(default)]
    done_reason: String,
    #[serde(default)]
    eval_count: u32,
}

/// Everything a scan needs, collected from the UI so it can move to a worker thread.
#[derive(Debug, Clone)]
pub struct AiScanRequest {
    pub raw_prompt: String,
    pub config: ScanConfig,
    pub mode: AiMode,
    pub host: String,
    pub model: String,
    /// e.g. "30m"; big models use a short value so their memory is freed soon.
    pub keep_alive: String,
    /// Let the model reason before answering (Gemma 4 / Qwen3). Slower; experimental.
    pub think: bool,
}

fn changes_schema(needs_negative: bool, dynamic: bool, natural: bool, max_issues: usize, max_changes: usize) -> serde_json::Value {
    let text = json!({ "type": "string" });
    let negative = if needs_negative { json!({ "type": "string", "minLength": 40 }) } else { text.clone() };
    let mut required = vec!["scene_summary", "scene_breakdown", "issues", "improved_prompt"];
    if natural {
        required.push("natural_sentence");
    }
    if dynamic {
        required.push("dynamic_prompt");
    }
    required.extend(["recommended_negative", "changes"]);
    // Property order matters: the model writes the analysis before the rewrite.
    let mut schema = json!({
        "type": "object",
        "properties": {
            "scene_summary": text,
            "scene_breakdown": {
                "type": "object",
                "properties": {
                    "subject": text, "pose": text, "framing": text,
                    "setting": text, "lighting": text, "mood": text
                },
                "required": ["subject", "pose", "framing", "setting", "lighting", "mood"]
            },
            "issues": { "type": "array", "maxItems": max_issues, "items": text },
            "improved_prompt": text,
            "natural_sentence": text,
            "dynamic_prompt": text,
            "recommended_negative": negative,
            "changes": {
                "type": "array",
                "maxItems": max_changes,
                "items": {
                    "type": "object",
                    "properties": {
                        "original": text,
                        "action": { "type": "string", "enum": ["replace", "remove", "add", "reweight", "framing_safe_swap"] },
                        "reason": text,
                        "suggested_alternative": text,
                        "framing_alternative": text
                    },
                    "required": ["original", "action", "reason", "suggested_alternative", "framing_alternative"]
                }
            }
        },
        "required": required
    });
    let props = schema["properties"].as_object_mut().expect("object");
    if !natural {
        props.remove("natural_sentence");
    }
    if !dynamic {
        props.remove("dynamic_prompt");
    }
    schema
}

fn generate_body<'a>(req: &'a AiScanRequest, system: &'a str, prompt: &'a str, seed: u32) -> GenerateRequest<'a> {
    let family = ModelFamily::detect(&req.model);
    let (max_issues, max_changes) = family.max_issues_changes();
    GenerateRequest {
        model: &req.model,
        system,
        prompt,
        format: changes_schema(
            !req.config.checkpoint.default_negative().is_empty(),
            req.config.dynamic,
            wants_sentence(&req.config),
            max_issues,
            max_changes,
        ),
        stream: false,
        keep_alive: &req.keep_alive,
        think: Some(req.think),
        options: GenerateOptions {
            seed,
            temperature: ModelFamily::detect(&req.model).temperature(),
            // reasoning tokens count against this too
            num_predict: if req.think {
                5000
            } else {
                let rich = req.config.detail == crate::profiles::DetailLevel::Rich;
                1800 + if req.config.dynamic { 600 } else { 0 } + if rich { 800 } else { 0 }
            },
            num_ctx: 16384,
            num_gpu: if req.mode == AiMode::CpuOnly { Some(0) } else { None },
        },
    }
}

/// Runs the deep scan via `POST /api/generate` and reports where it ran.
pub fn run_ai_scan(req: &AiScanRequest) -> Result<(AiJsonOutput, String)> {
    if req.mode == AiMode::Disabled {
        return Err(anyhow!("AI is turned off (rules only)."));
    }
    if req.model.trim().is_empty() {
        return Err(anyhow!("No AI model selected."));
    }
    let base = base_url(&req.host)?;
    let family = ModelFamily::detect(&req.model);
    let (system, prompt) = build_ai_system_and_user_prompt_for(&req.raw_prompt, &req.config, family);
    // thinking can take several minutes on modest GPUs
    let http = client(Some(Duration::from_secs(if req.think { 900 } else { 300 })))?;
    let mut send_think = true;
    let mut seed = 42u32;

    // A scan must come back complete: scene analysis, negative prompt and usable cards.
    // Whatever is missing is asked for again (up to 2 more scans), keeping the good
    // parts of each attempt.
    // No new follow-up or re-scan starts after this: the user gets the best result so far
    // (the badge lists what's still missing) instead of "Working…" for many minutes.
    let budget = ScanBudget::new(DEEP_SCAN_BUDGET);
    if is_big_model(&req.model) {
        // an earlier image read leaves the tagger loaded; the 12B needs that memory
        crate::tagger::release();
    }
    let mut best = scan_once(req, &http, &base, &system, &prompt, &mut send_think, &mut seed)?;
    let needs_negative = !req.config.checkpoint.default_negative().is_empty();
    if needs_negative && best.recommended_negative.trim().is_empty() && !best.improved_prompt.trim().is_empty() && !budget.spent() {
        // The model sometimes skips the negative; a short follow-up asks for just that.
        best.recommended_negative = generate_negative(req, &best.improved_prompt).unwrap_or_default();
    }
    let mut rescans = 0;
    while !scan_problems(&best, req).is_empty() && rescans < 2 && !budget.spent() {
        rescans += 1;
        let Ok(next) = scan_once(req, &http, &base, &system, &prompt, &mut send_think, &mut seed) else { continue };
        merge_scan(&mut best, next, req);
    }
    best.recommended_negative = clean_negative(&best.recommended_negative);
    enforce_checkpoint_rules(&mut best, req.config.checkpoint);
    let mut placement = model_placement(&req.host, &req.model).unwrap_or_else(|| "unknown device".to_string());
    let missing = scan_problems(&best, req);
    if rescans > 0 {
        placement.push_str(&format!(" • re-scanned {rescans}×"));
    }
    if !missing.is_empty() {
        placement.push_str(&format!(" • still missing: {}", missing.join(", ")));
    }
    Ok((best, placement))
}

/// Rules a small model sometimes ignores, applied to its reply: Flux has no negative prompt,
/// and the checkpoint's quality tags lead the rewrite (AI text only; the rule-based output
/// never gets tags the user didn't write).
fn enforce_checkpoint_rules(out: &mut AiJsonOutput, checkpoint: CheckpointProfile) {
    if checkpoint.default_negative().is_empty() {
        out.recommended_negative.clear();
    }
    out.improved_prompt = strip_placeholders(&out.improved_prompt);
    out.dynamic_prompt = strip_placeholders(&out.dynamic_prompt);
    out.recommended_negative = strip_placeholders(&out.recommended_negative);
    out.improved_prompt = ensure_quality_tags(&out.improved_prompt, checkpoint);
    out.dynamic_prompt = ensure_quality_tags(&out.dynamic_prompt, checkpoint);
}

/// Removes what a model copies out of its instructions instead of writing a tag:
/// `[UNANCHORED_LIMBS_FLAG resolved: hands in lap]` becomes `hands in lap`, a bare `*_FLAG`
/// goes, and so do placeholder tags such as `[framing]`, `{none}` or
/// `{missing framing tag placeholder}`. Real tags, weights `(tag:1.2)` and dynamic groups
/// `{a|b}` are kept; the separators and line breaks between the kept tags stay as written.
pub fn strip_placeholders(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    // Only words that are never a tag of their own. Real tags in brackets are A1111 / ComfyUI
    // de-emphasis (`[background]`, `[lighting]`, `[missing tooth]`) and must stay.
    const PLACEHOLDER_WORDS: &[&str] = &["none", "n/a", "null", "nil", "empty", "tbd", "todo", "...", "framing", "tag", "tags"];
    let placeholder = |inner: &str| {
        let l = inner.trim().to_lowercase();
        l.is_empty() || PLACEHOLDER_WORDS.contains(&l.as_str()) || l.contains("placeholder") || l.contains("_flag")
    };
    let mut kept: Vec<String> = Vec::new();
    for seg in text.split(',') {
        let lead: String = seg.chars().take_while(|c| c.is_whitespace()).collect();
        let tag = seg.trim();
        // "[X_FLAG resolved: hands in lap]" -> "hands in lap"
        // ASCII lowercasing keeps every byte where it was, so `i` is valid in `tag` (full Unicode
        // lowercasing can change lengths and made this slice panic on some characters)
        if let Some(i) = tag.to_ascii_lowercase().find("resolved:") {
            if tag.contains("_FLAG") || tag.contains("_flag") {
                let fix = tag[i + "resolved:".len()..].trim().trim_end_matches([']', ')', '}']).trim();
                if !fix.is_empty() {
                    kept.push(format!("{lead}{fix}"));
                }
                continue;
            }
        }
        if tag.ends_with("_FLAG") && !tag.contains(' ') {
            continue;
        }
        let bracketed = (tag.starts_with('[') && tag.ends_with(']')) || (tag.starts_with('{') && tag.ends_with('}'));
        if bracketed && !tag.contains('|') && placeholder(&tag[1..tag.len() - 1]) {
            continue;
        }
        kept.push(seg.to_string());
    }
    let joined = kept.join(",");
    // a dropped first tag leaves the next one's leading space at the very start
    if text.starts_with(char::is_whitespace) { joined } else { joined.trim_start().to_string() }
}

/// Turns a prompt a model wrote in a section layout back into one flat tag list: `[Scene …]:` and
/// `[Character 1]:` labels go, a `[tag, tag]:` header and `(tag, tag)` groups are unwrapped (weights
/// such as `(tag:1.2)` stay), prose sentences of 8+ words go (the app writes its own sentence), and
/// repeated tags are kept once.
pub fn flatten_section_layout(prompt: &str) -> String {
    if !prompt.contains("]:") && !prompt.contains("[Scene") && !prompt.contains("[Character") {
        return prompt.to_string();
    }
    let mut text = prompt.to_string();
    // "[...]:" headers: a label is dropped, a tag list inside is kept
    while let Some(open) = text.find('[') {
        let Some(rel) = text[open..].find("]:") else { break };
        let close = open + rel;
        let inner = text[open + 1..close].to_string();
        let keep = if inner.contains(',') { format!("{inner},") } else { String::new() };
        text.replace_range(open..close + 2, &keep);
    }
    // "(tag, tag, …)" groups without a weight
    let mut out = String::new();
    let mut depth_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        match c {
            '(' if depth_start.is_none() => depth_start = Some(i),
            ')' if depth_start.is_some() => {
                let start = depth_start.take().unwrap_or(i);
                let inner = &text[start + 1..i];
                if inner.contains(',') && !inner.contains(':') {
                    // commas around it, so the next words (often a sentence) don't fuse with the last tag
                    out.push_str(", ");
                    out.push_str(inner);
                    out.push_str(", ");
                } else {
                    out.push_str(&text[start..=i]);
                }
            }
            _ if depth_start.is_none() => out.push(c),
            _ => {}
        }
    }
    if let Some(start) = depth_start {
        out.push_str(&text[start..]);
    }
    let mut seen = std::collections::HashSet::new();
    let tags: Vec<String> = out
        .split([',', '\n'])
        .map(|t| t.trim().trim_end_matches('.').trim().to_string())
        .filter(|t| !t.is_empty() && t.split_whitespace().count() < 8)
        .filter(|t| seen.insert(t.to_lowercase()))
        .collect();
    tags.join(", ")
}

/// A tag the tagger can be trusted on when it is very sure: skin tone, animal species parts
/// (`raccoon ears`, `fox tail`, `cat girl`) and person counts (`1girl`, `2boys`).
fn is_guarded_tag(name: &str) -> bool {
    let t = name.to_lowercase();
    let count = t.chars().next().is_some_and(|c| c.is_ascii_digit())
        && ["girl", "girls", "boy", "boys", "other", "others"].iter().any(|s| t.ends_with(s));
    let skin = t.ends_with(" skin") || t.contains("-skinned") || t == "tan" || t == "tanlines";
    let species = (t.ends_with(" ears") || t.ends_with(" tail") || t.ends_with(" girl") || t.ends_with(" boy"))
        && !t.starts_with("animal ")
        && !t.starts_with("fake ")
        && !count;
    count || skin || species
}

/// The animal a one-word species tag stands for (`kitsune` → fox, `nekomimi` → cat …).
fn species_word_animal(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "kitsune" | "kyuubi" => "fox",
        "nekomimi" | "catgirl" | "nekomata" => "cat",
        "inumimi" | "doggirl" => "dog",
        "usagimimi" | "bunnygirl" => "rabbit",
        "okamimimi" | "wolfgirl" => "wolf",
        "tanuki" => "tanuki",
        "kemonomimi mode" => return None,
        _ => return None,
    })
}

/// The tagger's skin / species / count tags it is at least 80% sure of (prompt spelling).
pub fn strong_tags(tags: &[crate::tagger::ScoredTag]) -> Vec<String> {
    tags.iter()
        .filter(|t| t.score >= 0.8 && is_guarded_tag(&t.name.replace('_', " ")))
        .map(|t| t.name.replace('_', " "))
        .collect()
}

/// Puts the tagger's sure tags back when the writer dropped or contradicted them: a different animal on
/// the same part (`cat ears` vs. a sure `raccoon ears`) is replaced, an opposite skin tone removed, a
/// different count of the same kind replaced; missing ones are added after the leading quality tags.
pub fn enforce_strong_tags(prompt: &str, strong: &[String]) -> String {
    if strong.is_empty() || prompt.trim().is_empty() {
        return prompt.to_string();
    }
    let mut tags: Vec<String> = prompt.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let key = |t: &str| crate::danbooru::key(t);
    // sure tags never push each other out ("striped tail" is a pattern, not a rival to "raccoon tail")
    let sure_keys: std::collections::HashSet<String> = strong.iter().map(|t| key(t)).collect();
    for sure in strong {
        let s = sure.to_lowercase();
        let part = ["ears", "tail", "girl", "boy"].iter().find(|p| s.ends_with(&format!(" {p}")) ).copied();
        let is_count = s.chars().next().is_some_and(|c| c.is_ascii_digit());
        let is_skin = s.ends_with(" skin") || s.contains("-skinned");
        tags.retain(|t| {
            let l = t.to_lowercase();
            if key(&l) == key(&s) || sure_keys.contains(&key(&l)) {
                return true;
            }
            if is_count {
                let kind = if s.contains("girl") { "girl" } else if s.contains("boy") { "boy" } else { "other" };
                return !(l.chars().next().is_some_and(|c| c.is_ascii_digit()) && l.contains(kind));
            }
            if is_skin {
                let dark = s.contains("dark") || s.contains("tan") || s.contains("brown");
                let opposite = if dark { ["pale skin", "fair skin", "white skin", "light skin"] } else { ["dark skin", "tan skin", "brown skin", "dark-skinned female"] };
                return !opposite.contains(&l.as_str());
            }
            if let Some(p) = part {
                // another animal on the same part ("cat ears" vs "raccoon ears"); generic "animal ears" stays
                let other = l.ends_with(&format!(" {p}")) && !l.starts_with("animal ") && !l.starts_with("fake ") && l.split_whitespace().count() == 2;
                // or a one-word species tag for another animal ("kitsune" next to a sure "raccoon girl")
                let sure_animal = s.split_whitespace().next().unwrap_or("");
                let rival_species = species_word_animal(&l).is_some_and(|a| a != sure_animal);
                return !other && !rival_species;
            }
            true
        });
        if !tags.iter().any(|t| key(t) == key(&s)) {
            let quality = ["masterpiece", "best quality", "score 9", "score_9", "score 8 up", "score_8_up", "score 7 up", "score_7_up", "absurdres", "highres"];
            let at = tags.iter().take_while(|t| quality.contains(&t.to_lowercase().as_str())).count();
            tags.insert(at, sure.clone());
        }
    }
    tags.join(", ")
}

/// Removes instruction text a model copied into its prompt: tag-count notes ("25-45 tags total",
/// "tag count: 40") and explanatory parentheses ("raccoon tail (inferred from tagger/context)" becomes
/// "raccoon tail"). Weights such as "(red hair:1.2)" and real parenthesised tags are kept.
pub fn strip_instruction_leaks(prompt: &str) -> String {
    const NOTE_WORDS: &[&str] = &[
        "inferred", "tagger", "context", "note", "assumed", "assume", "guess", "likely", "possibly", "maybe",
        "if visible", "optional", "approx", "uncertain", "presumably",
    ];
    let mut kept: Vec<String> = Vec::new();
    for seg in prompt.split(',') {
        let lead: String = seg.chars().take_while(|c| c.is_whitespace()).collect();
        let mut tag = seg.trim().to_string();
        let l = tag.to_lowercase();
        let count_note = l.contains("tags total")
            || l.contains("tag count")
            || l.contains("total tags")
            || (l.ends_with(" tags") && l.chars().next().is_some_and(|c| c.is_ascii_digit()));
        if count_note {
            continue;
        }
        // "(...)" with no weight inside and a note word: an explanation, not part of the tag
        if let (Some(open), Some(close)) = (tag.find('('), tag.rfind(')')) {
            if close > open {
                let inner = tag[open + 1..close].to_lowercase();
                if !inner.contains(':') && NOTE_WORDS.iter().any(|w| inner.contains(w)) {
                    tag = format!("{}{}", tag[..open].trim_end(), &tag[close + 1..]).trim().to_string();
                }
            }
        }
        if !tag.is_empty() {
            kept.push(format!("{lead}{tag}"));
        }
    }
    let joined = kept.join(",");
    if prompt.starts_with(char::is_whitespace) { joined } else { joined.trim_start().to_string() }
}

/// Puts the checkpoint's missing quality tags at the front when the prompt lacks its first one.
/// An empty prompt stays empty.
pub fn ensure_quality_tags(prompt: &str, checkpoint: CheckpointProfile) -> String {
    let quality = checkpoint.canonical_quality_tags();
    let Some(first) = quality.first() else { return prompt.to_string() };
    if prompt.trim().is_empty() {
        return prompt.to_string();
    }
    let norm = |t: &str| t.trim().trim_matches(|c| c == '(' || c == ')').split(':').next().unwrap_or("").trim().to_lowercase();
    let present: Vec<String> = prompt.split(|c| c == ',' || c == '\n').map(norm).collect();
    if present.iter().any(|t| *t == *first) {
        return prompt.to_string();
    }
    let missing: Vec<&str> = quality.iter().copied().filter(|q| !present.iter().any(|t| t == q)).collect();
    format!("{}, {}", missing.join(", "), prompt.trim_start())
}

/// One deep-scan request; an unreadable reply is retried (up to 3 tries).
fn scan_once(
    req: &AiScanRequest,
    http: &reqwest::blocking::Client,
    base: &str,
    system: &str,
    prompt: &str,
    send_think: &mut bool,
    seed: &mut u32,
) -> Result<AiJsonOutput> {
    for _attempt in 0..3 {
        let mut body = generate_body(req, system, prompt, *seed);
        *seed += 1;
        if !*send_think {
            body.think = None;
        }
        let resp = http
            .post(format!("{base}/api/generate"))
            .json(&body)
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow!("The model took over 5 minutes to answer; try CPU/GPU mode or a smaller model")
                } else {
                    anyhow!("Couldn't reach Ollama at {base}")
                }
            })?;
        let resp = match check(resp, "AI scan") {
            // older/other models reject the think setting: resend without it
            Err(e) if *send_think && e.to_string().to_lowercase().contains("think") => {
                *send_think = false;
                continue;
            }
            r => r?,
        };
        let gen: GenerateResponse = resp.json()?;
        if std::env::var_os("PC_DEBUG_SCAN").is_some() {
            eprintln!(
                "[scan raw reply: done_reason {}, {} tokens of {}]\n{}\n[/scan]",
                gen.done_reason, gen.eval_count, body.options.num_predict, gen.response
            );
        }
        if let Some(mut o) = extract_json_from_llm(&gen.response) {
            o.thinking = gen.thinking.clone();
            return Ok(o);
        }
    }
    Err(anyhow!("Ollama answered, but the model's reply was unreadable. Run it again, or try a bigger model."))
}

fn analysis_ok(o: &AiJsonOutput) -> bool {
    let b = &o.scene_breakdown;
    let filled = [&b.subject, &b.pose, &b.framing, &b.setting, &b.lighting, &b.mood]
        .iter()
        .filter(|f| f.trim().len() >= 3)
        .count();
    o.scene_summary.trim().len() >= 20 && filled >= 4
}

fn negative_ok(o: &AiJsonOutput, req: &AiScanRequest) -> bool {
    req.config.checkpoint.default_negative().is_empty()
        || clean_negative(&o.recommended_negative).split(',').filter(|t| !t.trim().is_empty()).count() >= 5
}

/// The rewrite plus cards that would actually show up: when the improved prompt differs
/// from the user's, at least one AI card must survive `convert_ai_changes`.
/// The user's tags the scan was about (every character on the 2-4 Characters tabs).
pub fn request_tokens(req: &AiScanRequest) -> Vec<crate::parser::PromptToken> {
    let mut t = crate::parser::parse_prompt(&req.raw_prompt);
    if req.config.two_char {
        t.extend(crate::parser::parse_prompt(&req.config.scene_prompt_raw));
        for i in 1..req.config.char_count() {
            t.extend(crate::parser::parse_prompt(req.config.prompt_of(i)));
        }
    }
    t
}

fn cards_ok(o: &AiJsonOutput, req: &AiScanRequest) -> bool {
    if o.improved_prompt.trim().is_empty() {
        return false;
    }
    let tokens = request_tokens(req);
    // Layout tags (quality, group count, "far left"…) are added by the app itself (`two::combine`,
    // `ensure_quality_tags`), so they are no change that needs a card. Counting them made every
    // 3-4 character scan re-scan twice and end with "still missing: cards".
    // A layout tag the user wrote (their own `2girls`, `left`…) is content: changing it to `3girls`
    // must still need a card. Only layout the AI added on top is ignored.
    let before: std::collections::HashSet<String> = tokens.iter().map(|t| t.normalized.clone()).collect();
    let content = |t: &String| before.contains(t) || !is_layout_tag(t, req.config.checkpoint);
    let after: std::collections::HashSet<String> = crate::parser::parse_prompt(&o.improved_prompt)
        .into_iter()
        .map(|t| t.normalized)
        .filter(content)
        .collect();
    if before == after {
        return true; // nothing to change, so no cards needed
    }
    let rules = if req.config.two_char {
        let mut r = Vec::new();
        let add_for = |prompt: &str, r: &mut Vec<PromptChange>| {
            let t = crate::parser::parse_prompt(prompt);
            r.extend(crate::analyzer::analyze_with_rules(&t, &req.config).changes);
        };
        add_for(&req.raw_prompt, &mut r);
        add_for(&req.config.scene_prompt_raw, &mut r);
        for i in 1..req.config.char_count() {
            add_for(req.config.prompt_of(i), &mut r);
        }
        r
    } else {
        crate::analyzer::analyze_with_rules(&tokens, &req.config).changes
    };
    if !convert_ai_changes(&o.changes, &tokens, &rules).is_empty() {
        return true;
    }
    // If the improved prompt has edits and all edits are already handled by existing rule cards,
    // the scan is valid and we do not trigger wasteful re-scans.
    let covered: std::collections::HashSet<String> = rules
        .iter()
        .flat_map(|c| crate::parser::parse_prompt(&c.original))
        .map(|t| t.normalized)
        .collect();
    let diff = before.symmetric_difference(&after).collect::<Vec<_>>();
    !diff.is_empty() && diff.iter().all(|t| covered.contains(*t))
}

fn dynamic_ok(o: &AiJsonOutput, req: &AiScanRequest) -> bool {
    !req.config.dynamic || o.dynamic_prompt.contains('{')
}

/// What a scan is missing, in words for the status bar (empty = complete).
pub fn scan_problems(o: &AiJsonOutput, req: &AiScanRequest) -> Vec<&'static str> {
    let mut p = Vec::new();
    if !analysis_ok(o) {
        p.push("scene analysis");
    }
    if !negative_ok(o, req) {
        p.push("negative prompt");
    }
    if !cards_ok(o, req) {
        p.push("cards");
    }
    if !dynamic_ok(o, req) {
        p.push("dynamic prompt");
    }
    p
}

/// Fills the parts `best` is missing from a re-scan. The rewrite, its cards and its
/// dynamic version belong together, so they're taken as a set.
fn merge_scan(best: &mut AiJsonOutput, next: AiJsonOutput, req: &AiScanRequest) {
    if !analysis_ok(best) && analysis_ok(&next) {
        best.scene_summary = next.scene_summary.clone();
        best.scene_breakdown = next.scene_breakdown.clone();
        best.issues = next.issues.clone();
    }
    if !cards_ok(best, req) && cards_ok(&next, req) {
        best.improved_prompt = next.improved_prompt.clone();
        best.changes = next.changes.clone();
        best.dynamic_prompt = next.dynamic_prompt.clone();
        best.natural_sentence = next.natural_sentence.clone();
        best.thinking = next.thinking.clone();
    } else if !dynamic_ok(best, req) && dynamic_ok(&next, req) {
        best.dynamic_prompt = next.dynamic_prompt.clone();
    }
    if !negative_ok(best, req) && negative_ok(&next, req) {
        best.recommended_negative = next.recommended_negative;
    }
}

/// Asks only for a negative prompt matching `positive` (used when the scan's reply had none).
fn generate_negative(req: &AiScanRequest, positive: &str) -> Result<String> {
    let base = base_url(&req.host)?;
    let system = format!(
        "You write negative prompts for {} (ComfyUI). Reply with JSON only.",
        req.config.checkpoint.display_name()
    );
    let view_negs = req.config.target_view.negative_tags();
    let view_hint = if view_negs.is_empty() {
        String::new()
    } else {
        format!(", plus '{view_negs}' for the target framing")
    };
    let prompt = format!(
        "Positive prompt:\n{positive}\n\nWrite a negative prompt of 12-30 comma-separated tags: start from these usual \
         negatives ({}{view_hint}), then add concrete Danbooru tags for things that would spoil THIS scene (e.g. 'full body, lower body' \
         for an upper body shot, 'crowd, multiple girls' when solo, the opposite of its setting, time of day or mood), \
         each a real Danbooru tag naming the unwanted thing. Never 'no X', 'wrong X' or 'unwanted X'; never anything \
         the positive prompt asks for.",
        req.config.checkpoint.default_negative()
    );
    let body = json!({
        "model": req.model,
        "system": system,
        "prompt": prompt,
        "format": { "type": "object", "properties": { "negative": { "type": "string" } }, "required": ["negative"] },
        "stream": false,
        "keep_alive": req.keep_alive,
        "options": { "temperature": 0.2, "num_predict": 300, "num_ctx": 16384, "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> } },
    });
    let resp = client(Some(Duration::from_secs(120)))?.post(format!("{base}/api/generate")).json(&body).send()?;
    let gen: GenerateResponse = check(resp, "Negative prompt")?.json()?;
    let v: serde_json::Value = serde_json::from_str(gen.response.trim()).unwrap_or_default();
    Ok(v.get("negative").and_then(|n| n.as_str()).unwrap_or("").to_string())
}

/// Removes negative tags that backfire or say nothing: "no smile" in a negative prompt
/// fights the smile; "wrong framing" means nothing to the model. "no humans" is a real tag.
pub fn clean_negative(negative: &str) -> String {
    negative
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.replace('_', " "))
        .map(|t| t.strip_prefix("unwanted ").map(str::to_string).unwrap_or(t))
        .filter(|t| {
            let l = t.to_lowercase();
            l == "no humans"
                || !(l.starts_with("no ")
                    || l.starts_with("wrong ")
                    || l.starts_with("non-")
                    || l.starts_with("non ")
                    || l.starts_with("lack of ")
                    || l.starts_with("not ")
                    || l.starts_with("without "))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
struct PsModel {
    #[serde(default)]
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    size_vram: u64,
}

/// Describes how much of a loaded model sits in GPU memory, like `ollama ps`.
pub fn describe_placement(size: u64, size_vram: u64) -> String {
    if size == 0 || size_vram == 0 {
        "CPU".to_string()
    } else if size_vram >= size {
        "GPU 100%".to_string()
    } else {
        let gpu = size_vram * 100 / size;
        format!("{gpu}% GPU / {}% CPU", 100 - gpu)
    }
}

/// `GET /api/ps`: where the given model is currently loaded.
pub fn model_placement(host: &str, model: &str) -> Option<String> {
    let base = base_url(host).ok()?;
    let ps: PsResponse = client(Some(Duration::from_secs(2)))
        .ok()?
        .get(format!("{base}/api/ps"))
        .send()
        .ok()?
        .json()
        .ok()?;
    ps.models
        .iter()
        .find(|m| same_model(&m.name, model))
        .map(|m| describe_placement(m.size, m.size_vram))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_url_normalization() {
        assert_eq!(normalize_ollama_url("192.168.1.50").as_deref(), Some("http://192.168.1.50:11434"));
        assert_eq!(normalize_ollama_url("192.168.1.50:11434").as_deref(), Some("http://192.168.1.50:11434"));
        assert_eq!(normalize_ollama_url("localhost:8080/").as_deref(), Some("http://localhost:8080"));
        assert_eq!(normalize_ollama_url("http://gpu-box:11434/api/tags").as_deref(), Some("http://gpu-box:11434"));
        assert_eq!(normalize_ollama_url("https://ollama.lan").as_deref(), Some("https://ollama.lan:11434"));
        assert_eq!(normalize_ollama_url("[::1]").as_deref(), Some("http://[::1]:11434"));
        assert_eq!(normalize_ollama_url("[::1]:9000").as_deref(), Some("http://[::1]:9000"));
        assert_eq!(normalize_ollama_url("  "), None);
        assert_eq!(normalize_ollama_url("host:notaport"), None);
    }

    #[test]
    fn local_host_detection() {
        assert!(is_local_host("localhost"));
        assert!(is_local_host("127.0.0.1:11434"));
        assert!(!is_local_host("192.168.1.50"));
    }

    #[test]
    fn placement_text() {
        assert_eq!(describe_placement(100, 100), "GPU 100%");
        assert_eq!(describe_placement(100, 0), "CPU");
        assert_eq!(describe_placement(100, 60), "60% GPU / 40% CPU");
    }

    #[test]
    fn cpu_only_sets_num_gpu_zero() {
        use crate::profiles::{CheckpointProfile, TargetView};
        let mut req = AiScanRequest {
            raw_prompt: "1girl".into(),
            config: ScanConfig {
                checkpoint: CheckpointProfile::Anima,
                target_view: TargetView::None,
                pinned_emphasis_raw: String::new(),
                emphasis_weight: 1.25,
                cleanup_enabled: true,
                resolve_conflicts_enabled: true,
                clamp_weights_enabled: true,

                camera_angle: 0,

                pose: 0,
                proportion: 0,

                dynamic: false,

                lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),

                detail: crate::profiles::DetailLevel::Balanced,
            },
            mode: AiMode::CpuOnly,
            host: "localhost".into(),
            model: "m".into(),
            keep_alive: "30m".into(),
            think: false,
        };
        let v = serde_json::to_value(generate_body(&req, "s", "p", 42)).unwrap();
        assert_eq!(v["options"]["num_gpu"], 0);
        req.mode = AiMode::Gpu;
        let v = serde_json::to_value(generate_body(&req, "s", "p", 42)).unwrap();
        assert!(v["options"].get("num_gpu").is_none());
        assert_eq!(v["format"]["type"], "object");
    }

    #[test]
    fn same_model_ignores_latest_tag() {
        assert!(same_model("prompt-condenser-low:latest", "prompt-condenser-low"));
        assert!(!same_model("qwen2.5:3b", "qwen2.5:7b"));
    }

    #[test]
    fn json_extraction_ignores_surrounding_text() {
        let raw = "Sure! ```json\n{\"changes\":[{\"original\":\"ass\",\"suggested_alternative\":\"curvy\"}]}\n```";
        let out = extract_json_from_llm(raw).unwrap();
        assert_eq!(out.changes[0].original, "ass");
    }

    /// Minimal fake Ollama: answers by path, records each request's path and body.
    fn fake_ollama() -> (String, std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let host = listener.local_addr().unwrap().to_string();
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log2 = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // read headers, then the body by Content-Length
                let (head_end, len) = loop {
                    let n = stream.read(&mut chunk).unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                            .unwrap_or(0);
                        break (pos + 4, len);
                    }
                };
                while buf.len() < head_end + len {
                    let n = stream.read(&mut chunk).unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                }
                let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
                let body = String::from_utf8_lossy(&buf[head_end..head_end + len]).to_string();
                log2.lock().unwrap().push((path.clone(), body));
                let reply = match path.as_str() {
                    "/api/tags" => r#"{"models":[{"name":"qwen2.5:7b"},{"name":"prompt-condenser-low:latest"}]}"#.to_string(),
                    "/api/ps" => r#"{"models":[{"name":"prompt-condenser-low:latest","size":1000,"size_vram":1000}]}"#.to_string(),
                    "/api/generate" => serde_json::json!({
                        "response": "{\"changes\":[{\"original\":\"ass\",\"action\":\"framing_safe_swap\",\"reason\":\"r\",\"suggested_alternative\":\"curvy\",\"framing_alternative\":\"\"}]}"
                    }).to_string(),
                    "/api/pull" => "{\"status\":\"pulling manifest\"}\n{\"status\":\"downloading\",\"total\":100,\"completed\":50}\n{\"status\":\"success\"}\n".to_string(),
                    _ => r#"{"status":"success"}"#.to_string(),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    reply.len(),
                    reply
                );
            }
        });
        (host, log)
    }

    #[test]
    fn rules_a_small_model_forgets_are_enforced() {
        use crate::profiles::CheckpointProfile::*;
        // missing quality tags are put in front, only the missing ones
        assert_eq!(ensure_quality_tags("1girl, smile", PonyV6), "score_9, score_8_up, score_7_up, 1girl, smile");
        assert_eq!(ensure_quality_tags("best quality, 1girl", Anima), "masterpiece, best quality, 1girl");
        // already there (also weighted, or on the first line of an arranged prompt): untouched
        assert_eq!(ensure_quality_tags("(masterpiece:1.2), 1girl", Anima), "(masterpiece:1.2), 1girl");
        assert_eq!(ensure_quality_tags("score_9, 1girl", PonyV6), "score_9, 1girl");
        assert_eq!(ensure_quality_tags("masterpiece,\n1girl", Anima), "masterpiece,\n1girl");
        // Flux has no quality tags; empty stays empty
        assert_eq!(ensure_quality_tags("a girl on a beach", Flux), "a girl on a beach");
        assert_eq!(ensure_quality_tags("", PonyV6), "");
        // Flux never keeps a negative, other checkpoints do
        let mut o = AiJsonOutput { recommended_negative: "blurry, bad hands".into(), improved_prompt: "a girl".into(), ..Default::default() };
        enforce_checkpoint_rules(&mut o, Flux);
        assert!(o.recommended_negative.is_empty());
        let mut o = AiJsonOutput { recommended_negative: "blurry, bad hands".into(), improved_prompt: "1girl".into(), ..Default::default() };
        enforce_checkpoint_rules(&mut o, PonyV6);
        assert_eq!(o.recommended_negative, "blurry, bad hands");
        assert!(o.improved_prompt.starts_with("score_9"));
    }

    /// `PC_DEBUG_GEMMA=1 AUDIT_MODEL=<text model> cargo test live_synthesis -- --ignored --nocapture`:
    /// runs the dual-pass pipeline with Gemma synthesis on packaging/arch/character.jpg and prints
    /// the pipeline warnings (e.g. "Gemma output unusable") and the raw model reply.
    /// `IMAGE_TEST=a.png VISION_ENCODER=<model> cargo test live_checklist -- --ignored --nocapture`:
    /// the structured observation form for one image.
    #[test]
    #[ignore]
    fn live_checklist() {
        let img = std::env::var("IMAGE_TEST").expect("IMAGE_TEST");
        let enc = std::env::var("VISION_ENCODER").unwrap_or_else(|_| "huihui_ai/qwen2.5-vl-abliterated:3b".into());
        let t = std::time::Instant::now();
        let form = observe_with_checklist(std::path::Path::new(&img), "localhost:11434", AiMode::Gpu, &enc, &[]).unwrap();
        println!("=== {enc} ({:.0} s)\n{form}\n===", t.elapsed().as_secs_f32());
    }

    /// `IMAGE_TEST=a.png VISION_ENCODER=<model> cargo test live_vision_dump -- --ignored --nocapture`:
    /// prints the vision model's raw multi-zone description of an image (what the writer gets to see).
    #[test]
    #[ignore]
    fn live_vision_dump() {
        let img = std::env::var("IMAGE_TEST").expect("IMAGE_TEST");
        let enc = std::env::var("VISION_ENCODER").unwrap_or_else(|_| "huihui_ai/qwen2.5-vl-abliterated:3b".into());
        let t = std::time::Instant::now();
        let desc = extract_visual_description_from_path(std::path::Path::new(&img), "localhost:11434", AiMode::Gpu, &enc).unwrap();
        println!("=== {enc} ({:.0} s)\n{desc}\n===", t.elapsed().as_secs_f32());
    }

    #[test]
    #[ignore]
    fn live_synthesis() {
        use crate::prompt_condenser::{Mode, PipelineOptions};
        let model = std::env::var("AUDIT_MODEL").unwrap_or_else(|_| AiModelTier::Best.ollama_name().to_string());
        let vision = std::env::var("VISION_ENCODER").unwrap_or_else(|_| "huihui_ai/qwen2.5-vl-abliterated:3b".to_string());
        let img_path = std::env::var("IMAGE_TEST").unwrap_or_else(|_| "packaging/arch/character.jpg".to_string());
        let img = std::path::Path::new(&img_path);
        let vision_pass = std::env::var_os("GEMMA_VISION_PASS").is_some();
        let tags = image::open(img)
            .ok()
            .and_then(|d| crate::tagger::tag_dynamic(&d).ok())
            .map(|t| t.iter().filter(|x| x.score >= 0.35).map(|x| x.name.clone()).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        for mode in [Mode::SceneShape, Mode::ScenePose] {
            let opts = PipelineOptions { mode, use_llm_synthesis: true, gemma_vision_pass: vision_pass };
            let r = run_dual_pass_asymmetric_pipeline("localhost:11434", AiMode::Gpu, &vision, &vision, &model, img, &tags, opts);
            println!("== {img_path} {mode:?}\nWARNINGS: {:?}\nscene: {}\ncharacter: {}", r.warnings, r.scene_prompt, r.character_prompt);
        }
    }

    #[test]
    fn copied_flags_and_placeholders_are_removed() {
        // the three leaks seen in the live Anima audit
        assert_eq!(
            strip_placeholders("1girl, shrine, night, moonlit, [UNANCHORED_LIMBS_FLAG resolved: hands in lap]"),
            "1girl, shrine, night, moonlit, hands in lap"
        );
        assert_eq!(strip_placeholders("2girls, park, sunset, [framing], BREAK, 1girl"), "2girls, park, sunset, BREAK, 1girl");
        assert_eq!(strip_placeholders("bad hands, close-up, solo, crowded, {none}"), "bad hands, close-up, solo, crowded");
        assert_eq!(strip_placeholders("worst quality, {missing framing tag placeholder}, blurry"), "worst quality, blurry");
        assert_eq!(strip_placeholders("1girl, MISSING_BOTTOMS_FLAG, skirt"), "1girl, skirt");
        // real syntax survives: dynamic groups, weights, de-emphasis of a real tag, line breaks
        for keep in ["1girl, {smile|frown}, (red hair:1.2), [blurry background]", "masterpiece,\n1girl, beach", "{beach|forest}"] {
            assert_eq!(strip_placeholders(keep), keep);
        }
        assert_eq!(strip_placeholders("{none}, 1girl"), "1girl");
        // real de-emphasis tags stay
        for keep in ["[background], 1girl", "1girl, [lighting]", "[missing tooth], smile", "[pose]"] {
            assert_eq!(strip_placeholders(keep), keep);
        }
        // non-ASCII text before "resolved:" can't break the slicing (it used to panic)
        assert_eq!(strip_placeholders("[İK UNANCHORED_LIMBS_FLAG resolved: hands in lap]"), "hands in lap");
        assert_eq!(strip_placeholders(""), "");
    }

    #[test]
    fn layout_only_rewrites_need_no_cards_and_misfiled_adds_count() {
        use crate::profiles::{CheckpointProfile, DetailLevel, TargetView};
        let config = ScanConfig {
            checkpoint: CheckpointProfile::Anima, target_view: TargetView::None, pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
            camera_angle: 0, pose: 0, proportion: 0, dynamic: false, lora_tags_raw: String::new(), arranged: false, natural: true,
            two_char: true, char2_prompt_raw: "1girl, blue hair, bob cut, sailor uniform".into(), char2_lora_tags_raw: String::new(),
            char_break: false, scene_prompt_raw: "beach, sunset, standing in a row, wide shot".into(),
            more_chars: vec![("1girl, green hair, twintails, hoodie".into(), String::new()), ("1girl, pink hair, braid, white sundress".into(), String::new())],
            detail: DetailLevel::Balanced,
        };
        let req = AiScanRequest {
            raw_prompt: "1girl, red hair, long hair, red dress".into(), config, mode: AiMode::Gpu,
            host: String::new(), model: "huihui_ai/gemma-4-abliterated:12b".into(), keep_alive: "30m".into(), think: false,
        };
        let add = |o: &str, sug: &str| AiChangeItem { original: o.into(), action: "add".into(), reason: "r".into(), suggested_alternative: sug.into(), framing_alternative: String::new() };
        // the real 12B reply for 4 characters: only quality, count and position tags were added
        let mut out = AiJsonOutput {
            improved_prompt: "masterpiece, best quality, 4girls, standing in a row, beach, sunset, wide shot, BREAK, 1girl, on the far left, red hair, long hair, red dress, BREAK, 1girl, second from left, blue hair, bob cut, sailor uniform, BREAK, 1girl, second from right, green hair, twintails, hoodie, BREAK, 1girl, on the far right, pink hair, braid, white sundress".into(),
            changes: vec![add("4girls", "4girls"), add("far left", "far left"), add("beach", "beach")],
            ..Default::default()
        };
        assert!(cards_ok(&out, &req), "a layout-only rewrite needs no cards");
        let tokens = request_tokens(&req);
        assert!(convert_ai_changes(&out.changes, &tokens, &[]).is_empty(), "layout and existing tags make no cards");
        // a real new detail filed the same wrong way still becomes an add card
        out.improved_prompt.push_str(", seashells");
        out.changes.push(add("seashells", "seashells"));
        let cards = convert_ai_changes(&out.changes, &tokens, &[]);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].suggested_alternative, "seashells");
        assert!(cards_ok(&out, &req));
        // an "add masterpiece" is no card either: the app writes the quality tags itself
        assert!(convert_ai_changes(&[add("masterpiece", "masterpiece"), add("", "score_9")], &tokens, &[]).is_empty());
        // a layout tag the user wrote is content: changing their own count needs a card
        let mut own = req.clone();
        own.raw_prompt = "2girls, red hair, long hair, red dress".into();
        let changed = AiJsonOutput { improved_prompt: out.improved_prompt.replace("4girls", "3girls"), ..Default::default() };
        assert!(!cards_ok(&changed, &own), "2girls -> 3girls with no card must not pass");
        // the layout test itself
        assert!(is_layout_tag("score 9", CheckpointProfile::PonyV6) && is_layout_tag("score_8_up", CheckpointProfile::PonyV6));
        for t in ["break", "4girls", "2boys", "multiple girls", "on the far left", "second from the right", "in the middle", "masterpiece"] {
            assert!(is_layout_tag(t, CheckpointProfile::Anima), "{t}");
        }
        for t in ["1girl", "red hair", "left hand", "beach"] {
            assert!(!is_layout_tag(t, CheckpointProfile::Anima), "{t}");
        }
    }

    #[test]
    fn form_lines_become_tags_and_prompts_get_one_tag_per_part() {
        // the 3B's real answers for the raccoon-girl image
        let form = "PEOPLE: 1\nPOSE: standing, facing away from the camera, arms relaxed at sides\n\
                    GAZE AND EXPRESSION:\n- Looking over shoulder towards back of head with a slight smile\n\
                    FRAMING: from chest up to mid-thighs, slightly angled left side view\n\
                    BACKGROUND: plain light grey or off-white color; no specific contents visible";
        let t = tags_from_form(form);
        for want in ["cowboy shot", "from side", "simple background", "grey background", "looking back", "smile"] {
            assert!(t.iter().any(|x| x == want), "{want} missing from {t:?}");
        }
        assert!(!t.iter().any(|x| x == "arms at sides"), "the pose line is not trusted");
        // one tag per part: a dictionary tag beats an unknown one, else the first wins
        let one = one_per_part(
            ["red eyes", "reddish-brown eyes", "dark skin", "light brown skin", "thigh boots", "knee high boots", "medium hair", "long hair", "smile"]
                .iter().map(|x| x.to_string()).collect(),
        );
        assert_eq!(one, vec!["red eyes", "dark skin", "thigh boots", "medium hair", "smile"]);
        // the dictionary check needs the tagger's tag list on disk
        if crate::danbooru::local().is_some() {
            let v = validate_tags(
                ["black leather shorts", "standing sideways", "soft shadows", "blue jacket", "curvy hips", "gold wing hair clip", "raccoon girl", "masterpiece"]
                    .iter().map(|x| x.to_string()).collect(),
            );
            assert_eq!(v, vec!["black leather shorts", "from side", "blue jacket", "wide hips", "gold wing hair clip", "raccoon girl", "masterpiece"], "{v:?}");
            // descriptive tags the list doesn't have are kept; names as labels are not tags
            let v2 = validate_tags(["white teal bikini top", "Nagi: black hair", "light skin tones"].iter().map(|x| x.to_string()).collect());
            assert_eq!(v2, vec!["white teal bikini top", "black hair"], "{v2:?}");
        }
    }

    #[test]
    fn merged_tags_are_split_and_contradicted_hair_goes() {
        if crate::danbooru::local().is_some() {
            assert_eq!(split_merged_tags("blue jacket open jacket off the shoulders".into()), vec!["blue jacket", "open jacket", "off shoulder"]);
            // descriptive tags and real multi-word tags stay whole
            for keep in ["very short black leather shorts", "short hair with long locks", "golden wing hair ornament", "white teal bikini top"] {
                assert_eq!(split_merged_tags(keep.into()), vec![keep.to_string()], "{keep}");
            }
        }
        let hair: Vec<String> = ["white hair", "short hair with long locks", "low twintails", "long hair", "twintails", "smile"].iter().map(|x| x.to_string()).collect();
        assert_eq!(drop_contradicted_hair(hair), vec!["white hair", "short hair with long locks", "smile"]);
        let plain: Vec<String> = ["long hair", "twintails"].iter().map(|x| x.to_string()).collect();
        assert_eq!(drop_contradicted_hair(plain.clone()), plain, "untouched without the short-hair-with-locks tag");
    }

    #[test]
    fn the_taggers_sure_tags_cannot_be_dropped_or_contradicted() {
        let sure: Vec<String> =
            ["raccoon ears", "raccoon tail", "dark skin", "1girl", "striped tail"].iter().map(|t| t.to_string()).collect();
        // the 12B run: fox instead of raccoon, no dark skin
        let fixed = enforce_strong_tags("masterpiece, best quality, 1girl, fox ears, animal ears, fluffy tail, fox tail, pale skin, glasses", &sure);
        for want in ["raccoon ears", "raccoon tail", "striped tail", "dark skin", "animal ears", "glasses", "1girl"] {
            assert!(fixed.split(", ").any(|t| t == want), "{want} missing: {fixed}");
        }
        for gone in ["fox ears", "fox tail", "pale skin"] {
            assert!(!fixed.split(", ").any(|t| t == gone), "{gone} kept: {fixed}");
        }
        assert!(fixed.starts_with("masterpiece, best quality,"), "added tags go after the quality tags: {fixed}");
        // a one-word species tag for another animal goes too; generic "kemonomimi" stays
        let fixed2 = enforce_strong_tags("1girl, kitsune, kemonomimi, nekomimi, smile", &["raccoon girl".to_string()]);
        assert_eq!(fixed2, "raccoon girl, 1girl, kemonomimi, smile");
        // a wrong count is replaced, nothing changes without sure tags
        assert_eq!(enforce_strong_tags("2girls, smile", &["1girl".to_string()]), "1girl, smile");
        assert_eq!(enforce_strong_tags("2girls, smile", &[]), "2girls, smile");
        // only these kinds are guarded
        for g in ["raccoon ears", "fox tail", "cat girl", "dark skin", "dark-skinned female", "1girl", "2boys"] {
            assert!(is_guarded_tag(g), "{g}");
        }
        for n in ["animal ears", "fake animal ears", "long hair", "glasses", "multiple girls"] {
            assert!(!is_guarded_tag(n), "{n}");
        }
    }

    #[test]
    fn section_layouts_are_flattened_out_of_image_prompts() {
        // the 7B run: a tag list, then the same content again in section blocks
        let doubled = "1girl, cat ears, dark skin, glasses, [Scene / Environment & Shared Framing]: (masterpiece, best quality, cowboy shot) \
             A solo female character stands upright against a minimalistic light-colored backdrop., \
             [Character 1]: (1girl, solo, dark skin, glasses, black shorts)";
        assert_eq!(
            flatten_section_layout(doubled),
            "1girl, cat ears, dark skin, glasses, masterpiece, best quality, cowboy shot, solo, black shorts"
        );
        // the 3B run: a tag list used as the header
        assert_eq!(
            flatten_section_layout("[masterpiece, best quality, solo]: 1girl, smile"),
            "masterpiece, best quality, solo, 1girl, smile"
        );
        // a normal prompt is untouched, weights included
        for keep in ["1girl, (red hair:1.2), [blurry background], smile", "masterpiece, 1girl"] {
            assert_eq!(flatten_section_layout(keep), keep);
        }
        // and the image reader no longer asks for the layout
        assert!(!image_directives().contains("CORE OUTPUT STRUCTURE") && image_directives().contains("DIRECTIVE 6"));
    }

    #[test]
    fn leaked_instructions_are_stripped_from_image_prompts() {
        assert_eq!(
            strip_instruction_leaks("1girl, raccoon tail (inferred from tagger/context), 25-45 tags total, smile"),
            "1girl, raccoon tail, smile"
        );
        assert_eq!(strip_instruction_leaks("glasses, tag count: 40, boots (if visible)"), "glasses, boots");
        // weights and real parenthesised tags stay
        for keep in ["(red hair:1.2), 1girl", "kafuu chino (gochiusa), 1girl", "solo, (masterpiece)"] {
            assert_eq!(strip_instruction_leaks(keep), keep);
        }
    }

    #[test]
    fn combined_mode_always_asks_the_text_model() {
        use crate::profiles::{CheckpointProfile, DetailLevel, TargetView};
        let (host, log) = fake_ollama();
        let config = ScanConfig {
            checkpoint: CheckpointProfile::Anima, target_view: TargetView::None, pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
            camera_angle: 0, pose: 0, proportion: 0, dynamic: false, lora_tags_raw: String::new(), arranged: false, natural: false,
            two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false,
            scene_prompt_raw: String::new(), more_chars: Vec::new(), detail: DetailLevel::Balanced,
        };
        let req = ImageAiRequest {
            host, model: "prompt-condenser-gemma4-e4b".into(), vision_encoder: Some("huihui_ai/qwen2.5-vl-abliterated:3b".into()),
            mode: AiMode::Gpu, image_path: std::path::PathBuf::from("/nonexistent.png"), config,
            tag_hints: "1girl, green hair, thighhighs".into(), strong_tags: Vec::new(), second_pass_vision: false, scene_body_shape: false,
        };
        let out = describe_image_with_desc(&req, "A girl with green hair wearing black thigh-highs.", "qwen").unwrap();
        let calls: Vec<String> = log.lock().unwrap().iter().filter(|(p, _)| p == "/api/generate").map(|(_, b)| b.clone()).collect();
        // the visual check is off, yet the text model still writes the prompt (without the image)
        assert_eq!(calls.len(), 1, "the text model was not asked");
        let body: serde_json::Value = serde_json::from_str(&calls[0]).unwrap();
        assert_eq!(body["model"], "prompt-condenser-gemma4-e4b");
        assert!(body.get("images").is_none(), "no image without the 2nd Pass Visual Check");
        assert!(!body["prompt"].as_str().unwrap_or("").contains("An image is attached"), "no 'image attached' note");
        // this fake reply has no prompt, so the read falls back to the tag merge instead of failing
        assert!(out.prompt.contains("green hair"), "{}", out.prompt);
    }

    #[test]
    fn only_big_models_get_the_short_keep_alive() {
        assert!(is_big_model("huihui_ai/gemma-4-abliterated:12b"));
        assert!(!is_big_model("prompt-condenser-gemma4-e4b"));
        assert!(!is_big_model("richardyoung/llama-3.1-8b-instruct-abliterated"));
        assert!(!is_big_model("huihui_ai/qwen2.5-vl-abliterated:3b"));
        // a model that isn't loaded is left alone (fake server: only prompt-condenser-low is loaded)
        let (host, log) = fake_ollama();
        set_keep_alive(&host, "huihui_ai/gemma-4-abliterated:12b", "2m");
        assert!(log.lock().unwrap().iter().all(|(path, _)| path != "/api/generate"));
        set_keep_alive(&host, "prompt-condenser-low", "2m");
        assert!(log.lock().unwrap().iter().any(|(path, body)| path == "/api/generate" && body.contains("\"2m\"")));
    }

    #[test]
    fn vision_retry_is_a_real_second_attempt() {
        use crate::prompt_condenser::{VLM_RETRY_TEMP_BUMP, VLM_TEMPERATURE};
        let (t1, s1) = vlm_sampling(VLM_TEMPERATURE, VLM_TEMPERATURE);
        let (t2, s2) = vlm_sampling(VLM_TEMPERATURE + VLM_RETRY_TEMP_BUMP, VLM_TEMPERATURE);
        // a warmer configured first try is still a first try
        assert_eq!(vlm_sampling(0.3, 0.3).1, s1);
        assert!(t1 <= 0.05, "first try stays near-deterministic: {t1}");
        assert!(t2 > t1 + 0.1, "the retry really is warmer: {t2} vs {t1}");
        assert_ne!(s1, s2, "the retry uses another seed");
    }

    #[test]
    fn scan_budgets_expire() {
        assert!(!ScanBudget::new(Duration::from_secs(60)).spent());
        assert!(ScanBudget::new(Duration::from_secs(0)).spent());
        // the limits the app uses
        assert_eq!(DEEP_SCAN_BUDGET, Duration::from_secs(180));
        assert_eq!(MULTIZONE_BUDGET, Duration::from_secs(60));
        assert_eq!(MULTIZONE_MAX_ZONE_CALLS, 6);
    }

    #[test]
    fn dedicated_encoders_are_recognized_and_unloaded_quickly() {
        assert!(is_toriigate("hf.co/DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M") && is_toriigate("minthy/torii_gate") && !is_toriigate("gemma"));
        for enc in ["moondream:1.8b", "huihui_ai/qwen2.5-vl-abliterated:3b", "llava:7b", "Qwen2.5-VL-7B", "hf.co/DraconicDragon/ToriiGate-0.5-GGUF:Q4_K_M"] {
            assert!(is_dedicated_encoder(enc), "{enc}");
            assert_eq!(encoder_keep_alive(enc), "30s", "{enc}");
        }
        // text models (even ones that can see) stay loaded for the next scan
        for text in ["huihui_ai/gemma-4-abliterated:12b", "prompt-condenser-gemma4-e4b", "richardyoung/llama-3.1-8b-instruct-abliterated"] {
            assert!(!is_dedicated_encoder(text), "{text}");
            assert_eq!(encoder_keep_alive(text), "30m", "{text}");
        }
        // only dedicated encoders that are actually loaded get unloaded, never anything else
        let (host, log) = fake_ollama(); // /api/ps lists only prompt-condenser-low
        unload_dedicated_encoders(&format!("http://{host}"), &["moondream:1.8b", "prompt-condenser-low"], "x");
        assert!(log.lock().unwrap().iter().all(|(path, _)| path != "/api/generate"), "{:?}", log.lock().unwrap());
    }

    #[test]
    fn dual_pass_honors_the_selected_mode() {
        use crate::prompt_condenser::{Mode, PipelineOptions};
        let (host, log) = fake_ollama();
        let dir = std::env::temp_dir().join(format!("pc-mode-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let img = dir.join("t.png");
        image::DynamicImage::new_rgb8(64, 96).save(&img).unwrap();
        for (mode, wanted, other) in [
            (Mode::ScenePose, Mode::ScenePose.pass1_prompt(), Mode::SceneShape.pass1_prompt()),
            (Mode::SceneShape, Mode::SceneShape.pass1_prompt(), Mode::ScenePose.pass1_prompt()),
        ] {
            log.lock().unwrap().clear();
            let opts = PipelineOptions { mode, use_llm_synthesis: false, gemma_vision_pass: false };
            let _ = run_dual_pass_asymmetric_pipeline(
                &host, AiMode::Gpu, "moondream:1.8b", "moondream:1.8b", "prompt-condenser-low", &img, "1girl, standing", opts,
            );
            let prompts: Vec<String> = log
                .lock()
                .unwrap()
                .iter()
                .filter(|(path, _)| path == "/api/generate")
                .filter_map(|(_, body)| serde_json::from_str::<serde_json::Value>(body).ok())
                .filter_map(|v| v["prompt"].as_str().map(str::to_string))
                .collect();
            assert!(prompts.iter().any(|p| p == wanted), "{mode:?} never sent its own pass-1 prompt: {prompts:?}");
            assert!(!prompts.iter().any(|p| p == other), "{mode:?} sent the other mode's pass-1 prompt");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn talks_to_ollama_api() {
        use crate::profiles::{CheckpointProfile, TargetView};
        let (host, log) = fake_ollama();

        let (base, models) = list_ollama_models(&host).unwrap();
        assert_eq!(base, format!("http://{host}"));
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["prompt-condenser-low:latest", "qwen2.5:7b"]);

        let req = AiScanRequest {
            raw_prompt: "1girl, upper body, ass".into(),
            config: ScanConfig {
                checkpoint: CheckpointProfile::Anima,
                target_view: TargetView::UpperBody,
                pinned_emphasis_raw: String::new(),
                emphasis_weight: 1.25,
                cleanup_enabled: true,
                resolve_conflicts_enabled: true,
                clamp_weights_enabled: true,

                camera_angle: 0,

                pose: 0,
                proportion: 0,

                dynamic: false,

                lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),

                detail: crate::profiles::DetailLevel::Balanced,
            },
            mode: AiMode::Gpu,
            host: host.clone(),
            model: "prompt-condenser-low".into(),
            keep_alive: "30m".into(),
            think: false,
        };
        let (out, placement) = run_ai_scan(&req).unwrap();
        assert_eq!(out.changes[0].suggested_alternative, "curvy");
        // the fake reply has no scene analysis, negative or usable card, so it was re-scanned twice
        assert_eq!(placement, "GPU 100% • re-scanned 2× • still missing: scene analysis, negative prompt, cards");

        let (tx, rx) = std::sync::mpsc::channel();
        pull_tier_model(&host, AiModelTier::Good, tx).unwrap();
        let progress: Vec<String> = rx.try_iter().collect();
        assert!(progress.iter().any(|m| m.contains("(50%)")));

        let log = log.lock().unwrap();
        let paths: Vec<&str> = log.iter().map(|(p, _)| p.as_str()).collect();
        // 1 scan + 2 re-scans for the incomplete fake reply
        assert_eq!(
            paths,
            ["/api/tags", "/api/generate", "/api/generate", "/api/generate", "/api/ps", "/api/pull", "/api/copy", "/api/delete"]
        );
        let copy: serde_json::Value = serde_json::from_str(&log[6].1).unwrap();
        assert_eq!(copy["destination"], AiModelTier::Good.ollama_name());
        assert_eq!(copy["source"], AiModelTier::Good.hf_ref());
        let generate: serde_json::Value = serde_json::from_str(&log[1].1).unwrap();
        assert_eq!(generate["format"]["properties"]["changes"]["type"], "array");
        assert_eq!(generate["keep_alive"], "30m");
    }

    /// Talks to a real Ollama: `cargo test live_ollama -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_ollama() {
        let host = std::env::var("OLLAMA_TEST_HOST").unwrap_or_else(|_| "localhost:11434".into());
        println!("{:?}", list_ollama_models(&host));
        let model = std::env::var("OLLAMA_TEST_MODEL").unwrap_or_else(|_| "prompt-condenser-low".into());
        let req = AiScanRequest {
            raw_prompt: std::env::var("OLLAMA_TEST_PROMPT").unwrap_or_else(|_| "masterpiece, 8k, 1girl, upper body, (ass:1.4), looking at viewer, silver hair, booty shorts".into()),
            config: ScanConfig {
                checkpoint: crate::profiles::CheckpointProfile::Anima,
                target_view: crate::profiles::TargetView::UpperBody,
                pinned_emphasis_raw: String::new(),
                emphasis_weight: 1.25,
                cleanup_enabled: true,
                resolve_conflicts_enabled: true,
                clamp_weights_enabled: true,

                camera_angle: 0,

                pose: 0,
                proportion: 0,

                dynamic: false,

                lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),

                detail: crate::profiles::DetailLevel::Balanced,
            },
            mode: AiMode::Gpu,
            host,
            model,
            keep_alive: "30m".into(),
            think: false,
        };
        let t = std::time::Instant::now();
        let result = run_ai_scan(&req);
        println!("took {:?}", t.elapsed());
        let Ok((out, placement)) = result else { println!("{result:?}"); return };
        println!("placement: {placement}\nSUMMARY: {}\nBREAKDOWN: {:#?}\nISSUES: {:#?}", out.scene_summary, out.scene_breakdown, out.issues);
        println!("IMPROVED (tidied): {}", crate::condenser::tidy_ai_prompt(&out.improved_prompt, req.config.checkpoint));
        println!("NEGATIVE (raw): {:?}", out.recommended_negative);
        let toks = crate::parser::parse_prompt(&req.raw_prompt);
        let rules = crate::analyzer::analyze_with_rules(&toks, &req.config);
        for c in convert_ai_changes(&out.changes, &toks, &rules.changes) {
            println!("CARD {} | {:?} -> {:?} | alt {:?}", c.reason, c.original, c.suggested_alternative, c.framing_alternative);
        }
    }

    #[test]
    fn cut_off_reply_keeps_complete_entries() {
        let raw = r#"{"changes":[{"original":"ass","action":"framing_safe_swap","reason":"pulls {camera} down","suggested_alternative":"curvy","framing_alternative":""},{"original":"feet","action":"framing_safe_swap","reason":"zooms o"#;
        let out = extract_json_from_llm(raw).unwrap();
        assert_eq!(out.changes.len(), 1);
        assert_eq!(out.changes[0].original, "ass");
        assert!(extract_json_from_llm("total nonsense").is_none());
    }

    #[test]
    fn cut_off_reply_keeps_analysis_and_rewrite() {
        let raw = r#"{"scene_summary":"A woman on a rooftop at night","scene_breakdown":{"subject":"woman","pose":"standing","framing":"upper body","setting":"rooftop","lighting":"neon","mood":"moody"},"issues":["ass pulls camera","no lighting dir"],"improved_prompt":"1girl, upper body, neon rim light","changes":[{"original":"ass","action":"framing_safe_swap","reason":"r","suggested_alternative":"curvy","framing_alternative":""},{"original":"8k","action":"rem"#;
        let out = extract_json_from_llm(raw).unwrap();
        assert_eq!(out.scene_summary, "A woman on a rooftop at night");
        assert_eq!(out.scene_breakdown.lighting, "neon");
        assert_eq!(out.issues.len(), 2);
        assert_eq!(out.improved_prompt, "1girl, upper body, neon rim light");
        assert_eq!(out.changes.len(), 1);
    }

    #[test]
    fn incomplete_scans_are_detected_and_merged() {
        let req = AiScanRequest {
            raw_prompt: "1girl, smile, beach".into(),
            config: ScanConfig {
                checkpoint: crate::profiles::CheckpointProfile::PonyV6, target_view: crate::profiles::TargetView::None, pinned_emphasis_raw: String::new(),
                emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
                camera_angle: 0, pose: 0, proportion: 0, dynamic: false, lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),
                detail: crate::profiles::DetailLevel::Balanced,
            },
            mode: AiMode::Gpu, host: String::new(), model: String::new(), keep_alive: "30m".into(), think: false,
        };
        let card = AiChangeItem {
            original: String::new(), action: "add".into(), reason: "light".into(),
            suggested_alternative: "sunset".into(), framing_alternative: String::new(),
        };
        let complete = AiJsonOutput {
            scene_summary: "A girl smiling on a beach at sunset.".into(),
            scene_breakdown: SceneBreakdown {
                subject: "1girl".into(), pose: "standing".into(), framing: "cowboy shot".into(),
                setting: "beach".into(), lighting: "sunset".into(), mood: "happy".into(),
            },
            improved_prompt: "score_9, 1girl, smile, beach, sunset".into(),
            recommended_negative: "worst quality, low quality, blurry, extra fingers, bad hands, indoors".into(),
            changes: vec![card],
            ..Default::default()
        };
        assert!(scan_problems(&complete, &req).is_empty(), "{:?}", scan_problems(&complete, &req));

        let mut broken = AiJsonOutput { improved_prompt: complete.improved_prompt.clone(), ..Default::default() };
        assert_eq!(scan_problems(&broken, &req), vec!["scene analysis", "negative prompt", "cards"]);
        merge_scan(&mut broken, complete.clone(), &req);
        assert!(scan_problems(&broken, &req).is_empty());
    }

    /// Danbooru translator on real phrases:
    /// `TRANSLATE_MODEL=name cargo test live_translate -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_translate() {
        let model = std::env::var("TRANSLATE_MODEL").unwrap_or_else(|_| "richardyoung/llama-3.1-8b-instruct-abliterated".into());
        for (text, ckpt) in [
            ("a forlorn maiden sheltering beneath a parasol during a downpour, her tresses plastered to her cheeks", crate::profiles::CheckpointProfile::Illustrious),
            ("she's doing the thing where you peek out from behind a door, all shy, wearing those socks that go above the knee", crate::profiles::CheckpointProfile::Anima),
            ("a buxom warrior woman in a skimpy metal bikini, sweaty after battle, smirking smugly", crate::profiles::CheckpointProfile::PonyV6),
        ] {
            let t = std::time::Instant::now();
            let r = danbooru_translate(&TranslateRequest {
                host: "localhost:11434".into(), model: model.clone(), mode: AiMode::Gpu, text: text.into(), checkpoint: ckpt,
            }).unwrap();
            println!("\n[{:.0}s] {text}\n  TAGS   : {}\n  NATURAL: {}\n  FIXED  : {:?}\n  UNKNOWN: {:?}", t.elapsed().as_secs_f32(), r.tags, r.natural, r.fixed, r.unknown);
            for (f, to) in &r.mapping { println!("    “{f}” → {to}"); }
        }
    }

    /// Reads real images with a vision model, tagger hints included:
    /// `IMAGE_TEST=a.png cargo test live_image -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_image() {
        let model = std::env::var("IMAGE_MODEL").unwrap_or_else(|_| "prompt-condenser-gemma4-e4b:latest".into());
        let vision_encoder = match std::env::var("VISION_ENCODER").as_deref() {
            Ok("none") | Ok("direct") | Ok("self") => None,
            Ok(enc) => Some(enc.to_string()),
            Err(_) => Some("moondream:1.8b".into()),
        };
        for path in std::env::var("IMAGE_TEST").unwrap().split(':') {
            let path = std::path::PathBuf::from(path);
            let scored = crate::twoimg::tag_one_fast(&path, &|_| ()).unwrap_or_default();
            let hints = crate::tagger::build_prompt(&scored, 0.35).prompt;
            let strong = strong_tags(&scored);
            println!("STRONG  : {}", strong.join(", "));
            let config = ScanConfig {
                checkpoint: crate::profiles::CheckpointProfile::Anima, target_view: crate::profiles::TargetView::None, pinned_emphasis_raw: String::new(),
                emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
                camera_angle: 0, pose: 0, proportion: 0, dynamic: false, lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(), detail: crate::profiles::DetailLevel::Balanced,
            };
            let req = ImageAiRequest {
                host: "localhost:11434".into(),
                model: model.clone(),
                vision_encoder: vision_encoder.clone(),
                mode: AiMode::Gpu,
                image_path: path.clone(),
                config,
                tag_hints: hints.clone(),
                strong_tags: strong.clone(),
                second_pass_vision: false,
                scene_body_shape: false,
            };
            let t = std::time::Instant::now();
            let r = describe_image(&req).unwrap();
            println!("\n=======================================================");
            println!("IMAGE   : {}", path.display());
            println!("TIME    : {:.2}s (Model: {})", t.elapsed().as_secs_f32(), model);
            println!("TAGGER  : {hints}");
            println!("DESC    : {}", r.description);
            println!("PROMPT  : {}", r.prompt);
            println!("=======================================================\n");
        }
    }

    /// Downloads a tier into a real Ollama: `cargo test live_pull -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_pull() {
        let host = std::env::var("OLLAMA_TEST_HOST").unwrap_or_else(|_| "localhost:11434".into());
        let tier = match std::env::var("OLLAMA_TEST_TIER").as_deref() {
            Ok("good") | Ok("e4b") => AiModelTier::Good,
            Ok("better") | Ok("llama8b") | Ok("llama8") | Ok("llama") => AiModelTier::Better,
            Ok("best") | Ok("12b") => AiModelTier::Best,
            _ => AiModelTier::Better,
        };
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let printer = std::thread::spawn(move || {
            let mut last = String::new();
            for m in rx {
                if !m.ends_with("%)") || m.contains("0%)") {
                    println!("{m}");
                }
                last = m;
            }
            println!("{last}");
        });
        let result = pull_tier_model(&host, tier, tx);
        printer.join().unwrap();
        println!("{result:?}");
        result.unwrap();
    }



    #[test]
    fn negative_cleanup() {
        assert_eq!(
            clean_negative("worst quality, no smile, wrong framing, unwanted extra people, no humans, lack of detail, daytime, non-traditional elements"),
            "worst quality, extra people, no humans, daytime"
        );
        assert_eq!(clean_negative("bad_hands, wrong_camera_angle"), "bad hands");
    }

    /// Compares models on the same prompts:
    /// `COMPARE_MODELS=a,b cargo test live_compare -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_compare() {
        use crate::condenser::{dropped_tags, highlight_against_original, tidy_ai_prompt};
        use crate::profiles::{CheckpointProfile, TargetView};
        let models: Vec<String> = std::env::var("COMPARE_MODELS").unwrap().split(',').map(String::from).collect();
        let cases = [
            ("kitsune (Anima)", CheckpointProfile::Anima, TargetView::None,
             "((((masterpiece)))), ultra detailed, 8k, 1girl, solo, kitsune, fox ears, fox tail, long white hair, red eyes, kimono, off shoulder, sitting, upper body, (ass:1.3), shrine, night, cherry blossoms, looking at viewer, smile, detailed eyes, full body"),
            ("classroom (Illustrious)", CheckpointProfile::Illustrious, TargetView::None,
             "masterpiece, best quality, 1girl, solo, brown hair, ponytail, glasses, school uniform, sitting at desk, classroom, window, afternoon sunlight, reading book, calm"),
            ("knight from behind (Pony)", CheckpointProfile::PonyV6, TargetView::FromBehind,
             "score_9, score_8_up, masterpiece, 8k, 1girl, solo, knight, silver armor, red cape, holding sword, standing, castle, sunset, looking at viewer, cleavage, wind"),
            // uncensored check: explicit tags must survive and nothing may be refused or softened
            ("explicit (Pony)", CheckpointProfile::PonyV6, TargetView::None,
             "score_9, score_8_up, 1girl, solo, adult woman, nude, completely nude, nipples, large breasts, lying on bed, on back, spread legs, blush, sweat, bedroom, dim lighting"),
        ];
        let report = std::env::var("COMPARE_OUT").unwrap_or_else(|_| "/tmp/compare.txt".into());
        let mut full = String::new();
        println!("{:<26} {:<30} {:>6} {:>5} {:>8} {:>6} {:>5} {:>6}", "case", "model", "secs", "ok", "dropped", "added", "neg", "cards");
        for model in &models {
            for (name, ckpt, view, prompt) in cases {
                let req = AiScanRequest {
                    raw_prompt: prompt.into(),
                    config: ScanConfig {
                        checkpoint: ckpt, target_view: view, pinned_emphasis_raw: String::new(), emphasis_weight: 1.25,
                        cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
                        camera_angle: std::env::var("COMPARE_ANGLE").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
                        pose: 0,
                proportion: 0,
                        dynamic: std::env::var("COMPARE_DYNAMIC").is_ok(),
                        lora_tags_raw: std::env::var("COMPARE_LORA").unwrap_or_default(), arranged: false, natural: std::env::var("COMPARE_NATURAL").is_ok(), two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),
                        detail: match std::env::var("COMPARE_DETAIL").as_deref() { Ok("rich") => crate::profiles::DetailLevel::Rich, Ok("faithful") => crate::profiles::DetailLevel::Faithful, _ => crate::profiles::DetailLevel::Balanced },
                    },
                    mode: AiMode::Gpu,
                    host: "localhost:11434".into(),
                    model: model.clone(),
                    keep_alive: "30m".into(),
                    think: std::env::var("COMPARE_THINK").is_ok(),
                };
                let t = std::time::Instant::now();
                let r = run_ai_scan(&req);
                let secs = t.elapsed().as_secs_f32();
                match r {
                    Ok((out, place)) => {
                        let improved = tidy_ai_prompt(&out.improved_prompt, ckpt);
                        let dropped = dropped_tags(prompt, &improved, ckpt);
                        let (_, added) = highlight_against_original(prompt, &improved);
                        let toks = crate::parser::parse_prompt(prompt);
                        let rules = crate::analyzer::analyze_with_rules(&toks, &req.config);
                        let cards = convert_ai_changes(&out.changes, &toks, &rules.changes).len();
                        let neg = out.recommended_negative.split(',').filter(|t| !t.trim().is_empty()).count();
                        let explicit_kept = ["nude", "nipples", "spread legs"].iter().filter(|w| improved.contains(*w)).count();
                        let refusal = ["i can't", "i cannot", "i'm sorry", "not able to", "inappropriate"]
                            .iter()
                            .any(|w| out.scene_summary.to_lowercase().contains(w) || improved.to_lowercase().contains(w));
                        println!(
                            "{:<26} {:<30} {:>6.1} {:>5} {:>8} {:>6} {:>5} {:>6}{}",
                            name, model, secs, "yes", dropped.len(), added, neg, cards,
                            if name.starts_with("explicit") {
                                format!("   explicit tags kept {explicit_kept}/3{}", if refusal { ", REFUSED/SOFTENED" } else { ", no refusal" })
                            } else {
                                String::new()
                            }
                        );
                        full.push_str(&format!(
                            "\n===== {name} | {model} | {secs:.1}s | {place} | thinking {} chars\nSUMMARY: {}\nISSUES: {}\nIMPROVED: {}\nDROPPED: {}\nNEGATIVE: {}\nCARDS: {}\nDYNAMIC: {}\nNATURAL: {}\n",
                            out.thinking.len(), out.scene_summary, out.issues.join(" | "), improved, dropped.join(", "), out.recommended_negative,
                            out.changes.iter().map(|c| format!("[{}] '{}' -> '{}'", c.action, c.original, c.suggested_alternative)).collect::<Vec<_>>().join(" ; "),
                            out.dynamic_prompt, out.natural_sentence
                        ));
                    }
                    Err(e) => {
                        println!("{:<26} {:<30} {:>6.1} {:>5}  {}", name, model, secs, "FAIL", e);
                        full.push_str(&format!("\n===== {name} | {model} | FAILED: {e}\n"));
                    }
                }
            }
        }
        std::fs::write(&report, full).unwrap();
        println!("full outputs: {report}");
    }

    #[test]
    fn instructions_do_their_job() {
        use crate::profiles::{CheckpointProfile, DetailLevel, TargetView};
        let base = || ScanConfig {
            checkpoint: CheckpointProfile::PonyV6, target_view: TargetView::None, pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
            camera_angle: 0, pose: 0, proportion: 0, dynamic: false, lora_tags_raw: String::new(), arranged: false, natural: false,
            two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false,
            scene_prompt_raw: String::new(), more_chars: Vec::new(), detail: DetailLevel::Balanced,
        };
        // the Deep Scan has its own role and JSON reply; the image compiler's directives
        // (flags, [Scene]/[Character 1] layout, dual-image merge) must not leak into it
        for family in [ModelFamily::Gemma, ModelFamily::Generic, ModelFamily::Small] {
            let (deep, _) = build_ai_system_and_user_prompt_for("1girl, smile, beach", &base(), family);
            assert!(deep.contains("uncensored") && deep.contains("never refuse"), "{family:?}: no uncensored role");
            assert!(deep.contains("DIRECTIVE 1") && deep.contains("DIRECTIVE 5") && deep.contains("DIRECTIVE 7"), "{family:?}: rules missing");
            assert!(deep.contains("to the character they belong to"), "{family:?}: directive 2 not reworded");
            for leak in ["AntigravityAI", "[Character 1]", "[Scene]", "DIRECTIVE 6", "Image 1 Authority", "CORE OUTPUT STRUCTURE"] {
                assert!(!deep.contains(leak), "{family:?}: Deep Scan prompt contains {leak:?}");
            }
        }
        // ...while the image reader keeps its directives and the uncensored / never-refuse line
        let img = image_system_prompt(&base(), "");
        assert!(img.contains("uncensored") && img.contains("never refuse") && img.contains("DIRECTIVE 6"), "{img}");
        assert!(img.contains("more reliable than the visual description") && img.contains("dark skin"), "tagger rule missing");
        // image -> prompt: the quality tags are where the prompt says they are
        let sys = image_system_prompt(&base(), "");
        assert!(sys.contains("starting with the quality tags (score_9"), "{sys}");
        // the model writes the Deep Scan fields in the instructed order
        let schema = changes_schema(true, false, false, 8, 12).to_string();
        let at = |k: &str| schema.find(&format!("\"{k}\"")).unwrap();
        assert!(at("scene_summary") < at("improved_prompt") && at("improved_prompt") < at("changes"), "{schema}");
        assert!(!schema.contains("dynamic_prompt"));
        assert!(!schema.contains("natural_sentence"));
        // small models limit issues and changes
        let small_schema = changes_schema(false, true, true, 4, 6).to_string();
        assert!(small_schema.contains("\"maxItems\":4") && small_schema.contains("\"maxItems\":6"));
        // the Light model keeps the user's own choices
        let mut c = base();
        c.lora_tags_raw = "<lora:fox:0.8>".into();
        c.pinned_emphasis_raw = "red scarf".into();
        c.pose = 2;
        let (small, _) = build_ai_system_and_user_prompt_for("1girl, smile", &c, ModelFamily::Small);
        assert!(small.contains("<lora:fox:0.8>") && small.contains("red scarf") && small.contains("pose"), "{small}");
        assert!(!small.contains("Cleanup:"));
        // Faithful never gets told to expand
        let mut c = base();
        c.detail = DetailLevel::Faithful;
        let g = prompt_guidance("1girl, smile, red dress", &c).join("\n");
        assert!(!g.contains("Expand it") && !g.contains("Fill these in"), "{g}");
        // Rich isn't told to squeeze into 75 tokens
        c.detail = DetailLevel::Rich;
        let long = (0..40).map(|i| format!("tag{i} word")).collect::<Vec<_>>().join(", ");
        assert!(!prompt_guidance(&long, &c).join("\n").contains("75 tokens"));
        // "explicit" only for explicit words, not words that contain them
        for innocent in ["canal, cumulus clouds, floral print kimono, cucumber salad", "analog camera, coral reef, sussex"] {
            let g = prompt_guidance(innocent, &base()).join("\n");
            assert!(!g.contains("explicit"), "{innocent}: {g}");
        }
        assert!(prompt_guidance("1girl, nude, spread legs", &base()).join("\n").contains("explicit"));
        // 2others / 3others are characters
        assert!(prompt_guidance("2others, forest, mist", &base()).join("\n").contains("several characters"));
        // on a multi-character tab, character 1's LoRA doesn't strip the other characters' tags
        let text = "1girl, smile, blue background, sunlight, standing, upper body";
        let mut c = base();
        c.lora_tags_raw = "red background".into();
        // 1 Character: the LoRA's "red background" replaces the prompt's "blue background"
        let one = prompt_guidance(text, &c).join("\n");
        assert!(one.contains("setting / background"), "{one}");
        // several characters: the first LoRA box only speaks for character 1, so the setting counts
        c.two_char = true;
        c.char2_prompt_raw = "1girl, grey eyes".into();
        let many = prompt_guidance(text, &c).join("\n");
        assert!(!many.contains("setting / background") && many.contains("Already specified"), "{many}");
        // the 75-token note only exists for CLIP checkpoints: not for Anima or Flux
        let long = (0..60).map(|i| format!("tag{i} word")).collect::<Vec<_>>().join(", ");
        let mut c = base();
        c.checkpoint = CheckpointProfile::Anima;
        assert!(!prompt_guidance(&long, &c).join("\n").contains("The prompt is long"), "Anima has no 75-token limit");
        c.checkpoint = CheckpointProfile::Flux;
        assert!(!prompt_guidance(&long, &c).join("\n").contains("The prompt is long"));
        c.checkpoint = CheckpointProfile::PonyV6;
        assert!(prompt_guidance(&long, &c).join("\n").contains("75 tokens"));
        c.two_char = true;
        c.char2_prompt_raw = "1girl, blue hair".into();
        // two characters get 75 tokens each: 120 tokens is fine, 240 is long
        assert!(!prompt_guidance(&long, &c).join("\n").contains("The prompt is long"));
        let longer = (0..120).map(|i| format!("tag{i} word")).collect::<Vec<_>>().join(", ");
        assert!(prompt_guidance(&longer, &c).join("\n").contains("for 2 characters"));
        // multi-character: the notes read the Scene box too, and the example has BREAK, not solo
        let mut c = base();
        c.two_char = true;
        c.scene_prompt_raw = "park, sunset, upper body".into();
        c.char2_prompt_raw = "1girl, blue hair".into();
        let (sys, _) = build_ai_system_and_user_prompt_for("1girl, red hair", &c, ModelFamily::Gemma);
        assert!(!sys.contains("Not specified yet: camera framing, setting"), "{sys}");
        assert!(sys.contains("BREAK, \\\n") || sys.contains("standing, BREAK"), "example should show BREAK");
        assert!(sys.contains("Character 1's LoRA") || c.lora_tags_raw.is_empty());
    }

    #[test]
    fn guidance_adapts_to_the_prompt() {
        use crate::profiles::{CheckpointProfile, TargetView};
        let cfg = |view| ScanConfig {
            checkpoint: CheckpointProfile::Anima, target_view: view, pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
 camera_angle: 0,
 pose: 0,
 proportion: 0,
 dynamic: false,
 lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),
 detail: crate::profiles::DetailLevel::Balanced,
        };
        let joined = |p: &str, v| prompt_guidance(p, &cfg(v)).join("\n");

        let g = joined("no humans, scenery, forest, river, mountain, mist, pine trees, rocks, moss, waterfall", TargetView::None);
        assert!(g.contains("scenery") && g.contains("Never add characters"), "{g}");

        let g = joined("2girls, hatsune miku, kagamine rin, stage, concert, smile, singing, microphone, spotlight, crowd", TargetView::None);
        assert!(g.contains("several characters (2girls)"), "{g}");

        let g = joined("1girl, <lora:flatcolor_v2:0.8>, flatcolor, smile", TargetView::None);
        assert!(g.contains("<lora:flatcolor_v2:0.8>") && g.contains("short"), "{g}");

        let g = joined("1girl, solo, upper body, ass, smile, kimono, shrine, night, moonlight, sitting", TargetView::None);
        assert!(g.contains("framing conflicts with: ass"), "{g}");
        assert!(g.contains("Already specified:") && g.contains("light / time"), "{g}");

        // the guidance ends up in the system prompt
        let (sys, _) = build_ai_system_and_user_prompt("no humans, scenery, forest", &cfg(TargetView::None));
        assert!(sys.contains("ABOUT THIS PROMPT"));
    }

    #[test]
    fn instructions_follow_the_model_family() {
        use crate::profiles::{CheckpointProfile, TargetView};
        assert_eq!(ModelFamily::detect("prompt-condenser-gemma4-12b-q6:latest"), ModelFamily::Gemma);
        assert_eq!(ModelFamily::detect("prompt-condenser-qwen3-14b"), ModelFamily::Qwen3);
        assert_eq!(ModelFamily::detect("prompt-condenser-gemma4-e2b"), ModelFamily::Small);
        assert_eq!(ModelFamily::detect("prompt-condenser-gemma4-e4b"), ModelFamily::Gemma);
        assert_eq!(ModelFamily::detect("llama3.2:3b"), ModelFamily::Small);
        assert_eq!(ModelFamily::detect("qwen2.5:14b"), ModelFamily::Qwen25);
        assert_eq!(ModelFamily::detect("mistral-nemo"), ModelFamily::Generic);
        assert_eq!(ModelFamily::detect("qwen2.5:32b"), ModelFamily::Qwen25);
        assert_eq!(ModelFamily::detect("prompt-condenser-gemma4-12b"), ModelFamily::Gemma);
        let mut cfg = ScanConfig {
            checkpoint: CheckpointProfile::Anima, target_view: TargetView::None, pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
            camera_angle: 6, pose: 0, proportion: 0, dynamic: true, lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(), detail: crate::profiles::DetailLevel::Rich,
        };
        let p = "1girl, solo, smile, kimono, shrine, night";
        let (g, gu) = build_ai_system_and_user_prompt_for(p, &cfg, ModelFamily::Gemma);
        assert!(g.contains("## Task") && g.contains("```json") && g.contains("dynamic_prompt") && g.contains("Dutch angle"));
        assert!(g.contains("DETAIL LEVEL = RICH") && g.contains("CHARACTER (hair"));
        assert!(gu.contains("JSON object only"));
        cfg.dynamic = false;
        let (q, _) = build_ai_system_and_user_prompt_for(p, &cfg, ModelFamily::Qwen25);
        assert!(q.contains("STEPS") && !q.contains("dynamic_prompt") && !q.contains("```"));
        let (sm, _) = build_ai_system_and_user_prompt_for(p, &cfg, ModelFamily::Small);
        assert!(sm.len() < q.len() && sm.contains("up to 6 individual edits"));
        assert!(g.contains("DIRECTIVE 1: CAMERA & SUBMERSION FRAMING") && g.contains("DIRECTIVE 7: VLM + TAGGER SYNTHESIS"));

        // AntigravityTagFilter runs on raw prompt tags before the LLM handoff
        let (_, user_msg) = build_ai_system_and_user_prompt_for(
            "(1girl:1.2), bunny girl, maid, white shirt, thighhighs, standing",
            &cfg,
            ModelFamily::Gemma,
        );
        assert!(user_msg.contains("full body"));
        assert!(!user_msg.contains("bunny girl"));
        assert!(user_msg.contains("MISSING_BOTTOMS_FLAG"));
        assert!(user_msg.contains("UNANCHORED_LIMBS_FLAG"));
    }

    #[test]
    #[ignore]
    fn live_dual_pass() {
        let char_path = std::path::Path::new("packaging/arch/character.jpg");
        let scene_path = std::path::Path::new("packaging/arch/scene.jpg");
        assert!(char_path.exists(), "character.jpg exists");
        assert!(scene_path.exists(), "scene.jpg exists");
        assert!(crate::tagger::is_installed(), "tagger is installed");

        println!("\n>>> STEP 1: WD14 Tagging on character.jpg and scene.jpg...");
        let char_tags = crate::tagger::tag_image(char_path).expect("WD14 character tagging failed");
        let raw_char_wd14 = crate::tagger::build_prompt(&char_tags, crate::prompt_condenser::WD14_GENERAL_THRESHOLD).prompt;

        let scene_tags = crate::tagger::tag_image(scene_path).expect("WD14 scene tagging failed");
        let raw_scene_wd14 = crate::tagger::build_prompt(&scene_tags, crate::prompt_condenser::WD14_GENERAL_THRESHOLD).prompt;

        let opts_shape = crate::prompt_condenser::PipelineOptions {
            mode: crate::prompt_condenser::Mode::SceneShape,
            use_llm_synthesis: true,
            gemma_vision_pass: true,
        };
        let opts_pose = crate::prompt_condenser::PipelineOptions {
            mode: crate::prompt_condenser::Mode::ScenePose,
            use_llm_synthesis: true,
            gemma_vision_pass: true,
        };

        println!("\n================================================================================");
        println!(">>> [MODE A: SCENE + BODY SHAPE] - Scene provides Environment, Pose & Silhouette");
        println!("================================================================================");
        println!(">>> Running Mode A on scene.jpg...");
        let scene_shape_res = run_dual_pass_pipeline(
            "http://localhost:11434",
            AiMode::Gpu,
            "huihui_ai/qwen2.5-vl-abliterated:3b",
            "huihui_ai/gemma-4-abliterated:12b",
            scene_path,
            &raw_scene_wd14,
            opts_shape,
        );

        println!(">>> Running Mode A on character.jpg...");
        let char_shape_res = run_dual_pass_pipeline(
            "http://localhost:11434",
            AiMode::Gpu,
            "huihui_ai/qwen2.5-vl-abliterated:3b",
            "huihui_ai/gemma-4-abliterated:12b",
            char_path,
            &raw_char_wd14,
            opts_shape,
        );

        // Merge Mode A (scene_body_shape = true)
        let combined_scene_a = format!("{}, {}", scene_shape_res.scene_prompt, scene_shape_res.character_prompt);
        let actions_a = crate::sceneref::from_single_with_shape(&combined_scene_a, false, true);
        let plan_a = crate::sceneref::plan(&actions_a, &crate::sceneref::Overrides::default(), 1);
        let combined_char_a = format!("{}, {}", char_shape_res.character_prompt, char_shape_res.scene_prompt);
        let char_id_a = crate::scene::filter_identity_only_with_shape(&combined_char_a, true);
        let merged_char_a = crate::sceneref::inject(&char_id_a, &plan_a.chars[0], &crate::sceneref::is_action);

        println!("\n================================================================================");
        println!(">>> [MODE B: SCENE + SKELETAL POSE] - Scene provides Environment & Pure Skeleton");
        println!("================================================================================");
        println!(">>> Running Mode B on scene.jpg...");
        let scene_pose_res = run_dual_pass_pipeline(
            "http://localhost:11434",
            AiMode::Gpu,
            "huihui_ai/qwen2.5-vl-abliterated:3b",
            "huihui_ai/gemma-4-abliterated:12b",
            scene_path,
            &raw_scene_wd14,
            opts_pose,
        );

        println!(">>> Running Mode B on character.jpg...");
        let char_pose_res = run_dual_pass_pipeline(
            "http://localhost:11434",
            AiMode::Gpu,
            "huihui_ai/qwen2.5-vl-abliterated:3b",
            "huihui_ai/gemma-4-abliterated:12b",
            char_path,
            &raw_char_wd14,
            opts_pose,
        );

        // Merge Mode B (scene_body_shape = false)
        let combined_scene_b = format!("{}, {}", scene_pose_res.scene_prompt, scene_pose_res.character_prompt);
        let actions_b = crate::sceneref::from_single_with_shape(&combined_scene_b, false, false);
        let plan_b = crate::sceneref::plan(&actions_b, &crate::sceneref::Overrides::default(), 1);
        let combined_char_b = format!("{}, {}", char_pose_res.character_prompt, char_pose_res.scene_prompt);
        let char_id_b = crate::scene::filter_identity_only_with_shape(&combined_char_b, false);
        let merged_char_b = crate::sceneref::inject(&char_id_b, &plan_b.chars[0], &crate::sceneref::is_action);

        println!("\n================================================================================");
        println!(">>> DUAL-REFERENCE MERGE COMPARISON (Character + Scene)");
        println!("================================================================================");
        println!("\n--- [MODE A: SCENE + BODY SHAPE] ---");
        println!("SCENE PROMPT:\n  {}", plan_a.scene.join(", "));
        println!("MERGED CHARACTER (Wardrobe from Char, Body Shape & Pose from Scene):\n  {}", merged_char_a);

        println!("\n--- [MODE B: SCENE + SKELETAL POSE] ---");
        println!("SCENE PROMPT:\n  {}", plan_b.scene.join(", "));
        println!("MERGED CHARACTER (Wardrobe & Body Shape from Char, Pure Skeletal Pose from Scene):\n  {}", merged_char_b);
        println!("================================================================================");
    }

    #[test]
    #[ignore]
    fn live_before_and_after_quadrant_test() {
        use std::path::Path;
        use std::time::Instant;

        let img_path_str = std::env::var("TEST_IMAGE")
            .unwrap_or_else(|_| "tests/eval_dataset/HWP1dffawAA5Y7K.jpg".to_string());
        let img_path = Path::new(&img_path_str);
        if !img_path.is_file() {
            println!("Image not found at {}, skipping live test.", img_path.display());
            return;
        }

        println!("\n================================================================================");
        println!(">>> LIVE BEFORE & AFTER BENCHMARK ON: {}", img_path.display());
        println!("================================================================================");

        // ==========================================
        // 1. BEFORE: Legacy Single-Pass Full Frame
        // ==========================================
        println!("\n[1/2] RUNNING 'BEFORE' (Single-Pass Full Frame, No Quadrants, No 3-Zone Crops)...");
        let t_before_start = Instant::now();

        // Single pass WD14 on full image only
        let img_dynamic = image::open(img_path).expect("open image");
        let raw_full_scores = crate::tagger::tag_dynamic(&img_dynamic).expect("tag full");
        let before_wd14_prompt = crate::tagger::build_prompt(&raw_full_scores, 0.22).prompt;

        // Legacy full frame vision prompt at temp 0.20
        use base64::Engine as _;
        let b64_img = base64::engine::general_purpose::STANDARD.encode(shrink_for_vision(img_path).unwrap());
        let legacy_ask = "Describe the subject, appearance, clothing, pose, framing, setting, light and style.";
        let before_vision_desc = ask_vision_encoder("http://localhost:11434", AiMode::Gpu, "huihui_ai/qwen2.5-vl-abliterated:3b", b64_img, legacy_ask).unwrap_or_default();
        let before_prose_tags = crate::danbooru::extract_tags_from_prose(&before_vision_desc);

        // Merge before tags
        let mut before_merged_tags = before_wd14_prompt.clone();
        for pt in &before_prose_tags {
            if !before_merged_tags.to_lowercase().contains(&pt.to_lowercase()) {
                before_merged_tags.push_str(", ");
                before_merged_tags.push_str(pt);
            }
        }
        let before_elapsed = t_before_start.elapsed().as_secs_f32();

        // ==========================================
        // 2. AFTER: 4-Quadrant + 3-Zone Auto-Detector
        // ==========================================
        println!("\n[2/2] RUNNING 'AFTER' (4-Quadrant Objects + 3-Zone Character Pinpointing at temp 0.05)...");
        let t_after_start = Instant::now();

        // Upgraded multi-zone + 4-quadrant WD14 tagger
        let noop_progress = |_: String| {};
        let after_scores = crate::twoimg::tag_one_fast(img_path, &noop_progress).expect("tag after");
        let after_wd14_prompt = crate::tagger::build_prompt(&after_scores, 0.22).prompt;

        // Upgraded 4-quadrant + universal 3-zone Qwen 2.5-VL scan
        let after_vision_desc = extract_visual_description_from_path(
            img_path,
            "http://localhost:11434",
            AiMode::Gpu,
            "huihui_ai/qwen2.5-vl-abliterated:3b",
        ).expect("extract after vision");
        let after_prose_tags = crate::danbooru::extract_tags_from_prose(&after_vision_desc);

        // Merge after tags
        let mut after_merged_tags = after_wd14_prompt.clone();
        for pt in &after_prose_tags {
            if !after_merged_tags.to_lowercase().contains(&pt.to_lowercase()) {
                after_merged_tags.push_str(", ");
                after_merged_tags.push_str(pt);
            }
        }
        let after_deduped = crate::danbooru::dedup_color_parents(&after_merged_tags);
        let after_ordered = crate::danbooru::sort_canonical_buckets(&after_deduped);
        let after_elapsed = t_after_start.elapsed().as_secs_f32();

        println!("\n================================================================================");
        println!(">>> BEFORE VS AFTER SCAN RESULTS COMPARISON");
        println!("================================================================================");
        println!("BEFORE (Single Pass): {:.2}s, {} tags", before_elapsed, before_merged_tags.split(',').count());
        println!("AFTER  (4-Quadrant + 3-Zone): {:.2}s, {} tags", after_elapsed, after_ordered.split(',').count());

        println!("\n--- [BEFORE: VISUAL DESCRIPTION (Single Pass Full Frame)] ---");
        println!("{}", before_vision_desc);
        println!("\n--- [BEFORE: EXTRACTED TAGS] ---");
        println!("{}", before_merged_tags);

        println!("\n--- [AFTER: MULTI-ZONE & QUADRANT PINPOINTED DESCRIPTION] ---");
        println!("{}", after_vision_desc);
        println!("\n--- [AFTER: CANONICALLY ORDERED PINPOINTED TAGS] ---");
        println!("{}", after_ordered);

        // Identify new tags discovered exclusively by the After pass
        let before_lower = before_merged_tags.to_lowercase();
        let new_discoveries: Vec<&str> = after_ordered
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty() && !before_lower.contains(&t.to_lowercase()))
            .collect();

        println!("\n--- [NEW EXCLUSIVE DISCOVERIES BY AFTER PASS] ---");
        println!("({} new pinpointed tags discovered):", new_discoveries.len());
        for nd in &new_discoveries {
            println!("  + {}", nd);
        }
        println!("================================================================================");
    }
}

