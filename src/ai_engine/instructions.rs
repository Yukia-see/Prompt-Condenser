//! Part of `ai_engine`; see mod.rs.
use super::*;

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
pub(crate) fn image_directives() -> String {
    let all = IMAGE_COMPILER_DIRECTIVES;
    match (all.find("# CORE OUTPUT STRUCTURE"), all.find("# DIRECTIVE 1")) {
        (Some(a), Some(b)) => format!("{}{}", &all[..a], &all[b..]),
        _ => all.to_string(),
    }
}

/// The compiler directives as the Deep Scan needs them: the flag / layering / framing rules
/// (directives 1-5 and 7), without the image reader's persona, its `[Scene]` / `[Character 1]`
/// layout (the Deep Scan answers in JSON) and the two-image directive 6 (it has no images).
pub(crate) fn deep_scan_directives() -> String {
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

/// Words that mark explicit content (so the AI keeps it precise instead of softening it).
pub(crate) const EXPLICIT_WORDS: &[&str] = &[
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

pub(crate) fn param_billions(name: &str) -> Option<f32> {
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

    pub(crate) fn max_issues_changes(&self) -> (usize, usize) {
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
        (crate::profiles::BUSTS, config.bust, "bust size"),
        (crate::profiles::FRAMES, config.frame, "body frame"),
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
        // (LoRA tags, Keep list, angle / pose / bust / frame picks): only generic lines go
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
pub(crate) fn wants_sentence(config: &ScanConfig) -> bool {
    config.natural && config.checkpoint.supports_natural_language()
}

/// The improved_prompt task, worded for the chosen detail level.
pub(crate) fn improved_prompt_step(config: &ScanConfig) -> String {
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
