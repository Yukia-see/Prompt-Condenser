//! Part of `ai_engine`; see mod.rs.
use super::*;

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
    /// The tagger's breast-size tags at >= 0.5 with their scores (`confident_size_tags`): they win over the writer's size.
    pub size_tags: Vec<(String, f32)>,
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
pub(crate) const CHARACTER_CHECKLIST: &str = "Capture each character exactly: hair (length, color, bangs or parting, \
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

pub(crate) fn image_system_prompt(config: &ScanConfig, tag_hints: &str) -> String {
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
           * Include contrast meta-tags in the scene header only when they really differ: 'different hair colors', 'contrasting poses'.\n\
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
pub(crate) fn scene_pose_system_prompt(tag_hints: &str, scene_body_shape: bool) -> String {
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
    // No separate vision model: the same sketch / panel notes and cleanup as the Combined path
    let panels = image::open(&req.image_path).map(|i| panel_count(&i)).unwrap_or(1);
    let ask = format!("Write the prompt for this image.{}", image_style_note(&req.tag_hints, panels));
    let mut out = describe_image_with(req, &image_system_prompt(&req.config, &req.tag_hints), &ask)?;
    if std::env::var_os("PC_NO_POLISH").is_none() {
        out.prompt = polish_image_prompt_with(&out.prompt, "", &req.strong_tags, &req.tag_hints, &req.size_tags);
        out.prompt = apply_image_style(&out.prompt, &req.tag_hints, panels);
    }
    Ok(out)
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
         framing and the background. Do not summarize or drop details. Leave out character names, scores and notes. \
         For items the tagger detected and you agree with, copy the tagger's tag spelling word for word instead of rewording it.";
    let panels = image::open(&req.image_path).map(|i| panel_count(&i)).unwrap_or(1);
    let keep_details = format!("{keep_details}{}", image_style_note(&req.tag_hints, panels));
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
    check_cancel()?;
    let mut res = match describe_with_text_model(req, &system, &user_prompt) {
        Ok(r) => r,
        Err(_) => tag_merge(),
    };
    // PC_NO_POLISH=1 turns the last pass off, to measure what it adds (testing only)
    if std::env::var_os("PC_NO_POLISH").is_none() {
        res.prompt = polish_image_prompt_with(&res.prompt, vision_desc, &req.strong_tags, &req.tag_hints, &req.size_tags);
        res.prompt = apply_image_style(&res.prompt, &req.tag_hints, panels);
    }
    Ok(res)
}

/// Encodes an in-memory DynamicImage to JPEG base64 string, scaling down if over 1024px.
pub(crate) fn encode_dynamic_image_b64(img: &image::DynamicImage) -> Result<String> {
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
        // Zone by zone across all people (every head first, then every torso, then legs), so when the
        // time budget runs out each person has their most useful zone; going person by person used to
        // give the leftmost all three zones and the others nothing.
        let people: Vec<(&str, Vec<(&'static str, image::DynamicImage)>)> = chars
            .iter()
            .enumerate()
            .map(|(i, b)| (pos_labels.get(i).copied().unwrap_or("Character"), crate::detector::three_zone_crops(&img, b)))
            .collect();
        let mut lines: Vec<Vec<String>> = vec![Vec::new(); people.len()];
        'zones: for zone in 0..3 {
            if upper_only && zone > 0 {
                break; // 3+ people: one call each (head, hair, upper outfit)
            }
            for (pi, (label, zones)) in people.iter().enumerate() {
                let Some((z_name, z_img)) = zones.get(zone) else { continue };
                if zone_calls >= MULTIZONE_MAX_ZONE_CALLS || budget.spent() || cancelled() {
                    break 'zones;
                }
                zone_calls += 1;
                let label = *label;
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
                        lines[pi].push(format!("- **{}**: {}", z_name, desc.trim()));
                    }
                }
            }
        }
        for ((label, _), zone_lines) in people.iter().zip(lines) {
            if !zone_lines.is_empty() {
                char_sections.push(format!("### Character: {}\n{}", label, zone_lines.join("\n")));
            }
        }
    }

    // 3. 4-Quadrant environmental objects & props (plants, furniture, lamps, windows, decor)
    let quads = crate::detector::quadrant_crops(&img);
    let mut quad_parts = Vec::new();
    for (name, q_img) in &quads {
        if budget.spent() || cancelled() {
            break; // out of time (or cancelled): the people above matter more than the decor
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
pub(crate) fn vlm_sampling(temp: f32, first_try_temp: f32) -> (f32, u32) {
    let retry = temp > first_try_temp + 1e-3;
    if retry {
        (temp, 1042)
    } else {
        ((temp * 0.5).min(0.05), 42)
    }
}

/// Total time the multi-zone scan may spend on its extra (zone and quadrant) vision calls.
pub(crate) const MULTIZONE_BUDGET: Duration = Duration::from_secs(60);
/// After this long, a Deep Scan starts no more re-scans or follow-ups.
pub(crate) const DEEP_SCAN_BUDGET: Duration = Duration::from_secs(180);
/// Most per-person zone calls one multi-zone scan makes (2 people x 3 zones, or 1 each for 3-4).
pub(crate) const MULTIZONE_MAX_ZONE_CALLS: usize = 6;

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
    let base = base_url(host)?;
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    let prompt = toriigate_request_text(booru_tags);
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

/// The user message ToriiGate expects (its `make_user_query` with the "min_structured_md" format):
/// the format, the image's booru tags, and the recognized character names or "avoid to guess names".
pub(crate) fn toriigate_request_text(booru_tags: &str) -> String {
    const FORMAT: &str = "Your answer must contain 3 parts:\n<format>\n# 1. Thoughts about characters\n\
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.\n\
If no characters are listed in input - just write here \"No named characters\"\n# 2. Key details\n\
Here you need to write about the key details on image, prefere using regular text.\n# 3. Structured description\n## General\n\
Write about general composition, content of image, background and all things that are not related to characters directly.\n\
## Character name 1 (put here the name if any)\nWrite about datails and content related to specific character, including features, poses, look, used objects, interactions, and other things.\n\
## Character name 2 (put here the name if any)\nSame for each character.\n## Image effects\nMention image effect, style, camera angle\n</format>\nIn general stick to shorter descriptions.\n";
    let mut prompt = format!("# Captioning format:\n{FORMAT}\n");
    if !booru_tags.trim().is_empty() {
        prompt.push_str(&format!("# Booru tags for the image\n[{}]\n\n", booru_tags.trim()));
    }
    // Named characters the tagger recognized (it lists one only at >= 0.85): ToriiGate is trained to
    // use them ("make sure to use them"); without any it is told not to guess (it invents names).
    let names: Vec<String> = booru_tags
        .split(',')
        .map(|t| t.trim().replace("\\(", "(").replace("\\)", ")"))
        .filter(|t| !t.is_empty() && crate::danbooru::is_character(t))
        .collect();
    if names.is_empty() {
        prompt.push_str("# Characters on picture:\nAvoid to guess names for characters.\n");
    } else {
        prompt.push_str(&format!(
            "# Characters on picture:\nHere are names/tags for characters from the picture, make sure to use them: [{}].\n\n",
            names.join(", ")
        ));
    }
    prompt
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
