//! Part of `ai_engine`; see mod.rs.
use super::*;

/// Which vision model reads which pass: the mode's pass-1 prompt goes to the spatial /
/// body model, everything else to the identity / wardrobe model.
pub(crate) fn vision_model_for_prompt<'a>(
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
pub(crate) fn encoder_keep_alive(encoder: &str) -> &'static str {
    if is_dedicated_encoder(encoder) { "30s" } else { "30m" }
}

/// Unloads the dedicated vision encoders that are currently loaded (never the text model,
/// and never a model that isn't loaded: asking Ollama to unload one would load it first).
pub(crate) fn unload_dedicated_encoders(base: &str, encoders: &[&str], text_model: &str) {
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
pub(crate) fn describe_with_text_model(
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
pub(crate) fn parse_image_ai_result(raw: &str) -> Result<ImageAiResult> {
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

pub(crate) fn describe_image_with(req: &ImageAiRequest, system: &str, ask: &str) -> Result<ImageAiResult> {
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

pub(crate) fn clean_image_ai_prompt(out: &mut ImageAiResult, checkpoint: crate::profiles::CheckpointProfile) {
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

pub(crate) fn split_image_pipelined(req: &TwoImageRequest, encoder: &str) -> Result<(crate::twoimg::SplitPrompt, String)> {
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

pub(crate) fn split_image_direct(req: &TwoImageRequest) -> Result<(crate::twoimg::SplitPrompt, String)> {
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

pub(crate) fn split_once(base: &str, body: &serde_json::Value, n: usize) -> Result<(crate::twoimg::SplitPrompt, String)> {
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
