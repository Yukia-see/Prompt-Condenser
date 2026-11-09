//! Part of `ai_engine`; see mod.rs.
use super::*;

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

pub(crate) fn translate_system_prompt(req: &TranslateRequest, candidates: &str) -> String {
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
