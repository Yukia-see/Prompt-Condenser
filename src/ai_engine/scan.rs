//! Part of `ai_engine`; see mod.rs.
use super::*;

pub(crate) fn changes_schema(needs_negative: bool, dynamic: bool, natural: bool, max_issues: usize, max_changes: usize) -> serde_json::Value {
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

pub(crate) fn generate_body<'a>(req: &'a AiScanRequest, system: &'a str, prompt: &'a str, seed: u32) -> GenerateRequest<'a> {
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
    // the short negative follow-up always runs (a slow first scan used to come back without a negative);
    // only the full re-scans are cut by the time limit
    if needs_negative && best.recommended_negative.trim().is_empty() && !best.improved_prompt.trim().is_empty() {
        // The model sometimes skips the negative; a short follow-up asks for just that.
        if !cancelled() {
            best.recommended_negative = generate_negative(req, &best.improved_prompt).unwrap_or_default();
        }
    }
    let mut rescans = 0;
    while !scan_problems(&best, req).is_empty() && rescans < 2 && !budget.spent() && !cancelled() {
        rescans += 1;
        let Ok(next) = scan_once(req, &http, &base, &system, &prompt, &mut send_think, &mut seed) else { continue };
        merge_scan(&mut best, next, req);
    }
    best.recommended_negative = clean_negative(&best.recommended_negative);
    enforce_checkpoint_rules(&mut best, req.config.checkpoint);
    enforce_body_choices(&mut best, &req.config);
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

/// One deep-scan request; an unreadable reply is retried (up to 3 tries).
pub(crate) fn scan_once(
    req: &AiScanRequest,
    http: &reqwest::blocking::Client,
    base: &str,
    system: &str,
    prompt: &str,
    send_think: &mut bool,
    seed: &mut u32,
) -> Result<AiJsonOutput> {
    for _attempt in 0..3 {
        check_cancel()?;
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

pub(crate) fn analysis_ok(o: &AiJsonOutput) -> bool {
    let b = &o.scene_breakdown;
    let filled = [&b.subject, &b.pose, &b.framing, &b.setting, &b.lighting, &b.mood]
        .iter()
        .filter(|f| f.trim().len() >= 3)
        .count();
    o.scene_summary.trim().len() >= 20 && filled >= 4
}

pub(crate) fn negative_ok(o: &AiJsonOutput, req: &AiScanRequest) -> bool {
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

pub(crate) fn cards_ok(o: &AiJsonOutput, req: &AiScanRequest) -> bool {
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
    let multi = req.config.two_char;
    let content = |t: &String| before.contains(t) || !(is_quality_tag_any(t) || (multi && is_layout_tag(t, CheckpointProfile::Flux)));
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
    if !convert_ai_changes_for(&o.changes, &tokens, &rules, multi).is_empty() {
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

pub(crate) fn dynamic_ok(o: &AiJsonOutput, req: &AiScanRequest) -> bool {
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
pub(crate) fn merge_scan(best: &mut AiJsonOutput, next: AiJsonOutput, req: &AiScanRequest) {
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
pub(crate) fn generate_negative(req: &AiScanRequest, positive: &str) -> Result<String> {
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
pub(crate) struct PsResponse {
    #[serde(default)]
    models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PsModel {
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
