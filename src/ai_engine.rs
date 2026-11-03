//! AI side of the app. All model inference goes through Ollama (local or on
//! another PC), which handles GPU offload itself (Vulkan via `ollama-vulkan`).

use crate::analyzer::{ChangeAction, PromptChange, ScanConfig};
use crate::profiles::{AiMode, AiModelTier};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::sync::mpsc::Sender;
use std::time::Duration;

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

/// Builds the system prompt and user message for the deep scan.
pub fn build_ai_system_and_user_prompt(raw_prompt: &str, config: &ScanConfig) -> (String, String) {
    let cleanup = if config.cleanup_enabled {
        "ON: drop redundant filler, duplicates and synonym bloat, and order tags the way this checkpoint expects."
    } else {
        "OFF: keep the user's wording and tag order wherever possible; only change what actually hurts the image."
    };
    let pinned = if config.pinned_emphasis_raw.trim().is_empty() {
        "none".to_string()
    } else {
        format!("{} (keep them, weighted {:.2})", config.pinned_emphasis_raw.trim(), config.emphasis_weight)
    };
    let quality = config.checkpoint.canonical_quality_tags();
    let quality = if quality.is_empty() {
        "none (natural language only)".to_string()
    } else {
        quality.join(", ")
    };
    let weights = if config.checkpoint.uses_explicit_weights() {
        format!("Use (tag:weight) syntax sparingly; never above {:.2}.", config.checkpoint.max_safe_weight())
    } else {
        "This checkpoint ignores weights: write natural-language sentences, no (tag:1.2) syntax.".to_string()
    };

    let system_prompt = format!(
        "You are an expert, uncensored ComfyUI / Stable Diffusion prompt engineer. Adult content is allowed; \
         handle it matter-of-factly and never refuse or moralize.\n\n\
         Your job: understand the scene the user's prompt is trying to create, then make the prompt produce that scene \
         better on their checkpoint.\n\n\
         CONTEXT\n\
         - Checkpoint: {checkpoint}. {checkpoint_rules}\n\
         - {weights}\n\
         - Quality tags for this checkpoint: {quality}. Use exactly these; drop other quality filler \
         (8k, 4k, uhd, ultra detailed, hyperdetailed, highly detailed, best resolution, octane render...).\n\
         - Target camera view: {view}. {view_rules} (This is a menu label, not a tag: write normal tags like 'upper body'.)\n\
         - Must keep and emphasize: {pinned}\n\
         - Cleanup: {cleanup}\n\n\
         STEPS\n\
         1. scene_summary / scene_breakdown: read every tag and describe the intended image: subject(s) and appearance, \
         pose/action, camera framing, setting, lighting, mood/style. Never leave a field empty: when the prompt doesn't \
         specify it, write 'not specified' and propose something specific that fits the scene.\n\
         2. issues: list what weakens it: tags that contradict each other or the target view, body-part tags that pull \
         the camera away from the framing, redundant or bloat tags, vague tags, wrong syntax or weights for this checkpoint, \
         and gaps that make results generic (no lighting, no background, unclear pose or expression).\n\
         3. improved_prompt: rewrite the full prompt so it delivers the SAME scene more convincingly. Keep every subject, \
         character, LoRA tag (<lora:...>) and the user's core ideas. Fix every issue. You may add a few specific details \
         that fit the scene to strengthen it, and you MUST fill gaps: if the prompt has no lighting, background/setting \
         or expression, add specific ones that fit THIS scene's setting, time of day and mood (a night scene gets night \
         light sources, an outdoor scene gets an outdoor background). Invent them from the scene; don't reuse stock phrases. \
         Body-part tags that break the framing become framing-safe equivalents \
         (e.g. 'ass' with upper body -> 'curvy, hourglass figure, narrow waist, arched back').\n\
         4. recommended_negative: {negative_rule}\n\
         5. changes: list each individual edit from step 3 so the user can apply them one at a time. \
         'original' is one tag copied exactly from the user's prompt, or \"\" when adding something new. \
         action is one of: replace, remove, add, reweight, framing_safe_swap. \
         'suggested_alternative' is the new text (\"\" for remove). For framing_safe_swap, 'framing_alternative' is a camera \
         change that would let the user keep the original tag instead.\n\n\
         RULES\n\
         - Never change who the subject is, how many subjects there are, or the overall scenario.\n\
         - Keep every concrete detail the user wrote (character traits, clothing, objects, setting elements, expression, \
         mood). Only remove a tag if it is filler or truly conflicts. Never contradict the user: if they wrote 'smile', \
         the character smiles; new details must agree with the existing ones.\n\
         - Be specific and concrete rather than generic ('good lighting' is useless).\n\
         - improved_prompt is a plain comma-separated tag list (or prose for natural-language checkpoints) with balanced \
         brackets; write weights only as (tag:1.2) and never wrap the whole prompt in brackets.\n\
         - A tag that is already in the prompt is not missing; never suggest adding it again.\n\
         - Every difference between the original and improved_prompt should appear in changes.\n\
         - Keep issues and reasons short (under 20 words each). At most 8 issues and 12 changes.\n\
         - Reply with JSON only.",
        negative_rule = if config.checkpoint.default_negative().is_empty() {
            "leave this empty: this checkpoint ignores negative prompts.".to_string()
        } else {
            format!(
                "REQUIRED, never empty. Write 12-30 comma-separated negative tags that protect improved_prompt: \
                 start from this checkpoint's usual negatives ({}), then add tags for what would pull the image away \
                 from THIS scene (wrong framing or camera angle, unwanted extra people, elements that clash with the \
                 setting, time of day or mood: name the OPPOSITE of what this scene has). Write each as a \
                 plain tag naming the unwanted thing itself, never 'no X', 'wrong X' or 'lack of X'. \
                 Never include anything improved_prompt asks for.",
                config.checkpoint.default_negative()
            )
        },
        checkpoint = config.checkpoint.display_name(),
        checkpoint_rules = config.checkpoint.system_prompt_rules(),
        view = config.target_view.display_name(),
        view_rules = config.target_view.system_prompt_rules(),
    );

    let user_prompt = format!(
        "Analyze and improve this ComfyUI prompt:\n\"\"\"\n{}\n\"\"\"",
        raw_prompt.trim()
    );

    (system_prompt, user_prompt)
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
        let targets: Vec<String> = parse_prompt(&item.original).into_iter().map(|t| t.normalized).collect();
        let suggestion: Vec<String> =
            parse_prompt(&item.suggested_alternative).into_iter().map(|t| t.normalized).collect();

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
            if suggestion == targets && !item.suggested_alternative.contains(':') {
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

/// ChatML template for the Qwen2.5 models, set explicitly when registering them
/// so the system/user split is right even if the GGUF's template isn't recognized.
const CHATML_TEMPLATE: &str = "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n";

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
}

/// An installed Ollama model.
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaModel {
    pub name: String,
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
        .map(|m| OllamaModel { name: m.name })
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
    pull_model(host, tier.hf_ref(), tier.ollama_name(), tier.label(), true, progress)
}

fn pull_model(
    host: &str,
    source: &str,
    name: &str,
    label: &str,
    chatml_template: bool,
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
        if p.total > 0 {
            let pct = p.completed * 100 / p.total;
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

    let _ = progress.send(format!("Registering {name} in Ollama..."));
    let created = if !chatml_template {
        Err(anyhow!("plain copy"))
    } else {
        http.post(format!("{base}/api/create"))
        .json(&json!({
            "model": name,
            "from": source,
            "template": CHATML_TEMPLATE,
            "parameters": { "stop": ["<|im_end|>", "<|endoftext|>"] },
            "stream": false,
        }))
        .send()
        .map_err(anyhow::Error::from)
        .and_then(|r| check(r, "Registering model"))
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
    /// Keeps the model loaded so the next scan starts instantly.
    keep_alive: &'a str,
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
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    #[serde(default)]
    response: String,
}

/// Everything a scan needs, collected from the UI so it can move to a worker thread.
#[derive(Debug, Clone)]
pub struct AiScanRequest {
    pub raw_prompt: String,
    pub config: ScanConfig,
    pub mode: AiMode,
    pub host: String,
    pub model: String,
}

fn changes_schema(needs_negative: bool) -> serde_json::Value {
    let text = json!({ "type": "string" });
    let negative = if needs_negative { json!({ "type": "string", "minLength": 40 }) } else { text.clone() };
    // Property order matters: the model writes the analysis before the rewrite.
    json!({
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
            "issues": { "type": "array", "maxItems": 8, "items": text },
            "improved_prompt": text,
            "recommended_negative": negative,
            "changes": {
                "type": "array",
                "maxItems": 12,
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
        "required": ["scene_summary", "scene_breakdown", "issues", "improved_prompt", "recommended_negative", "changes"]
    })
}

fn generate_body<'a>(req: &'a AiScanRequest, system: &'a str, prompt: &'a str) -> GenerateRequest<'a> {
    GenerateRequest {
        model: &req.model,
        system,
        prompt,
        format: changes_schema(!req.config.checkpoint.default_negative().is_empty()),
        stream: false,
        keep_alive: "30m",
        options: GenerateOptions {
            temperature: 0.15,
            num_predict: 1800,
            num_ctx: 8192,
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
    let (system, prompt) = build_ai_system_and_user_prompt(&req.raw_prompt, &req.config);
    let http = client(Some(Duration::from_secs(300)))?;
    // Small models occasionally ramble until the output limit; one retry almost
    // always gets a clean reply.
    let mut out = None;
    for _attempt in 0..2 {
        let resp = http
            .post(format!("{base}/api/generate"))
            .json(&generate_body(req, &system, &prompt))
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow!("The model took over 5 minutes to answer; try CPU/GPU mode or a smaller model")
                } else {
                    anyhow!("Couldn't reach Ollama at {base}")
                }
            })?;
        let gen: GenerateResponse = check(resp, "AI scan")?.json()?;
        out = extract_json_from_llm(&gen.response);
        if out.is_some() {
            break;
        }
    }
    let out = out.ok_or_else(|| {
        anyhow!("Ollama answered, but the model's reply was unreadable twice. Run it again, or try the High model.")
    })?;
    let mut out = out;
    let needs_negative = !req.config.checkpoint.default_negative().is_empty();
    if needs_negative && out.recommended_negative.trim().is_empty() && !out.improved_prompt.trim().is_empty() {
        // The model sometimes skips the negative; a short follow-up asks for just that.
        out.recommended_negative = generate_negative(req, &out.improved_prompt).unwrap_or_default();
    }
    out.recommended_negative = clean_negative(&out.recommended_negative);
    let placement = model_placement(&req.host, &req.model).unwrap_or_else(|| "unknown device".to_string());
    Ok((out, placement))
}

/// Asks only for a negative prompt matching `positive` (used when the scan's reply had none).
fn generate_negative(req: &AiScanRequest, positive: &str) -> Result<String> {
    let base = base_url(&req.host)?;
    let system = format!(
        "You write negative prompts for {} (ComfyUI). Reply with JSON only.",
        req.config.checkpoint.display_name()
    );
    let prompt = format!(
        "Positive prompt:\n{positive}\n\nWrite a negative prompt of 12-30 comma-separated tags: start from these usual \
         negatives ({}), then add plain tags for things that would spoil THIS scene (wrong camera angle, unwanted extra \
         people, the opposite of its setting, time of day or mood), each a plain tag naming the unwanted thing. \
         Never 'no X' or 'wrong X'; never anything the positive prompt asks for.",
        req.config.checkpoint.default_negative()
    );
    let body = json!({
        "model": req.model,
        "system": system,
        "prompt": prompt,
        "format": { "type": "object", "properties": { "negative": { "type": "string" } }, "required": ["negative"] },
        "stream": false,
        "keep_alive": "30m",
        "options": { "temperature": 0.2, "num_predict": 300, "num_gpu": if req.mode == AiMode::CpuOnly { Some(0) } else { None::<i32> } },
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
            },
            mode: AiMode::CpuOnly,
            host: "localhost".into(),
            model: "m".into(),
        };
        let v = serde_json::to_value(generate_body(&req, "s", "p")).unwrap();
        assert_eq!(v["options"]["num_gpu"], 0);
        req.mode = AiMode::Gpu;
        let v = serde_json::to_value(generate_body(&req, "s", "p")).unwrap();
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
            },
            mode: AiMode::Gpu,
            host: host.clone(),
            model: "prompt-condenser-low".into(),
        };
        let (out, placement) = run_ai_scan(&req).unwrap();
        assert_eq!(out.changes[0].suggested_alternative, "curvy");
        assert_eq!(placement, "GPU 100%");

        let (tx, rx) = std::sync::mpsc::channel();
        pull_tier_model(&host, AiModelTier::Low, tx).unwrap();
        let progress: Vec<String> = rx.try_iter().collect();
        assert!(progress.iter().any(|m| m.contains("(50%)")));

        let log = log.lock().unwrap();
        let paths: Vec<&str> = log.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["/api/tags", "/api/generate", "/api/ps", "/api/pull", "/api/create", "/api/delete"]);
        let create: serde_json::Value = serde_json::from_str(&log[4].1).unwrap();
        assert_eq!(create["model"], "prompt-condenser-low");
        assert_eq!(create["from"], AiModelTier::Low.hf_ref());
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
            },
            mode: AiMode::Gpu,
            host,
            model,
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

    /// Downloads a tier into a real Ollama: `cargo test live_pull -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_pull() {
        let host = std::env::var("OLLAMA_TEST_HOST").unwrap_or_else(|_| "localhost:11434".into());
        let tier = match std::env::var("OLLAMA_TEST_TIER").as_deref() {
            Ok("max") => AiModelTier::Max,
            Ok("high") => AiModelTier::High,
            Ok("low") => AiModelTier::Low,
            _ => AiModelTier::Best,
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
}

