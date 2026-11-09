//! Part of `ai_engine`; see mod.rs.
use super::*;

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
pub(crate) fn tier_template(_tier: AiModelTier) -> Option<&'static str> {
    None
}

pub(crate) fn client(timeout: Option<Duration>) -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(timeout)
        .build()?)
}

pub(crate) fn base_url(host: &str) -> Result<String> {
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
pub(crate) fn check(resp: reqwest::blocking::Response, what: &str) -> Result<reqwest::blocking::Response> {
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
pub(crate) struct OllamaTagsResponse {
    #[serde(default)]
    pub(crate) models: Vec<OllamaModelEntry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaModelEntry {
    #[serde(default)]
    pub(crate) name: String,
    /// e.g. ["completion", "vision"]
    #[serde(default)]
    pub(crate) capabilities: Vec<String>,
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
pub(crate) struct PullProgress {
    #[serde(default)]
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) total: u64,
    #[serde(default)]
    pub(crate) completed: u64,
    #[serde(default)]
    pub(crate) error: Option<String>,
}

/// Downloads a pre-packaged tier through Ollama (`/api/pull` of the Hugging Face GGUF),
/// then registers it under the app's short name with the ChatML template.
pub fn pull_tier_model(host: &str, tier: AiModelTier, progress: Sender<String>) -> Result<()> {
    pull_model(host, tier.hf_ref(), tier.ollama_name(), tier.label(), tier_template(tier), progress)
}

pub(crate) fn pull_model(
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
pub(crate) struct GenerateRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) system: &'a str,
    pub(crate) prompt: &'a str,
    /// JSON schema: Ollama constrains generation to exactly this shape.
    pub(crate) format: serde_json::Value,
    pub(crate) stream: bool,
    /// How long the model stays loaded after the scan.
    pub(crate) keep_alive: &'a str,
    /// `false` turns off "thinking" (Qwen3, Gemma 4) so the answer comes straight away.
    /// Left out for models that reject the setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) think: Option<bool>,
    pub(crate) options: GenerateOptions,
}

#[derive(Debug, Serialize)]
pub(crate) struct GenerateOptions {
    pub(crate) temperature: f32,
    pub(crate) num_predict: i32,
    /// Room for the long system prompt plus a full analysis.
    pub(crate) num_ctx: i32,
    /// 0 = keep every layer on the CPU. Omitted = Ollama offloads to the GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) num_gpu: Option<i32>,
    /// Fixed per attempt, so the same prompt gives the same scan (a retry uses the next one).
    pub(crate) seed: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenerateResponse {
    #[serde(default)]
    pub(crate) response: String,
    /// The model's reasoning, when thinking was on and the model supports it.
    #[serde(default)]
    pub(crate) thinking: String,
    /// "stop", or "length" when the reply hit `num_predict`.
    #[serde(default)]
    pub(crate) done_reason: String,
    #[serde(default)]
    pub(crate) eval_count: u32,
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
