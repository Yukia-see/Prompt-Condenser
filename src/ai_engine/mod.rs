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

/// System/user prompts for the Deep Scan, per model family.
mod instructions;
pub use instructions::*;
/// JSON repair and turning AI changes into opt-in cards.
mod cards;
pub use cards::*;
/// Image requests, dual pass, multi-zone scan, ToriiGate / checklist observation.
mod vision;
pub use vision::*;
/// Cleanup of image-read tags (form tags, validation, one size per part).
mod polish;
pub use polish::*;
/// Cancel flag, vision encoders, the image writer, multi-character image split.
mod image_read;
pub use image_read::*;
/// Danbooru Words dialog.
mod translate;
pub use translate::*;
/// HTTP client: model list, pull, generate request types.
mod ollama;
pub use ollama::*;
/// Deep Scan run, completeness checks, re-scans, placement.
mod scan;
pub use scan::*;
/// Rules enforced on AI text (placeholders, quality tags, strong-tag guard).
mod cleanup;
pub use cleanup::*;


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

static CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The Cancel button: the running Deep Scan / image read stops after its current step (a request
/// already sent to Ollama can't be interrupted, so that step finishes and its result is dropped).
pub fn request_cancel() {
    CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Cleared when a new Deep Scan or image read starts.
pub fn reset_cancel() {
    CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
}

pub fn cancelled() -> bool {
    CANCEL.load(std::sync::atomic::Ordering::SeqCst)
}

/// `Err("Cancelled")` once Cancel was pressed: put between the steps of a long job.
pub fn check_cancel() -> Result<()> {
    if cancelled() { Err(anyhow!("Cancelled")) } else { Ok(()) }
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
    /// Tests that run scans or image reads take turns: the Cancel flag is process-wide, and the
    /// cancel test sets it.
    static SCAN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn scan_lock() -> std::sync::MutexGuard<'static, ()> {
        SCAN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A plain Anima Deep Scan request against `host`.
    fn test_request(host: &str) -> AiScanRequest {
        AiScanRequest {
            raw_prompt: "1girl, upper body".into(),
            config: ScanConfig {
                checkpoint: CheckpointProfile::Anima, target_view: crate::profiles::TargetView::None, pinned_emphasis_raw: String::new(),
                emphasis_weight: 1.25, cleanup_enabled: true, resolve_conflicts_enabled: true, clamp_weights_enabled: true,
                camera_angle: 0, pose: 0, proportion: 0, dynamic: false, lora_tags_raw: String::new(), arranged: false,
                natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(),
                char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),
                detail: crate::profiles::DetailLevel::Balanced,
            },
            mode: AiMode::Gpu,
            host: host.to_string(),
            model: "prompt-condenser-low".into(),
            keep_alive: "30m".into(),
            think: false,
        }
    }

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
        // other spellings of the same quality tags don't get a second block
        assert_eq!(ensure_quality_tags("score 9, score 8 up, 1girl", PonyV6), "score 9, score 8 up, 1girl");
        assert_eq!(ensure_quality_tags("[score_9], 1girl", PonyV6).matches("score_9").count(), 1);
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
        // a resolved flag whose fix contains commas, and weighted / lowercase bare flags
        assert_eq!(
            strip_placeholders("1girl, [UNANCHORED_LIMBS_FLAG resolved: hands in lap, legs crossed], smile"),
            "1girl, hands in lap, legs crossed, smile"
        );
        assert_eq!(strip_placeholders("1girl, (MISSING_BOTTOMS_FLAG:1.1), missing_bottoms_flag, skirt"), "1girl, skirt");
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
        assert!(convert_ai_changes_for(&out.changes, &tokens, &[], true).is_empty(), "layout and existing tags make no cards");
        // a real new detail filed the same wrong way still becomes an add card
        out.improved_prompt.push_str(", seashells");
        out.changes.push(add("seashells", "seashells"));
        let cards = convert_ai_changes_for(&out.changes, &tokens, &[], true);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].suggested_alternative, "seashells");
        assert_eq!(cards[0].original, "", "an addition carries no original, so accepting it adds the tag");
        // on the 1 Character tab a count tag is content: "2girls" becomes a card there
        assert_eq!(convert_ai_changes_for(&[add("", "2girls")], &tokens, &[], false).len(), 1);
        assert!(convert_ai_changes_for(&[add("", "2girls")], &tokens, &[], true).is_empty());
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

    /// `IMAGE_TEST=a.png:b.png cargo test live_panels -- --ignored --nocapture`: panels found per image.
    #[test]
    #[ignore]
    fn live_panels() {
        for path in std::env::var("IMAGE_TEST").unwrap().split(':') {
            let img = image::open(path).unwrap();
            println!("PANELS {} {}", panel_count(&img), path);
        }
    }

    #[test]
    fn plainly_described_body_and_pose_details_come_back() {
        let desc = "She wears a navy top with a visible navel showing midriff exposure; one arm is extended forward \
                    toward the viewer. Her shoes are not visible.";
        let out = polish_image_prompt_with("1girl, sailor collar, invisible", desc, &[], "1girl", &[]);
        for want in ["navel", "midriff", "outstretched arm", "reaching towards viewer"] {
            assert!(out.split(", ").any(|t| t == want), "{want} missing: {out}");
        }
        assert!(!out.contains("invisible"), "{out}");
        // negated: no navel claimed
        let out = polish_image_prompt_with("1girl", "Her navel is not visible.", &[], "1girl", &[]);
        assert!(!out.contains("navel"), "{out}");
    }

    #[test]
    fn writer_filler_and_unbacked_named_uniforms_go() {
        let v = |s: &str| s.split(", ").map(str::to_string).collect::<Vec<String>>();
        // named uniforms need the description to name them; generic ones always stay
        let desc = "Kinomoto Sakura in a navy sailor top with red trim and a white pleated skirt.";
        let out = drop_unbacked_named_uniforms(
            v("1girl, tomoeda elementary school uniform, ooarai military uniform, school uniform, sailor collar"),
            desc,
        );
        assert_eq!(out, v("1girl, school uniform, sailor collar"));
        let backed = drop_unbacked_named_uniforms(v("1girl, anzio school uniform"), "She wears the Anzio school uniform.");
        assert_eq!(backed, v("1girl, anzio school uniform"));
        assert_eq!(drop_unbacked_named_uniforms(v("1girl, ooarai military uniform"), ""), v("1girl, ooarai military uniform"));
        // filler about the drawing goes; a known tag inside it stays; descriptions stay
        if crate::danbooru::local().is_some() {
            let out = validate_tags(v("anime style, soft shading, clenched hand in foreground, pink heart chest piece"));
            assert!(!out.iter().any(|t| t.contains("style") || t.contains("shading") || t.contains("foreground")), "{out:?}");
            assert!(out.iter().any(|t| t == "clenched hand"), "{out:?}");
            assert!(out.iter().any(|t| t == "pink heart chest piece"), "{out:?}");
        }
    }

    #[test]
    fn colorful_pictures_are_not_monochrome_and_rivals_are_resolved() {
        // warm paper with grey lines is not colorful; a few strong colors are
        let mut paper = image::RgbImage::from_pixel(400, 400, image::Rgb([240, 228, 205]));
        for x in 0..400 {
            paper.put_pixel(x, 200, image::Rgb([60, 60, 60]));
        }
        assert!(!is_colorful(&image::DynamicImage::ImageRgb8(paper.clone())));
        let mut painting = paper;
        for x in 0..80 {
            for y in 0..40 {
                painting.put_pixel(x, y, image::Rgb([30, 170, 90])); // 2% emerald
            }
        }
        assert!(is_colorful(&image::DynamicImage::ImageRgb8(painting.clone())));
        let file = std::env::temp_dir().join(format!("pc_colorful_{}.png", std::process::id()));
        painting.save(&file).unwrap();
        assert_eq!(color_checked_hints("1girl, monochrome, greyscale, armor", &file), "1girl, armor");
        let _ = std::fs::remove_file(&file);
        // recovery only brings back tags the tagger is sure of
        let v = |s: &str| s.split(", ").map(str::to_string).collect::<Vec<String>>();
        let scores = vec![("white shirt".to_string(), 0.9), ("ooarai military uniform".to_string(), 0.38)];
        let out = restore_tagger_tags(v("1girl"), "white shirt, ooarai military uniform, style parody", "", &scores);
        assert_eq!(out, v("1girl, white shirt"));
        // one hair color for one character; a group keeps every color
        assert_eq!(one_per_part(v("1girl, pink hair, orange hair, blonde hair, multicolored hair")), v("1girl, pink hair, multicolored hair"));
        assert_eq!(one_per_part(v("2girls, purple hair, blonde hair, purple eyes, green eyes")), v("2girls, purple hair, blonde hair, purple eyes, green eyes"));
        // barefoot goes next to shoes, stays next to thighhighs
        assert_eq!(drop_contradicted_feet(v("barefoot, mary janes, white socks")), v("mary janes, white socks"));
        assert_eq!(drop_contradicted_feet(v("barefoot, black thighhighs")), v("barefoot, black thighhighs"));
    }

    #[test]
    fn the_writers_dropped_tagger_tags_come_back_with_the_same_checks() {
        let v = |s: &str| s.split(", ").map(str::to_string).collect::<Vec<String>>();
        // unbacked defaults go; backed ones (tagger or description) stay
        let out = restore_tagger_tags(v("1girl, smile, looking at viewer, full body, thigh gap"), "", "she stands, full body visible", &[]);
        assert_eq!(out, v("1girl, smile, full body"));
        let out = restore_tagger_tags(v("1girl, looking at viewer"), "looking at viewer", "", &[]);
        assert_eq!(out, v("1girl, looking at viewer"));
        // a sure tagger size replaces the writer's, in place
        let out = restore_tagger_tags(v("1girl, large breasts, smile"), "gigantic breasts", "", &[("gigantic breasts".into(), 0.93)]);
        assert_eq!(out, v("1girl, gigantic breasts, smile"));
        // recovery: tagger order, nothing present / generic / noise / count, at most 18
        let tagger = "1girl, solo, breasts, virtual youtuber, white shirt, smile, shirt, loose socks";
        let out = restore_tagger_tags(v("1girl, smile"), tagger, "", &[]);
        assert_eq!(out, v("1girl, smile, white shirt, loose socks"));
        let many: Vec<String> = (0..30).map(|i| format!("tag{i}")).collect();
        assert_eq!(restore_tagger_tags(v("1girl"), &many.join(", "), "", &[]).len(), 1 + 18);
        // with scores, every sure tag comes back (no cap of 18); unsure ones still don't
        let mut scored: Vec<(String, f32)> = many.iter().map(|t| (t.clone(), 0.9)).collect();
        scored[29].1 = 0.3;
        assert_eq!(restore_tagger_tags(v("1girl"), &many.join(", "), "", &scored).len(), 1 + 29);
        // through the whole polish: the writer's size and hair win over recovered rivals
        let out = polish_image_prompt_with(
            "1girl, short hair, long sidelocks, huge breasts, crop top",
            "",
            &[],
            "1girl, long hair, large breasts, shirt, crop top, navel",
            &[],
        );
        assert!(!out.split(", ").any(|t| t == "long hair" || t == "large breasts" || t == "shirt"), "{out}");
        assert!(out.contains("huge breasts") && out.contains("navel"), "{out}");
        // bare parents go next to a specific version
        assert_eq!(crate::danbooru::dedup_color_parents("raccoon tail, tail, large breasts, breasts"), "raccoon tail, large breasts");
    }

    #[test]
    fn sketches_panels_and_names_are_fixed_up() {
        // a sketch: no colors, no robot parts, style tags kept
        let out = apply_image_style(
            "1girl, black hair, aqua eyes, humanoid robot, joints, serafuku, white background",
            "1girl, monochrome, sketch, robot, traditional media",
            1,
        );
        assert_eq!(out, "1girl, serafuku, white background, monochrome, sketch, traditional media");
        // filler phrases become the tags inside them; color descriptions stay whole
        if crate::danbooru::local().is_some() {
            let v = validate_tags(vec!["from behind looking back at viewer".into(), "tight white shirt".into(), "teal bikini top".into()]);
            assert_eq!(v, vec!["from behind", "looking back", "looking at viewer", "white shirt", "teal bikini top"], "{v:?}");
        }
        // no humans: the invented girl goes
        assert_eq!(apply_image_style("masterpiece, solo, 1girl, logo, paw print", "no humans, paw print", 1), "masterpiece, no humans, logo, paw print");
        // a sketch loses hues (and black / white on a line drawing) on hair, eyes, clothes; keeps the rest
        let out = apply_image_style(
            "1girl, long black hair, white thighhighs, red serafuku, light blush, light smile, dark aura, dark skin, white background",
            "1girl, sketch",
            1,
        );
        assert_eq!(out, "1girl, light blush, light smile, dark aura, dark skin, white background, sketch");
        // filled greyscale manga: black hair can be right, a red dress can't
        let out = apply_image_style("1girl, black hair, red dress, smile", "1girl, greyscale, monochrome", 1);
        assert_eq!(out, "1girl, black hair, smile, monochrome, greyscale");
        // materials and extra colors keep a description whole
        if crate::danbooru::local().is_some() {
            let v = validate_tags(vec!["turquoise bikini".into(), "sheer black dress".into()]);
            assert_eq!(v, vec!["turquoise bikini", "sheer black dress"], "{v:?}");
        }
        // panels add multiple views
        assert!(apply_image_style("1girl, smile", "1girl", 2).ends_with("multiple views"));
        // a character name the tagger recognized comes back after the count tags
        if crate::danbooru::local().is_some() {
            let out = apply_image_style("masterpiece, 1girl, solo, bat wings", "rouge the bat, furry, bat wings", 1);
            assert_eq!(out, "masterpiece, 1girl, solo, rouge the bat, bat wings");
        }
        // a dark full-height line in the middle splits the picture into two panels; a plain one has none
        let mut img = image::RgbImage::from_pixel(800, 600, image::Rgb([240, 200, 220]));
        assert_eq!(panel_count(&image::DynamicImage::ImageRgb8(img.clone())), 1);
        for x in 398..404 {
            for y in 0..600 {
                img.put_pixel(x, y, image::Rgb([10, 10, 10]));
            }
        }
        assert_eq!(panel_count(&image::DynamicImage::ImageRgb8(img)), 2);
    }

    #[test]
    fn cancel_stops_a_scan_before_its_next_request() {
        let _turn = scan_lock();
        let (host, log) = fake_ollama();
        request_cancel();
        assert!(check_cancel().is_err());
        let err = run_ai_scan(&test_request(&host)).unwrap_err().to_string();
        assert!(err.contains("Cancelled"), "{err}");
        assert!(log.lock().unwrap().iter().all(|(p, _)| p != "/api/generate"), "no model call after Cancel");
        reset_cancel();
        assert!(check_cancel().is_ok());
    }

    #[test]
    fn toriigate_gets_recognized_names_or_is_told_not_to_guess() {
        let none = toriigate_request_text("1girl, raccoon ears, dark skin");
        assert!(none.contains("Avoid to guess names") && none.contains("[1girl, raccoon ears, dark skin]"), "{none}");
        if crate::danbooru::local().is_some() {
            let named = toriigate_request_text("1girl, hatsune miku, twintails");
            assert!(named.contains("make sure to use them: [hatsune miku]") && !named.contains("Avoid to guess"), "{named}");
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
        // archetypes and tail descriptions are not species: only other animals go
        let fixed3 = enforce_strong_tags(
            "1girl, magical girl, school girl, fox girl, striped tail, fluffy tail, cat tail, animal ears, fox ears",
            &["raccoon girl".to_string(), "raccoon tail".to_string()],
        );
        assert_eq!(fixed3, "raccoon tail, raccoon girl, 1girl, magical girl, school girl, striped tail, fluffy tail, animal ears, fox ears");
        assert!(!is_guarded_tag("magical girl") && !is_guarded_tag("striped tail"));
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
        let _turn = scan_lock();
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
            tag_hints: "1girl, green hair, thighhighs".into(), strong_tags: Vec::new(), tagger_scores: Vec::new(), second_pass_vision: false, scene_body_shape: false,
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
        let _turn = scan_lock();
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
        let _turn = scan_lock();
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
                tagger_scores: tag_scores(&scored),
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

