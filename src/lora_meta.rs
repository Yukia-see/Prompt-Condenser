//! LoRA trigger-word suggestions: the most frequent training tags stored in a LoRA file's
//! safetensors header (`__metadata__.ss_tag_frequency`, written by kohya's sd-scripts).

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

/// Suggestions per LoRA file.
pub const MAX_SUGGESTIONS: usize = 5;
/// A bigger header is refused (real ones are a few hundred KB at most).
const MAX_HEADER: u64 = 100 * 1024 * 1024;
/// How deep the folder search goes (also stops symlink loops).
const MAX_DEPTH: usize = 8;

/// Training tags that say nothing about the LoRA's subject: counts, framing, plain backgrounds,
/// quality / meta tags.
const GENERIC: &[&str] = &[
    "solo", "1girl", "1boy", "1other", "looking at viewer", "simple background", "white background",
    "grey background", "gray background", "transparent background", "black background", "upper body", "full body",
    "cowboy shot", "portrait", "close-up", "highres", "absurdres", "lowres", "signature", "watermark", "artist name",
    "english text", "blurry", "realistic", "masterpiece", "best quality", "official art", "commentary request", "translated", "text",
];

/// The LoRA folder to search: `$LORA_DIR` when set, else the Options LoRA folder, else
/// `{ComfyUI folder}/models/loras` (`comfyui_folder`).
pub fn effective_folder(lora_setting: &str, comfyui_setting: &str) -> String {
    let env = |k: &str| std::env::var(k).ok();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    resolve_lora_folder(env("LORA_DIR").as_deref(), lora_setting, comfyui_setting, env("COMFYUI_PATH").as_deref(), home.as_deref())
}

/// The ComfyUI folder: the Options setting, else `$COMFYUI_PATH`, else `$HOME/ComfyUI` if it exists.
pub fn comfyui_folder(setting: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    resolve_comfyui(setting, std::env::var("COMFYUI_PATH").ok().as_deref(), home.as_deref())
}

fn resolve_comfyui(setting: &str, env: Option<&str>, home: Option<&Path>) -> Option<PathBuf> {
    let given = |s: &str| Some(s.trim()).filter(|s| !s.is_empty()).map(PathBuf::from);
    given(setting)
        .or_else(|| env.and_then(given))
        .or_else(|| home.map(|h| h.join("ComfyUI")).filter(|p| p.is_dir()))
}

fn resolve_lora_folder(lora_env: Option<&str>, lora_setting: &str, comfyui_setting: &str, comfyui_env: Option<&str>, home: Option<&Path>) -> String {
    let given = |s: &str| Some(s.trim().to_string()).filter(|s| !s.is_empty());
    lora_env
        .and_then(given)
        .or_else(|| given(lora_setting))
        .or_else(|| resolve_comfyui(comfyui_setting, comfyui_env, home).map(|c| c.join("models").join("loras").to_string_lossy().into_owned()))
        .unwrap_or_default()
}

/// The LoRA names in a prompt: `<lora:name:0.8>` → `name` (a sub-folder in front is dropped,
/// and so is a `.safetensors` ending).
pub fn lora_names(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // ASCII lowercasing keeps byte offsets, so they work on `text` too
    let lower = text.to_ascii_lowercase();
    let mut rest = lower.as_str();
    let mut offset = 0;
    while let Some(i) = rest.find("<lora:") {
        let start = offset + i + "<lora:".len();
        let Some(end) = text[start..].find('>').map(|e| start + e) else { break };
        let inner = &text[start..end];
        // name:weight (the name itself may hold a Windows drive colon only in a path, which is rare)
        let name = inner.rsplit_once(':').filter(|(_, w)| w.trim().parse::<f32>().is_ok()).map_or(inner, |(n, _)| n);
        let name = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
        let name = name.strip_suffix(".safetensors").unwrap_or(name);
        if !name.is_empty() && !out.iter().any(|n| n.eq_ignore_ascii_case(name)) {
            out.push(name.to_string());
        }
        offset = end;
        rest = &lower[end..];
    }
    out
}

/// `<folder>/**/name.safetensors`, the file stem matched case-insensitively.
pub fn find_lora(folder: &Path, name: &str) -> Option<PathBuf> {
    fn walk(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
        let mut dirs = Vec::new();
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
                && path.file_stem().is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case(name))
            {
                return Some(path);
            }
        }
        if depth >= MAX_DEPTH {
            return None;
        }
        dirs.sort();
        dirs.iter().find_map(|d| walk(d, name, depth + 1))
    }
    walk(folder, name, 0)
}

/// A LoRA file's `__metadata__` object from its safetensors header (`Null` when it has none).
fn read_metadata(path: &Path) -> anyhow::Result<serde_json::Value> {
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len > MAX_HEADER {
        anyhow::bail!("safetensors header of {len} bytes is too big");
    }
    let mut header = vec![0u8; len as usize];
    f.read_exact(&mut header)?;
    let mut json: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(json.get_mut("__metadata__").map(serde_json::Value::take).unwrap_or_default())
}

/// The trainer's trigger words (`modelspec.trigger_phrase`, comma-separated), in order.
fn trigger_phrases(meta: &serde_json::Value) -> Vec<String> {
    meta.get("modelspec.trigger_phrase")
        .and_then(|v| v.as_str())
        .map(|p| p.split(',').map(|t| t.trim().replace('_', " ")).filter(|t| !t.is_empty()).collect())
        .unwrap_or_default()
}

/// Every training tag in a LoRA file with its count summed over the datasets, most frequent first.
/// No metadata (or no `ss_tag_frequency`) gives an empty list.
#[cfg(test)]
pub fn read_tag_frequency(path: &Path) -> anyhow::Result<Vec<(String, u64)>> {
    tag_frequency(&read_metadata(path)?)
}

fn tag_frequency(meta: &serde_json::Value) -> anyhow::Result<Vec<(String, u64)>> {
    let Some(freq) = meta.get("ss_tag_frequency").and_then(|v| v.as_str()) else {
        return Ok(Vec::new());
    };
    let datasets: HashMap<String, HashMap<String, serde_json::Value>> = serde_json::from_str(freq)?;
    let mut sums: Vec<(String, u64)> = Vec::new();
    for tags in datasets.values() {
        for (tag, count) in tags {
            let tag = tag.trim().replace('_', " ");
            let count = count.as_u64().or_else(|| count.as_f64().map(|c| c as u64)).unwrap_or(0);
            if tag.is_empty() {
                continue;
            }
            match sums.iter_mut().find(|(t, _)| *t == tag) {
                Some((_, c)) => *c += count,
                None => sums.push((tag, count)),
            }
        }
    }
    // most frequent first, ties alphabetically so the order is stable
    sums.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(sums)
}

/// The `MAX_SUGGESTIONS` most frequent tags that say something about the LoRA's subject.
#[cfg(test)]
pub fn top_tags(freq: &[(String, u64)]) -> Vec<String> {
    pick(freq.iter().map(|(t, _)| t.clone()))
}

/// The first `MAX_SUGGESTIONS` tags that say something about the LoRA's subject, without duplicates.
fn pick(tags: impl Iterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let n = t.to_lowercase();
        if GENERIC.contains(&n.as_str())
            || crate::ai_engine::is_quality_tag_any(&n)
            || crate::ai_engine::is_layout_tag(&n, crate::profiles::CheckpointProfile::Anima)
            || out.iter().any(|o| o.to_lowercase() == n)
        {
            continue;
        }
        out.push(t);
        if out.len() == MAX_SUGGESTIONS {
            break;
        }
    }
    out
}

/// A LoRA file's suggestions: its trigger phrase first, then its most frequent training tags.
fn read_suggestions(path: &Path) -> anyhow::Result<Vec<String>> {
    let meta = read_metadata(path)?;
    let freq = tag_frequency(&meta).unwrap_or_default();
    Ok(pick(trigger_phrases(&meta).into_iter().chain(freq.into_iter().map(|(t, _)| t))))
}

/// Parsed suggestions per file, kept while its modification time stays the same.
static CACHE: Mutex<Option<HashMap<PathBuf, (SystemTime, Vec<String>)>>> = Mutex::new(None);

/// Suggestions for one LoRA name in `folder` (empty when the file or its metadata is missing).
/// Reads the disk: call it off the UI thread.
pub fn suggestions(folder: &Path, name: &str) -> Vec<String> {
    let Some(path) = find_lora(folder, name) else { return Vec::new() };
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
    if let Some((t, tags)) = CACHE.lock().unwrap().get_or_insert_with(HashMap::new).get(&path) {
        if *t == mtime {
            return tags.clone();
        }
    }
    let tags = read_suggestions(&path).unwrap_or_default();
    CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(path, (mtime, tags.clone()));
    tags
}

/// Suggestions for every LoRA in a LoRA box, in order, without duplicates.
pub fn suggestions_for_box(folder: &Path, lora_box: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for name in lora_names(lora_box) {
        for t in suggestions(folder, &name) {
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
    out
}

/// The suggestions not already in the box (whole tags, underscores and case ignored).
pub fn not_in_box(suggested: &[String], lora_box: &str) -> Vec<String> {
    let have: Vec<String> = lora_box
        .split([',', '\n'])
        .map(|t| t.trim().replace('_', " ").to_lowercase())
        .collect();
    suggested.iter().filter(|s| !have.contains(&s.to_lowercase())).cloned().collect()
}

/// Weight step of the LoRA chips' − / + buttons, and the range (negative weights are valid: slider LoRAs).
pub const WEIGHT_STEP: f32 = 0.05;
pub const WEIGHT_MIN: f32 = -5.0;
pub const WEIGHT_MAX: f32 = 5.0;

/// Each `<lora:…>` in a box: byte range of the text between `<lora:` and `>`, the name, and the weights
/// written after it (`name:0.8:0.6` → [0.8, 0.6]; a name may itself hold colons, the weights are the
/// trailing numbers).
fn lora_spans(text: &str) -> Vec<(std::ops::Range<usize>, String, Vec<f32>)> {
    let lower = text.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = lower[from..].find("<lora:") {
        let start = from + i + "<lora:".len();
        let Some(end) = text[start..].find('>').map(|e| start + e) else { break };
        let parts: Vec<&str> = text[start..end].split(':').collect();
        let mut j = parts.len();
        while j > 1 && parts[j - 1].trim().parse::<f32>().is_ok() {
            j -= 1;
        }
        let weights = parts[j..].iter().filter_map(|p| p.trim().parse::<f32>().ok()).collect();
        out.push((start..end, parts[..j].join(":"), weights));
        from = end;
    }
    out
}

/// The box's LoRAs with their (first) weight; a LoRA written without one has 1.0.
pub fn loras_with_weights(box_text: &str) -> Vec<(String, f32)> {
    lora_spans(box_text).into_iter().map(|(_, name, w)| (name, w.first().copied().unwrap_or(1.0))).collect()
}

/// A weight as the box shows it: 2 decimals at most, no trailing zeros (0.85, 0.8, 1, -0.5).
pub fn format_weight(w: f32) -> String {
    let w = (w * 100.0).round() / 100.0;
    let w = if w == 0.0 { 0.0 } else { w };
    let s = format!("{w:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The box with the `index`-th LoRA's first weight moved by `delta` (rounded to 2 decimals, kept in
/// −5…5); a second weight (`<lora:n:0.8:0.6>`) and all other text stay as written. A missing weight
/// counts as 1.0 and is written out. An index past the last LoRA returns the box unchanged.
pub fn step_weight(box_text: &str, index: usize, delta: f32) -> String {
    let Some((range, name, weights)) = lora_spans(box_text).into_iter().nth(index) else {
        return box_text.to_string();
    };
    let w = (weights.first().copied().unwrap_or(1.0) + delta).clamp(WEIGHT_MIN, WEIGHT_MAX);
    let mut inner = format!("{name}:{}", format_weight(w));
    // keep any later weights exactly as written
    let rest: Vec<&str> = box_text[range.clone()].split(':').collect();
    let written = rest.len() - weights.len();
    for p in rest.iter().skip(written + 1) {
        inner.push(':');
        inner.push_str(p);
    }
    format!("{}{}{}", &box_text[..range.start], inner, &box_text[range.end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_weights_are_read_and_stepped_in_place() {
        let b = "trigger, <lora:style_v2:0.8>, more words";
        assert_eq!(loras_with_weights(b), vec![("style_v2".to_string(), 0.8)]);
        assert_eq!(step_weight(b, 0, WEIGHT_STEP), "trigger, <lora:style_v2:0.85>, more words");
        assert_eq!(step_weight(b, 0, -WEIGHT_STEP), "trigger, <lora:style_v2:0.75>, more words");
        // two weights: only the first moves
        let two = "<LORA:sub/char:0.8:0.6>, <lora:b:1>";
        assert_eq!(loras_with_weights(two), vec![("sub/char".to_string(), 0.8), ("b".to_string(), 1.0)]);
        assert_eq!(step_weight(two, 0, WEIGHT_STEP), "<LORA:sub/char:0.85:0.6>, <lora:b:1>");
        assert_eq!(step_weight(two, 1, -WEIGHT_STEP), "<LORA:sub/char:0.8:0.6>, <lora:b:0.95>");
        // no weight = 1.0, written out
        assert_eq!(loras_with_weights("<lora:plain>"), vec![("plain".to_string(), 1.0)]);
        assert_eq!(step_weight("x <lora:plain> y", 0, WEIGHT_STEP), "x <lora:plain:1.05> y");
        // negative weights (slider LoRAs) are fine, down to -5; no floor at 0.1
        assert_eq!(step_weight("<lora:slider:0.05>", 0, -WEIGHT_STEP), "<lora:slider:0>");
        assert_eq!(step_weight("<lora:slider:0>", 0, -WEIGHT_STEP), "<lora:slider:-0.05>");
        assert_eq!(step_weight("<lora:slider:-5>", 0, -WEIGHT_STEP), "<lora:slider:-5>");
        assert_eq!(step_weight("<lora:slider:4.98>", 0, WEIGHT_STEP), "<lora:slider:5>");
        // nothing to step
        assert_eq!(step_weight("just words", 0, WEIGHT_STEP), "just words");
        assert_eq!(step_weight(b, 3, WEIGHT_STEP), b);
        assert_eq!(format_weight(0.8), "0.8");
        assert_eq!(format_weight(-0.0001), "0");
    }

    /// A safetensors file with just a header (no tensor data).
    fn write_lora(dir: &Path, name: &str, header: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pc-lora-meta-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn folders_resolve_in_order() {
        let home = temp_dir("home");
        // no setting, no env, no ~/ComfyUI: nothing
        assert_eq!(resolve_comfyui("", None, Some(&home)), None);
        std::fs::create_dir_all(home.join("ComfyUI")).unwrap();
        assert_eq!(resolve_comfyui("", None, Some(&home)), Some(home.join("ComfyUI")));
        assert_eq!(resolve_comfyui("", Some("/env/comfy"), Some(&home)), Some(PathBuf::from("/env/comfy")));
        assert_eq!(resolve_comfyui(" /set/comfy ", Some("/env/comfy"), Some(&home)), Some(PathBuf::from("/set/comfy")));
        // LoRAs: $LORA_DIR, then the LoRA setting, then {comfyui}/models/loras
        assert_eq!(resolve_lora_folder(None, "", "/set/comfy", None, None), "/set/comfy/models/loras");
        assert_eq!(resolve_lora_folder(None, "/my/loras", "/set/comfy", None, None), "/my/loras");
        assert_eq!(resolve_lora_folder(Some("/env/loras"), "/my/loras", "/set/comfy", None, None), "/env/loras");
        assert_eq!(resolve_lora_folder(None, "", "", None, None), "");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn lora_names_are_read_from_the_box() {
        assert_eq!(
            lora_names("<lora:Kay_GuP:0.8>, kay, <LORA:styles/inkWash.safetensors:1>, <lora:kay_gup:0.5>, <lora:bare>"),
            vec!["Kay_GuP", "inkWash", "bare"]
        );
        assert!(lora_names("1girl, no loras").is_empty());
    }

    #[test]
    fn trigger_tags_come_from_the_safetensors_header() {
        let dir = temp_dir("ok");
        let freq = r#"{"10_kay": {"kay_(gup)": 40, "1girl": 40, "solo": 38, "short_hair": 30, "masterpiece": 39},
                       "5_extra": {"kay_(gup)": 5, "military_uniform": 12, "smile": 9, "blue_eyes": 9, "brown_hair": 8, "jacket": 1}}"#;
        let header = serde_json::json!({
            "__metadata__": { "ss_tag_frequency": freq, "ss_output_name": "kay" },
            "w": { "dtype": "F16", "shape": [1], "data_offsets": [0, 2] }
        })
        .to_string();
        write_lora(&dir.join("chars/GuP"), "Kay_GuP.safetensors", &header);
        // found in a sub-folder, case-insensitively
        let path = find_lora(&dir, "kay_gup").expect("found");
        let f = read_tag_frequency(&path).unwrap();
        assert_eq!(f[0], ("kay (gup)".to_string(), 45));
        // counts, quality and generic tags go; top 5, most frequent first
        assert_eq!(top_tags(&f), vec!["kay (gup)", "short hair", "military uniform", "blue eyes", "smile"]);
        assert_eq!(suggestions_for_box(&dir, "<lora:Kay_GuP:0.8>, kay"), top_tags(&f));
        assert_eq!(not_in_box(&top_tags(&f), "<lora:Kay_GuP:0.8>,\nkay_(gup), Short Hair"), vec!["military uniform", "blue eyes", "smile"]);
        assert!(suggestions(&dir, "missing").is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_trigger_phrase_comes_first() {
        let dir = temp_dir("trigger");
        let freq = r#"{"ds": {"solo": 50, "red_hair": 40, "kay": 30, "smile": 20, "jacket": 10, "boots": 5}}"#;
        let header = serde_json::json!({
            "__metadata__": { "modelspec.trigger_phrase": "kay_gup, masterpiece,  ink style ", "ss_tag_frequency": freq }
        })
        .to_string();
        write_lora(&dir, "kay.safetensors", &header);
        // trigger words in order (quality tag dropped), then training tags, max 5 in all
        assert_eq!(suggestions(&dir, "kay"), vec!["kay gup", "ink style", "red hair", "kay", "smile"]);
        // a trigger phrase alone is enough
        write_lora(&dir, "only.safetensors", r#"{"__metadata__": {"modelspec.trigger_phrase": "zzz style"}}"#);
        assert_eq!(suggestions(&dir, "only"), vec!["zzz style"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_or_bare_headers_give_no_suggestions() {
        let dir = temp_dir("bad");
        // no metadata at all, and metadata without tag frequencies
        let bare = write_lora(&dir, "bare.safetensors", r#"{"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#);
        assert!(read_tag_frequency(&bare).unwrap().is_empty());
        let named = write_lora(&dir, "named.safetensors", r#"{"__metadata__": {"ss_output_name": "x"}}"#);
        assert!(read_tag_frequency(&named).unwrap().is_empty());
        assert!(suggestions(&dir, "named").is_empty());
        // a header length over the limit is refused without reading it
        let huge = dir.join("huge.safetensors");
        std::fs::write(&huge, (200u64 * 1024 * 1024).to_le_bytes()).unwrap();
        assert!(read_tag_frequency(&huge).is_err());
        assert!(suggestions(&dir, "huge").is_empty());
        // a file shorter than its header length
        let cut = dir.join("cut.safetensors");
        std::fs::write(&cut, [50u8, 0, 0, 0, 0, 0, 0, 0, b'{']).unwrap();
        assert!(read_tag_frequency(&cut).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
