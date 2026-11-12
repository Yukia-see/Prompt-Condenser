//! The generation prompt saved in a PNG's text chunks: A1111 / Forge `parameters`, ComfyUI's `prompt`
//! graph, NovelAI's `Comment` JSON / `Description`.

use anyhow::Result;
use std::path::Path;

/// A prompt read from a PNG.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PngPrompt {
    pub positive: String,
    pub negative: Option<String>,
    /// "A1111 / Forge", "ComfyUI" or "NovelAI".
    pub source: &'static str,
    /// ComfyUI only: the positive prompt couldn't be followed from a sampler, so the longest text was taken.
    pub partial: bool,
    /// ComfyUI: the sampler node the prompt was taken from ("3 KSampler").
    pub sampler: Option<String>,
    /// ComfyUI: how many different positive prompts the samplers have (hires fix / refiner passes
    /// usually share one).
    pub distinct_positives: usize,
    /// ComfyUI: the samplers' positives differ and the first pass (no other sampler upstream: it made
    /// the picture's content; later passes refine it) was picked.
    pub first_pass_chosen: bool,
}

/// The prompt saved in a PNG, or `None` when it has no generation metadata.
pub fn read_png_prompt(path: &Path) -> Result<Option<PngPrompt>> {
    Ok(from_chunks(&text_chunks(path)?))
}

/// Every text chunk (tEXt, zTXt, iTXt; also the ones after the image data) as (keyword, text).
fn text_chunks(path: &Path) -> Result<Vec<(String, String)>> {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    let mut reader = decoder.read_info()?;
    // chunks after IDAT are only seen once the image is read
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    if reader.next_frame(&mut buf).is_ok() {
        let _ = reader.finish();
    }
    let info = reader.info();
    let mut out: Vec<(String, String)> = Vec::new();
    for t in &info.uncompressed_latin1_text {
        out.push((t.keyword.clone(), t.text.clone()));
    }
    for t in &info.compressed_latin1_text {
        if let Ok(text) = t.get_text() {
            out.push((t.keyword.clone(), text));
        }
    }
    for t in &info.utf8_text {
        if let Ok(text) = t.get_text() {
            out.push((t.keyword.clone(), text));
        }
    }
    Ok(out)
}

fn from_chunks(chunks: &[(String, String)]) -> Option<PngPrompt> {
    let get = |k: &str| chunks.iter().find(|(key, _)| key.eq_ignore_ascii_case(k)).map(|(_, t)| t.as_str());
    if let Some(p) = get("parameters").and_then(parse_a1111) {
        return Some(p);
    }
    if let Some(p) = get("prompt").and_then(parse_comfyui) {
        return Some(p);
    }
    // NovelAI: Comment = {"prompt": …, "uc": …}, Description = the prompt
    let comment: Option<serde_json::Value> = get("Comment").and_then(|c| serde_json::from_str(c).ok());
    let positive = comment
        .as_ref()
        .and_then(|c| c.get("prompt").and_then(|p| p.as_str()).map(str::to_string))
        .or_else(|| get("Description").map(str::to_string))
        .filter(|p| !p.trim().is_empty())?;
    let negative = comment.as_ref().and_then(|c| c.get("uc").and_then(|u| u.as_str())).map(str::to_string);
    Some(PngPrompt { positive: positive.trim().to_string(), negative, source: "NovelAI", ..Default::default() })
}

/// A1111 / Forge: the prompt, then "Negative prompt: …", then a "Steps: …, Sampler: …" line.
fn parse_a1111(text: &str) -> Option<PngPrompt> {
    let trailer = |s: &str| -> usize {
        // the settings line: starts with "Steps:" at the beginning of a line
        s.match_indices("Steps:").find(|(i, _)| *i == 0 || s[..*i].ends_with('\n')).map_or(s.len(), |(i, _)| i)
    };
    let (positive, negative) = match text.find("Negative prompt:") {
        Some(i) => {
            let rest = &text[i + "Negative prompt:".len()..];
            (&text[..i], Some(rest[..trailer(rest)].trim().to_string()))
        }
        None => (&text[..trailer(text)], None),
    };
    let positive = positive.trim();
    (!positive.is_empty()).then(|| PngPrompt {
        positive: positive.to_string(),
        negative: negative.filter(|n| !n.is_empty()),
        source: "A1111 / Forge",
        ..Default::default()
    })
}

/// ComfyUI's API graph: {"id": {"class_type": …, "inputs": {…}}}. The positive prompt is the text that
/// reaches a sampler's `positive` input (through conditioning nodes); else the longest text, as partial.
fn parse_comfyui(json: &str) -> Option<PngPrompt> {
    let graph: serde_json::Value = serde_json::from_str(json).ok()?;
    let nodes = graph.as_object()?;
    let class = |id: &str| nodes.get(id).and_then(|n| n.get("class_type")).and_then(|c| c.as_str()).unwrap_or("");
    let inputs = |id: &str| nodes.get(id).and_then(|n| n.get("inputs")).and_then(|i| i.as_object());
    // a text input: a string, or a link to a node holding one (a primitive / string node)
    let text_of = |id: &str| -> Option<String> {
        let ins = inputs(id)?;
        ["text", "text_g", "prompt", "positive"].iter().find_map(|k| string_value(nodes, ins.get(*k)?, 0))
    };
    let is_encoder = |id: &str| class(id).contains("TextEncode");
    // follows a conditioning link back to a text encoder (through combine / set-area nodes)
    let resolve = |start: &serde_json::Value| -> Option<String> {
        let mut id = start.as_array()?.first()?.as_str()?.to_string();
        for _ in 0..8 {
            if is_encoder(&id) {
                return text_of(&id);
            }
            let ins = inputs(&id)?;
            id = ins
                .iter()
                .filter(|(k, _)| k.starts_with("conditioning") || *k == "positive")
                .find_map(|(_, v)| v.as_array()?.first()?.as_str().map(str::to_string))?;
        }
        None
    };
    // node ids in number order ("3" before "10")
    let mut ids: Vec<&String> = nodes.keys().collect();
    ids.sort_by_key(|id| (id.parse::<u64>().unwrap_or(u64::MAX), id.to_string()));
    // a sampler node: KSampler, SamplerCustom(Advanced)… (not KSamplerSelect, which has no latent,
    // conditioning or guider input)
    let is_sampler = |id: &str| {
        class(id).contains("Sampler")
            && inputs(id).is_some_and(|i| ["latent_image", "latent", "positive", "guider"].iter().any(|k| i.contains_key(*k)))
    };
    // a sampler's conditioning: its own positive / negative, or its guider's (CFGGuider, BasicGuider…)
    let conditioning = |id: &str, which: &str| -> Option<String> {
        let ins = inputs(id)?;
        let link = match ins.get(which) {
            Some(v) => v.clone(),
            None => {
                let guider = ins.get("guider")?.as_array()?.first()?.as_str()?.to_string();
                let g = inputs(&guider)?;
                g.get(which).or_else(|| (which == "positive").then(|| g.get("conditioning")).flatten())?.clone()
            }
        };
        resolve(&link).map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
    };
    // every sampler with a positive prompt that leads back to text
    let samplers: Vec<(&String, String, Option<String>)> = ids
        .iter()
        .filter(|id| is_sampler(id))
        .filter_map(|id| Some((*id, conditioning(id, "positive")?, conditioning(id, "negative"))))
        .collect();
    if !samplers.is_empty() {
        let mut distinct: Vec<&str> = samplers.iter().map(|s| s.1.as_str()).collect();
        distinct.sort();
        distinct.dedup();
        // the first pass: no other sampler upstream (it starts from an empty latent, a reference latent or
        // a loaded image); hires-fix / refiner passes start from another sampler's output
        let first_pass = |id: &str| {
            let mut stack: Vec<String> = vec![id.to_string()];
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            while let Some(n) = stack.pop() {
                if !seen.insert(n.clone()) || seen.len() > 500 {
                    continue;
                }
                if n != id && is_sampler(&n) {
                    return false;
                }
                if let Some(ins) = inputs(&n) {
                    stack.extend(ins.values().filter_map(|v| v.as_array()?.first()?.as_str().map(str::to_string)));
                }
            }
            true
        };
        let chosen = samplers.iter().find(|s| first_pass(s.0));
        let first_pass_chosen = chosen.is_some() && distinct.len() > 1;
        let (id, positive, negative) = chosen.unwrap_or(&samplers[0]).clone();
        return Some(PngPrompt {
            positive,
            negative,
            source: "ComfyUI",
            sampler: Some(format!("{id} {}", class(id))),
            distinct_positives: distinct.len(),
            first_pass_chosen,
            ..Default::default()
        });
    }
    let longest = ids.iter().filter(|id| is_encoder(id)).filter_map(|id| text_of(id)).max_by_key(|t| t.len())?;
    (!longest.trim().is_empty()).then(|| PngPrompt { positive: longest.trim().to_string(), source: "ComfyUI", partial: true, ..Default::default() })
}

/// A text input's string: the value itself, or what a linked node holds (a primitive's `value`, a
/// string node's `text` / `string`), joining concatenate nodes (`string_a` + delimiter + `string_b` …).
fn string_value(nodes: &serde_json::Map<String, serde_json::Value>, v: &serde_json::Value, depth: usize) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(link) if depth < 8 => {
            let node = nodes.get(link.first()?.as_str()?)?;
            let class = node.get("class_type").and_then(|c| c.as_str()).unwrap_or("");
            let ins = node.get("inputs")?.as_object()?;
            if class.contains("Concat") || class.contains("Join") {
                let delimiter = ins.get("delimiter").and_then(|d| d.as_str()).unwrap_or("");
                let parts: Vec<String> = ins
                    .iter()
                    .filter(|(k, _)| k.starts_with("string") || k.starts_with("text"))
                    .filter_map(|(_, v)| string_value(nodes, v, depth + 1))
                    .filter(|p| !p.is_empty())
                    .collect();
                return (!parts.is_empty()).then(|| parts.join(delimiter));
            }
            ["value", "text", "string", "String", "prompt"].iter().find_map(|k| string_value(nodes, ins.get(*k)?, depth + 1))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temporary test file, deleted when dropped (tests run in parallel, so each owns its file).
    struct TempPng(std::path::PathBuf);

    impl std::ops::Deref for TempPng {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPng {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// A 1x1 PNG with the given text chunks (tEXt, or iTXt / zTXt by keyword prefix "i:" / "z:").
    fn png_with(chunks: &[(&str, &str)]) -> TempPng {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("pc-png-meta-{}-{n}.png", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 1, 1);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        for (k, t) in chunks {
            match k.split_once(':') {
                Some(("i", key)) => enc.add_itxt_chunk(key.into(), t.to_string()).unwrap(),
                Some(("z", key)) => enc.add_ztxt_chunk(key.into(), t.to_string()).unwrap(),
                _ => enc.add_text_chunk(k.to_string(), t.to_string()).unwrap(),
            }
        }
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&[0]).unwrap();
        w.finish().unwrap();
        TempPng(path)
    }

    #[test]
    fn a1111_parameters() {
        let p = png_with(&[(
            "parameters",
            "1girl, (red hair:1.2), smile\nNegative prompt: lowres, bad hands\nSteps: 28, Sampler: Euler a, CFG scale: 7, Seed: 1",
        )]);
        let r = read_png_prompt(&p).unwrap().unwrap();
        assert_eq!(r.positive, "1girl, (red hair:1.2), smile");
        assert_eq!(r.negative.as_deref(), Some("lowres, bad hands"));
        assert_eq!((r.source, r.partial), ("A1111 / Forge", false));
        // no negative; also as a compressed chunk
        let p = png_with(&[("z:parameters", "1boy, armor\nSteps: 20, Sampler: DPM++ 2M")]);
        let r = read_png_prompt(&p).unwrap().unwrap();
        assert_eq!((r.positive.as_str(), r.negative), ("1boy, armor", None));
    }

    #[test]
    fn comfyui_graph() {
        let graph = r#"{
            "3": {"class_type": "KSampler", "inputs": {"positive": ["10", 0], "negative": ["7", 0], "model": ["4", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, beach, sunset", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres", "clip": ["4", 1]}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "an unused but much longer text that is not wired anywhere", "clip": ["4", 1]}},
            "10": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["6", 0], "conditioning_2": ["6", 0]}}
        }"#;
        let r = read_png_prompt(&png_with(&[("prompt", graph)])).unwrap().unwrap();
        assert_eq!(r.positive, "1girl, beach, sunset");
        assert_eq!(r.negative.as_deref(), Some("lowres"));
        assert!(!r.partial);
        // no sampler to follow: the longest text, marked partial
        let loose = r#"{"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "short"}},
                        "2": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "1girl, long hair, library"}}}"#;
        let r = read_png_prompt(&png_with(&[("i:prompt", loose)])).unwrap().unwrap();
        assert_eq!((r.positive.as_str(), r.partial), ("1girl, long hair, library", true));
    }

    #[test]
    fn comfyui_text_through_string_nodes() {
        // the positive is two primitives joined by a concatenate node; the negative a primitive
        let graph = r#"{
            "1": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "masterpiece, "}},
            "2": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "1girl, red hair"}},
            "3": {"class_type": "StringConcatenate", "inputs": {"string_a": ["1", 0], "string_b": ["2", 0], "delimiter": ""}},
            "4": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a much longer negative prompt, lowres, bad hands, blurry"}},
            "9:47": {"class_type": "CLIPTextEncode", "inputs": {"text": ["3", 0]}},
            "9:78": {"class_type": "CLIPTextEncode", "inputs": {"text": ["4", 0]}},
            "9:80": {"class_type": "KSampler", "inputs": {"positive": ["9:47", 0], "negative": ["9:78", 0], "latent_image": ["9:28", 0]}},
            "9:28": {"class_type": "EmptyLatentImage", "inputs": {}}
        }"#;
        let r = read_png_prompt(&png_with(&[("prompt", graph)])).unwrap().unwrap();
        assert_eq!(r.positive, "masterpiece, 1girl, red hair");
        assert_eq!(r.negative.as_deref(), Some("a much longer negative prompt, lowres, bad hands, blurry"));
        assert!(!r.partial);
    }

    #[test]
    fn comfyui_picks_the_first_pass_when_samplers_differ() {
        // a second sampler (id 2, upscale pass) with its own prompt; the first pass (id 10) starts from
        // an empty latent, has no sampler upstream and holds the content prompt
        let graph = r#"{
            "2": {"class_type": "KSampler", "inputs": {"positive": ["21", 0], "negative": ["12", 0], "latent_image": ["30", 0]}},
            "10": {"class_type": "KSampler", "inputs": {"positive": ["11", 0], "negative": ["12", 0], "latent_image": ["5", 0]}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216}},
            "30": {"class_type": "LatentUpscale", "inputs": {"samples": ["10", 0]}},
            "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, garden, rain"}},
            "21": {"class_type": "CLIPTextEncode", "inputs": {"text": "highly detailed, sharp"}},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres"}}
        }"#;
        let r = read_png_prompt(&png_with(&[("prompt", graph)])).unwrap().unwrap();
        assert_eq!(r.positive, "1girl, garden, rain");
        assert_eq!((r.sampler.as_deref(), r.distinct_positives, r.first_pass_chosen), (Some("10 KSampler"), 2, true));
        // samplers sharing one prompt: still the first pass, but no note
        let shared = graph.replace(r#""positive": ["21", 0]"#, r#""positive": ["11", 0]"#);
        let r = read_png_prompt(&png_with(&[("prompt", &shared)])).unwrap().unwrap();
        assert_eq!((r.sampler.as_deref(), r.distinct_positives, r.first_pass_chosen), (Some("10 KSampler"), 1, false));
        // img2img (the first pass starts from an encoded image): still the pass with no sampler upstream
        let img2img = graph.replace("EmptyLatentImage", "VAEEncode");
        let r = read_png_prompt(&png_with(&[("prompt", &img2img)])).unwrap().unwrap();
        assert_eq!((r.sampler.as_deref(), r.first_pass_chosen), (Some("10 KSampler"), true));
    }

    #[test]
    fn comfyui_follows_guiders_and_takes_the_first_pass() {
        // pass 1: SamplerCustomAdvanced with a guider and a reference latent; pass 2: KSampler refining it
        let graph = r#"{
            "12": {"class_type": "AnimaRefLatent", "inputs": {"image": ["10", 0]}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, beach, sunset, long hair"}},
            "14": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres"}},
            "15": {"class_type": "CFGGuider", "inputs": {"positive": ["13", 0], "negative": ["14", 0]}},
            "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "19": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["15", 0], "sampler": ["16", 0], "latent_image": ["12", 1]}},
            "22": {"class_type": "CLIPTextEncode", "inputs": {"text": "detailed"}},
            "23": {"class_type": "KSampler", "inputs": {"positive": ["22", 0], "negative": ["14", 0], "latent_image": ["19", 0]}}
        }"#;
        let r = read_png_prompt(&png_with(&[("prompt", graph)])).unwrap().unwrap();
        assert_eq!(r.positive, "1girl, beach, sunset, long hair");
        assert_eq!(r.negative.as_deref(), Some("lowres"));
        assert_eq!((r.sampler.as_deref(), r.distinct_positives, r.first_pass_chosen), (Some("19 SamplerCustomAdvanced"), 2, true));
    }

    /// Real ComfyUI PNGs, without printing their prompts (private):
    /// `PNG_TEST=a.png:b.png cargo test live_png_prompt -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_png_prompt() {
        for path in std::env::var("PNG_TEST").unwrap().split(':') {
            let name = Path::new(path).file_name().unwrap().to_string_lossy().into_owned();
            match read_png_prompt(Path::new(path)) {
                Ok(Some(p)) => println!(
                    "\n{name}: format {}, partial {}, sampler {}, distinct positives {}, positive {} words",
                    p.source,
                    if p.partial { "yes" } else { "no" },
                    p.sampler.as_deref().unwrap_or("-"),
                    p.distinct_positives,
                    p.positive.split_whitespace().count(),
                ),
                Ok(None) => println!("\n{name}: no prompt found"),
                // the error text never holds chunk contents (png decoder / io errors only)
                Err(e) => println!("\n{name}: error {e}"),
            }
        }
    }

    #[test]
    fn novelai_comment_and_description() {
        let p = png_with(&[
            ("Description", "1girl, kimono"),
            ("Comment", r#"{"prompt": "1girl, {kimono}, shrine", "uc": "lowres", "steps": 28}"#),
        ]);
        let r = read_png_prompt(&p).unwrap().unwrap();
        assert_eq!((r.positive.as_str(), r.negative.as_deref(), r.source), ("1girl, {kimono}, shrine", Some("lowres"), "NovelAI"));
        let r = read_png_prompt(&png_with(&[("Description", "1boy, rain")])).unwrap().unwrap();
        assert_eq!(r.positive, "1boy, rain");
    }

    #[test]
    fn no_metadata_gives_nothing() {
        assert_eq!(read_png_prompt(&png_with(&[])).unwrap(), None);
        assert_eq!(read_png_prompt(&png_with(&[("Software", "GIMP"), ("parameters", "")])).unwrap(), None);
        // not a PNG at all
        let bad = TempPng(std::env::temp_dir().join(format!("pc-png-meta-bad-{}.png", std::process::id())));
        std::fs::write(&*bad, b"not a png").unwrap();
        assert!(read_png_prompt(&bad).is_err());
    }
}
