//! Danbooru tagger: SmilingWolf's WD EVA02-Large v3, run locally with ONNX Runtime.
//! Deterministic (same image -> same tags) and trained on Danbooru, so its output
//! uses real Danbooru tag names.

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::Mutex;

pub const REPO: &str = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main";
const INPUT_SIZE: u32 = 448;
/// Character tags need much more confidence than general ones (model card defaults).
const CHARACTER_THRESHOLD: f32 = 0.85;

/// Danbooru tag categories used in `selected_tags.csv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    General,
    Character,
    Rating,
}

#[derive(Debug, Clone)]
pub struct ScoredTag {
    pub name: String,
    pub kind: TagKind,
    pub score: f32,
}

pub fn tagger_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("prompt-condenser").join("wd-eva02-large-tagger-v3")
}

pub fn is_installed() -> bool {
    let dir = tagger_dir();
    dir.join("model.onnx").is_file() && dir.join("selected_tags.csv").is_file()
}

/// Downloads the model (~1.2 GB) and tag list, reporting progress.
pub fn download(progress: Sender<String>) -> Result<()> {
    let dir = tagger_dir();
    std::fs::create_dir_all(&dir)?;
    let http = reqwest::blocking::Client::builder().timeout(None).build()?;
    for file in ["selected_tags.csv", "model.onnx"] {
        let dest = dir.join(file);
        if dest.is_file() {
            continue;
        }
        let mut resp = http
            .get(format!("{REPO}/{file}"))
            .header("User-Agent", "prompt-condenser")
            .send()
            .with_context(|| format!("Couldn't download {file}"))?;
        if !resp.status().is_success() {
            return Err(anyhow!("Downloading {file} failed: HTTP {}", resp.status()));
        }
        let total = resp.content_length().unwrap_or(0);
        let part = dest.with_extension("part");
        let mut out = std::fs::File::create(&part)?;
        let mut buf = vec![0u8; 1 << 16];
        let (mut done, mut last_pct) = (0u64, u64::MAX);
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            if let Some(pct) = (done * 100).checked_div(total) {
                if pct != last_pct {
                    last_pct = pct;
                    let _ = progress.send(format!(
                        "Downloading Danbooru tagger: {} / {} MB ({pct}%)",
                        done / (1 << 20),
                        total / (1 << 20)
                    ));
                }
            }
        }
        out.flush()?;
        std::fs::rename(&part, &dest)?;
    }
    Ok(())
}

/// ONNX Runtime allows one environment per process; the tagger and the person
/// detector share it. It lives for the whole run, so it outlives every session.
static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
static ENV_INIT: Mutex<()> = Mutex::new(());

pub(crate) fn ort_env() -> Result<&'static ort::environment::Environment> {
    let _guard = ENV_INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("prompt-condenser").build()?;
    Ok(ENV.get_or_init(|| env))
}

struct Loaded {
    session: Session,
    tags: Vec<(String, TagKind)>,
}

/// Loading the 1.2 GB model takes a few seconds, so it stays loaded after first use.
static MODEL: Mutex<Option<Loaded>> = Mutex::new(None);

/// Frees the loaded tagger (about 1.5 GB of RAM) and the person detector. Called before a big text
/// model loads: on this shared-memory machine the 12B didn't fit next to them and the system
/// swapped. The next image read loads them again (~5 s).
pub fn release() {
    *MODEL.lock().unwrap_or_else(|e| e.into_inner()) = None;
    *MODEL2.lock().unwrap_or_else(|e| e.into_inner()) = None;
    crate::detector::release();
}

fn load_tags(path: &Path) -> Result<Vec<(String, TagKind)>> {
    let text = std::fs::read_to_string(path)?;
    let mut tags = Vec::new();
    // tag_id,name,category,count
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let name = fields[1..fields.len() - 2].join(",");
        let kind = match fields[fields.len() - 2] {
            "9" => TagKind::Rating,
            "4" => TagKind::Character,
            _ => TagKind::General,
        };
        tags.push((name, kind));
    }
    Ok(tags)
}

/// WD v3 preprocessing: pad to a white square, resize to 448, BGR, 0-255, NHWC.
fn preprocess(img: &image::DynamicImage) -> Result<Vec<f32>> {
    let img = img.to_rgba8();
    let side = img.width().max(img.height());
    let mut canvas = image::RgbaImage::from_pixel(side, side, image::Rgba([255, 255, 255, 255]));
    image::imageops::overlay(
        &mut canvas,
        &img,
        ((side - img.width()) / 2) as i64,
        ((side - img.height()) / 2) as i64,
    );
    let resized = image::imageops::resize(&canvas, INPUT_SIZE, INPUT_SIZE, image::imageops::FilterType::CatmullRom);
    let mut data = Vec::with_capacity((INPUT_SIZE * INPUT_SIZE * 3) as usize);
    for p in resized.pixels() {
        data.extend_from_slice(&[p[2] as f32, p[1] as f32, p[0] as f32]);
    }
    Ok(data)
}

/// Scores every Danbooru tag for the image.
#[cfg(test)]
pub fn tag_image(path: &Path) -> Result<Vec<ScoredTag>> {
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    tag_dynamic(&img)
}

/// Scores every Danbooru tag for an image already in memory (e.g. one character's crop).
pub fn tag_dynamic(img: &image::DynamicImage) -> Result<Vec<ScoredTag>> {
    let first = tag_many(&[img])?.remove(0);
    match second_tagger_dir() {
        Some(dir) => Ok(blend_scores(first, second_tagger_scores(img, &dir)?)),
        None => Ok(first),
    }
}

/// Tags several images in one run of the model. (Batching measured ~5x slower than
/// one at a time on this CPU with little free memory, so callers tag one by one.)
fn tag_many(imgs: &[&image::DynamicImage]) -> Result<Vec<Vec<ScoredTag>>> {
    if !is_installed() {
        return Err(anyhow!("The Danbooru tagger isn't downloaded yet."));
    }
    let mut input = Vec::with_capacity(imgs.len() * (INPUT_SIZE * INPUT_SIZE * 3) as usize);
    for img in imgs {
        input.extend(preprocess(img)?);
    }
    with_model(|loaded| {
        let n = imgs.len();
        let tensor = Tensor::from_array(([n, INPUT_SIZE as usize, INPUT_SIZE as usize, 3], input.into_boxed_slice()))?;
        let outputs = loaded.session.run(ort::inputs![tensor])?;
        let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;
        let per = loaded.tags.len();
        if scores.len() < n * per {
            return Err(anyhow!("Tagger model returned {} scores, expected {}", scores.len(), n * per));
        }
        Ok((0..n)
            .map(|i| {
                loaded
                    .tags
                    .iter()
                    .zip(scores[i * per..(i + 1) * per].iter())
                    .map(|((name, kind), &score)| ScoredTag { name: name.clone(), kind: *kind, score })
                    .collect()
            })
            .collect())
    })
}

/// Experiment (testing only): a second tagger whose scores are averaged in, switched on by `PC_TAGGER2_DIR`
/// (a folder with `model.onnx` and `selected_tags.csv`). Built for Redstonexs danbooru-tagger-v1 (EVA02-Large,
/// 24k tags): pad to white, 448, BGR, (x / 255 - 0.5) / 0.5, NCHW, logits. Shared tags get the mean of both
/// scores, tags only one model knows keep that model's score (Gemini's ensemble benchmark).
static MODEL2: Mutex<Option<Loaded>> = Mutex::new(None);

fn second_tagger_dir() -> Option<PathBuf> {
    std::env::var_os("PC_TAGGER2_DIR").map(PathBuf::from).filter(|d| d.join("model.onnx").exists())
}

/// `tag_id,name,category,…`: name and category by position (the Redstonexs list has extra columns).
fn load_tags_by_position(path: &Path) -> Result<Vec<(String, TagKind)>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            (f.len() >= 3).then(|| {
                let kind = match f[2] {
                    "9" => TagKind::Rating,
                    "4" => TagKind::Character,
                    _ => TagKind::General,
                };
                (f[1].to_string(), kind)
            })
        })
        .collect())
}

fn second_tagger_scores(img: &image::DynamicImage, dir: &Path) -> Result<Vec<ScoredTag>> {
    let mut guard = MODEL2.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let session = Session::builder(ort_env()?)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(dir.join("model.onnx"))
            .context("Couldn't load the second tagger")?;
        *guard = Some(Loaded { session, tags: load_tags_by_position(&dir.join("selected_tags.csv"))? });
    }
    let loaded = guard.as_mut().expect("just loaded");
    // same padding / size as WD, then normalized BGR in NCHW
    let hwc = preprocess(img)?;
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut chw = vec![0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            chw[c * n + i] = (hwc[i * 3 + c] / 255.0 - 0.5) / 0.5;
        }
    }
    let tensor = Tensor::from_array(([1usize, 3, INPUT_SIZE as usize, INPUT_SIZE as usize], chw.into_boxed_slice()))?;
    let outputs = loaded.session.run(ort::inputs![tensor])?;
    let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(loaded
        .tags
        .iter()
        .zip(logits.iter())
        .map(|((name, kind), &l)| ScoredTag { name: name.clone(), kind: *kind, score: 1.0 / (1.0 + (-l).exp()) })
        .collect())
}

/// Averages the second tagger's scores into the first's (shared tags: mean; one model only: its score).
fn blend_scores(first: Vec<ScoredTag>, second: Vec<ScoredTag>) -> Vec<ScoredTag> {
    let mut other: std::collections::HashMap<String, ScoredTag> = second.into_iter().map(|t| (t.name.clone(), t)).collect();
    let mut out: Vec<ScoredTag> = first
        .into_iter()
        .map(|mut t| {
            if let Some(o) = other.remove(&t.name) {
                t.score = 0.5 * (t.score + o.score);
            }
            t
        })
        .collect();
    out.extend(other.into_values());
    out
}

fn with_model<T>(f: impl FnOnce(&mut Loaded) -> Result<T>) -> Result<T> {
    let mut guard = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let dir = tagger_dir();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let session = Session::builder(ort_env()?)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(dir.join("model.onnx"))
            .context("Couldn't load the tagger model")?;
        *guard = Some(Loaded { session, tags: load_tags(&dir.join("selected_tags.csv"))? });
    }
    f(guard.as_mut().expect("just loaded"))
}

/// Checks whether a normalized tag represents a concrete environmental object or prop
/// (plants, foliage, furniture, fixtures, handheld props) rather than generic framing or abstract setting words.
pub fn is_concrete_scene_object(norm: &str) -> bool {
    crate::prompt_condenser::is_concrete_scene_object(norm)
}

/// Adds what a zoomed-in crop (`detector::upper_crop`, `middle_crop`, `lower_crop`) sees better:
/// confident (>= `DETAIL_CROP_MIN`) appearance tags (hair, eyes, traits, body, outfit, expression, held props, micro-poses,
/// and concrete scene objects like plants or furniture). Framing, count, size tags and generic backgrounds
/// from the crop are ignored, since zooming in changes those.
/// The lowest score a zoomed-in crop's tag needs to be added (see `merge_detail_crop`).
const DETAIL_CROP_MIN: f32 = 0.75;

pub fn merge_detail_crop(base: &[ScoredTag], detail: &[ScoredTag]) -> Vec<ScoredTag> {
    use crate::prompt_condenser::{self, CropType};

    let base_tuples: Vec<(String, f32)> = base.iter().map(|t| (t.name.clone(), t.score)).collect();
    let detail_tuples: Vec<(String, f32)> = detail
        .iter()
        .filter(|d| d.kind == TagKind::General)
        .map(|t| (t.name.clone(), t.score))
        .collect();

    let merged = prompt_condenser::merge_crop_tags(
        &base_tuples,
        &detail_tuples,
        CropType::CharacterZone,
        &[],
        // Only tags the crop is sure of: at 0.55 / 0.40 it added 129 tags on the 12 test images, 34% of them
        // correct (thighs, legs, border, invisible…); at 0.75 the tagger-only score is ~1 point higher.
        DETAIL_CROP_MIN,
        DETAIL_CROP_MIN,
    );

    restore_kinds(base, merged)
}

/// Words that place a tag in a body zone: a zone crop may add a fitting tag at a lower bar than `DETAIL_CROP_MIN`
/// (the head crop can't really show thighs, so a head tag from it is trustworthy at 0.45). Measured on the upper
/// zone alone: fox_vtuber `hair flower` 0.46, sakura_wings `white wings` 0.46, both correct and both lost at 0.75.
const UPPER_ZONE_WORDS: &[&str] = &[
    "hair", "eyes", "ears", "ear", "earrings", "hat", "beret", "cap", "headwear", "hairband", "hairclip", "headband",
    "horns", "horn", "halo", "wings", "choker", "collar", "necklace", "glasses", "eyewear", "mouth", "smile", "blush",
    "teeth", "tongue", "fang", "eyebrows", "eyelashes", "lips", "makeup", "freckles", "mole", "face", "ahoge",
    "bangs", "sidelocks", "ponytail", "twintails", "braid", "bun", "bobbles", "flower", "ornament", "veil", "mask",
    "headdress", "tiara", "crown", "neckerchief", "necktie", "bowtie", "scrunchie", "pupils", "eyeshadow",
];
const MIDDLE_ZONE_WORDS: &[&str] = &[
    "shirt", "jacket", "top", "dress", "sleeves", "gloves", "navel", "midriff", "cleavage", "bikini", "bra", "sweater",
    "cardigan", "vest", "apron", "belt", "holding", "arm", "arms", "hand", "hands", "fingers", "nails", "bag", "blazer",
    "coat", "hoodie", "uniform", "serafuku", "leotard", "bodysuit", "armor", "cape", "cloak", "corset", "stomach",
    "shoulders", "shoulder", "underboob", "sideboob", "collarbone", "bracelet", "wristband", "cuffs", "sash", "obi",
];
const LOWER_ZONE_WORDS: &[&str] = &[
    "skirt", "shorts", "pants", "panties", "thighhighs", "socks", "kneehighs", "pantyhose", "legwear", "boots", "shoes",
    "footwear", "sneakers", "sandals", "heels", "loafers", "uwabaki", "barefoot", "feet", "legs", "thighs", "knees",
    "garter", "buruma", "bloomers", "thigh", "strap", "leggings", "stockings",
];

/// The bar a zone-fitting tag needs (`PC_ZONE_BAR` overrides it, testing only).
fn zone_bar(upper: bool) -> f32 {
    std::env::var("PC_ZONE_BAR").ok().and_then(|v| v.parse().ok()).unwrap_or(if upper { 0.45 } else { 0.50 })
}

/// `merge_detail_crop` for one of `detector::three_zone_crops` (label "Upper…", "Middle…", "Lower…"): tags that
/// fit the zone need only `zone_bar`, all others still `DETAIL_CROP_MIN`. Framing, size and artifact tags are
/// still rejected by `merge_crop_tags`.
pub fn merge_zone_crop(base: &[ScoredTag], detail: &[ScoredTag], zone_label: &str) -> Vec<ScoredTag> {
    let words: &[&str] = if zone_label.starts_with("Upper") {
        UPPER_ZONE_WORDS
    } else if zone_label.starts_with("Middle") {
        MIDDLE_ZONE_WORDS
    } else if zone_label.starts_with("Lower") {
        LOWER_ZONE_WORDS
    } else {
        return merge_detail_crop(base, detail);
    };
    let bar = zone_bar(zone_label.starts_with("Upper"));
    let fits = |name: &str| name.split(['_', ' ']).any(|w| words.contains(&w));
    let kept: Vec<ScoredTag> = detail
        .iter()
        .filter(|t| t.kind == TagKind::General && t.score >= if fits(&t.name) { bar } else { DETAIL_CROP_MIN })
        .cloned()
        .collect();
    let base_tuples: Vec<(String, f32)> = base.iter().map(|t| (t.name.clone(), t.score)).collect();
    let kept_tuples: Vec<(String, f32)> = kept.iter().map(|t| (t.name.clone(), t.score)).collect();
    let merged = crate::prompt_condenser::merge_crop_tags(
        &base_tuples,
        &kept_tuples,
        crate::prompt_condenser::CropType::CharacterZone,
        &[],
        bar.min(DETAIL_CROP_MIN),
        bar.min(DETAIL_CROP_MIN),
    );
    restore_kinds(base, merged)
}

/// Gives merged (name, score) pairs back their tag kind from `base` (General for new ones).
/// The tagger scores all ~10,000 tags, so this uses one lookup table: searching `base` for
/// every tag made each merge ~100 million string comparisons (minutes per image read).
fn restore_kinds(base: &[ScoredTag], merged: Vec<(String, f32)>) -> Vec<ScoredTag> {
    let mut kinds: std::collections::HashMap<String, TagKind> = std::collections::HashMap::with_capacity(base.len());
    for b in base {
        kinds.entry(crate::danbooru::key(&b.name)).or_insert(b.kind);
    }
    merged
        .into_iter()
        .map(|(name, score)| {
            let kind = kinds.get(&crate::danbooru::key(&name)).copied().unwrap_or(TagKind::General);
            ScoredTag { name, score, kind }
        })
        .collect()
}

/// Merges tags discovered in a scene quadrant crop (TL, TR, BL, BR), specifically keeping
/// confident (>= 0.40) concrete scene objects (plants, flowers, lamps, furniture, props)
/// and character micro-details (>= 0.55), while discarding false zoom-framing tags.
pub fn merge_quadrant_scene_crop(base: &[ScoredTag], quad: &[ScoredTag]) -> Vec<ScoredTag> {
    use crate::prompt_condenser::{self, CropType};

    let base_tuples: Vec<(String, f32)> = base.iter().map(|t| (t.name.clone(), t.score)).collect();
    let quad_tuples: Vec<(String, f32)> = quad
        .iter()
        .filter(|q| q.kind == TagKind::General)
        .map(|t| (t.name.clone(), t.score))
        .collect();

    let merged = prompt_condenser::merge_crop_tags(
        &base_tuples,
        &quad_tuples,
        CropType::Quadrant,
        &[],
        0.55,
        0.40,
    );

    restore_kinds(base, merged)
}

/// Extracts the top confident WD14 detections from a specific crop (Upper, Middle, Lower, or Quadrant)
/// to pass as explicit visual anchors to Qwen 2.5-VL so it pinpoints the exact object sub-type.
pub fn extract_zone_triggers(scores: &[ScoredTag], threshold: f32) -> Vec<String> {
    let mut picks: Vec<String> = scores
        .iter()
        .filter(|t| t.kind == TagKind::General && t.score >= threshold)
        .map(|t| to_prompt_tag(&t.name))
        .filter(|t| {
            let l = t.to_lowercase();
            !l.contains("background")
                && !l.ends_with("shot")
                && !l.ends_with("view")
                && !l.ends_with("angle")
                && l != "solo"
                && l != "1girl"
                && l != "2girls"
                && l != "3girls"
                && l != "4girls"
        })
        .collect();
    picks.truncate(15);
    picks
}

/// Emoticon tags keep their underscores (`^_^`, `>_<`...).
const KAOMOJI: &[&str] = &[
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@",
    "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
];

/// Danbooru name -> prompt tag: underscores become spaces and parentheses are escaped
/// so ComfyUI doesn't read `(series)` as a weight.
pub fn to_prompt_tag(name: &str) -> String {
    let t = if KAOMOJI.contains(&name) { name.to_string() } else { name.replace('_', " ") };
    t.replace('(', "\\(").replace(')', "\\)")
}

pub struct TagPrompt {
    pub prompt: String,
    pub rating: String,
    pub characters: usize,
    pub general: usize,
}

/// Builds the prompt: character tags first, then general tags by confidence.
pub fn build_prompt(scores: &[ScoredTag], general_threshold: f32) -> TagPrompt {
    let mut characters: Vec<&ScoredTag> =
        scores.iter().filter(|t| t.kind == TagKind::Character && t.score >= CHARACTER_THRESHOLD).collect();
    let mut general: Vec<&ScoredTag> =
        scores.iter().filter(|t| t.kind == TagKind::General && t.score >= general_threshold).collect();
    characters.sort_by(|a, b| b.score.total_cmp(&a.score));
    general.sort_by(|a, b| b.score.total_cmp(&a.score));
    // the tagger scores each size on its own, so "medium breasts" and "large breasts"
    // can both pass: keep only the most confident one (the list is sorted by score)
    let mut picked: Vec<&ScoredTag> = Vec::new();
    for t in general {
        let n = crate::danbooru::key(&t.name);
        if !picked.iter().any(|p| crate::analyzer::size_conflict(&crate::danbooru::key(&p.name), &n)) {
            picked.push(t);
        }
    }
    let general = picked;
    let rating = scores
        .iter()
        .filter(|t| t.kind == TagKind::Rating)
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .map(|t| t.name.clone())
        .unwrap_or_default();
    TagPrompt {
        prompt: characters.iter().chain(&general).map(|t| to_prompt_tag(&t.name)).collect::<Vec<_>>().join(", "),
        rating,
        characters: characters.len(),
        general: general.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_crop_adds_details_but_not_framing() {
        let t = |name: &str, score| ScoredTag { name: name.into(), kind: TagKind::General, score };
        let base = vec![t("1girl", 0.9), t("full_body", 0.8), t("thick_eyebrows", 0.2), t("glasses", 0.3)];
        let detail = vec![
            t("portrait", 0.9), t("close-up", 0.8), t("upper_body", 0.9), t("solo", 0.9), t("simple_background", 0.7),
            t("thick_eyebrows", 0.8), t("glasses", 0.7), t("hair_ornament", 0.8),
        ];
        let m = merge_detail_crop(&base, &detail);
        let score = |n: &str| m.iter().find(|x| x.name == n).map(|x| x.score);
        assert_eq!(score("thick_eyebrows"), Some(0.8));
        assert_eq!(score("glasses"), Some(0.3)); // below DETAIL_CROP_MIN in the crop: unchanged
        assert_eq!(score("hair_ornament"), Some(0.8));
        for n in ["portrait", "close-up", "upper_body", "simple_background"] {
            assert_eq!(score(n), None, "{n}");
        }
        assert_eq!(score("full_body"), Some(0.8));
        // no sizes from the zoomed crop
        let m = merge_detail_crop(&base, &[t("large_breasts", 0.9)]);
        assert!(m.iter().all(|x| x.name != "large_breasts"));
        // one size per body part in the prompt: the most confident
        let p = build_prompt(&[t("medium_breasts", 0.5), t("large_breasts", 0.7), t("long_hair", 0.9), t("very_long_hair", 0.6)], 0.35);
        assert_eq!(p.prompt, "long hair, large breasts, very long hair");
    }

    #[test]
    fn quadrant_crop_merges_concrete_scene_objects_and_held_props() {
        let t = |name: &str, score| ScoredTag { name: name.into(), kind: TagKind::General, score };
        let base = vec![t("1girl", 0.9), t("full_body", 0.8), t("simple_background", 0.8)];
        let quad = vec![
            t("potted_plant", 0.65),
            t("flower", 0.70),
            t("holding_sword", 0.60),
            t("white_background", 0.75),
            t("upper_body", 0.80),
        ];
        let m = merge_quadrant_scene_crop(&base, &quad);
        let score = |n: &str| m.iter().find(|x| x.name == n).map(|x| x.score);
        assert_eq!(score("potted_plant"), Some(0.65));
        assert_eq!(score("flower"), Some(0.70));
        assert_eq!(score("holding_sword"), Some(0.60));
        assert_eq!(score("white_background"), None);
        assert_eq!(score("upper_body"), None);
    }

    #[test]
    fn danbooru_names_become_prompt_tags() {
        assert_eq!(to_prompt_tag("long_hair"), "long hair");
        assert_eq!(to_prompt_tag("^_^"), "^_^");
        assert_eq!(to_prompt_tag("2b_(nier:automata)"), "2b \\(nier:automata\\)");
    }

    #[test]
    fn prompt_orders_characters_first_and_applies_thresholds() {
        let t = |name: &str, kind, score| ScoredTag { name: name.into(), kind, score };
        let scores = vec![
            t("general", TagKind::Rating, 0.9),
            t("smile", TagKind::General, 0.6),
            t("1girl", TagKind::General, 0.99),
            t("hatsune_miku", TagKind::Character, 0.95),
            t("some_char", TagKind::Character, 0.5),
            t("hat", TagKind::General, 0.2),
        ];
        let p = build_prompt(&scores, 0.35);
        assert_eq!(p.prompt, "hatsune miku, 1girl, smile");
        assert_eq!(p.rating, "general");
        assert_eq!((p.characters, p.general), (1, 2));
    }

    /// Downloads the tagger if needed and tags real images:
    /// `TAG_TEST_IMAGES=a.png:b.jpg cargo test live_tagger -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_tagger() {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let printer = std::thread::spawn(move || {
            for m in rx {
                if m.ends_with("0%)") {
                    println!("{m}");
                }
            }
        });
        download(tx).unwrap();
        printer.join().unwrap();
        for path in std::env::var("TAG_TEST_IMAGES").unwrap().split(':') {
            let t = std::time::Instant::now();
            let scores = tag_image(Path::new(path)).unwrap();
            let p = build_prompt(&scores, 0.35);
            println!("{path}: {:.1}s, rating {}, {} tags\n  {}", t.elapsed().as_secs_f32(), p.rating, p.characters + p.general, p.prompt);
        }
    }
}

