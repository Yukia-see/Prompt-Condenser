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
pub fn tag_image(path: &Path) -> Result<Vec<ScoredTag>> {
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    tag_dynamic(&img)
}

/// Scores every Danbooru tag for an image already in memory (e.g. one character's crop).
pub fn tag_dynamic(img: &image::DynamicImage) -> Result<Vec<ScoredTag>> {
    Ok(tag_many(&[img])?.remove(0))
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
    let n = norm.replace('_', " ");
    const OBJECTS: &[&str] = &[
        "plant", "potted plant", "houseplant", "flower", "flowers", "bouquet", "petal", "rose",
        "sunflower", "lily", "tulip", "cherry blossoms", "tree", "leaf", "leaves", "foliage",
        "fern", "vines", "ivy", "grass", "bush", "hedge", "cactus", "succulent", "vase",
        "flowerpot", "flower pot", "window", "curtain", "curtains", "drapes", "chair", "armchair",
        "couch", "sofa", "table", "desk", "bookshelf", "bookcase", "shelf", "lamp", "lantern",
        "chandelier", "street lamp", "mirror", "clock", "poster", "painting", "picture frame",
        "signboard", "pillow", "cushion", "carpet", "rug", "mat", "bed", "television", "monitor",
        "laptop", "computer", "cup", "mug", "teacup", "bottle", "wine glass", "book", "umbrella",
        "parasol", "weapon", "sword", "gun", "staff", "wand", "fan", "food", "drink", "phone",
        "cellphone", "smartphone", "camera", "plushie", "doll", "instrument", "guitar", "violin",
    ];
    OBJECTS.iter().any(|&obj| n == obj || n.contains(obj))
}

/// Adds what a zoomed-in crop (`detector::upper_crop`, `middle_crop`, `lower_crop`) sees better:
/// confident (>= 0.55) appearance tags (hair, eyes, traits, body, outfit, expression, held props, micro-poses,
/// and concrete scene objects like plants or furniture). Framing, count, size tags and generic backgrounds
/// from the crop are ignored, since zooming in changes those.
pub fn merge_detail_crop(base: &[ScoredTag], detail: &[ScoredTag]) -> Vec<ScoredTag> {
    use crate::scene::Slot;
    let mut out = base.to_vec();
    for d in detail.iter().filter(|d| d.kind == TagKind::General && d.score >= 0.55) {
        let prompt = to_prompt_tag(&d.name);
        let Some(t) = crate::parser::parse_prompt(&prompt).into_iter().next() else { continue };
        let is_concrete_scene = is_concrete_scene_object(&t.normalized);
        let is_micro_pose = crate::scene::slot(&t) == Slot::Pose && (
            t.normalized.starts_with("holding")
            || t.normalized.starts_with("adjusting")
            || t.normalized.starts_with("pulling")
            || t.normalized.contains("knee")
            || t.normalized.contains("hand")
            || t.normalized.contains("leg")
            || t.normalized == "v"
            || t.normalized == "v sign"
            || t.normalized == "peace sign"
        );
        let detail_slot = matches!(
            crate::scene::slot(&t),
            Slot::Hair | Slot::Eyes | Slot::Traits | Slot::Body | Slot::Outfit | Slot::Expression | Slot::Other
        ) || is_micro_pose || (crate::scene::slot(&t) == Slot::Setting && is_concrete_scene);
        // zooming in makes the body look bigger, so sizes and proportions come from the
        // full view only
        if !detail_slot
            || crate::two::count_tag(&t.normalized).is_some()
            || t.normalized == "solo"
            || crate::analyzer::is_size_tag(&t.normalized)
        {
            continue;
        }
        match out.iter_mut().find(|b| b.name == d.name) {
            Some(b) => b.score = b.score.max(d.score),
            None => out.push(d.clone()),
        }
    }
    out
}

/// Merges tags discovered in a scene quadrant crop (TL, TR, BL, BR), specifically keeping
/// confident (>= 0.40) concrete scene objects (plants, flowers, lamps, furniture, props)
/// and character micro-details (>= 0.55), while discarding false zoom-framing tags.
pub fn merge_quadrant_scene_crop(base: &[ScoredTag], quad: &[ScoredTag]) -> Vec<ScoredTag> {
    use crate::scene::Slot;
    let mut out = base.to_vec();
    for q in quad.iter().filter(|q| q.kind == TagKind::General) {
        let norm = crate::danbooru::key(&q.name);
        // Exclude framing, camera, and generic background tags
        if crate::two::count_tag(&norm).is_some()
            || norm == "solo"
            || crate::analyzer::is_size_tag(&norm)
            || norm == "simple background"
            || norm == "white background"
            || norm == "grey background"
            || norm == "monochrome"
            || (norm.ends_with("background") && !is_concrete_scene_object(&norm))
            || norm.ends_with("shot")
            || norm.ends_with("view")
            || norm.ends_with("angle")
            || norm == "close-up"
            || norm == "portrait"
            || norm == "upper body"
            || norm == "lower body"
            || norm == "full body"
            || norm == "cowboy shot"
        {
            continue;
        }

        let is_concrete = is_concrete_scene_object(&norm);
        let min_thresh = if is_concrete { 0.40 } else { 0.55 };
        if q.score < min_thresh {
            continue;
        }

        let prompt = to_prompt_tag(&q.name);
        let Some(t) = crate::parser::parse_prompt(&prompt).into_iter().next() else { continue };
        let s = crate::scene::slot(&t);
        let is_valid = is_concrete
            || matches!(s, Slot::Hair | Slot::Eyes | Slot::Traits | Slot::Outfit | Slot::Expression | Slot::Other)
            || (s == Slot::Pose && (norm.starts_with("holding") || norm.contains("knee") || norm.contains("leg") || norm.contains("hand") || norm.contains("adjusting")));

        if !is_valid {
            continue;
        }

        match out.iter_mut().find(|b| b.name == q.name) {
            Some(b) => b.score = b.score.max(q.score),
            None => out.push(q.clone()),
        }
    }
    out
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
            t("thick_eyebrows", 0.7), t("glasses", 0.5), t("hair_ornament", 0.8),
        ];
        let m = merge_detail_crop(&base, &detail);
        let score = |n: &str| m.iter().find(|x| x.name == n).map(|x| x.score);
        assert_eq!(score("thick_eyebrows"), Some(0.7));
        assert_eq!(score("glasses"), Some(0.3)); // below 0.55 in the crop: unchanged
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

