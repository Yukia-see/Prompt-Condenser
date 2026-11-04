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
            if total > 0 {
                let pct = done * 100 / total;
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

struct Loaded {
    /// The ONNX Runtime environment must outlive the session.
    _env: ort::environment::Environment,
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
fn preprocess(path: &Path) -> Result<Vec<f32>> {
    let img = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?.to_rgba8();
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
    if !is_installed() {
        return Err(anyhow!("The Danbooru tagger isn't downloaded yet."));
    }
    let input = preprocess(path)?;
    let mut guard = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let dir = tagger_dir();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let env = ort::init().with_name("prompt-condenser").build()?;
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(dir.join("model.onnx"))
            .context("Couldn't load the tagger model")?;
        *guard = Some(Loaded { _env: env, session, tags: load_tags(&dir.join("selected_tags.csv"))? });
    }
    let loaded = guard.as_mut().expect("just loaded");

    let tensor = Tensor::from_array(([1usize, INPUT_SIZE as usize, INPUT_SIZE as usize, 3], input.into_boxed_slice()))?;
    let outputs = loaded.session.run(ort::inputs![tensor])?;
    let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(loaded
        .tags
        .iter()
        .zip(scores.iter())
        .map(|((name, kind), &score)| ScoredTag { name: name.clone(), kind: *kind, score })
        .collect())
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

