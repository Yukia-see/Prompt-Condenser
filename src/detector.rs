//! Anime person detector: deepghs' person_detect v1.3 (YOLOv8s, ONNX, ~45 MB), run
//! locally with ONNX Runtime. Finds a box around each character so the 2 Characters
//! tab can tag each one separately.

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::Mutex;

const URL: &str =
    "https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.3_s/model.onnx";
/// The model card's best-F1 confidence threshold.
const THRESHOLD: f32 = 0.324;
const IOU: f32 = 0.7;
/// Longest side the image is scaled to (sides rounded up to a multiple of 32).
const MAX_SIZE: u32 = 640;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl PersonBox {
    pub fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
    fn iou(&self, o: &PersonBox) -> f32 {
        let w = (self.x1.min(o.x1) - self.x0.max(o.x0)).max(0.0);
        let h = (self.y1.min(o.y1) - self.y0.max(o.y0)).max(0.0);
        let inter = w * h;
        inter / (self.area() + o.area() - inter).max(1e-6)
    }
    /// Share of the smaller box covered by the other (how much two characters overlap).
    pub fn overlap(&self, o: &PersonBox) -> f32 {
        let w = (self.x1.min(o.x1) - self.x0.max(o.x0)).max(0.0);
        let h = (self.y1.min(o.y1) - self.y0.max(o.y0)).max(0.0);
        w * h / self.area().min(o.area()).max(1e-6)
    }
}

fn model_path() -> PathBuf {
    crate::tagger::tagger_dir().with_file_name("anime-person-detection-v1.3s").join("model.onnx")
}

pub fn is_installed() -> bool {
    model_path().is_file()
}

/// Downloads the model (~45 MB), reporting progress.
pub fn download(progress: &Sender<String>) -> Result<()> {
    let dest = model_path();
    if dest.is_file() {
        return Ok(());
    }
    std::fs::create_dir_all(dest.parent().expect("has a parent"))?;
    let mut resp = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()?
        .get(URL)
        .header("User-Agent", "prompt-condenser")
        .send()
        .context("Couldn't download the person detector")?;
    if !resp.status().is_success() {
        return Err(anyhow!("Downloading the person detector failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let part = dest.with_extension("part");
    let mut out = std::fs::File::create(&part)?;
    let mut buf = vec![0u8; 1 << 16];
    let (mut done, mut last) = (0u64, u64::MAX);
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if total > 0 && done * 100 / total != last {
            last = done * 100 / total;
            let _ = progress.send(format!("Downloading the person detector: {} / {} MB ({last}%)", done >> 20, total >> 20));
        }
    }
    out.flush()?;
    std::fs::rename(&part, &dest)?;
    Ok(())
}

struct Loaded {
    session: Session,
}

static MODEL: Mutex<Option<Loaded>> = Mutex::new(None);

/// Finds the characters in an image, most confident first (coordinates in pixels).
pub fn detect(img: &image::DynamicImage) -> Result<Vec<PersonBox>> {
    detect_at(img, THRESHOLD)
}

/// Like `detect`, but when fewer than `n` people pass the normal threshold, looks again
/// at a lower one for the others, keeping only boxes that are clearly different people
/// (tangled poses often score low).
pub fn detect_n(img: &image::DynamicImage, n: usize) -> Result<Vec<PersonBox>> {
    let boxes = detect_at(img, THRESHOLD)?;
    if boxes.len() >= n {
        return Ok(boxes);
    }
    let mut kept: Vec<PersonBox> = boxes.clone();
    for b in detect_at(img, 0.12)? {
        // a box that mostly covers one already found is the same person again
        let biggest = kept.iter().map(|k| k.area()).fold(b.area(), f32::max);
        if kept.iter().all(|k| b.overlap(k) < 0.75) && b.area() > biggest * 0.15 {
            kept.push(b);
        }
    }
    Ok(if kept.len() > boxes.len() { kept } else { boxes })
}

fn detect_at(img: &image::DynamicImage, threshold: f32) -> Result<Vec<PersonBox>> {
    if !is_installed() {
        return Err(anyhow!("The person detector isn't downloaded yet."));
    }
    let (ow, oh) = (img.width(), img.height());
    let r = (MAX_SIZE as f32 / ow.max(oh) as f32).min(1.0);
    let align = |v: f32| ((v / 32.0).ceil() * 32.0) as u32;
    let (nw, nh) = (align(ow as f32 * r).max(32), align(oh as f32 * r).max(32));
    let resized = img.resize_exact(nw, nh, image::imageops::FilterType::Triangle).to_rgb8();
    // RGB, 0-1, NCHW
    let plane = (nw * nh) as usize;
    let mut data = vec![0f32; plane * 3];
    for (i, p) in resized.pixels().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = p[c] as f32 / 255.0;
        }
    }

    let mut guard = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let session = Session::builder(crate::tagger::ort_env()?)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(model_path())
            .context("Couldn't load the person detector")?;
        *guard = Some(Loaded { session });
    }
    let loaded = guard.as_mut().expect("just loaded");
    let tensor = Tensor::from_array(([1usize, 3, nh as usize, nw as usize], data.into_boxed_slice()))?;
    let outputs = loaded.session.run(ort::inputs![tensor])?;
    let (shape, out) = outputs[0].try_extract_tensor::<f32>()?;
    // YOLOv8: [1, 4 + classes, candidates] with cx, cy, w, h in the resized image
    let (rows, n) = (shape[1] as usize, shape[2] as usize);
    if rows < 5 {
        return Err(anyhow!("Unexpected detector output {shape:?}"));
    }
    let (sx, sy) = (ow as f32 / nw as f32, oh as f32 / nh as f32);
    let mut boxes: Vec<PersonBox> = (0..n)
        .filter_map(|i| {
            let score = out[4 * n + i];
            (score >= threshold).then(|| {
                let (cx, cy, w, h) = (out[i], out[n + i], out[2 * n + i], out[3 * n + i]);
                PersonBox {
                    x0: ((cx - w / 2.0) * sx).max(0.0),
                    y0: ((cy - h / 2.0) * sy).max(0.0),
                    x1: ((cx + w / 2.0) * sx).min(ow as f32),
                    y1: ((cy + h / 2.0) * sy).min(oh as f32),
                    score,
                }
            })
        })
        .collect();
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<PersonBox> = Vec::new();
    for b in boxes {
        if kept.iter().all(|k| k.iou(&b) < IOU) {
            kept.push(b);
        }
    }
    Ok(kept)
}

/// The `n` main characters, left to right: the `n` biggest boxes (people in the
/// background are usually small). `None` unless at least `n` are found.
pub fn main_n(boxes: &[PersonBox], n: usize) -> Option<Vec<PersonBox>> {
    let mut by_size = boxes.to_vec();
    by_size.sort_by(|a, b| b.area().total_cmp(&a.area()));
    if by_size.len() < n {
        return None;
    }
    by_size.truncate(n);
    by_size.sort_by(|a, b| (a.x0 + a.x1).total_cmp(&(b.x0 + b.x1)));
    Some(by_size)
}

/// Crops a box with a little margin, so hair and hands at the edge stay in.
pub fn crop(img: &image::DynamicImage, b: &PersonBox) -> image::DynamicImage {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (mx, my) = ((b.x1 - b.x0) * 0.06, (b.y1 - b.y0) * 0.04);
    let x0 = (b.x0 - mx).max(0.0).min((w - 1.0).max(0.0));
    let y0 = (b.y0 - my).max(0.0).min((h - 1.0).max(0.0));
    let x1 = (b.x1 + mx).min(w).max(x0 + 1.0);
    let y1 = (b.y1 + my).min(h).max(y0 + 1.0);
    let cw = ((x1 - x0) as u32).max(1).min(img.width().saturating_sub(x0 as u32));
    let ch = ((y1 - y0) as u32).max(1).min(img.height().saturating_sub(y0 as u32));
    img.crop_imm(x0 as u32, y0 as u32, cw.max(1), ch.max(1))
}

/// The top 55% of a character's box (face, hair, eyes, the top of the outfit), so the
/// tagger sees those details bigger.
pub fn upper_crop(img: &image::DynamicImage, b: &PersonBox) -> image::DynamicImage {
    let top = PersonBox { y1: b.y0 + (b.y1 - b.y0) * 0.55, ..*b };
    crop(img, &top)
}

/// The middle 40% of a character's box (30% to 70%: torso, waist, hands, hips, tail root, belts).
pub fn middle_crop(img: &image::DynamicImage, b: &PersonBox) -> image::DynamicImage {
    let h = b.y1 - b.y0;
    let mid = PersonBox {
        y0: b.y0 + h * 0.30,
        y1: b.y0 + h * 0.70,
        ..*b
    };
    crop(img, &mid)
}

/// The lower 45% of a character's box (55% to 100%: legs, knees, calves, feet, shoes, claws, tail tip).
pub fn lower_crop(img: &image::DynamicImage, b: &PersonBox) -> image::DynamicImage {
    let h = b.y1 - b.y0;
    let low = PersonBox {
        y0: b.y0 + h * 0.55,
        ..*b
    };
    crop(img, &low)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_two_biggest_left_to_right() {
        let b = |x0: f32, x1: f32, s: f32| PersonBox { x0, y0: 0.0, x1, y1: 100.0, score: s };
        let boxes = [b(500.0, 700.0, 0.9), b(0.0, 300.0, 0.8), b(800.0, 820.0, 0.95)];
        let two = main_n(&boxes, 2).unwrap();
        assert_eq!((two[0].x0, two[1].x0), (0.0, 500.0));
        assert!(main_n(&boxes[..1], 2).is_none());
        let three = main_n(&boxes, 3).unwrap();
        assert_eq!(three.iter().map(|b| b.x0).collect::<Vec<_>>(), vec![0.0, 500.0, 800.0]);
    }

    /// `DETECT_TEST=a.png cargo test live_detect -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_detect() {
        for p in std::env::var("DETECT_TEST").unwrap().split(':') {
            let img = image::open(p).unwrap();
            let t = std::time::Instant::now();
            let boxes = detect_n(&img, 2).unwrap();
            println!("{p} ({}x{}): {:.2}s, {} found", img.width(), img.height(), t.elapsed().as_secs_f32(), boxes.len());
            for b in &boxes {
                println!("   {:.0},{:.0} -> {:.0},{:.0}  score {:.2}", b.x0, b.y0, b.x1, b.y1, b.score);
            }
        }
    }
}
