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

/// Frees the loaded detector (see `tagger::release`).
pub fn release() {
    *MODEL.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

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

/// Crops a bounding box, optionally masking out overlapping regions from other detected figures
/// to prevent visual bleed of neighboring outfits/hair/limbs into the crop.
/// If overlap is extreme (>65%), returns a warning recommending full-frame fallback.
pub fn crop_with_overlap_mask(
    img: &image::DynamicImage,
    b: &PersonBox,
    others: &[PersonBox],
) -> (image::DynamicImage, Option<String>) {
    let mut warning = None;
    let mut cropped = crop(img, b);

    let (w, h) = (img.width() as f32, img.height() as f32);
    let (mx, my) = ((b.x1 - b.x0) * 0.06, (b.y1 - b.y0) * 0.04);
    let x0 = (b.x0 - mx).max(0.0).min((w - 1.0).max(0.0));
    let y0 = (b.y0 - my).max(0.0).min((h - 1.0).max(0.0));
    let cw = cropped.width();
    let ch = cropped.height();

    for o in others {
        if b.x0 == o.x0 && b.y0 == o.y0 && b.x1 == o.x1 && b.y1 == o.y1 {
            continue;
        }
        let ov = b.overlap(o);
        if ov > 0.65 {
            warning = Some(format!(
                "Extreme figure overlap ({:.1}%); neighboring figure dominates the crop region. Fallback to full-frame recommended.",
                ov * 100.0
            ));
        } else if ov > 0.10 {
            let ix0 = ((o.x0 - x0).max(0.0).min(cw as f32)) as u32;
            let iy0 = ((o.y0 - y0).max(0.0).min(ch as f32)) as u32;
            let ix1 = ((o.x1 - x0).max(0.0).min(cw as f32)) as u32;
            let iy1 = ((o.y1 - y0).max(0.0).min(ch as f32)) as u32;

            if ix1 > ix0 && iy1 > iy0 {
                let slice_w = ix1 - ix0;
                let slice_h = iy1 - iy0;
                let is_peripheral = ix0 == 0 || ix1 == cw || iy0 == 0 || iy1 == ch;
                if is_peripheral && (slice_w * slice_h) < (cw * ch / 2) {
                    let mut rgba = cropped.to_rgba8();
                    let neutral_pixel = image::Rgba([128, 128, 128, 255]);
                    for py in iy0..iy1 {
                        for px in ix0..ix1 {
                            rgba.put_pixel(px, py, neutral_pixel);
                        }
                    }
                    cropped = image::DynamicImage::ImageRgba8(rgba);
                }
            }
        }
    }

    (cropped, warning)
}

/// The top 55% of a character's box (face, hair, eyes, the top of the outfit), so the
/// tagger sees those details bigger.
pub fn upper_crop(img: &image::DynamicImage, b: &PersonBox) -> image::DynamicImage {
    let top = PersonBox { y1: b.y0 + (b.y1 - b.y0) * 0.55, ..*b };
    crop(img, &top)
}

/// The middle 40% of a character's box (30% to 70%: torso, waist, hands, hips, tail root, belts).
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn lower_crop(img: &image::DynamicImage, b: &PersonBox) -> image::DynamicImage {
    let h = b.y1 - b.y0;
    let low = PersonBox {
        y0: b.y0 + h * 0.55,
        ..*b
    };
    crop(img, &low)
}

/// Returns the valid overlapping anatomical zones for a character bounding box using
/// pure zone geometry (Upper: 0-55%, Middle: 30-70%, Lower: 55-100%).
/// Skips the Lower zone for half-body, bust, or bottom-edge-clipped figures.
pub fn three_zone_crops(img: &image::DynamicImage, b: &PersonBox) -> Vec<(&'static str, image::DynamicImage)> {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let zone_boxes = crate::prompt_condenser::compute_zone_boxes(w, h, b.x0, b.y0, b.x1, b.y1);
    zone_boxes
        .into_iter()
        .map(|zb| {
            let label = match zb.name {
                "Upper" => "Upper (Head, Hair & Face)",
                "Middle" => "Middle (Torso, Neckline, Arms & Held Items)",
                "Lower" => "Lower (Hips, Legs & Footwear)",
                _ => zb.name,
            };
            let z_box = PersonBox {
                x0: zb.x0,
                y0: zb.y0,
                x1: zb.x1,
                y1: zb.y1,
                score: b.score,
            };
            (label, crop(img, &z_box))
        })
        .collect()
}

/// Slices the full image into 4 overlapping quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
/// with a 10% center overlap (0.0..0.55 and 0.45..1.0 on X and Y) so background objects (plants, lamps,
/// furniture, props) crossing the center axes are never cut in half.
pub fn quadrant_crops(img: &image::DynamicImage) -> [(&'static str, image::DynamicImage); 4] {
    let (w, h) = (img.width().max(2), img.height().max(2));
    let x_mid_end = ((w as f32 * 0.55).round() as u32).clamp(1, w);
    let x_mid_start = ((w as f32 * 0.45).round() as u32).min(w - 1);
    let y_mid_end = ((h as f32 * 0.55).round() as u32).clamp(1, h);
    let y_mid_start = ((h as f32 * 0.45).round() as u32).min(h - 1);

    let rw = (w - x_mid_start).max(1);
    let bh = (h - y_mid_start).max(1);

    [
        ("Top-Left Quadrant", img.crop_imm(0, 0, x_mid_end, y_mid_end)),
        ("Top-Right Quadrant", img.crop_imm(x_mid_start, 0, rw, y_mid_end)),
        ("Bottom-Left Quadrant", img.crop_imm(0, y_mid_start, x_mid_end, bh)),
        ("Bottom-Right Quadrant", img.crop_imm(x_mid_start, y_mid_start, rw, bh)),
    ]
}

/// Returns a bounding box centered on a wrist position with a proportional margin
/// (e.g. 18% of figure height, min 32px) clamped to image boundaries.
#[allow(dead_code)] // not wired to the UI yet
pub fn wrist_hand_box(
    wrist_x: f32,
    wrist_y: f32,
    figure_height: f32,
    img_w: f32,
    img_h: f32,
) -> PersonBox {
    let radius = (figure_height * 0.18).max(32.0);
    PersonBox {
        x0: (wrist_x - radius).max(0.0),
        y0: (wrist_y - radius).max(0.0),
        x1: (wrist_x + radius).min(img_w),
        y1: (wrist_y + radius).min(img_h),
        score: 1.0,
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
#[allow(dead_code)] // not wired to the UI yet
pub struct OpenPosePerson {
    pub person_id: Vec<i32>,
    pub pose_keypoints_2d: Vec<f32>,
    pub face_keypoints_2d: Vec<f32>,
    pub hand_left_keypoints_2d: Vec<f32>,
    pub hand_right_keypoints_2d: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[allow(dead_code)] // not wired to the UI yet
pub struct OpenPoseExport {
    pub version: f32,
    pub people: Vec<OpenPosePerson>,
    pub canvas_width: u32,
    pub canvas_height: u32,
}

#[allow(dead_code)] // not wired to the UI yet
pub fn export_openpose_json(
    people: &[OpenPosePerson],
    canvas_width: u32,
    canvas_height: u32,
) -> String {
    let export = OpenPoseExport {
        version: 1.3,
        people: people.to_vec(),
        canvas_width,
        canvas_height,
    };
    serde_json::to_string_pretty(&export).unwrap_or_else(|_| "{}".to_string())
}

/// Synthesizes estimated 18-point OpenPose keypoints from a detected PersonBox
/// using standard anime anatomical ratios when full skeleton detector is offline.
#[allow(dead_code)] // not wired to the UI yet
pub fn person_box_to_estimated_keypoints(b: &PersonBox) -> Vec<f32> {
    let w = b.x1 - b.x0;
    let h = b.y1 - b.y0;
    let cx = b.x0 + w * 0.5;

    let pts = [
        (cx, b.y0 + h * 0.12),           // 0: Nose
        (cx, b.y0 + h * 0.20),           // 1: Neck
        (cx - w * 0.22, b.y0 + h * 0.22), // 2: RShoulder
        (cx - w * 0.28, b.y0 + h * 0.38), // 3: RElbow
        (cx - w * 0.25, b.y0 + h * 0.52), // 4: RWrist
        (cx + w * 0.22, b.y0 + h * 0.22), // 5: LShoulder
        (cx + w * 0.28, b.y0 + h * 0.38), // 6: LElbow
        (cx + w * 0.25, b.y0 + h * 0.52), // 7: LWrist
        (cx, b.y0 + h * 0.55),           // 8: MidHip
        (cx - w * 0.16, b.y0 + h * 0.56), // 9: RHip
        (cx - w * 0.18, b.y0 + h * 0.76), // 10: RKnee
        (cx - w * 0.20, b.y0 + h * 0.96), // 11: RAnkle
        (cx + w * 0.16, b.y0 + h * 0.56), // 12: LHip
        (cx + w * 0.18, b.y0 + h * 0.76), // 13: LKnee
        (cx + w * 0.20, b.y0 + h * 0.96), // 14: LAnkle
        (cx - w * 0.08, b.y0 + h * 0.09), // 15: REye
        (cx + w * 0.08, b.y0 + h * 0.09), // 16: LEye
        (cx - w * 0.18, b.y0 + h * 0.12), // 17: REar
    ];

    let mut out = Vec::with_capacity(pts.len() * 3);
    for (x, y) in pts {
        out.push(x);
        out.push(y);
        out.push(b.score);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrant_and_three_zone_crops_have_valid_dimensions() {
        let img = image::DynamicImage::new_rgb8(1000, 800);
        let quads = quadrant_crops(&img);
        assert_eq!(quads.len(), 4);
        for (_, q) in &quads {
            assert_eq!(q.width(), 550);
            assert_eq!(q.height(), 440);
        }
        let b = PersonBox { x0: 100.0, y0: 50.0, x1: 600.0, y1: 750.0, score: 0.95 };
        let zones = three_zone_crops(&img, &b);
        assert_eq!(zones.len(), 3);
        for (_, z) in &zones {
            assert!(z.width() > 400);
            assert!(z.height() > 200);
        }
    }

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

    #[test]
    fn crop_overlap_masking_and_warning() {
        let img = image::DynamicImage::new_rgb8(1000, 1000);
        let b1 = PersonBox { x0: 100.0, y0: 100.0, x1: 500.0, y1: 900.0, score: 0.9 };
        // Moderate peripheral overlap (b2 overlaps b1 by ~20%)
        let b2 = PersonBox { x0: 450.0, y0: 100.0, x1: 850.0, y1: 900.0, score: 0.85 };
        let (crop1, warn1) = crop_with_overlap_mask(&img, &b1, &[b1, b2]);
        assert!(warn1.is_none());
        assert!(crop1.width() > 300);

        // Extreme overlap (>65%)
        let b3 = PersonBox { x0: 120.0, y0: 110.0, x1: 490.0, y1: 890.0, score: 0.88 };
        let (_crop_ext, warn_ext) = crop_with_overlap_mask(&img, &b1, &[b1, b3]);
        assert!(warn_ext.is_some());
        assert!(warn_ext.unwrap().contains("Extreme figure overlap"));
    }

    #[test]
    fn wrist_hand_box_and_openpose_export() {
        let b = PersonBox { x0: 100.0, y0: 100.0, x1: 500.0, y1: 900.0, score: 0.95 };
        let hand = wrist_hand_box(200.0, 500.0, 800.0, 1000.0, 1000.0);
        // Radius = 800 * 0.18 = 144
        assert!((hand.x0 - 56.0).abs() < 1e-3);
        assert!((hand.x1 - 344.0).abs() < 1e-3);
        assert!((hand.y0 - 356.0).abs() < 1e-3);
        assert!((hand.y1 - 644.0).abs() < 1e-3);

        let kpts = person_box_to_estimated_keypoints(&b);
        assert_eq!(kpts.len(), 18 * 3);

        let person = OpenPosePerson {
            person_id: vec![-1],
            pose_keypoints_2d: kpts,
            face_keypoints_2d: vec![],
            hand_left_keypoints_2d: vec![],
            hand_right_keypoints_2d: vec![],
        };
        let json_str = export_openpose_json(&[person], 1000, 1000);
        assert!(json_str.contains("\"version\": 1.3"));
        assert!(json_str.contains("\"canvas_width\": 1000"));
        assert!(json_str.contains("\"pose_keypoints_2d\""));
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
