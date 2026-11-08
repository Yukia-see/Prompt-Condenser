//! An image with several characters -> Character 1..N + Scene.
//!
//! 1. The person detector finds the characters (left to right = character 1, 2, …).
//! 2. The Danbooru tagger reads each crop and the whole image: a tag clearly strongest
//!    in one crop belongs to that character; scene tags (camera, setting, light, style)
//!    and interaction tags are shared.
//! 3. In AI mode, a vision model gets the image (each character in a colored box),
//!    every crop and those tag lists, and fixes what the crops get wrong (overlapping
//!    characters bleed into each other's crop).

use crate::detector::{self, PersonBox};
use crate::parser::{parse_prompt, PromptToken};
use crate::scene::{slot, Slot};
use crate::tagger::{self, ScoredTag};
use crate::two::{count_tag, is_interaction};
use anyhow::{anyhow, Context, Result};
use std::path::Path;

/// The box colors drawn around characters 1..4 for the vision model, and their names.
pub const BOX_COLORS: [([u8; 3], &str); 4] =
    [([235, 30, 30], "RED"), ([30, 110, 245], "BLUE"), ([40, 190, 60], "GREEN"), ([245, 200, 20], "YELLOW")];

/// What the detector and tagger found (kept so the threshold slider and the
/// Tagger / AI switch don't have to run the models again).
#[derive(Clone)]
pub struct TagSplit {
    pub image: image::DynamicImage,
    pub full: Vec<ScoredTag>,
    /// Each character, left to right: their crop and its tags (`None` when fewer
    /// characters than wanted were found).
    pub chars: Option<Vec<(image::DynamicImage, Vec<ScoredTag>)>>,
    /// How much the boxes overlap (the worst pair; 0 = apart, 1 = one inside another).
    pub overlap: f32,
    /// The characters' boxes, left to right.
    pub boxes: Option<Vec<PersonBox>>,
    pub people: usize,
    /// How many characters were asked for.
    pub wanted: usize,
}

/// The boxes' contents: one prompt per character, plus the shared scene.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SplitPrompt {
    pub chars: Vec<String>,
    pub scene: String,
}

/// `close_ups`: also tag each character's face and outfit zoomed in (Tagger mode; in
/// AI mode the vision model sees the crops itself, so it's skipped to save time).
pub fn read(path: &Path, n: usize, close_ups: bool) -> Result<TagSplit> {
    let image = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    let boxes = detector::detect_n(&image, n)?;
    let picked = detector::main_n(&boxes, n);
    let (full, chars, overlap) = match &picked {
        Some(bs) => {
            let mut overlap: f32 = 0.0;
            for i in 0..bs.len() {
                for j in i + 1..bs.len() {
                    overlap = overlap.max(bs[i].overlap(&bs[j]));
                }
            }
            // zoom in on each face and outfit, unless they overlap so much that the
            // zoomed crop would mostly show another character
            let zoom = close_ups && overlap <= 0.35;
            // one view at a time: batching several in one run measured ~5x slower on CPU
            let full = tagger::tag_dynamic(&image)?;
            let mut chars = Vec::new();
            for b in bs {
                let crop = detector::crop(&image, b);
                let mut tags = tagger::tag_dynamic(&crop)?;
                if zoom {
                    tags = tagger::merge_detail_crop(&tags, &tagger::tag_dynamic(&detector::upper_crop(&image, b))?);
                }
                chars.push((crop, tags));
            }
            (full, Some(chars), overlap)
        }
        None => (tagger::tag_dynamic(&image)?, None, 0.0),
    };
    Ok(TagSplit { image, full, chars, overlap, boxes: picked, people: boxes.len(), wanted: n })
}

/// One character's image (the 1 Character tab, or a Char 1 / Char 2 reference): the
/// full image's tags plus a zoomed look at the main character's face and outfit.
/// Downloads the person detector the first time; without it, just the full image.
pub fn tag_one(path: &Path, progress: &(dyn Fn(String) + Sync)) -> Result<Vec<ScoredTag>> {
    let image = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    let full = tagger::tag_dynamic(&image)?;
    if !detector::is_installed() {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let downloaded = std::thread::scope(|s| {
            s.spawn(|| rx.into_iter().for_each(progress));
            let r = detector::download(&tx);
            drop(tx);
            r
        });
        if downloaded.is_err() {
            return Ok(full);
        }
    }
    let Ok(boxes) = detector::detect(&image) else { return Ok(full) };
    let Some(main) = boxes.iter().copied().max_by(|a, b| a.area().total_cmp(&b.area())) else { return Ok(full) };
    progress("Looking closer at the face and outfit…".into());
    Ok(tagger::merge_detail_crop(&full, &tagger::tag_dynamic(&detector::upper_crop(&image, &main))?))
}

/// Same as `tag_one`, but runs the detector before tagging, so the image is opened once.
pub fn tag_one_fast(path: &Path, progress: &(dyn Fn(String) + Sync)) -> Result<Vec<ScoredTag>> {
    if !detector::is_installed() {
        return tag_one(path, progress);
    }
    let image = image::open(path).with_context(|| format!("Couldn't open {}", path.display()))?;
    let mut boxes = detector::detect(&image).unwrap_or_default();
    if boxes.is_empty() {
        // Fallback to full-frame bounding box when anime person detector finds no box (e.g. realistic/cosplay photos)
        boxes.push(PersonBox {
            x0: 0.0,
            y0: 0.0,
            x1: image.width() as f32,
            y1: image.height() as f32,
            score: 1.0,
        });
    }
    let full = tagger::tag_dynamic(&image)?;
    if boxes.len() == 1 {
        let main = &boxes[0];
        let up = detector::upper_crop(&image, main);
        let is_tall = (main.y1 - main.y0) / (image.height() as f32).max(1.0) >= 0.35;
        if is_tall {
            let mid = detector::middle_crop(&image, main);
            let low = detector::lower_crop(&image, main);
            let mut tags = [&up, &mid, &low]
                .iter()
                .map(|v| tagger::tag_dynamic(v))
                .collect::<Result<Vec<_>>>()?
                .into_iter();
            let up_detail = tags.next().unwrap_or_default();
            let mid_detail = tags.next().unwrap_or_default();
            let low_detail = tags.next().unwrap_or_default();
            let merged = tagger::merge_detail_crop(&full, &up_detail);
            let merged = tagger::merge_detail_crop(&merged, &mid_detail);
            return Ok(tagger::merge_detail_crop(&merged, &low_detail));
        } else {
            let detail = tagger::tag_dynamic(&up)?;
            return Ok(tagger::merge_detail_crop(&full, &detail));
        }
    }
    // Multiple characters: zoom in on each main character so individual clothing/hair/breast tags are detected
    let count = boxes.len().min(4);
    let chars = detector::main_n(&boxes, count).unwrap_or_else(|| boxes.clone());
    let mut merged = full;
    for b in &chars {
        let crop = detector::crop(&image, b);
        let tags = tagger::tag_dynamic(&crop)?;
        merged = tagger::merge_detail_crop(&merged, &tags);
    }
    Ok(merged)
}

fn tokens(scores: &[ScoredTag], threshold: f32) -> Vec<PromptToken> {
    parse_prompt(&tagger::build_prompt(scores, threshold).prompt)
}

fn is_scene(t: &PromptToken) -> bool {
    let n = t.normalized.as_str();
    if crate::scene::is_character_like(n) || n.contains("body") {
        return false;
    }
    is_interaction(n) || matches!(slot(t), Slot::Setting | Slot::Light | Slot::Style)
}

/// A tag's score in a crop, by prompt-form name ("black hair").
fn score_map(scores: &[ScoredTag]) -> std::collections::HashMap<String, f32> {
    // content ratings ("general", "sensitive"...) aren't tags
    scores.iter().filter(|t| t.kind != tagger::TagKind::Rating).map(|t| (crate::danbooru::key(&t.name), t.score)).collect()
}

/// Which characters a tag belongs to, from its score in each crop: clearly higher in
/// one crop (1.6x the next) -> that character; similar in several -> all of those
/// (they share it, e.g. long hair). Overlapping crops show part of other characters,
/// but their features score lower there.
fn owners(scores: &[f32], threshold: f32) -> Vec<usize> {
    const CLEARLY: f32 = 1.6;
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let Some(&best) = order.first() else { return Vec::new() };
    if scores[best] < threshold {
        return Vec::new();
    }
    let mut out: Vec<usize> = order
        .into_iter()
        .filter(|&i| scores[i] >= threshold && scores[best] <= scores[i] * CLEARLY)
        .collect();
    out.sort();
    out
}

/// Tags the tagger is sure belong to one character only (strong in its crop, faint in
/// every other). The AI may not move these.
pub fn locked(ts: &TagSplit) -> Vec<Vec<String>> {
    let Some(chars) = &ts.chars else { return Vec::new() };
    let maps: Vec<_> = chars.iter().map(|(_, t)| score_map(t)).collect();
    let mut out = vec![Vec::new(); maps.len()];
    for (i, own) in maps.iter().enumerate() {
        for (name, &s) in own.iter() {
            let others_faint = maps.iter().enumerate().all(|(j, m)| j == i || m.get(name).copied().unwrap_or(0.0) < 0.2);
            let tok = parse_prompt(name);
            let Some(t) = tok.first() else { continue };
            if s >= 0.6 && others_faint && !is_scene(t) && count_tag(&t.normalized).is_none() && t.normalized != "solo" {
                out[i].push(t.normalized.clone());
            }
        }
        out[i].sort();
    }
    out
}

/// The split from the tags alone (Tagger mode, and the hints for AI mode).
pub fn rule_split(ts: &TagSplit, threshold: f32) -> SplitPrompt {
    let full = tokens(&ts.full, threshold);
    let mut scene: Vec<String> = Vec::new();
    let add = |v: &mut Vec<String>, t: &PromptToken| {
        if !v.iter().any(|x| crate::danbooru::key(x) == t.normalized) {
            v.push(t.text.clone());
        }
    };
    let Some(chars) = &ts.chars else {
        // fewer people found than wanted: everything about people goes to character 1
        let mut c1 = Vec::new();
        for t in &full {
            if is_scene(t) {
                add(&mut scene, t);
            } else if t.normalized != "solo" && count_tag(&t.normalized).map(|(n, _)| n == 1).unwrap_or(true) {
                add(&mut c1, t);
            }
        }
        let mut out = vec![c1.join(", ")];
        out.resize(ts.wanted.max(1), String::new());
        return SplitPrompt { chars: out, scene: scene.join(", ") };
    };
    let maps: Vec<_> = chars.iter().map(|(_, t)| score_map(t)).collect();
    let mut sides: Vec<Vec<String>> = vec![Vec::new(); chars.len()];
    for (i, (_, scores)) in chars.iter().enumerate() {
        for t in tokens(scores, threshold) {
            let n = t.normalized.as_str();
            if n == "solo" {
                continue;
            }
            match count_tag(n) {
                Some((1, _)) => add(&mut sides[i], &t),
                Some(_) => {}
                None if is_scene(&t) => add(&mut scene, &t),
                None => {
                    let s: Vec<f32> = maps.iter().map(|m| m.get(n).copied().unwrap_or(0.0)).collect();
                    let own = owners(&s, threshold);
                    if own.contains(&i) {
                        // shared or its own: the other owners add it from their own crop
                        add(&mut sides[i], &t);
                    } else {
                        // bled into this crop from another character
                        for o in own {
                            add(&mut sides[o], &t);
                        }
                    }
                }
            }
        }
    }
    // the whole image adds the scene, the interaction and the group count
    for t in &full {
        if is_scene(t) {
            add(&mut scene, t);
        }
    }
    SplitPrompt { chars: sides.into_iter().map(|v| v.join(", ")).collect(), scene: scene.join(", ") }
}

/// The full picture with each character in a colored box (red, blue, green, yellow),
/// so the vision model can't mix them up ("left" / "right" alone isn't enough).
pub fn marked(ts: &TagSplit) -> image::DynamicImage {
    let Some(boxes) = &ts.boxes else { return ts.image.clone() };
    let mut img = ts.image.to_rgb8();
    let (w, h) = (img.width() as i64, img.height() as i64);
    let t = (w.min(h) / 120).max(3);
    for (b, (rgb, _)) in boxes.iter().zip(BOX_COLORS) {
        let color = image::Rgb(rgb);
        let (x0, y0, x1, y1) = (b.x0 as i64, b.y0 as i64, b.x1 as i64, b.y1 as i64);
        for y in y0.max(0)..y1.min(h) {
            for x in x0.max(0)..x1.min(w) {
                if x - x0 < t || x1 - x <= t || y - y0 < t || y1 - y <= t {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
    image::DynamicImage::ImageRgb8(img)
}

/// Makes the AI's split agree with the detector: if its characters match the crops
/// better in another order (the model mixed them up), reorder them; and put locked
/// tags back where they belong.
pub fn settle(ai: SplitPrompt, ts: &TagSplit, threshold: f32) -> SplitPrompt {
    let Some(chars) = &ts.chars else { return ai };
    let n = chars.len();
    let hints = rule_split(ts, threshold);
    let set = |s: &str| parse_prompt(s).into_iter().map(|t| t.normalized).collect::<Vec<_>>();
    let hint_sets: Vec<Vec<String>> = hints.chars.iter().map(|c| set(c)).collect();
    let mut ai_chars = ai.chars.clone();
    ai_chars.resize(n, String::new());
    let ai_sets: Vec<Vec<String>> = ai_chars.iter().map(|c| set(c)).collect();
    let hits = |a: &[String], b: &[String]| a.iter().filter(|x| b.contains(x)).count() as i64;
    // perm[i] = which AI list goes to character i
    let score = |perm: &[usize]| perm.iter().enumerate().map(|(i, &p)| hits(&ai_sets[p], &hint_sets[i])).sum::<i64>();
    let identity: Vec<usize> = (0..n).collect();
    let mut best = identity.clone();
    for perm in crate::two::permutations(n) {
        if score(&perm) > score(&best) {
            best = perm;
        }
    }
    // only reorder when it's clearly better (not on a 1-tag difference)
    if score(&best) <= score(&identity) + 2 {
        best = identity;
    }
    let mut lists: Vec<Vec<String>> = best
        .iter()
        .map(|&p| ai_chars[p].split(", ").filter(|x| !x.is_empty()).map(str::to_string).collect())
        .collect();
    for (i, locks) in locked(ts).iter().enumerate() {
        for l in locks {
            for (j, list) in lists.iter_mut().enumerate() {
                if j == i {
                    if !list.iter().any(|t| crate::danbooru::key(t) == *l) {
                        list.push(l.clone());
                    }
                } else {
                    list.retain(|t| crate::danbooru::key(t) != *l);
                }
            }
        }
    }
    SplitPrompt { chars: lists.into_iter().map(|v| v.join(", ")).collect(), scene: ai.scene }
}

/// Scales an image down (longest side `max`) and encodes it as base64 JPEG for Ollama.
pub fn encode(img: &image::DynamicImage, max: u32) -> Result<String> {
    use base64::Engine as _;
    let small = if img.width().max(img.height()) > max { img.resize(max, max, image::imageops::FilterType::Triangle) } else { img.clone() };
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90).encode_image(&small.to_rgb8())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(out))
}

/// Cleans one of the AI's lists: no quality tags, no "solo", and the group count
/// ("2girls") only in the scene.
pub fn clean_list(text: &str, is_scene_list: bool) -> String {
    let mut seen: Vec<String> = Vec::new();
    // labels the model sometimes writes ("art style: comic", "lighting: dim") and
    // references to "character 2" aren't tags
    let text: String = text
        .split(',')
        .map(|t| {
            let t = t.trim();
            let t = match t.split_once(':') {
                Some((label, rest)) if !t.starts_with('(') && !t.starts_with('<') && label.chars().all(|c| c.is_alphabetic() || c == ' ') => {
                    let rest = rest.trim();
                    if label.trim() == "lighting" && !rest.contains("light") { format!("{rest} lighting") } else { rest.to_string() }
                }
                _ => t.to_string(),
            };
            // notes like "long hair (shared)" or "cat tail (incorrect tag from hint)" aren't
            // part of the tag; a tag the model itself calls wrong is dropped
            let t = match (t.rfind(" ("), t.ends_with(')')) {
                (Some(i), true) if !t.contains("\\(") && !t[i..].contains(':') => {
                    let note = t[i + 2..t.len() - 1].to_lowercase();
                    if ["incorrect", "wrong", "not ", "unsure", "uncertain"].iter().any(|w| note.contains(w)) {
                        String::new()
                    } else {
                        t[..i].trim().to_string()
                    }
                }
                _ => t.to_string(),
            };
            const RATINGS: &[&str] = &["general", "sensitive", "questionable", "explicit"];
            if RATINGS.contains(&t.as_str()) {
                return String::new();
            }
            const CATEGORY: &[&str] = &["art style", "style", "lighting", "background", "setting", "pose", "expression", "outfit", "clothing", "hair", "eyes"];
            if CATEGORY.contains(&t.as_str()) {
                return String::new();
            }
            if (1..=4).any(|k| t.contains(&format!("character {k}"))) {
                if t.starts_with("looking at") { "looking at another".to_string() } else { String::new() }
            } else {
                t
            }
        })
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    parse_prompt(&text)
        .into_iter()
        .filter(|t| {
            let l = t.raw.to_lowercase();
            !t.normalized.ends_with(" flag") && !t.normalized.ends_with("_flag") && !t.normalized.contains("flag") && !l.contains("flag")
        })
        .filter(|t| t.category != crate::parser::TokenCategory::QualityMeta && t.normalized != "solo")
        .filter(|t| match count_tag(&t.normalized) {
            Some((1, _)) => !is_scene_list,
            Some(_) => is_scene_list,
            None => true,
        })
        .filter(|t| {
            if is_scene_list {
                if count_tag(&t.normalized).is_some() {
                    return true;
                }
                if !crate::sceneref::is_true_scene_tag(t) {
                    return false;
                }
            }
            // a duplicate, or a second size for the same body part: the first one wins
            let new = !seen.iter().any(|s| *s == t.normalized || crate::analyzer::size_conflict(s, &t.normalized));
            seen.push(t.normalized.clone());
            new
        })
        .map(|t| t.raw.trim().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn not_found_note(ts: &TagSplit) -> Option<String> {
    ts.chars.is_none().then(|| match ts.people {
        0 => "No characters found".to_string(),
        1 => "Only one character found".to_string(),
        n => format!("Only {n} characters found"),
    })
}

/// Fails early with a readable message when a model is missing.
pub fn check_models() -> Result<()> {
    if !tagger::is_installed() {
        return Err(anyhow!("Download the Danbooru tagger first (the button under From Image)."));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagger::TagKind;

    fn tags(list: &[(&str, f32)]) -> Vec<ScoredTag> {
        list.iter().map(|(n, s)| ScoredTag { name: n.to_string(), kind: TagKind::General, score: *s }).collect()
    }

    #[test]
    fn overlap_bleed_goes_to_the_stronger_crop() {
        assert_eq!(owners(&[0.9, 0.4], 0.35), vec![0]);
        assert_eq!(owners(&[0.5, 0.9], 0.35), vec![1]);
        assert_eq!(owners(&[0.8, 0.7], 0.35), vec![0, 1]);
        assert_eq!(owners(&[0.2, 0.3], 0.35), Vec::<usize>::new());
        assert_eq!(owners(&[0.8, 0.1, 0.75], 0.35), vec![0, 2]);
    }

    fn split3(blank: &image::DynamicImage, lists: [&[(&str, f32)]; 3]) -> TagSplit {
        TagSplit {
            image: blank.clone(),
            full: Vec::new(),
            chars: Some(lists.iter().map(|l| (blank.clone(), tags(l))).collect()),
            overlap: 0.0,
            boxes: None,
            people: 3,
            wanted: 3,
        }
    }

    #[test]
    fn settle_reorders_and_restores_locks() {
        let blank = image::DynamicImage::new_rgb8(1, 1);
        let ts = split3(&blank, [
            &[("1girl", 0.9), ("red_hair", 0.9), ("dress", 0.8), ("hat", 0.7)],
            &[("1girl", 0.9), ("blue_hair", 0.9), ("armor", 0.8), ("sword", 0.7)],
            &[("1boy", 0.9), ("green_hair", 0.9), ("cape", 0.8), ("staff", 0.7)],
        ]);
        // the model mixed up the order and moved the hat
        let ai = SplitPrompt {
            chars: vec![
                "1girl, blue hair, armor, sword, hat".into(),
                "1boy, green hair, cape, staff".into(),
                "1girl, red hair, dress".into(),
            ],
            scene: "park".into(),
        };
        let s = settle(ai, &ts, 0.35);
        assert_eq!(s.chars, vec!["1girl, red hair, dress, hat", "1girl, blue hair, armor, sword", "1boy, green hair, cape, staff"]);
    }

    /// Where the time goes: `TIMING_TEST=a.png cargo test live_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_timing() {
        let p = std::env::var("TIMING_TEST").unwrap();
        let img = image::open(&p).unwrap();
        let t = std::time::Instant::now();
        tagger::tag_dynamic(&img).unwrap();
        println!("tagger, first run (loads the model): {:.1}s", t.elapsed().as_secs_f32());
        for round in 0..2 {
            let t = std::time::Instant::now();
            tagger::tag_dynamic(&img).unwrap();
            let tag = t.elapsed().as_secs_f32();
            let t = std::time::Instant::now();
            let boxes = detector::detect_n(&img, 2).unwrap();
            let det = t.elapsed().as_secs_f32();
            let t = std::time::Instant::now();
            let _ = read(Path::new(&p), 2, true).unwrap();
            let t2 = std::time::Instant::now();
            let _ = read(Path::new(&p), 2, false).unwrap();
            println!("   without close-ups: {:.1}s", t2.elapsed().as_secs_f32());
            println!("round {round}: one tagger pass {tag:.1}s, detector {det:.2}s, whole 2-character read {:.1}s ({} people)", t.elapsed().as_secs_f32(), boxes.len());
        }
    }

    /// `TWOIMG_TEST=a.png cargo test live_two_image -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_two_image() {
        for p in std::env::var("TWOIMG_TEST").unwrap().split(':') {
            let t = std::time::Instant::now();
            let n: usize = std::env::var("TWOIMG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            let ts = read(Path::new(p), n, true).unwrap();
            let rules = rule_split(&ts, 0.35);
            let ts2 = ts.clone();
            println!("\n== {p}: {} people, overlap {:.2}, tags {:.1}s", ts.people, ts.overlap, t.elapsed().as_secs_f32());
            for (i, c) in rules.chars.iter().enumerate() { println!("TAGGER c{}   : {c}", i + 1); }
            println!("TAGGER scene: {}", rules.scene);
            let t = std::time::Instant::now();
            let model = std::env::var("IMAGE_MODEL").unwrap_or_else(|_| "prompt-condenser-gemma4-e4b:latest".into());
            let (ai, who) = crate::ai_engine::split_image(&crate::ai_engine::TwoImageRequest {
                host: "localhost:11434".into(), model,
                vision_encoder: Some("moondream:1.8b".into()),
                mode: crate::profiles::AiMode::Gpu, split: ts.clone(), hints: rules,
                second_pass_vision: false,
                scene_body_shape: false,
            }).unwrap();
            let ai = settle(ai, &ts2, 0.35);
            println!("AI ({:.0}s) who: {who}", t.elapsed().as_secs_f32());
            for (i, c) in ai.chars.iter().enumerate() { println!("AI c{}   : {c}", i + 1); }
            println!("AI scene: {}", ai.scene);
        }
    }

    #[test]
    fn splits_tags_by_crop() {
        let blank = image::DynamicImage::new_rgb8(1, 1);
        let ts = TagSplit {
            image: blank.clone(),
            full: tags(&[("2girls", 0.9), ("classroom", 0.8), ("holding_hands", 0.7), ("black_hair", 0.9)]),
            chars: Some(vec![
                (blank.clone(), tags(&[("1girl", 0.9), ("solo", 0.8), ("black_hair", 0.9), ("v", 0.7), ("classroom", 0.6)])),
                (blank, tags(&[("1girl", 0.9), ("black_hair", 0.9), ("sweater", 0.8), ("crossed_legs", 0.6)])),
            ]),
            overlap: 0.0,
            boxes: None,
            people: 2,
            wanted: 2,
        };
        let s = rule_split(&ts, 0.35);
        assert_eq!(s.chars, vec!["1girl, black hair, v", "1girl, black hair, sweater, crossed legs"]);
        assert_eq!(s.scene, "classroom, holding hands");
        assert_eq!(clean_list("masterpiece, 2girls, 1girl, solo, red hair, red hair", false), "1girl, red hair");
        assert_eq!(clean_list("2girls, 1girl, park", true), "2girls, park");
        assert_eq!(
            clean_list("art style: comic, lighting: dim, background: wall, looking at character 2, (smile:1.2)", false),
            "comic, dim lighting, wall, looking at another, (smile:1.2)"
        );
        assert_eq!(clean_list("cat ears (shared), art style, long hair", false), "cat ears, long hair");
        assert_eq!(clean_list("large breasts, medium breasts, long hair, very long hair", false), "large breasts, long hair, very long hair");
        assert_eq!(clean_list("cat tail (incorrect tag from hint), sensitive, 2b \\(nier:automata\\)", false), "2b \\(nier:automata\\)");
    }
}
