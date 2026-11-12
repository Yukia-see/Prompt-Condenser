//! Evaluation Harness for Phase 1 Baseline & Multi-Arm Ablation
//!
//! Evaluates:
//! - Arm 1: Full Frame Baseline (Standard 1024px)
//! - Arm 2: Full Frame Native/High-Res (2048px/dynamic uncropped)
//! - Arm 3: WD14 Crops Only (WD14 4-quadrant + 3-zone, full-frame VLM)
//! - Arm 4: Full Decomposition (WD14 crops + VLM crops)
//!
//! Measures micro-attribute and global framing precision/recall,
//! counts crop-boundary artifacts, and generates per-image delta tables.

use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::ai_engine::{
    ask_vision_encoder, extract_visual_description_from_path, shrink_for_vision,
    shrink_for_vision_custom,
};
use crate::danbooru::{dedup_color_parents, extract_tags_from_prose, sort_canonical_buckets};
use crate::profiles::AiMode;
use crate::prompt_condenser::MODE_A_PASS1_KEYWORDS;
use crate::tagger;
use crate::twoimg;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvalArm {
    Arm1Baseline,
    Arm2NativeHighRes,
    Arm3Wd14CropsOnly,
    Arm4FullDecomp,
}

impl EvalArm {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Arm1Baseline => "Arm 1: Full Frame Baseline (1024px)",
            Self::Arm2NativeHighRes => "Arm 2: Full Frame High-Res (2048px)",
            Self::Arm3Wd14CropsOnly => "Arm 3: WD14 Crops Only",
            Self::Arm4FullDecomp => "Arm 4: Full Decomposition (WD14+VLM Crops)",
        }
    }

    pub fn short_code(&self) -> &'static str {
        match self {
            Self::Arm1Baseline => "arm1",
            Self::Arm2NativeHighRes => "arm2",
            Self::Arm3Wd14CropsOnly => "arm3",
            Self::Arm4FullDecomp => "arm4",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "arm1" | "baseline" | "arm1_baseline" => Some(Self::Arm1Baseline),
            "arm2" | "highres" | "native" | "arm2_highres" => Some(Self::Arm2NativeHighRes),
            "arm3" | "wd14_crops" | "arm3_wd14_crops" => Some(Self::Arm3Wd14CropsOnly),
            "arm4" | "full" | "full_decomp" | "arm4_full_decomp" => Some(Self::Arm4FullDecomp),
            _ => None,
        }
    }
}

#[allow(dead_code)]
pub struct TagTaxonomy {
    pub ratings: HashSet<String>,
    pub characters: HashSet<String>,
    pub general: HashSet<String>,
    pub crop_artifacts: HashSet<String>,
}

impl TagTaxonomy {
    pub fn load_or_fallback(csv_path: Option<&Path>) -> Self {
        let mut ratings = HashSet::new();
        let mut characters = HashSet::new();
        let mut general = HashSet::new();

        let default_path = crate::tagger::tagger_dir().join("selected_tags.csv");
        let path = csv_path.unwrap_or(&default_path);

        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines().skip(1) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 3 {
                    let name = parts[1].trim().replace('_', " ").to_lowercase();
                    let cat = parts[2].trim();
                    match cat {
                        "9" => { ratings.insert(name); }
                        "4" => { characters.insert(name); }
                        "0" => { general.insert(name); }
                        _ => {}
                    }
                }
            }
        }

        // Standard crop-boundary artifact blacklist
        let crop_artifacts: HashSet<String> = [
            "head out of frame", "cropped legs", "out of frame", "portrait",
            "close-up", "solo focus", "breast focus", "upper body", "lower body",
            "feet out of frame", "face out of frame",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            ratings,
            characters,
            general,
            crop_artifacts,
        }
    }

    pub fn is_crop_artifact(&self, tag: &str) -> bool {
        self.crop_artifacts.contains(&tag.to_lowercase())
    }

    pub fn is_global(&self, tag: &str) -> bool {
        let t = tag.to_lowercase();
        // Framing, camera, lighting, environment/scene keywords
        MODE_A_PASS1_KEYWORDS.iter().any(|k| *k == t.as_str())
            || t.contains("angle")
            || t.contains("view")
            || t.contains("shot")
            || t.contains("background")
            || t.contains("lighting")
            || t.contains("sky")
            || t.contains("indoors")
            || t.contains("outdoors")
    }

    pub fn is_micro_attribute(&self, tag: &str) -> bool {
        let t = tag.to_lowercase();
        !self.is_crop_artifact(&t)
            && !self.is_global(&t)
            && !self.ratings.contains(&t)
            && !self.characters.contains(&t)
    }
}

#[allow(dead_code)]
pub struct ImageMetrics {
    pub image_name: String,
    pub is_tuning_set: bool,
    pub arm: EvalArm,
    pub elapsed_sec: f32,
    pub gt_total: usize,
    pub gt_micro: usize,
    pub gt_global: usize,
    pub pred_total: usize,
    pub pred_micro: usize,
    pub pred_global: usize,
    pub micro_prec: f32,
    pub micro_rec: f32,
    pub micro_f1: f32,
    pub global_prec: f32,
    pub global_rec: f32,
    pub global_f1: f32,
    pub crop_artifacts: usize,
    pub rating_leaks: usize,
    pub character_leaks: usize,
    pub raw_pred_tags: String,
}

pub fn run_arm_on_image(
    image_path: &Path,
    arm: EvalArm,
    taxonomy: &TagTaxonomy,
    gt_tags: &[String],
    is_tuning: bool,
) -> Result<ImageMetrics> {
    let t0 = Instant::now();
    let host = "http://localhost:11434";
    let vision_model = "huihui_ai/qwen2.5-vl-abliterated:3b";
    let img_dynamic = image::open(image_path).with_context(|| format!("Open {}", image_path.display()))?;

    let pred_tags_str = match arm {
        EvalArm::Arm1Baseline => {
            // WD14 full frame only
            let raw_scores = tagger::tag_dynamic(&img_dynamic)?;
            let wd14_prompt = tagger::build_prompt(&raw_scores, 0.22).prompt;

            // VLM full frame at 1024px
            let b64 = base64::engine::general_purpose::STANDARD.encode(shrink_for_vision(image_path)?);
            let legacy_ask = "Describe the subject, appearance, clothing, pose, framing, setting, light and style.";
            let desc = ask_vision_encoder(host, AiMode::Gpu, vision_model, b64, legacy_ask).unwrap_or_default();
            let prose_tags = extract_tags_from_prose(&desc);

            let mut merged = wd14_prompt;
            for pt in &prose_tags {
                if !merged.to_lowercase().contains(&pt.to_lowercase()) {
                    merged.push_str(", ");
                    merged.push_str(pt);
                }
            }
            dedup_color_parents(&merged)
        }
        EvalArm::Arm2NativeHighRes => {
            // WD14 full frame only
            let raw_scores = tagger::tag_dynamic(&img_dynamic)?;
            let wd14_prompt = tagger::build_prompt(&raw_scores, 0.22).prompt;

            // VLM full frame at 2048px (native dynamic resolution test)
            let b64 = base64::engine::general_purpose::STANDARD.encode(shrink_for_vision_custom(image_path, 2048)?);
            let legacy_ask = "Describe the subject, appearance, clothing, pose, framing, setting, light and style in fine detail.";
            let desc = ask_vision_encoder(host, AiMode::Gpu, vision_model, b64, legacy_ask).unwrap_or_default();
            let prose_tags = extract_tags_from_prose(&desc);

            let mut merged = wd14_prompt;
            for pt in &prose_tags {
                if crate::prompt_condenser::is_crop_artifact_tag(pt)
                    || crate::prompt_condenser::is_rating_tag(pt)
                {
                    continue;
                }
                if !merged.to_lowercase().contains(&pt.to_lowercase()) {
                    merged.push_str(", ");
                    merged.push_str(pt);
                }
            }
            dedup_color_parents(&merged)
        }
        EvalArm::Arm3Wd14CropsOnly => {
            // WD14 4-quadrant + 3-zone crops
            let noop = |_: String| {};
            let crop_scores = twoimg::tag_one_fast(image_path, &noop)?;
            let wd14_prompt = tagger::build_prompt(&crop_scores, 0.22).prompt;

            // VLM full frame at 1024px (no VLM crops)
            let b64 = base64::engine::general_purpose::STANDARD.encode(shrink_for_vision(image_path)?);
            let legacy_ask = "Describe the subject, appearance, clothing, pose, framing, setting, light and style.";
            let desc = ask_vision_encoder(host, AiMode::Gpu, vision_model, b64, legacy_ask).unwrap_or_default();
            let prose_tags = extract_tags_from_prose(&desc);

            let mut merged = wd14_prompt;
            for pt in &prose_tags {
                if crate::prompt_condenser::is_crop_artifact_tag(pt)
                    || crate::prompt_condenser::is_rating_tag(pt)
                {
                    continue;
                }
                if !merged.to_lowercase().contains(&pt.to_lowercase()) {
                    merged.push_str(", ");
                    merged.push_str(pt);
                }
            }
            dedup_color_parents(&merged)
        }
        EvalArm::Arm4FullDecomp => {
            // WD14 4-quadrant + 3-zone crops
            let noop = |_: String| {};
            let crop_scores = twoimg::tag_one_fast(image_path, &noop)?;
            let wd14_prompt = tagger::build_prompt(&crop_scores, 0.22).prompt;

            // VLM 4-quadrant + 3-zone crops
            let desc = extract_visual_description_from_path(image_path, host, AiMode::Gpu, vision_model)
                .unwrap_or_default();
            let prose_tags = extract_tags_from_prose(&desc);

            let mut merged = wd14_prompt;
            for pt in &prose_tags {
                if crate::prompt_condenser::is_crop_artifact_tag(pt)
                    || crate::prompt_condenser::is_rating_tag(pt)
                {
                    continue;
                }
                if !merged.to_lowercase().contains(&pt.to_lowercase()) {
                    merged.push_str(", ");
                    merged.push_str(pt);
                }
            }
            let deduped = dedup_color_parents(&merged);
            sort_canonical_buckets(&deduped)
        }
    };

    let elapsed = t0.elapsed().as_secs_f32();

    // Parse tags
    let pred_tags: Vec<String> = pred_tags_str
        .split(',')
        .map(|s| s.trim().to_lowercase().replace('_', " "))
        .filter(|s| !s.is_empty())
        .collect();

    let gt_set: HashSet<String> = gt_tags
        .iter()
        .map(|s| s.trim().to_lowercase().replace('_', " "))
        .filter(|s| !s.is_empty())
        .collect();

    // Partition Ground Truth
    let mut gt_micro = HashSet::new();
    let mut gt_global = HashSet::new();
    for t in &gt_set {
        if taxonomy.is_global(t) {
            gt_global.insert(t.clone());
        } else if taxonomy.is_micro_attribute(t) {
            gt_micro.insert(t.clone());
        }
    }

    // Partition Predictions & Count Leaks
    let mut pred_micro = HashSet::new();
    let mut pred_global = HashSet::new();
    let mut crop_artifacts = 0;
    let mut rating_leaks = 0;
    let mut character_leaks = 0;

    for t in &pred_tags {
        if taxonomy.is_crop_artifact(t) {
            crop_artifacts += 1;
        }
        if taxonomy.ratings.contains(t) {
            rating_leaks += 1;
        }
        if taxonomy.characters.contains(t) {
            character_leaks += 1;
        }

        if taxonomy.is_global(t) {
            pred_global.insert(t.clone());
        } else if taxonomy.is_micro_attribute(t) {
            pred_micro.insert(t.clone());
        }
    }

    // Calculate metrics
    let calc_pr = |pred: &HashSet<String>, gt: &HashSet<String>| -> (f32, f32, f32) {
        if pred.is_empty() && gt.is_empty() {
            return (1.0, 1.0, 1.0);
        }
        let tp = pred.intersection(gt).count() as f32;
        let p = if pred.is_empty() { 0.0 } else { tp / (pred.len() as f32) };
        let r = if gt.is_empty() { 1.0 } else { tp / (gt.len() as f32) };
        let f1 = if (p + r) > 0.0 { 2.0 * (p * r) / (p + r) } else { 0.0 };
        (p, r, f1)
    };

    let (m_prec, m_rec, m_f1) = calc_pr(&pred_micro, &gt_micro);
    let (g_prec, g_rec, g_f1) = calc_pr(&pred_global, &gt_global);

    let image_name = image_path.file_name().unwrap_or_default().to_string_lossy().to_string();

    Ok(ImageMetrics {
        image_name,
        is_tuning_set: is_tuning,
        arm,
        elapsed_sec: elapsed,
        gt_total: gt_set.len(),
        gt_micro: gt_micro.len(),
        gt_global: gt_global.len(),
        pred_total: pred_tags.len(),
        pred_micro: pred_micro.len(),
        pred_global: pred_global.len(),
        micro_prec: m_prec,
        micro_rec: m_rec,
        micro_f1: m_f1,
        global_prec: g_prec,
        global_rec: g_rec,
        global_f1: g_f1,
        crop_artifacts,
        rating_leaks,
        character_leaks,
        raw_pred_tags: pred_tags_str,
    })
}

/// Runs the evaluation CLI
pub fn run_eval_cli(args: &[String]) -> Result<()> {
    let mut dir_path = PathBuf::from("tests/eval_dataset");
    let mut arms_to_run = vec![
        EvalArm::Arm1Baseline,
        EvalArm::Arm2NativeHighRes,
        EvalArm::Arm3Wd14CropsOnly,
        EvalArm::Arm4FullDecomp,
    ];
    let mut split_ratio = 0.67f32;
    let mut output_report = PathBuf::from("baseline_eval_report.md");
    let mut csv_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" | "-d" => {
                if i + 1 < args.len() {
                    dir_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--arm" | "-a" => {
                if i + 1 < args.len() {
                    let a_str = &args[i + 1];
                    if a_str.to_lowercase() != "all" {
                        if let Some(arm) = EvalArm::from_str(a_str) {
                            arms_to_run = vec![arm];
                        } else {
                            eprintln!("Unknown arm '{}', using all arms.", a_str);
                        }
                    }
                    i += 1;
                }
            }
            "--split" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() {
                        split_ratio = v;
                    }
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_report = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--csv" => {
                if i + 1 < args.len() {
                    csv_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Prompt Condenser Evaluation Harness (Phase 1 Baseline & Ablation)");
                println!("Usage: prompt-condenser eval [options]");
                println!("Options:");
                println!("  --dir, -d <PATH>       Directory of evaluation images + .txt files (default: tests/eval_dataset)");
                println!("  --arm, -a <ARM>        Arm to run (all, arm1, arm2, arm3, arm4) (default: all)");
                println!("  --split <FLOAT>        Tuning/Holdout split ratio (default: 0.67)");
                println!("  --output, -o <PATH>    Markdown report output path (default: baseline_eval_report.md)");
                println!("  --csv <PATH>           Path to selected_tags.csv");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    if !dir_path.exists() {
        return Err(anyhow!("Evaluation dataset directory not found at {}", dir_path.display()));
    }

    let taxonomy = TagTaxonomy::load_or_fallback(csv_path.as_deref());

    // Discover test images (.jpg, .png) that have a corresponding .txt file
    let mut image_files = Vec::new();
    for entry in std::fs::read_dir(&dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();
            if ext == "jpg" || ext == "png" || ext == "jpeg" || ext == "webp" {
                let txt_path = path.with_extension("txt");
                if txt_path.exists() {
                    image_files.push((path, txt_path));
                }
            }
        }
    }

    image_files.sort_by(|a, b| a.0.cmp(&b.0));

    if image_files.is_empty() {
        return Err(anyhow!(
            "No evaluation image + .txt pairs found in {}. Ensure each image has a matching .txt ground truth file.",
            dir_path.display()
        ));
    }

    let total_images = image_files.len();
    let tuning_count = ((total_images as f32) * split_ratio).round() as usize;

    println!("\n================================================================================");
    println!(">>> PROMPT CONDENSER PHASE 1 EVALUATION HARNESS");
    println!("================================================================================");
    println!("Dataset: {} ({} paired images)", dir_path.display(), total_images);
    println!("Split: {} Tuning (67%) / {} Holdout (33%)", tuning_count, total_images.saturating_sub(tuning_count));
    println!("Arms: {:?}", arms_to_run.iter().map(|a| a.short_code()).collect::<Vec<_>>());
    println!("================================================================================\n");

    let mut all_results: HashMap<EvalArm, Vec<ImageMetrics>> = HashMap::new();

    for arm in &arms_to_run {
        println!(">>> Evaluating {}...", arm.name());
        let mut arm_metrics = Vec::new();

        for (idx, (img_path, txt_path)) in image_files.iter().enumerate() {
            let is_tuning = idx < tuning_count;
            let gt_raw = std::fs::read_to_string(txt_path)?;
            let gt_tags: Vec<String> = gt_raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            print!("  [{}/{}] {} ({})... ", idx + 1, total_images, img_path.file_name().unwrap_or_default().to_string_lossy(), if is_tuning { "Tuning" } else { "Holdout" });
            std::io::Write::flush(&mut std::io::stdout())?;

            match run_arm_on_image(img_path, *arm, &taxonomy, &gt_tags, is_tuning) {
                Ok(m) => {
                    println!(
                        "Done in {:.1}s | Micro P: {:.1}%, R: {:.1}% | Artifacts: {}",
                        m.elapsed_sec,
                        m.micro_prec * 100.0,
                        m.micro_rec * 100.0,
                        m.crop_artifacts
                    );
                    arm_metrics.push(m);
                }
                Err(e) => {
                    println!("FAILED: {e}");
                }
            }
        }
        all_results.insert(*arm, arm_metrics);
        println!();
    }

    // Generate Markdown Report
    let mut report = String::new();
    report.push_str("# Prompt Condenser: Phase 1 Baseline & Multi-Arm Ablation Report\n\n");
    report.push_str(&format!("**Date:** Nov 8, 2026\n**Images Evaluated:** {}\n**Dataset:** `{}`\n**Split:** {} Tuning / {} Holdout Confirmation\n\n", total_images, dir_path.display(), tuning_count, total_images.saturating_sub(tuning_count)));

    report.push_str("## 1. Executive Summary & Aggregate Averages\n\n");
    report.push_str("| Arm | Dataset Split | Micro Prec | Micro Rec | Micro F1 | Global Prec | Global Rec | Crop Artifacts | Avg Time |\n");
    report.push_str("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n");

    for arm in &arms_to_run {
        if let Some(metrics) = all_results.get(arm) {
            let (tuning, holdout): (Vec<&ImageMetrics>, Vec<&ImageMetrics>) = metrics.iter().partition(|m| m.is_tuning_set);

            let summarize = |subset: &[&ImageMetrics]| -> (f32, f32, f32, f32, f32, usize, f32) {
                if subset.is_empty() { return (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0); }
                let n = subset.len() as f32;
                let mp = subset.iter().map(|m| m.micro_prec).sum::<f32>() / n;
                let mr = subset.iter().map(|m| m.micro_rec).sum::<f32>() / n;
                let mf1 = subset.iter().map(|m| m.micro_f1).sum::<f32>() / n;
                let gp = subset.iter().map(|m| m.global_prec).sum::<f32>() / n;
                let gr = subset.iter().map(|m| m.global_rec).sum::<f32>() / n;
                let ca = subset.iter().map(|m| m.crop_artifacts).sum::<usize>();
                let time = subset.iter().map(|m| m.elapsed_sec).sum::<f32>() / n;
                (mp, mr, mf1, gp, gr, ca, time)
            };

            let (tmp, tmr, tmf1, tgp, tgr, tca, ttime) = summarize(&tuning);
            report.push_str(&format!(
                "| **{}** | Tuning (2/3) | {:.1}% | {:.1}% | {:.2} | {:.1}% | {:.1}% | {} | {:.1}s |\n",
                arm.short_code(), tmp * 100.0, tmr * 100.0, tmf1, tgp * 100.0, tgr * 100.0, tca, ttime
            ));

            if !holdout.is_empty() {
                let (hmp, hmr, hmf1, hgp, hgr, hca, htime) = summarize(&holdout);
                report.push_str(&format!(
                    "| **{}** | Holdout (1/3) | {:.1}% | {:.1}% | {:.2} | {:.1}% | {:.1}% | {} | {:.1}s |\n",
                    arm.short_code(), hmp * 100.0, hmr * 100.0, hmf1, hgp * 100.0, hgr * 100.0, hca, htime
                ));
            }
        }
    }

    // Per-image delta table against Arm 1 baseline
    if let Some(baseline_metrics) = all_results.get(&EvalArm::Arm1Baseline) {
        let baseline_map: HashMap<&str, &ImageMetrics> = baseline_metrics.iter().map(|m| (m.image_name.as_str(), m)).collect();

        report.push_str("\n## 2. Per-Image Delta Tables (vs. Arm 1 Baseline)\n\n");

        for arm in &arms_to_run {
            if *arm == EvalArm::Arm1Baseline { continue; }
            if let Some(metrics) = all_results.get(arm) {
                report.push_str(&format!("### {}\n\n", arm.name()));
                report.push_str("| Image | Set | Baseline Micro P/R | Arm Micro P/R | $\\Delta$ Micro P | $\\Delta$ Micro R | Artifacts (Base $\\rightarrow$ Arm) |\n");
                report.push_str("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n");

                for m in metrics {
                    if let Some(b) = baseline_map.get(m.image_name.as_str()) {
                        let delta_p = (m.micro_prec - b.micro_prec) * 100.0;
                        let delta_r = (m.micro_rec - b.micro_rec) * 100.0;
                        report.push_str(&format!(
                            "| `{}` | {} | {:.1}% / {:.1}% | {:.1}% / {:.1}% | {:+.1}% | {:+.1}% | {} $\\rightarrow$ {} |\n",
                            m.image_name,
                            if m.is_tuning_set { "Tuning" } else { "Holdout" },
                            b.micro_prec * 100.0, b.micro_rec * 100.0,
                            m.micro_prec * 100.0, m.micro_rec * 100.0,
                            delta_p, delta_r,
                            b.crop_artifacts, m.crop_artifacts
                        ));
                    }
                }
                report.push_str("\n");
            }
        }
    }

    std::fs::write(&output_report, &report)?;
    println!("Evaluation report written to: {}", output_report.display());

    Ok(())
}

#[cfg(test)]
mod category_tests {
    use crate::scene::Slot;
    use std::collections::{BTreeMap, HashSet};
    use std::path::{Path, PathBuf};

    /// Ignored on both sides, as in tools/score_image_prompts.py.
    const IGNORED: &[&str] = &[
        "masterpiece", "best quality", "high quality", "amazing quality", "very aesthetic", "absurdres", "highres",
        "solo", "score 9", "score 8 up", "score 7 up",
    ];

    /// The scorer's `norm`: lowercase, spaces for underscores, `(tag:1.2)` → tag, `\(` → `(`.
    fn norm(tag: &str) -> String {
        let mut t = tag.trim().to_lowercase().replace('_', " ");
        if t.starts_with('(') && t.ends_with(')') && !t.ends_with("\\)") {
            let inner = &t[1..t.len() - 1];
            let inner = match inner.rsplit_once(':') {
                Some((a, w)) if w.parse::<f32>().is_ok() => a,
                _ => inner,
            };
            t = inner.to_string();
        }
        t.replace("\\(", "(").replace("\\)", ")").trim().to_string()
    }

    fn tags(text: &str) -> HashSet<String> {
        text.split([',', '\n']).map(norm).filter(|t| !t.is_empty() && !IGNORED.contains(&t.as_str())).collect()
    }

    fn category(tag: &str) -> &'static str {
        let Some(t) = crate::parser::parse_prompt(tag).into_iter().next() else { return "other" };
        match crate::scene::slot(&t) {
            Slot::Setting | Slot::Light => "setting/bg",
            Slot::Hair => "hair",
            Slot::Eyes => "eyes",
            Slot::Outfit => "outfit",
            _ => "other",
        }
    }

    /// Fast Tagger mode (one pass + `clean_tagger_rules`) scored per tag category at several thresholds,
    /// on the top-level images of tests/eval_local that have answer files (never ai_generated/ or excluded/):
    /// `cargo test live_fast_categories -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_fast_categories() {
        let dir = std::env::var("EVAL_DIR").unwrap_or_else(|_| "tests/eval_local".into());
        let mut images: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| p.extension().is_some_and(|e| matches!(e.to_string_lossy().to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp")))
            .filter(|p| p.with_extension("txt").exists())
            .collect();
        images.sort();
        let thresholds = [0.25f32, 0.30, 0.35, 0.42];
        let cats = ["setting/bg", "hair", "eyes", "outfit", "other"];
        // per threshold: category -> (output, right, missed); and per-image (P, R, F1)
        let mut table: Vec<BTreeMap<&str, (usize, usize, usize)>> = vec![BTreeMap::new(); thresholds.len()];
        let mut per_image: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); thresholds.len()];
        let mut missed_list: Vec<BTreeMap<&str, Vec<String>>> = vec![BTreeMap::new(); thresholds.len()];
        for path in &images {
            let scored = crate::twoimg::tag_one_passes(path, &|_| (), false, false).unwrap_or_default();
            let key = tags(&std::fs::read_to_string(path.with_extension("txt")).unwrap());
            let accept_file = path.with_file_name(format!("{}.acceptable.txt", path.file_stem().unwrap().to_string_lossy()));
            let acceptable = std::fs::read_to_string(&accept_file).map(|t| tags(&t)).unwrap_or_default();
            for (ti, &th) in thresholds.iter().enumerate() {
                let raw = crate::tagger::build_prompt(&scored, th).prompt;
                let got = tags(&crate::ai_engine::clean_tagger_rules(&raw, Path::new(path)));
                let implied: HashSet<String> = key
                    .iter()
                    .filter(|k| !got.contains(*k) && !k.contains(' ') && got.iter().any(|g| g.ends_with(&format!(" {k}"))))
                    .cloned()
                    .collect();
                let hit: HashSet<&String> = key.iter().filter(|k| got.contains(*k) || implied.contains(*k)).collect();
                let right: HashSet<&String> = got.iter().filter(|g| key.contains(*g) || acceptable.contains(*g)).collect();
                for g in &got {
                    let e = table[ti].entry(category(g)).or_default();
                    e.0 += 1;
                    if right.contains(g) {
                        e.1 += 1;
                    }
                }
                for k in key.iter().filter(|k| !hit.contains(k)) {
                    table[ti].entry(category(k)).or_default().2 += 1;
                    missed_list[ti].entry(category(k)).or_default().push(k.clone());
                }
                let rec = hit.len() as f64 / key.len().max(1) as f64;
                let prec = right.len() as f64 / got.len().max(1) as f64;
                let f1 = 2.0 * rec * prec / (rec + prec).max(1e-9);
                per_image[ti].push((prec, rec, f1));
                println!("{:<28} th {th:.2}: P {:3.0}% R {:3.0}% F1 {:3.0}% ({} tags)", path.file_name().unwrap().to_string_lossy(), prec * 100.0, rec * 100.0, f1 * 100.0, got.len());
            }
        }
        println!("\n{} images: {}", images.len(), images.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>().join(", "));
        for (ti, th) in thresholds.iter().enumerate() {
            let n = per_image[ti].len().max(1) as f64;
            let avg = |f: fn(&(f64, f64, f64)) -> f64| per_image[ti].iter().map(f).sum::<f64>() / n * 100.0;
            println!("\nthreshold {th:.2}: avg P {:.1}%  R {:.1}%  F1 {:.1}%", avg(|x| x.0), avg(|x| x.1), avg(|x| x.2));
            println!("  {:<11} {:>6} {:>6} {:>6} {:>7}", "category", "output", "right", "prec", "missed");
            for c in cats {
                let (o, r, m) = table[ti].get(c).copied().unwrap_or_default();
                println!("  {c:<11} {o:>6} {r:>6} {:>5.0}% {m:>7}", r as f64 * 100.0 / o.max(1) as f64);
            }
            for c in cats {
                if let Some(m) = missed_list[ti].get(c) {
                    println!("  missed {c}: {}", m.join(", "));
                }
            }
        }
    }

    /// Per-category thresholds (`category` of each general tag) instead of one, on the same images and
    /// scoring as `live_fast_categories`: prints the top combinations of a small grid, the plain 0.30
    /// baseline and the best combination's per-image change.
    /// `cargo test live_fast_category_thresholds -- --ignored --nocapture`
    /// (`EVAL_DIR=tests/eval_local/fresh CATEGORY_COMBOS="0.3,0.3,0.3,0.3,0.3;…"` scores given combinations per image)
    #[test]
    #[ignore]
    fn live_fast_category_thresholds() {
        use crate::tagger::{ScoredTag, TagKind};
        let dir = std::env::var("EVAL_DIR").unwrap_or_else(|_| "tests/eval_local".into());
        let mut images: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| p.extension().is_some_and(|e| matches!(e.to_string_lossy().to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp")))
            .filter(|p| p.with_extension("txt").exists())
            .collect();
        images.sort();
        struct Img {
            path: PathBuf,
            scored: Vec<ScoredTag>,
            cats: Vec<&'static str>,
            key: HashSet<String>,
            acceptable: HashSet<String>,
        }
        let data: Vec<Img> = images
            .iter()
            .map(|path| {
                let scored = crate::twoimg::tag_one_passes(path, &|_| (), false, false).unwrap_or_default();
                let cats = scored.iter().map(|t| category(&t.name.replace('_', " "))).collect();
                let accept_file = path.with_file_name(format!("{}.acceptable.txt", path.file_stem().unwrap().to_string_lossy()));
                Img {
                    path: path.clone(),
                    scored,
                    cats,
                    key: tags(&std::fs::read_to_string(path.with_extension("txt")).unwrap()),
                    acceptable: std::fs::read_to_string(&accept_file).map(|t| tags(&t)).unwrap_or_default(),
                }
            })
            .collect();
        // [setting/bg, hair, eyes, outfit, other]
        let score = |img: &Img, th: [f32; 5]| -> (f64, f64, f64) {
            let idx = |c: &str| ["setting/bg", "hair", "eyes", "outfit", "other"].iter().position(|x| *x == c).unwrap();
            let kept: Vec<ScoredTag> = img
                .scored
                .iter()
                .zip(&img.cats)
                .filter(|(t, c)| t.kind != TagKind::General || t.score >= th[idx(c)])
                .map(|(t, _)| t.clone())
                .collect();
            let min = th.iter().cloned().fold(1.0f32, f32::min);
            let raw = crate::tagger::build_prompt(&kept, min).prompt;
            let got = tags(&crate::ai_engine::clean_tagger_rules(&raw, &img.path));
            let key = &img.key;
            let implied = |k: &String| !k.contains(' ') && got.iter().any(|g| g.ends_with(&format!(" {k}")));
            let hit = key.iter().filter(|k| got.contains(*k) || implied(k)).count();
            let right = got.iter().filter(|g| key.contains(*g) || img.acceptable.contains(*g)).count();
            let rec = hit as f64 / key.len().max(1) as f64;
            let prec = right as f64 / got.len().max(1) as f64;
            (prec, rec, 2.0 * rec * prec / (rec + prec).max(1e-9))
        };
        let run = |th: [f32; 5]| -> Vec<(f64, f64, f64)> { data.iter().map(|img| score(img, th)).collect() };
        let avg = |v: &[(f64, f64, f64)]| {
            let n = v.len().max(1) as f64;
            (v.iter().map(|x| x.0).sum::<f64>() / n * 100.0, v.iter().map(|x| x.1).sum::<f64>() / n * 100.0, v.iter().map(|x| x.2).sum::<f64>() / n * 100.0)
        };
        // CATEGORY_COMBOS="bg,hair,eyes,outfit,other;…": just these, per image (for held-out images)
        if let Ok(list) = std::env::var("CATEGORY_COMBOS") {
            for combo in list.split(';') {
                let v: Vec<f32> = combo.split(',').map(|x| x.trim().parse().unwrap()).collect();
                let th = [v[0], v[1], v[2], v[3], v[4]];
                let per = run(th);
                let (p, r, f) = avg(&per);
                println!("\n{th:?}: avg P {p:.1}%  R {r:.1}%  F1 {f:.1}%");
                for (img, (p, r, f)) in data.iter().zip(&per) {
                    println!("  {:<28} P {:5.1}%  R {:5.1}%  F1 {:5.1}%", img.path.file_name().unwrap().to_string_lossy(), p * 100.0, r * 100.0, f * 100.0);
                }
            }
            return;
        }
        let base = run([0.30; 5]);
        let (bp, br, bf) = avg(&base);
        let mut results: Vec<([f32; 5], (f64, f64, f64), Vec<(f64, f64, f64)>)> = Vec::new();
        for bg in [0.30f32, 0.35, 0.42] {
            for eyes in [0.30f32, 0.35, 0.42] {
                for hair in [0.30f32, 0.35] {
                    for outfit in [0.25f32, 0.30] {
                        for other in [0.25f32, 0.30] {
                            let th = [bg, hair, eyes, outfit, other];
                            let per = run(th);
                            results.push((th, avg(&per), per));
                        }
                    }
                }
            }
        }
        results.sort_by(|a, b| b.1 .2.partial_cmp(&a.1 .2).unwrap());
        println!("\n{} images, {} combinations", data.len(), results.len());
        println!("baseline 0.30 everywhere: P {bp:.1}%  R {br:.1}%  F1 {bf:.1}%");
        println!("top 5 (setting/bg, hair, eyes, outfit, other):");
        for (th, (p, r, f), _) in results.iter().take(5) {
            println!("  bg {:.2} hair {:.2} eyes {:.2} outfit {:.2} other {:.2}: P {p:.1}%  R {r:.1}%  F1 {f:.1}% ({:+.1})", th[0], th[1], th[2], th[3], th[4], f - bf);
        }
        let (th, _, per) = &results[0];
        println!("best {th:?} per image, F1 vs 0.30:");
        for ((img, b), n) in data.iter().zip(&base).zip(per) {
            println!("  {:<28} {:5.1}% -> {:5.1}% ({:+.1})", img.path.file_name().unwrap().to_string_lossy(), b.2 * 100.0, n.2 * 100.0, (n.2 - b.2) * 100.0);
        }
    }
}
