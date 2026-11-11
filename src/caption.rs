//! Batch captioning: `prompt-condenser --caption <folder> [--threshold 0.30] [--overwrite] [--recursive]`
//! writes a Danbooru caption (`<image>.txt`) next to every image, for LoRA training datasets. It uses the fast
//! Tagger mode: one tagger pass plus the cleanup rules and picture checks (`ai_engine::clean_tagger_prompt`),
//! which scored like the full AI pipeline on the test images (78-79%) at ~5 s per image. No Ollama needed.

use crate::{ai_engine, detector, tagger, twoimg};
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

/// Measured best for the fast mode on the 11 test images (0.30: 79.2%, 0.35: 78.5%, 0.25: 77.8%).
const DEFAULT_THRESHOLD: f32 = 0.30;
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp"];

#[derive(Debug, PartialEq)]
struct Options {
    folder: PathBuf,
    threshold: f32,
    overwrite: bool,
    recursive: bool,
}

const USAGE: &str = "usage: prompt-condenser --caption <folder> [--threshold 0.30] [--overwrite] [--recursive]";

fn parse_args(args: &[String]) -> Result<Options> {
    let mut folder = None;
    let mut threshold = DEFAULT_THRESHOLD;
    let (mut overwrite, mut recursive) = (false, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--overwrite" => overwrite = true,
            "--recursive" | "-r" => recursive = true,
            "--threshold" => {
                let v = it.next().ok_or_else(|| anyhow!("--threshold needs a value\n{USAGE}"))?;
                threshold = v.parse().map_err(|_| anyhow!("--threshold must be a number like 0.30\n{USAGE}"))?;
                if !(0.05..=0.95).contains(&threshold) {
                    return Err(anyhow!("--threshold must be between 0.05 and 0.95"));
                }
            }
            "-h" | "--help" => return Err(anyhow!("{USAGE}")),
            f if !f.starts_with('-') && folder.is_none() => folder = Some(PathBuf::from(f)),
            other => return Err(anyhow!("unknown option {other}\n{USAGE}")),
        }
    }
    let folder = folder.ok_or_else(|| anyhow!("{USAGE}"))?;
    Ok(Options { folder, threshold, overwrite, recursive })
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

fn collect_images(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if recursive {
                collect_images(&p, recursive, out)?;
            }
        } else if is_image(&p) {
            out.push(p);
        }
    }
    Ok(())
}

/// A prompt's tags written for a training caption: no prompt escaping (`kay (girls und panzer)`).
fn as_caption(prompt: &str) -> String {
    prompt.replace("\\(", "(").replace("\\)", ")")
}

/// Downloads the tagger (and the person detector for the picture checks) when missing, printing progress.
fn ensure_models() -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let printer = std::thread::spawn(move || {
        let mut last = String::new();
        for msg in rx {
            if msg != last {
                eprint!("\r{msg:<70}");
                last = msg;
            }
        }
        eprintln!();
    });
    let result = (|| -> Result<()> {
        if !tagger::is_installed() {
            tagger::download(tx.clone())?;
        }
        if !detector::is_installed() {
            // optional: without it the footwear / inset checks are skipped
            if let Err(e) = detector::download(&tx) {
                let _ = tx.send(format!("person detector not downloaded ({e:#}); picture checks skipped"));
            }
        }
        Ok(())
    })();
    drop(tx);
    let _ = printer.join();
    result
}

pub fn run_caption_cli(args: &[String]) -> Result<()> {
    let opts = parse_args(args)?;
    if !opts.folder.is_dir() {
        return Err(anyhow!("{} is not a folder", opts.folder.display()));
    }
    let mut images = Vec::new();
    collect_images(&opts.folder, opts.recursive, &mut images)?;
    if images.is_empty() {
        return Err(anyhow!("no images ({}) in {}", IMAGE_EXTENSIONS.join(", "), opts.folder.display()));
    }
    ensure_models()?;
    let (mut written, mut skipped, mut failed) = (0, 0, 0);
    let started = std::time::Instant::now();
    for (i, path) in images.iter().enumerate() {
        let txt = path.with_extension("txt");
        let name = path.strip_prefix(&opts.folder).unwrap_or(path).display();
        if txt.exists() && !opts.overwrite {
            println!("[{}/{}] {name}: caption exists, skipped (--overwrite replaces it)", i + 1, images.len());
            skipped += 1;
            continue;
        }
        let t = std::time::Instant::now();
        let caption = twoimg::tag_one_passes(path, &|_| (), false, false).map(|scores| {
            let raw = tagger::build_prompt(&scores, opts.threshold).prompt;
            as_caption(&ai_engine::clean_tagger_prompt(&raw, path))
        });
        match caption.and_then(|c| std::fs::write(&txt, format!("{c}\n")).map(|_| c).map_err(Into::into)) {
            Ok(c) => {
                written += 1;
                println!("[{}/{}] {name}: {} tags ({:.1} s)", i + 1, images.len(), c.split(", ").count(), t.elapsed().as_secs_f32());
            }
            Err(e) => {
                failed += 1;
                eprintln!("[{}/{}] {name}: failed: {e:#}", i + 1, images.len());
            }
        }
    }
    println!(
        "\n{written} captions written, {skipped} skipped, {failed} failed in {:.0} s (threshold {:.2})",
        started.elapsed().as_secs_f32(),
        opts.threshold
    );
    if failed > 0 && written == 0 {
        return Err(anyhow!("no caption could be written"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn caption_options_are_parsed() {
        let o = parse_args(&args("pics --threshold 0.4 --overwrite -r")).unwrap();
        assert_eq!(o, Options { folder: "pics".into(), threshold: 0.4, overwrite: true, recursive: true });
        let d = parse_args(&args("pics")).unwrap();
        assert_eq!((d.threshold, d.overwrite, d.recursive), (DEFAULT_THRESHOLD, false, false));
        assert!(parse_args(&args("")).is_err());
        assert!(parse_args(&args("pics --threshold x")).is_err());
        assert!(parse_args(&args("pics --threshold 1.5")).is_err());
        assert!(parse_args(&args("pics --bogus")).is_err());
    }

    #[test]
    fn captions_are_unescaped_and_only_images_are_picked() {
        assert_eq!(as_caption("1girl, kay \\(girls und panzer\\), smile"), "1girl, kay (girls und panzer), smile");
        assert!(is_image(Path::new("a/b.PNG")) && is_image(Path::new("c.webp")));
        assert!(!is_image(Path::new("c.txt")) && !is_image(Path::new("noext")));
    }
}
