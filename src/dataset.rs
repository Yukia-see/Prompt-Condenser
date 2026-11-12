//! Dataset caption cleanup: `prompt-condenser --dataset <folder> [--recursive] [--no-backup]
//! [--keep-first N] [--sort] [--one-size]` tidies the existing LoRA training captions (`*.txt`).
//!
//! Captions must keep describing their image, so nothing is added or swapped: none of the prompt rules
//! (framing / body substitutions, views, quality tags, LoRA box) run. Only: separators and whitespace,
//! underscores → spaces (kaomoji kept), `\(` `\)` unescaped, weight syntax dropped (`(tag:1.2)` → `tag`),
//! exact duplicates dropped (case-insensitive), generic tags implied by a specific one dropped
//! (`analyzer::implies`), and with `--one-size` one size per body part (`analyzer::size_conflict`, the
//! first wins). The order stays (kohya's `keep_tokens` relies on the first tags); the first
//! `--keep-first` tags (default 1, the trigger word) are never dropped or moved, and `--sort` orders
//! the rest by `scene::slot`.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq)]
struct Options {
    folder: PathBuf,
    recursive: bool,
    backup: bool,
    keep_first: usize,
    sort: bool,
    one_size: bool,
}

const USAGE: &str =
    "usage: prompt-condenser --dataset <folder> [--recursive] [--no-backup] [--keep-first N] [--sort] [--one-size]";

fn parse_args(args: &[String]) -> Result<Options> {
    let mut o = Options { folder: PathBuf::new(), recursive: false, backup: true, keep_first: 1, sort: false, one_size: false };
    let mut folder = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--recursive" | "-r" => o.recursive = true,
            "--no-backup" => o.backup = false,
            "--sort" => o.sort = true,
            "--one-size" => o.one_size = true,
            "--keep-first" => {
                let v = it.next().ok_or_else(|| anyhow!("--keep-first needs a number\n{USAGE}"))?;
                o.keep_first = v.parse().map_err(|_| anyhow!("--keep-first must be a whole number like 1\n{USAGE}"))?;
            }
            "-h" | "--help" => return Err(anyhow!("{USAGE}")),
            f if !f.starts_with('-') && folder.is_none() => folder = Some(PathBuf::from(f)),
            other => return Err(anyhow!("unknown option {other}\n{USAGE}")),
        }
    }
    o.folder = folder.ok_or_else(|| anyhow!("{USAGE}"))?;
    Ok(o)
}

/// What cleaning one caption removed.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Stats {
    duplicates: usize,
    implied: usize,
    weights: usize,
    sizes: usize,
}

impl std::ops::AddAssign for Stats {
    fn add_assign(&mut self, o: Self) {
        self.duplicates += o.duplicates;
        self.implied += o.implied;
        self.weights += o.weights;
        self.sizes += o.sizes;
    }
}

/// `(tag:1.2)`, `((tag))`, `[tag]` → (`tag`, true). Escaped parentheses (`\(series\)`) and a name's
/// own parenthesis (`kay (girls und panzer)`) are not weights.
fn strip_weight(tag: &str) -> (String, bool) {
    let mut t = tag.trim().to_string();
    let mut weighted = false;
    loop {
        let wrapped = (t.starts_with('(') && t.ends_with(')') && !t.ends_with("\\)"))
            || (t.starts_with('[') && t.ends_with(']') && !t.ends_with("\\]"));
        if !wrapped || t.len() < 2 {
            break;
        }
        t = t[1..t.len() - 1].trim().to_string();
        weighted = true;
    }
    if weighted {
        if let Some((name, w)) = t.rsplit_once(':') {
            if w.trim().parse::<f32>().is_ok() {
                t = name.trim().to_string();
            }
        }
    }
    (t, weighted)
}

/// One caption tag in training form: no weight, unescaped, underscores as spaces (not in kaomoji),
/// single spaces.
fn normalize_tag(raw: &str) -> (String, bool) {
    let (t, weighted) = strip_weight(raw);
    let t = t.replace("\\(", "(").replace("\\)", ")");
    let t = if crate::tagger::is_kaomoji(t.trim()) { t.trim().to_string() } else { t.replace('_', " ") };
    (t.split_whitespace().collect::<Vec<_>>().join(" "), weighted)
}

/// Cleans one caption; returns the new text and what was removed.
fn clean_caption(text: &str, keep_first: usize, sort: bool, one_size: bool) -> (String, Stats) {
    let mut stats = Stats::default();
    let mut tags: Vec<String> = Vec::new();
    for raw in text.split([',', '\n']) {
        if raw.trim().is_empty() {
            continue;
        }
        let (tag, weighted) = normalize_tag(raw);
        stats.weights += weighted as usize;
        if tag.is_empty() {
            continue;
        }
        if tags.iter().any(|t| t.eq_ignore_ascii_case(&tag)) {
            stats.duplicates += 1;
            continue;
        }
        tags.push(tag);
    }
    let lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    let pinned = keep_first.min(tags.len());
    let mut keep = vec![true; tags.len()];
    for i in pinned..tags.len() {
        if lower.iter().enumerate().any(|(j, o)| j != i && crate::analyzer::implies(o, &lower[i])) {
            keep[i] = false;
            stats.implied += 1;
        } else if one_size && (0..i).any(|j| keep[j] && crate::analyzer::size_conflict(&lower[j], &lower[i])) {
            keep[i] = false;
            stats.sizes += 1;
        }
    }
    let kept: Vec<String> = tags.into_iter().zip(keep).filter(|(_, k)| *k).map(|(t, _)| t).collect();
    let (head, rest) = kept.split_at(pinned.min(kept.len()));
    let mut rest = rest.to_vec();
    if sort {
        // stable: tags of the same kind keep their order
        rest.sort_by_key(|t| {
            let escaped = t.replace('(', "\\(").replace(')', "\\)");
            crate::parser::parse_prompt(&escaped).first().map(crate::scene::slot).unwrap_or(crate::scene::Slot::Other)
        });
    }
    (head.iter().chain(&rest).cloned().collect::<Vec<_>>().join(", "), stats)
}

fn collect_captions(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if recursive {
                collect_captions(&p, recursive, out)?;
            }
        } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt")) {
            out.push(p);
        }
    }
    Ok(())
}

/// Cleans every caption in the folder; prints changed files and a summary.
fn run(o: &Options) -> Result<(usize, usize, Stats)> {
    if !o.folder.is_dir() {
        return Err(anyhow!("{} is not a folder", o.folder.display()));
    }
    let mut files = Vec::new();
    collect_captions(&o.folder, o.recursive, &mut files)?;
    let (mut changed, mut total, mut sum) = (0, 0, Stats::default());
    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("{}: not readable as text, skipped", path.display());
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        total += 1;
        let (clean, stats) = clean_caption(&text, o.keep_first, o.sort, o.one_size);
        let original = text.trim_end_matches(['\n', '\r']);
        if clean == original {
            continue;
        }
        if o.backup {
            let bak = PathBuf::from(format!("{}.bak", path.display()));
            if !bak.exists() {
                std::fs::write(&bak, &text)?;
            }
        }
        let newline = if text.ends_with('\n') { "\n" } else { "" };
        std::fs::write(path, format!("{clean}{newline}"))?;
        changed += 1;
        sum += stats;
        let name = path.strip_prefix(&o.folder).unwrap_or(path).display();
        println!(
            "{name}: {} duplicates, {} implied, {} weights{}",
            stats.duplicates,
            stats.implied,
            stats.weights,
            if o.one_size { format!(", {} sizes", stats.sizes) } else { String::new() }
        );
    }
    Ok((changed, total, sum))
}

pub fn run_dataset_cli(args: &[String]) -> Result<()> {
    let o = parse_args(args)?;
    let (changed, total, s) = run(&o)?;
    if total == 0 {
        return Err(anyhow!("no non-empty .txt captions in {}", o.folder.display()));
    }
    println!(
        "\nCleaned {changed} of {total} files: removed {} duplicates, {} implied tags, {} weights{}{}",
        s.duplicates,
        s.implied,
        s.weights,
        if o.one_size { format!(", {} sizes", s.sizes) } else { String::new() },
        if o.backup && changed > 0 { " (originals saved as .txt.bak)" } else { "" }
    );
    Ok(())
}

/// Captions every image in a folder for LoRA training, like `--caption` (the same pipeline,
/// `caption::caption_for`): writes `<image>.txt`, with `trigger_word` first when given. An existing caption is
/// never overwritten (it may hold hand edits); those images are skipped. Returns (written, images found).
#[allow(dead_code)] // for a future "caption a folder" button; --caption is the CLI way
pub fn batch_tag_images_in_folder(
    folder: &Path,
    trigger_word: Option<&str>,
    threshold: f32,
    recursive: bool,
) -> Result<(usize, usize)> {
    let mut images = Vec::new();
    crate::caption::collect_images(folder, recursive, &mut images)?;
    let trigger = trigger_word.map(str::trim).filter(|t| !t.is_empty());
    let mut written = 0;
    for img in &images {
        let txt = img.with_extension("txt");
        if txt.exists() {
            continue;
        }
        let Ok(caption) = crate::caption::caption_for(img, threshold) else { continue };
        let caption = match trigger {
            // the trigger word goes first (kohya keep_tokens), once
            Some(t) => {
                let rest: Vec<&str> = caption.split(", ").filter(|x| !x.eq_ignore_ascii_case(t)).collect();
                std::iter::once(t).chain(rest).collect::<Vec<_>>().join(", ")
            }
            None => caption,
        };
        if std::fs::write(&txt, format!("{caption}\n")).is_ok() {
            written += 1;
        }
    }
    Ok((written, images.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn options_are_parsed() {
        let o = parse_args(&args("data --recursive --no-backup --keep-first 2 --sort --one-size")).unwrap();
        assert_eq!(o, Options { folder: "data".into(), recursive: true, backup: false, keep_first: 2, sort: true, one_size: true });
        let o = parse_args(&args("data")).unwrap();
        assert_eq!((o.backup, o.keep_first, o.sort, o.one_size), (true, 1, false, false));
        assert!(parse_args(&args("data --keep-first x")).is_err());
        assert!(parse_args(&[]).is_err());
    }

    #[test]
    fn captions_are_tidied_without_adding_or_swapping_tags() {
        let (out, s) = clean_caption(
            "mychar,  1girl, long_hair ,(smile:1.2), ((blush)), [solo], Long hair, bikini, swimsuit, kay \\(girls und panzer\\), ^_^, full body, close-up",
            1,
            false,
            false,
        );
        // order kept, framing conflict (full body + close-up) left alone: captions describe the picture
        assert_eq!(out, "mychar, 1girl, long hair, smile, blush, solo, bikini, kay (girls und panzer), ^_^, full body, close-up");
        assert_eq!(s, Stats { duplicates: 1, implied: 1, weights: 3, sizes: 0 });
        // a name's own parenthesis is not a weight
        assert_eq!(clean_caption("kay (girls und panzer), 1girl", 1, false, false).0, "kay (girls und panzer), 1girl");
    }

    #[test]
    fn the_first_tags_are_pinned_and_sort_and_one_size_are_opt_in() {
        // the trigger word is never pruned, even when a later tag implies it
        assert_eq!(clean_caption("hat, witch hat, 1girl", 1, false, false).0, "hat, witch hat, 1girl");
        assert_eq!(clean_caption("hat, witch hat, 1girl", 0, false, false).0, "witch hat, 1girl");
        // sizes: only with --one-size, the first wins
        assert_eq!(clean_caption("trig, large breasts, small breasts", 1, false, false).0, "trig, large breasts, small breasts");
        let (out, s) = clean_caption("trig, large breasts, small breasts", 1, false, true);
        assert_eq!((out.as_str(), s.sizes), ("trig, large breasts", 1));
        // --sort reorders everything after the pinned tags by kind
        let sorted = clean_caption("trig, beach, smile, 1girl, red hair", 1, true, false).0;
        assert!(sorted.starts_with("trig, 1girl"), "{sorted}");
        assert!(sorted.find("red hair").unwrap() < sorted.find("beach").unwrap(), "{sorted}");
    }

    #[test]
    fn a_folder_is_cleaned_with_backups() {
        let dir = std::env::temp_dir().join(format!("pc-dataset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), "trig, 1girl, 1girl, (smile:1.1)\n").unwrap();
        std::fs::write(dir.join("b.txt"), "trig, 1girl, smile").unwrap();
        std::fs::write(dir.join("empty.txt"), "  \n").unwrap();
        std::fs::write(dir.join("a.txt.bak"), "older backup").unwrap();
        std::fs::write(dir.join("sub/c.txt"), "trig, sword, katana").unwrap();
        let mut o = parse_args(&args(dir.to_str().unwrap())).unwrap();
        let (changed, total, s) = run(&o).unwrap();
        assert_eq!((changed, total), (1, 2), "b is clean, empty skipped, sub not recursed");
        assert_eq!((s.duplicates, s.weights), (1, 1));
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "trig, 1girl, smile\n");
        assert_eq!(std::fs::read_to_string(dir.join("a.txt.bak")).unwrap(), "older backup", "an existing .bak is never overwritten");
        o.recursive = true;
        o.backup = false;
        let (changed, _, s) = run(&o).unwrap();
        assert_eq!((changed, s.implied), (1, 1));
        assert_eq!(std::fs::read_to_string(dir.join("sub/c.txt")).unwrap(), "trig, katana");
        assert!(!dir.join("sub/c.txt.bak").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
