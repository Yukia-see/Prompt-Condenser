//! Prompt cleaning without the window, for scripts and the ComfyUI node:
//! `prompt-condenser --prompt "<text>" [--checkpoint anima|pony|illustrious|sdxl|sd15|flux] [--lora "<tags>"]
//! [--arranged] [--natural]` (text `-` = read standard input) prints JSON: the cleaned positive prompt (the
//! window's rule-based After view, all suggestions applied), the recommended negative and the changes made.
//! No AI: Deep Scans and image reads stay in the window (images: `--caption`).
//!
//! `--view none|turnaround|360|upper-body|close-up|front|from-behind|side|full-body` is the window's
//! Target View dropdown (`ScanConfig.target_view`): framing conflicts are judged against it and its
//! negative tags go into `negative`. Without it, like the window, the prompt's own framing tags decide.
//!
//! `--vrp` with 2-4 `--char "<prompt>"` options also prints `vrp_state`: the LuCiD VRP "Anima VRP (core)"
//! JSON (a string, exactly what the window's Copy VRP regions puts on the clipboard), one region per
//! character. The `--prompt` text is the shared scene: it stays the `positive` (the node's global prompt)
//! and the `--lora` box belongs to it. Each character goes through the same rule engine as in the window's
//! multi-character tabs: its own analysis, the `--view` (else the scene's detected framing), no LoRA box,
//! `solo` removed, cards labelled "Character N:" in `changes`. The `negative` leaves out anything the scene
//! or a character asks for. Regions: `--regions "x,y,w,h;…"` (fractions of the canvas) when it has one box
//! per character, else the person detector's boxes on `--image <path>` (downloaded on first use; the
//! canvas keeps the image's shape), else even left-to-right strips; `regions_source` says which, and
//! `warnings` why a better source wasn't used.

use crate::analyzer::{self, ScanConfig};
use crate::detector::{self, PersonBox};
use crate::profiles::{CheckpointProfile, DetailLevel, TargetView};
use crate::{condenser, parser, scene, two};
use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str = "usage: prompt-condenser --prompt \"<text>\" | - [--checkpoint anima|pony|illustrious|sdxl|sd15|flux] \
                     [--lora \"<tags>\"] [--arranged] [--natural] \
                     [--view none|turnaround|360|upper-body|close-up|front|from-behind|side|full-body] \
                     [--vrp --char \"<prompt>\" --char \"<prompt>\" (2-4) [--image <path>] [--regions \"x,y,w,h;...\"]]";

#[derive(Debug, PartialEq)]
struct Options {
    text: String,
    checkpoint: CheckpointProfile,
    lora: String,
    arranged: bool,
    natural: bool,
    view: TargetView,
    vrp: bool,
    chars: Vec<String>,
    image: Option<PathBuf>,
    regions: Option<Vec<[f64; 4]>>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            text: String::new(),
            checkpoint: CheckpointProfile::Anima,
            lora: String::new(),
            arranged: false,
            natural: false,
            view: TargetView::None,
            vrp: false,
            chars: Vec::new(),
            image: None,
            regions: None,
        }
    }
}

/// `--view` names: case doesn't matter, spaces / underscores / dashes are the same.
fn view_from_name(name: &str) -> Option<TargetView> {
    let n: String = name.trim().to_ascii_lowercase().chars().map(|c| if c == ' ' || c == '_' { '-' } else { c }).collect();
    Some(match n.as_str() {
        "none" => TargetView::None,
        "turnaround" | "360" | "turnaround-360" | "360-view" => TargetView::Turnaround360,
        "upper-body" | "cowboy-shot" => TargetView::UpperBody,
        "close-up" | "closeup" | "portrait" => TargetView::CloseUpPortrait,
        "front" | "front-view" => TargetView::FrontView,
        "from-behind" | "behind" | "back-view" => TargetView::FromBehind,
        "side" | "side-profile" | "from-side" => TargetView::SideProfile,
        "full-body" => TargetView::FullBody,
        _ => return None,
    })
}

/// `--regions "x,y,w,h;x,y,w,h"`: boxes as fractions of the canvas.
fn parse_regions(s: &str) -> Result<Vec<[f64; 4]>> {
    let bad = || anyhow!("--regions wants \"x,y,w,h;x,y,w,h;...\" with fractions from 0 to 1, got {s:?}");
    s.split(';')
        .filter(|r| !r.trim().is_empty())
        .map(|r| {
            let v: Vec<f64> = r.split(',').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| bad())?;
            match v.as_slice() {
                [x, y, w, h] if v.iter().all(|f| (0.0..=1.0).contains(f)) && *w > 0.0 && *h > 0.0 => Ok([*x, *y, *w, *h]),
                _ => Err(bad()),
            }
        })
        .collect()
}

fn checkpoint_from_name(name: &str) -> Option<CheckpointProfile> {
    Some(match name.to_ascii_lowercase().as_str() {
        "anima" => CheckpointProfile::Anima,
        "pony" | "ponyv6" | "pony-v6" => CheckpointProfile::PonyV6,
        "illustrious" | "noobai" => CheckpointProfile::Illustrious,
        "sdxl" => CheckpointProfile::Sdxl,
        "sd15" | "sd1.5" => CheckpointProfile::Sd15,
        "flux" => CheckpointProfile::Flux,
        _ => return None,
    })
}

fn parse_args(args: &[String], stdin: impl FnOnce() -> Result<String>) -> Result<Options> {
    let mut text = None;
    let mut opts = Options::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--checkpoint" => {
                let v = it.next().ok_or_else(|| anyhow!("--checkpoint needs a name\n{USAGE}"))?;
                opts.checkpoint = checkpoint_from_name(v).ok_or_else(|| anyhow!("unknown checkpoint {v}\n{USAGE}"))?;
            }
            "--lora" => opts.lora = it.next().ok_or_else(|| anyhow!("--lora needs tags\n{USAGE}"))?.clone(),
            "--arranged" => opts.arranged = true,
            "--view" => {
                let v = it.next().ok_or_else(|| anyhow!("--view needs a name\n{USAGE}"))?;
                opts.view = view_from_name(v).ok_or_else(|| {
                    anyhow!("unknown view {v}: use none, turnaround (or 360), upper-body, close-up, front, from-behind, side or full-body")
                })?;
            }
            "--vrp" => opts.vrp = true,
            "--char" => opts.chars.push(it.next().ok_or_else(|| anyhow!("--char needs a prompt\n{USAGE}"))?.clone()),
            "--image" => opts.image = Some(PathBuf::from(it.next().ok_or_else(|| anyhow!("--image needs a path\n{USAGE}"))?)),
            "--regions" => opts.regions = Some(parse_regions(it.next().ok_or_else(|| anyhow!("--regions needs boxes\n{USAGE}"))?)?),
            "--natural" => opts.natural = true,
            "-h" | "--help" => return Err(anyhow!("{USAGE}")),
            t if text.is_none() && (t == "-" || !t.starts_with("--")) => text = Some(t.to_string()),
            other => return Err(anyhow!("unknown option {other}\n{USAGE}")),
        }
    }
    let text = text.ok_or_else(|| anyhow!("{USAGE}"))?;
    opts.text = if text == "-" { stdin()? } else { text };
    // with --vrp the characters carry the prompt, so the shared scene may be empty
    if opts.text.trim().is_empty() && !opts.vrp {
        return Err(anyhow!("the prompt is empty\n{USAGE}"));
    }
    if !opts.vrp && (!opts.chars.is_empty() || opts.image.is_some() || opts.regions.is_some()) {
        return Err(anyhow!("--char, --image and --regions only go with --vrp\n{USAGE}"));
    }
    if opts.vrp && !(2..=4).contains(&opts.chars.len()) {
        return Err(anyhow!("--vrp needs 2 to 4 --char prompts (got {})\n{USAGE}", opts.chars.len()));
    }
    if opts.chars.iter().any(|c| c.trim().is_empty()) {
        return Err(anyhow!("a --char prompt is empty\n{USAGE}"));
    }
    Ok(opts)
}

/// The window's settings for a fresh 1 Character prompt (no angle / pose picks, cleanup and conflict fixes on;
/// the Target View from `--view`).
fn config_for(o: &Options) -> ScanConfig {
    ScanConfig {
        checkpoint: o.checkpoint,
        target_view: o.view,
        pinned_emphasis_raw: String::new(),
        emphasis_weight: 1.25,
        cleanup_enabled: true,
        resolve_conflicts_enabled: true,
        clamp_weights_enabled: true,
        camera_angle: 0,
        pose: 0,
        bust: 0,
        frame: 0,
        dynamic: false,
        lora_tags_raw: o.lora.clone(),
        arranged: o.arranged,
        natural: o.natural,
        two_char: false,
        char2_prompt_raw: String::new(),
        char2_lora_tags_raw: String::new(),
        char_break: false,
        scene_prompt_raw: String::new(),
        more_chars: Vec::new(),
        detail: DetailLevel::Balanced,
    }
}

/// What `--image` gave: the person detector's boxes (left to right, one per character) and the image size.
#[derive(Default)]
struct Detected {
    boxes: Option<Vec<PersonBox>>,
    size: Option<(u32, u32)>,
    warning: Option<String>,
}

/// One rule-engine pass, like the window's After view: the finished prompt plus the changes made.
fn clean(text: &str, config: &ScanConfig, sentence_allowed: bool) -> (analyzer::RuleAnalysisResult, condenser::CondensedOutput, String) {
    let result = analyzer::analyze_with_rules(&parser::parse_prompt(text), config);
    let out = condenser::build_condensed_output(text, &result.processed_tokens, &result.changes, config);
    let sentence = (sentence_allowed && config.natural && config.checkpoint.supports_natural_language())
        .then(|| scene::prompt_sentence(&parser::parse_prompt(&out.final_prompt)));
    let finished = analyzer::finish_prompt(&out.final_prompt, config, true, sentence.as_deref());
    (result, out, finished)
}

fn change_json(c: &crate::analyzer::PromptChange) -> serde_json::Value {
    serde_json::json!({
        "original": c.original,
        "suggestion": c.suggested_alternative,
        "reason": c.reason,
    })
}

/// The JSON `--prompt` prints.
fn condense(o: &Options, detected: &Detected) -> serde_json::Value {
    let config = config_for(o);
    let (result, out, positive) = clean(&o.text, &config, true);
    let mut changes: Vec<serde_json::Value> = result.changes.iter().map(change_json).collect();

    // --vrp: each character on its own, like the window's multi-character tabs
    let mut regions_prompts = Vec::new();
    for (i, text) in o.chars.iter().enumerate() {
        let mut ci = config.clone();
        ci.lora_tags_raw = String::new();
        ci.natural = false;
        if ci.target_view == TargetView::None {
            ci.target_view = result.active_framing_detected;
        }
        let mut r = analyzer::analyze_with_rules(&parser::parse_prompt(text), &ci);
        crate::label_two_char_cards(&mut r, i + 1);
        let out = condenser::build_condensed_output(text, &r.processed_tokens, &r.changes, &ci);
        regions_prompts.push(analyzer::finish_prompt(&out.final_prompt, &ci, true, None));
        changes.extend(r.changes.iter().map(change_json));
    }

    // nothing the scene or a character asks for goes into the negative
    let asked = std::iter::once(positive.clone()).chain(regions_prompts.iter().cloned()).collect::<Vec<_>>().join(",\n");
    let negative = condenser::recommended_negative(&config, &asked);
    let mut v = serde_json::json!({
        "checkpoint": format!("{:?}", config.checkpoint),
        "view": format!("{:?}", config.target_view),
        "positive": positive,
        "negative": negative,
        "changes": changes,
        "tokens_before": out.before_tokens,
        "tokens_after": out.after_tokens,
    });
    if o.vrp {
        let n = o.chars.len();
        let mut warnings: Vec<String> = detected.warning.iter().cloned().collect();
        let given = o.regions.as_ref().filter(|r| r.len() == n);
        if let Some(r) = &o.regions {
            if r.len() != n {
                warnings.push(format!("--regions has {} boxes for {n} characters; ignored", r.len()));
            }
        }
        let boxes = detected.boxes.as_deref().filter(|b| b.len() == n);
        let (regions, source) = match (given, boxes) {
            (Some(r), _) => (r.clone(), "regions"),
            (None, Some(b)) => (two::vrp_regions(n, Some(b), detected.size), "detector"),
            (None, None) => (two::vrp_regions(n, None, detected.size), "strips"),
        };
        v["characters"] = serde_json::json!(regions_prompts);
        v["vrp_state"] = serde_json::json!(two::vrp_state_json_from(&regions_prompts, &regions, detected.size));
        v["regions_source"] = serde_json::json!(source);
        v["warnings"] = serde_json::json!(warnings);
    }
    v
}

/// Reads `--image`: its size, and the person detector's `n` biggest boxes (downloading the detector once).
fn detect_people(path: &std::path::Path, n: usize) -> Detected {
    let image = match image::open(path).with_context(|| format!("couldn't open {}", path.display())) {
        Ok(i) => i,
        Err(e) => return Detected { warning: Some(format!("{e:#}; even strips used")), ..Default::default() },
    };
    let size = Some((image.width(), image.height()));
    let found = (|| -> Result<Option<Vec<PersonBox>>> {
        if !detector::is_installed() {
            let (tx, _rx) = std::sync::mpsc::channel();
            eprintln!("downloading the person detector (~45 MB, once)…");
            detector::download(&tx)?;
        }
        Ok(detector::main_n(&detector::detect_n(&image, n)?, n))
    })();
    match found {
        Ok(Some(boxes)) => Detected { boxes: Some(boxes), size, warning: None },
        Ok(None) => Detected { size, warning: Some(format!("the detector found fewer than {n} people; even strips used")), ..Default::default() },
        Err(e) => Detected { size, warning: Some(format!("person detection failed ({e:#}); even strips used")), ..Default::default() },
    }
}

pub fn run_prompt_cli(args: &[String]) -> Result<()> {
    let opts = parse_args(args, || {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    })?;
    let n = opts.chars.len();
    // the detector is only needed when --regions doesn't already give one box per character
    let detected = match &opts.image {
        Some(path) if opts.regions.as_ref().map_or(true, |r| r.len() != n) => detect_people(path, n),
        Some(path) => Detected { size: image::image_dimensions(path).ok(), ..Default::default() },
        None => Detected::default(),
    };
    println!("{}", serde_json::to_string_pretty(&condense(&opts, &detected))?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn prompt_options_are_parsed() {
        let no_stdin = || -> Result<String> { panic!("stdin not expected") };
        let o = parse_args(&args(&["1girl, smile", "--checkpoint", "pony", "--lora", "<lora:x:1>", "--arranged"]), no_stdin).unwrap();
        assert_eq!((o.text.as_str(), o.checkpoint, o.lora.as_str(), o.arranged, o.natural), ("1girl, smile", CheckpointProfile::PonyV6, "<lora:x:1>", true, false));
        let piped = parse_args(&args(&["-", "--checkpoint", "flux"]), || Ok("1girl, beach".to_string())).unwrap();
        assert_eq!((piped.text.as_str(), piped.checkpoint), ("1girl, beach", CheckpointProfile::Flux));
        assert!(parse_args(&args(&[]), no_stdin).is_err());
        assert!(parse_args(&args(&["x", "--checkpoint", "dalle"]), no_stdin).is_err());
        assert!(parse_args(&args(&["x", "--bogus"]), no_stdin).is_err());
    }

    #[test]
    fn prompts_are_cleaned_like_the_window_does() {
        let o = Options {
            text: "1girl, solo, upper body, ass, smile, smile, (masterpiece:1.8)".into(),
            checkpoint: CheckpointProfile::PonyV6,
            ..Default::default()
        };
        let v = condense(&o, &Detected::default());
        let positive = v["positive"].as_str().unwrap();
        // the duplicate goes, the framing conflict (ass in an upper-body shot) is reported
        assert_eq!(positive.matches("smile").count(), 1, "{positive}");
        assert!(!v["changes"].as_array().unwrap().is_empty(), "{v}");
        assert!(!v["negative"].as_str().unwrap().is_empty(), "Pony has a recommended negative: {v}");
        // Flux: no negative
        let flux = condense(&Options { checkpoint: CheckpointProfile::Flux, ..o }, &Detected::default());
        assert_eq!(flux["negative"], "");
        assert!(flux.get("vrp_state").is_none(), "no VRP without --vrp");
    }

    #[test]
    fn view_and_vrp_options_are_parsed() {
        let no_stdin = || -> Result<String> { panic!("stdin not expected") };
        for (name, view) in [
            ("none", TargetView::None),
            ("360", TargetView::Turnaround360),
            ("Turnaround", TargetView::Turnaround360),
            ("upper_body", TargetView::UpperBody),
            ("Close Up", TargetView::CloseUpPortrait),
            ("front", TargetView::FrontView),
            ("FROM-BEHIND", TargetView::FromBehind),
            ("side", TargetView::SideProfile),
            ("full body", TargetView::FullBody),
        ] {
            assert_eq!(parse_args(&args(&["x", "--view", name]), no_stdin).unwrap().view, view, "{name}");
        }
        let err = parse_args(&args(&["x", "--view", "worm's eye"]), no_stdin).unwrap_err().to_string();
        assert!(err.contains("upper-body") && err.contains("full-body"), "{err}");

        let o = parse_args(
            &args(&["beach", "--vrp", "--char", "1girl, red hair", "--char", "1boy", "--image", "a.png", "--regions", "0,0,0.5,1; 0.5,0,0.5,1"]),
            no_stdin,
        )
        .unwrap();
        assert!(o.vrp);
        assert_eq!(o.chars, vec!["1girl, red hair".to_string(), "1boy".to_string()]);
        assert_eq!(o.image, Some(PathBuf::from("a.png")));
        assert_eq!(o.regions, Some(vec![[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0]]));
        // --char without --vrp, --vrp with one character or five, bad boxes
        assert!(parse_args(&args(&["x", "--char", "1girl", "--char", "1boy"]), no_stdin).is_err());
        assert!(parse_args(&args(&["x", "--vrp", "--char", "1girl"]), no_stdin).is_err());
        let five: Vec<&str> = ["x", "--vrp"].into_iter().chain(std::iter::repeat(["--char", "1girl"]).take(5).flatten()).collect();
        assert!(parse_args(&args(&five), no_stdin).is_err());
        assert!(parse_args(&args(&["x", "--vrp", "--char", "a", "--char", "b", "--regions", "0,0,1.5,1;0,0,1,1"]), no_stdin).is_err());
        assert!(parse_args(&args(&["x", "--vrp", "--char", "a", "--char", "b", "--regions", "0,0,1"]), no_stdin).is_err());
    }

    #[test]
    fn the_view_works_like_the_window_dropdown() {
        let base = Options { text: "1girl, smile, boots, beach".into(), checkpoint: CheckpointProfile::PonyV6, ..Default::default() };
        let none = condense(&base, &Detected::default());
        let upper = condense(&Options { view: TargetView::UpperBody, ..base }, &Detected::default());
        assert_eq!(upper["view"], "UpperBody");
        // the dropdown's negative tags, and boots (out of an upper-body shot) get a card
        let neg = upper["negative"].as_str().unwrap();
        assert!(neg.contains("full body") && !none["negative"].as_str().unwrap().contains("full body"), "{neg}");
        assert!(upper["changes"].as_array().unwrap().iter().any(|c| c["original"] == "boots"), "{upper}");
        assert!(!upper["positive"].as_str().unwrap().contains("boots"), "{upper}");
    }

    fn vrp_of(v: &serde_json::Value) -> serde_json::Value {
        serde_json::from_str(v["vrp_state"].as_str().expect("vrp_state is a string")).unwrap()
    }

    #[test]
    fn vrp_accepts_an_empty_scene() {
        let no_stdin = || -> Result<String> { panic!("stdin not expected") };
        assert!(parse_args(&args(&[""]), no_stdin).is_err(), "an empty prompt without --vrp");
        let o = parse_args(&args(&["", "--vrp", "--char", "1girl, red hair", "--char", "1girl, blue hair"]), no_stdin).unwrap();
        let piped = parse_args(&args(&["-", "--vrp", "--char", "1girl", "--char", "1boy"]), || Ok("  \n".to_string()));
        assert!(piped.is_ok());
        let v = condense(&o, &Detected::default());
        assert_eq!(v["positive"], "", "no tags added to an empty scene: {v}");
        assert!(!v["negative"].as_str().unwrap().contains("red hair"), "{v}");
        assert_eq!(v["characters"].as_array().unwrap().len(), 2);
        assert_eq!(vrp_of(&v)["regions"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn vrp_state_has_one_cleaned_region_per_character() {
        let o = Options {
            text: "masterpiece, beach, sunset".into(),
            vrp: true,
            chars: vec!["1girl, solo, red hair, smile, smile".into(), "1boy, black hair".into(), "1girl, blonde hair".into()],
            ..Default::default()
        };
        let v = condense(&o, &Detected::default());
        assert_eq!(v["regions_source"], "strips");
        let vrp = vrp_of(&v);
        assert_eq!(vrp["version"], 2);
        let regions = vrp["regions"].as_array().unwrap();
        assert_eq!(regions.len(), 3);
        // even strips, left to right
        assert_eq!(regions[0]["x"], 0.0);
        assert!(regions[1]["x"].as_f64().unwrap() > 0.3 && regions[2]["x"].as_f64().unwrap() > 0.6, "{vrp}");
        assert_eq!(regions[0]["height"], 1.0);
        // cleaned: the duplicate and "solo" go, with a labelled card
        let p0 = regions[0]["prompt"].as_str().unwrap();
        assert_eq!(p0.matches("smile").count(), 1, "{p0}");
        assert!(!p0.contains("solo"), "{p0}");
        assert!(v["changes"].as_array().unwrap().iter().any(|c| c["reason"].as_str().unwrap().starts_with("Character 1:")), "{v}");
        assert!(regions[1]["prompt"].as_str().unwrap().contains("black hair"));
        // the scene stays the global prompt, and the negative doesn't fight a character
        assert!(v["positive"].as_str().unwrap().contains("beach"));
        assert!(!v["negative"].as_str().unwrap().contains("blonde hair"));
    }

    #[test]
    fn vrp_regions_option_overrides_the_strips_when_the_count_matches() {
        let o = Options {
            text: "park".into(),
            vrp: true,
            chars: vec!["1girl".into(), "1boy".into()],
            regions: Some(vec![[0.1, 0.2, 0.3, 0.7], [0.55, 0.1, 0.4, 0.85]]),
            ..Default::default()
        };
        let v = condense(&o, &Detected::default());
        assert_eq!(v["regions_source"], "regions");
        let vrp = vrp_of(&v);
        let r = &vrp["regions"];
        assert_eq!((r[0]["x"].as_f64(), r[0]["y"].as_f64(), r[0]["width"].as_f64(), r[0]["height"].as_f64()), (Some(0.1), Some(0.2), Some(0.3), Some(0.7)));
        assert_eq!(r[1]["x"].as_f64(), Some(0.55));
        // the image's shape sets the canvas (portrait image -> portrait canvas)
        let tall = condense(&o, &Detected { size: Some((832, 1216)), ..Default::default() });
        let c = &vrp_of(&tall)["canvas"];
        assert!(c["height"].as_u64() > c["width"].as_u64(), "{c}");
        // a count that doesn't match falls back to strips, with a warning
        let wrong = condense(&Options { regions: Some(vec![[0.0, 0.0, 1.0, 1.0]]), ..o }, &Detected::default());
        assert_eq!(wrong["regions_source"], "strips");
        assert_eq!(wrong["warnings"].as_array().unwrap().len(), 1);
    }
}
