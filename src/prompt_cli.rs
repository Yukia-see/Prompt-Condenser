//! Prompt cleaning without the window, for scripts and the ComfyUI node:
//! `prompt-condenser --prompt "<text>" [--checkpoint anima|pony|illustrious|sdxl|sd15|flux] [--lora "<tags>"]
//! [--arranged] [--natural]` (text `-` = read standard input) prints JSON: the cleaned positive prompt (the
//! window's rule-based After view, all suggestions applied), the recommended negative and the changes made.
//! No AI: Deep Scans and image reads stay in the window (images: `--caption`).

use crate::analyzer::{self, ScanConfig};
use crate::profiles::{CheckpointProfile, DetailLevel, TargetView};
use crate::{condenser, parser, scene};
use anyhow::{anyhow, Result};
use std::io::Read;

const USAGE: &str = "usage: prompt-condenser --prompt \"<text>\" | - [--checkpoint anima|pony|illustrious|sdxl|sd15|flux] \
                     [--lora \"<tags>\"] [--arranged] [--natural]";

#[derive(Debug, PartialEq)]
struct Options {
    text: String,
    checkpoint: CheckpointProfile,
    lora: String,
    arranged: bool,
    natural: bool,
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
    let mut opts = Options { text: String::new(), checkpoint: CheckpointProfile::Anima, lora: String::new(), arranged: false, natural: false };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--checkpoint" => {
                let v = it.next().ok_or_else(|| anyhow!("--checkpoint needs a name\n{USAGE}"))?;
                opts.checkpoint = checkpoint_from_name(v).ok_or_else(|| anyhow!("unknown checkpoint {v}\n{USAGE}"))?;
            }
            "--lora" => opts.lora = it.next().ok_or_else(|| anyhow!("--lora needs tags\n{USAGE}"))?.clone(),
            "--arranged" => opts.arranged = true,
            "--natural" => opts.natural = true,
            "-h" | "--help" => return Err(anyhow!("{USAGE}")),
            t if text.is_none() && (t == "-" || !t.starts_with("--")) => text = Some(t.to_string()),
            other => return Err(anyhow!("unknown option {other}\n{USAGE}")),
        }
    }
    let text = text.ok_or_else(|| anyhow!("{USAGE}"))?;
    opts.text = if text == "-" { stdin()? } else { text };
    if opts.text.trim().is_empty() {
        return Err(anyhow!("the prompt is empty\n{USAGE}"));
    }
    Ok(opts)
}

/// The window's settings for a fresh 1 Character prompt (no angle / pose picks, cleanup and conflict fixes on).
fn config_for(o: &Options) -> ScanConfig {
    ScanConfig {
        checkpoint: o.checkpoint,
        target_view: TargetView::None,
        pinned_emphasis_raw: String::new(),
        emphasis_weight: 1.25,
        cleanup_enabled: true,
        resolve_conflicts_enabled: true,
        clamp_weights_enabled: true,
        camera_angle: 0,
        pose: 0,
        proportion: 0,
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

/// The JSON `--prompt` prints.
fn condense(o: &Options) -> serde_json::Value {
    let config = config_for(o);
    let tokens = parser::parse_prompt(&o.text);
    let result = analyzer::analyze_with_rules(&tokens, &config);
    let out = condenser::build_condensed_output(&o.text, &result.processed_tokens, &result.changes, &config);
    let sentence = (config.natural && config.checkpoint.supports_natural_language())
        .then(|| scene::prompt_sentence(&parser::parse_prompt(&out.final_prompt)));
    let positive = analyzer::finish_prompt(&out.final_prompt, &config, true, sentence.as_deref());
    let negative = if config.checkpoint.default_negative().is_empty() {
        String::new()
    } else {
        condenser::recommended_negative(&config, &positive)
    };
    let changes: Vec<serde_json::Value> = result
        .changes
        .iter()
        .map(|c| {
            serde_json::json!({
                "original": c.original,
                "suggestion": c.suggested_alternative,
                "reason": c.reason,
            })
        })
        .collect();
    serde_json::json!({
        "checkpoint": format!("{:?}", config.checkpoint),
        "positive": positive,
        "negative": negative,
        "changes": changes,
        "tokens_before": out.before_tokens,
        "tokens_after": out.after_tokens,
    })
}

pub fn run_prompt_cli(args: &[String]) -> Result<()> {
    let opts = parse_args(args, || {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    })?;
    println!("{}", serde_json::to_string_pretty(&condense(&opts))?);
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
            lora: String::new(),
            arranged: false,
            natural: false,
        };
        let v = condense(&o);
        let positive = v["positive"].as_str().unwrap();
        // the duplicate goes, the framing conflict (ass in an upper-body shot) is reported
        assert_eq!(positive.matches("smile").count(), 1, "{positive}");
        assert!(!v["changes"].as_array().unwrap().is_empty(), "{v}");
        assert!(!v["negative"].as_str().unwrap().is_empty(), "Pony has a recommended negative: {v}");
        // Flux: no negative
        let flux = condense(&Options { checkpoint: CheckpointProfile::Flux, ..o });
        assert_eq!(flux["negative"], "");
    }
}
