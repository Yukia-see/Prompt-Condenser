//! 💡 Ideas: suggestions for the parts a prompt leaves out (camera, setting, lighting, expression, pose).
//! The app decides which parts are missing (`scene::slot`) and gives the model a menu of real Danbooru tags
//! for each (the scene keyword lists that are real tags); the model only picks options that fit the scene,
//! and anything not on the menu is dropped. Each part becomes an opt-in card (`ChangeAction::Idea`) whose
//! tags are switched on one by one. Measured before the menus (free choice, 2 prompts): E4B 3.2-3.4 s warm,
//! half its tags weren't real Danbooru tags.

use crate::analyzer::{ChangeAction, PromptChange};
use crate::parser;
use crate::profiles::AiMode;
use crate::scene::{self, Slot};
use anyhow::{anyhow, Result};
use serde_json::json;
use std::time::Duration;

/// Idea card ids start here (rule cards: 0 / `two::card_base`, AI cards: `ai_engine::AI_CARD_ID_BASE`).
pub const IDEA_CARD_ID_BASE: usize = 60_000;

/// A part the prompt leaves out, with the real tags the model may pick from.
#[derive(Debug, Clone)]
pub struct Gap {
    pub name: &'static str,
    pub menu: Vec<String>,
}

const PARTS: &[(Slot, &str)] = &[
    (Slot::Camera, "camera framing"),
    (Slot::Setting, "setting / background"),
    (Slot::Light, "lighting / atmosphere"),
    (Slot::Expression, "expression"),
    (Slot::Pose, "pose"),
];

/// The parts the prompt has no tag for, each with its menu of real Danbooru tags (needs the tag list on disk;
/// without it the menus would be unchecked, so nothing is suggested).
pub fn gaps(prompt: &str) -> Vec<Gap> {
    let Some(list) = crate::danbooru::local() else { return Vec::new() };
    let tokens = parser::parse_prompt(prompt);
    let present: Vec<Slot> = tokens.iter().map(scene::slot).collect();
    // a scenery prompt has no expression or pose to fill: only a character count (1girl, 2boys, solo) says
    // someone is in it ("scenery" itself sorts as a subject word)
    let has_subject = tokens.iter().any(|t| t.normalized == "solo" || crate::two::count_tag(&t.normalized).is_some());
    PARTS
        .iter()
        .filter(|(slot, _)| !present.contains(slot))
        .filter(|(slot, _)| has_subject || !matches!(slot, Slot::Expression | Slot::Pose))
        .filter_map(|(slot, name)| {
            let mut menu: Vec<String> = Vec::new();
            for w in scene::slot_words(*slot) {
                let w = w.trim().replace("\\(", "(").replace("\\)", ")");
                if let Some(t) = list.get(&w).filter(|t| !t.character && t.count >= 1_000) {
                    if !menu.contains(&t.name) {
                        menu.push(t.name.clone());
                    }
                }
            }
            (!menu.is_empty()).then_some(Gap { name, menu })
        })
        .collect()
}

/// Asks the model for up to 3 options (1-2 tags each) per gap, from the menus only.
pub fn suggest(host: &str, model: &str, mode: AiMode, prompt: &str, gaps: &[Gap]) -> Result<Vec<(String, Vec<Vec<String>>)>> {
    if gaps.is_empty() {
        return Ok(Vec::new());
    }
    let base = crate::ai_engine::base_url(host)?;
    let menus: String = gaps.iter().map(|g| format!("- {}: {}\n", g.name, g.menu.join(", "))).collect();
    let system = "You add missing details to an anime image-generation prompt. For each missing part, choose 3 \
        different options that fit the scene and the character; each option is 1 or 2 tags. Choose ONLY tags from \
        that part's menu, spelled exactly as in the menu. Never repeat a tag already in the prompt. Answer in JSON.";
    let user = format!("Prompt: {prompt}\n\nMissing parts and their menus:\n{menus}");
    let schema = json!({
        "type": "object",
        "properties": { "suggestions": { "type": "array", "items": { "type": "object",
            "properties": { "part": { "type": "string" }, "options": { "type": "array", "items": { "type": "string" } } },
            "required": ["part", "options"] } } },
        "required": ["suggestions"],
    });
    let body = json!({
        "model": model, "system": system, "prompt": user, "format": schema, "stream": false, "think": false,
        "keep_alive": "30m",
        "options": { "temperature": 0.7, "num_predict": 500, "num_gpu": if mode == AiMode::CpuOnly { Some(0) } else { None::<i32> } },
    });
    let http = crate::ai_engine::client(Some(Duration::from_secs(180)))?;
    let mut resp = http.post(format!("{base}/api/generate")).json(&body).send().map_err(|_| anyhow!("Couldn't reach Ollama at {base}"))?;
    if resp.status().as_u16() == 400 {
        let mut b = body.clone();
        b.as_object_mut().expect("object").remove("think");
        resp = http.post(format!("{base}/api/generate")).json(&b).send().map_err(|_| anyhow!("Couldn't reach Ollama at {base}"))?;
    }
    if !resp.status().is_success() {
        return Err(anyhow!("Ollama answered {} for the idea request", resp.status()));
    }
    let v: serde_json::Value = resp.json()?;
    let raw = v["response"].as_str().unwrap_or("");
    let reply: serde_json::Value = serde_json::from_str(&raw[raw.find('{').unwrap_or(0)..]).map_err(|_| anyhow!("The model's reply was unreadable. Try again."))?;
    Ok(validate(&reply, prompt, gaps))
}

/// Keeps only menu tags that aren't in the prompt yet, at most 3 options per gap, in the gaps' order.
pub fn validate(reply: &serde_json::Value, prompt: &str, gaps: &[Gap]) -> Vec<(String, Vec<Vec<String>>)> {
    let key = |t: &str| t.trim().to_lowercase().replace('_', " ").replace("\\(", "(").replace("\\)", ")");
    let in_prompt: Vec<String> = parser::parse_prompt(prompt).iter().map(|t| key(&t.normalized)).collect();
    let items: Vec<&serde_json::Value> = reply["suggestions"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    let mut out = Vec::new();
    for (gi, gap) in gaps.iter().enumerate() {
        // match the reply's part by name, else by position
        let item = items
            .iter()
            .find(|it| it["part"].as_str().is_some_and(|p| key(p) == key(gap.name) || gap.name.starts_with(&key(p))))
            .or_else(|| items.get(gi))
            .copied();
        let Some(item) = item else { continue };
        let mut options: Vec<Vec<String>> = Vec::new();
        let mut used: Vec<String> = Vec::new();
        for opt in item["options"].as_array().into_iter().flatten().filter_map(|o| o.as_str()) {
            let tags: Vec<String> = opt
                .split(',')
                .map(key)
                .filter(|t| gap.menu.iter().any(|m| key(m) == *t) && !in_prompt.contains(t) && !used.contains(t))
                .take(2)
                .collect();
            if tags.is_empty() {
                continue;
            }
            used.extend(tags.iter().cloned());
            options.push(tags);
            if options.len() == 3 {
                break;
            }
        }
        if !options.is_empty() {
            out.push((gap.name.to_string(), options));
        }
    }
    out
}

/// One opt-in card per gap: every option's tags as chips (the caller switches off all but the first option).
pub fn cards(suggestions: &[(String, Vec<Vec<String>>)]) -> Vec<PromptChange> {
    suggestions
        .iter()
        .enumerate()
        .map(|(i, (part, options))| PromptChange {
            id: IDEA_CARD_ID_BASE + i,
            original: String::new(),
            action: ChangeAction::Idea,
            reason: format!(
                "Idea for the {part}, which the prompt doesn't say. Options: {}. Pick the tags you want, then Add It.",
                options.iter().map(|o| o.join(" + ")).collect::<Vec<_>>().join(" · ")
            ),
            suggested_alternative: options.iter().flatten().cloned().collect::<Vec<_>>().join(", "),
            framing_alternative: String::new(),
            mode: 2,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replies_keep_only_menu_tags_not_in_the_prompt() {
        let gaps = vec![
            Gap { name: "lighting / atmosphere", menu: vec!["sunset".into(), "backlighting".into(), "moonlight".into()] },
            Gap { name: "pose", menu: vec!["sitting".into(), "standing".into()] },
        ];
        let reply = json!({ "suggestions": [
            { "part": "lighting / atmosphere", "options": ["sunset, backlighting", "golden hour", "moonlight"] },
            { "part": "pose", "options": ["sitting", "standing", "leaning"] },
        ]});
        let out = validate(&reply, "1girl, standing", &gaps);
        assert_eq!(out[0].1, vec![vec!["sunset".to_string(), "backlighting".to_string()], vec!["moonlight".to_string()]]);
        // "standing" is already in the prompt, "leaning" isn't on the menu
        assert_eq!(out[1].1, vec![vec!["sitting".to_string()]]);
        let c = cards(&out);
        assert_eq!(c[0].suggested_alternative, "sunset, backlighting, moonlight");
        assert!(c.iter().all(|c| c.mode == 2 && c.action == ChangeAction::Idea));
    }

    #[test]
    fn gaps_follow_what_the_prompt_already_has() {
        if crate::danbooru::local().is_none() {
            return; // needs the tag list on disk
        }
        let names: Vec<&str> = gaps("1girl, smile, standing, upper body").iter().map(|g| g.name).collect();
        assert_eq!(names, vec!["setting / background", "lighting / atmosphere"]);
        // scenery: no expression or pose
        let names: Vec<&str> = gaps("scenery, mountain").iter().map(|g| g.name).collect();
        assert!(!names.contains(&"pose") && !names.contains(&"expression"), "{names:?}");
        for g in gaps("1girl") {
            assert!(!g.menu.is_empty() && g.menu.iter().all(|t| crate::danbooru::local().unwrap().get(t).is_some()), "{g:?}");
        }
    }
}
