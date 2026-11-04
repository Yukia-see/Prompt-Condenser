//! Scene / Pose reference image -> who does what.
//!
//! The reference is read per character (the person detector finds them left to right,
//! like the image split), and every tag that survives the pose-and-scene filter becomes
//! an `Action` with an owner: one character, all of them, the shared scene, or nobody.
//! Pose, gaze and holding tags start with the character they were seen on; camera,
//! setting, light, style and interaction tags start in the scene. The Scene panel lets
//! the user move them (the standing character gets the tea, not the sitting one) and
//! override any part; `plan` turns all of it into the tags each box receives.

use crate::danbooru::key;
use crate::parser::{parse_prompt, PromptToken};
use crate::scene::{filter_scene_and_pose_only, slot, Slot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// Character 1..4 (0-based).
    Char(usize),
    /// Every character.
    All,
    /// The shared Scene box (Before on the 1 Character tab).
    Scene,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub tag: String,
    pub owner: Owner,
    /// Where Auto Scene puts it back.
    pub auto: Owner,
}

/// The Advanced overrides; an empty field means "use what the image gave".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides {
    /// Replaces the scene's tags.
    pub scene: String,
    /// Tags never taken from the image.
    pub exclude: String,
    /// Replaces character i's actions.
    pub chars: [String; 4],
}

/// The tags each box receives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    pub chars: Vec<Vec<String>>,
    pub scene: Vec<String>,
}

/// Is this tag definitely a scene environment, camera, lighting, style, or interaction tag?
/// Body parts, anatomy and clothing are strictly forbidden from the Scene. Camera
/// framing ("full body", "cowboy shot", "from below") is part of what a pose reference
/// gives, so it's kept even though "full body" contains "body".
pub fn is_true_scene_tag(t: &PromptToken) -> bool {
    let n = t.normalized.as_str();
    if slot(t) == Slot::Camera {
        return true;
    }
    if crate::scene::is_character_like(n) || n.contains("body") {
        return false;
    }
    crate::two::is_interaction(n) || matches!(slot(t), Slot::Setting | Slot::Light | Slot::Style)
}

/// Something a character does (pose, action, holding, gaze, facing, hand gesture) rather than
/// something about the place or the camera. Body parts and identity are strictly excluded.
pub fn is_action(t: &PromptToken) -> bool {
    let n = t.normalized.as_str();
    if crate::two::is_interaction(n) {
        return false; // shared by the characters involved
    }
    // Reject body parts, anatomy, clothing, and character traits
    if crate::scene::is_character_like(n)
        && !n.contains("gesture")
        && !n.contains("hand on")
        && !n.contains("hands on")
        && !n.contains("arms up")
        && !n.contains("arms behind")
        && !n.contains("crossed arms")
        && !n.contains("arms crossed")
        && !n.contains("spread legs")
        && !n.contains("crossed legs")
    {
        return false;
    }
    n.starts_with("looking ")
        || n.starts_with("facing ")
        || n.ends_with(" gaze")
        || n == "eye contact"
        || slot(t) == Slot::Pose
        || n.starts_with("holding ")
        || n.contains("gesture")
        || n.contains("hand on")
        || n.contains("hands on")
        || n.contains("arms up")
        || n.contains("arms behind")
        || n.contains("crossed arms")
        || n.contains("arms crossed")
        || n.contains("spread legs")
        || n.contains("crossed legs")
        || n.contains("bent over")
        || n.contains("all fours")
        || n.contains("drinking")
        || n.contains("eating")
        || n.contains("reading")
        || n.contains("sleeping")
        || n.contains("sitting")
        || n.contains("standing")
        || n.contains("kneeling")
        || n.contains("lying")
        || n.contains("squatting")
        || n.contains("crouching")
        || n.contains("leaning")
        || n.contains("stretching")
        || n.contains("walking")
        || n.contains("running")
        || n.contains("jumping")
        || n.contains("dancing")
        || n.contains("waving")
        || n.contains("pointing")
        || n.contains("salute")
        || n.contains("shushing")
        || n == "v"
        || n == "v sign"
        || n == "peace sign"
}

/// Clears all actions assigned to character `char_idx`.
pub fn clear_char_actions(actions: &mut Vec<Action>, char_idx: usize) {
    actions.retain(|a| !matches!(a.owner, Owner::Char(c) if c == char_idx));
}

/// Moves actions from one character to another.
pub fn move_char_actions(actions: &mut [Action], from: usize, to: usize) {
    for a in actions.iter_mut() {
        if a.owner == Owner::Char(from) {
            a.owner = Owner::Char(to);
        }
    }
}

fn tags(text: &str) -> Vec<String> {
    parse_prompt(text).into_iter().map(|t| t.raw.trim().to_string()).filter(|t| !t.is_empty()).collect()
}

fn push(out: &mut Vec<Action>, tag: &str, owner: Owner) {
    if !out.iter().any(|a| a.owner == owner && key(&a.tag) == key(tag)) {
        out.push(Action { tag: tag.to_string(), owner, auto: owner });
    }
}

/// One character's reading (the 1 Character tab): its actions go to character 1,
/// setting/camera/lighting go to the scene. Body parts and identity are dropped.
pub fn from_single(text: &str, keep_unknown: bool) -> Vec<Action> {
    let mut out = Vec::new();
    for t in parse_prompt(&filter_scene_and_pose_only(text, keep_unknown)) {
        if is_action(&t) {
            push(&mut out, t.raw.trim(), Owner::Char(0));
        } else if is_true_scene_tag(&t) {
            push(&mut out, t.raw.trim(), Owner::Scene);
        }
    }
    out
}

/// A per-character reading (one prompt per detected character, left to right, plus the
/// shared part). An action seen on every character becomes one "All" action.
/// Non-action character tags (body parts, identity, clothes) are NEVER pushed to Scene.
pub fn from_split(chars: &[String], scene: &str, keep_unknown: bool) -> Vec<Action> {
    let parts: Vec<Vec<PromptToken>> =
        chars.iter().map(|c| parse_prompt(&filter_scene_and_pose_only(c, keep_unknown))).collect();
    let n = parts.iter().filter(|p| !p.is_empty()).count();
    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        for t in part {
            if is_action(t) {
                let everyone = n >= 2 && parts.iter().filter(|p| !p.is_empty()).all(|p| p.iter().any(|x| x.normalized == t.normalized));
                push(&mut out, t.raw.trim(), if everyone { Owner::All } else { Owner::Char(i) });
            } else if is_true_scene_tag(t) {
                push(&mut out, t.raw.trim(), Owner::Scene);
            }
        }
    }
    // the whole picture: setting, light, interactions only
    for t in parse_prompt(&filter_scene_and_pose_only(scene, keep_unknown)) {
        if is_true_scene_tag(&t) && !out.iter().any(|a| key(&a.tag) == t.normalized) {
            push(&mut out, t.raw.trim(), Owner::Scene);
        }
    }
    out
}

/// What goes into each box for `n` characters. Owners past the last character are
/// dropped (the tab has fewer characters than the reading).
pub fn plan(actions: &[Action], ov: &Overrides, n: usize) -> Plan {
    let excluded: Vec<String> = parse_prompt(&ov.exclude).into_iter().map(|t| t.normalized).collect();
    let keep = |list: Vec<String>| -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for t in list {
            let k = key(&t);
            if !excluded.contains(&k) && !out.iter().any(|x| key(x) == k) {
                out.push(t);
            }
        }
        out
    };
    let owned = |pred: &dyn Fn(Owner) -> bool| -> Vec<String> {
        actions.iter().filter(|a| pred(a.owner)).map(|a| a.tag.clone()).collect()
    };
    let scene = if ov.scene.trim().is_empty() { owned(&|o| o == Owner::Scene) } else { tags(&ov.scene) };
    let chars = (0..n.clamp(1, 4))
        .map(|i| {
            let own = &ov.chars[i];
            keep(if own.trim().is_empty() { owned(&|o| o == Owner::Char(i) || o == Owner::All) } else { tags(own) })
        })
        .collect();
    Plan { chars, scene: keep(scene) }
}

/// Puts `refs` into a box: the box keeps its own tags except the ones `replaced`
/// says the reference takes over (only when it brings any), then gets the refs.
pub fn inject(original: &str, refs: &[String], replaced: &dyn Fn(&PromptToken) -> bool) -> String {
    let mut kept: Vec<String> = parse_prompt(original)
        .into_iter()
        .filter(|t| refs.is_empty() || !replaced(t))
        .map(|t| t.raw.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for r in refs {
        if !kept.iter().any(|k| key(k) == key(r)) {
            kept.push(r.clone());
        }
    }
    kept.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner_of(a: &[Action], tag: &str) -> Vec<Owner> {
        a.iter().filter(|x| x.tag == tag).map(|x| x.owner).collect()
    }

    /// Auto Scene (Fast) on real images, the way main.rs runs it without the AI:
    /// `SCENE_TEST=a.png cargo test live_auto_scene -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_auto_scene() {
        for p in std::env::var("SCENE_TEST").unwrap().split(':') {
            let path = std::path::Path::new(p);
            for n in 1..=3usize {
                let t = std::time::Instant::now();
                let actions = if n == 1 {
                    let tags = crate::tagger::tag_image(path).unwrap();
                    let hints = filter_scene_and_pose_only(&crate::tagger::build_prompt(&tags, 0.35).prompt, false);
                    println!("  hints: {hints}");
                    from_single(&hints, false)
                } else {
                    let ts = crate::twoimg::read(path, n, false).unwrap();
                    let split = crate::twoimg::rule_split(&ts, 0.35);
                    println!("  people found: {}, split chars: {:?}\n  split scene: {}", ts.people, split.chars, split.scene);
                    from_split(&split.chars, &split.scene, false)
                };
                let plan = plan(&actions, &Overrides::default(), n);
                println!("== {p} n={n} ({:.1}s): {} actions\n  plan chars: {:?}\n  plan scene: {:?}", t.elapsed().as_secs_f32(), actions.len(), plan.chars, plan.scene);
            }
        }
    }

    #[test]
    fn split_assigns_actions_to_who_did_them() {
        let chars = vec![
            "1girl, silver hair, armor, standing, drinking tea, holding cup".to_string(),
            "1girl, blonde hair, maid, sitting, chair".to_string(),
        ];
        let a = from_split(&chars, "indoors, cafe, sunlight", false);
        assert_eq!(owner_of(&a, "standing"), vec![Owner::Char(0)]);
        assert_eq!(owner_of(&a, "drinking tea"), vec![Owner::Char(0)]);
        assert_eq!(owner_of(&a, "sitting"), vec![Owner::Char(1)]);
        assert_eq!(owner_of(&a, "indoors"), vec![Owner::Scene]);
        // identity never comes from the reference
        assert!(a.iter().all(|x| !x.tag.contains("hair") && x.tag != "armor" && x.tag != "1girl"));
    }

    #[test]
    fn body_parts_never_enter_scene() {
        let chars = vec![
            "1girl, large breasts, navel, pussy, standing, looking at viewer".to_string(),
            "1boy, penis, muscular, thighs, sitting".to_string(),
        ];
        let a = from_split(&chars, "classroom, sunlight, nipples, cleavage", false);
        assert_eq!(owner_of(&a, "standing"), vec![Owner::Char(0)]);
        assert_eq!(owner_of(&a, "sitting"), vec![Owner::Char(1)]);
        assert_eq!(owner_of(&a, "classroom"), vec![Owner::Scene]);
        assert_eq!(owner_of(&a, "sunlight"), vec![Owner::Scene]);
        // All body parts MUST be absent from both Scene and actions
        for forbidden in &["breasts", "navel", "pussy", "penis", "thighs", "nipples", "cleavage", "muscular"] {
            assert!(a.iter().all(|x| !x.tag.contains(forbidden)), "Found forbidden tag {forbidden} in actions!");
        }
    }

    #[test]
    fn shared_action_becomes_all() {
        let chars = vec!["1girl, looking at viewer, smile".to_string(), "1boy, looking at viewer".to_string()];
        let a = from_split(&chars, "", false);
        assert_eq!(owner_of(&a, "looking at viewer"), vec![Owner::All]);
    }

    #[test]
    fn plan_moves_and_overrides() {
        let chars = vec!["standing".to_string(), "sitting, drinking tea".to_string()];
        let mut a = from_split(&chars, "indoors, sunlight", false);
        // the standing character drinks the tea instead
        for x in a.iter_mut().filter(|x| x.tag == "drinking tea") {
            x.owner = Owner::Char(0);
        }
        let p = plan(&a, &Overrides::default(), 2);
        assert_eq!(p.chars[0], vec!["standing", "drinking tea"]);
        assert_eq!(p.chars[1], vec!["sitting"]);
        assert_eq!(p.scene, vec!["indoors", "sunlight"]);

        let ov = Overrides { scene: "beach, sunset".into(), exclude: "sitting".into(), chars: ["".into(), "".into(), "".into(), "".into()] };
        let p = plan(&a, &ov, 2);
        assert_eq!(p.scene, vec!["beach", "sunset"]);
        assert!(p.chars[1].is_empty());

        let ov = Overrides { chars: ["kneeling".into(), "".into(), "".into(), "".into()], ..Default::default() };
        assert_eq!(plan(&a, &ov, 2).chars[0], vec!["kneeling"]);
    }

    #[test]
    fn inject_replaces_only_when_the_reference_brings_something() {
        let replaced = |t: &PromptToken| is_action(t);
        assert_eq!(inject("1girl, red hair, sitting", &["standing".into()], &replaced), "1girl, red hair, standing");
        assert_eq!(inject("1girl, sitting", &[], &replaced), "1girl, sitting");
    }

    #[test]
    fn camera_framing_comes_with_the_reference() {
        let a = from_single("1girl, full body, from below, straight-on, cowboy shot, standing, beach", false);
        for cam in ["full body", "from below", "straight-on", "cowboy shot"] {
            assert_eq!(owner_of(&a, cam), vec![Owner::Scene], "{cam}");
        }
        assert_eq!(owner_of(&a, "standing"), vec![Owner::Char(0)]);
    }

    #[test]
    fn single_reading_splits_actions_from_scene() {
        let a = from_single("1girl, long hair, sitting, looking at viewer, bedroom, night", false);
        assert_eq!(owner_of(&a, "sitting"), vec![Owner::Char(0)]);
        assert_eq!(owner_of(&a, "looking at viewer"), vec![Owner::Char(0)]);
        assert_eq!(owner_of(&a, "bedroom"), vec![Owner::Scene]);
    }
}
