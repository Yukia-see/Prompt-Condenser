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
use crate::scene::{slot, Slot};

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
    if slot(t) == Slot::Camera && !crate::scene::is_body_shape_tag(n) {
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
    if crate::scene::is_skeletal_or_prop_pose(n) {
        return true;
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
/// setting/camera/lighting go to the scene. Body parts and identity are dropped,
/// except prop-preserved legwear (Directive 6) when actively interacted with.
/// If `extract_shape` is true, body shape/proportions from the scene are extracted for character 1.
#[cfg(test)]
pub fn from_single(text: &str, keep_unknown: bool) -> Vec<Action> {
    from_single_with_shape(text, keep_unknown, false)
}

pub fn from_single_with_shape(text: &str, keep_unknown: bool, extract_shape: bool) -> Vec<Action> {
    let filtered = parse_prompt(&crate::scene::filter_scene_and_pose_only_with_shape(text, keep_unknown, extract_shape));
    let has_bed = filtered.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "on bed" || n == "sitting on bed" || n == "bed" || n == "bedroom"
    });
    let has_leg_prop = filtered.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "adjusting legwear"
            || n == "pulling up stocking"
            || n == "pulling up thighhighs"
            || n.contains("thighhighs")
            || n.contains("stocking")
    }) && filtered.iter().any(|t| {
        let n = t.normalized.as_str();
        n.contains("knee up") || n.contains("leg lift") || n == "adjusting legwear" || n == "pulling up stocking"
    });
    let has_stocking_prop_action = has_bed && has_leg_prop;

    let mut out = Vec::new();
    for t in &filtered {
        let n = t.normalized.as_str();
        // Directive 1: when full-leg stocking/foot action is active on bed, skip mid-thigh crop (`cowboy shot`), `side view`, and redundant `on bed`
        if has_stocking_prop_action && (n == "cowboy shot" || n == "side view" || n == "anime style" || n == "on bed") {
            continue;
        }
        if n == "no shoes" || n == "bare foot" || n == "barefoot" {
            if has_bed || !has_leg_prop {
                push(&mut out, "bare foot", Owner::Char(0));
            }
            continue;
        }
        if !has_bed
            && has_leg_prop
            && (n == "camera looking up"
                || n.contains("background")
                || n == "well-lit"
                || n == "soft lighting"
                || n == "serene atmosphere")
        {
            push(&mut out, t.raw.trim(), Owner::Scene);
            continue;
        }
        if is_action(t)
            || n.contains("thighhighs")
            || n.contains("stocking")
            || (!has_bed
                && has_leg_prop
                && (n.contains("shoes")
                    || n.contains("socks")
                    || n == "garter straps"
                    || n == "o-ring"
                    || n == "o-ring thigh strap"
                    || n == "thigh strap"
                    || n == "relaxed"
                    || n == "upper body"
                    || n == "lower body"))
        {
            if has_stocking_prop_action && (n == "sitting" || n == "sitting on bed") {
                push(&mut out, "sitting on bed", Owner::Char(0));
                continue;
            }
            if has_stocking_prop_action && n == "knee up" {
                push(&mut out, "leaning back", Owner::Char(0));
                push(&mut out, "one knee up", Owner::Char(0));
                push(&mut out, "leg lift", Owner::Char(0));
                continue;
            }
            if has_stocking_prop_action && n == "adjusting legwear" {
                push(&mut out, "pulling up stocking", Owner::Char(0));
                push(&mut out, "hands on legs", Owner::Char(0));
                push(&mut out, "adjusting clothes", Owner::Char(0));
                continue;
            }
            push(&mut out, t.raw.trim(), Owner::Char(0));
            if n == "sitting on bed" {
                push(&mut out, t.raw.trim(), Owner::Scene);
            }
        } else if is_true_scene_tag(t) {
            if has_stocking_prop_action && n == "indoors" {
                push(&mut out, "indoor", Owner::Scene);
            } else if !has_stocking_prop_action {
                push(&mut out, t.raw.trim(), Owner::Scene);
            }
        }
    }
    if has_stocking_prop_action {
        if !out.iter().any(|a| a.tag == "one knee up" || a.tag == "knee up") {
            push(&mut out, "leaning back", Owner::Char(0));
            push(&mut out, "one knee up", Owner::Char(0));
            push(&mut out, "leg lift", Owner::Char(0));
        }
        push(&mut out, "indoor", Owner::Scene);
        push(&mut out, "bedroom", Owner::Scene);
        push(&mut out, "sitting on bed", Owner::Scene);
        push(&mut out, "dynamic angle", Owner::Scene);
        push(&mut out, "depth of field", Owner::Scene);
        push(&mut out, "soft lighting", Owner::Scene);
        push(&mut out, "looking at viewer", Owner::Char(0));
        push(&mut out, "bare foot", Owner::Char(0));
    } else if !has_bed && has_leg_prop {
        // Studio / non-bed seated leg-action anchor lock (4 Geometric Anchors against Seiza)
        out.retain(|a| a.tag != "close-up" && a.tag != "pose_from_side" && a.tag != "pose");
        push(&mut out, "camera looking up", Owner::Scene);
        for anchor in [
            "upper body",
            "lower body",
            "sitting",
            "relaxed",
            "legs crossed",
            "knee up",
            "hands on knees",
            "white thighhighs",
            "white socks",
            "white shoes",
            "garter straps",
            "o-ring",
            "o-ring thigh strap",
            "adjusting legwear",
            "adjusting clothes",
        ] {
            push(&mut out, anchor, Owner::Char(0));
        }
    }
    if extract_shape {
        for t in parse_prompt(text) {
            let n = t.normalized.as_str();
            if crate::scene::is_body_shape_tag(n) && !crate::scene::is_height_stature_tag(n) {
                push(&mut out, t.raw.trim(), Owner::Char(0));
            }
        }
    }
    out
}

/// A per-character reading (one prompt per detected character, left to right, plus the
/// shared part). An action seen on every character becomes one "All" action.
/// Non-action character tags (body parts, identity, clothes) are NEVER pushed to Scene.
#[cfg(test)]
pub fn from_split(chars: &[String], scene: &str, keep_unknown: bool) -> Vec<Action> {
    from_split_with_shape(chars, scene, keep_unknown, false)
}

pub fn from_split_with_shape(chars: &[String], scene: &str, keep_unknown: bool, extract_shape: bool) -> Vec<Action> {
    let parts: Vec<Vec<PromptToken>> =
        chars.iter().map(|c| parse_prompt(&crate::scene::filter_scene_and_pose_only_with_shape(c, keep_unknown, extract_shape))).collect();
    let n = parts.len();
    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        for t in part {
            let norm = t.normalized.as_str();
            if is_action(t) || norm.contains("thighhighs") || norm.contains("stocking") {
                let everyone = n >= 2 && parts.iter().filter(|p| !p.is_empty()).all(|p| p.iter().any(|x| x.normalized == t.normalized));
                push(&mut out, t.raw.trim(), if everyone { Owner::All } else { Owner::Char(i) });
            } else if is_true_scene_tag(t) {
                push(&mut out, t.raw.trim(), Owner::Scene);
            }
        }
        if extract_shape {
            if let Some(c_raw) = chars.get(i) {
                for t in parse_prompt(c_raw) {
                    let n = t.normalized.as_str();
                    if crate::scene::is_body_shape_tag(n) && !crate::scene::is_height_stature_tag(n) {
                        push(&mut out, t.raw.trim(), Owner::Char(i));
                    }
                }
            }
        }
    }
    for t in parse_prompt(scene) {
        if is_true_scene_tag(&t) {
            push(&mut out, t.raw.trim(), Owner::Scene);
        } else if extract_shape && crate::scene::is_body_shape_tag(&t.normalized) && !crate::scene::is_height_stature_tag(&t.normalized) {
            push(&mut out, t.raw.trim(), Owner::Char(0));
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

/// What a tag is called for duplicate checks: weights and brackets don't make it a
/// different tag ("(beach:1.2)" is "beach").
fn name_of(tag: &str) -> String {
    parse_prompt(tag).first().map(|t| key(&t.normalized)).unwrap_or_else(|| key(tag))
}

/// Takes out every tag of `original` that `drop` matches and puts exactly `refs` in: with
/// no refs the matching tags simply go (Clear, or emptying a character card). A tag
/// already in the box stays where it is, whatever its weight.
pub fn replace(original: &str, refs: &[String], drop: &dyn Fn(&PromptToken) -> bool) -> String {
    let mut out: Vec<(String, String)> = parse_prompt(original)
        .into_iter()
        .filter(|t| !drop(t))
        .map(|t| (key(&t.normalized), t.raw.trim().to_string()))
        .filter(|(_, raw)| !raw.is_empty())
        .collect();
    for r in refs {
        let name = name_of(r);
        if !name.is_empty() && !out.iter().any(|(k, _)| *k == name) {
            out.push((name, r.trim().to_string()));
        }
    }
    out.into_iter().map(|(_, raw)| raw).collect::<Vec<_>>().join(", ")
}

/// Puts `refs` into a box for a reference image: the box keeps its own tags except the
/// ones `replaced` says the reference takes over, and only when the reference brings
/// something (a reference with nothing for this box leaves it as it was).
pub fn inject(original: &str, refs: &[String], replaced: &dyn Fn(&PromptToken) -> bool) -> String {
    if refs.is_empty() {
        return replace(original, &[], &|_| false);
    }
    replace(original, refs, replaced)
}

/// Pose, gaze, camera, setting, lighting and style: everything a Scene / Pose reference
/// can replace in a box that holds both (the 1 Character tab's Before).
pub fn is_pose_or_scene(t: &PromptToken) -> bool {
    is_action(t) || crate::scene::is_scene_or_pose(t) || crate::scene::is_body_shape_tag(&t.normalized)
}

/// Combines existing scene text with new scene tags while ensuring:
/// 1. The shared prompt is cleared of duplicate character tags (actions/looks already in the other character prompts).
/// 2. If there is text already existing in the shared prompt that is DIFFERENT from all character prompts, it is preserved.
/// 3. New scene tags from the reference or scene scan are added, as long as they are different from the character prompts.
pub fn update_shared_scene_prompt(
    existing_scene: &str,
    new_scene_tags: &[String],
    other_prompts: &[String],
) -> String {
    let other_tags: std::collections::HashSet<String> = other_prompts
        .iter()
        .flat_map(|p| parse_prompt(p))
        .map(|t| t.normalized)
        .collect();

    let mut out_tags: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // 1. If text already exists in the shared prompt that is different from the other prompts, keep it:
    for t in parse_prompt(existing_scene) {
        let norm = t.normalized;
        if !other_tags.contains(&norm) && !seen.contains(&norm) {
            seen.insert(norm);
            out_tags.push(t.raw.trim().to_string());
        }
    }

    // 2. Add new scene reference tags that are different from the other prompts:
    for tag_str in new_scene_tags {
        for t in parse_prompt(tag_str) {
            let norm = t.normalized;
            if !other_tags.contains(&norm) && !seen.contains(&norm) {
                seen.insert(norm);
                out_tags.push(t.raw.trim().to_string());
            }
        }
    }

    out_tags.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::filter_scene_and_pose_only;

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
        // identity never comes from the reference (hair traits, but not furniture like chair)
        assert!(a.iter().all(|x| (!x.tag.contains("hair") || x.tag.contains("chair")) && x.tag != "armor" && x.tag != "1girl"));
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
    fn replace_can_remove_and_inject_does_not_duplicate_weighted_tags() {
        let acts = |t: &PromptToken| is_action(t);
        // an explicit replace with nothing removes the actions (Clear, emptying a character card)
        assert_eq!(replace("1girl, red hair, sitting, holding cup", &[], &acts), "1girl, red hair");
        // inject leaves a box alone when the reference brings nothing
        assert_eq!(inject("1girl, sitting", &[], &acts), "1girl, sitting");
        // a weighted tag is the same tag
        assert_eq!(inject("1girl, (beach:1.2), sitting", &["beach".into(), "standing".into()], &acts), "1girl, (beach:1.2), standing");
        assert_eq!(inject("1girl", &["Sitting".into(), "sitting".into(), "(sitting:1.1)".into()], &acts), "1girl, Sitting");
        // facial expressions are not poses, so a reference never overwrites them
        for face in ["winking", "pouting", "smirking", "blushing", "smiling"] {
            assert!(!is_action(&parse_prompt(face)[0]), "{face}");
        }
        assert!(is_action(&parse_prompt("sitting")[0]) && is_action(&parse_prompt("hands on hips")[0]));
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

    #[test]
    fn update_shared_scene_prompt_clears_duplicates_and_preserves_unique() {
        let c1 = "1girl, brown hair, gym shirt, blue shorts, standing, looking at viewer";
        let c2 = "1girl, green hair, dress, from behind, looking at viewer";
        let existing = "school bag, from behind, striped background, full body, gym, standing";
        let new_tags = vec!["gymnasium".to_string(), "polished floor".to_string(), "looking at viewer".to_string()];

        let updated = update_shared_scene_prompt(existing, &new_tags, &[c1.to_string(), c2.to_string()]);
        // "from behind", "standing", "looking at viewer" must be cleared because they exist in character prompts:
        assert!(!updated.contains("from behind"), "{updated}");
        assert!(!updated.contains("standing"), "{updated}");
        assert!(!updated.contains("looking at viewer"), "{updated}");

        // Unique existing tags must be preserved:
        assert!(updated.contains("school bag"), "{updated}");
        assert!(updated.contains("striped background"), "{updated}");
        assert!(updated.contains("full body"), "{updated}");
        assert!(updated.contains("gym"), "{updated}");

        // New unique tags must be added:
        assert!(updated.contains("gymnasium"), "{updated}");
        assert!(updated.contains("polished floor"), "{updated}");
    }

    #[test]
    fn scene_body_shape_extraction_and_identity_filtering() {
        let scene_tags = "1girl, solo, curvy, large breasts, sitting on bed, bedroom, soft lighting";
        let char_tags = "1girl, solo, tall, petite, flat chest, blonde hair, long hair, blue dress";

        // Without shape extraction:
        let a_off = from_single_with_shape(scene_tags, false, false);
        assert!(!a_off.iter().any(|a| a.tag == "curvy" || a.tag == "large breasts"));

        // With shape extraction:
        let a_on = from_single_with_shape(scene_tags, false, true);
        assert!(a_on.iter().any(|a| a.tag == "curvy" && a.owner == Owner::Char(0)));
        assert!(a_on.iter().any(|a| a.tag == "large breasts" && a.owner == Owner::Char(0)));

        // Character identity filtering with shape drop:
        let char_filtered = crate::scene::filter_identity_only_with_shape(char_tags, true);
        assert!(!char_filtered.contains("flat chest"));
        assert!(char_filtered.contains("tall"));
        assert!(char_filtered.contains("blonde hair"));
        assert!(char_filtered.contains("blue dress"));
    }

}
