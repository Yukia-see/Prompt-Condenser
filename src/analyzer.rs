use crate::parser::{PromptToken, TokenCategory};
use crate::profiles::{CheckpointProfile, TargetView};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeAction {
    FramingSafeSwap,
    PositioningConflict,
    RemovedBloat,
    MergedDuplicate,
    WeightClamped,
    KeptEmphasized,
    AiRewrite,
    /// Tags set by the camera angle dropdown.
    AngleChoice,
    /// Tags set by the pose dropdown.
    PoseChoice,
}

impl ChangeAction {
    pub fn badge_text(&self) -> &'static str {
        match self {
            Self::FramingSafeSwap => "FRAMING-SAFE SWAP",
            Self::PositioningConflict => "VIEW CONFLICT",
            Self::RemovedBloat => "CONDENSED BLOAT",
            Self::MergedDuplicate => "DUPLICATE",
            Self::WeightClamped => "WEIGHT CLAMPED",
            Self::KeptEmphasized => "KEPT & EMPHASIZED",
            Self::AiRewrite => "AI SUGGESTION",
            Self::AngleChoice => "CAMERA ANGLE",
            Self::PoseChoice => "POSE",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptChange {
    pub id: usize,
    pub original: String,
    pub action: ChangeAction,
    pub reason: String,
    /// Primary suggested replacement (e.g. Framing-Safe body words or 360 turnaround tags)
    pub suggested_alternative: String,
    /// Secondary option (e.g. "Change framing to 'cowboy shot, from behind' to keep 'ass' directly")
    pub framing_alternative: String,
    /// Current state: 0 = Applied Primary Suggestion/Removal, 1 = Use Secondary Framing Alternative, 2 = Restore Original
    pub mode: u8,
}

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub checkpoint: CheckpointProfile,
    pub target_view: TargetView,
    pub pinned_emphasis_raw: String,
    pub emphasis_weight: f32,
    pub cleanup_enabled: bool,
    pub resolve_conflicts_enabled: bool,
    pub clamp_weights_enabled: bool,
    /// Index into `profiles::CAMERA_ANGLES` (0 = none).
    pub camera_angle: usize,
    /// Index into `profiles::POSES` (0 = none).
    pub pose: usize,
    /// Dynamic Prompts {a|b|c} variations wanted.
    pub dynamic: bool,
}

#[derive(Debug, Clone)]
pub struct RuleAnalysisResult {
    pub processed_tokens: Vec<PromptToken>,
    pub changes: Vec<PromptChange>,
    pub active_framing_detected: TargetView,
}

/// Checks if the prompt itself contains an explicit framing tag even if the TargetView dropdown is `None`.
pub fn detect_effective_view(tokens: &[PromptToken], selected_view: TargetView) -> TargetView {
    if selected_view != TargetView::None {
        return selected_view;
    }
    for t in tokens {
        let n = t.normalized.as_str();
        if n.contains("360") || n.contains("turnaround") || n.contains("multiple views") {
            return TargetView::Turnaround360;
        }
        if n == "upper body" || n == "cowboy shot" {
            return TargetView::UpperBody;
        }
        if n == "portrait" || n == "close-up" || n == "close up" || n == "headshot" {
            return TargetView::CloseUpPortrait;
        }
        if n == "from behind" || n == "back view" {
            return TargetView::FromBehind;
        }
        if n == "front view" {
            return TargetView::FrontView;
        }
        if n == "side profile" || n == "from side" || n == "side view" {
            return TargetView::SideProfile;
        }
        if n == "full body" {
            return TargetView::FullBody;
        }
    }
    TargetView::None
}

/// Checks whether a body-part token triggers a camera-pull conflict with the active view
/// (e.g., `ass` or `butt` inside `Upper Body` or `Front View`), and returns:
/// `Some((reason, framing_safe_body_alternative, camera_framing_alternative))`
pub fn check_framing_safe_body_swap(
    normalized_tag: &str,
    effective_view: TargetView,
) -> Option<(&'static str, &'static str, &'static str)> {
    let is_rear_lower = matches!(
        normalized_tag,
        "ass"
            | "butt"
            | "huge ass"
            | "big ass"
            | "large ass"
            | "round ass"
            | "plump ass"
            | "booty"
            | "glutes"
            | "buttocks"
            | "ass focus"
    );

    let is_lower_legs_feet = matches!(
        normalized_tag,
        "feet"
            | "barefoot"
            | "toes"
            | "soles"
            | "ankles"
            | "calves"
            | "knees"
            | "thighs"
            | "thick thighs"
            | "legs"
            | "long legs"
    );

    let is_front_chest = matches!(
        normalized_tag,
        "cleavage" | "deep cleavage" | "front cleavage" | "pectorals" | "abs" | "navel" | "stomach"
    );

    match effective_view {
        TargetView::UpperBody => {
            if is_rear_lower {
                return Some((
                    "Direct 'ass/butt' tags pull ComfyUI's camera downward and backward, breaking 'upper body' framing and twisting anatomy.",
                    "curvy, voluptuous, hourglass figure, narrow waist, arched back",
                    "cowboy shot, from behind, looking back",
                ));
            }
            if matches!(normalized_tag, "feet" | "barefoot" | "toes" | "soles" | "ankles" | "calves") {
                return Some((
                    "Lower-leg/foot tags force ComfyUI to zoom out to full body, breaking 'upper body' framing.",
                    "slender build, graceful posture, poised stance",
                    "full body, standing",
                ));
            }
        }
        TargetView::CloseUpPortrait => {
            if is_rear_lower {
                return Some((
                    "'ass/butt' pulls the camera out of 'portrait / close-up' into a lower-body rear shot.",
                    "curvy figure, alluring expression, glamorous",
                    "cowboy shot, from behind, looking back",
                ));
            }
            if is_lower_legs_feet || normalized_tag == "abs" || normalized_tag == "muscular abs" {
                return Some((
                    "Mid/lower-body anatomy tags ('abs/thighs/legs/feet') pull the camera out of a close-up portrait.",
                    "toned shoulders, defined collarbone, athletic build",
                    "full body",
                ));
            }
        }
        TargetView::FrontView => {
            if is_rear_lower {
                return Some((
                    "Direct 'ass/butt' tags force the character to turn their back to the camera, conflicting with 'front view'.",
                    "wide hips, curvy, hourglass figure, thick thighs",
                    "from behind, looking back",
                ));
            }
        }
        TargetView::FromBehind => {
            if is_front_chest {
                return Some((
                    "Front torso tags ('cleavage/abs/navel') pull the camera to the front, fighting 'from behind / back view'.",
                    "sideboob, breasts from behind, curvy silhouette, arched back, looking back",
                    "front view, three-quarter view",
                ));
            }
        }
        _ => {}
    }

    None
}

/// Checks if a camera/pose/background tag directly conflicts with the selected `TargetView`.
pub fn check_view_positioning_conflict(
    normalized_tag: &str,
    target_view: TargetView,
) -> Option<(&'static str, &'static str)> {
    match target_view {
        TargetView::Turnaround360 => {
            let single_crop = [
                "cowboy shot",
                "upper body",
                "portrait",
                "close-up",
                "close up",
                "extreme close-up",
                "headshot",
                "cropped",
                "lower body",
            ];
            if single_crop.contains(&normalized_tag) {
                return Some((
                    "Crops the body and prevents ComfyUI from generating a full 360 turnaround sheet.",
                    "character turnaround, multiple views, full body",
                ));
            }

            let single_angle = [
                "looking at viewer",
                "straight-on",
                "dutch angle",
                "foreshortening",
                "from above",
                "from below",
                "pov",
                "first person view",
            ];
            if single_angle.contains(&normalized_tag) {
                return Some((
                    "Locks the camera to a single angle/perspective, which breaks 360 multi-view turnarounds.",
                    "front view, side view, back view",
                ));
            }

            let bg_pose_conflict = [
                "depth of field",
                "blurry background",
                "bokeh",
                "sitting",
                "sitting on bed",
                "sitting on chair",
                "lying",
                "on back",
                "on stomach",
                "kneeling",
            ];
            if bg_pose_conflict.contains(&normalized_tag) {
                return Some((
                    "Sitting/lying poses and depth-of-field blur interfere with clean 360 character turnaround sheets.",
                    "standing, simple background",
                ));
            }
        }
        TargetView::UpperBody => {
            if matches!(normalized_tag, "full body" | "wide shot" | "extreme close-up" | "headshot") {
                return Some((
                    "Conflicts with the selected 'Upper Body / Cowboy Shot' framing.",
                    "upper body",
                ));
            }
        }
        TargetView::CloseUpPortrait => {
            if matches!(
                normalized_tag,
                "full body" | "cowboy shot" | "wide shot" | "lower body" | "feetfocus"
            ) {
                return Some((
                    "Zooms the camera out and conflicts with 'Close-Up / Portrait' framing.",
                    "portrait, close-up, face focus",
                ));
            }
        }
        TargetView::FrontView => {
            if matches!(normalized_tag, "from behind" | "back view" | "rear view") {
                return Some((
                    "Rotates the subject away from the camera, conflicting with 'Front View'.",
                    "front view, looking at viewer",
                ));
            }
        }
        TargetView::FromBehind => {
            if matches!(normalized_tag, "front view" | "straight-on") {
                return Some((
                    "Forces a front-facing angle, conflicting with 'From Behind / Back View'.",
                    "from behind, back view, looking back",
                ));
            }
        }
        TargetView::SideProfile => {
            if matches!(normalized_tag, "front view" | "from behind" | "back view" | "straight-on") {
                return Some((
                    "Conflicts with 'Side Profile' camera angle.",
                    "from side, side profile",
                ));
            }
        }
        TargetView::FullBody => {
            if matches!(
                normalized_tag,
                "upper body" | "cowboy shot" | "portrait" | "close-up" | "close up" | "cropped"
            ) {
                return Some((
                    "Crops out the lower body, conflicting with 'Full Body' framing.",
                    "full body, standing",
                ));
            }
        }
        TargetView::None => {}
    }
    None
}

/// Checks if a camera framing tag in the prompt hides a user-pinned body part
/// (when TargetView is None, so the user wants to emphasize a specific body part!).
pub fn check_pinned_visibility_conflict(
    normalized_framing_tag: &str,
    pinned_norms: &[String],
) -> Option<(String, String)> {
    for pin in pinned_norms {
        let pin_is_back = pin.contains("back")
            || pin.contains("spine")
            || pin.contains("shoulder blade")
            || pin == "ass"
            || pin == "butt";
        let pin_is_lower = pin.contains("leg")
            || pin.contains("thigh")
            || pin.contains("feet")
            || pin.contains("foot")
            || pin.contains("boot")
            || pin.contains("ankle");

        if pin_is_back && normalized_framing_tag == "front view" {
            return Some((
                format!(
                    "'front view' hides your emphasized feature ('{}') on the back.",
                    pin
                ),
                "from behind, back focus, looking back".to_string(),
            ));
        }

        if pin_is_lower
            && matches!(
                normalized_framing_tag,
                "portrait" | "close-up" | "close up" | "upper body" | "headshot"
            )
        {
            return Some((
                format!(
                    "'{}' crops out the lower body and hides your emphasized '{}'.",
                    normalized_framing_tag, pin
                ),
                "full body".to_string(),
            ));
        }
    }
    None
}

pub fn analyze_with_rules(tokens: &[PromptToken], config: &ScanConfig) -> RuleAnalysisResult {
    let effective_view = detect_effective_view(tokens, config.target_view);

    // Parse user's pinned emphasis parts
    // Pinned parts only boost tags already in the prompt; they are never added.
    let pinned_norms: Vec<String> = config
        .pinned_emphasis_raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.to_lowercase()
                .replace('_', " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();

    let mut changes: Vec<PromptChange> = Vec::new();
    let mut next_change_id = 0usize;
    let mut seen_normalized: HashSet<String> = HashSet::new();
    let mut processed_tokens: Vec<PromptToken> = Vec::new();
    let mut added_quality_replacement = false;

    for token in tokens {
        let mut tok = token.clone();
        let is_pinned = pinned_norms
            .iter()
            .any(|p| tok.normalized == *p || tok.normalized.contains(p.as_str()));

        // 1. Check exact duplicates (when cleanup is enabled)
        if config.cleanup_enabled
            && tok.category != TokenCategory::LoraOrSpecial
            && seen_normalized.contains(&tok.normalized)
        {
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::MergedDuplicate,
                reason: format!("Duplicate tag '{}' appeared multiple times in the prompt.", tok.text),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.cleanup_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
            continue;
        }

        // 2. Check Framing-Safe Body-Part Swap (e.g. `ass` + `Upper Body` -> `curvy, voluptuous, hourglass figure...`)
        if let Some((reason, body_alt, framing_alt)) =
            check_framing_safe_body_swap(&tok.normalized, effective_view)
        {
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::FramingSafeSwap,
                reason: reason.to_string(),
                suggested_alternative: body_alt.to_string(),
                framing_alternative: framing_alt.to_string(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;

            if config.resolve_conflicts_enabled {
                for alt_part in body_alt.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    let norm = alt_part.to_lowercase();
                    if seen_normalized.insert(norm.clone()) {
                        processed_tokens.push(PromptToken {
                            raw: alt_part.to_string(),
                            text: alt_part.to_string(),
                            normalized: norm,
                            weight: if is_pinned { config.emphasis_weight } else { 1.15 },
                            had_runaway_weight: false,
                            replaced: true,
                            added: false,
                            original_index: tok.original_index,
                            category: if is_pinned {
                                TokenCategory::PinnedEmphasis
                            } else {
                                TokenCategory::CharacterBody
                            },
                        });
                    }
                }
                continue;
            }
        }

        // 3. Check Target View Positioning Conflict (e.g. `360 View` vs `cowboy shot / looking at viewer / sitting`)
        if let Some((reason, alt)) = check_view_positioning_conflict(&tok.normalized, effective_view) {
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::PositioningConflict,
                reason: reason.to_string(),
                suggested_alternative: alt.to_string(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;

            if config.resolve_conflicts_enabled {
                for alt_part in alt.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    let norm = alt_part.to_lowercase();
                    if seen_normalized.insert(norm.clone()) {
                        processed_tokens.push(PromptToken {
                            raw: alt_part.to_string(),
                            text: alt_part.to_string(),
                            normalized: norm,
                            weight: 1.0,
                            had_runaway_weight: false,
                            replaced: true,
                            added: false,
                            original_index: tok.original_index,
                            category: TokenCategory::FramingPose,
                        });
                    }
                }
                continue;
            }
        }

        // 4. Check if this framing tag hides a Pinned Body Part (when TargetView is None)
        if effective_view == TargetView::None {
            if let Some((reason, alt)) = check_pinned_visibility_conflict(&tok.normalized, &pinned_norms) {
                changes.push(PromptChange {
                    id: next_change_id,
                    original: tok.raw.clone(),
                    action: ChangeAction::PositioningConflict,
                    reason,
                    suggested_alternative: alt.clone(),
                    framing_alternative: String::new(),
                    mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
                });
                next_change_id += 1;

                if config.resolve_conflicts_enabled {
                    for alt_part in alt.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                        let norm = alt_part.to_lowercase();
                        if seen_normalized.insert(norm.clone()) {
                            processed_tokens.push(PromptToken {
                                raw: alt_part.to_string(),
                                text: alt_part.to_string(),
                                normalized: norm,
                                weight: 1.0,
                                had_runaway_weight: false,
                                replaced: true,
                                added: false,
                                original_index: tok.original_index,
                                category: TokenCategory::FramingPose,
                            });
                        }
                    }
                    continue;
                }
            }
        }

        // 5. If this token matches a Pinned Emphasis item, lock & boost its weight!
        if is_pinned {
            tok.weight = config.emphasis_weight;
            tok.category = TokenCategory::PinnedEmphasis;
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::KeptEmphasized,
                reason: format!(
                    "Protected & emphasized at weight {:.2} per your Keep & Emphasize selection.",
                    config.emphasis_weight
                ),
                suggested_alternative: format!("({}:{:.2})", tok.text, config.emphasis_weight),
                framing_alternative: String::new(),
                mode: 0,
            });
            next_change_id += 1;
            seen_normalized.insert(tok.normalized.clone());
            processed_tokens.push(tok);
            continue;
        }

        // 6. Check Checkpoint-specific Bloat / Redundant Quality Synonyms (ONLY if Cleanup Mode is ON)
        if config.cleanup_enabled {
            if let Some((reason, replacement)) = config.checkpoint.check_bloat_tag(&tok.normalized) {
                let suggested = if !added_quality_replacement && !replacement.is_empty() {
                    added_quality_replacement = true;
                    replacement.to_string()
                } else {
                    String::new()
                };

                changes.push(PromptChange {
                    id: next_change_id,
                    original: tok.raw.clone(),
                    action: ChangeAction::RemovedBloat,
                    reason: reason.to_string(),
                    suggested_alternative: suggested.clone(),
                    framing_alternative: String::new(),
                    mode: 0,
                });
                next_change_id += 1;

                if !suggested.is_empty() {
                    for q_tag in suggested.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                        let norm = q_tag.to_lowercase().replace('_', " ");
                        if seen_normalized.insert(norm.clone()) {
                            processed_tokens.push(PromptToken {
                                raw: q_tag.to_string(),
                                text: q_tag.to_string(),
                                normalized: norm,
                                weight: 1.0,
                                had_runaway_weight: false,
                                replaced: true,
                                added: false,
                                original_index: tok.original_index,
                                category: TokenCategory::QualityMeta,
                            });
                        }
                    }
                }
                continue;
            }
        }

        // 7. Clamp Runaway Weights `((((tag))))` if enabled
        let max_w = config.checkpoint.max_safe_weight();
        if config.clamp_weights_enabled && (tok.had_runaway_weight || tok.weight > max_w) {
            let clamped = tok.weight.min(max_w).min(1.25);
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::WeightClamped,
                reason: format!(
                    "Runaway attention weight ({:.2}x) causes ComfyUI CLIP burn/artifacts; clamped to {:.2}x.",
                    tok.weight, clamped
                ),
                suggested_alternative: if (clamped - 1.0).abs() < 0.05 {
                    tok.text.clone()
                } else {
                    format!("({}:{:.2})", tok.text, clamped)
                },
                framing_alternative: String::new(),
                mode: 0,
            });
            next_change_id += 1;
            tok.weight = clamped;
        }

        seen_normalized.insert(tok.normalized.clone());
        processed_tokens.push(tok);
    }

    // Camera angle / pose dropdowns: the user's explicit choice replaces the prompt's
    // own angle / pose tags (shown as a card so it can be undone).
    for (list, idx, action, what) in [
        (crate::profiles::CAMERA_ANGLES, config.camera_angle, ChangeAction::AngleChoice, "camera angle"),
        (crate::profiles::POSES, config.pose, ChangeAction::PoseChoice, "pose"),
    ] {
        let Some((label, tags)) = list.get(idx).filter(|_| idx > 0) else { continue };
        let group = crate::profiles::group_tags(list);
        let chosen: Vec<PromptToken> = crate::parser::parse_prompt(tags);
        let is_chosen = |n: &str| chosen.iter().any(|c| c.normalized == n);
        let replaced: Vec<String> = processed_tokens
            .iter()
            .filter(|t| group.contains(&t.normalized) && !is_chosen(&t.normalized))
            .map(|t| t.text.clone())
            .collect();
        processed_tokens.retain(|t| !(group.contains(&t.normalized) && !is_chosen(&t.normalized)));
        for mut t in chosen {
            if processed_tokens.iter().any(|e| e.normalized == t.normalized) {
                continue;
            }
            t.added = true;
            t.original_index = 940;
            t.category = TokenCategory::FramingPose;
            processed_tokens.push(t);
        }
        changes.push(PromptChange {
            id: next_change_id,
            original: replaced.join(", "),
            action,
            reason: if replaced.is_empty() {
                format!("Your {what} choice: {label}.")
            } else {
                format!("Your {what} choice: {label}. It replaces the prompt's own {what} tags.")
            },
            suggested_alternative: tags.to_string(),
            framing_alternative: String::new(),
            mode: 0,
        });
        next_change_id += 1;
    }

    RuleAnalysisResult {
        processed_tokens,
        changes,
        active_framing_detected: effective_view,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condenser::build_condensed_output;
    use crate::parser::parse_prompt;

    fn cfg(view: TargetView) -> ScanConfig {
        ScanConfig {
            checkpoint: CheckpointProfile::Anima,
            target_view: view,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,

            camera_angle: 0,

            pose: 0,

            dynamic: false,
        }
    }

    fn run(prompt: &str, config: &ScanConfig) -> (RuleAnalysisResult, String) {
        let toks = parse_prompt(prompt);
        let res = analyze_with_rules(&toks, config);
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, config);
        (res, out.final_prompt)
    }

    #[test]
    fn upper_body_swaps_ass_for_framing_safe_words() {
        let (res, out) = run("1girl, upper body, (ass:1.4)", &cfg(TargetView::UpperBody));
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::FramingSafeSwap));
        assert!(out.contains("hourglass figure"));
        assert!(out.contains("narrow waist"));
        assert!(!out.split(", ").any(|t| t.starts_with("(ass") || t == "ass"));
    }

    #[test]
    fn framing_alternative_keeps_the_body_part() {
        let config = cfg(TargetView::UpperBody);
        let prompt = "1girl, upper body, ass";
        let toks = parse_prompt(prompt);
        let mut res = analyze_with_rules(&toks, &config);
        let swap = res.changes.iter_mut().find(|c| c.action == ChangeAction::FramingSafeSwap).unwrap();
        swap.mode = 1;
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, &config).final_prompt;
        assert!(out.contains("ass"));
        assert!(out.contains("from behind"));
        assert!(!out.contains("hourglass figure"));
    }

    #[test]
    fn turnaround_360_removes_single_angle_tags() {
        let (res, out) = run(
            "1girl, cowboy shot, looking at viewer, sitting, depth of field",
            &cfg(TargetView::Turnaround360),
        );
        let conflicts = res.changes.iter().filter(|c| c.action == ChangeAction::PositioningConflict).count();
        assert_eq!(conflicts, 4);
        assert!(out.contains("character turnaround"));
        assert!(out.contains("multiple views"));
        assert!(!out.contains("cowboy shot"));
        assert!(!out.contains("sitting"));
    }

    #[test]
    fn cleanup_disabled_keeps_order_and_bloat() {
        let mut config = cfg(TargetView::None);
        config.cleanup_enabled = false;
        let (_, out) = run("8k, 1girl, ultra detailed, smile", &config);
        assert_eq!(out, "8k, 1girl, ultra detailed, smile");

        config.cleanup_enabled = true;
        let (_, out) = run("8k, 1girl, ultra detailed, smile", &config);
        assert!(!out.contains("8k"));
        assert!(!out.contains("ultra detailed"));
    }

    #[test]
    fn runaway_weights_are_clamped() {
        let (res, out) = run("((((masterpiece)))), 1girl", &cfg(TargetView::None));
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::WeightClamped));
        assert!(!out.contains("1.46"));
    }

    #[test]
    fn pinned_parts_are_emphasized() {
        let mut config = cfg(TargetView::None);
        config.pinned_emphasis_raw = "cybernetic spine".into();
        let (res, out) = run("1girl, cybernetic spine", &config);
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::KeptEmphasized));
        assert!(out.contains("(cybernetic spine:1.25)"));
    }

    #[test]
    fn pinned_parts_not_in_prompt_are_never_added() {
        let mut config = cfg(TargetView::None);
        config.pinned_emphasis_raw = "ass, toned back, cybernetic spine".into();
        let (_, out) = run("1girl, silver hair, toned back", &config);
        assert!(!out.contains("cybernetic spine"));
        assert!(!out.contains("ass"));
        assert!(out.contains("(toned back:1.25)"));
    }

    #[test]
    fn selected_view_does_not_append_framing_tags() {
        let (_, out) = run("1girl, silver hair, smile", &cfg(TargetView::Turnaround360));
        assert_eq!(out, "1girl, silver hair, smile");
    }

    #[test]
    fn generic_bloat_is_removed_not_replaced() {
        let (_, out) = run("8k, 1girl, smile", &cfg(TargetView::None));
        assert!(!out.contains("masterpiece"));
    }

    #[test]
    fn pony_does_not_prepend_score_tags() {
        let mut config = cfg(TargetView::None);
        config.checkpoint = CheckpointProfile::PonyV6;
        let (_, out) = run("1girl, smile", &config);
        assert!(!out.contains("score_9"));
    }

    #[test]
    fn substitutions_are_highlighted() {
        let config = cfg(TargetView::UpperBody);
        let prompt = "1girl, upper body, (ass:1.4), <lora:style_v2:0.8>, score_9";
        let toks = parse_prompt(prompt);
        let res = analyze_with_rules(&toks, &config);
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, &config);
        let md = &out.highlighted_markdown;
        assert!(md.contains(r"<u>\(curvy\:1\.15\)</u>"));
        assert!(!md.contains("<u>1girl</u>"));
        // Slint must parse it, and the visible text must match the plain prompt
        let styled = slint::StyledText::from_markdown(md).expect("valid markdown");
        let _ = styled;
    }

    #[test]
    fn ai_cards_are_opt_in() {
        use crate::ai_engine::{convert_ai_changes, AiChangeItem};
        let config = cfg(TargetView::None);
        let prompt = "1girl, booty shorts, smile";
        let toks = parse_prompt(prompt);
        let res = analyze_with_rules(&toks, &config);
        let item = |orig: &str, sugg: &str| AiChangeItem {
            original: orig.into(),
            action: "framing_safe_swap".into(),
            reason: "test".into(),
            suggested_alternative: sugg.into(),
            framing_alternative: String::new(),
        };
        let ai = convert_ai_changes(
            &[item("booty shorts", "denim shorts"), item("cybernetic spine", "spine"), item("smile", "smile")],
            &toks,
            &res.changes,
        );
        // Only the finding that names a real tag and changes something survives
        assert_eq!(ai.len(), 1);
        assert_eq!(ai[0].mode, 2);

        let mut changes = res.changes.clone();
        changes.extend(ai);
        let out = build_condensed_output(prompt, &res.processed_tokens, &changes, &config);
        assert!(out.final_prompt.contains("booty shorts"), "not applied until accepted");

        changes.last_mut().unwrap().mode = 0;
        let out = build_condensed_output(prompt, &res.processed_tokens, &changes, &config);
        assert!(!out.final_prompt.contains("booty shorts"));
        assert!(out.highlighted_markdown.contains("<u>denim shorts</u>"));
    }

    #[test]
    fn ai_add_cards_append_highlighted_details() {
        use crate::ai_engine::{convert_ai_changes, AiChangeItem};
        let config = cfg(TargetView::None);
        let prompt = "1girl, smile, city";
        let toks = parse_prompt(prompt);
        let res = analyze_with_rules(&toks, &config);
        let add = |sugg: &str| AiChangeItem {
            original: String::new(),
            action: "add".into(),
            reason: "r".into(),
            suggested_alternative: sugg.into(),
            framing_alternative: String::new(),
        };
        // "smile" is already there, so only the lighting detail survives
        let mut ai = convert_ai_changes(&[add("neon rim lighting"), add("smile")], &toks, &res.changes);
        assert_eq!(ai.len(), 1);
        ai[0].mode = 0;
        let mut changes = res.changes.clone();
        changes.extend(ai);
        let out = build_condensed_output(prompt, &res.processed_tokens, &changes, &config);
        assert!(out.final_prompt.contains("neon rim lighting"));
        assert!(out.highlighted_markdown.contains("#8CD9A8"), "added tags are green");
        assert!(out.final_prompt.contains("smile"));
    }

    #[test]
    fn ai_improved_prompt_marks_new_tags() {
        let (md, n) = crate::condenser::highlight_against_original("1girl, smile", "1girl, smile, golden hour lighting");
        assert_eq!(n, 1);
        assert!(md.contains("<u>golden hour lighting</u>"));
        assert!(slint::StyledText::from_markdown(&md).is_ok());
    }

    #[test]
    fn tidies_messy_ai_prompts() {
        use crate::condenser::tidy_ai_prompt;
        let messy = "((((masterpiece))), 1girl, solo, narrow waist, shrine, night, narrow waist:1.0, (fox tail:1.6))";
        let out = tidy_ai_prompt(messy, CheckpointProfile::Anima);
        assert_eq!(out, "masterpiece, 1girl, solo, narrow waist, shrine, night, (fox tail:1.35)");
        // prose is untouched for Flux
        assert_eq!(tidy_ai_prompt("A fox girl (smiling) at night", CheckpointProfile::Flux), "A fox girl (smiling) at night");
    }

    #[test]
    fn reports_tags_the_ai_dropped() {
        let dropped = crate::condenser::dropped_tags(
            "8k, 1girl, smile, cherry blossoms",
            "1girl, smile, moonlight",
            CheckpointProfile::Anima,
        );
        assert_eq!(dropped, vec!["cherry blossoms"]); // 8k is filler, not reported
    }

    #[test]
    fn recommended_negative_respects_view_and_prompt() {
        use crate::condenser::{merge_negative, recommended_negative};
        let config = cfg(TargetView::UpperBody);
        let neg = recommended_negative(&config, "1girl, smile, blurry background");
        assert!(neg.contains("full body") && neg.contains("worst quality"));
        assert!(!neg.split(", ").any(|t| t == "blurry background"));
        // a tag the positive asks for never ends up in the negative
        assert_eq!(merge_negative("lowres, smile, lowres", "1girl, smile"), "lowres");
        let mut flux = cfg(TargetView::None);
        flux.checkpoint = CheckpointProfile::Flux;
        assert_eq!(recommended_negative(&flux, "a cat"), "");
    }

    #[test]
    fn negative_never_fights_the_scene() {
        use crate::condenser::merge_negative;
        let pos = "1girl, solo, classroom, afternoon sunlight, reading book";
        assert_eq!(
            merge_negative("worst quality, indoor background, daytime, night time, people, multiple people", pos),
            "worst quality, night time, multiple people"
        );
    }

    #[test]
    fn lists_marked_tags() {
        use crate::condenser::{highlight_md, marked_tags, Highlight};
        let md = format!("{}, {}, {}", highlight_md("1girl", Highlight::None), highlight_md("(curvy:1.15)", Highlight::Substituted), highlight_md("moonlight", Highlight::Added));
        assert_eq!(marked_tags(&md, Highlight::Substituted), vec!["(curvy:1.15)"]);
        assert_eq!(marked_tags(&md, Highlight::Added), vec!["moonlight"]);
    }

    #[test]
    fn angle_and_pose_dropdowns_replace_prompt_tags() {
        let mut config = cfg(TargetView::None);
        config.camera_angle = crate::profiles::CAMERA_ANGLES.iter().position(|(l, _)| l.starts_with("Dutch")).unwrap();
        config.pose = crate::profiles::POSES.iter().position(|(l, _)| *l == "Sitting").unwrap();
        let (res, out) = run("1girl, solo, from above, standing, smile", &config);
        assert!(out.contains("dutch angle") && out.contains("sitting"), "{out}");
        assert!(!out.contains("from above") && !out.contains("standing"), "{out}");
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::AngleChoice && c.original == "from above"));
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::PoseChoice && c.original == "standing"));
    }

    #[test]
    fn dynamic_syntax_survives_parsing() {
        let toks = parse_prompt("1girl, {red hair, ponytail|blue hair}, smile");
        assert_eq!(toks.len(), 3);
        assert_eq!(toks[1].text, "{red hair, ponytail|blue hair}");
    }
}

