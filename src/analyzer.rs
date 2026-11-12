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
    /// Two sizes / lengths for the same thing (medium + large breasts).
    SizeConflict,
    /// Incompatible poses or arm gestures.
    PoseConflict,
    /// Species purity, four-ears triggers, facial state conflicts, eyewear conflict.
    AnatomyConflict,
    /// 3D bleed tags on 2D anime checkpoints.
    StyleConflict,
    /// Recommended chain tags (clothing interactions, pose/physics, negative locks).
    Recommendation,
    /// Bust or frame set by the Bust / Frame dropdowns.
    ProportionChoice,
    /// A generic tag a more specific one already implies (`swimsuit` next to `bikini`).
    ImpliedTag,
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
            Self::SizeConflict => "SIZE CONFLICT",
            Self::PoseConflict => "POSE CONFLICT",
            Self::AnatomyConflict => "ANATOMY CONFLICT",
            Self::StyleConflict => "STYLE CONFLICT",
            Self::AiRewrite => "AI SUGGESTION",
            Self::AngleChoice => "CAMERA ANGLE",
            Self::PoseChoice => "POSE",
            Self::Recommendation => "FIT & DYNAMICS",
            Self::ProportionChoice => "BODY PROPORTION",
            Self::ImpliedTag => "IMPLIED TAG",
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
    /// Suggest fabric fit and dynamic poses for existing garments/anatomy.
    pub fit_suggestions: bool,
    /// Allow exaggerated/stylized cartoon proportions for shortstack (bypasses anatomy conflict rules, drops bad anatomy from negative).
    pub exaggerated_proportions: bool,
    /// Index into `profiles::CAMERA_ANGLES` (0 = none).
    pub camera_angle: usize,
    /// Index into `profiles::POSES` (0 = none).
    pub pose: usize,
    /// Index into `profiles::BUSTS` (0 = none).
    pub bust: usize,
    /// Index into `profiles::FRAMES` (0 = none).
    pub frame: usize,
    /// Dynamic Prompts {a|b|c} variations wanted.
    pub dynamic: bool,
    /// The LoRA box: tags put into the output word for word.
    pub lora_tags_raw: String,
    /// Show After as grouped lines in the best order (not for Flux prose).
    pub arranged: bool,
    /// Add natural-language sentences (only where the checkpoint understands them).
    pub natural: bool,
    /// The 2 Characters tab: Before is character 1 + the scene, plus a second prompt.
    pub two_char: bool,
    /// Character 2's prompt and LoRA box (2 Characters tab only).
    pub char2_prompt_raw: String,
    pub char2_lora_tags_raw: String,
    /// Put BREAK between the two characters (CLIP checkpoints only).
    pub char_break: bool,
    /// The Scene box (2 Characters tab): setting, camera, light, mood, interaction.
    /// Its tags are always shared, never one character's.
    pub scene_prompt_raw: String,
    /// Characters 3 and 4 (3 / 4 Characters tabs): (prompt, LoRA box).
    pub more_chars: Vec<(String, String)>,
    /// How much the deep scan may enrich the prompt.
    pub detail: crate::profiles::DetailLevel,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            checkpoint: CheckpointProfile::Anima,
            target_view: TargetView::None,
            pinned_emphasis_raw: String::new(),
            emphasis_weight: 1.25,
            cleanup_enabled: true,
            resolve_conflicts_enabled: true,
            clamp_weights_enabled: true,
            fit_suggestions: true,
            exaggerated_proportions: false,
            camera_angle: 0,
            pose: 0,
            bust: 0,
            frame: 0,
            dynamic: false,
            lora_tags_raw: String::new(),
            arranged: false,
            natural: false,
            two_char: false,
            char2_prompt_raw: String::new(),
            char2_lora_tags_raw: String::new(),
            char_break: false,
            scene_prompt_raw: String::new(),
            more_chars: Vec::new(),
            detail: crate::profiles::DetailLevel::Balanced,
        }
    }
}

impl ScanConfig {
    /// How many characters the tab has (1 on the 1 Character tab).
    pub fn char_count(&self) -> usize {
        if self.two_char { 2 + self.more_chars.len() } else { 1 }
    }

    /// Character `i`'s LoRA box (0-based).
    pub fn lora_of(&self, i: usize) -> &str {
        match i {
            0 => &self.lora_tags_raw,
            1 => &self.char2_lora_tags_raw,
            n => self.more_chars.get(n - 2).map(|c| c.1.as_str()).unwrap_or(""),
        }
    }

    /// Character `i`'s prompt for characters 2 and up (character 1's is the Before box).
    pub fn prompt_of(&self, i: usize) -> &str {
        match i {
            1 => &self.char2_prompt_raw,
            n if n >= 2 => self.more_chars.get(n - 2).map(|c| c.0.as_str()).unwrap_or(""),
            _ => "",
        }
    }
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
        if n == "front view" || n == "straight-on" || n == "straight on" || n == "facing viewer" {
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

/// Tags specifying small or flat chest.
pub fn is_small_chest_tag(tag: &str) -> bool {
    matches!(
        tag,
        "small breasts"
            | "flat chest"
            | "tiny breasts"
            | "petite breasts"
            | "perky breasts"
            | "aa cup"
            | "a cup"
            | "b cup"
            | "flat"
    )
}

/// Tags specifying a large or prominent bust.
#[allow(dead_code)]
pub fn is_large_chest_tag(tag: &str) -> bool {
    matches!(
        tag,
        "large breasts"
            | "huge breasts"
            | "gigantic breasts"
            | "hyper breasts"
            | "enormous breasts"
            | "massive breasts"
            | "full breasts"
            | "heavy breasts"
            | "e cup"
            | "f cup"
            | "g cup"
            | "h cup"
            | "i cup"
            | "oppai"
            | "busty"
    )
}

/// Chest exposure / cleavage tags that require breast volume.
#[allow(dead_code)]
pub fn is_cleavage_or_chest_exposure_tag(tag: &str) -> bool {
    matches!(
        tag,
        "cleavage"
            | "deep cleavage"
            | "underboob"
            | "sideboob"
            | "cleavage cutout"
    )
}

/// Poses that physically require prominent breast volume.
#[allow(dead_code)]
pub fn is_large_chest_pose_tag(tag: &str) -> bool {
    matches!(
        tag,
        "breast rest"
            | "resting breasts on table"
            | "breasts on table"
            | "holding breasts"
            | "holding own breasts"
            | "cupping breasts"
            | "breast grab"
            | "groping own breasts"
            | "grabbing own breasts"
            | "breast press"
            | "cleavage peek"
            | "paizuri"
            | "hand bra"
    )
}

/// Kemonomimi / animal ears target tags.
pub fn is_kemonomimi_tag(tag: &str) -> bool {
    matches!(
        tag,
        "kemonomimi mode"
            | "kemonomimi"
            | "animal ears"
            | "rabbit ears"
            | "bunny ears"
            | "fox ears"
            | "cat ears"
            | "dog ears"
            | "tanuki ears"
            | "wolf ears"
            | "horse ears"
            | "kitsune"
            | "neko"
    )
}

/// Furry bleed tags incompatible with human-skin kemonomimi.
pub fn is_furry_bleed_tag(tag: &str) -> bool {
    if matches!(
        tag,
        "furry"
            | "furry female"
            | "furry male"
            | "anthro"
            | "fur"
            | "white fur"
            | "pink fur"
            | "black fur"
            | "brown fur"
            | "grey fur"
            | "gray fur"
            | "orange fur"
            | "blue fur"
            | "snout"
            | "muzzle"
            | "animal nose"
            | "paws"
            | "paw pads"
            | "digitigrade"
            | "furry body"
            | "body fur"
    ) {
        return true;
    }
    if tag.ends_with(" fur") && tag.split_whitespace().count() == 2 {
        let first = tag.split_whitespace().next().unwrap();
        if COLOR_WORDS.contains(&first) {
            return true;
        }
    }
    false
}

/// Human ear / accessory tags that trigger four-ear generation artifacts on kemonomimi.
pub fn is_four_ears_trigger_tag(tag: &str) -> bool {
    if matches!(
        tag,
        "human ears"
            | "earrings"
            | "hoop earrings"
            | "stud earrings"
            | "ear piercing"
            | "ear piercings"
            | "hair tucked behind ear"
            | "elf ears"
            | "pointy ears"
            | "pointed ears"
    ) {
        return true;
    }
    if tag.contains("earring") || tag.contains("ear piercing") {
        return true;
    }
    false
}

/// Closed eyes tags.
pub fn is_closed_eyes_tag(tag: &str) -> bool {
    matches!(tag, "closed eyes" | "eyes closed" | "sleeping")
}

/// Gaze, pupil, half-closed, and eye color tags that force eyelids open.
pub fn is_closed_eyes_conflict_tag(tag: &str) -> bool {
    if matches!(
        tag,
        "looking at viewer"
            | "looking away"
            | "looking back"
            | "looking up"
            | "looking down"
            | "eye contact"
            | "staring"
            | "glaring"
            | "half-closed eyes"
            | "half closed eyes"
            | "jitome"
            | "tareme"
            | "tsurime"
            | "heterochromia"
    ) {
        return true;
    }
    if tag.contains("pupil") || tag.contains("sclera") {
        return true;
    }
    let eye_colors = [
        "red eyes", "blue eyes", "green eyes", "yellow eyes", "amber eyes",
        "purple eyes", "pink eyes", "brown eyes", "black eyes", "aqua eyes",
        "orange eyes", "grey eyes", "gray eyes",
    ];
    if eye_colors.contains(&tag) {
        return true;
    }
    if tag.ends_with(" eyes") && tag.split_whitespace().count() == 2 {
        let first = tag.split_whitespace().next().unwrap();
        if COLOR_WORDS.contains(&first) {
            return true;
        }
    }
    false
}

/// Closed mouth tags.
pub fn is_closed_mouth_tag(tag: &str) -> bool {
    matches!(
        tag,
        "closed mouth" | ":3" | "3" | "cat mouth" | "pout" | "tight lips"
    )
}

/// Mouth anatomy and expressions that force the mouth open.
pub fn is_closed_mouth_conflict_tag(tag: &str) -> bool {
    if matches!(
        tag,
        "open mouth"
            | "parted lips"
            | "slightly open mouth"
            | "uvula"
            | "saliva"
            | "drooling"
            | "laughing"
            | "shouting"
            | "yawning"
            | "sigh"
    ) {
        return true;
    }
    if tag.contains("teeth") || tag.contains("fang") || tag.contains("tongue") {
        return true;
    }
    false
}

/// Eyewear worn on the face.
pub fn is_face_eyewear_tag(tag: &str) -> bool {
    matches!(
        tag,
        "glasses"
            | "sunglasses"
            | "round eyewear"
            | "rimmed eyewear"
            | "semi-rimless eyewear"
            | "under-rim eyewear"
            | "pince-nez"
    )
}

/// Eyewear worn on the head or removed.
pub fn is_head_or_off_eyewear_tag(tag: &str) -> bool {
    matches!(
        tag,
        "eyewear on head"
            | "glasses on head"
            | "sunglasses on head"
            | "goggles on head"
            | "holding eyewear"
            | "removing eyewear"
            | "eyewear hang"
    )
}

/// Petite and short stature tags.
pub fn is_short_petite_tag(tag: &str) -> bool {
    matches!(tag, "short" | "petite" | "loli" | "small stature" | "chibi")
}

/// Tags that stretch proportions or props that hijack framing away from petite stature.
pub fn is_short_petite_conflict_tag(tag: &str) -> bool {
    matches!(
        tag,
        "tall"
            | "tall female"
            | "tall male"
            | "long legs"
            | "model body"
            | "mature female"
            | "milf"
            | "statuesque"
            | "towering"
            | "high heels"
            | "stilettos"
            | "huge weapon"
            | "giant weapon"
            | "polearm"
            | "staff"
            | "oversized weapon"
            | "scythe"
    )
}

/// Tall and statuesque stature tags.
pub fn is_tall_statuesque_tag(tag: &str) -> bool {
    matches!(
        tag,
        "tall" | "tall female" | "tall male" | "long legs" | "mature female" | "statuesque"
    )
}

/// Childlike and oversized clothing tags that compress head-to-body ratio.
pub fn is_tall_statuesque_conflict_tag(tag: &str) -> bool {
    if matches!(
        tag,
        "short"
            | "petite"
            | "loli"
            | "chibi"
            | "big head"
            | "round face"
            | "oversized shirt"
            | "oversized clothes"
            | "sleeves past fingers"
            | "sleeves past wrists"
    ) {
        return true;
    }
    if tag.starts_with("oversized ") {
        return true;
    }
    false
}

/// 3D and semi-realistic bleed tags that degrade 2D anime checkpoints.
pub fn is_3d_bleed_tag(tag: &str) -> bool {
    matches!(
        tag,
        "realistic"
            | "photorealistic"
            | "3d"
            | "blender"
            | "octane render"
            | "unreal engine"
            | "unreal engine 5"
            | "shiny skin"
            | "oily skin"
            | "pores"
            | "subsurface scattering"
            | "lips"
            | "nose nostrils"
    )
}

/// Dark skin and tanned skin tags.
pub fn is_dark_skin_tag(tag: &str) -> bool {
    matches!(
        tag,
        "dark skin"
            | "dark-skinned female"
            | "dark-skinned male"
            | "dark-skinned"
            | "tan"
            | "brown skin"
            | "tanned"
            | "tanned skin"
    )
}

/// Pale and bleached skin tags that wash out dark skin tones.
pub fn is_dark_skin_conflict_tag(tag: &str) -> bool {
    matches!(
        tag,
        "pale skin"
            | "fair skin"
            | "white skin"
            | "porcelain skin"
            | "albino"
            | "overexposed"
            | "bleached"
            | "pale-skinned"
            | "fair-skinned"
            | "light skin"
    )
}

/// Caution tags that risk washing out dark skin.
pub fn is_dark_skin_caution_tag(tag: &str) -> bool {
    matches!(tag, "tanlines" | "sunburn" | "full-face blush" | "full face blush")
}

/// Standard breast tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreastTier {
    FlatChest,
    SmallBreasts,
    MediumBreasts,
    LargeBreasts,
    HugeBreasts,
    GiganticBreasts,
}

pub fn detect_breast_tier(tag: &str) -> Option<BreastTier> {
    match tag {
        "flat chest" | "aa cup" | "flat" => Some(BreastTier::FlatChest),
        "small breasts" | "a cup" | "b cup" | "petite breasts" | "perky breasts" => {
            Some(BreastTier::SmallBreasts)
        }
        "medium breasts" | "c cup" | "d cup" | "average breasts" => {
            Some(BreastTier::MediumBreasts)
        }
        "large breasts" | "e cup" | "f cup" | "full breasts" => {
            Some(BreastTier::LargeBreasts)
        }
        "huge breasts" | "g cup" | "h cup" | "heavy breasts" => {
            Some(BreastTier::HugeBreasts)
        }
        "gigantic breasts"
        | "hyper breasts"
        | "enormous breasts"
        | "massive breasts"
        | "breasts bigger than head"
        | "i cup"
        | "oppai" => Some(BreastTier::GiganticBreasts),
        _ => None,
    }
}

pub fn breast_tier_conflicts(tier: BreastTier, tag: &str) -> Option<(&'static str, ChangeAction)> {
    match tier {
        BreastTier::FlatChest => {
            let is_size = matches!(
                tag,
                "breasts"
                    | "small breasts"
                    | "medium breasts"
                    | "large breasts"
                    | "huge breasts"
                    | "gigantic breasts"
                    | "hyper breasts"
                    | "a cup"
                    | "b cup"
                    | "c cup"
                    | "d cup"
                    | "e cup"
                    | "f cup"
                    | "g cup"
                    | "h cup"
                    | "voluptuous"
                    | "curvy"
                    | "busty"
            );
            if is_size {
                return Some((
                    "Removes all larger breast tiers, generic 'breasts' tag bias, and cleavage/physics tags that force breast volume.",
                    ChangeAction::SizeConflict,
                ));
            }
            let is_hijacker = matches!(
                tag,
                "cleavage"
                    | "deep cleavage"
                    | "underboob"
                    | "sideboob"
                    | "bouncing breasts"
                    | "jiggling"
                    | "breast press"
                    | "boob shelf"
                    | "breast rest"
                    | "breasts on table"
                    | "taut shirt"
                    | "button gap"
                    | "popping button"
                    | "paizuri"
                    | "hand bra"
                    | "underbust"
                    | "framed breasts"
                    | "strap between breasts"
                    | "seatbelt between breasts"
                    | "sagging breasts"
                    | "hanging breasts"
                    | "breast curtains"
                    | "bursting breasts"
                    | "boob pocket"
                    | "virgin killer sweater"
            );
            if is_hijacker {
                let action = if matches!(
                    tag,
                    "breast press" | "breast rest" | "breasts on table" | "paizuri" | "hand bra"
                ) {
                    ChangeAction::PoseConflict
                } else {
                    ChangeAction::SizeConflict
                };
                return Some((
                    "Removes all larger breast tiers, generic 'breasts' tag bias, and cleavage/physics tags that force breast volume.",
                    action,
                ));
            }
        }
        BreastTier::SmallBreasts => {
            let is_size = matches!(
                tag,
                "flat chest"
                    | "aa cup"
                    | "medium breasts"
                    | "large breasts"
                    | "huge breasts"
                    | "gigantic breasts"
                    | "hyper breasts"
                    | "c cup"
                    | "d cup"
                    | "e cup"
                    | "f cup"
                    | "g cup"
                    | "h cup"
                    | "voluptuous"
                    | "busty"
            );
            if is_size {
                return Some((
                    "Removes flat/larger tiers and heavy-physics tags that inflate small breasts to large.",
                    ChangeAction::SizeConflict,
                ));
            }
            let is_hijacker = matches!(
                tag,
                "cleavage"
                    | "deep cleavage"
                    | "underboob"
                    | "sideboob"
                    | "cleavage cutout"
                    | "cleavage peek"
                    | "boob shelf"
                    | "breast rest"
                    | "resting breasts on table"
                    | "breasts on table"
                    | "holding breasts"
                    | "holding own breasts"
                    | "cupping breasts"
                    | "breast grab"
                    | "groping own breasts"
                    | "grabbing own breasts"
                    | "breast press"
                    | "hand bra"
                    | "button gap"
                    | "popping button"
                    | "paizuri"
                    | "sagging breasts"
                    | "hanging breasts"
                    | "heavy breasts"
                    | "bursting breasts"
                    | "framed breasts"
                    | "boob pocket"
                    | "breasts bigger than head"
            );
            if is_hijacker {
                let action = if is_large_chest_pose_tag(tag)
                    || matches!(tag, "breast rest" | "breasts on table" | "paizuri" | "holding breasts" | "breast grab")
                {
                    ChangeAction::PoseConflict
                } else {
                    ChangeAction::SizeConflict
                };
                let reason = if action == ChangeAction::PoseConflict {
                    "Pose physically requires prominent breasts and forces the model to inflate breast size. Removes flat/larger tiers and heavy-physics tags that inflate small breasts to large."
                } else {
                    "Removes flat/larger tiers and heavy-physics tags that inflate small breasts to large."
                };
                return Some((reason, action));
            }
        }
        BreastTier::MediumBreasts => {
            let is_size = matches!(
                tag,
                "flat chest"
                    | "aa cup"
                    | "small breasts"
                    | "a cup"
                    | "b cup"
                    | "large breasts"
                    | "huge breasts"
                    | "gigantic breasts"
                    | "hyper breasts"
                    | "e cup"
                    | "f cup"
                    | "g cup"
                    | "h cup"
            );
            if is_size {
                return Some((
                    "Prevents extreme deflation (flat triggers) or extreme inflation (huge/shelf triggers).",
                    ChangeAction::SizeConflict,
                ));
            }
            let is_hijacker = matches!(
                tag,
                "boob shelf"
                    | "breast rest"
                    | "breasts on table"
                    | "popping button"
                    | "bursting breasts"
                    | "sagging breasts"
                    | "heavy breasts"
                    | "breasts bigger than head"
                    | "supporting own breasts"
                    | "visible ribs"
                    | "pectorals"
            );
            if is_hijacker {
                let action = if matches!(tag, "breast rest" | "breasts on table" | "supporting own breasts") {
                    ChangeAction::PoseConflict
                } else {
                    ChangeAction::SizeConflict
                };
                return Some((
                    "Prevents extreme deflation (flat triggers) or extreme inflation (huge/shelf triggers).",
                    action,
                ));
            }
        }
        BreastTier::LargeBreasts => {
            let is_size = matches!(
                tag,
                "flat chest"
                    | "aa cup"
                    | "small breasts"
                    | "a cup"
                    | "b cup"
                    | "medium breasts"
                    | "c cup"
                    | "d cup"
                    | "huge breasts"
                    | "gigantic breasts"
                    | "hyper breasts"
            );
            if is_size {
                return Some((
                    "Strips smaller/larger tiers and hyper-scaling tags.",
                    ChangeAction::SizeConflict,
                ));
            }
            let is_hijacker = matches!(
                tag,
                "visible ribs"
                    | "pectorals"
                    | "flat"
                    | "boyish"
                    | "androgynous"
                    | "breasts bigger than head"
                    | "breasts bigger than torso"
                    | "hyper"
            );
            if is_hijacker {
                return Some((
                    "Strips smaller/larger tiers and hyper-scaling tags.",
                    ChangeAction::SizeConflict,
                ));
            }
        }
        BreastTier::HugeBreasts => {
            let is_size = matches!(
                tag,
                "flat chest"
                    | "aa cup"
                    | "small breasts"
                    | "a cup"
                    | "b cup"
                    | "medium breasts"
                    | "c cup"
                    | "d cup"
                    | "large breasts"
                    | "e cup"
                    | "f cup"
            );
            if is_size {
                return Some((
                    "Strips smaller tiers and slender body tags that fight against huge breast rendering.",
                    ChangeAction::SizeConflict,
                ));
            }
            let is_hijacker = matches!(
                tag,
                "visible ribs"
                    | "pectorals"
                    | "perky breasts"
                    | "petite"
                    | "skinny"
                    | "slender"
                    | "boyish"
                    | "androgynous"
            );
            if is_hijacker {
                return Some((
                    "Strips smaller tiers and slender body tags that fight against huge breast rendering.",
                    ChangeAction::SizeConflict,
                ));
            }
        }
        BreastTier::GiganticBreasts => {
            let is_size = matches!(
                tag,
                "flat chest"
                    | "aa cup"
                    | "small breasts"
                    | "a cup"
                    | "b cup"
                    | "medium breasts"
                    | "c cup"
                    | "d cup"
                    | "large breasts"
                    | "e cup"
                    | "f cup"
                    | "proportionate"
            );
            if is_size {
                return Some((
                    "Strips smaller tiers and proportionate tags that fight against gigantic breast rendering.",
                    ChangeAction::SizeConflict,
                ));
            }
            let is_hijacker = matches!(
                tag,
                "visible ribs"
                    | "pectorals"
                    | "perky breasts"
                    | "petite"
                    | "skinny"
                    | "slender"
                    | "boyish"
                    | "androgynous"
            );
            if is_hijacker {
                return Some((
                    "Strips smaller tiers and slender body tags that fight against gigantic breast rendering.",
                    ChangeAction::SizeConflict,
                ));
            }
        }
    }
    None
}

/// Thick and curvy lower body tags.
pub fn is_thick_curvy_lower_tag(tag: &str) -> bool {
    matches!(
        tag,
        "thick thighs"
            | "wide hips"
            | "broad hips"
            | "childbearing hips"
            | "huge hips"
            | "pronounced hips"
            | "narrow waist wide hips"
            | "wide pelvis"
            | "curvy"
            | "plump thighs"
            | "bottom heavy"
            | "huge ass"
            | "pear shaped"
    )
}

/// Tags incompatible with thick and curvy lower body.
pub fn is_thick_curvy_conflict(tag: &str) -> bool {
    matches!(
        tag,
        "thin legs"
            | "slender legs"
            | "stick legs"
            | "thigh gap"
            | "narrow hips"
            | "boyish hips"
            | "flat hips"
            | "straight torso"
            | "tubular waist"
            | "tubular body"
            | "flat ass"
            | "small ass"
            | "skinny"
            | "lanky"
            | "frail"
    )
}

/// Slender and petite lower body tags.
pub fn is_slender_petite_lower_tag(tag: &str) -> bool {
    matches!(
        tag,
        "slender" | "skinny" | "thin legs" | "thigh gap" | "narrow hips" | "petite"
    )
}

/// Tags incompatible with slender and petite lower body.
pub fn is_slender_petite_conflict(tag: &str) -> bool {
    matches!(
        tag,
        "thick thighs"
            | "wide hips"
            | "broad hips"
            | "childbearing hips"
            | "huge hips"
            | "pronounced hips"
            | "narrow waist wide hips"
            | "curvy"
            | "voluptuous"
            | "plump"
            | "chubby"
            | "huge ass"
            | "bottom heavy"
            | "pear shaped"
            | "skindentation"
            | "thigh squish"
            | "muffin top"
            | "stomach bulge"
            | "belly"
    )
}

/// Muscular and toned physique tags.
pub fn is_muscular_toned_tag(tag: &str) -> bool {
    matches!(
        tag,
        "abs" | "toned" | "muscular" | "muscular female" | "six pack" | "athletic"
    )
}

/// Tags incompatible with muscular and toned physique.
pub fn is_muscular_conflict(tag: &str) -> bool {
    matches!(
        tag,
        "chubby"
            | "plump"
            | "soft body"
            | "belly"
            | "stomach bulge"
            | "pudge"
            | "muffin top"
            | "love handles"
            | "flabby"
            | "frail"
            | "skinny"
    )
}

/// Soft and chubby body tags.
pub fn is_soft_chubby_tag(tag: &str) -> bool {
    matches!(
        tag,
        "chubby" | "plump" | "soft body" | "belly" | "stomach bulge" | "muffin top"
    )
}

/// Tags incompatible with soft and chubby body.
pub fn is_soft_chubby_conflict(tag: &str) -> bool {
    matches!(
        tag,
        "abs"
            | "toned"
            | "muscular"
            | "six pack"
            | "obliques"
            | "narrow waist"
            | "slim waist"
            | "flat stomach"
            | "skinny"
            | "slender"
            | "visible ribs"
    )
}

/// Proportion tiers for chain tag recommendations, negative locking, and prompt reinforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProportionTier {
    FlatChest,
    SmallBreasts,
    MediumBreasts,
    LargeBreasts,
    HugeBreasts,
    GiganticBreasts,
    ThickThighsAndHips,
    MuscularFemale,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProportionRecommendations {
    pub tier: ProportionTier,
    pub primary_name: &'static str,
    pub core_reinforcers: &'static [&'static str],
    pub clothing_interaction: &'static [&'static str],
    pub pose_and_physics: &'static [&'static str],
    pub auto_negative_lock: &'static [&'static str],
}

pub fn get_proportion_recommendations(tier: ProportionTier) -> &'static ProportionRecommendations {
    match tier {
        ProportionTier::FlatChest => &ProportionRecommendations {
            tier: ProportionTier::FlatChest,
            primary_name: "flat_chest",
            core_reinforcers: &["aa_cup", "small_underbust", "slender_torso"],
            clothing_interaction: &[
                "loose_shirt",
                "baggy_clothes",
            ],
            pose_and_physics: &[
                "collarbone",
            ],
            auto_negative_lock: &[
                "breasts",
                "cleavage",
                "medium_breasts",
                "large_breasts",
                "underboob",
                "sideboob",
                "boob_shadow",
            ],
        },
        ProportionTier::SmallBreasts => &ProportionRecommendations {
            tier: ProportionTier::SmallBreasts,
            primary_name: "small_breasts",
            core_reinforcers: &["a_cup", "b_cup", "perky_breasts", "petite_breasts"],
            clothing_interaction: &[
                "tight_shirt",
            ],
            pose_and_physics: &["collarbone", "arched_back"],
            auto_negative_lock: &[
                "flat_chest",
                "large_breasts",
                "huge_breasts",
                "deep_cleavage",
                "sagging_breasts",
            ],
        },
        ProportionTier::MediumBreasts => &ProportionRecommendations {
            tier: ProportionTier::MediumBreasts,
            primary_name: "medium_breasts",
            core_reinforcers: &["c_cup", "d_cup", "proportionate_body", "full_breasts"],
            clothing_interaction: &[],
            pose_and_physics: &["perky_breasts"],
            auto_negative_lock: &[
                "flat_chest",
                "small_breasts",
                "huge_breasts",
                "gigantic_breasts",
                "hyper_breasts",
            ],
        },
        ProportionTier::LargeBreasts => &ProportionRecommendations {
            tier: ProportionTier::LargeBreasts,
            primary_name: "large_breasts",
            core_reinforcers: &["e_cup", "f_cup", "curvy", "voluptuous", "deep_cleavage"],
            clothing_interaction: &[
                "taut_shirt",
                "taut_clothes",
                "button_gap",
            ],
            pose_and_physics: &[
                "framed_breasts",
                "leaning_forward",
            ],
            auto_negative_lock: &[
                "flat_chest",
                "small_breasts",
                "medium_breasts",
                "petite",
            ],
        },
        ProportionTier::HugeBreasts => &ProportionRecommendations {
            tier: ProportionTier::HugeBreasts,
            primary_name: "huge_breasts",
            core_reinforcers: &["g_cup", "h_cup", "heavy_breasts", "voluptuous", "deep_cleavage"],
            clothing_interaction: &[
                "taut_shirt",
                "taut_clothes",
                "button_gap",
                "popped_button",
            ],
            pose_and_physics: &[
                "leaning_forward",
            ],
            auto_negative_lock: &[
                "flat_chest",
                "small_breasts",
                "medium_breasts",
                "perky_breasts",
                "slender",
                "skinny",
            ],
        },
        ProportionTier::GiganticBreasts => &ProportionRecommendations {
            tier: ProportionTier::GiganticBreasts,
            primary_name: "gigantic_breasts",
            core_reinforcers: &[
                "hyper_breasts",
                "massive_breasts",
                "enormous_breasts",
                "breasts_bigger_than_head",
                "voluptuous",
            ],
            clothing_interaction: &[
                "taut_shirt",
                "taut_clothes",
                "button_gap",
                "popped_button",
            ],
            pose_and_physics: &[
                "leaning_forward",
                "hands_on_own_chest",
            ],
            auto_negative_lock: &[
                "flat_chest",
                "small_breasts",
                "medium_breasts",
                "large_breasts",
                "proportionate",
                "petite",
                "skinny",
            ],
        },
        ProportionTier::ThickThighsAndHips => &ProportionRecommendations {
            tier: ProportionTier::ThickThighsAndHips,
            primary_name: "thick_thighs_and_hips",
            core_reinforcers: &[
                "thick_thighs",
                "wide_hips",
                "broad_hips",
                "narrow_waist_wide_hips",
                "pronounced_hips",
                "hourglass_figure",
                "plump_thighs",
                "curvy",
                "bottom_heavy",
                "pear_shaped",
            ],
            clothing_interaction: &[
                "skindentation",
            ],
            pose_and_physics: &[
                "contrapposto",
                "hands_on_own_hips",
            ],
            auto_negative_lock: &[
                "thin_legs",
                "stick_legs",
                "thigh_gap",
                "narrow_hips",
                "boyish_hips",
                "flat_hips",
                "straight_torso",
                "tubular_body",
                "skinny",
                "slender_legs",
                "flat_ass",
            ],
        },
        ProportionTier::MuscularFemale => &ProportionRecommendations {
            tier: ProportionTier::MuscularFemale,
            primary_name: "muscular_female",
            core_reinforcers: &[
                "muscular_female",
                "abs",
                "toned",
                "athletic",
                "defined_muscles",
                "six_pack",
            ],
            clothing_interaction: &[
                "taut_shirt",
                "taut_clothes",
            ],
            pose_and_physics: &[
                "flexing",
            ],
            auto_negative_lock: &[
                "chubby",
                "plump",
                "soft_body",
                "flabby",
                "frail",
                "skinny",
                "narrow_shoulders",
            ],
        },
    }
}

/// Detects if a tag corresponds to a proportion tier.
pub fn detect_proportion_tier(tag: &str) -> Option<ProportionTier> {
    match tag {
        "flat chest" | "aa cup" | "flat" | "small underbust" | "slender torso" => {
            Some(ProportionTier::FlatChest)
        }
        "small breasts" | "a cup" | "b cup" | "petite breasts" | "perky breasts" => {
            Some(ProportionTier::SmallBreasts)
        }
        "medium breasts"
        | "c cup"
        | "d cup"
        | "average breasts"
        | "proportionate body"
        | "full breasts" => Some(ProportionTier::MediumBreasts),
        "large breasts" | "e cup" | "f cup" => Some(ProportionTier::LargeBreasts),
        "huge breasts" | "g cup" | "h cup" | "heavy breasts" => Some(ProportionTier::HugeBreasts),
        "gigantic breasts"
        | "hyper breasts"
        | "massive breasts"
        | "enormous breasts"
        | "breasts bigger than head" => Some(ProportionTier::GiganticBreasts),
        "thick thighs"
        | "wide hips"
        | "broad hips"
        | "childbearing hips"
        | "huge hips"
        | "pronounced hips"
        | "narrow waist wide hips"
        | "wide pelvis"
        | "plump thighs"
        | "bottom heavy"
        | "pear shaped" => Some(ProportionTier::ThickThighsAndHips),
        "muscular female" | "defined muscles" | "six pack" | "abs" | "toned" | "athletic" => {
            Some(ProportionTier::MuscularFemale)
        }
        _ => None,
    }
}

/// Detects all proportion tiers present in a list of tokens.
pub fn detect_all_proportion_tiers(tokens: &[PromptToken]) -> Vec<(ProportionTier, PromptToken)> {
    let mut detected = Vec::new();
    let mut seen_breast = false;
    let mut seen_thighs = false;
    let mut seen_muscular = false;

    for t in tokens {
        if let Some(tier) = detect_proportion_tier(&t.normalized) {
            match tier {
                ProportionTier::FlatChest
                | ProportionTier::SmallBreasts
                | ProportionTier::MediumBreasts
                | ProportionTier::LargeBreasts
                | ProportionTier::HugeBreasts
                | ProportionTier::GiganticBreasts => {
                    if !seen_breast {
                        seen_breast = true;
                        detected.push((tier, t.clone()));
                    }
                }
                ProportionTier::ThickThighsAndHips => {
                    if !seen_thighs {
                        seen_thighs = true;
                        detected.push((tier, t.clone()));
                    }
                }
                ProportionTier::MuscularFemale => {
                    if !seen_muscular {
                        seen_muscular = true;
                        detected.push((tier, t.clone()));
                    }
                }
            }
        }
    }
    detected
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ProportionLockResult {
    pub final_positive_tags: Vec<String>,
    pub removed_conflicts: Vec<String>,
    pub recommended_clothing: Vec<String>,
    pub recommended_physics_poses: Vec<String>,
    pub auto_negative_additions: Vec<String>,
}

/// Returns checkpoint-adaptive weight for proportion tags:
/// - Pony V6: clamped to 1.15 to avoid burning and color shift.
/// - Illustrious: clamped to 1.20 for optimal Danbooru cross-attention.
/// - SD1.5 / Anima: up to 1.25.
/// - Flux: 1.0 (natural language models don't use numeric multipliers).
#[allow(dead_code)]
pub fn adaptive_weight_for_checkpoint(checkpoint: CheckpointProfile, requested_weight: f32) -> f32 {
    match checkpoint {
        CheckpointProfile::PonyV6 => requested_weight.clamp(1.0, 1.15),
        CheckpointProfile::Illustrious => requested_weight.clamp(1.0, 1.20),
        CheckpointProfile::Sd15 | CheckpointProfile::Anima | CheckpointProfile::Sdxl => requested_weight.clamp(1.0, 1.25),
        CheckpointProfile::Flux => 1.0,
    }
}

#[allow(dead_code)]
pub fn apply_proportion_lock_with_checkpoint(
    current_tags: &[String],
    selected_tier: ProportionTier,
    auto_inject_core: bool,
    weight_multiplier: f32,
    checkpoint: CheckpointProfile,
) -> ProportionLockResult {
    let adaptive_weight = adaptive_weight_for_checkpoint(checkpoint, weight_multiplier);
    apply_proportion_lock(current_tags, selected_tier, auto_inject_core, adaptive_weight)
}

/// Applies proportion locking:
/// 1. Strips conflicting size/hijacker tags.
/// 2. Locks in the primary tag (optionally with prompt weighting).
/// 3. Returns categorized recommendations and auto-negative tags.
#[allow(dead_code)]
pub fn apply_proportion_lock(
    current_tags: &[String],
    selected_tier: ProportionTier,
    auto_inject_core: bool,
    weight_multiplier: f32,
) -> ProportionLockResult {
    let rec = get_proportion_recommendations(selected_tier);
    let mut cleaned_tags = Vec::new();
    let mut removed_conflicts = Vec::new();

    for tag_str in current_tags {
        let norm = tag_str.trim().to_lowercase().replace('_', " ");
        let is_conflict = match selected_tier {
            ProportionTier::FlatChest
            | ProportionTier::SmallBreasts
            | ProportionTier::MediumBreasts
            | ProportionTier::LargeBreasts
            | ProportionTier::HugeBreasts
            | ProportionTier::GiganticBreasts => {
                let breast_tier = match selected_tier {
                    ProportionTier::FlatChest => BreastTier::FlatChest,
                    ProportionTier::SmallBreasts => BreastTier::SmallBreasts,
                    ProportionTier::MediumBreasts => BreastTier::MediumBreasts,
                    ProportionTier::LargeBreasts => BreastTier::LargeBreasts,
                    ProportionTier::HugeBreasts => BreastTier::HugeBreasts,
                    ProportionTier::GiganticBreasts => BreastTier::GiganticBreasts,
                    _ => unreachable!(),
                };
                breast_tier_conflicts(breast_tier, &norm).is_some()
            }
            ProportionTier::ThickThighsAndHips => is_thick_curvy_conflict(&norm),
            ProportionTier::MuscularFemale => is_muscular_conflict(&norm),
        };

        if is_conflict {
            removed_conflicts.push(tag_str.clone());
        } else {
            cleaned_tags.push(tag_str.clone());
        }
    }

    let primary_clean = rec.primary_name.replace('_', " ");
    let locked_tag = if weight_multiplier > 1.0 {
        format!("({}:{:.2})", primary_clean, weight_multiplier)
    } else {
        primary_clean.clone()
    };

    let already_has_primary = cleaned_tags.iter().any(|t| {
        let n = t.trim().to_lowercase().replace('_', " ");
        n == primary_clean || n.contains(&primary_clean)
    });

    if weight_multiplier > 1.0 {
        if let Some(pos) = cleaned_tags.iter().position(|t| {
            let n = t.trim().to_lowercase().replace('_', " ");
            n == primary_clean || n.starts_with(&format!("({primary_clean}:"))
        }) {
            cleaned_tags[pos] = locked_tag;
        } else {
            cleaned_tags.insert(0, locked_tag);
        }
    } else if !already_has_primary {
        cleaned_tags.insert(0, locked_tag);
    }

    if auto_inject_core {
        for core in rec.core_reinforcers {
            let core_clean = core.replace('_', " ");
            let has_core = cleaned_tags.iter().any(|t| {
                let n = t.trim().to_lowercase().replace('_', " ");
                n == core_clean || n.contains(&core_clean)
            });
            if !has_core {
                cleaned_tags.push(core_clean);
            }
        }
    }

    ProportionLockResult {
        final_positive_tags: cleaned_tags,
        removed_conflicts,
        recommended_clothing: rec.clothing_interaction.iter().map(|s| s.to_string()).collect(),
        recommended_physics_poses: rec.pose_and_physics.iter().map(|s| s.to_string()).collect(),
        auto_negative_additions: rec.auto_negative_lock.iter().map(|s| s.to_string()).collect(),
    }
}

/// Checks whether a body-part token triggers a camera-pull conflict with the active view
/// (e.g., `ass` or `butt` inside `Upper Body` or `Front View`), and returns:
/// `Some((reason, framing_safe_body_alternative, camera_framing_alternative))`
pub fn check_framing_safe_body_swap(
    normalized_tag: &str,
    effective_view: TargetView,
    has_small_chest: bool,
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
            | "butt crack"
            | "back dimples"
            | "sacral dimples"
            | "tailplug"
            | "buttplug"
            | "back bow"
    );

    let is_shoulder_blades = normalized_tag.contains("shoulder blade")
        || normalized_tag.contains("scapula")
        || normalized_tag == "scapulae";

    let is_rear_upper = is_shoulder_blades
        || normalized_tag == "back"
        || matches!(
            normalized_tag,
            "bare back"
                | "back cutout"
                | "backless"
                | "backless dress"
                | "backless outfit"
                | "muscular back"
                | "spine"
                | "nape"
                | "back tattoo"
                | "reverse bunnysuit"
                | "sideboob"
        );

    let is_front_chest = matches!(
        normalized_tag,
        "cleavage"
            | "deep cleavage"
            | "front cleavage"
            | "underboob"
            | "collarbone"
            | "chest"
            | "breasts"
            | "large breasts"
            | "medium breasts"
            | "small breasts"
            | "flat chest"
            | "huge breasts"
            | "gigantic breasts"
            | "pectorals"
    );

    let is_front_midriff_lower = matches!(
        normalized_tag,
        "navel"
            | "stomach"
            | "midriff"
            | "abs"
            | "muscular abs"
            | "cameltoe"
            | "bulge"
    );

    let is_lower_legs_feet = matches!(
        normalized_tag,
        "feet"
            | "barefoot"
            | "toes"
            | "soles"
            | "ankles"
            | "calves"
            | "shins"
            | "knees"
            | "thighs"
            | "thick thighs"
            | "thigh gap"
            | "legs"
            | "long legs"
            | "hips"
            | "wide hips"
    );

    let is_footwear_legwear = matches!(
        normalized_tag,
        "shoes"
            | "boots"
            | "high heels"
            | "sneakers"
            | "sandals"
            | "footwear"
            | "anklet"
            | "socks"
            | "kneehighs"
            | "thighhighs"
            | "pantyhose"
            | "legwear"
    );

    let is_lower_clothing = matches!(
        normalized_tag,
        "panties" | "skirt" | "shorts" | "pants"
    );

    let is_lower_or_full_action = matches!(
        normalized_tag,
        "standing" | "walking" | "running" | "sitting" | "squatting"
    );

    let is_micro_face = matches!(
        normalized_tag,
        "detailed eyes"
            | "pupils"
            | "slit pupils"
            | "sclera"
            | "lip gloss"
            | "uvula"
            | "mole under eye"
    );

    match effective_view {
        TargetView::UpperBody => {
            if is_rear_lower {
                if has_small_chest {
                    return Some((
                        "Direct 'ass/butt' tags pull ComfyUI's camera downward and backward, breaking 'upper body' framing and twisting anatomy.",
                        "slender, narrow waist, athletic build, arched back",
                        "cowboy shot, from behind, looking back",
                    ));
                }
                return Some((
                    "Direct 'ass/butt' tags pull ComfyUI's camera downward and backward, breaking 'upper body' framing and twisting anatomy.",
                    "curvy, voluptuous, hourglass figure, narrow waist, arched back",
                    "cowboy shot, from behind, looking back",
                ));
            }
            if is_lower_legs_feet
                || is_footwear_legwear
                || matches!(normalized_tag, "standing" | "walking" | "running" | "squatting")
            {
                return Some((
                    "Forces framing below the waist/hips.",
                    "slender build, graceful posture, poised stance",
                    "full body, standing",
                ));
            }
        }
        TargetView::CloseUpPortrait => {
            if is_rear_lower
                || is_lower_legs_feet
                || is_footwear_legwear
                || is_lower_clothing
                || is_lower_or_full_action
                || is_front_midriff_lower
                || is_shoulder_blades
            {
                return Some((
                    "Pulls camera down/back to render lower body or full-body actions.",
                    "toned shoulders, defined collarbone, expressive face, styled hair",
                    "full body, cowboy shot",
                ));
            }
        }
        TargetView::FrontView => {
            if is_rear_lower {
                if has_small_chest {
                    return Some((
                        "Forces rear or side rotation to render back/side anatomy.",
                        "narrow waist, slender build, athletic stance",
                        "from behind, looking back",
                    ));
                }
                return Some((
                    "Forces rear or side rotation to render back/side anatomy.",
                    "wide hips, curvy, hourglass figure, thick thighs",
                    "from behind, looking back",
                ));
            }
            if is_rear_upper {
                let alt = if matches!(normalized_tag, "backless dress" | "backless outfit" | "reverse bunnysuit") {
                    "cleavage cutout, plunge dress, front cutout"
                } else {
                    "defined collarbone, toned shoulders, clavicle"
                };
                return Some((
                    "Forces rear or side rotation to render back/side anatomy.",
                    alt,
                    "from behind, looking back over shoulder",
                ));
            }
        }
        TargetView::FromBehind => {
            if is_front_chest {
                if has_small_chest {
                    return Some((
                        "Forces torso to rotate forward to render front anatomy.",
                        "small breasts from behind, slender silhouette, arched back",
                        "front view, straight-on",
                    ));
                }
                return Some((
                    "Forces torso to rotate forward to render front anatomy.",
                    "sideboob, breasts from behind, curvy silhouette, arched back",
                    "front view, straight-on",
                ));
            }
            if is_front_midriff_lower {
                return Some((
                    "Forces torso to rotate forward to render front anatomy.",
                    "toned back, curvy hips, back dimples, defined spine, rear silhouette",
                    "front view, straight-on",
                ));
            }
        }
        TargetView::FullBody if is_micro_face => {
            return Some((
                "Forces tight facial crop to resolve micro-details.",
                "expressive face, clear facial features",
                "portrait, close-up, face focus",
            ));
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
            if matches!(
                normalized_tag,
                "full body" | "lower body" | "wide shot" | "extreme close-up" | "headshot"
            ) {
                return Some((
                    "Forces framing below the waist/hips.",
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
                    "Pulls camera down/back to render lower body or full-body actions.",
                    "portrait, close-up, face focus",
                ));
            }
        }
        TargetView::FrontView => {
            if matches!(
                normalized_tag,
                "from behind"
                    | "back view"
                    | "rear view"
                    | "back focus"
                    | "over the shoulder"
                    | "looking back"
                    | "profile"
                    | "from side"
                    | "side view"
            ) {
                return Some((
                    "Forces rear or side rotation to render back/side anatomy.",
                    "front view, looking at viewer",
                ));
            }
        }
        TargetView::FromBehind => {
            if matches!(normalized_tag, "front view" | "straight-on" | "straight on") {
                return Some((
                    "Forces torso to rotate forward to render front anatomy.",
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
                "upper body"
                    | "cowboy shot"
                    | "portrait"
                    | "close-up"
                    | "close up"
                    | "headshot"
                    | "cropped"
                    | "face focus"
                    | "eye focus"
            ) {
                return Some((
                    "Forces tight facial crop to resolve micro-details.",
                    "full body, standing",
                ));
            }
        }
        TargetView::None => {}
    }
    None
}

/// Checks if an extreme high/low camera pitch tag conflicts with eye-level camera framing.
pub fn check_eye_level_conflict(
    normalized_tag: &str,
    is_eye_level_active: bool,
) -> Option<(&'static str, &'static str, &'static str)> {
    if !is_eye_level_active {
        return None;
    }
    let is_extreme_pitch = matches!(
        normalized_tag,
        "from below"
            | "upskirt"
            | "pantyshot"
            | "underboob"
            | "undershot"
            | "soles"
            | "stepping on viewer"
            | "from above"
            | "downblouse"
            | "top of head"
            | "low angle"
            | "high angle"
            | "bird's-eye view"
            | "worm's-eye view"
    );
    if is_extreme_pitch {
        Some((
            "Hijacks camera pitch (Z-axis) to extreme high or low angles.",
            "",
            "straight-on, eye level",
        ))
    } else {
        None
    }
}

/// Checks if an arm gesture or grounded posture directly conflicts with an active resting/upright pose.
pub fn check_pose_conflict(
    normalized_tag: &str,
    is_arms_down_or_hips: bool,
    is_standing_upright: bool,
) -> Option<(&'static str, &'static str, &'static str)> {
    if is_arms_down_or_hips {
        let is_gesture = matches!(
            normalized_tag,
            "armpits"
                | "bare armpits"
                | "armpit crease"
                | "armpit hair"
                | "arms up"
                | "hands up"
                | "stretching"
                | "tying hair"
                | "arms behind head"
                | "hands behind head"
                | "reaching towards viewer"
                | "peace sign"
                | "waving"
                | "salute"
        );
        if is_gesture {
            return Some((
                "Armpits and gesture tags force arms to lift or break resting position.",
                "",
                "",
            ));
        }
    }

    if is_standing_upright {
        let is_grounded = matches!(
            normalized_tag,
            "sitting"
                | "seiza"
                | "wariza"
                | "indian style"
                | "hugging knees"
                | "lying"
                | "on back"
                | "on stomach"
                | "on side"
                | "all fours"
                | "top-down bottom-up"
                | "squatting"
                | "crouching"
                | "kneeling"
                | "soles"
        );
        if is_grounded {
            return Some((
                "Forces grounded, seated, or bent spine postures.",
                "",
                "",
            ));
        }
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

pub fn is_two_hand_pose(norm: &str) -> bool {
    matches!(
        norm,
        "hands on hips"
            | "hands on own hips"
            | "hands on waist"
            | "crossed arms"
            | "arms crossed"
            | "arms at sides"
            | "hands in pockets"
            | "heart hands"
            | "praying"
            | "own hands together"
            | "arms behind head"
            | "hands behind head"
            | "arms behind back"
            | "hands behind back"
            | "two-handed breast hold"
            | "grabbing own breast"
            | "breast hold"
            | "holding breasts"
            | "breast grab"
            | "hands on own chest"
            | "arms up"
            | "hands up"
            | "stretching"
            | "holding own hair"
            | "twintails grab"
    )
}

pub fn is_one_hand_gesture(norm: &str) -> bool {
    matches!(
        norm,
        "peace sign"
            | "v"
            | "double v"
            | "waving"
            | "salute"
            | "hand on hip"
            | "hand on own hip"
            | "pointing"
            | "thumbs up"
            | "hand to mouth"
            | "finger to mouth"
            | "shushing"
            | "adjusting eyewear"
            | "adjusting glasses"
            | "chin rest"
            | "reaching towards viewer"
            | "outstretched arm"
            | "outstretched hand"
            | "beckoning"
    ) || (norm.starts_with("holding ") && !matches!(norm, "holding hands" | "holding own hair"))
}

pub fn posture_group(norm: &str) -> Option<&'static str> {
    match norm {
        "standing" | "standing upright" | "straight posture" | "standing on one leg" => Some("standing"),
        "sitting" | "indian style" | "hugging knees" => Some("sitting"),
        "kneeling" => Some("kneeling"),
        // seiza / wariza are sitting on the knees: both a kind of sitting and of kneeling
        "seiza" | "wariza" => Some("seated_kneel"),
        "squatting" | "crouching" => Some("squatting"),
        "lying" | "on back" | "on stomach" | "on side" => Some("lying"),
        "all fours" | "top-down bottom-up" => Some("all_fours"),
        _ => None,
    }
}

/// Whether two posture groups can't both be true for one person. The same group never conflicts, and
/// seiza / wariza (`seated_kneel`) agree with plain sitting or kneeling: those pairs are implications
/// (the generic tag gets an ImpliedTag card instead).
pub fn postures_conflict(a: &str, b: &str) -> bool {
    if a == b {
        return false;
    }
    let seated_kneel_ok = |x: &str, y: &str| x == "seated_kneel" && (y == "sitting" || y == "kneeling");
    !(seated_kneel_ok(a, b) || seated_kneel_ok(b, a))
}

/// Leg positions or postures that can't go with a leg position (`crossed legs`, `spread legs`…):
/// seiza / wariza fold the legs under, standing on one leg lifts one.
pub fn leg_cross_conflict(posture: &str, leg_group: &str) -> bool {
    match posture {
        "seiza" | "wariza" => matches!(leg_group, "crossed" | "spread"),
        "standing on one leg" => matches!(leg_group, "spread" | "together" | "wide"),
        _ => false,
    }
}

/// Boot heights are mutually exclusive on one person (one_per_part's rule for image reads).
pub fn boot_height(norm: &str) -> Option<&'static str> {
    match norm {
        "thigh boots" | "thighhigh boots" | "thigh-high boots" => Some("thigh"),
        "knee boots" | "knee-high boots" => Some("knee"),
        "ankle boots" => Some("ankle"),
        _ => None,
    }
}

pub fn leg_stance_group(norm: &str) -> Option<&'static str> {
    match norm {
        "crossed legs" => Some("crossed"),
        "spread legs" | "legs spread" | "legs apart" => Some("spread"),
        "feet together" => Some("together"),
        "pigeon-toed" => Some("pigeon"),
        "wide stance" => Some("wide"),
        _ => None,
    }
}

pub fn is_footwear(norm: &str) -> bool {
    !crate::ai_engine::is_unworn_footwear(norm) && matches!(
        norm,
        "shoes"
            | "boots"
            | "loafers"
            | "sneakers"
            | "sandals"
            | "high heels"
            | "slippers"
            | "footwear"
            | "geta"
            | "zori"
            | "uwabaki"
            | "mary janes"
            | "ballet slippers"
    ) || norm.ends_with(" shoes")
        || norm.ends_with(" boots")
        || norm.ends_with(" sneakers")
        || norm.ends_with(" sandals")
        || norm.ends_with(" heels")
        || norm.ends_with(" loafers")
}

/// Poses that hold the legs apart, so the thighs separate (a thigh gap is natural then).
pub fn legs_apart(norm: &str) -> bool {
    matches!(norm, "spread legs" | "legs spread" | "legs apart" | "wide stance" | "squatting" | "knees apart")
}

pub fn is_barefoot(norm: &str) -> bool {
    matches!(norm, "barefoot" | "bare feet" | "bare foot" | "no shoes")
}

pub fn is_barefoot_exception(norm: &str) -> bool {
    crate::ai_engine::is_unworn_footwear(norm)
}

pub fn analyze_with_rules(tokens: &[PromptToken], config: &ScanConfig) -> RuleAnalysisResult {
    let effective_view = detect_effective_view(tokens, config.target_view);

    let angle_label = crate::profiles::CAMERA_ANGLES
        .get(config.camera_angle)
        .map(|(l, _)| *l)
        .unwrap_or("");
    let is_eye_level_active = angle_label.starts_with("Eye level")
        || tokens.iter().any(|t| t.normalized == "eye level");

    let pose_label = crate::profiles::POSES
        .get(config.pose)
        .map(|(l, _)| *l)
        .unwrap_or("");
    let is_pose_standing_upright = pose_label.starts_with("Standing")
        || pose_label.starts_with("Contrapposto")
        || pose_label.starts_with("Hands on hips")
        || pose_label.starts_with("Arms crossed");
    let is_pose_arms_down_or_hips = pose_label.starts_with("Hands on hips")
        || pose_label.starts_with("Arms crossed");

    let prompt_has_standing = tokens.iter().any(|t| {
        matches!(
            t.normalized.as_str(),
            "standing" | "straight posture" | "standing upright"
        )
    });
    let prompt_has_arms_down_or_hips = tokens.iter().any(|t| {
        matches!(
            t.normalized.as_str(),
            "arms at sides"
                | "hands on hips"
                | "hands on own hips"
                | "hands in pockets"
                | "crossed arms"
        )
    });

    let is_multi_person = tokens.iter().any(|t| {
        matches!(
            t.normalized.as_str(),
            "2girls" | "3girls" | "4girls" | "5girls" | "6+girls" | "multiple girls"
            | "2boys" | "3boys" | "4boys" | "5boys" | "6+boys" | "multiple boys"
            | "2others" | "3others" | "multiple others"
        )
    }) || (tokens.iter().filter(|t| matches!(t.normalized.as_str(), "1girl" | "1boy" | "1other")).count() > 1);
    // with several people one can stand while another sits, or one waves while another crosses her arms,
    // so the prompt's own poses never conflict then; a Pose dropdown pick still applies
    let standing_upright_active = is_pose_standing_upright
        || (config.pose == 0 && prompt_has_standing && !is_multi_person);
    let arms_down_or_hips_active = is_pose_arms_down_or_hips
        || (config.pose == 0 && prompt_has_arms_down_or_hips && !is_multi_person);

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

    let mut pose_conflict_reasons: std::collections::HashMap<String, &'static str> = std::collections::HashMap::new();

    if !is_multi_person {
        // 1. Two-Hand Budget
        let hand_tokens: Vec<&PromptToken> = tokens.iter().filter(|t| is_two_hand_pose(&t.normalized) || is_one_hand_gesture(&t.normalized)).collect();
        if !hand_tokens.is_empty() {
            let two_hand_winner = hand_tokens.iter().find(|t| is_two_hand_pose(&t.normalized) && (t.weight > 1.0 || is_pose_arms_down_or_hips))
                .or_else(|| hand_tokens.iter().find(|t| is_two_hand_pose(&t.normalized)))
                .or_else(|| hand_tokens.iter().find(|t| t.weight > 1.0))
                .or_else(|| hand_tokens.first());

            if let Some(winner) = two_hand_winner {
                if is_two_hand_pose(&winner.normalized) || is_pose_arms_down_or_hips {
                    for t in &hand_tokens {
                        if t.normalized != winner.normalized {
                            pose_conflict_reasons.insert(
                                t.normalized.clone(),
                                "Hands are already occupied by a two-handed pose. Conflicting hand pose exceeds the human 2-hand limit.",
                            );
                        }
                    }
                } else {
                    let mut allowed_count = 0;
                    for t in &hand_tokens {
                        if allowed_count < 2 {
                            allowed_count += 1;
                        } else {
                            pose_conflict_reasons.insert(
                                t.normalized.clone(),
                                "Human anatomy limit (2 hands) already reached. Third hand gesture causes deformed extra arms.",
                            );
                        }
                    }
                }
            }
        }

        // 2. Postures (Mutually exclusive body locations)
        let posture_tokens: Vec<&PromptToken> = tokens.iter().filter(|t| posture_group(&t.normalized).is_some()).collect();
        if posture_tokens.len() > 1 {
            let win_posture = posture_tokens.iter().find(|t| t.weight > 1.0 || (is_pose_standing_upright && posture_group(&t.normalized) == Some("standing")))
                .or_else(|| posture_tokens.first())
                .unwrap();
            let win_grp = posture_group(&win_posture.normalized).unwrap();
            for t in &posture_tokens {
                if postures_conflict(posture_group(&t.normalized).unwrap(), win_grp) {
                    pose_conflict_reasons.insert(
                        t.normalized.clone(),
                        "Forces grounded, seated, or contradictory spine postures.",
                    );
                }
            }
        }

        // 3. Leg Positions (Mutually exclusive leg stances)
        let leg_tokens: Vec<&PromptToken> = tokens.iter().filter(|t| leg_stance_group(&t.normalized).is_some()).collect();
        if leg_tokens.len() > 1 {
            let win_leg = leg_tokens.iter().find(|t| t.weight > 1.0).or_else(|| leg_tokens.first()).unwrap();
            let win_grp = leg_stance_group(&win_leg.normalized).unwrap();
            for t in &leg_tokens {
                if leg_stance_group(&t.normalized).unwrap() != win_grp {
                    pose_conflict_reasons.insert(
                        t.normalized.clone(),
                        "Leg stance directly contradicts an active leg pose.",
                    );
                }
            }
        }

        // 3b. Postures that fold or lift the legs vs a leg position (seiza + crossed legs)
        for p in tokens.iter().filter(|t| matches!(t.normalized.as_str(), "seiza" | "wariza" | "standing on one leg")) {
            for l in tokens.iter() {
                if let Some(grp) = leg_stance_group(&l.normalized) {
                    if leg_cross_conflict(&p.normalized, grp) {
                        // a weighted tag beats an unweighted one; otherwise the first written wins
                        let posture_wins = match (p.weight > 1.0, l.weight > 1.0) {
                            (true, false) => true,
                            (false, true) => false,
                            _ => p.original_index < l.original_index,
                        };
                        let drop = if posture_wins { l } else { p };
                        pose_conflict_reasons
                            .entry(drop.normalized.clone())
                            .or_insert("This leg position can't go with the posture: seiza / wariza fold the legs under, standing on one leg lifts one.");
                    }
                }
            }
        }

        // 3c. One boot height per person (first written wins, weighted beats unweighted)
        let boots: Vec<&PromptToken> = tokens.iter().filter(|t| boot_height(&t.normalized).is_some()).collect();
        if boots.len() > 1 {
            let win = boots.iter().find(|t| t.weight > 1.0).or_else(|| boots.first()).unwrap();
            let win_h = boot_height(&win.normalized).unwrap();
            for t in &boots {
                if boot_height(&t.normalized).unwrap() != win_h {
                    pose_conflict_reasons.insert(t.normalized.clone(), "Boots can only have one height: thigh, knee or ankle boots.");
                }
            }
        }

        // 3d. A typed `thigh gap` next to thick thighs / wide hips: full thighs touch unless the legs are apart
        if tokens.iter().any(|t| t.normalized == "thigh gap")
            && tokens.iter().any(|t| matches!(t.normalized.as_str(), "thick thighs" | "wide hips"))
            && !tokens.iter().any(|t| legs_apart(&t.normalized))
        {
            pose_conflict_reasons.insert(
                "thigh gap".to_string(),
                "'thigh gap' contradicts thick thighs: full thighs touch unless the legs are apart (spread legs, wide stance…).",
            );
        }

        // 4. Barefoot vs Footwear
        let has_footwear = tokens.iter().find(|t| is_footwear(&t.normalized));
        let has_barefoot = tokens.iter().find(|t| is_barefoot(&t.normalized));
        let has_exception = tokens.iter().any(|t| is_barefoot_exception(&t.normalized));
        if let (Some(_fw), Some(bf)) = (has_footwear, has_barefoot) {
            if !has_exception {
                pose_conflict_reasons.insert(
                    bf.normalized.clone(),
                    "Cannot be simultaneously barefoot and wearing footwear.",
                );
            }
        }
    }

    let mut changes: Vec<PromptChange> = Vec::new();
    let mut next_change_id = 0usize;
    let mut seen_normalized: HashSet<String> = HashSet::new();
    let mut processed_tokens: Vec<PromptToken> = Vec::new();
    let mut added_quality_replacement = false;
    let has_small_chest = tokens.iter().any(|t| is_small_chest_tag(&t.normalized));

    for token in tokens {
        let mut tok = token.clone();
        let is_pinned = pinned_norms
            .iter()
            .any(|p| tok.normalized == *p || crate::parser::has_phrase(&tok.normalized, p));

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
            check_framing_safe_body_swap(&tok.normalized, effective_view, has_small_chest)
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

        // 3.1 Check Eye-Level Pitch Conflict (extreme high/low pitch vs eye level)
        if let Some((reason, alt, framing_alt)) = check_eye_level_conflict(&tok.normalized, is_eye_level_active) {
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::PositioningConflict,
                reason: reason.to_string(),
                suggested_alternative: alt.to_string(),
                framing_alternative: framing_alt.to_string(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;

            if config.resolve_conflicts_enabled {
                continue;
            }
        }

        // 3.2 Check Pose Conflict (resting arms vs arm gestures, upright standing vs grounded/bent poses, two-hand budget, postures, footwear)
        let legacy_check = check_pose_conflict(
            &tok.normalized,
            arms_down_or_hips_active,
            standing_upright_active,
        );
        let custom_reason = if legacy_check.is_none() {
            pose_conflict_reasons.get(&tok.normalized).copied()
        } else {
            None
        };

        if let Some(reason) = legacy_check.map(|(r, _, _)| r).or(custom_reason) {
            let alt = legacy_check.map(|(_, a, _)| a).unwrap_or("");
            let framing_alt = legacy_check.map(|(_, _, f)| f).unwrap_or("");
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::PoseConflict,
                reason: reason.to_string(),
                suggested_alternative: alt.to_string(),
                framing_alternative: framing_alt.to_string(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;

            if config.resolve_conflicts_enabled {
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

        // 6. Check 2D Anime Flat Protection vs 3D Bleed Tags (if Cleanup or Conflict Resolution is ON and checkpoint is 2D anime)
        if (config.cleanup_enabled || config.resolve_conflicts_enabled)
            && config.checkpoint.is_2d_anime_checkpoint()
            && is_3d_bleed_tag(&tok.normalized)
        {
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.raw.clone(),
                action: ChangeAction::StyleConflict,
                reason: "Strips tags that pull 2D anime checkpoints toward uncanny 3D/semi-realistic rendering.".to_string(),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: 0,
            });
            next_change_id += 1;
            continue;
        }

        // 6.1 Check Checkpoint-specific Bloat / Redundant Quality Synonyms (ONLY if Cleanup Mode is ON)
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

    // Camera angle / pose / bust / frame dropdowns: the user's explicit choice replaces the prompt's
    // own tags of that list (shown as a card so it can be undone). A bust choice also replaces any
    // other breast size the prompt has ("c cup", "perky breasts").
    let mut dropdown_body: Vec<PromptToken> = Vec::new();
    for (list, idx, action, what) in [
        (crate::profiles::CAMERA_ANGLES, config.camera_angle, ChangeAction::AngleChoice, "camera angle"),
        (crate::profiles::POSES, config.pose, ChangeAction::PoseChoice, "pose"),
        (crate::profiles::BUSTS, config.bust, ChangeAction::ProportionChoice, "bust"),
        (crate::profiles::FRAMES, config.frame, ChangeAction::ProportionChoice, "frame"),
    ] {
        let Some((label, tags)) = list.get(idx).filter(|_| idx > 0) else { continue };
        let is_bust = what == "bust";
        let group = crate::profiles::group_tags(list);
        let in_group = |n: &str| group.iter().any(|g| g == n) || (is_bust && detect_breast_tier(n).is_some());
        let chosen: Vec<PromptToken> = crate::parser::parse_prompt(tags);
        let is_chosen = |n: &str| chosen.iter().any(|c| c.normalized == n);
        let replaced: Vec<String> = processed_tokens
            .iter()
            .filter(|t| in_group(&t.normalized) && !is_chosen(&t.normalized))
            .map(|t| t.text.clone())
            .collect();
        processed_tokens.retain(|t| !(in_group(&t.normalized) && !is_chosen(&t.normalized)));
        for mut t in chosen {
            if action == ChangeAction::ProportionChoice {
                dropdown_body.push(t.clone());
            }
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

    // "Exaggerated": a pear or shortstack figure has narrow shoulders. There's no Danbooru tag for it, so it's
    // only added where the checkpoint reads plain words (Anima / SDXL / Flux); it belongs to the Frame pick.
    if config.exaggerated_proportions
        && matches!(crate::condenser::frame_label(config.frame), "pear" | "shortstack")
        && config.checkpoint.supports_natural_language()
        && !processed_tokens.iter().any(|t| t.normalized == "narrow shoulders")
    {
        let mut t = crate::parser::parse_prompt("narrow shoulders").remove(0);
        t.added = true;
        t.original_index = 940;
        t.category = TokenCategory::FramingPose;
        dropdown_body.push(t.clone());
        processed_tokens.push(t);
    }

    // The LoRA box beats a Bust / Frame choice too, so those aren't protected below.
    let loras = crate::parser::parse_prompt(&config.lora_tags_raw);
    dropdown_body.retain(|d| !loras.iter().any(|l| l.normalized == d.normalized || tags_conflict(&l.normalized, &d.normalized)));
    let is_dropdown = |n: &str| dropdown_body.iter().any(|d| d.normalized == n);

    let has_shortstack = processed_tokens.iter().any(|tok| tok.normalized == "shortstack")
        || is_dropdown("shortstack");
    if config.exaggerated_proportions && has_shortstack {
        if !processed_tokens.iter().any(|t| t.normalized == "exaggerated proportions") {
            let t = PromptToken {
                raw: "exaggerated proportions".to_string(),
                text: "exaggerated proportions".to_string(),
                normalized: "exaggerated proportions".to_string(),
                weight: 1.0,
                had_runaway_weight: false,
                replaced: false,
                added: true,
                original_index: 941,
                category: TokenCategory::CharacterBody,
            };
            processed_tokens.push(t);
            changes.push(PromptChange {
                id: next_change_id,
                original: String::new(),
                action: ChangeAction::ProportionChoice,
                reason: "Exaggerated proportions enabled for shortstack to bypass realistic human anatomy constraints.".to_string(),
                suggested_alternative: "exaggerated proportions".to_string(),
                framing_alternative: String::new(),
                mode: 0,
            });
            next_change_id += 1;
        }
    }

    // LoRA tags (their own box) go into the output word for word (`insert_lora_block`).
    // Here the prompt's own copies are dropped, and tags that contradict one are
    // replaced by it ("large breasts" beats "gigantic breasts"), with a card each.
    for lora in crate::parser::parse_prompt(&config.lora_tags_raw) {
        processed_tokens.retain(|t| t.normalized != lora.normalized);
        let conflicting: Vec<String> = processed_tokens
            .iter()
            .filter(|t| tags_conflict(&lora.normalized, &t.normalized))
            .map(|t| t.text.clone())
            .collect();
        if conflicting.is_empty() {
            continue;
        }
        processed_tokens.retain(|t| !tags_conflict(&lora.normalized, &t.normalized));
        changes.push(PromptChange {
            id: next_change_id,
            original: conflicting.join(", "),
            action: ChangeAction::KeptEmphasized,
            reason: format!("Your LoRA tag '{}' takes precedence over '{}'.", lora.text, conflicting.join("', '")),
            suggested_alternative: String::new(),
            framing_alternative: String::new(),
            mode: 0,
        });
        next_change_id += 1;
    }

    // 1. Species and Anatomy Purity (Kemonomimi)
    let kemonomimi_tok = processed_tokens
        .iter()
        .find(|t| is_kemonomimi_tag(&t.normalized))
        .cloned();

    if let Some(kemo_tok) = kemonomimi_tok {
        let mut conflicts: Vec<PromptToken> = Vec::new();
        for t in &processed_tokens {
            if t.normalized == kemo_tok.normalized || is_kemonomimi_tag(&t.normalized) {
                continue;
            }
            if is_furry_bleed_tag(&t.normalized) || is_four_ears_trigger_tag(&t.normalized) {
                conflicts.push(t.clone());
            }
        }
        for c in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::AnatomyConflict,
                reason: "Prevents kemonomimi characters from turning into full anthros/furries or growing a second pair of human ears.".to_string(),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 2. Facial State Conflicts: Closed Eyes
    let closed_eyes_tok = processed_tokens
        .iter()
        .find(|t| is_closed_eyes_tag(&t.normalized))
        .cloned();

    if let Some(ce_tok) = closed_eyes_tok {
        let mut conflicts: Vec<PromptToken> = Vec::new();
        for t in &processed_tokens {
            if t.normalized == ce_tok.normalized || is_closed_eyes_tag(&t.normalized) {
                continue;
            }
            if is_closed_eyes_conflict_tag(&t.normalized) {
                conflicts.push(t.clone());
            }
        }
        for c in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::AnatomyConflict,
                reason: "Eye color, pupil, and gaze tags force the eyelids open even when closed_eyes is specified.".to_string(),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 3. Facial State Conflicts: Closed Mouth
    let closed_mouth_tok = processed_tokens
        .iter()
        .find(|t| is_closed_mouth_tag(&t.normalized))
        .cloned();

    if let Some(cm_tok) = closed_mouth_tok {
        let mut conflicts: Vec<PromptToken> = Vec::new();
        for t in &processed_tokens {
            if t.normalized == cm_tok.normalized || is_closed_mouth_tag(&t.normalized) {
                continue;
            }
            if is_closed_mouth_conflict_tag(&t.normalized) {
                conflicts.push(t.clone());
            }
        }
        for c in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::AnatomyConflict,
                reason: "Internal mouth anatomy (teeth, tongue, fang) forces the mouth open.".to_string(),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 4. Single Eyewear Rule: Face-worn vs Head-worn / removed eyewear
    let face_eyewear_idx = processed_tokens.iter().position(|t| is_face_eyewear_tag(&t.normalized));
    let head_eyewear_idx = processed_tokens.iter().position(|t| is_head_or_off_eyewear_tag(&t.normalized));

    if let (Some(f_idx), Some(h_idx)) = (face_eyewear_idx, head_eyewear_idx) {
        let drop_face = h_idx < f_idx;
        let mut conflicts: Vec<PromptToken> = Vec::new();
        for t in &processed_tokens {
            if (drop_face && is_face_eyewear_tag(&t.normalized))
                || (!drop_face && is_head_or_off_eyewear_tag(&t.normalized))
            {
                conflicts.push(t.clone());
            }
        }
        for drop_tok in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != drop_tok.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: drop_tok.text.clone(),
                action: ChangeAction::AnatomyConflict,
                reason: "Combining face-worn and head-worn eyewear tags generates two pairs of glasses simultaneously.".to_string(),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 5. Stature & Ratio Conflicts
    let first_short_idx = processed_tokens.iter().position(|t| is_short_petite_tag(&t.normalized));
    let first_tall_idx = processed_tokens.iter().position(|t| is_tall_statuesque_tag(&t.normalized));

    let stature_winner = match (first_short_idx, first_tall_idx) {
        (Some(s), Some(t)) => {
            if s < t {
                Some((true, processed_tokens[s].clone()))
            } else {
                Some((false, processed_tokens[t].clone()))
            }
        }
        (Some(s), None) => Some((true, processed_tokens[s].clone())),
        (None, Some(t)) => Some((false, processed_tokens[t].clone())),
        (None, None) => None,
    };

    if let Some((is_short, winner_tok)) = stature_winner {
        let mut conflicts: Vec<(PromptToken, &'static str)> = Vec::new();
        for t in &processed_tokens {
            if is_short {
                if is_short_petite_tag(&t.normalized) {
                    continue;
                }
                if is_short_petite_conflict_tag(&t.normalized) {
                    let reason = if matches!(
                        t.normalized.as_str(),
                        "huge weapon"
                            | "giant weapon"
                            | "polearm"
                            | "staff"
                            | "oversized weapon"
                            | "scythe"
                    ) {
                        "Flags oversized props that zoom the camera out."
                    } else {
                        "Prevents mature/tall tags from stretching limb ratios."
                    };
                    conflicts.push((t.clone(), reason));
                }
            } else {
                if is_tall_statuesque_tag(&t.normalized) {
                    continue;
                }
                if is_tall_statuesque_conflict_tag(&t.normalized) {
                    conflicts.push((
                        t.clone(),
                        "Oversized clothing and round-face tags compress the character's head-to-body ratio into a childlike stature.",
                    ));
                }
            }
        }
        for (c, reason) in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::SizeConflict,
                reason: format!("'{}' conflicts with '{}'; {reason}", c.text, winner_tok.text),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 6. Dark Skin Protection
    let dark_skin_tok = processed_tokens
        .iter()
        .find(|t| is_dark_skin_tag(&t.normalized))
        .cloned();

    if let Some(ds_tok) = dark_skin_tok {
        let mut conflicts: Vec<(PromptToken, bool)> = Vec::new();
        for t in &processed_tokens {
            if t.normalized == ds_tok.normalized || is_dark_skin_tag(&t.normalized) {
                continue;
            }
            if is_dark_skin_conflict_tag(&t.normalized) {
                conflicts.push((t.clone(), false));
            } else if is_dark_skin_caution_tag(&t.normalized) {
                conflicts.push((t.clone(), true));
            }
        }
        for (c, is_caution) in conflicts {
            if config.resolve_conflicts_enabled && !is_caution {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::SizeConflict,
                reason: if is_caution {
                    format!("'{}' with '{}': Caution - pale-biased lighting or blush effects can wash out dark skin tones.", c.text, ds_tok.text)
                } else {
                    "Prevents pale-biased lighting or skin tags from washing out dark skin tones.".to_string()
                },
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled && !is_caution { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 7. Comprehensive Breast Size Tier Resolution
    // Whichever breast tier is prompted first sets the character's breast tier.
    let active_tier_tok = processed_tokens
        .iter()
        .find_map(|t| detect_breast_tier(&t.normalized).map(|tier| (tier, t.clone())));

    if let Some((tier, tier_tok)) = active_tier_tok {
        let has_shortstack = processed_tokens.iter().any(|tok| tok.normalized == "shortstack")
            || is_dropdown("shortstack");
        let mut conflicting: Vec<(PromptToken, &'static str, ChangeAction)> = Vec::new();
        for t in &processed_tokens {
            if t.normalized == tier_tok.normalized || is_dropdown(&t.normalized) {
                continue;
            }
            if has_shortstack && (config.exaggerated_proportions || t.normalized == "petite" || t.normalized == "short") {
                continue;
            }
            if let Some((reason, action)) = breast_tier_conflicts(tier, &t.normalized) {
                conflicting.push((t.clone(), reason, action));
            }
        }
        for (c, reason, action) in conflicting {
            let suggested_alt = if tier == BreastTier::FlatChest {
                match c.normalized.as_str() {
                    "virgin killer sweater" => "loose sweater",
                    "cleavage cutout" | "boob window" => "sarashi",
                    "taut shirt" => "loose shirt",
                    "boob pocket" => "chest pocket",
                    _ => "",
                }
            } else {
                ""
            };

            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
                if !suggested_alt.is_empty() {
                    let norm = suggested_alt.to_string();
                    if !processed_tokens.iter().any(|t| t.normalized == norm) {
                        processed_tokens.push(PromptToken {
                            raw: suggested_alt.to_string(),
                            text: suggested_alt.to_string(),
                            normalized: norm,
                            weight: 1.0,
                            had_runaway_weight: false,
                            replaced: true,
                            added: false,
                            original_index: c.original_index,
                            category: TokenCategory::Attire,
                        });
                    }
                }
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action,
                reason: format!("'{}' conflicts with your prompted '{}'; {reason}", c.text, tier_tok.text),
                suggested_alternative: suggested_alt.to_string(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // Pose conflicts with huge/gigantic breasts (e.g. prone / crossed arms)
    let large_chest_tok = processed_tokens
        .iter()
        .find(|t| matches!(t.normalized.as_str(), "huge breasts" | "gigantic breasts" | "massive breasts" | "enormous breasts"))
        .cloned();

    if let Some(large_tok) = large_chest_tok {
        let mut conflicting_poses: Vec<(PromptToken, String, &'static str)> = Vec::new();
        for t in &processed_tokens {
            if matches!(t.normalized.as_str(), "lying on stomach" | "on stomach" | "prone") {
                conflicting_poses.push((
                    t.clone(),
                    format!("Lying flat on stomach causes clipping and severe anatomical distortion with '{}'.", large_tok.text),
                    "",
                ));
            } else if matches!(t.normalized.as_str(), "crossed arms" | "arms crossed") {
                conflicting_poses.push((
                    t.clone(),
                    format!("Crossing arms flat across the chest causes clipping and impossible geometry with '{}'.", large_tok.text),
                    "",
                ));
            }
        }
        for (tok, reason, alt) in conflicting_poses {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != tok.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: tok.text.clone(),
                action: ChangeAction::PoseConflict,
                reason,
                suggested_alternative: alt.to_string(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 8. Body Proportion Conflicts: Thick/Curvy vs Slender/Petite Lower
    let has_shortstack_lower = processed_tokens.iter().any(|t| t.normalized == "shortstack")
        || is_dropdown("shortstack");
    let (first_thick_idx, first_slender_idx) = if config.exaggerated_proportions && has_shortstack_lower {
        (None, None)
    } else {
        (
            processed_tokens.iter().position(|t| is_thick_curvy_lower_tag(&t.normalized)),
            processed_tokens.iter().position(|t| {
                if has_shortstack_lower && (t.normalized == "petite" || t.normalized == "short") {
                    false
                } else {
                    is_slender_petite_lower_tag(&t.normalized)
                }
            }),
        )
    };

    let proportion_lower_winner = match (first_thick_idx, first_slender_idx) {
        (Some(th), Some(sl)) => {
            if th < sl {
                Some((true, processed_tokens[th].clone()))
            } else {
                Some((false, processed_tokens[sl].clone()))
            }
        }
        (Some(th), None) => Some((true, processed_tokens[th].clone())),
        (None, Some(sl)) => Some((false, processed_tokens[sl].clone())),
        (None, None) => None,
    };

    if let Some((is_thick, winner_tok)) = proportion_lower_winner {
        let mut conflicts: Vec<(PromptToken, &'static str)> = Vec::new();
        let spread = processed_tokens.iter().any(|t| legs_apart(&t.normalized));
        for t in &processed_tokens {
            if is_thick {
                if is_thick_curvy_lower_tag(&t.normalized) {
                    continue;
                }
                // spread legs separate even thick thighs, so a thigh gap fits that pose
                if spread && t.normalized == "thigh gap" {
                    continue;
                }
                if is_thick_curvy_conflict(&t.normalized) {
                    conflicts.push((
                        t.clone(),
                        "Prevents thin/slender tags from neutralizing hip and thigh volume.",
                    ));
                }
            } else {
                if is_slender_petite_lower_tag(&t.normalized) {
                    continue;
                }
                if is_slender_petite_conflict(&t.normalized) {
                    conflicts.push((
                        t.clone(),
                        "Strips volume and compression tags (like skindentation) that force thick thighs.",
                    ));
                }
            }
        }
        for (c, reason) in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::SizeConflict,
                reason: format!("'{}' conflicts with '{}'; {reason}", c.text, winner_tok.text),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // 9. Body Proportion Conflicts: Muscular/Toned vs Soft/Chubby
    let first_muscular_idx = processed_tokens.iter().position(|t| is_muscular_toned_tag(&t.normalized));
    let first_soft_idx = processed_tokens.iter().position(|t| is_soft_chubby_tag(&t.normalized));

    let physique_winner = match (first_muscular_idx, first_soft_idx) {
        (Some(m), Some(s)) => {
            if m < s {
                Some((true, processed_tokens[m].clone()))
            } else {
                Some((false, processed_tokens[s].clone()))
            }
        }
        (Some(m), None) => Some((true, processed_tokens[m].clone())),
        (None, Some(s)) => Some((false, processed_tokens[s].clone())),
        (None, None) => None,
    };

    if let Some((is_muscular, winner_tok)) = physique_winner {
        let mut conflicts: Vec<(PromptToken, &'static str)> = Vec::new();
        for t in &processed_tokens {
            if is_muscular {
                if is_muscular_toned_tag(&t.normalized) {
                    continue;
                }
                if is_muscular_conflict(&t.normalized) {
                    conflicts.push((
                        t.clone(),
                        "Prevents body-fat tags from smoothing out muscle definition.",
                    ));
                }
            } else {
                if is_soft_chubby_tag(&t.normalized) {
                    continue;
                }
                if is_soft_chubby_conflict(&t.normalized) {
                    conflicts.push((
                        t.clone(),
                        "Prevents fitness and slim-waist tags from erasing soft midsection anatomy.",
                    ));
                }
            }
        }
        for (c, reason) in conflicts {
            if config.resolve_conflicts_enabled {
                processed_tokens.retain(|t| t.normalized != c.normalized);
            }
            changes.push(PromptChange {
                id: next_change_id,
                original: c.text.clone(),
                action: ChangeAction::SizeConflict,
                reason: format!("'{}' conflicts with '{}'; {reason}", c.text, winner_tok.text),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if config.resolve_conflicts_enabled { 0 } else { 2 },
            });
            next_change_id += 1;
        }
    }

    // A generic tag next to a more specific one that implies it (`swimsuit` + `bikini`, `sword` +
    // `holding sword`): the generic one goes, with a card to keep it. Weighted / pinned tags stay.
    if config.cleanup_enabled {
        let norms: Vec<String> = processed_tokens.iter().map(|t| t.normalized.clone()).collect();
        let mut drop: Vec<usize> = Vec::new();
        for (i, t) in processed_tokens.iter().enumerate() {
            if (t.weight - 1.0).abs() > f32::EPSILON || t.category == TokenCategory::PinnedEmphasis || t.added {
                continue;
            }
            let Some(child) = norms.iter().enumerate().find(|(j, o)| *j != i && implies(o, &t.normalized)).map(|(j, _)| j) else {
                continue;
            };
            changes.push(PromptChange {
                id: next_change_id,
                original: t.text.clone(),
                action: ChangeAction::ImpliedTag,
                reason: format!(
                    "'{}' already means '{}', so the generic tag only repeats it.",
                    processed_tokens[child].text, t.text
                ),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: 0,
            });
            next_change_id += 1;
            drop.push(i);
        }
        let mut i = 0;
        processed_tokens.retain(|_| {
            let keep = !drop.contains(&i);
            i += 1;
            keep
        });
    }

    // 8.5 Mutually Exclusive Taxonomy Groups Resolution (Mexes Danbooru Taxonomy)
    let has_shortstack_taxonomy = processed_tokens.iter().any(|tok| tok.normalized == "shortstack")
        || is_dropdown("shortstack");
    resolve_exclusive_taxonomy_groups(
        &mut processed_tokens,
        &mut changes,
        &mut next_change_id,
        config.resolve_conflicts_enabled,
        config.exaggerated_proportions && has_shortstack_taxonomy,
    );

    // Two sizes for the same body part (or two hair lengths) can't both be right:
    // keep the one written first, with a card to bring the other back.
    if config.resolve_conflicts_enabled {
        let mut kept: Vec<PromptToken> = Vec::new();
        for t in processed_tokens {
            // a Bust / Frame choice beats a size the prompt wrote
            if is_dropdown(&t.normalized) {
                if let Some(pos) = kept.iter().position(|k| size_conflict(&k.normalized, &t.normalized) && !is_dropdown(&k.normalized)) {
                    let k = kept.remove(pos);
                    changes.push(PromptChange {
                        id: next_change_id,
                        original: k.text.clone(),
                        action: ChangeAction::SizeConflict,
                        reason: format!("'{}' contradicts your choice '{}'.", k.text, t.text),
                        suggested_alternative: String::new(),
                        framing_alternative: String::new(),
                        mode: 0,
                    });
                    next_change_id += 1;
                    kept.push(t);
                    continue;
                }
            }
            match kept.iter().find(|k| size_conflict(&k.normalized, &t.normalized)) {
                Some(k) => {
                    changes.push(PromptChange {
                        id: next_change_id,
                        original: t.text.clone(),
                        action: ChangeAction::SizeConflict,
                        reason: format!("'{}' and '{}' contradict each other; kept '{}', written first.", k.text, t.text, k.text),
                        suggested_alternative: String::new(),
                        framing_alternative: String::new(),
                        mode: 0,
                    });
                    next_change_id += 1;
                }
                None => kept.push(t),
            }
        }
        processed_tokens = kept;

        // A slender body type next to a big chest / hips: Danbooru's "petite" (and slim,
        // slender, skinny) comes with a small chest in training, so the model shrinks
        // "large breasts". "short" keeps a height difference without that pull.
        // Not for a Bust / Frame choice, and "shortstack" means short and curvy, so its petite stays.
        let curvy = processed_tokens.iter().any(|t| is_big_body_tag(&t.normalized));
        let shortstack = processed_tokens.iter().any(|t| t.normalized == "shortstack") || is_dropdown("shortstack");
        if curvy && !(shortstack && config.exaggerated_proportions) {
            for t in processed_tokens.iter_mut() {
                if !SLENDER_TAGS.contains(&t.normalized.as_str())
                    || is_dropdown(&t.normalized)
                    || (shortstack && t.normalized == "petite")
                {
                    continue;
                }
                let original = t.text.clone();
                let swap = if original == "petite" { "short" } else { "" };
                changes.push(PromptChange {
                    id: next_change_id,
                    original: original.clone(),
                    action: ChangeAction::SizeConflict,
                    reason: format!(
                        "'{original}' means a small, slender body, so the model shrinks the chest / hips you asked for. {}",
                        if swap.is_empty() { "Removed.".to_string() } else { format!("Replaced with '{swap}' to keep the height.") }
                    ),
                    suggested_alternative: swap.to_string(),
                    framing_alternative: String::new(),
                    mode: 0,
                });
                next_change_id += 1;
                t.text = swap.to_string();
                t.raw = swap.to_string();
                t.normalized = swap.to_string();
                t.replaced = true;
            }
            processed_tokens.retain(|t| !t.normalized.is_empty());
        }
    }

    // The other conflict rules (stature, breast tiers, thick vs slender, taxonomy) never remove a
    // Bust / Frame choice: put back any they took, and drop their cards about it.
    for t in &dropdown_body {
        if !processed_tokens.iter().any(|e| e.normalized == t.normalized) {
            changes.retain(|c| !(c.original == t.text && c.action != ChangeAction::ProportionChoice));
            let mut t = t.clone();
            t.added = true;
            t.original_index = 940;
            t.category = TokenCategory::FramingPose;
            processed_tokens.push(t);
        }
    }

    // The hips lock puts `thigh gap` in the negative (condenser::recommended_negative), except with the legs apart:
    // say so, so a missing `thigh gap` in the negative isn't a surprise
    if processed_tokens.iter().any(|t| matches!(t.normalized.as_str(), "thick thighs" | "wide hips"))
        && processed_tokens.iter().any(|t| legs_apart(&t.normalized))
    {
        changes.push(PromptChange {
            id: next_change_id,
            original: String::new(),
            action: ChangeAction::ProportionChoice,
            reason: "Legs apart: 'thigh gap' is left out of the negative, so it doesn't fight the pose (spread thighs separate).".to_string(),
            suggested_alternative: String::new(),
            framing_alternative: String::new(),
            mode: 0,
        });
        next_change_id += 1;
    }

    // "Exaggerated": the Bust / Frame picks get their emphasis (`body_emphasis`). Set after the clamp,
    // on purpose: the user picked them, and a typed weight that's already higher stays.
    if config.exaggerated_proportions {
        let bust_tags: Vec<String> = crate::profiles::BUSTS
            .get(config.bust)
            .filter(|_| config.bust > 0)
            .map(|(_, tags)| crate::parser::parse_prompt(tags).into_iter().map(|t| t.normalized).collect())
            .unwrap_or_default();
        for t in processed_tokens.iter_mut().filter(|t| is_dropdown(&t.normalized)) {
            let w = body_emphasis(config.checkpoint, &t.normalized, bust_tags.contains(&t.normalized), config.frame);
            if w > t.weight {
                t.weight = w;
            }
        }
    }

    // 10. Proportion Chain Recommendations
    if config.fit_suggestions {
        let prop_tiers = detect_all_proportion_tiers(&processed_tokens);
        let has_shirt = processed_tokens.iter().any(|t| {
            matches!(
                t.normalized.as_str(),
                "shirt" | "blouse" | "top" | "sweater" | "turtleneck" | "jacket" | "uniform" | "school uniform" | "dress shirt"
            )
        });
        let has_legwear = processed_tokens.iter().any(|t| {
            matches!(
                t.normalized.as_str(),
                "thighhighs" | "kneehighs" | "socks" | "stockings" | "pantyhose" | "legwear"
            ) || ["thighhighs", "kneehighs", "socks", "stockings", "pantyhose", "legwear"]
                .iter()
                .any(|w| t.normalized.ends_with(&format!(" {w}")))
        });

        for (tier, tier_tok) in prop_tiers {
            let rec = get_proportion_recommendations(tier);

            // Filter clothing tags to only match garments currently worn in prompt
            let valid_clothing: Vec<&str> = rec.clothing_interaction.iter().copied().filter(|c| {
                if *c == "skindentation" {
                    has_legwear
                } else if c.contains("shirt") || c.contains("clothes") || c.contains("button") {
                    has_shirt
                } else {
                    false
                }
            }).collect();

            let pretty = |v: &[&str]| v.iter().map(|t| t.replace('_', " ")).collect::<Vec<_>>().join(", ");
            let clothing_str = pretty(&valid_clothing);
            let pose_str = pretty(rec.pose_and_physics);

            // auto-negative lock filtering: if shortstack is present, do not negative-lock "petite"
            let is_shortstack = processed_tokens.iter().any(|t| t.normalized == "shortstack");
            let neg_list: Vec<&str> = rec.auto_negative_lock.iter().copied().filter(|n| {
                if is_shortstack && *n == "petite" {
                    false
                } else {
                    true
                }
            }).collect();
            let neg_str = pretty(&neg_list);

            let mut top_companions = Vec::new();
            for c in valid_clothing.iter().take(2) {
                top_companions.push(c.replace('_', " "));
            }
            for p in rec.pose_and_physics.iter().take(2) {
                top_companions.push(p.replace('_', " "));
            }
            let suggested_alt = top_companions.join(", ");

            if !suggested_alt.is_empty() {
                changes.push(PromptChange {
                    id: next_change_id,
                    original: String::new(),
                    action: ChangeAction::Recommendation,
                    reason: {
                        // only the lines that have something (fit tags exist only for clothes in the prompt)
                        let mut r = format!("Reinforce '{}':", tier_tok.text);
                        for (label, v) in [("Fit", &clothing_str), ("Pose", &pose_str), ("Negative locks", &neg_str)] {
                            if !v.is_empty() {
                                r.push_str(&format!("\n• {label}: {v}"));
                            }
                        }
                        r
                    },
                    suggested_alternative: suggested_alt,
                    framing_alternative: String::new(),
                    mode: 2, // Informational by default; user can click 'Add It' to insert suggested companion tags
                });
                next_change_id += 1;
            }
        }
    }

    // 11. Front-View Camera Angle Protection Card
    let has_front_view = processed_tokens.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "front view" || n == "straight-on" || n == "straight on" || n == "facing viewer"
    });
    let has_back_biased_pose = processed_tokens.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "arms up" || n == "sitting on bed" || n == "on bed" || n == "lying on back" || n == "lying on bed"
    });
    if has_front_view && has_back_biased_pose {
        changes.push(PromptChange {
            id: next_change_id,
            original: "front view".to_string(),
            action: ChangeAction::Recommendation,
            reason: "Front-View Angle Lock: Poses like 'arms up, sitting on bed' carry a strong rear-angle bias in diffusion models. Added negative locks ('from behind, back view, over the shoulder, looking back') to guarantee a straight-on front angle.".to_string(),
            suggested_alternative: "straight-on, facing viewer".to_string(),
            framing_alternative: String::new(),
            mode: 2,
        });
        next_change_id += 1;
    }

    // 12. Directive 6 Prop-Preservation & Leg Grounding Card
    let has_prop_action = processed_tokens.iter().any(|t| {
        let n = t.normalized.as_str();
        n == "pulling up stocking" || n == "adjusting legwear" || n == "pulling up thighhighs"
    });
    if has_prop_action {
        let has_stocking = processed_tokens.iter().any(|t| t.normalized.contains("thighhighs") || t.normalized.contains("stocking"));
        let has_barefoot = processed_tokens.iter().any(|t| t.normalized == "bare foot" || t.normalized == "barefoot" || t.normalized == "no shoes");
        if !has_stocking || !has_barefoot {
            let mut companions = Vec::new();
            if !has_stocking { companions.push("white thighhighs"); }
            if !has_barefoot { companions.push("bare foot"); }
            changes.push(PromptChange {
                id: next_change_id,
                original: String::new(),
                action: ChangeAction::Recommendation,
                reason: "Directive 6 Prop Preservation: Active legwear adjustments require keeping the interacted garment prop and grounding the leg with barefoot anatomy.".to_string(),
                suggested_alternative: companions.join(", "),
                framing_alternative: String::new(),
                mode: 2,
            });
            next_change_id += 1;
        }
    }

    // 13. Redundant Scene Setting & Bed Surface Cleanup
    if config.cleanup_enabled {
        let has_sitting_on_bed = processed_tokens.iter().any(|t| t.normalized == "sitting on bed");
        if has_sitting_on_bed {
            if let Some(pos) = processed_tokens.iter().position(|t| t.normalized == "on bed") {
                let tok = processed_tokens.remove(pos);
                changes.push(PromptChange {
                    id: next_change_id,
                    original: tok.text,
                    action: ChangeAction::RemovedBloat,
                    reason: "'on bed' is redundant when 'sitting on bed' is already present.".to_string(),
                    suggested_alternative: String::new(),
                    framing_alternative: String::new(),
                    mode: 0,
                });
                let _ = next_change_id;
            }
        }
    }

    // 14. Combo Synergy Recommendation Cards (Mexes Danbooru Taxonomy & 8_combo_synergy_matrix)
    if config.fit_suggestions {
        generate_combo_synergy_cards(&processed_tokens, &mut changes, &mut next_change_id);
    }

    RuleAnalysisResult {
        processed_tokens,
        changes,
        active_framing_detected: effective_view,
    }
}

/// The same body-type fix for text the AI wrote: within one character's tags, a slender
/// tag next to a big chest / hips goes ("petite" becomes "short").
#[cfg(test)]
pub fn fix_body_type(tokens: Vec<PromptToken>) -> Vec<PromptToken> {
    fix_body_type_keeping(tokens, &[])
}

/// `fix_body_type`, but tags in `keep` (normalized: the Bust / Frame dropdown picks, `chosen_body_tags`)
/// are never changed, the same protection the rule path gives them.
pub fn fix_body_type_keeping(tokens: Vec<PromptToken>, keep: &[String]) -> Vec<PromptToken> {
    if !tokens.iter().any(|t| is_big_body_tag(&t.normalized)) {
        return tokens;
    }
    // "shortstack" means short and curvy: its petite stays
    let shortstack = tokens.iter().any(|t| t.normalized == "shortstack");
    tokens
        .into_iter()
        .filter_map(|mut t| {
            if !SLENDER_TAGS.contains(&t.normalized.as_str())
                || (shortstack && t.normalized == "petite")
                || keep.contains(&t.normalized)
            {
                return Some(t);
            }
            if t.normalized != "petite" {
                return None;
            }
            t.text = "short".into();
            t.raw = "short".into();
            t.normalized = "short".into();
            Some(t)
        })
        .collect()
}

/// The Bust / Frame dropdown picks' tags (normalized), minus any the LoRA box overrides (the LoRA box beats
/// a dropdown, as in `analyze_with_rules`).
pub fn chosen_body_tags(config: &ScanConfig) -> Vec<String> {
    let loras = crate::parser::parse_prompt(&config.lora_tags_raw);
    [(crate::profiles::BUSTS, config.bust), (crate::profiles::FRAMES, config.frame)]
        .into_iter()
        .filter_map(|(list, i)| list.get(i).filter(|_| i > 0))
        .flat_map(|(_, tags)| crate::parser::parse_prompt(tags))
        .map(|t| t.normalized)
        .filter(|n| !loras.iter().any(|l| l.normalized == *n || tags_conflict(&l.normalized, n)))
        .collect()
}

/// Puts the Bust / Frame dropdown picks into AI text (one character's tags): the text's own tags of that
/// list go (for a bust, any other breast size too), the chosen tags are added once at the end, then the
/// body-type fix runs without touching them. Returns the text unchanged when nothing had to change.
/// The weight an "Exaggerated" Bust / Frame pick gets (1.0 = none). Measured by the user on Anima, where
/// `(gigantic breasts:2.5)` gave the wanted size and several heavy weights at once broke the anatomy, so only
/// the bust and the frame's key tags are emphasized. CLIP checkpoints burn above ~1.5, so they get less;
/// Flux ignores weights.
pub fn body_emphasis(checkpoint: crate::profiles::CheckpointProfile, tag: &str, is_bust: bool, frame: usize) -> f32 {
    use crate::profiles::CheckpointProfile as C;
    let key_frame_tag = matches!(tag, "shortstack" | "wide hips" | "thick thighs" | "muscular female");
    let label = crate::condenser::frame_label(frame);
    match checkpoint {
        C::Flux => 1.0,
        // the waist cinch of the figures that have one (an hourglass leans on it most), and Muscular's abs
        C::Anima if tag == "narrow waist" && label == "hourglass" => 1.3,
        _ if tag == "narrow waist" && matches!(label, "shortstack" | "pear" | "hourglass") => 1.2,
        _ if tag == "abs" && label == "muscular" => 1.2,
        C::Anima if is_bust => 2.5,
        C::Anima if tag == "shortstack" => 1.4,
        C::Anima if key_frame_tag => 1.5,
        _ if is_bust => 1.4,
        _ if key_frame_tag => 1.2,
        _ => 1.0,
    }
}

#[cfg(test)]
pub fn apply_body_choices(text: &str, bust: usize, frame: usize) -> String {
    apply_body_choices_with(text, bust, frame, None)
}

/// `apply_body_choices` with the "Exaggerated" emphasis (`body_emphasis`) for `exaggerate = Some(checkpoint)`.
pub fn apply_body_choices_with(
    text: &str,
    bust: usize,
    frame: usize,
    exaggerate: Option<crate::profiles::CheckpointProfile>,
) -> String {
    let mut tokens = crate::parser::parse_prompt(text);
    let before: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
    let mut keep: Vec<String> = Vec::new();
    for (list, idx, is_bust) in [(crate::profiles::BUSTS, bust, true), (crate::profiles::FRAMES, frame, false)] {
        let Some((_, tags)) = list.get(idx).filter(|_| idx > 0) else { continue };
        let group = crate::profiles::group_tags(list);
        let chosen = crate::parser::parse_prompt(tags);
        let is_chosen = |n: &str| chosen.iter().any(|c| c.normalized == n);
        let unwanted = |n: &str| (group.iter().any(|g| g == n) || (is_bust && detect_breast_tier(n).is_some())) && !is_chosen(n);
        // a dynamic group `{large breasts|huge breasts}` loses its unwanted options (and goes when none is left)
        for t in tokens.iter_mut() {
            let Some(inner) = t.raw.trim().strip_prefix('{').and_then(|r| r.strip_suffix('}')) else { continue };
            let opts: Vec<&str> = inner.split('|').collect();
            let kept: Vec<&str> = opts.iter().copied().filter(|o| !unwanted(&crate::parser::parse_prompt(o).first().map(|p| p.normalized.clone()).unwrap_or_default())).collect();
            if kept.len() == opts.len() {
                continue;
            }
            let raw = match kept.len() {
                0 => String::new(),
                1 => kept[0].trim().to_string(),
                _ => format!("{{{}}}", kept.join("|")),
            };
            *t = crate::parser::parse_prompt(&raw).into_iter().next().unwrap_or_else(|| PromptToken { raw: String::new(), ..t.clone() });
        }
        tokens.retain(|t| !t.raw.is_empty() && !unwanted(&t.normalized));
        let chosen_norms: Vec<String> = chosen.iter().map(|c| c.normalized.clone()).collect();
        for c in chosen {
            keep.push(c.normalized.clone());
            if !tokens.iter().any(|t| t.normalized == c.normalized) {
                tokens.push(c);
            }
        }
        if let Some(cp) = exaggerate {
            for t in tokens.iter_mut().filter(|t| chosen_norms.contains(&t.normalized)) {
                let w = body_emphasis(cp, &t.normalized, is_bust, frame);
                if w > t.weight {
                    t.weight = w;
                    t.raw = t.format_for_output(true);
                }
            }
        }
    }
    let tokens = fix_body_type_keeping(tokens, &keep);
    let after: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
    if after == before {
        return text.to_string();
    }
    after.join(", ")
}

/// Body-type tags that pull toward a small, slender figure.
const SLENDER_TAGS: &[&str] = &["petite", "slim", "slender", "skinny", "thin"];

/// Tags asking for a big chest, hips or a curvy figure.
pub fn is_big_body_tag(n: &str) -> bool {
    let w: Vec<&str> = n.split_whitespace().collect();
    let big = w.iter().any(|x| matches!(*x, "large" | "huge" | "gigantic" | "big" | "enormous" | "massive" | "wide" | "thick"));
    let part = w.last().map(|x| matches!(*x, "breasts" | "ass" | "hips" | "thighs" | "butt")).unwrap_or(false);
    (big && part) || matches!(n, "curvy" | "voluptuous" | "plump" | "hourglass figure" | "busty")
}

/// Words that set one attribute of a body part or feature. Two tags about the same thing
/// (same last word) that both set the same attribute contradict each other.
const SIZE_WORDS: &[&str] = &[
    "flat", "tiny", "small", "petite", "medium", "large", "big", "huge", "gigantic", "giant", "enormous",
    "massive", "hyper", "wide", "narrow", "thick", "slim", "plump",
];
const LENGTH_WORDS: &[&str] = &["short", "medium", "long", "very", "absurdly", "shoulder-length", "bob"];
const COLOR_WORDS: &[&str] = &[
    "black", "white", "silver", "grey", "gray", "blonde", "blond", "brown", "red", "blue", "green", "pink",
    "purple", "orange", "yellow", "aqua", "platinum", "multicolored", "two-tone", "gradient", "streaked",
    "dark", "light", "golden", "teal", "violet", "crimson", "auburn",
    // Common anime eye, hair, and skin colors
    "amber", "hazel", "cyan", "magenta", "turquoise", "lavender", "maroon", "emerald", "sapphire", "ruby",
    "beige", "tan", "tanned", "pale", "fair", "copper", "bronze", "lilac", "indigo", "charcoal", "navy",
    "olive", "lime", "peach", "coral", "mint", "rose", "scarlet", "ivory", "brunette",
];

fn head_word(word: &str) -> &str {
    match word {
        "chest" | "breast" | "boobs" | "tits" => "breasts",
        "butt" | "buttocks" => "ass",
        "eye" => "eyes",
        "thigh" => "thighs",
        "ear" => "ears",
        "tail" => "tail",
        "pupil" => "pupils",
        "eyebrow" => "eyebrows",
        "eyelash" => "eyelashes",
        "lip" => "lips",
        w => w,
    }
}

/// True when `tag` contradicts `keep` (a LoRA tag): same subject, same attribute,
/// different value ("gigantic breasts" vs "large breasts", "black hair" vs "silver hair",
/// "amber eyes" vs "grey eyes").
/// Long clauses (prose) are never treated as conflicts.
pub fn tags_conflict(keep: &str, tag: &str) -> bool {
    if crate::prompt_condenser::can_coexist(keep, tag) {
        return false;
    }
    if crate::prompt_condenser::is_occluded_by(keep, tag) {
        return true;
    }
    if attribute_conflict(keep, tag, &[SIZE_WORDS, LENGTH_WORDS, COLOR_WORDS]) {
        return true;
    }
    // Species and Anatomy Purity (Kemonomimi)
    if (is_kemonomimi_tag(keep) && (is_furry_bleed_tag(tag) || is_four_ears_trigger_tag(tag)))
        || ((is_furry_bleed_tag(keep) || is_four_ears_trigger_tag(keep)) && is_kemonomimi_tag(tag))
    {
        return true;
    }
    // Facial State Conflicts
    if (is_closed_eyes_tag(keep) && is_closed_eyes_conflict_tag(tag))
        || (is_closed_eyes_conflict_tag(keep) && is_closed_eyes_tag(tag))
    {
        return true;
    }
    if (is_closed_mouth_tag(keep) && is_closed_mouth_conflict_tag(tag))
        || (is_closed_mouth_conflict_tag(keep) && is_closed_mouth_tag(tag))
    {
        return true;
    }
    if (is_face_eyewear_tag(keep) && is_head_or_off_eyewear_tag(tag))
        || (is_head_or_off_eyewear_tag(keep) && is_face_eyewear_tag(tag))
    {
        return true;
    }
    // Stature and Ratio Conflicts
    if (is_short_petite_tag(keep) && is_short_petite_conflict_tag(tag))
        || (is_short_petite_conflict_tag(keep) && is_short_petite_tag(tag))
    {
        return true;
    }
    if (is_tall_statuesque_tag(keep) && is_tall_statuesque_conflict_tag(tag))
        || (is_tall_statuesque_conflict_tag(keep) && is_tall_statuesque_tag(tag))
    {
        return true;
    }
    // Skin Tone Protection
    if (is_dark_skin_tag(keep) && is_dark_skin_conflict_tag(tag))
        || (is_dark_skin_conflict_tag(keep) && is_dark_skin_tag(tag))
    {
        return true;
    }
    // Breast Size Tier Rules
    if let Some(tier) = detect_breast_tier(keep) {
        if breast_tier_conflicts(tier, tag).is_some() {
            return true;
        }
    }
    if let Some(tier) = detect_breast_tier(tag) {
        if breast_tier_conflicts(tier, keep).is_some() {
            return true;
        }
    }
    // Body Proportion Conflicts
    if (is_thick_curvy_lower_tag(keep) && is_thick_curvy_conflict(tag))
        || (is_thick_curvy_conflict(keep) && is_thick_curvy_lower_tag(tag))
    {
        return true;
    }
    if (is_slender_petite_lower_tag(keep) && is_slender_petite_conflict(tag))
        || (is_slender_petite_conflict(keep) && is_slender_petite_lower_tag(tag))
    {
        return true;
    }
    if (is_muscular_toned_tag(keep) && is_muscular_conflict(tag))
        || (is_muscular_conflict(keep) && is_muscular_toned_tag(tag))
    {
        return true;
    }
    if (is_soft_chubby_tag(keep) && is_soft_chubby_conflict(tag))
        || (is_soft_chubby_conflict(keep) && is_soft_chubby_tag(tag))
    {
        return true;
    }
    false
}

/// Two sizes or lengths of the same thing: "medium breasts" / "large breasts",
/// "flat chest" / "huge breasts", "short hair" / "long hair". Colors don't count
/// (multicolored or streaked hair really has two), and "long hair" + "very long hair"
/// is how Danbooru tags it, so that pair is fine.
/// "+ Quality": puts the checkpoint's quality tags (`canonical_quality_tags`) that Before lacks in
/// front of it. An explicit button, so it isn't the "never add tags" rule's business. No rating tags.
pub fn add_quality_tags(prompt: &str, checkpoint: crate::profiles::CheckpointProfile) -> (String, Vec<&'static str>) {
    // "score_9", "score 9", "(score_9:1.1)" and "[score_9]" are the same tag
    let norm = |t: &str| t.trim().trim_matches(['(', ')', '[', ']']).split(':').next().unwrap_or("").trim().to_lowercase().replace('_', " ");
    let present: Vec<String> = prompt.split([',', '\n']).map(norm).collect();
    let missing: Vec<&'static str> =
        checkpoint.canonical_quality_tags().iter().copied().filter(|q| !present.contains(&norm(q))).collect();
    if missing.is_empty() {
        return (prompt.to_string(), missing);
    }
    let rest = prompt.trim_start();
    let text = if rest.is_empty() { missing.join(", ") } else { format!("{}, {rest}", missing.join(", ")) };
    (text, missing)
}

/// Generic tag → the specific tags that imply it: (generic, exact children, child suffixes). Only
/// clear parent / child pairs; a suffix child is `<words> <suffix>` (`witch hat`, `red bikini`).
const IMPLIED_PARENTS: &[(&str, &[&str], &[&str])] = &[
    ("swimsuit", &["bikini", "bikini top", "bikini bottom", "bikini top only"], &[" swimsuit", " bikini"]),
    ("bikini", &[], &[" bikini"]),
    ("sword", &["katana"], &[" sword"]),
    ("weapon", &["sword", "katana", "gun", "rifle", "handgun"], &[" weapon", " sword", " gun"]),
    ("gun", &["handgun", "rifle", "assault rifle", "shotgun"], &[" gun"]),
    ("hat", &[], &[" hat"]),
    ("horns", &[], &[" horns"]),
    ("wings", &[], &[" wings"]),
    ("glasses", &[], &[" glasses"]),
    ("dress", &[], &[" dress"]),
    ("skirt", &["miniskirt"], &[" skirt"]),
    ("shirt", &[], &[" shirt"]),
    ("jacket", &[], &[" jacket"]),
    ("boots", &[], &[" boots"]),
    ("gloves", &[], &[" gloves"]),
    ("thighhighs", &[], &[" thighhighs"]),
    ("footwear", &["shoes", "boots", "loafers", "sneakers", "sandals", "high heels", "slippers", "geta", "zori", "uwabaki", "mary janes"], &[" footwear", " shoes", " boots"]),
    ("shoes", &["loafers", "sneakers", "sandals", "high heels", "slippers", "geta", "zori", "ballet slippers", "mary janes", "uwabaki"], &[" shoes"]),
    ("sitting", &["seiza", "wariza", "indian style"], &[]),
    ("kneeling", &["seiza"], &[]),
    ("supporting own breasts", &["two-handed breast hold", "grabbing own breast", "breast hold", "hands on own chest"], &[]),
];

/// Whether the tag `child` implies the generic tag `generic` (`bikini` → `swimsuit`, `holding sword`
/// → `sword`, `witch hat` → `hat`). Absent / removed items imply nothing (`no hat`, `unworn hat`), and
/// "holding …" implies only weapons (a held hat isn't a worn one).
pub fn implies(child: &str, generic: &str) -> bool {
    if child == generic {
        return false;
    }
    let Some((_, exact, suffixes)) = IMPLIED_PARENTS.iter().find(|(g, _, _)| *g == generic) else {
        return false;
    };
    if ["no ", "without ", "unworn ", "removed ", "missing ", "broken "].iter().any(|p| child.starts_with(p)) {
        return false;
    }
    let weapon = matches!(generic, "sword" | "weapon" | "gun");
    if child.starts_with("holding ") && !weapon {
        return false;
    }
    let base = child.strip_prefix("holding ").filter(|_| weapon).unwrap_or(child);
    exact.contains(&base) || suffixes.iter().any(|s| child.ends_with(s)) || (weapon && base == generic)
}

pub fn size_conflict(a: &str, b: &str) -> bool {
    attribute_conflict(a, b, &[SIZE_WORDS, LENGTH_WORDS])
}

/// A body size or proportion tag ("large breasts", "thick thighs", "wide hips"), not
/// a face detail like "thick eyebrows".
#[allow(dead_code)]
pub fn is_size_tag(tag: &str) -> bool {
    const BODY: &[&str] = &[
        "breasts", "ass", "hips", "thighs", "waist", "figure", "body", "belly", "stomach", "penis", "testicles",
        "areolae", "nipples", "legs", "arms", "shoulders", "muscles",
    ];
    let words: Vec<&str> = tag.split_whitespace().collect();
    let Some(last) = words.last() else { return false };
    words.len() <= 3 && BODY.contains(&head_word(last)) && words.iter().any(|w| SIZE_WORDS.contains(w))
}

fn attribute_conflict(keep: &str, tag: &str, classes: &[&[&str]]) -> bool {
    if keep == tag || tag.contains(keep) || keep.contains(tag) {
        return false;
    }
    let kw: Vec<&str> = keep.split_whitespace().collect();
    let tw: Vec<&str> = tag.split_whitespace().collect();
    if kw.len() < 2 || tw.len() < 2 || tw.len() > 4 {
        return false;
    }
    if head_word(kw[kw.len() - 1]) != head_word(tw[tw.len() - 1]) {
        return false;
    }
    let (km, tm) = (&kw[..kw.len() - 1], &tw[..tw.len() - 1]);
    classes.iter().any(|class| km.iter().any(|w| class.contains(w)) && tm.iter().any(|w| class.contains(w)))
}

/// Puts the LoRA box's text into a finished prompt word for word, right after the
/// leading quality tags, and drops the prompt's own copies of those tags.
/// With `strip_conflicts`, tags that contradict a LoRA tag are dropped too (used on AI
/// text; the rule output already handled them as cards).
pub fn insert_lora_block(text: &str, config: &ScanConfig, strip_conflicts: bool) -> String {
    let block = config.lora_tags_raw.trim().trim_matches(',').trim();
    if block.is_empty() {
        return text.to_string();
    }
    let parts = without_lora_tags(text, config, strip_conflicts);
    let at = parts.iter().take_while(|t| t.category == TokenCategory::QualityMeta).count();
    let mut out: Vec<String> = parts.iter().map(|t| t.raw.trim().to_string()).collect();
    out.insert(at, block.to_string());
    out.join(", ")
}

/// The prompt's tags minus copies of the LoRA box's tags (and, with `strip_conflicts`,
/// minus tags that contradict them).
pub fn without_lora_tags(text: &str, config: &ScanConfig, strip_conflicts: bool) -> Vec<PromptToken> {
    let tokens = strip_lora_tokens(crate::parser::parse_prompt(text), &config.lora_tags_raw, strip_conflicts);
    // AI text gets the same body-type fix the rule cards apply (never on a Bust / Frame pick)
    if strip_conflicts { fix_body_type_keeping(tokens, &chosen_body_tags(config)) } else { tokens }
}

/// `tokens` minus copies of the tags in `lora_raw` (and, with `strip_conflicts`, minus
/// tags that contradict them).
pub fn strip_lora_tokens(tokens: Vec<PromptToken>, lora_raw: &str, strip_conflicts: bool) -> Vec<PromptToken> {
    let loras = crate::parser::parse_prompt(lora_raw);
    tokens
        .into_iter()
        .filter(|t| !loras.iter().any(|l| l.normalized == t.normalized))
        .filter(|t| !strip_conflicts || !loras.iter().any(|l| tags_conflict(&l.normalized, &t.normalized)))
        .collect()
}

/// Final touches for any After view: the LoRA box, the natural-language sentence
/// (only when on and the checkpoint understands it) and the arranged layout.
pub fn finish_prompt(text: &str, config: &ScanConfig, strip_conflicts: bool, sentence: Option<&str>) -> String {
    let sentence = sentence
        .map(str::trim)
        .filter(|s| config.natural && config.checkpoint.supports_natural_language() && !s.is_empty());
    if config.arranged && config.checkpoint != crate::profiles::CheckpointProfile::Flux {
        let tokens = without_lora_tags(text, config, strip_conflicts);
        return crate::scene::arrange(&tokens, &config.lora_tags_raw, sentence);
    }
    let t = insert_lora_block(text, config, strip_conflicts);
    match sentence {
        Some(s) if !t.contains(s) => {
            let t = t.trim().trim_end_matches(',').trim();
            if t.is_empty() { s.to_string() } else { format!("{t}, {s}") }
        }
        _ => t,
    }
}

pub fn resolve_exclusive_taxonomy_groups(
    tokens: &mut Vec<PromptToken>,
    changes: &mut Vec<PromptChange>,
    next_change_id: &mut usize,
    resolve_conflicts: bool,
    bypass_proportions: bool,
) {
    // 1. Stature: Tall vs Short
    const TALL_TAGS: &[&str] = &["tall female", "tall", "long legs"];
    const SHORT_TAGS: &[&str] = &["short female", "short legs"];
    let has_tall = tokens.iter().find(|t| TALL_TAGS.contains(&t.normalized.as_str())).cloned();
    if let Some(tall_tok) = has_tall {
        let losers: Vec<PromptToken> = tokens
            .iter()
            .filter(|t| SHORT_TAGS.contains(&t.normalized.as_str()))
            .cloned()
            .collect();
        for loser in losers {
            if resolve_conflicts {
                tokens.retain(|t| t.normalized != loser.normalized);
            }
            changes.push(PromptChange {
                id: *next_change_id,
                original: loser.text.clone(),
                action: ChangeAction::SizeConflict,
                reason: format!(
                    "'{}' conflicts with dominant '{}' in exclusive stature taxonomy.",
                    loser.text, tall_tok.text
                ),
                suggested_alternative: String::new(),
                framing_alternative: String::new(),
                mode: if resolve_conflicts { 0 } else { 2 },
            });
            *next_change_id += 1;
        }
    }

    // 2. Hip & Thigh scale: Curvy/Thick vs Thin/Narrow
    if !bypass_proportions {
        const CURVY_LOWER: &[&str] = &["wide hips", "thick thighs", "plump thighs"];
        const THIN_LOWER: &[&str] = &["narrow hips", "thin legs", "skinny legs", "stick legs"];
        let has_curvy_lower = tokens.iter().find(|t| CURVY_LOWER.contains(&t.normalized.as_str())).cloned();
        if let Some(curvy_tok) = has_curvy_lower {
            let losers: Vec<PromptToken> = tokens
                .iter()
                .filter(|t| THIN_LOWER.contains(&t.normalized.as_str()))
                .cloned()
                .collect();
            for loser in losers {
                if resolve_conflicts {
                    tokens.retain(|t| t.normalized != loser.normalized);
                }
                changes.push(PromptChange {
                    id: *next_change_id,
                    original: loser.text.clone(),
                    action: ChangeAction::SizeConflict,
                    reason: format!(
                        "'{}' conflicts with dominant '{}' in exclusive hip_thigh_scale taxonomy.",
                        loser.text, curvy_tok.text
                    ),
                    suggested_alternative: String::new(),
                    framing_alternative: String::new(),
                    mode: if resolve_conflicts { 0 } else { 2 },
                });
                *next_change_id += 1;
            }
        }
    }

    // 3. Body Frame: Curvy/Full vs Slender/Skinny
    if !bypass_proportions {
        const CURVY_FRAME: &[&str] = &["shortstack", "plump", "chubby", "curvy", "voluptuous"];
        const SLENDER_FRAME: &[&str] = &["slender", "skinny"];
        let has_curvy_frame = tokens.iter().find(|t| CURVY_FRAME.contains(&t.normalized.as_str())).cloned();
        if let Some(curvy_tok) = has_curvy_frame {
            let losers: Vec<PromptToken> = tokens
                .iter()
                .filter(|t| SLENDER_FRAME.contains(&t.normalized.as_str()))
                .cloned()
                .collect();
            for loser in losers {
                if resolve_conflicts {
                    tokens.retain(|t| t.normalized != loser.normalized);
                }
                changes.push(PromptChange {
                    id: *next_change_id,
                    original: loser.text.clone(),
                    action: ChangeAction::SizeConflict,
                    reason: format!(
                        "'{}' conflicts with dominant '{}' in exclusive body_frame taxonomy.",
                        loser.text, curvy_tok.text
                    ),
                    suggested_alternative: String::new(),
                    framing_alternative: String::new(),
                    mode: if resolve_conflicts { 0 } else { 2 },
                });
                *next_change_id += 1;
            }
        }
    }
}

pub struct ComboSynergyRule {
    pub name: &'static str,
    pub anatomy_any: &'static [&'static str],
    pub garment_any: &'static [&'static str],
    pub pose_any: &'static [&'static str],
    pub recommend_tags: &'static [&'static str],
    pub reason: &'static str,
}

pub const COMBO_SYNERGY_RULES: &[ComboSynergyRule] = &[
    ComboSynergyRule {
        name: "Bust & Button-Up Tension",
        anatomy_any: &["large breasts", "huge breasts", "gigantic breasts", "hyper breasts", "enormous breasts"],
        garment_any: &["shirt", "blouse", "collared shirt", "dress shirt", "cardigan", "vest", "button-up"],
        pose_any: &[],
        recommend_tags: &["button gap", "popped button", "taut shirt"],
        reason: "Full bust volume with a buttoned shirt creates natural fabric tension. Recommend adding button gap, popped button, or taut shirt.",
    },
    ComboSynergyRule {
        name: "Bust & Knit Sweater Tension",
        anatomy_any: &["large breasts", "huge breasts", "gigantic breasts", "enormous breasts"],
        garment_any: &["sweater", "turtleneck", "knit sweater"],
        pose_any: &[],
        recommend_tags: &["taut clothes", "framed breasts"],
        reason: "Full bust volume with a knit sweater forms defined contours. Recommend taut clothes or framed breasts.",
    },
    ComboSynergyRule {
        name: "Curvy Thighs & Legwear Indentation",
        anatomy_any: &["thick thighs", "wide hips", "plump thighs", "curvy"],
        garment_any: &["thighhighs", "kneehighs", "pantyhose", "stockings"],
        pose_any: &[],
        recommend_tags: &["skindentation"],
        reason: "Curvy thighs paired with snug legwear naturally produce skindentation and thigh squish.",
    },
    ComboSynergyRule {
        name: "Hourglass & High-Slit Dress",
        anatomy_any: &["wide hips", "broad hips", "pear shaped", "huge hips", "narrow waist wide hips", "hourglass figure"],
        garment_any: &["china dress", "qipao", "cheongsam", "slit dress", "slit skirt"],
        pose_any: &[],
        recommend_tags: &["side slit", "hip vent"],
        reason: "Hourglass proportions with a slit dress or qipao benefit from side slits and tailored contours.",
    },
    ComboSynergyRule {
        name: "Pronounced Hips & Posture Synergy",
        anatomy_any: &["wide hips", "pronounced hips", "broad hips", "narrow waist wide hips"],
        garment_any: &[],
        pose_any: &["standing", "front view", "full body", "cowboy shot"],
        recommend_tags: &["contrapposto", "hand on own hip", "hands on own hips"],
        reason: "Accentuate hip proportions with classic weight-shifted posture. Recommend contrapposto or hand on own hip.",
    },
    ComboSynergyRule {
        name: "Petite Stature & Weapon Framing",
        anatomy_any: &["short", "petite", "loli"],
        garment_any: &["staff", "polearm", "huge weapon", "giant weapon", "scythe", "greatsword", "sword"],
        pose_any: &[],
        recommend_tags: &["upper body", "cowboy shot"],
        reason: "Petite stature with large weapons can be dwarfed in wide shots. Recommend cowboy shot or upper body to keep head in frame.",
    },
];

pub fn generate_combo_synergy_cards(
    tokens: &[PromptToken],
    changes: &mut Vec<PromptChange>,
    next_change_id: &mut usize,
) {
    let norms: Vec<&str> = tokens.iter().map(|t| t.normalized.as_str()).collect();
    let tag_matches = |norm: &str, target: &str| -> bool {
        norm == target || norm.split_whitespace().any(|w| w == target)
    };
    let has_any = |list: &[&str]| -> bool {
        list.is_empty() || list.iter().any(|target| norms.iter().any(|n| tag_matches(n, target)))
    };
    // body tags must be the whole tag: whole-word matching let "short" fire on "short hair" / "short sleeves"
    let has_body = |list: &[&str]| -> bool { list.is_empty() || list.iter().any(|target| norms.iter().any(|n| n == target)) };

    for rule in COMBO_SYNERGY_RULES {
        let match_anatomy = has_body(rule.anatomy_any);
        let match_garment = has_any(rule.garment_any);
        let match_pose = has_any(rule.pose_any);

        if match_anatomy && match_garment && match_pose {
            let missing: Vec<&str> = rule
                .recommend_tags
                .iter()
                .filter(|rec| !norms.iter().any(|n| tag_matches(n, rec)))
                .copied()
                .collect();

            if !missing.is_empty() {
                let suggested_alt = missing.join(", ");
                if !changes.iter().any(|c| c.action == ChangeAction::Recommendation && c.suggested_alternative == suggested_alt) {
                    changes.push(PromptChange {
                        id: *next_change_id,
                        original: String::new(),
                        action: ChangeAction::Recommendation,
                        reason: format!("Combo Synergy ({}): {}", rule.name, rule.reason),
                        suggested_alternative: suggested_alt,
                        framing_alternative: String::new(),
                        mode: 2,
                    });
                    *next_change_id += 1;
                }
            }
        }
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
            fit_suggestions: true,
            exaggerated_proportions: false,

            camera_angle: 0,

            pose: 0,
            bust: 0,
            frame: 0,

            dynamic: false,

            lora_tags_raw: String::new(), arranged: false, natural: false, two_char: false, char2_prompt_raw: String::new(), char2_lora_tags_raw: String::new(), char_break: false, scene_prompt_raw: String::new(), more_chars: Vec::new(),

            detail: crate::profiles::DetailLevel::Balanced,
        }
    }

    fn run(prompt: &str, config: &ScanConfig) -> (RuleAnalysisResult, String) {
        let toks = parse_prompt(prompt);
        let res = analyze_with_rules(&toks, config);
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, config);
        (res, out.final_prompt)
    }

    #[test]
    fn exaggerated_picks_are_emphasized_per_checkpoint() {
        use crate::profiles::CheckpointProfile as C;
        let out_for = |cp: C, exaggerated: bool, prompt: &str| {
            let mut c = cfg(TargetView::None);
            c.checkpoint = cp;
            c.bust = 6; // gigantic breasts
            c.frame = 2; // shortstack
            c.exaggerated_proportions = exaggerated;
            run(prompt, &c).1
        };
        let anima = out_for(C::Anima, true, "1girl, smile");
        assert!(anima.contains("(gigantic breasts:2.5)") && anima.contains("(shortstack:1.4)") && anima.contains("(wide hips:1.5)"), "{anima}");
        // the waist cinch gets a light 1.2 (the user's plan); curvy and petite stay plain
        assert!(anima.contains("(narrow waist:1.2)") && !anima.contains("(curvy:") && !anima.contains("(petite:"), "{anima}");
        let pony = out_for(C::PonyV6, true, "1girl, smile");
        assert!(pony.contains("(gigantic breasts:1.4)") && pony.contains("(shortstack:1.2)"), "{pony}");
        // off: plain tags
        let plain = out_for(C::Anima, false, "1girl, smile");
        assert!(plain.contains("gigantic breasts") && !plain.contains("(gigantic breasts:"), "{plain}");
        // a typed weight that's already higher stays (and the clamp doesn't cut the emphasis)
        let typed = out_for(C::Anima, true, "1girl, (gigantic breasts:3)");
        assert!(typed.contains("(gigantic breasts:2.5)") || typed.contains("(gigantic breasts:3)"), "{typed}");
        // the AI views get the same emphasis
        let ai = apply_body_choices_with("1girl, large breasts, slender", 6, 2, Some(C::Anima));
        assert!(ai.contains("(gigantic breasts:2.5)") && !ai.contains("large breasts"), "{ai}");
        assert_eq!(body_emphasis(C::Flux, "gigantic breasts", true, 2), 1.0);
    }

    #[test]
    fn exaggerated_frame_reinforcement_follows_the_plan() {
        use crate::profiles::CheckpointProfile as C;
        let both = |cp: C, frame: usize, bust: usize, exaggerated: bool| {
            let mut c = cfg(TargetView::None);
            c.checkpoint = cp;
            c.frame = frame;
            c.bust = bust;
            c.exaggerated_proportions = exaggerated;
            let out = run("1girl, smile", &c).1;
            let neg = crate::condenser::recommended_negative(&c, &out);
            (out, neg)
        };
        // Pear (4) on Anima: narrow shoulders + waist cinch, shoulder / top-heavy guards, the 17-term error guard
        let (out, neg) = both(C::Anima, 4, 0, true);
        assert!(out.contains("narrow shoulders") && out.contains("(narrow waist:1.2)"), "{out}");
        for t in ["broad shoulders", "wide shoulders", "V-taper", "top-heavy", "anatomical nonsense", "fused breasts"] {
            assert!(neg.contains(t), "{t}: {neg}");
        }
        // Pear on Pony: no narrow shoulders (CLIP doesn't read the words), guards there, only the short error guard
        let (out, neg) = both(C::PonyV6, 4, 0, true);
        assert!(!out.contains("narrow shoulders") && neg.contains("broad shoulders") && neg.contains("malformed hands"), "{out} | {neg}");
        assert!(!neg.contains("anatomical nonsense"), "{neg}");
        // Hourglass (5) on Anima: a stronger waist; Muscular (6): abs and its negatives
        assert!(both(C::Anima, 5, 0, true).0.contains("(narrow waist:1.3)"));
        let (out, neg) = both(C::Anima, 6, 0, true);
        assert!(out.contains("(abs:1.2)") && neg.contains("noodle arms") && neg.contains("soft body"), "{out} | {neg}");
        // Flat chest (bust 1) + Petite (1): the flat guard, and no "mature female"
        let (_, neg) = both(C::Anima, 1, 1, true);
        assert!(neg.contains("breast press") && neg.contains("cleavage") && neg.contains("giantess"), "{neg}");
        assert!(!neg.contains("mature female"), "{neg}");
        // off: none of it
        for frame in 0..=6 {
            let (out, neg) = both(C::Anima, frame, 6, false);
            assert!(!out.contains("narrow shoulders") && !out.contains("(narrow waist:") && !out.contains("(abs:"), "{frame}: {out}");
            assert!(!neg.contains("broad shoulders") && !neg.contains("extra limbs") && !neg.contains("V-taper"), "{frame}: {neg}");
        }
    }

    #[test]
    fn thigh_gap_goes_with_thick_thighs_only_when_the_legs_are_apart() {
        let config = cfg(TargetView::None);
        let gap_cards = |p: &str| run(p, &config).0.changes.iter().filter(|c| c.original == "thigh gap").count();
        assert_eq!(gap_cards("1girl, thick thighs, thigh gap"), 1);
        assert_eq!(gap_cards("1girl, thick thighs, thigh gap, spread legs"), 0);
        // the hips lock leaves "thigh gap" out of the negative with the legs apart, and says so
        let neg = |p: &str| crate::condenser::recommended_negative(&config, p);
        assert!(neg("1girl, thick thighs").contains("thigh gap"));
        assert!(!neg("1girl, thick thighs, spread legs").contains("thigh gap"));
        let (res, _) = run("1girl, thick thighs, wide stance", &config);
        assert!(res.changes.iter().any(|c| c.reason.starts_with("Legs apart")));
    }

    #[test]
    fn every_suggested_tag_is_a_real_danbooru_tag() {
        // what a FIT & DYNAMICS card can offer (core_reinforcers are never offered on a card)
        let Some(home) = std::env::var_os("HOME") else { return };
        let csv = std::path::Path::new(&home).join(".local/share/prompt-condenser/wd-eva02-large-tagger-v3/selected_tags.csv");
        let Ok(text) = std::fs::read_to_string(csv) else { return };
        let known: HashSet<String> = text.lines().skip(1).filter_map(|l| l.split(',').nth(1)).map(|n| n.replace('_', " ")).collect();
        let mut offered: Vec<&str> = COMBO_SYNERGY_RULES.iter().flat_map(|r| r.recommend_tags.iter().copied()).collect();
        for tier in [
            ProportionTier::FlatChest, ProportionTier::SmallBreasts, ProportionTier::MediumBreasts, ProportionTier::LargeBreasts,
            ProportionTier::HugeBreasts, ProportionTier::GiganticBreasts, ProportionTier::ThickThighsAndHips, ProportionTier::MuscularFemale,
        ] {
            let rec = get_proportion_recommendations(tier);
            offered.extend(rec.clothing_interaction.iter().chain(rec.pose_and_physics).copied());
        }
        let unknown: Vec<&str> = offered.into_iter().filter(|t| !known.contains(&t.replace('_', " "))).collect();
        assert!(unknown.is_empty(), "suggested tags that aren't Danbooru tags: {unknown:?}");
    }

    #[test]
    fn posture_leg_and_boot_conflicts_follow_the_rules() {
        let config = cfg(TargetView::None);
        let cards = |p: &str| {
            let (res, out) = run(p, &config);
            let c: Vec<(String, ChangeAction)> = res.changes.iter().map(|c| (c.original.clone(), c.action)).collect();
            (c, out)
        };
        // seiza is a kind of sitting: the generic tag goes, never the specific one
        let (c, out) = cards("1girl, sitting, seiza");
        assert!(c.contains(&("sitting".into(), ChangeAction::ImpliedTag)) && out.contains("seiza"), "{c:?} {out}");
        // seiza folds the legs under; standing on one leg lifts one
        let (c, _) = cards("1girl, seiza, crossed legs");
        assert!(c.contains(&("crossed legs".into(), ChangeAction::PoseConflict)), "{c:?}");
        let (c, _) = cards("1girl, standing on one leg, spread legs");
        assert!(c.contains(&("spread legs".into(), ChangeAction::PoseConflict)), "{c:?}");
        // normal combinations stay
        for p in ["1girl, sitting, crossed legs", "1girl, standing, leaning forward", "1girl, sitting, leaning forward", "1girl, squatting, spread legs"] {
            let (c, _) = cards(p);
            assert!(!c.iter().any(|(_, a)| *a == ChangeAction::PoseConflict), "{p}: {c:?}");
        }
        // one boot height, the first written wins
        let (c, out) = cards("1girl, thigh boots, knee boots");
        assert!(c.contains(&("knee boots".into(), ChangeAction::PoseConflict)) && out.contains("thigh boots"), "{c:?}");
        // several people: one can stand while another sits, one waves while another crosses her arms
        for p in ["2girls, standing, sitting", "2girls, crossed arms, waving, v", "1girl, 1boy, standing, kneeling"] {
            let (c, _) = cards(p);
            assert!(!c.iter().any(|(_, a)| *a == ChangeAction::PoseConflict), "{p}: {c:?}");
        }
        // footwear that isn't worn doesn't cancel barefoot
        for p in ["1girl, barefoot, shoes removed", "1girl, barefoot, unworn shoes", "1girl, barefoot, holding shoes"] {
            let (c, _) = cards(p);
            assert!(!c.iter().any(|(o, _)| o == "barefoot"), "{p}: {c:?}");
        }
        let (c, _) = cards("1girl, barefoot, sneakers");
        assert!(c.iter().any(|(o, _)| o == "barefoot"), "{c:?}");
    }

    #[test]
    fn body_type_card_lists_only_what_applies() {
        let config = cfg(TargetView::None);
        let (res, _) = run("1girl, gigantic breasts", &config);
        let card = res.changes.iter().find(|c| c.reason.starts_with("Reinforce")).expect("body-type card");
        assert!(!card.reason.contains("Clothing") && !card.reason.contains("Fit:") && !card.reason.contains('_'), "{}", card.reason);
        // striped thighhighs count as legwear, so the fit tag is offered
        let (res, _) = run("1girl, wide hips, striped thighhighs", &config);
        assert!(res.changes.iter().any(|c| c.reason.starts_with("Reinforce") && c.suggested_alternative.contains("skindentation")));
        // the switch turns every FIT & DYNAMICS card off
        let mut off = config.clone();
        off.fit_suggestions = false;
        let (res, _) = run("1girl, gigantic breasts, shirt, wide hips, thighhighs", &off);
        assert!(!res.changes.iter().any(|c| c.action == ChangeAction::Recommendation));
    }

    #[test]
    fn generic_tags_implied_by_specific_ones_get_a_card() {
        let tags = |out: &str| out.split(", ").map(str::to_string).collect::<Vec<_>>();
        let (res, out) = run("1girl, bikini, swimsuit, holding sword, sword, witch hat, hat, beach", &cfg(TargetView::None));
        let implied: Vec<&str> =
            res.changes.iter().filter(|c| c.action == ChangeAction::ImpliedTag).map(|c| c.original.as_str()).collect();
        assert_eq!(implied, ["swimsuit", "sword", "hat"]);
        let t = tags(&out);
        for kept in ["bikini", "holding sword", "witch hat", "beach"] {
            assert!(t.contains(&kept.to_string()), "{kept} kept: {out}");
        }
        for gone in ["swimsuit", "sword", "hat"] {
            assert!(!t.contains(&gone.to_string()), "{gone} pruned: {out}");
        }
        // Keep Original brings the generic tag back
        let config = cfg(TargetView::None);
        let prompt = "1girl, red bikini, bikini";
        let mut res = analyze_with_rules(&parse_prompt(prompt), &config);
        let card = res.changes.iter_mut().find(|c| c.action == ChangeAction::ImpliedTag).unwrap();
        card.mode = 2;
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, &config).final_prompt;
        assert!(tags(&out).contains(&"bikini".to_string()), "{out}");
        // weighted generic tags, "holding" non-weapons, absent items and unrelated pairs stay
        for prompt in ["1girl, bikini, (swimsuit:1.2)", "1girl, holding hat, hat", "1girl, no hat, hat", "1girl, hair bow, bow", "1girl, sunglasses, glasses"] {
            let (res, _) = run(prompt, &cfg(TargetView::None));
            assert!(!res.changes.iter().any(|c| c.action == ChangeAction::ImpliedTag), "{prompt}");
        }
        // Clean Bloat off: nothing pruned
        let mut off = cfg(TargetView::None);
        off.cleanup_enabled = false;
        assert!(!run("1girl, bikini, swimsuit", &off).0.changes.iter().any(|c| c.action == ChangeAction::ImpliedTag));
    }

    #[test]
    fn implication_table_is_safe() {
        assert!(implies("one-piece swimsuit", "swimsuit"));
        assert!(implies("school swimsuit", "swimsuit"));
        assert!(implies("katana", "sword"));
        assert!(implies("holding weapon", "weapon"));
        assert!(implies("straw hat", "hat"));
        assert!(!implies("hat ribbon", "hat"));
        assert!(!implies("dress shirt", "dress"));
        assert!(!implies("unworn hat", "hat"));
        assert!(!implies("swimsuit", "swimsuit"));
        assert!(!implies("ponytail", "tail"));
    }

    #[test]
    fn quality_button_adds_only_missing_checkpoint_tags() {
        use crate::profiles::CheckpointProfile as C;
        assert_eq!(add_quality_tags("1girl, beach", C::Anima), ("masterpiece, best quality, 1girl, beach".to_string(), vec!["masterpiece", "best quality"]));
        assert_eq!(add_quality_tags("(score_9:1.1), 1girl", C::PonyV6).0, "score 8 up, score 7 up, (score_9:1.1), 1girl".replace("score 8 up, score 7 up", "score_8_up, score_7_up"));
        assert_eq!(add_quality_tags("masterpiece, best quality, 1girl", C::Sdxl).1.len(), 0);
        assert_eq!(add_quality_tags("", C::Sd15).0, "masterpiece, best quality, highres");
        assert_eq!(add_quality_tags("1girl", C::Flux), ("1girl".to_string(), vec![]));
        // never a rating tag
        assert!(!add_quality_tags("1girl", C::PonyV6).0.contains("rating"));
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
    fn front_view_swaps_shoulder_blades_for_framing_safe_words() {
        let (res, out) = run("1girl, front view, (shoulder blades:1.2)", &cfg(TargetView::FrontView));
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::FramingSafeSwap && c.original.contains("shoulder blades")));
        assert!(out.contains("defined collarbone"));
        assert!(out.contains("toned shoulders"));
        assert!(!out.contains("shoulder blades"));
    }

    #[test]
    fn front_view_shoulder_blades_framing_alternative() {
        let config = cfg(TargetView::FrontView);
        let prompt = "1girl, front view, shoulder blades";
        let toks = parse_prompt(prompt);
        let mut res = analyze_with_rules(&toks, &config);
        let swap = res.changes.iter_mut().find(|c| c.action == ChangeAction::FramingSafeSwap && c.original == "shoulder blades").unwrap();
        swap.mode = 1;
        let out = build_condensed_output(prompt, &res.processed_tokens, &res.changes, &config).final_prompt;
        assert!(out.contains("shoulder blades"));
        assert!(out.contains("from behind"));
        assert!(!out.contains("front view"));
        assert!(!out.contains("defined collarbone"));
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
    fn lora_tags_beat_conflicting_tags_and_stay_verbatim() {
        let mut config = cfg(TargetView::None);
        config.lora_tags_raw = "<lora:mewson:0.8>, mewson, large breasts, silver hair,".into();
        let (_, out) = run("masterpiece, 1girl, gigantic breasts, black hair, long hair, blue eyes, mewson", &config);
        let out = insert_lora_block(&out, &config, false);
        assert!(out.contains("<lora:mewson:0.8>, mewson, large breasts, silver hair"), "{out}");
        assert_eq!(out.matches("mewson,").count(), 1, "{out}");
        assert!(!out.contains("gigantic") && !out.contains("black hair"), "{out}");
        assert!(out.contains("long hair") && out.contains("blue eyes"), "{out}");
        assert!(tags_conflict("flat chest", "huge breasts"));
        assert!(!tags_conflict("large breasts", "breasts out"));
        assert!(!tags_conflict("long hair", "silver hair"));
        assert!(tags_conflict("amber eyes", "grey eyes"));
        assert!(tags_conflict("white hair", "long black hair"));
        assert!(tags_conflict("short hair", "long black hair"));
        assert!(tags_conflict("dark skin", "pale skin"));
    }

    #[test]
    fn finishing_respects_the_checkpoint() {
        let mut config = cfg(TargetView::None);
        config.natural = true;
        let s = Some("Two friends share an umbrella.");
        // Anima understands sentences
        assert_eq!(finish_prompt("1girl, rain", &config, false, s), "1girl, rain, Two friends share an umbrella.");
        // Pony doesn't: no sentence even with the toggle on
        config.checkpoint = CheckpointProfile::PonyV6;
        assert_eq!(finish_prompt("1girl, rain", &config, false, s), "1girl, rain");
        config.arranged = true;
        assert_eq!(finish_prompt("rain, 1girl", &config, false, s), "1girl,\nrain");
    }

    #[test]
    fn bust_and_frame_choices_keep_their_tags() {
        let tags = |out: &str| crate::parser::parse_prompt(out).into_iter().map(|t| t.normalized).collect::<Vec<_>>();
        let size_cards = |res: &RuleAnalysisResult| res.changes.iter().filter(|c| c.action == ChangeAction::SizeConflict).map(|c| c.original.clone()).collect::<Vec<_>>();
        // Shortstack: all six tags, no size-conflict card
        let mut config = cfg(TargetView::None);
        config.frame = 2;
        let (res, out) = run("1girl, smile", &config);
        for t in crate::profiles::FRAMES[2].1.split(", ") {
            assert!(tags(&out).contains(&t.to_string()), "{t} missing: {out}");
        }
        assert!(size_cards(&res).is_empty(), "{:?}", size_cards(&res));
        // Gigantic + Petite: both stay
        config.bust = 6;
        config.frame = 1;
        let (res, out) = run("1girl, smile", &config);
        assert!(tags(&out).contains(&"gigantic breasts".into()) && tags(&out).contains(&"petite".into()), "{out}");
        assert!(size_cards(&res).is_empty(), "{:?}", size_cards(&res));
        // Bust and Frame together replace only their own groups
        config.bust = 4;
        config.frame = 5;
        let (res, out) = run("1girl, small breasts, c cup, thick thighs, muscular female, smile", &config);
        let t = tags(&out);
        for want in ["large breasts", "curvy", "wide hips", "narrow waist", "smile"] {
            assert!(t.contains(&want.to_string()), "{want} missing: {out}");
        }
        for gone in ["small breasts", "c cup", "thick thighs", "muscular female"] {
            assert!(!t.contains(&gone.to_string()), "{gone} kept: {out}");
        }
        assert_eq!(res.changes.iter().filter(|c| c.action == ChangeAction::ProportionChoice).count(), 2);
        // a typed shortstack keeps its petite
        let (res, out) = run("1girl, shortstack, petite, huge breasts, wide hips", &cfg(TargetView::None));
        assert!(tags(&out).contains(&"petite".into()), "{out}");
        assert!(!size_cards(&res).contains(&"petite".to_string()));
        let ai = fix_body_type(crate::parser::parse_prompt("shortstack, petite, huge breasts"));
        assert!(ai.iter().any(|t| t.normalized == "petite"));
        // a typed "petite, huge breasts" still gets today's card
        let (res, out) = run("1girl, petite, huge breasts", &cfg(TargetView::None));
        assert!(!tags(&out).contains(&"petite".into()) && size_cards(&res).contains(&"petite".to_string()), "{out}");
        // the old single Body dropdown's saved index
        use crate::profiles::{bust_frame_from_old_body as old, BUSTS, FRAMES};
        assert_eq!(old(0), (0, 0));
        assert_eq!(BUSTS[old(6).0 as usize].1, "gigantic breasts");
        assert_eq!(FRAMES[old(7).1 as usize].0, "Thick thighs & wide hips");
        assert_eq!(FRAMES[old(8).1 as usize].1, "muscular female, abs, toned");
        assert_eq!(old(1), (1, 0));
    }

    #[test]
    fn petite_with_large_breasts_becomes_short() {
        let config = cfg(TargetView::None);
        let (res, out) = run("1girl, petite, (large breasts:1.25), wide hips, short hair", &config);
        assert!(!out.contains("petite") && out.contains("short,") || out.contains("short\n") || out.contains(", short"), "{out}");
        assert!(out.contains("(large breasts:1.25)"), "{out}");
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::SizeConflict && c.original == "petite"));
        // petite alone stays
        let (_, out) = run("1girl, petite, small breasts", &config);
        assert!(out.contains("petite"), "{out}");
    }

    #[test]
    fn conflicting_sizes_keep_the_first() {
        let config = cfg(TargetView::None);
        let (res, out) = run("1girl, medium breasts, long hair, large breasts, very long hair, short hair, blue hair, white hair", &config);
        assert!(out.contains("medium breasts") && !out.contains("large breasts"), "{out}");
        assert!(out.contains("long hair") && out.contains("very long hair") && !out.contains("short hair"), "{out}");
        assert!(out.contains("blue hair") && out.contains("white hair"), "{out}"); // two colors are fine
        assert_eq!(res.changes.iter().filter(|c| c.action == ChangeAction::SizeConflict).count(), 2);
        assert!(size_conflict("flat chest", "huge breasts"));
        assert!(!size_conflict("long hair", "very long hair"));
    }

    #[test]
    fn ai_prompt_gets_the_lora_block() {
        let mut config = cfg(TargetView::None);
        config.checkpoint = CheckpointProfile::PonyV6;
        config.lora_tags_raw = "large breasts, red scarf".into();
        let out = insert_lora_block("score_9, score_8_up, 1girl, gigantic breasts, smile, red scarf", &config, true);
        assert_eq!(out, "score_9, score_8_up, large breasts, red scarf, 1girl, smile");
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

    #[test]
    fn enriched_tags_are_not_reported_as_dropped() {
        let dropped = crate::condenser::dropped_tags(
            "1girl, kimono, brown hair, smile, cherry blossoms, red cape, long white hair",
            "1girl, red silk kimono with gold embroidery, chestnut brown hair, gentle smile, flowing red velvet cape, long flowing white hair",
            CheckpointProfile::Anima,
        );
        assert_eq!(dropped, vec!["cherry blossoms"]);
    }

    #[test]
    fn front_view_json_conflicts_are_detected() {
        let config = cfg(TargetView::FrontView);
        let prompt = "1girl, front view, backless dress, nape, spine, sideboob, butt crack, over the shoulder";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "backless dress" && c.reason.contains("Forces rear or side rotation")));
        assert!(res.changes.iter().any(|c| c.original == "nape" && c.reason.contains("Forces rear or side rotation")));
        assert!(res.changes.iter().any(|c| c.original == "spine" && c.reason.contains("Forces rear or side rotation")));
        assert!(res.changes.iter().any(|c| c.original == "sideboob" && c.reason.contains("Forces rear or side rotation")));
        assert!(res.changes.iter().any(|c| c.original == "butt crack" && c.reason.contains("Forces rear or side rotation")));
        assert!(res.changes.iter().any(|c| c.original == "over the shoulder" && c.action == ChangeAction::PositioningConflict));
        assert!(!out.contains("backless dress"));
        assert!(!out.contains("spine"));
    }

    #[test]
    fn back_view_json_conflicts_are_detected() {
        let config = cfg(TargetView::FromBehind);
        let prompt = "1girl, from behind, front view, navel, stomach, midriff, cleavage, collarbone";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "front view" && c.reason.contains("Forces torso to rotate forward")));
        assert!(res.changes.iter().any(|c| c.original == "navel" && c.reason.contains("Forces torso to rotate forward")));
        assert!(res.changes.iter().any(|c| c.original == "cleavage" && c.reason.contains("Forces torso to rotate forward")));
        assert!(res.changes.iter().any(|c| c.original == "collarbone" && c.reason.contains("Forces torso to rotate forward")));
        assert!(!out.contains("cleavage"));
        assert!(!out.contains("navel"));
    }

    #[test]
    fn eye_level_json_conflicts_are_detected() {
        let mut config = cfg(TargetView::None);
        config.camera_angle = 1; // Eye level / straight-on
        let prompt = "1girl, from below, upskirt, downblouse, soles";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "from below" && c.reason.contains("Hijacks camera pitch")));
        assert!(res.changes.iter().any(|c| c.original == "upskirt" && c.reason.contains("Hijacks camera pitch")));
        assert!(res.changes.iter().any(|c| c.original == "downblouse" && c.reason.contains("Hijacks camera pitch")));
        assert!(res.changes.iter().any(|c| c.original == "soles" && c.reason.contains("Hijacks camera pitch")));
        assert!(!out.contains("from below"));
        assert!(!out.contains("upskirt"));
    }

    #[test]
    fn portrait_json_conflicts_are_detected() {
        let config = cfg(TargetView::CloseUpPortrait);
        let prompt = "1girl, portrait, shoes, skirt, thighs, standing, navel";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "shoes" && c.reason.contains("Pulls camera down/back")));
        assert!(res.changes.iter().any(|c| c.original == "skirt" && c.reason.contains("Pulls camera down/back")));
        assert!(res.changes.iter().any(|c| c.original == "thighs" && c.reason.contains("Pulls camera down/back")));
        assert!(res.changes.iter().any(|c| c.original == "standing" && c.reason.contains("Pulls camera down/back")));
        assert!(!out.contains("shoes"));
        assert!(!out.contains("standing"));
    }

    #[test]
    fn upper_body_json_conflicts_are_detected() {
        let config = cfg(TargetView::UpperBody);
        let prompt = "1girl, upper body, feet, high heels, knees, calves, squatting";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "feet" && c.reason.contains("Forces framing below the waist/hips")));
        assert!(res.changes.iter().any(|c| c.original == "high heels" && c.reason.contains("Forces framing below the waist/hips")));
        assert!(res.changes.iter().any(|c| c.original == "knees" && c.reason.contains("Forces framing below the waist/hips")));
        assert!(res.changes.iter().any(|c| c.original == "squatting" && c.reason.contains("Forces framing below the waist/hips")));
        assert!(!out.contains("feet"));
        assert!(!out.contains("squatting"));
    }

    #[test]
    fn full_body_json_conflicts_are_detected() {
        let config = cfg(TargetView::FullBody);
        let prompt = "1girl, full body, portrait, detailed eyes, pupils, lip gloss, uvula, mole under eye";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "portrait" && c.action == ChangeAction::PositioningConflict));
        assert!(res.changes.iter().any(|c| c.original == "detailed eyes" && c.reason.contains("Forces tight facial crop")));
        assert!(res.changes.iter().any(|c| c.original == "pupils" && c.reason.contains("Forces tight facial crop")));
        assert!(res.changes.iter().any(|c| c.original == "lip gloss" && c.reason.contains("Forces tight facial crop")));
        assert!(res.changes.iter().any(|c| c.original == "uvula" && c.reason.contains("Forces tight facial crop")));
        assert!(!out.contains("lip gloss"));
        assert!(!out.contains("uvula"));
    }

    #[test]
    fn arms_down_pose_conflicts_are_detected() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, hands on hips, arms up, bare armpits, peace sign, waving";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "arms up" && c.action == ChangeAction::PoseConflict && c.reason.contains("force arms to lift")));
        assert!(res.changes.iter().any(|c| c.original == "bare armpits" && c.action == ChangeAction::PoseConflict));
        assert!(res.changes.iter().any(|c| c.original == "peace sign" && c.action == ChangeAction::PoseConflict));
        assert!(res.changes.iter().any(|c| c.original == "waving" && c.action == ChangeAction::PoseConflict));
        assert!(out.contains("hands on hips"));
        assert!(!out.contains("arms up"));
        assert!(!out.contains("peace sign"));
    }

    #[test]
    fn standing_upright_pose_conflicts_are_detected() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, standing, sitting, seiza, lying, bent over, crouching";
        let (res, out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.original == "sitting" && c.action == ChangeAction::PoseConflict && c.reason.contains("Forces grounded")));
        assert!(res.changes.iter().any(|c| c.original == "seiza" && c.action == ChangeAction::PoseConflict));
        assert!(res.changes.iter().any(|c| c.original == "lying" && c.action == ChangeAction::PoseConflict));
        // bending over is done standing (and leaning forward goes with any posture), so it stays
        assert!(!res.changes.iter().any(|c| c.original == "bent over"));
        assert!(res.changes.iter().any(|c| c.original == "crouching" && c.action == ChangeAction::PoseConflict));
        assert!(out.contains("standing"));
        assert!(!out.contains("sitting"));
        assert!(!out.contains("seiza"));
    }

    #[test]
    fn small_breasts_keeps_size_and_drops_cleavage_and_large_breasts() {
        let config = cfg(TargetView::None);
        // When small breasts is prompted, it preserves the breast size and strips cleavage, large breasts, etc.
        let (res, out) = run("1girl, small breasts, cleavage, large breasts, busty, underboob", &config);
        assert!(out.contains("small breasts"));
        assert!(!out.contains("cleavage"));
        assert!(!out.contains("large breasts"));
        assert!(!out.contains("busty"));
        assert!(!out.contains("underboob"));
        assert!(res.changes.iter().any(|c| c.original == "cleavage" && c.action == ChangeAction::SizeConflict));
        assert!(res.changes.iter().any(|c| c.original == "large breasts" && c.action == ChangeAction::SizeConflict));
        assert!(res.changes.iter().any(|c| c.original == "busty" && c.action == ChangeAction::SizeConflict));
        assert!(res.changes.iter().any(|c| c.original == "underboob" && c.action == ChangeAction::SizeConflict));
    }

    #[test]
    fn small_breasts_drops_prominent_breast_poses() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, small breasts, breast rest, holding breasts, breast grab";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("small breasts"));
        assert!(!out.contains("breast rest"));
        assert!(!out.contains("holding breasts"));
        assert!(!out.contains("breast grab"));
        assert!(res.changes.iter().any(|c| c.original == "breast rest" && c.action == ChangeAction::PoseConflict && c.reason.contains("prominent breasts")));
        assert!(res.changes.iter().any(|c| c.original == "holding breasts" && c.action == ChangeAction::PoseConflict));
        assert!(res.changes.iter().any(|c| c.original == "breast grab" && c.action == ChangeAction::PoseConflict));
    }

    #[test]
    fn lora_overrides_small_breasts_when_specified() {
        let mut config = cfg(TargetView::None);
        config.lora_tags_raw = "<lora:hero:1>, large breasts".into();
        let prompt = "1girl, small breasts, smile";
        let (res, out) = run(prompt, &config);
        let finished = insert_lora_block(&out, &config, false);
        assert!(finished.contains("large breasts"));
        assert!(!finished.contains("small breasts"));
        assert!(res.changes.iter().any(|c| c.original == "small breasts" && c.action == ChangeAction::KeptEmphasized));
    }

    #[test]
    fn lora_small_breasts_overrides_prompt_cleavage_and_busty() {
        let mut config = cfg(TargetView::None);
        config.lora_tags_raw = "<lora:hero:1>, small breasts".into();
        let prompt = "1girl, cleavage, busty, smile";
        let (res, out) = run(prompt, &config);
        let finished = insert_lora_block(&out, &config, false);
        assert!(finished.contains("small breasts"));
        assert!(!finished.contains("cleavage"));
        assert!(!finished.contains("busty"));
        assert!(res.changes.iter().any(|c| c.original.contains("cleavage") && c.action == ChangeAction::KeptEmphasized));
    }

    #[test]
    fn huge_breasts_lying_on_stomach_causes_pose_conflict() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, gigantic breasts, lying on stomach";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("gigantic breasts"));
        assert!(!out.contains("lying on stomach"));
        assert!(res.changes.iter().any(|c| c.original == "lying on stomach" && c.action == ChangeAction::PoseConflict && c.reason.contains("clipping")));
    }

    #[test]
    fn small_breasts_framing_swap_preserves_slender_physique() {
        let config = cfg(TargetView::UpperBody);
        let prompt = "1girl, small breasts, upper body, ass";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("small breasts"));
        assert!(out.contains("slender"));
        assert!(out.contains("narrow waist"));
        assert!(!out.contains("voluptuous"));
        assert!(!out.contains("hourglass figure"));
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::FramingSafeSwap));
    }

    #[test]
    fn kemonomimi_strips_furry_bleed_and_four_ears_triggers() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, cat ears, furry, muzzle, paws, human ears, earrings, pointy ears";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("cat ears"));
        assert!(!out.contains("furry"));
        assert!(!out.contains("muzzle"));
        assert!(!out.contains("paws"));
        assert!(!out.contains("human ears"));
        assert!(!out.contains("earrings"));
        assert!(!out.contains("pointy ears"));
        assert!(res.changes.iter().any(|c| c.original == "furry" && c.action == ChangeAction::AnatomyConflict && c.reason.contains("Prevents kemonomimi")));
        assert!(res.changes.iter().any(|c| c.original == "human ears" && c.action == ChangeAction::AnatomyConflict));
    }

    #[test]
    fn closed_eyes_strips_gaze_pupils_and_eye_colors() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, closed eyes, looking at viewer, blue eyes, slit pupils, eye contact";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("closed eyes"));
        assert!(!out.contains("looking at viewer"));
        assert!(!out.contains("blue eyes"));
        assert!(!out.contains("slit pupils"));
        assert!(!out.contains("eye contact"));
        assert!(res.changes.iter().any(|c| c.original == "looking at viewer" && c.action == ChangeAction::AnatomyConflict && c.reason.contains("force the eyelids open")));
        assert!(res.changes.iter().any(|c| c.original == "blue eyes" && c.action == ChangeAction::AnatomyConflict));
    }

    #[test]
    fn closed_mouth_strips_open_mouth_teeth_and_fangs() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, closed mouth, open mouth, teeth, fangs, tongue out";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("closed mouth"));
        assert!(!out.contains("open mouth"));
        assert!(!out.contains("teeth"));
        assert!(!out.contains("fangs"));
        assert!(!out.contains("tongue out"));
        assert!(res.changes.iter().any(|c| c.original == "open mouth" && c.action == ChangeAction::AnatomyConflict && c.reason.contains("Internal mouth anatomy")));
    }

    #[test]
    fn single_eyewear_rule_drops_second_pair() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, glasses, glasses on head, holding eyewear";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("glasses"));
        assert!(!out.contains("glasses on head"));
        assert!(!out.contains("holding eyewear"));
        assert!(res.changes.iter().any(|c| c.original == "glasses on head" && c.action == ChangeAction::AnatomyConflict && c.reason.contains("two pairs of glasses")));
        assert!(res.changes.iter().any(|c| c.original == "holding eyewear" && c.action == ChangeAction::AnatomyConflict));
    }

    #[test]
    fn short_petite_strips_tall_and_giant_weapon_framing_hijackers() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, short, petite, tall, long legs, scythe, giant weapon";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("short"));
        assert!(!out.contains("tall"));
        assert!(!out.contains("long legs"));
        assert!(!out.contains("scythe"));
        assert!(!out.contains("giant weapon"));
        assert!(res.changes.iter().any(|c| c.original == "tall" && c.action == ChangeAction::SizeConflict));
        assert!(res.changes.iter().any(|c| c.original == "scythe" && c.action == ChangeAction::SizeConflict && c.reason.contains("oversized props")));
    }

    #[test]
    fn tall_statuesque_strips_childlike_and_oversized_clothes() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, tall, mature female, round face, oversized shirt, chibi";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("tall"));
        assert!(!out.contains("round face"));
        assert!(!out.contains("oversized shirt"));
        assert!(!out.contains("chibi"));
        assert!(res.changes.iter().any(|c| c.original == "round face" && c.action == ChangeAction::SizeConflict && c.reason.contains("childlike stature")));
    }

    #[test]
    fn anime_flat_protection_strips_3d_bleed_tags() {
        let mut config = cfg(TargetView::None);
        config.cleanup_enabled = true;
        config.checkpoint = CheckpointProfile::PonyV6;
        let prompt = "1girl, photorealistic, 3d, blender, octane render, shiny skin, pores";
        let (res, out) = run(prompt, &config);
        assert!(!out.contains("photorealistic"));
        assert!(!out.contains("3d"));
        assert!(!out.contains("blender"));
        assert!(!out.contains("octane render"));
        assert!(!out.contains("shiny skin"));
        assert!(!out.contains("pores"));
        assert!(res.changes.iter().any(|c| c.original == "photorealistic" && c.action == ChangeAction::StyleConflict && c.reason.contains("2D anime checkpoints")));
    }

    #[test]
    fn dark_skin_protection_strips_pale_skin_and_cautions_tanlines() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, dark skin, pale skin, white skin, tanlines";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("dark skin"));
        assert!(!out.contains("pale skin"));
        assert!(!out.contains("white skin"));
        assert!(out.contains("tanlines"));
        assert!(res.changes.iter().any(|c| c.original == "pale skin" && c.action == ChangeAction::SizeConflict && c.reason.contains("washing out dark skin tones")));
        assert!(res.changes.iter().any(|c| c.original == "tanlines" && c.action == ChangeAction::SizeConflict && c.reason.contains("Caution")));
    }

    #[test]
    fn flat_chest_strips_larger_tiers_cleavage_and_physics_hijackers() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, flat chest, large breasts, cleavage, deep cleavage, boob shelf, breast press, virgin killer sweater";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("flat chest"));
        assert!(!out.contains("large breasts"));
        assert!(!out.contains("cleavage"));
        assert!(!out.contains("deep cleavage"));
        assert!(!out.contains("boob shelf"));
        assert!(!out.contains("breast press"));
        assert!(!out.contains("virgin killer sweater"));
        assert!(res.changes.iter().any(|c| c.original == "cleavage" && c.action == ChangeAction::SizeConflict && c.reason.contains("Removes all larger breast tiers")));
        assert!(res.changes.iter().any(|c| c.original == "breast press" && c.action == ChangeAction::PoseConflict));
    }

    #[test]
    fn medium_breasts_strips_deflation_and_inflation_hijackers() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, medium breasts, flat chest, huge breasts, breasts on table, visible ribs";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("medium breasts"));
        assert!(!out.contains("flat chest"));
        assert!(!out.contains("huge breasts"));
        assert!(!out.contains("breasts on table"));
        assert!(!out.contains("visible ribs"));
        assert!(res.changes.iter().any(|c| c.original == "flat chest" && c.action == ChangeAction::SizeConflict && c.reason.contains("Prevents extreme deflation")));
        assert!(res.changes.iter().any(|c| c.original == "breasts on table" && c.action == ChangeAction::PoseConflict));
    }

    #[test]
    fn large_breasts_strips_smaller_larger_tiers_and_hyper() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, large breasts, flat chest, small breasts, huge breasts, visible ribs, hyper";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("large breasts"));
        assert!(!out.contains("flat chest"));
        assert!(!out.contains("small breasts"));
        assert!(!out.contains("huge breasts"));
        assert!(!out.contains("visible ribs"));
        assert!(!out.contains("hyper"));
        assert!(res.changes.iter().any(|c| c.original == "flat chest" && c.action == ChangeAction::SizeConflict && c.reason.contains("Strips smaller/larger tiers")));
    }

    #[test]
    fn huge_breasts_strips_slender_tags() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, huge breasts, small breasts, flat chest, visible ribs, slender, petite";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("huge breasts"));
        assert!(!out.contains("small breasts"));
        assert!(!out.contains("flat chest"));
        assert!(!out.contains("visible ribs"));
        assert!(!out.contains("slender"));
        assert!(res.changes.iter().any(|c| c.original == "slender" && c.action == ChangeAction::SizeConflict && c.reason.contains("slender body tags that fight against huge breast")));
    }

    #[test]
    fn thick_curvy_lower_strips_thin_legs() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, thick thighs, wide hips, thin legs, stick legs, flat ass";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("thick thighs"));
        assert!(out.contains("wide hips"));
        assert!(!out.contains("thin legs"));
        assert!(!out.contains("stick legs"));
        assert!(!out.contains("flat ass"));
        assert!(res.changes.iter().any(|c| c.original == "thin legs" && c.action == ChangeAction::SizeConflict && c.reason.contains("neutralizing hip and thigh volume")));
    }

    #[test]
    fn slender_petite_lower_strips_thick_thighs_and_skindentation() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, slender, thin legs, thigh gap, thick thighs, skindentation, muffin top";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("slender"));
        assert!(out.contains("thin legs"));
        assert!(!out.contains("thick thighs"));
        assert!(!out.contains("skindentation"));
        assert!(!out.contains("muffin top"));
        assert!(res.changes.iter().any(|c| c.original == "thick thighs" && c.action == ChangeAction::SizeConflict && c.reason.contains("force thick thighs")));
    }

    #[test]
    fn muscular_toned_strips_chubby_and_flabby() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, abs, toned, muscular, chubby, soft body, love handles";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("abs"));
        assert!(out.contains("toned"));
        assert!(!out.contains("chubby"));
        assert!(!out.contains("soft body"));
        assert!(!out.contains("love handles"));
        assert!(res.changes.iter().any(|c| c.original == "chubby" && c.action == ChangeAction::SizeConflict && c.reason.contains("smoothing out muscle definition")));
    }

    #[test]
    fn soft_chubby_strips_abs_and_slim_waist() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, chubby, belly, abs, six pack, slim waist";
        let (res, out) = run(prompt, &config);
        assert!(out.contains("chubby"));
        assert!(out.contains("belly"));
        assert!(!out.contains("abs"));
        assert!(!out.contains("six pack"));
        assert!(!out.contains("slim waist"));
        assert!(res.changes.iter().any(|c| c.original == "abs" && c.action == ChangeAction::SizeConflict && c.reason.contains("erasing soft midsection anatomy")));
    }

    #[test]
    fn lora_overrides_kemonomimi_closed_eyes_and_dark_skin() {
        let mut config = cfg(TargetView::None);
        config.lora_tags_raw = "<lora:neko:1>, cat ears, closed eyes, dark skin, abs".into();
        let prompt = "1girl, human ears, furry, blue eyes, pale skin, chubby";
        let (res, out) = run(prompt, &config);
        let finished = insert_lora_block(&out, &config, false);
        assert!(finished.contains("cat ears"));
        assert!(finished.contains("closed eyes"));
        assert!(finished.contains("dark skin"));
        assert!(finished.contains("abs"));
        assert!(!finished.contains("human ears"));
        assert!(!finished.contains("furry"));
        assert!(!finished.contains("blue eyes"));
        assert!(!finished.contains("pale skin"));
        assert!(!finished.contains("chubby"));
        assert!(res.changes.iter().any(|c| c.original.contains("human ears") && c.action == ChangeAction::KeptEmphasized));
        assert!(res.changes.iter().any(|c| c.original.contains("blue eyes") && c.action == ChangeAction::KeptEmphasized));
        assert!(res.changes.iter().any(|c| c.original.contains("pale skin") && c.action == ChangeAction::KeptEmphasized));
        assert!(res.changes.iter().any(|c| c.original.contains("chubby") && c.action == ChangeAction::KeptEmphasized));
    }

    #[test]
    fn apply_proportion_lock_locks_tier_and_returns_recommendations() {
        let current_tags = vec![
            "1girl".to_string(),
            "flat chest".to_string(),
            "large breasts".to_string(),
            "cleavage".to_string(),
            "smile".to_string(),
        ];
        let lock_res = apply_proportion_lock(&current_tags, ProportionTier::FlatChest, true, 1.2);
        assert!(lock_res.final_positive_tags.iter().any(|t| t == "(flat chest:1.20)"));
        assert!(lock_res.final_positive_tags.contains(&"aa cup".to_string()));
        assert!(lock_res.final_positive_tags.contains(&"small underbust".to_string()));
        assert!(lock_res.final_positive_tags.contains(&"slender torso".to_string()));
        assert!(!lock_res.final_positive_tags.contains(&"large breasts".to_string()));
        assert!(!lock_res.final_positive_tags.contains(&"cleavage".to_string()));
        assert!(lock_res.removed_conflicts.contains(&"large breasts".to_string()));
        assert!(lock_res.removed_conflicts.contains(&"cleavage".to_string()));
        assert!(lock_res.recommended_clothing.contains(&"loose_shirt".to_string()));
        assert!(lock_res.recommended_physics_poses.contains(&"collarbone".to_string()));
        assert!(lock_res.auto_negative_additions.contains(&"cleavage".to_string()));
    }

    #[test]
    fn recommended_negative_auto_locks_detected_proportion_tiers() {
        use crate::condenser::recommended_negative;
        let config = cfg(TargetView::None);
        let neg = recommended_negative(&config, "1girl, flat chest, thick thighs, smile");
        assert!(neg.contains("breasts"));
        assert!(neg.contains("cleavage"));
        assert!(neg.contains("large breasts"));
        assert!(neg.contains("thin legs"));
        assert!(neg.contains("stick legs"));
        assert!(neg.contains("thigh gap"));
    }

    #[test]
    fn proportion_chain_recommendations_generate_cards_and_can_be_applied() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, flat chest, smile";
        let (res, _out) = run(prompt, &config);
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::Recommendation && c.reason.contains("flat chest")));
        let rec_card = res.changes.iter().find(|c| c.action == ChangeAction::Recommendation).unwrap();
        assert_eq!(rec_card.mode, 2);

        // When user applies the recommendation card (mode 0)
        let mut applied_card = rec_card.clone();
        applied_card.mode = 0;
        let condensed = crate::condenser::build_condensed_output(
            prompt,
            &res.processed_tokens,
            &[applied_card],
            &config,
        );
        assert!(
            condensed.final_prompt.contains("loose shirt")
                || condensed.final_prompt.contains("sarashi")
                || condensed.final_prompt.contains("baggy clothes")
                || condensed.final_prompt.contains("collarbone")
        );
    }

    #[test]
    fn front_view_angle_lock_and_prop_preservation_cards() {
        let config = cfg(TargetView::None);
        let prompt = "1girl, front view, sitting on bed, on bed, arms up, pulling up stocking";
        let (res, out) = run(prompt, &config);

        // 1. Redundant 'on bed' removed
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::RemovedBloat && c.original == "on bed"));
        assert!(!out.contains(", on bed"));

        // 2. Front-view angle lock card generated
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::Recommendation && c.reason.contains("Front-View Angle Lock")));

        // 3. Directive 6 prop preservation card generated
        assert!(res.changes.iter().any(|c| c.action == ChangeAction::Recommendation && c.reason.contains("Directive 6 Prop Preservation")));
    }

    #[test]
    fn exclusive_taxonomy_groups_resolve_by_dominance_rank() {
        let config = cfg(TargetView::None);

        // 1. Camera framing: full body beats portrait
        let (res, out) = run("1girl, full body, portrait", &config);
        assert!(out.contains("full body"));
        assert!(!out.contains("portrait"));
        assert!(res.changes.iter().any(|c| c.original == "portrait" && c.action == ChangeAction::PositioningConflict));

        // 2. Stature: tall female beats short female
        let (res, out) = run("1girl, tall female, short female", &config);
        assert!(out.contains("tall female"));
        assert!(!out.contains("short female"));
        assert!(res.changes.iter().any(|c| c.original == "short female" && c.action == ChangeAction::SizeConflict && c.reason.contains("exclusive stature taxonomy")));

        // 3. Hip thigh scale: thick thighs and wide hips together strip thin legs
        let (res, out) = run("1girl, thick thighs, wide hips, thin legs", &config);
        assert!(out.contains("thick thighs"));
        assert!(out.contains("wide hips"));
        assert!(!out.contains("thin legs"));
        assert!(res.changes.iter().any(|c| c.original == "thin legs" && c.action == ChangeAction::SizeConflict));
    }

    #[test]
    fn combo_synergy_cards_generation_and_application() {
        let config = cfg(TargetView::None);

        // 1. Combo 0: Bust volume + buttoned shirt
        let (res, _) = run("1girl, large breasts, shirt", &config);
        let card = res.changes.iter().find(|c| c.action == ChangeAction::Recommendation && c.reason.contains("Bust & Button-Up Tension")).expect("Combo 0 card");
        assert_eq!(card.mode, 2);
        assert!(card.suggested_alternative.contains("button gap"));

        // User accepts the card (mode: 0)
        let mut applied = card.clone();
        applied.mode = 0;
        let condensed = crate::condenser::build_condensed_output("1girl, large breasts, shirt", &res.processed_tokens, &[applied], &config);
        assert!(condensed.final_prompt.contains("button gap"));
        assert!(condensed.final_prompt.contains("popped button"));

        // 2. Combo 2: Curvy thighs + fitted legwear
        let (res, _) = run("1girl, thick thighs, white thighhighs", &config);
        let card = res.changes.iter().find(|c| c.action == ChangeAction::Recommendation && c.reason.contains("Curvy Thighs & Legwear Indentation")).expect("Combo 2 card");
        assert_eq!(card.mode, 2);
        assert!(card.suggested_alternative.contains("skindentation"));

        let mut applied = card.clone();
        applied.mode = 0;
        let condensed = crate::condenser::build_condensed_output("1girl, thick thighs, white thighhighs", &res.processed_tokens, &[applied], &config);
        assert!(condensed.final_prompt.contains("skindentation"));

        // 3. Combo 4: Hourglass + High-slit dress (a generic dress or skirt isn't on the garment list)
        let (res, _) = run("1girl, wide hips, narrow waist wide hips, china dress", &config);
        let card = res.changes.iter().find(|c| c.action == ChangeAction::Recommendation && c.reason.contains("Hourglass & High-Slit Dress")).expect("Combo 4 card");
        assert!(card.suggested_alternative.contains("side slit"));

        let names = |prompt: &str| -> Vec<String> {
            run(prompt, &config).0.changes.iter().filter(|c| c.action == ChangeAction::Recommendation).map(|c| c.reason.clone()).collect()
        };
        let has = |prompt: &str, rule: &str| names(prompt).iter().any(|r| r.contains(rule));
        // Purged garment rules: Snug Shorts and High-Leg Cut no longer exist
        assert!(!has("1girl, shortstack, wide hips", "Snug Shorts"), "{:?}", names("1girl, shortstack, wide hips"));
        assert!(!has("1girl, wide hips, denim shorts", "Snug Shorts"));
        // a pleated skirt is no slit skirt
        assert!(!has("1girl, wide hips, pleated skirt", "High-Slit"));
        // "short hair" is no short stature
        assert!(!has("1girl, short hair, sword", "Petite Stature"));
        assert!(!has("1girl, short sleeves, holding sword", "Petite Stature"));
        assert!(has("1girl, petite, sword", "Petite Stature"));
    }

    #[test]
    fn two_hand_budget_and_limb_conflicts_are_resolved() {
        let config = cfg(TargetView::None);

        // 1. Two-handed pose overrides 1-handed gestures
        let (res, out) = run("1girl, hands on own hips, peace sign, waving", &config);
        assert!(out.contains("hands on own hips"));
        assert!(!out.contains("peace sign"));
        assert!(!out.contains("waving"));
        assert!(res.changes.iter().any(|c| c.original == "peace sign" && c.action == ChangeAction::PoseConflict));

        // 2. Postures conflict (standing vs sitting)
        let (res, out) = run("1girl, standing, sitting", &config);
        assert!(out.contains("standing"));
        assert!(!out.contains("sitting"));
        assert!(res.changes.iter().any(|c| c.original == "sitting" && c.action == ChangeAction::PoseConflict));

        // 3. Posture + valid leg position is preserved (sitting + crossed legs)
        let (_, out) = run("1girl, sitting, crossed legs", &config);
        assert!(out.contains("sitting"));
        assert!(out.contains("crossed legs"));

        // 4. Incompatible leg stances conflict (crossed legs vs wide stance)
        let (res, out) = run("1girl, crossed legs, wide stance", &config);
        assert!(out.contains("crossed legs"));
        assert!(!out.contains("wide stance"));
        assert!(res.changes.iter().any(|c| c.original == "wide stance" && c.action == ChangeAction::PoseConflict));

        // 5. Barefoot vs Footwear conflict
        let (res, out) = run("1girl, sneakers, barefoot", &config);
        assert!(out.contains("sneakers"));
        assert!(!out.contains("barefoot"));
        assert!(res.changes.iter().any(|c| c.original == "barefoot" && c.action == ChangeAction::PoseConflict));

        // 6. Footwear implication: loafers prunes generic shoes
        let (res, out) = run("1girl, loafers, shoes", &config);
        assert!(out.contains("loafers"));
        assert!(!out.contains("shoes"));
        assert!(res.changes.iter().any(|c| c.original == "shoes" && c.action == ChangeAction::ImpliedTag));

        // 7. Shortstack does NOT lock petite into negative prompt
        let (res, _) = run("1girl, shortstack, gigantic breasts", &config);
        for c in &res.changes {
            if c.action == ChangeAction::Recommendation {
                assert!(!c.reason.contains("Auto-Negative Locks: petite"), "petite should not be locked into negative when shortstack is used");
            }
        }
    }

    #[test]
    fn shortstack_exaggerated_proportions_bypasses_conflicts_and_adds_tag() {
        let mut config = cfg(TargetView::None);
        config.exaggerated_proportions = true;

        let (res, out) = run("1girl, shortstack, gigantic breasts, slender, wide hips", &config);
        // 1. Positive prompt contains "exaggerated proportions"
        assert!(out.contains("exaggerated proportions"), "Output should contain exaggerated proportions: {out}");
        // 2. Both gigantic breasts and wide hips and slender are preserved
        assert!(out.contains("gigantic breasts"), "gigantic breasts missing: {out}");
        assert!(out.contains("wide hips"), "wide hips missing: {out}");
        assert!(out.contains("slender"), "slender missing: {out}");
        // 3. No size conflicts or anatomy conflicts for these tags
        assert!(!res.changes.iter().any(|c| c.action == ChangeAction::SizeConflict && c.original == "slender"));
        assert!(!res.changes.iter().any(|c| c.action == ChangeAction::SizeConflict && c.original == "gigantic breasts"));
        assert!(!res.changes.iter().any(|c| c.action == ChangeAction::SizeConflict && c.original == "wide hips"));
    }
}





