use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointProfile {
    Anima,
    PonyV6,
    Illustrious,
    Sdxl,
    Sd15,
    Flux,
}

impl CheckpointProfile {
    pub fn from_index(idx: i32) -> Self {
        match idx {
            0 => Self::Anima,
            1 => Self::PonyV6,
            2 => Self::Illustrious,
            3 => Self::Sdxl,
            4 => Self::Sd15,
            5 => Self::Flux,
            _ => Self::Anima,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Anima => "Anima (Hybrid Tags + Natural Language)",
            Self::PonyV6 => "Pony Diffusion V6 XL",
            Self::Illustrious => "Illustrious / NoobAI XL",
            Self::Sdxl => "SDXL (Base / Finetunes)",
            Self::Sd15 => "SD 1.5",
            Self::Flux => "Flux (Natural Language)",
        }
    }

    pub fn max_safe_weight(&self) -> f32 {
        match self {
            Self::Anima => 1.35,
            Self::PonyV6 => 1.35,
            Self::Illustrious => 1.30,
            Self::Sdxl => 1.30,
            Self::Sd15 => 1.25,
            Self::Flux => 1.15,
        }
    }

    pub fn supports_natural_language(&self) -> bool {
        matches!(self, Self::Anima | Self::Sdxl | Self::Flux)
    }

    pub fn is_2d_anime_checkpoint(&self) -> bool {
        matches!(self, Self::Anima | Self::PonyV6 | Self::Illustrious | Self::Sd15)
    }

    pub fn uses_explicit_weights(&self) -> bool {
        !matches!(self, Self::Flux)
    }

    pub fn system_prompt_rules(&self) -> &'static str {
        match self {
            Self::Anima => {
                "Anima uses 'masterpiece, best quality' at the start, followed by clean comma-separated canonical Danbooru tags \
                 (always spaces instead of underscores, and no inline (tag:weight) syntax unless explicitly pinned) and 1-2 \
                 concise natural-language sentences in the Scene block locking subject, primary verb/posture, environment, and \
                 clothing layers (no Pony score_9 tags)."
            }
            Self::PonyV6 => {
                "Pony V6 XL requires 'score_9, score_8_up, score_7_up' at the very start. \
                 Strip non-Pony SD1.5 bloat like '8k, uhd, unreal engine, octane render, photorealistic, best quality, masterpiece'. \
                 Use clean Danbooru tags ordered by: Score/Source -> Subject Count -> Character/Body -> Framing/Pose -> Clothing -> Background."
            }
            Self::Illustrious => {
                "Illustrious / NoobAI XL uses Danbooru tags with 'masterpiece, best quality, absurdres' at the front. \
                 Keep weights <= 1.30 to prevent color burn. Strip redundant quality synonyms like 'ultra detailed, hyperdetailed, 8k, uhd'."
            }
            Self::Sdxl => {
                "SDXL works best with concise tags and short descriptive phrases grouped cleanly inside 75-token chunks. \
                 Avoid excessive nested parentheses and redundant synonyms."
            }
            Self::Sd15 => {
                "SD 1.5 uses strict comma-separated Danbooru tags with 'masterpiece, best quality, highres' first. \
                 Keep weights <= 1.25 so CLIP attention does not burn the image."
            }
            Self::Flux => {
                "Flux uses pure natural language prose and ignores (tag:1.3) weight syntax. \
                 Convert tag soup into cohesive, descriptive sentences and emphasize key body parts using vivid descriptive focus ('with prominent focus on...')."
            }
        }
    }

    /// Standard negative prompt for this checkpoint ("" = negatives aren't used).
    pub fn default_negative(&self) -> &'static str {
        match self {
            Self::Anima => "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, \
                bad anatomy, bad hands, extra fingers, missing fingers, watermark, signature, text",
            Self::PonyV6 => "score_4, score_5, score_6, worst quality, low quality, bad anatomy, bad hands, \
                extra fingers, missing fingers, blurry, jpeg artifacts, watermark, signature, text",
            Self::Illustrious => "worst quality, low quality, lowres, bad anatomy, bad hands, extra fingers, \
                missing fingers, extra digits, fewer digits, jpeg artifacts, signature, watermark, username, \
                artist name, blurry",
            Self::Sdxl => "worst quality, low quality, lowres, blurry, deformed, bad anatomy, bad hands, \
                extra limbs, disfigured, watermark, text, signature",
            Self::Sd15 => "(worst quality, low quality:1.4), lowres, bad anatomy, bad hands, extra fingers, \
                missing fingers, text, error, cropped, jpeg artifacts, signature, watermark, username, blurry",
            Self::Flux => "",
        }
    }

    pub fn canonical_quality_tags(&self) -> &'static [&'static str] {
        match self {
            Self::Anima => &["masterpiece", "best quality"],
            Self::PonyV6 => &["score_9", "score_8_up", "score_7_up"],
            Self::Illustrious => &["masterpiece", "best quality", "absurdres"],
            Self::Sdxl => &["masterpiece", "best quality"],
            Self::Sd15 => &["masterpiece", "best quality", "highres"],
            Self::Flux => &[],
        }
    }

    /// Returns Some(reason) if the tag is redundant bloat for this checkpoint.
    pub fn check_bloat_tag(&self, normalized: &str) -> Option<(&'static str, &'static str)> {
        let common_bloat = [
            "ultra detailed",
            "extremely detailed",
            "hyperdetailed",
            "hyper detailed",
            "super detailed",
            "intricate details",
            "intricate",
            "highly detailed",
            "insanely detailed",
            "8k",
            "8k uhd",
            "4k",
            "uhd",
            "hdr",
            "award winning",
            "award-winning",
            "trending on artstation",
            "trending on pixiv",
            "unreal engine",
            "unreal engine 5",
            "octane render",
            "ray tracing",
            "vray",
            "unity 3d",
            "cgsociety",
            "fine details",
            "maximum detail",
            "perfect quality",
            "amazing quality",
            "very aesthetic",
        ];

        if common_bloat.contains(&normalized) {
            return Some((
                "Redundant quality/detail filler that wastes CLIP tokens and can degrade checkpoint fidelity.",
                "",
            ));
        }

        if matches!(self, Self::PonyV6)
            && matches!(
                normalized,
                "masterpiece" | "best quality" | "absurdres" | "highres" | "ultra high res"
            )
        {
            return Some((
                "Pony V6 XL uses score tags ('score_9, score_8_up') instead of SD1.5 'masterpiece/best quality' tags.",
                "score_9, score_8_up, score_7_up",
            ));
        }

        if !matches!(self, Self::PonyV6)
            && (normalized.starts_with("score_9")
                || normalized.starts_with("score_8")
                || normalized.starts_with("score_7")
                || normalized.starts_with("score_6"))
        {
            let suggestion = match self {
                Self::Flux => "",
                Self::Illustrious => "masterpiece, best quality, absurdres",
                _ => "masterpiece, best quality",
            };
            return Some((
                "Pony score tags ('score_9...') have no effect or add noise on non-Pony checkpoints.",
                suggestion,
            ));
        }

        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetView {
    None,
    Turnaround360,
    UpperBody,
    CloseUpPortrait,
    FrontView,
    FromBehind,
    SideProfile,
    FullBody,
}

impl TargetView {
    /// Negative tags that keep the camera on this view.
    pub fn negative_tags(&self) -> &'static str {
        match self {
            Self::None => "",
            Self::Turnaround360 => "close-up, cropped, out of frame, blurry background, depth of field, dutch angle",
            Self::UpperBody => "full body, lower body, feet, shoes",
            Self::CloseUpPortrait => "full body, lower body, wide shot, distant",
            Self::FrontView => "from behind, back view, rear view, over the shoulder, looking back",
            Self::FromBehind => "front view",
            Self::SideProfile => "front view, from behind",
            Self::FullBody => "close-up, portrait, cropped, out of frame, head out of frame, feet out of frame",
        }
    }

    pub fn from_index(idx: i32) -> Self {
        match idx {
            0 => Self::None,
            1 => Self::Turnaround360,
            2 => Self::UpperBody,
            3 => Self::CloseUpPortrait,
            4 => Self::FrontView,
            5 => Self::FromBehind,
            6 => Self::SideProfile,
            7 => Self::FullBody,
            _ => Self::None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::None => "None (Keep Current Framing)",
            Self::Turnaround360 => "360 View / Character Turnaround",
            Self::UpperBody => "Upper Body / Cowboy Shot",
            Self::CloseUpPortrait => "Close-Up / Portrait",
            Self::FrontView => "Front View",
            Self::FromBehind => "From Behind / Back View",
            Self::SideProfile => "Side Profile",
            Self::FullBody => "Full Body",
        }
    }

    pub fn system_prompt_rules(&self) -> &'static str {
        match self {
            Self::None => "Keep the existing camera framing unless it contradicts an emphasized body part.",
            Self::Turnaround360 => {
                "The user wants a 360-degree character turnaround sheet (multiple views: front, side, back). \
                 Remove single-angle or cropped tags like 'looking at viewer, close-up, portrait, upper body, cowboy shot, \
                 depth of field, blurry background, sitting, dutch angle, foreshortening' and suggest \
                 'character turnaround, multiple views, full body, front view, side view, back view, simple background, standing'."
            }
            Self::UpperBody => {
                "The user wants an Upper Body / Cowboy Shot framing. \
                 Remove conflicting crops like 'full body, lower body, wide shot'. \
                 Remove lower body, leg, footwear, and action tags ('feet, barefoot, toes, shoes, boots, high heels, sneakers, sandals, footwear, ankles, anklet, socks, kneehighs, thighhighs, thighs, thick thighs, knees, calves, shins, standing, walking, running, squatting') \
                 or replace them with upper-body poise tags ('slender build, graceful posture, poised stance'). \
                 If the prompt contains lower/rear body tags like 'ass, butt, huge ass' that pull the camera down/back, \
                 replace them with FRAMING-SAFE upper-body equivalents like 'curvy, voluptuous, hourglass figure, narrow waist, arched back'."
            }
            Self::CloseUpPortrait => {
                "The user wants a Close-Up / Portrait shot. \
                 Remove 'full body, cowboy shot, wide shot, lower body' and replace lower-body physique, clothing, footwear, and action tags \
                 ('feet, barefoot, toes, shoes, boots, high heels, sneakers, sandals, footwear, ankles, anklet, socks, kneehighs, thighhighs, pantyhose, legwear, thighs, thick thighs, thigh gap, hips, wide hips, butt, ass, panties, skirt, shorts, pants, navel, midriff, standing, walking, running, sitting, squatting') \
                 with portrait-safe equivalents ('athletic build, toned shoulders, defined collarbone, expressive face, styled hair')."
            }
            Self::FrontView => {
                "The user wants a Front View shot. \
                 Remove 'from behind, back view, rear view, back focus, over the shoulder, looking back, profile, from side, side view' \
                 and replace rear-only anatomical and outfit tags ('from behind, back, bare back, shoulder blades, scapulae, nape, spine, back dimples, sacral dimples, butt, ass, ass focus, butt crack, backless dress, backless outfit, back bow, tailplug, buttplug, back tattoo, reverse bunnysuit, sideboob') \
                 with front-compatible silhouette tags ('defined collarbone, toned shoulders, wide hips, curvy, hourglass figure, thick thighs, cleavage cutout')."
            }
            Self::FromBehind => {
                "The user wants a From Behind / Back View shot. \
                 Remove 'front view, straight-on' and replace front-only torso tags \
                 ('navel, stomach, midriff, abs, cleavage, underboob, cameltoe, bulge, collarbone, chest, breasts, large breasts, medium breasts, small breasts, flat chest') \
                 with rear-compatible equivalents ('sideboob, breasts from behind, curvy silhouette, arched back, toned back, curvy hips, back dimples, defined spine')."
            }
            Self::SideProfile => {
                "The user wants a Side Profile / From Side view. \
                 Remove 'front view, from behind, straight-on' and suggest 'from side, side profile, curvy silhouette'."
            }
            Self::FullBody => {
                "The user wants a Full Body shot showing head to toe. \
                 Remove 'portrait, close-up, face focus, eye focus, cropped' and replace micro-facial detail tags \
                 ('detailed eyes, pupils, slit pupils, sclera, lip gloss, uvula, mole under eye') \
                 with full-body safe equivalents ('expressive face, clear facial features, head to toe, standing')."
            }
        }
    }
}

/// How the deep AI scan runs. Ollama chooses the GPU (Vulkan via `ollama-vulkan`)
/// and falls back to CPU on its own when the model doesn't fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiMode {
    Gpu,
    CpuOnly,
    Disabled,
}

impl AiMode {
    pub fn from_index(idx: i32) -> Self {
        match idx {
            1 => Self::CpuOnly,
            2 => Self::Disabled,
            _ => Self::Gpu,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Gpu => "GPU (Vulkan via Ollama, auto CPU fallback)",
            Self::CpuOnly => "CPU only (keep VRAM free for ComfyUI)",
            Self::Disabled => "AI off (rules only)",
        }
    }
}

/// The AI models the app offers: Good, Better, Best, uncensored, registered in Ollama.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiModelTier {
    /// Good: Gemma 4 E4B (~4B active) Q4_K_M (~3.5 GB VRAM).
    Good,
    /// Better: Llama 3.1 8B Instruct Abliterated (~5 GB VRAM).
    Better,
    /// Best: Gemma 4 12B Abliterated (~8 GB VRAM).
    Best,
}

impl AiModelTier {
    /// Dropdown order, Good to Best: the app selects the last installed one by default.
    pub const ALL: [AiModelTier; 3] = [Self::Good, Self::Better, Self::Best];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Good => "Good — Gemma 4 E4B (~3.5 GB VRAM)",
            Self::Better => "Better — Llama 3.1 8B Abliterated (~5 GB VRAM)",
            Self::Best => "Best — Gemma 4 12B Abliterated (~8 GB VRAM)",
        }
    }

    /// Name the model has inside Ollama (set by `install.sh` or the in-app download).
    pub fn ollama_name(&self) -> &'static str {
        match self {
            Self::Good => "prompt-condenser-gemma4-e4b",
            Self::Better => "richardyoung/llama-3.1-8b-instruct-abliterated",
            Self::Best => "huihui_ai/gemma-4-abliterated:12b",
        }
    }

    /// How long Ollama keeps the model loaded after a scan, so the next scan starts
    /// instantly.
    pub fn keep_alive(&self) -> &'static str {
        "30m"
    }

    /// Where Ollama downloads it from.
    pub fn hf_ref(&self) -> &'static str {
        match self {
            Self::Good => "hf.co/TrevorJS/gemma-4-E4B-it-uncensored-GGUF:Q4_K_M",
            Self::Better => "richardyoung/llama-3.1-8b-instruct-abliterated:latest",
            Self::Best => "huihui_ai/gemma-4-abliterated:12b",
        }
    }
}

/// Camera angles (prompting + photography). (dropdown label, tags). Index 0 = none.
/// A tags value starting with '{' is a Dynamic Prompts variation group.
pub const CAMERA_ANGLES: &[(&str, &str)] = &[
    ("None (keep the prompt's angle)", ""),
    ("Eye level / straight-on", "straight-on, eye level"),
    ("High angle (looking down)", "from above, high angle"),
    ("Low angle (looking up)", "from below, low angle"),
    ("Bird's-eye view (overhead)", "from above, bird's-eye view, overhead shot"),
    ("Worm's-eye view (ground up)", "from below, worm's-eye view, extreme low angle"),
    ("Dutch angle (tilted frame)", "dutch angle, tilted frame"),
    ("Three-quarter view", "three-quarter view"),
    ("Profile / side on", "from side, profile"),
    ("Over the shoulder", "over the shoulder, from behind, looking back"),
    ("POV / first person", "pov, first-person view"),
    ("Selfie", "selfie, from above, holding phone"),
    ("Wide-angle / fisheye", "wide-angle lens, fisheye"),
    ("Telephoto (compressed, blurred background)", "telephoto lens, depth of field, blurry background"),
    ("Foreshortening (dynamic perspective)", "foreshortening, dynamic angle, perspective"),
    ("Cinematic angle", "cinematic angle, dynamic angle"),
    ("Upside down", "upside-down"),
    ("Random angle (dynamic prompt)", "{from above|from below|dutch angle|three-quarter view|from side|straight-on}"),
];

/// Camera framing for one character in the Scene Panel (dropdown label, tags). Index 0 = none.
pub const VIEWS: &[(&str, &str)] = &[
    ("None (keep the prompt's framing)", ""),
    ("Close-up / portrait", "portrait, close-up"),
    ("Upper body", "upper body"),
    ("Cowboy shot (thighs up)", "cowboy shot"),
    ("Full body", "full body"),
    ("Lower body", "lower body"),
    ("Front view", "front view"),
    ("From behind", "from behind, back view"),
    ("Side profile", "from side, profile"),
    ("Wide shot (small in frame)", "wide shot"),
];

/// Body positions and poses (prompting + photography). (dropdown label, tags). Index 0 = none.
pub const POSES: &[(&str, &str)] = &[
    ("None (keep the prompt's pose)", ""),
    ("Standing", "standing"),
    ("Contrapposto / S-curve", "standing, contrapposto, hand on hip"),
    ("Hands on hips (power pose)", "standing, hands on hips"),
    ("Arms crossed", "standing, crossed arms"),
    ("Arms up / stretching", "arms up, stretching"),
    ("Leaning against a wall", "leaning against wall, leaning back"),
    ("Hand on face / thinking", "hand on own chin, thinking"),
    ("Peace sign", "peace sign, v"),
    ("Waving", "waving"),
    ("Walking", "walking"),
    ("Running", "running, motion blur"),
    ("Jumping / midair", "jumping, midair"),
    ("Dancing", "dancing"),
    ("Dynamic / action pose", "dynamic pose, action pose"),
    ("Fighting stance", "fighting stance"),
    ("Looking back over shoulder", "looking back, looking over shoulder"),
    ("Sitting", "sitting"),
    ("Sitting cross-legged", "sitting, crossed legs, indian style"),
    ("Sitting on a chair", "sitting on chair"),
    ("Hugging knees", "sitting, knees to chest, hugging own legs"),
    ("Seiza (kneeling on heels)", "seiza"),
    ("Wariza (W-sitting)", "wariza"),
    ("Kneeling", "kneeling"),
    ("Squatting", "squatting"),
    ("Lying on back", "lying, on back"),
    ("Lying on stomach", "lying, on stomach"),
    ("Lying on side", "lying, on side"),
    ("Reclining", "reclining"),
    ("All fours", "all fours"),
    ("Bent over", "bent over"),
    ("Arched back", "arched back"),
    ("Floating", "floating"),
    ("Sleeping", "sleeping, closed eyes"),
    ("Candid (natural, unposed)", "candid, natural pose"),
    ("Random pose (dynamic prompt)", "{standing|sitting|leaning against wall|dynamic pose|looking back|arms up}"),
];

/// Body proportion presets (dropdown label, tags). Index 0 = none.
pub const PROPORTIONS: &[(&str, &str)] = &[
    ("None (keep prompt's proportion)", ""),
    ("Flat chest (slender, aa-cup)", "flat chest, aa cup, small underbust, slender torso"),
    ("Small breasts (perky, a/b-cup)", "small breasts, a cup, perky breasts"),
    ("Medium breasts (full, c/d-cup)", "medium breasts, c cup, proportionate body"),
    ("Large breasts (curvy, e/f-cup)", "large breasts, e cup, curvy, voluptuous"),
    ("Huge breasts (heavy, g/h-cup)", "huge breasts, g cup, heavy breasts, deep cleavage"),
    ("Gigantic breasts (hyper, massive)", "gigantic breasts, massive breasts, voluptuous"),
    ("Thick thighs & wide hips (curvy)", "thick thighs, wide hips, plump thighs, curvy"),
    ("Muscular & athletic (toned, abs)", "muscular female, abs, toned, athletic, defined muscles"),
];

/// Every tag that belongs to a list above (lowercase), used to spot existing angle /
/// pose tags in a prompt that a dropdown choice should replace.
pub fn group_tags(list: &[(&str, &str)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (_, tags) in list {
        if tags.starts_with('{') {
            continue;
        }
        for t in tags.split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()) {
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
    out
}

/// How much the deep scan may enrich the prompt. Index order matches the UI dropdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetailLevel {
    /// Fix problems only; add almost nothing.
    Faithful,
    /// Fix problems and fill obvious gaps (lighting, background, expression).
    #[default]
    Balanced,
    /// Full descriptive expansion: a detail checklist, descriptive pairs, 40-60 tags.
    Rich,
}

impl DetailLevel {
    pub fn from_index(idx: i32) -> Self {
        match idx {
            0 => Self::Faithful,
            2 => Self::Rich,
            _ => Self::Balanced,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloat_tag_detection() {
        let sd = CheckpointProfile::Sd15;
        assert!(sd.check_bloat_tag("trending on pixiv").is_some());
        assert!(sd.check_bloat_tag("trending on artstation").is_some());

        let flux = CheckpointProfile::Flux;
        let (_, sug) = flux.check_bloat_tag("score_9").expect("Pony score tag on Flux is bloat");
        assert_eq!(sug, "", "Flux should not suggest quality tags");

        let ill = CheckpointProfile::Illustrious;
        let (_, sug) = ill.check_bloat_tag("score_9").expect("Pony score tag on Illustrious is bloat");
        assert_eq!(sug, "masterpiece, best quality, absurdres");

        let pony = CheckpointProfile::PonyV6;
        assert!(pony.check_bloat_tag("score_9").is_none());
        assert!(pony.check_bloat_tag("masterpiece").is_some());
    }
}
