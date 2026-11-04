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

    pub fn uses_explicit_weights(&self) -> bool {
        !matches!(self, Self::Flux)
    }

    pub fn system_prompt_rules(&self) -> &'static str {
        match self {
            Self::Anima => {
                "Anima supports BOTH concise Danbooru tags AND natural-language descriptive clauses. \
                 Keep core quality/subject/character tags at the start, followed by clean natural-language \
                 or hybrid tag descriptions for pose, emphasized body features, and scene. \
                 Use (feature:weight) for emphasized parts."
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
            " trending on pixiv",
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
            return Some((
                "Pony score tags ('score_9...') have no effect or add noise on non-Pony checkpoints.",
                "masterpiece, best quality",
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
            Self::FrontView => "from behind, back view",
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
                 Remove conflicting crops like 'full body, feet out of frame, extreme close-up, portrait'. \
                 CRITICAL: If the prompt contains lower/rear body tags like 'ass, butt, huge ass, feet' that pull the camera down/back, \
                 replace them with FRAMING-SAFE upper-body equivalents like 'curvy, voluptuous, hourglass figure, narrow waist, arched back' \
                 so the character keeps the exact same curvy appeal without breaking Upper Body framing!"
            }
            Self::CloseUpPortrait => {
                "The user wants a Close-Up / Portrait shot. \
                 Remove 'full body, cowboy shot, wide shot' and replace lower/mid-body physique tags (like 'abs, thick thighs, ass, legs') \
                 with portrait-safe equivalents ('athletic build, toned shoulders, defined collarbone, expressive face')."
            }
            Self::FrontView => {
                "The user wants a Front View shot. \
                 Remove 'from behind, back view' and replace rear-only anatomical tags (like 'ass, butt, back tattoo') \
                 with front-compatible silhouette tags ('wide hips, curvy, hourglass figure, thick thighs')."
            }
            Self::FromBehind => {
                "The user wants a From Behind / Back View shot. \
                 Remove 'front view, straight-on' and replace front-only torso tags (like 'cleavage, front chest') \
                 with rear-compatible equivalents ('sideboob, breasts from behind, curvy silhouette, arched back, looking back')."
            }
            Self::SideProfile => {
                "The user wants a Side Profile / From Side view. \
                 Remove 'front view, from behind, straight-on' and suggest 'from side, side profile, curvy silhouette'."
            }
            Self::FullBody => {
                "The user wants a Full Body shot showing head to toe. \
                 Remove 'portrait, close-up, upper body, cowboy shot, cropped' and suggest 'full body, standing, head to toe'."
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

/// The AI models the app offers: all Gemma 4, uncensored, registered in Ollama
/// under short names. (Qwen2.5 / Qwen3 / Qwen3.5 were tried and lost to Gemma.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiModelTier {
    /// Gemma 4 E2B (~2B active) at Q8: small models lose more to compression, and Q8
    /// fixed Q4's habit of softening explicit prompts, for under 1 GB more.
    GemmaE2B,
    /// Gemma 4 E4B (~4B active).
    GemmaE4B,
    /// Gemma 4 12B Q4: the recommended default.
    Gemma12,
}

impl AiModelTier {
    /// Dropdown order, lightest to best: the app selects the last installed one by default.
    pub const ALL: [AiModelTier; 3] = [Self::GemmaE2B, Self::GemmaE4B, Self::Gemma12];

    pub fn label(&self) -> &'static str {
        match self {
            Self::GemmaE2B => "Light — Gemma 4 E2B Q8 (~3 GB VRAM)",
            Self::GemmaE4B => "Balanced — Gemma 4 E4B (~3.5 GB VRAM)",
            Self::Gemma12 => "Maximum Fast — Gemma 4 12B Q4 (~8 GB VRAM)",
        }
    }

    /// Name the model has inside Ollama (set by `install.sh` or the in-app download).
    pub fn ollama_name(&self) -> &'static str {
        match self {
            Self::GemmaE2B => "prompt-condenser-gemma4-e2b-q8",
            Self::GemmaE4B => "prompt-condenser-gemma4-e4b",
            Self::Gemma12 => "prompt-condenser-gemma4-12b",
        }
    }

    /// How long Ollama keeps the model loaded after a scan, so the next scan starts
    /// instantly. (All current tiers fit comfortably; shorten this for bigger models on
    /// shared-memory GPUs, where a loaded model takes system RAM.)
    pub fn keep_alive(&self) -> &'static str {
        "30m"
    }

    /// Where Ollama downloads it from. The E2B/E4B builds are text-only (no 1 GB image
    /// projector), which keeps their memory use down.
    pub fn hf_ref(&self) -> &'static str {
        match self {
            Self::GemmaE2B => "hf.co/TrevorJS/gemma-4-E2B-it-uncensored-GGUF:Q8_0",
            Self::GemmaE4B => "hf.co/TrevorJS/gemma-4-E4B-it-uncensored-GGUF:Q4_K_M",
            Self::Gemma12 => "hf.co/culturerevolt/gemma-4-12b-heretic-abliterated-GGUF:Q4_K_M",
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
