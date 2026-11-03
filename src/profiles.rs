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

/// The two pre-packaged uncensored models, registered in Ollama under short names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiModelTier {
    Low,
    High,
    Best,
    Max,
}

impl AiModelTier {
    pub const ALL: [AiModelTier; 4] = [Self::Low, Self::High, Self::Best, Self::Max];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "Low — Fast 1.5B Uncensored",
            Self::High => "High — 3B Uncensored",
            Self::Best => "Best — 7B Uncensored (recommended, ~4.7 GB)",
            Self::Max => "Max — 14B Uncensored (~9 GB)",
        }
    }

    /// Name the model has inside Ollama (set by `install.sh` or the in-app download).
    pub fn ollama_name(&self) -> &'static str {
        match self {
            Self::Low => "prompt-condenser-low",
            Self::High => "prompt-condenser-high",
            Self::Best => "prompt-condenser-best",
            Self::Max => "prompt-condenser-max",
        }
    }

    /// Where Ollama downloads it from (same GGUF files the installer bundles).
    pub fn hf_ref(&self) -> &'static str {
        match self {
            Self::Low => "hf.co/mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF:Q4_K_M",
            Self::High => "hf.co/mradermacher/Qwen2.5-3B-Instruct-abliterated-GGUF:Q4_K_M",
            Self::Best => "hf.co/mradermacher/Qwen2.5-7B-Instruct-abliterated-GGUF:Q4_K_M",
            Self::Max => "hf.co/mradermacher/Qwen2.5-14B-Instruct-abliterated-GGUF:Q4_K_M",
        }
    }
}

