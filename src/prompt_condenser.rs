//! Prompt-Condenser: dual-pass Moondream + WD14 pipeline (Rust port of
//! `prompt_condenser_dual_pass.py`, which stays the reference implementation).
//!
//! Uses only the standard library, so it drops into any crate: add this file as
//! `src/prompt_condenser.rs` and declare it in main.rs / lib.rs:
//!
//! ```text
//! #[allow(dead_code)] // until every helper is wired up
//! mod prompt_condenser;
//! #[cfg(test)]
//! mod prompt_condenser_parity; // optional: checks Rust == the Python reference
//! ```
//!
//! Flow (same as the Python reference):
//!   1. WD14 tags the image; Moondream describes it in two passes (prose).
//!   2. Moondream prose is converted to WD-style tags.
//!   3. WD14 adjusts the Moondream tags: WD14 wins on attributes (colors, hair
//!      length, breast size), Moondream wins on pose and camera angle.
//!   4. The merged tags are split into scene / character prompts by mode.
//!   5. Optionally Gemma (Ollama) polishes the result, optionally with the image.
//!
//! All I/O is injected: you pass closures for Moondream and Gemma, so this module
//! has no HTTP, image or async dependencies. Every failure falls back to the
//! mechanical merge and is reported in `PipelineResult::warnings`.
//!
//! The single-pass path is not here: keep your existing code for it and call
//! `run_dual_pass` only when the dual-pass toggle is on.

use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants (Phase 0)
// ---------------------------------------------------------------------------
pub const VLM_TEMPERATURE: f32 = 0.05;
pub const MOONDREAM_TEMPERATURE: f32 = VLM_TEMPERATURE;
pub const GEMMA_TEMPERATURE: f32 = 0.10;
/// Pass these to your WD14 tagger; this module only receives the tag string.
pub const WD14_GENERAL_THRESHOLD: f32 = 0.22;
pub const WD14_CHARACTER_THRESHOLD: f32 = 0.85;
/// The model supports 128K, but the input is only ~1.5K tokens and every extra token of
/// context costs KV-cache memory (shared with the system on the BC-250, and on a LAN
/// server this request decides how much memory Ollama reserves). 16K matches the app's
/// other calls.
pub const GEMMA_CONTEXT_WINDOW: u32 = 16_384;
pub const GEMMA_MAX_TOKENS: u32 = 768;
pub const GEMMA_MAX_TOKENS_RETRY: u32 = 1024;
pub const VLM_RETRIES: u32 = 1;
pub const MOONDREAM_RETRIES: u32 = VLM_RETRIES;
pub const VLM_RETRY_TEMP_BUMP: f32 = 0.2;
pub const MOONDREAM_RETRY_TEMP_BUMP: f32 = VLM_RETRY_TEMP_BUMP;

#[derive(Clone, Debug, PartialEq)]
pub struct ZoneBox {
    pub name: &'static str,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClothingLayerDef {
    pub tag: &'static str,
    pub region: &'static str,
    pub layer: &'static str,
    pub rank: u8,
    pub covers: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropType {
    CharacterZone,
    Quadrant,
}

/// Runtime settings; `Config::default()` matches the Python constants.
#[derive(Clone, Debug)]
pub struct Config {
    /// Prepended to the scene prompt; empty string disables it.
    pub quality_prefix: String,
    /// Write "(" as "\(" (A1111/ComfyUI prompt syntax).
    pub escape_parentheses: bool,
    /// Mechanical path only: also append VLM wording that matched no tag.
    pub keep_unmatched_prose: bool,
    /// Two-source rule: prune VLM tags that have only 1 source and were not in WD14.
    pub require_two_sources: bool,
    pub vlm_temperature: f32,
    pub vlm_retries: u32,
    pub vlm_retry_temp_bump: f32,
    pub moondream_temperature: f32,
    pub moondream_retries: u32,
    pub moondream_retry_temp_bump: f32,
    pub gemma_temperature: f32,
    pub gemma_max_tokens: u32,
    pub gemma_max_tokens_retry: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            quality_prefix: QUALITY_PREFIX_DEFAULT.to_string(),
            escape_parentheses: false,
            keep_unmatched_prose: false,
            require_two_sources: false,
            vlm_temperature: VLM_TEMPERATURE,
            vlm_retries: VLM_RETRIES,
            vlm_retry_temp_bump: VLM_RETRY_TEMP_BUMP,
            moondream_temperature: MOONDREAM_TEMPERATURE,
            moondream_retries: MOONDREAM_RETRIES,
            moondream_retry_temp_bump: MOONDREAM_RETRY_TEMP_BUMP,
            gemma_temperature: GEMMA_TEMPERATURE,
            gemma_max_tokens: GEMMA_MAX_TOKENS,
            gemma_max_tokens_retry: GEMMA_MAX_TOKENS_RETRY,
        }
    }
}

// ---- GENERATED from prompt_condenser_dual_pass.py by tools/gen_data.py: do not edit by hand ----

pub const ZONE_FORMAT_HINT: &str = " Format each zone on its own line exactly as 'N. Zone name: text'. Plain text only, no markdown.";
pub const QUALITY_PREFIX_DEFAULT: &str = "masterpiece, best quality";
pub const GEMMA_OLLAMA_MODEL: &str = "huihui_ai/gemma-4-abliterated:12b";
pub const VISION_TAIL: &str = "You are also given the original image. Do a final check of the merged tags against it: remove tags that clearly contradict the image, add clearly visible details the tags missed, and keep everything else. Do not invent anything you cannot see.";
pub const MODE_A_PASS1_PROMPT: &str = "Analyze physical space and body proportions and silhouette as visible. Ignore clothing and face. 6 zones: 1. Scene & Lighting 2. Camera & Depth 3. Core Pose 4. Upper Body Shape 5. Lower Body Shape 6. Physical Anchors Format each zone on its own line exactly as 'N. Zone name: text'. Plain text only, no markdown.";
pub const MODE_A_PASS2_PROMPT: &str = "Analyze facial identity and wardrobe. Ignore body weight and environment. 6 zones: 1. Head & Hair 2. Face & Eyes 3. Torso Clothing 4. Leg Clothing 5. Footwear 6. Accessories & Props Format each zone on its own line exactly as 'N. Zone name: text'. Plain text only, no markdown.";
pub const MODE_A_LLM_INSTRUCTION: &str = "Merge 'Scene & Body Shape' (Pass 1) with 'Face & Clothes' (Pass 2). Dress the body described in Pass 1 using the wardrobe from Pass 2. Do not invent body proportions that are not in the input. Place environment, lighting and camera in scene_prompt. Place face, hair, body shape, clothing and props in character_prompt. The tag lists have already been converted to WD-style tags and reconciled (WD14 wins on attributes such as colors, Moondream wins on pose and camera): treat them as ground truth and use the blueprints only for detail the tags lack. Output Danbooru-style tags only. Output valid JSON with exactly two keys: \"scene_prompt\" and \"character_prompt\". Both values must be single strings of comma-separated tags. No markdown, no commentary.";
pub const MODE_A_PASS1_KEYWORDS: &[&str] = &[
    "all fours",
    "alley",
    "background",
    "backlighting",
    "balcony",
    "bathroom",
    "beach",
    "bed",
    "bedroom",
    "bench",
    "blurry",
    "bokeh",
    "bookshelf",
    "breasts",
    "bridge",
    "building",
    "cafe",
    "carpet",
    "castle",
    "cave",
    "ceiling",
    "chair",
    "chromatic aberration",
    "city",
    "cityscape",
    "classroom",
    "cleavage",
    "close-up",
    "cloud",
    "clouds",
    "couch",
    "cowboy shot",
    "crouching",
    "curtains",
    "dawn",
    "depth of field",
    "desert",
    "desk",
    "door",
    "dusk",
    "dutch angle",
    "dynamic angle",
    "evening",
    "field",
    "film grain",
    "fisheye",
    "floor",
    "fog",
    "foreshortening",
    "forest",
    "from above",
    "from behind",
    "from below",
    "from side",
    "full body",
    "garden",
    "grasping",
    "grass",
    "gym",
    "hill",
    "holding",
    "indoor",
    "indoors",
    "jumping",
    "kitchen",
    "kneeling",
    "lake",
    "lamp",
    "leaning",
    "lens flare",
    "library",
    "light particles",
    "light rays",
    "lighting",
    "locker room",
    "lower body",
    "lying",
    "mist",
    "moon",
    "moonlight",
    "motion blur",
    "mountain",
    "night",
    "ocean",
    "on back",
    "on stomach",
    "out of frame",
    "outdoor",
    "outdoors",
    "park",
    "pool",
    "poolside",
    "portrait",
    "pov",
    "pressed",
    "rain",
    "raining",
    "restaurant",
    "river",
    "road",
    "rooftop",
    "room",
    "ruins",
    "running",
    "sea",
    "selfie",
    "shadow",
    "shelf",
    "shrine",
    "silhouette",
    "sitting",
    "skindentation",
    "sky",
    "skyscraper",
    "snow",
    "snowing",
    "sofa",
    "squatting",
    "squished",
    "stage",
    "stairs",
    "standing",
    "starry sky",
    "street",
    "stretching",
    "sunbeam",
    "sunlight",
    "sunrise",
    "sunset",
    "table",
    "temple",
    "thick thighs",
    "tree",
    "trees",
    "underwater",
    "upper body",
    "vignetting",
    "walking",
    "wall",
    "waterfall",
    "wide angle",
    "wide hips",
    "wide shot",
    "window",
];
pub const MODE_B_PASS1_PROMPT: &str = "Analyze physical space and skeletal pose. Ignore clothing, face, and body thickness. 6 zones: 1. Scene & Lighting 2. Camera & Depth 3. Center of Gravity & Stance 4. Torso & Head Alignment 5. Limb Positioning 6. Environmental Anchors Format each zone on its own line exactly as 'N. Zone name: text'. Plain text only, no markdown.";
pub const MODE_B_PASS2_PROMPT: &str = "Analyze physical body proportions, facial identity, and wardrobe. Ignore environment and underlying pose mechanics. 6 zones: 1. Head & Face 2. Upper Body Proportions & Clothing 3. Lower Body Proportions & Clothing 4. Legwear & Footwear 5. Skin & Fabric Physics 6. Accessories & Props Format each zone on its own line exactly as 'N. Zone name: text'. Plain text only, no markdown.";
pub const MODE_B_LLM_INSTRUCTION: &str = "Merge 'Scene & Pose' (Pass 1) with 'Character & Body Shape' (Pass 2). Map the proportions and clothing from Pass 2 onto the pose from Pass 1 without inventing proportions that are not in the input. Place environment, lighting and camera in scene_prompt. Place face, hair, body proportions, clothing, props and the skeletal pose in character_prompt (the pose belongs to the character, so it never becomes shared scene). The tag lists have already been converted to WD-style tags and reconciled (WD14 wins on attributes such as colors, Moondream wins on pose and camera): treat them as ground truth and use the blueprints only for detail the tags lack. Output Danbooru-style tags only. Output valid JSON with exactly two keys: \"scene_prompt\" and \"character_prompt\". Both values must be single strings of comma-separated tags. No markdown, no commentary.";
pub const MODE_B_PASS1_KEYWORDS: &[&str] = &[
    "all fours",
    "alley",
    "arms behind",
    "arms up",
    "background",
    "backlighting",
    "balcony",
    "bathroom",
    "beach",
    "bed",
    "bedroom",
    "bench",
    "blurry",
    "bokeh",
    "bookshelf",
    "bridge",
    "building",
    "cafe",
    "carpet",
    "castle",
    "cave",
    "ceiling",
    "chair",
    "chromatic aberration",
    "city",
    "cityscape",
    "classroom",
    "close-up",
    "cloud",
    "clouds",
    "couch",
    "cowboy shot",
    "crossed legs",
    "crouching",
    "curtains",
    "dawn",
    "depth of field",
    "desert",
    "desk",
    "door",
    "dusk",
    "dutch angle",
    "dynamic angle",
    "evening",
    "field",
    "film grain",
    "fisheye",
    "floor",
    "fog",
    "foreshortening",
    "forest",
    "from above",
    "from behind",
    "from below",
    "from side",
    "full body",
    "garden",
    "grasping",
    "grass",
    "gym",
    "head tilt",
    "hill",
    "holding",
    "indoor",
    "indoors",
    "jumping",
    "kitchen",
    "kneeling",
    "lake",
    "lamp",
    "leaning",
    "legs crossed",
    "lens flare",
    "library",
    "light particles",
    "light rays",
    "lighting",
    "locker room",
    "looking back",
    "lower body",
    "lying",
    "mist",
    "moon",
    "moonlight",
    "motion blur",
    "mountain",
    "night",
    "object intersection",
    "ocean",
    "on back",
    "on stomach",
    "out of frame",
    "outdoor",
    "outdoors",
    "park",
    "pool",
    "poolside",
    "portrait",
    "pov",
    "rain",
    "raining",
    "restaurant",
    "river",
    "road",
    "rooftop",
    "room",
    "ruins",
    "running",
    "sea",
    "selfie",
    "shadow",
    "shelf",
    "shrine",
    "silhouette",
    "sitting",
    "sky",
    "skyscraper",
    "snow",
    "snowing",
    "sofa",
    "spread legs",
    "squatting",
    "stage",
    "stairs",
    "standing",
    "starry sky",
    "street",
    "stretching",
    "sunbeam",
    "sunlight",
    "sunrise",
    "sunset",
    "table",
    "temple",
    "tree",
    "trees",
    "underwater",
    "upper body",
    "vignetting",
    "walking",
    "wall",
    "waterfall",
    "wide angle",
    "wide shot",
    "window",
];
pub const BUILTIN_VOCAB: &[&str] = &[
    "arms up",
    "backlighting",
    "bag",
    "bangs",
    "barefoot",
    "bed",
    "belt",
    "bikini",
    "blue eyes",
    "blush",
    "boots",
    "braid",
    "brown eyes",
    "chair",
    "close-up",
    "closed mouth",
    "couch",
    "cowboy shot",
    "crossed legs",
    "cup",
    "denim shorts",
    "depth of field",
    "desk",
    "dress",
    "dynamic angle",
    "earrings",
    "floor",
    "foreshortening",
    "from above",
    "from behind",
    "from below",
    "from side",
    "full body",
    "glasses",
    "gloves",
    "green eyes",
    "hat",
    "high heels",
    "holding",
    "hoodie",
    "indoor",
    "indoors",
    "jacket",
    "jeans",
    "kneeling",
    "large breasts",
    "leaning",
    "long hair",
    "looking at viewer",
    "looking back",
    "lying",
    "medium breasts",
    "necklace",
    "night",
    "no humans",
    "open mouth",
    "outdoor",
    "outdoors",
    "pants",
    "phone",
    "ponytail",
    "portrait",
    "red eyes",
    "sandals",
    "school uniform",
    "shirt",
    "short hair",
    "shorts",
    "simple background",
    "sitting",
    "skirt",
    "sky",
    "small breasts",
    "smile",
    "sneakers",
    "socks",
    "spread legs",
    "squatting",
    "standing",
    "sunset",
    "sweater",
    "swimsuit",
    "t-shirt",
    "table",
    "thick thighs",
    "thighhighs",
    "twintails",
    "umbrella",
    "uniform",
    "upper body",
    "white background",
    "wide hips",
    "window",
];
pub const COLORS: &[&str] = &[
    "aqua",
    "beige",
    "black",
    "blonde",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "navy",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
];
pub const SINGLE_STOP: &[&str] = &[
    "air",
    "arms",
    "back",
    "body",
    "close",
    "dark",
    "edge",
    "eyes",
    "face",
    "front",
    "hair",
    "hand",
    "head",
    "large",
    "left",
    "legs",
    "light",
    "line",
    "long",
    "one",
    "open",
    "point",
    "right",
    "short",
    "side",
    "small",
    "top",
    "two",
];
pub const NEGATORS: &[&str] = &[
    "aren't",
    "doesn't",
    "don't",
    "hasn't",
    "haven't",
    "isn't",
    "lacking",
    "lacks",
    "missing",
    "neither",
    "never",
    "no",
    "none",
    "nor",
    "not",
    "wasn't",
    "without",
];
pub const HEDGES: &[&str] = &[
    "appear",
    "appearing",
    "appears",
    "could",
    "likely",
    "maybe",
    "might",
    "perhaps",
    "possibly",
    "presumably",
    "probably",
    "resembles",
    "seem",
    "seemingly",
    "seems",
    "suggests",
];
pub const SCOPE_END: &[&str] = &[
    "although",
    "but",
    "however",
    "though",
    "while",
    "with",
    "yet",
];
pub const NEW_STATEMENT_VERBS: &[&str] = &[
    "are",
    "carries",
    "has",
    "have",
    "holds",
    "is",
    "shows",
    "wears",
];
pub const CROP_ARTIFACT_TAGS: &[&str] = &[
    "arm out of frame",
    "armpit focus",
    "back focus",
    "breast focus",
    "close-up",
    "cowboy shot",
    "cropped legs",
    "eye focus",
    "face focus",
    "face out of frame",
    "feet out of frame",
    "focus",
    "foot focus",
    "full body",
    "hands out of frame",
    "head out of frame",
    "hip focus",
    "legs out of frame",
    "lower body",
    "navel focus",
    "out of frame",
    "partially submerged",
    "portrait",
    "solo focus",
    "upper body",
    "wide shot",
];
pub const RATING_TAGS: &[&str] = &[
    "explicit",
    "general",
    "questionable",
    "sensitive",
];
pub const BODY_SIZE_AND_COUNT_TAGS: &[&str] = &[
    "1boy",
    "1girl",
    "2boys",
    "2girls",
    "3boys",
    "3girls",
    "4boys",
    "4girls",
    "5boys",
    "5girls",
    "6+boys",
    "6+girls",
    "chibi",
    "flat chest",
    "gigantic breasts",
    "huge breasts",
    "large breasts",
    "medium breasts",
    "multiple boys",
    "multiple girls",
    "petite",
    "short",
    "small breasts",
    "solo",
    "tall",
];
pub const CONCRETE_SCENE_OBJECTS: &[&str] = &[
    "armchair",
    "bed",
    "book",
    "bookcase",
    "bookshelf",
    "bottle",
    "bouquet",
    "bush",
    "cactus",
    "camera",
    "carpet",
    "cellphone",
    "chair",
    "chandelier",
    "cherry blossoms",
    "clock",
    "computer",
    "couch",
    "cup",
    "curtain",
    "curtains",
    "cushion",
    "desk",
    "doll",
    "drapes",
    "drink",
    "fan",
    "fern",
    "flower",
    "flower pot",
    "flowerpot",
    "flowers",
    "foliage",
    "food",
    "grass",
    "guitar",
    "gun",
    "hedge",
    "houseplant",
    "instrument",
    "ivy",
    "lamp",
    "lantern",
    "laptop",
    "leaf",
    "leaves",
    "lily",
    "mat",
    "mirror",
    "monitor",
    "mug",
    "painting",
    "parasol",
    "petal",
    "phone",
    "picture frame",
    "pillow",
    "plant",
    "plushie",
    "poster",
    "potted plant",
    "rose",
    "rug",
    "shelf",
    "signboard",
    "smartphone",
    "sofa",
    "staff",
    "street lamp",
    "succulent",
    "sunflower",
    "sword",
    "table",
    "teacup",
    "television",
    "tree",
    "tulip",
    "umbrella",
    "vase",
    "vines",
    "violin",
    "wand",
    "weapon",
    "window",
    "wine glass",
];
pub const SCENE_KEYWORDS: &[&str] = &[
    "alley",
    "background",
    "backlighting",
    "balcony",
    "bathroom",
    "beach",
    "bed",
    "bedroom",
    "bench",
    "blurry",
    "bokeh",
    "bookshelf",
    "bridge",
    "building",
    "cafe",
    "carpet",
    "castle",
    "cave",
    "ceiling",
    "chair",
    "chromatic aberration",
    "city",
    "cityscape",
    "classroom",
    "close-up",
    "cloud",
    "clouds",
    "couch",
    "cowboy shot",
    "curtains",
    "dawn",
    "depth of field",
    "desert",
    "desk",
    "door",
    "dusk",
    "dutch angle",
    "dynamic angle",
    "evening",
    "field",
    "film grain",
    "fisheye",
    "floor",
    "fog",
    "foreshortening",
    "forest",
    "from above",
    "from behind",
    "from below",
    "from side",
    "full body",
    "garden",
    "grass",
    "gym",
    "hill",
    "indoor",
    "indoors",
    "kitchen",
    "lake",
    "lamp",
    "lens flare",
    "library",
    "light particles",
    "light rays",
    "lighting",
    "locker room",
    "lower body",
    "mist",
    "moon",
    "moonlight",
    "motion blur",
    "mountain",
    "night",
    "ocean",
    "out of frame",
    "outdoor",
    "outdoors",
    "park",
    "pool",
    "poolside",
    "portrait",
    "pov",
    "rain",
    "raining",
    "restaurant",
    "river",
    "road",
    "rooftop",
    "room",
    "ruins",
    "sea",
    "selfie",
    "shadow",
    "shelf",
    "shrine",
    "silhouette",
    "sky",
    "skyscraper",
    "snow",
    "snowing",
    "sofa",
    "stage",
    "stairs",
    "starry sky",
    "street",
    "sunbeam",
    "sunlight",
    "sunrise",
    "sunset",
    "table",
    "temple",
    "tree",
    "trees",
    "underwater",
    "upper body",
    "vignetting",
    "wall",
    "waterfall",
    "wide angle",
    "wide shot",
    "window",
];
pub const EXCLUSIVE_GROUPS: &[&[&str]] = &[
    &["kneeling", "lying", "sitting", "squatting", "standing"],
    &["from above", "from behind", "from below", "from side"],
];
pub const ATTRIBUTE_GROUPS: &[&[&str]] = &[
    &["long hair", "medium hair", "short hair", "very long hair", "very short hair"],
    &["flat chest", "gigantic breasts", "huge breasts", "large breasts", "medium breasts", "small breasts"],
];
pub const COEXISTING_PAIRS: &[(&str, &str)] = &[
    ("bare arms", "halterneck"),
    ("bare arms", "sleeveless dress"),
    ("bare arms", "sleeveless shirt"),
    ("bare arms", "strapless dress"),
    ("bare arms", "tube top"),
    ("bare shoulders", "halterneck"),
    ("bare shoulders", "off-shoulder shirt"),
    ("bare shoulders", "strapless dress"),
    ("bare shoulders", "tube top"),
    ("bare stomach", "bikini"),
    ("bikini", "bare stomach"),
    ("bikini", "cleavage"),
    ("bikini", "deep cleavage"),
    ("bikini", "navel"),
    ("boots", "kneehighs"),
    ("boots", "socks"),
    ("boots", "thighhighs"),
    ("camisole", "cardigan"),
    ("cardigan", "camisole"),
    ("cleavage", "bikini"),
    ("cleavage", "crop top"),
    ("cleavage", "halterneck"),
    ("cleavage", "jacket"),
    ("cleavage", "open jacket"),
    ("cleavage", "open shirt"),
    ("cleavage", "plunge dress"),
    ("cleavage", "strapless dress"),
    ("cleavage", "tube top"),
    ("cleavage", "unbuttoned shirt"),
    ("coat", "dress"),
    ("crop top", "cleavage"),
    ("crop top", "midriff"),
    ("crop top", "navel"),
    ("deep cleavage", "bikini"),
    ("deep cleavage", "open jacket"),
    ("deep cleavage", "plunge dress"),
    ("deep cleavage", "tube top"),
    ("dress", "coat"),
    ("gloves", "sleeveless"),
    ("halterneck", "bare arms"),
    ("halterneck", "bare shoulders"),
    ("halterneck", "cleavage"),
    ("jacket", "cleavage"),
    ("jacket", "shirt"),
    ("jacket", "t-shirt"),
    ("kneehighs", "boots"),
    ("midriff", "crop top"),
    ("midriff", "open jacket"),
    ("midriff", "open shirt"),
    ("midriff", "tube top"),
    ("navel", "bikini"),
    ("navel", "crop top"),
    ("navel", "open jacket"),
    ("navel", "open shirt"),
    ("navel", "tube top"),
    ("off-shoulder shirt", "bare shoulders"),
    ("open jacket", "cleavage"),
    ("open jacket", "deep cleavage"),
    ("open jacket", "midriff"),
    ("open jacket", "navel"),
    ("open jacket", "shirt"),
    ("open jacket", "t-shirt"),
    ("open shirt", "cleavage"),
    ("open shirt", "midriff"),
    ("open shirt", "navel"),
    ("plunge dress", "cleavage"),
    ("plunge dress", "deep cleavage"),
    ("shirt", "jacket"),
    ("shirt", "open jacket"),
    ("shoes", "socks"),
    ("shoes", "thighhighs"),
    ("shorts", "thighhighs"),
    ("skirt", "thighhighs"),
    ("sleeveless", "gloves"),
    ("sleeveless dress", "bare arms"),
    ("sleeveless shirt", "bare arms"),
    ("sneakers", "socks"),
    ("socks", "boots"),
    ("socks", "shoes"),
    ("socks", "sneakers"),
    ("strapless dress", "bare arms"),
    ("strapless dress", "bare shoulders"),
    ("strapless dress", "cleavage"),
    ("t-shirt", "jacket"),
    ("t-shirt", "open jacket"),
    ("thighhighs", "boots"),
    ("thighhighs", "shoes"),
    ("thighhighs", "shorts"),
    ("thighhighs", "skirt"),
    ("tube top", "bare arms"),
    ("tube top", "bare shoulders"),
    ("tube top", "cleavage"),
    ("tube top", "deep cleavage"),
    ("tube top", "midriff"),
    ("tube top", "navel"),
    ("unbuttoned shirt", "cleavage"),
];
pub const OCCLUSION_SUPPRESSIONS: &[(&str, &str)] = &[
    ("blindfold", "eyes"),
    ("boots", "barefoot"),
    ("burka", "face"),
    ("closed eyes", "looking at viewer"),
    ("faceless", "eyes"),
    ("gloves", "bare hands"),
    ("high heels", "barefoot"),
    ("jeans", "bare legs"),
    ("loafers", "barefoot"),
    ("long pants", "bare legs"),
    ("pants", "bare legs"),
    ("sandals", "barefoot"),
    ("shoes", "barefoot"),
    ("sneakers", "barefoot"),
    ("trousers", "bare legs"),
    ("turtleneck", "bare shoulders"),
    ("turtleneck", "cleavage"),
    ("turtleneck", "collarbone"),
    ("turtleneck", "deep cleavage"),
];
pub const CLOTHING_LAYER_DEFS: &[ClothingLayerDef] = &[
    ClothingLayerDef { tag: "bare arms", region: "arms", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "bare hands", region: "hands", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "bare legs", region: "legs", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "bare shoulders", region: "shoulders", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "bare stomach", region: "stomach", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "barefoot", region: "feet", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "bikini", region: "chest", layer: "swimwear", rank: 1, covers: &[] },
    ClothingLayerDef { tag: "blouse", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "boots", region: "feet", layer: "footwear", rank: 4, covers: &["feet"] },
    ClothingLayerDef { tag: "bra", region: "chest", layer: "underwear", rank: 1, covers: &[] },
    ClothingLayerDef { tag: "camisole", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "cardigan", region: "torso", layer: "middle", rank: 3, covers: &[] },
    ClothingLayerDef { tag: "cleavage", region: "chest", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "coat", region: "torso", layer: "outer", rank: 4, covers: &[] },
    ClothingLayerDef { tag: "collarbone", region: "neck", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "crop top", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "deep cleavage", region: "chest", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "gloves", region: "hands", layer: "outer", rank: 4, covers: &["hands"] },
    ClothingLayerDef { tag: "high heels", region: "feet", layer: "footwear", rank: 4, covers: &["feet"] },
    ClothingLayerDef { tag: "hoodie", region: "torso", layer: "outer", rank: 4, covers: &[] },
    ClothingLayerDef { tag: "jacket", region: "torso", layer: "outer", rank: 4, covers: &[] },
    ClothingLayerDef { tag: "jeans", region: "legs", layer: "outer", rank: 4, covers: &["legs"] },
    ClothingLayerDef { tag: "kneehighs", region: "legs", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "loafers", region: "feet", layer: "footwear", rank: 4, covers: &["feet"] },
    ClothingLayerDef { tag: "long pants", region: "legs", layer: "outer", rank: 4, covers: &["legs"] },
    ClothingLayerDef { tag: "midriff", region: "stomach", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "navel", region: "stomach", layer: "skin", rank: 0, covers: &[] },
    ClothingLayerDef { tag: "overcoat", region: "torso", layer: "outer", rank: 4, covers: &[] },
    ClothingLayerDef { tag: "panties", region: "pelvis", layer: "underwear", rank: 1, covers: &[] },
    ClothingLayerDef { tag: "pants", region: "legs", layer: "outer", rank: 4, covers: &["legs"] },
    ClothingLayerDef { tag: "pantyhose", region: "legs", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "sandals", region: "feet", layer: "footwear", rank: 4, covers: &["feet"] },
    ClothingLayerDef { tag: "shirt", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "shoes", region: "feet", layer: "footwear", rank: 4, covers: &["feet"] },
    ClothingLayerDef { tag: "shorts", region: "legs", layer: "middle", rank: 3, covers: &[] },
    ClothingLayerDef { tag: "skirt", region: "legs", layer: "middle", rank: 3, covers: &[] },
    ClothingLayerDef { tag: "sneakers", region: "feet", layer: "footwear", rank: 4, covers: &["feet"] },
    ClothingLayerDef { tag: "socks", region: "feet", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "sweater", region: "torso", layer: "middle", rank: 3, covers: &[] },
    ClothingLayerDef { tag: "t-shirt", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "tank top", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "thighhighs", region: "legs", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "tube top", region: "torso", layer: "inner", rank: 2, covers: &[] },
    ClothingLayerDef { tag: "turtleneck", region: "torso", layer: "outer", rank: 4, covers: &["neck", "chest", "shoulders"] },
    ClothingLayerDef { tag: "underwear", region: "pelvis", layer: "underwear", rank: 1, covers: &[] },
];
// ---- END GENERATED ----

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Pass 1 = Scene + Body Shape, Pass 2 = Face + Clothes.
    SceneShape,
    /// Pass 1 = Scene + Skeletal Pose, Pass 2 = Character + Body Shape.
    ScenePose,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::SceneShape, Mode::ScenePose];

    /// For the dropdown: "scene_shape" or "scene_pose". Unknown values are an error.
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s {
            "scene_shape" => Ok(Mode::SceneShape),
            "scene_pose" => Ok(Mode::ScenePose),
            other => Err(format!(
                "Unknown dual_pass_mode '{}'; expected one of ('scene_shape', 'scene_pose')",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::SceneShape => "scene_shape",
            Mode::ScenePose => "scene_pose",
        }
    }

    pub fn pass1_prompt(self) -> &'static str {
        match self {
            Mode::SceneShape => MODE_A_PASS1_PROMPT,
            Mode::ScenePose => MODE_B_PASS1_PROMPT,
        }
    }

    pub fn pass2_prompt(self) -> &'static str {
        match self {
            Mode::SceneShape => MODE_A_PASS2_PROMPT,
            Mode::ScenePose => MODE_B_PASS2_PROMPT,
        }
    }

    pub fn llm_instruction(self) -> &'static str {
        match self {
            Mode::SceneShape => MODE_A_LLM_INSTRUCTION,
            Mode::ScenePose => MODE_B_LLM_INSTRUCTION,
        }
    }

    pub fn pass1_keywords(self) -> &'static [&'static str] {
        match self {
            Mode::SceneShape => MODE_A_PASS1_KEYWORDS,
            Mode::ScenePose => MODE_B_PASS1_KEYWORDS,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared text helpers
// ---------------------------------------------------------------------------
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Same as the regex `\bphrase\b` for a phrase that starts and ends with a word char.
pub fn contains_whole_phrase(hay: &str, phrase: &str) -> bool {
    if phrase.is_empty() {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = hay[start..].find(phrase) {
        let abs = start + pos;
        let end = abs + phrase.len();
        let before_ok = match hay[..abs].chars().next_back() {
            Some(c) => !is_word_char(c),
            None => true,
        };
        let after_ok = match hay[end..].chars().next() {
            Some(c) => !is_word_char(c),
            None => true,
        };
        if before_ok && after_ok {
            return true;
        }
        start = abs + hay[abs..].chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

pub fn norm_tag(tag: &str) -> String {
    tag.trim().to_lowercase().replace('_', " ")
}

/// Join comma-separated chunks, drop empties, de-duplicate (case-insensitive), keep order.
pub fn clean_csv(parts: &[&str]) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        for item in part.split(',') {
            let item = item.trim().trim_matches('.').trim();
            if item.is_empty() {
                continue;
            }
            if seen.insert(item.to_lowercase()) {
                out.push(item.to_string());
            }
        }
    }
    out.join(", ")
}

fn is_header_sep(c: char) -> bool {
    matches!(c, ':' | '-' | '\u{2013}' | '\u{2014}')
}

/// Line-level equivalent of `^\s*\d+[.)]\s*\**[^:\n\-–—]+\**\s*[:\-–—]\s*`.
/// Like the Python version, a hyphen inside a zone name also ends the header.
fn strip_header_line(line: &str) -> &str {
    let s = line.trim_start();
    let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return line;
    }
    let rest = &s[digits..];
    if !(rest.starts_with('.') || rest.starts_with(')')) {
        return line;
    }
    let after = &rest[1..];
    match after.char_indices().find(|&(_, c)| is_header_sep(c)) {
        Some((p, c)) if p > 0 => after[p + c.len_utf8()..].trim_start(),
        _ => line,
    }
}

fn strip_zone_headers(text: &str) -> String {
    text.split('\n')
        .map(strip_header_line)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("**", "")
}

pub fn flatten_zones(text: &str) -> String {
    clean_csv(&[&strip_zone_headers(text).replace('\n', ", ")])
}

/// `1girl`, `2boys`, `6+girls`, `solo`, `multiple girls`, `multiple boys`.
pub fn is_subject_tag(tag: &str) -> bool {
    if matches!(tag, "solo" | "multiple girls" | "multiple boys") {
        return true;
    }
    let digits = tag.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return false;
    }
    let mut rest = &tag[digits..];
    if let Some(r) = rest.strip_prefix('+') {
        rest = r;
    }
    matches!(rest, "girl" | "boy" | "girls" | "boys" | "other" | "others")
}

/// Final output styling: optional escaping of parentheses (not double-escaped).
pub fn style_tags(s: &str, escape_parentheses: bool) -> String {
    if !escape_parentheses {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if (c == '(' || c == ')') && prev != Some('\\') {
            out.push('\\');
        }
        out.push(c);
        prev = Some(c);
    }
    out
}

// ---------------------------------------------------------------------------
// Tag vocabulary (WD14 selected_tags.csv)
// ---------------------------------------------------------------------------
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_quotes) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            ('"', _) => in_quotes = !in_quotes,
            (',', false) => fields.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

/// Tag names from a WD14 `selected_tags.csv` (the `name` column), normalized.
pub fn load_vocab_from_csv(text: &str) -> HashSet<String> {
    let mut vocab = HashSet::new();
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(h) => parse_csv_line(h.trim_start_matches('\u{feff}')),
        None => return vocab,
    };
    let idx = match header.iter().position(|h| h.trim() == "name") {
        Some(i) => i,
        None => return vocab,
    };
    for line in lines {
        if let Some(name) = parse_csv_line(line).get(idx) {
            let n = norm_tag(name);
            if !n.is_empty() {
                vocab.insert(n);
            }
        }
    }
    vocab
}

// ---------------------------------------------------------------------------
// Prose -> WD-style tags
// ---------------------------------------------------------------------------
/// Words and hyphenated words: `[a-z0-9']+(?:-[a-z0-9']+)*` on lowercased text.
fn tokens(clause: &str) -> Vec<String> {
    let lower = clause.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let is_w = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'';
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !is_w(chars[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && is_w(chars[i]) {
            i += 1;
        }
        while i + 1 < chars.len() && chars[i] == '-' && is_w(chars[i + 1]) {
            i += 1;
            while i < chars.len() && is_w(chars[i]) {
                i += 1;
            }
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

fn variants(toks: &[String]) -> Vec<String> {
    let mut out = vec![toks.join(" ")];
    if let Some(last) = toks.last() {
        if last.chars().count() > 3 && last.ends_with('s') {
            let mut v: Vec<String> = toks[..toks.len() - 1].to_vec();
            v.push(last[..last.len() - 1].to_string());
            out.push(v.join(" "));
        }
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProseAnalysis {
    /// Tags found (negated ones excluded), in order.
    pub tags: Vec<String>,
    /// Stated plainly.
    pub strong: BTreeSet<String>,
    /// Stated only with a hedge ("appears to be", "possibly").
    pub weak: BTreeSet<String>,
    /// Explicitly denied ("no glasses", "not wearing boots").
    pub negated: Vec<String>,
}

/// Greedy longest-match of known tags inside Moondream prose, with negation and
/// hedge handling. `is_known` says whether a candidate is in the tag vocabulary.
pub fn analyze_prose(text: &str, is_known: &dyn Fn(&str) -> bool) -> ProseAnalysis {
    const MAX_N: usize = 4;
    let mut res = ProseAnalysis::default();
    let flat = strip_zone_headers(text).replace('\n', ", ");
    for clause in flat.split([',', '.', ';', ':']) {
        let toks = tokens(clause);
        let (mut neg, mut hedge) = (false, false);
        let mut i = 0;
        while i < toks.len() {
            let mut hit: Option<(String, usize)> = None;
            let max = MAX_N.min(toks.len() - i);
            'search: for n in (1..=max).rev() {
                for cand in variants(&toks[i..i + n]) {
                    if is_known(&cand) && !(n == 1 && SINGLE_STOP.contains(&cand.as_str())) {
                        hit = Some((cand, n));
                        break 'search;
                    }
                }
            }
            if let Some((tag, n)) = hit {
                if neg {
                    if !res.negated.contains(&tag) {
                        res.negated.push(tag);
                    }
                } else {
                    if !res.tags.contains(&tag) {
                        res.tags.push(tag.clone());
                    }
                    if hedge {
                        if !res.strong.contains(&tag) {
                            res.weak.insert(tag);
                        }
                    } else {
                        res.weak.remove(&tag);
                        res.strong.insert(tag);
                    }
                }
                i += n;
                continue;
            }
            let tok = toks[i].as_str();
            if NEGATORS.contains(&tok) {
                neg = true;
            } else if HEDGES.contains(&tok) {
                hedge = true;
            } else if SCOPE_END.contains(&tok)
                || (NEW_STATEMENT_VERBS.contains(&tok) && i > 0 && toks[i - 1] == "and")
            {
                neg = false;
                hedge = false;
            }
            i += 1;
        }
    }
    res
}

/// Convenience wrapper: known tags come from a plain set.
pub fn analyze_prose_with_vocab(text: &str, vocab: &HashSet<String>) -> ProseAnalysis {
    analyze_prose(text, &|c| vocab.contains(c))
}

// ---------------------------------------------------------------------------
// Reconcile: WD14 adjusts Moondream
// ---------------------------------------------------------------------------
fn split_color(tag: &str) -> Option<(&str, &str)> {
    let (first, rest) = tag.split_once(' ')?;
    if COLORS.contains(&first) {
        Some((first, rest))
    } else {
        None
    }
}

/// Returns (drop_from_moondream, drop_from_wd14).
pub fn reconcile(md: &[String], wd: &[String]) -> (HashSet<String>, HashSet<String>) {
    let md_set: HashSet<&str> = md.iter().map(|s| s.as_str()).collect();
    let wd_set: HashSet<&str> = wd.iter().map(|s| s.as_str()).collect();
    let mut drop_md: HashSet<String> = HashSet::new();
    let mut drop_wd: HashSet<String> = HashSet::new();

    let mut wd_colors: HashMap<&str, HashSet<&str>> = HashMap::new();
    for t in &wd_set {
        if let Some((color, rest)) = split_color(t) {
            wd_colors.entry(rest).or_default().insert(color);
        }
    }
    for t in &md_set {
        match split_color(t) {
            Some((color, rest)) => {
                if let Some(colors) = wd_colors.get(rest) {
                    if !colors.contains(color) {
                        drop_md.insert(t.to_string()); // WD14 wins on colors
                    }
                }
            }
            None => {
                if wd_colors.contains_key(t) {
                    drop_md.insert(t.to_string()); // plain "dress" yields to "blue dress"
                }
            }
        }
    }

    for group in EXCLUSIVE_GROUPS {
        if group.iter().any(|g| md_set.contains(g)) {
            for g in group.iter() {
                if wd_set.contains(g) && !md_set.contains(g) {
                    drop_wd.insert(g.to_string()); // Moondream wins on pose/camera
                }
            }
        }
    }
    for group in ATTRIBUTE_GROUPS {
        if group.iter().any(|g| wd_set.contains(g)) {
            for g in group.iter() {
                if md_set.contains(g) && !wd_set.contains(g) {
                    drop_md.insert(g.to_string()); // WD14 wins on attribute sizes
                }
            }
        }
    }
    (drop_md, drop_wd)
}

// ---------------------------------------------------------------------------
// Tag router (WD14 tags -> Pass 1 / Pass 2)
// ---------------------------------------------------------------------------
pub fn normalized_tags(raw_wd14_tags: &str) -> Vec<String> {
    raw_wd14_tags
        .split(',')
        .map(norm_tag)
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn sort_tag_lists(raw_wd14_tags: &str, mode: Mode) -> (Vec<String>, Vec<String>) {
    let (mut p1, mut p2) = (Vec::new(), Vec::new());
    for tag in normalized_tags(raw_wd14_tags) {
        if mode.pass1_keywords().iter().any(|k| contains_whole_phrase(&tag, k)) {
            p1.push(tag);
        } else {
            p2.push(tag);
        }
    }
    (p1, p2)
}

// ---------------------------------------------------------------------------
// Merge: convert prose to tags, then let WD14 adjust them
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Merged {
    pub pass1_tags: Vec<String>,
    pub pass2_tags: Vec<String>,
    pub subject_tags: Vec<String>,
    pub dropped_vlm: Vec<String>,
    pub dropped_wd14: Vec<String>,
    pub dropped_hedged: Vec<String>,
    pub dropped_unconfirmed: Vec<String>,
    pub dropped_layer_conflicts: Vec<(String, String)>,
    pub negated_vlm: Vec<String>,
    /// WD14 tags that matched no scene keyword and defaulted to Pass 2; grow the
    /// keyword lists from these.
    pub unrouted_wd14: Vec<String>,
    pub dropped_moondream: Vec<String>,
    pub negated_moondream: Vec<String>,
}

// ---------------------------------------------------------------------------
// Phase 4: Layer-based conflict rules
// ---------------------------------------------------------------------------
pub fn can_coexist(tag_a: &str, tag_b: &str) -> bool {
    let na = norm_tag(tag_a);
    let nb = norm_tag(tag_b);
    if na == nb {
        return true;
    }
    COEXISTING_PAIRS
        .iter()
        .any(|&(a, b)| (a == na && b == nb) || (a == nb && b == na))
}

pub fn is_occluded_by(covering_tag: &str, inner_tag: &str) -> bool {
    let nc = norm_tag(covering_tag);
    let ni = norm_tag(inner_tag);
    if nc == ni {
        return false;
    }
    if can_coexist(&nc, &ni) {
        return false;
    }
    if OCCLUSION_SUPPRESSIONS
        .iter()
        .any(|&(s, t)| s == nc && t == ni)
    {
        return true;
    }
    let rule_c = CLOTHING_LAYER_DEFS.iter().find(|d| d.tag == nc);
    let rule_i = CLOTHING_LAYER_DEFS.iter().find(|d| d.tag == ni);
    if let (Some(rc), Some(ri)) = (rule_c, rule_i) {
        if rc.covers.contains(&ri.region) && rc.rank > ri.rank {
            return true;
        }
    }
    false
}

pub fn resolve_layer_conflicts(tags: &[String]) -> (Vec<String>, Vec<(String, String)>) {
    let mut kept = Vec::new();
    let mut suppressed = Vec::new();
    for (i, tag) in tags.iter().enumerate() {
        let mut is_suppressed = false;
        for (j, other) in tags.iter().enumerate() {
            if i == j {
                continue;
            }
            if is_occluded_by(other, tag) {
                suppressed.push((tag.clone(), other.clone()));
                is_suppressed = true;
                break;
            }
        }
        if !is_suppressed {
            kept.push(tag.clone());
        }
    }
    (kept, suppressed)
}

pub fn merge_tags(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
) -> Merged {
    merge_tags_with_options(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab, false)
}

pub fn merge_tags_with_options(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
    require_two_sources: bool,
) -> Merged {
    let wd_all = normalized_tags(raw_wd14_tags);
    let (wd1, wd2) = sort_tag_lists(raw_wd14_tags, mode);
    let wd_set: HashSet<String> = wd_all.iter().cloned().collect();
    let builtin: HashSet<&str> = BUILTIN_VOCAB.iter().copied().collect();
    let is_known = |c: &str| {
        builtin.contains(c) || wd_set.contains(c) || vocab.is_some_and(|v| v.contains(c))
    };

    let a1 = analyze_prose(pass1_txt, &is_known);
    let a2 = analyze_prose(pass2_txt, &is_known);

    // Hedged VLM tags count only if WD14 agrees.
    let weak: BTreeSet<&String> = a1.weak.iter().chain(a2.weak.iter()).collect();
    let strong: BTreeSet<&String> = a1.strong.iter().chain(a2.strong.iter()).collect();
    let hedged_unconfirmed: BTreeSet<String> = weak
        .difference(&strong)
        .filter(|t| !wd_set.contains(t.as_str()))
        .map(|t| (*t).clone())
        .collect();

    let mut md1: Vec<String> = a1.tags.iter().filter(|t| !hedged_unconfirmed.contains(*t)).cloned().collect();
    let mut md2: Vec<String> = a2.tags.iter().filter(|t| !hedged_unconfirmed.contains(*t)).cloned().collect();
    let mut negated: Vec<String> = a1
        .negated
        .iter()
        .chain(a2.negated.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    negated.sort();

    // Phase 3: Two-Source Rule
    let mut exclusive_all = HashSet::new();
    for group in EXCLUSIVE_GROUPS {
        for item in *group {
            exclusive_all.insert(*item);
        }
    }
    let mut unconfirmed = BTreeSet::new();
    let md1_set: HashSet<String> = md1.iter().cloned().collect();
    let md2_set: HashSet<String> = md2.iter().cloned().collect();

    for t in md1.iter().chain(md2.iter()) {
        if !wd_set.contains(t) && !exclusive_all.contains(t.as_str()) {
            let in_both = md1_set.contains(t) && md2_set.contains(t);
            if !in_both {
                unconfirmed.insert(t.clone());
            }
        }
    }
    if require_two_sources {
        md1.retain(|t| !unconfirmed.contains(t));
        md2.retain(|t| !unconfirmed.contains(t));
    }

    let all_md: Vec<String> = md1.iter().chain(md2.iter()).cloned().collect();
    let all_wd: Vec<String> = wd1.iter().chain(wd2.iter()).cloned().collect();
    let (drop_md, drop_wd) = reconcile(&all_md, &all_wd);

    let subject: Vec<String> = wd_all.iter().filter(|t| is_subject_tag(t)).cloned().collect();
    let keep_wd = |t: &&String| !drop_wd.contains(*t) && !is_subject_tag(t);
    let wd1: Vec<String> = wd1.iter().filter(keep_wd).cloned().collect();
    let wd2: Vec<String> = wd2.iter().filter(keep_wd).cloned().collect();
    let md1: Vec<String> = md1.into_iter().filter(|t| !drop_md.contains(t)).collect();
    let md2: Vec<String> = md2.into_iter().filter(|t| !drop_md.contains(t)).collect();

    // Phase 4: Layer-based conflict resolution
    let all_combined: Vec<String> = md1.iter().chain(wd1.iter()).chain(md2.iter()).chain(wd2.iter()).cloned().collect();
    let (_, mut layer_suppressed) = resolve_layer_conflicts(&all_combined);
    layer_suppressed.sort();
    let suppressed_norms: HashSet<String> = layer_suppressed.iter().map(|(sup, _)| norm_tag(sup)).collect();
    let md1: Vec<String> = md1.into_iter().filter(|t| !suppressed_norms.contains(&norm_tag(t))).collect();
    let wd1: Vec<String> = wd1.into_iter().filter(|t| !suppressed_norms.contains(&norm_tag(t))).collect();
    let md2: Vec<String> = md2.into_iter().filter(|t| !suppressed_norms.contains(&norm_tag(t))).collect();
    let wd2: Vec<String> = wd2.into_iter().filter(|t| !suppressed_norms.contains(&norm_tag(t))).collect();

    let sorted = |s: &HashSet<String>| {
        let mut v: Vec<String> = s.iter().cloned().collect();
        v.sort();
        v
    };
    let drop_vlm_sorted = sorted(&drop_md);
    Merged {
        pass1_tags: md1.into_iter().chain(wd1).collect(),
        pass2_tags: md2.into_iter().chain(wd2.iter().cloned()).collect(),
        subject_tags: subject,
        dropped_vlm: drop_vlm_sorted.clone(),
        dropped_wd14: sorted(&drop_wd),
        dropped_hedged: hedged_unconfirmed.into_iter().collect(),
        dropped_unconfirmed: unconfirmed.into_iter().collect(),
        dropped_layer_conflicts: layer_suppressed,
        negated_vlm: negated.clone(),
        unrouted_wd14: wd2,
        dropped_moondream: drop_vlm_sorted,
        negated_moondream: negated,
    }
}

// ---------------------------------------------------------------------------
// Mechanical (no-LLM) path
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompts {
    pub scene_prompt: String,
    pub character_prompt: String,
}

pub fn process_direct(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
    cfg: &Config,
) -> Prompts {
    let m = merge_tags_with_options(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab, cfg.require_two_sources);
    let (extra1, extra2) = if cfg.keep_unmatched_prose {
        (flatten_zones(pass1_txt), flatten_zones(pass2_txt))
    } else {
        (String::new(), String::new())
    };
    let scene = clean_csv(&[&cfg.quality_prefix, &m.pass1_tags.join(", "), &extra1]);
    let character = clean_csv(&[&m.subject_tags.join(", "), &m.pass2_tags.join(", "), &extra2]);
    Prompts {
        scene_prompt: style_tags(&scene, cfg.escape_parentheses),
        character_prompt: style_tags(&character, cfg.escape_parentheses),
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON (so the module needs no serde)
// ---------------------------------------------------------------------------
pub mod json {
    #[derive(Clone, Debug, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Value>),
        Obj(Vec<(String, Value)>),
    }

    impl Value {
        /// Last matching key wins, like Python's json.loads.
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Obj(items) => items.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
    }

    pub fn escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    struct Parser {
        chars: Vec<char>,
        pos: usize,
    }

    pub fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser { chars: text.chars().collect(), pos: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.pos != p.chars.len() {
            return Err(format!("Extra data at char {}", p.pos));
        }
        Ok(v)
    }

    impl Parser {
        fn ws(&mut self) {
            while self.pos < self.chars.len() && matches!(self.chars[self.pos], ' ' | '\t' | '\n' | '\r') {
                self.pos += 1;
            }
        }

        fn peek(&self) -> Option<char> {
            self.chars.get(self.pos).copied()
        }

        fn expect(&mut self, c: char) -> Result<(), String> {
            if self.peek() == Some(c) {
                self.pos += 1;
                Ok(())
            } else {
                Err(format!("Expecting '{}' at char {}", c, self.pos))
            }
        }

        fn literal(&mut self, word: &str, v: Value) -> Result<Value, String> {
            for c in word.chars() {
                self.expect(c)?;
            }
            Ok(v)
        }

        fn value(&mut self) -> Result<Value, String> {
            match self.peek() {
                Some('{') => self.object(),
                Some('[') => self.array(),
                Some('"') => Ok(Value::Str(self.string()?)),
                Some('t') => self.literal("true", Value::Bool(true)),
                Some('f') => self.literal("false", Value::Bool(false)),
                Some('n') => self.literal("null", Value::Null),
                Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
                _ => Err(format!("Expecting value at char {}", self.pos)),
            }
        }

        fn number(&mut self) -> Result<Value, String> {
            let start = self.pos;
            while self.pos < self.chars.len()
                && matches!(self.chars[self.pos], '-' | '+' | '.' | 'e' | 'E' | '0'..='9')
            {
                self.pos += 1;
            }
            let s: String = self.chars[start..self.pos].iter().collect();
            s.parse::<f64>().map(Value::Num).map_err(|_| format!("Bad number at char {}", start))
        }

        fn hex4(&mut self) -> Result<u32, String> {
            let mut n = 0u32;
            for _ in 0..4 {
                let c = self.peek().ok_or("Truncated \\u escape")?;
                n = n * 16 + c.to_digit(16).ok_or_else(|| format!("Bad \\u escape at char {}", self.pos))?;
                self.pos += 1;
            }
            Ok(n)
        }

        fn string(&mut self) -> Result<String, String> {
            self.expect('"')?;
            let mut out = String::new();
            loop {
                let c = self.peek().ok_or("Unterminated string")?;
                self.pos += 1;
                match c {
                    '"' => return Ok(out),
                    '\\' => {
                        let e = self.peek().ok_or("Unterminated string")?;
                        self.pos += 1;
                        match e {
                            '"' => out.push('"'),
                            '\\' => out.push('\\'),
                            '/' => out.push('/'),
                            'b' => out.push('\u{8}'),
                            'f' => out.push('\u{c}'),
                            'n' => out.push('\n'),
                            'r' => out.push('\r'),
                            't' => out.push('\t'),
                            'u' => {
                                let hi = self.hex4()?;
                                let cp = if (0xD800..0xDC00).contains(&hi)
                                    && self.peek() == Some('\\')
                                    && self.chars.get(self.pos + 1) == Some(&'u')
                                {
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF)
                                } else {
                                    hi
                                };
                                out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            _ => return Err(format!("Invalid escape at char {}", self.pos)),
                        }
                    }
                    c if (c as u32) < 0x20 => return Err(format!("Control character in string at char {}", self.pos)),
                    c => out.push(c),
                }
            }
        }

        fn array(&mut self) -> Result<Value, String> {
            self.expect('[')?;
            let mut items = Vec::new();
            self.ws();
            if self.peek() == Some(']') {
                self.pos += 1;
                return Ok(Value::Arr(items));
            }
            loop {
                self.ws();
                items.push(self.value()?);
                self.ws();
                match self.peek() {
                    Some(',') => self.pos += 1,
                    Some(']') => {
                        self.pos += 1;
                        return Ok(Value::Arr(items));
                    }
                    _ => return Err(format!("Expecting ',' or ']' at char {}", self.pos)),
                }
            }
        }

        fn object(&mut self) -> Result<Value, String> {
            self.expect('{')?;
            let mut items = Vec::new();
            self.ws();
            if self.peek() == Some('}') {
                self.pos += 1;
                return Ok(Value::Obj(items));
            }
            loop {
                self.ws();
                let key = self.string()?;
                self.ws();
                self.expect(':')?;
                self.ws();
                items.push((key, self.value()?));
                self.ws();
                match self.peek() {
                    Some(',') => self.pos += 1,
                    Some('}') => {
                        self.pos += 1;
                        return Ok(Value::Obj(items));
                    }
                    _ => return Err(format!("Expecting ',' or '}}' at char {}", self.pos)),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gemma: payload, reply parsing, truncation
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
pub struct Payload {
    pub system_instruction: String,
    pub pass_1_blueprint: String,
    pub pass_1_tags: String,
    pub pass_2_blueprint: String,
    pub pass_2_tags: String,
}

pub fn build_payload(
    raw_wd14_tags: &str,
    pass1_txt: &str,
    pass2_txt: &str,
    mode: Mode,
    vocab: Option<&HashSet<String>>,
    vision: bool,
) -> Payload {
    let m = merge_tags(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab);
    let instruction = if vision {
        format!("{} {}", mode.llm_instruction(), VISION_TAIL)
    } else {
        mode.llm_instruction().to_string()
    };
    let pass2: Vec<String> = m.subject_tags.iter().chain(m.pass2_tags.iter()).cloned().collect();
    Payload {
        system_instruction: instruction,
        pass_1_blueprint: flatten_zones(pass1_txt),
        pass_1_tags: m.pass1_tags.join(", "),
        pass_2_blueprint: flatten_zones(pass2_txt),
        pass_2_tags: pass2.join(", "),
    }
}

/// Gemma has no system role: the instruction goes at the top of the user turn.
pub fn payload_to_text(p: &Payload) -> String {
    format!(
        "{}\n\nINPUT:\n{{\n  \"pass_1_blueprint\": \"{}\",\n  \"pass_1_tags\": \"{}\",\n  \"pass_2_blueprint\": \"{}\",\n  \"pass_2_tags\": \"{}\"\n}}",
        p.system_instruction,
        json::escape(&p.pass_1_blueprint),
        json::escape(&p.pass_1_tags),
        json::escape(&p.pass_2_blueprint),
        json::escape(&p.pass_2_tags),
    )
}

fn strip_fences(raw: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in raw.trim().split('\n') {
        let mut l = line;
        if l.starts_with("```") {
            l = &l[3..];
            if l.len() >= 4 && l.is_char_boundary(4) && l[..4].eq_ignore_ascii_case("json") {
                l = &l[4..];
            }
            l = l.trim_start();
        }
        if l.ends_with("```") {
            l = l[..l.len() - 3].trim_end();
        }
        lines.push(l.to_string());
    }
    lines.join("\n").trim().to_string()
}

fn py_str(v: &json::Value) -> Option<String> {
    match v {
        json::Value::Str(s) => Some(s.clone()),
        json::Value::Num(n) => Some(if n.fract() == 0.0 && n.abs() < 1e15 {
            format!("{}", *n as i64)
        } else {
            format!("{}", n)
        }),
        json::Value::Bool(b) => Some(if *b { "True".into() } else { "False".into() }),
        json::Value::Null => Some("None".into()),
        _ => None,
    }
}

fn as_str(v: Option<&json::Value>) -> String {
    match v {
        Some(json::Value::Str(s)) => s.trim().to_string(),
        Some(json::Value::Arr(items)) => items
            .iter()
            .filter_map(py_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    }
}

/// `Some((scene, character))` (either may be empty) or `None` if unparseable.
pub fn try_parse_gemma_json(raw: &str) -> Option<(String, String)> {
    let text = strip_fences(raw);
    let candidate = match (text.find('{'), text.rfind('}')) {
        (Some(s), Some(e)) if e > s => &text[s..=e],
        _ => text.as_str(),
    };
    match json::parse(candidate) {
        Ok(v @ json::Value::Obj(_)) => {
            Some((as_str(v.get("scene_prompt")), as_str(v.get("character_prompt"))))
        }
        _ => None,
    }
}

/// Reads the string value of `"key": "..."` from a reply that may be cut off or not valid JSON.
/// Returns the text up to the closing quote, or to the end of the reply when it was cut off
/// (then `.1` is false, and the last, probably unfinished, tag is dropped by the caller).
fn partial_json_string(text: &str, key: &str) -> Option<(String, bool)> {
    let at = text.find(&format!("\"{key}\""))?;
    let rest = &text[at + key.len() + 2..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some((out, true)),
            '\\' => match chars.next() {
                Some('n') | Some('t') => out.push(' '),
                Some(o) => out.push(o),
                None => {}
            },
            _ => out.push(c),
        }
    }
    Some((out, false))
}

/// Keeps what a cut-off or slightly malformed reply did write: the complete tags of
/// `character_prompt` (and `scene_prompt` when it finished). `None` when fewer than 3 tags survive.
pub fn salvage_gemma_reply(raw: &str) -> Option<(String, String)> {
    let text = strip_fences(raw);
    let complete_tags = |(value, closed): (String, bool)| -> String {
        let mut tags: Vec<&str> = value.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
        if !closed {
            tags.pop(); // the tag the reply stopped in
        }
        let mut seen = std::collections::HashSet::new();
        tags.retain(|t| seen.insert(t.to_lowercase())); // a looping reply repeats its tags
        tags.join(", ")
    };
    let character = complete_tags(partial_json_string(&text, "character_prompt")?);
    if character.split(',').filter(|t| !t.trim().is_empty()).count() < 3 {
        return None;
    }
    let scene = partial_json_string(&text, "scene_prompt")
        .filter(|(_, closed)| *closed)
        .map(complete_tags)
        .unwrap_or_default();
    Some((scene, character))
}

/// True when a reply repeats the same tags over and over (a generation loop): at least 5 tags
/// in its character list are copies of an earlier one. A larger token cap can't finish such a reply.
pub fn looks_like_a_loop(text: &str) -> bool {
    let body = match text.find("character_prompt") {
        Some(i) => &text[i..],
        None => text,
    };
    let mut seen = std::collections::HashSet::new();
    let mut repeats = 0;
    for tag in body.split(',') {
        let t = tag.trim().trim_matches(|c: char| c == '"' || c == '{' || c == '}' || c == ':').to_lowercase();
        let t = t.trim_start_matches("character_prompt").trim_matches(|c: char| c == '"' || c == ':' || c == ' ').to_string();
        if !t.is_empty() && !seen.insert(t) {
            repeats += 1;
        }
    }
    repeats >= 5
}

/// Why `try_parse_gemma_json` gave nothing usable, for the warning shown to the user.
fn gemma_failure_reason(reply: &GemmaReply) -> String {
    let raw = reply.text.trim();
    let why = if raw.is_empty() {
        "the model returned nothing".to_string()
    } else if gemma_truncated(reply) && (reply.finish_reason.is_some() || raw.contains('{')) {
        // cut off for real: the server said "length", or a JSON object was started and never closed
        // (a reply with no `{` at all is plain text, reported as "not JSON" below)
        "the reply was cut off before any tags were complete".to_string()
    } else if try_parse_gemma_json(raw).is_some() {
        "character_prompt was empty".to_string()
    } else {
        let head: String = raw.chars().take(60).collect::<String>().replace('\n', " ");
        format!("the reply was not JSON, it began: {head}")
    };
    format!("Gemma output unusable ({why}); used the mechanical merge")
}

/// Never fails; on bad output returns a best-effort fallback.
pub fn parse_gemma_json_response(raw: &str, cfg: &Config) -> Prompts {
    if let Some((scene, character)) = try_parse_gemma_json(raw) {
        return Prompts {
            scene_prompt: if scene.is_empty() { cfg.quality_prefix.clone() } else { scene },
            character_prompt: character,
        };
    }
    let text = strip_fences(raw);
    let stripped: String = text.chars().filter(|c| !matches!(c, '{' | '}' | '"')).collect();
    Prompts {
        scene_prompt: cfg.quality_prefix.clone(),
        character_prompt: clean_csv(&[&stripped.replace('\n', ", ")]),
    }
}

pub struct GemmaRequest<'a> {
    pub text: &'a str,
    pub temperature: f32,
    pub max_tokens: u32,
    /// True on the vision pass: your closure attaches the image.
    pub with_image: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GemmaReply {
    pub text: String,
    /// Ollama's `done_reason`; "length" means the reply hit the token cap.
    pub finish_reason: Option<String>,
}

/// True if the reply was cut off. With no finish reason, a reply that does not end
/// in `}` (ignoring a closing code fence) counts as cut off.
pub fn gemma_truncated(reply: &GemmaReply) -> bool {
    if let Some(r) = &reply.finish_reason {
        return matches!(
            r.trim().to_lowercase().as_str(),
            "length" | "max_tokens" | "max_length" | "max_new_tokens"
        );
    }
    let t = reply.text.trim();
    let t = t.strip_suffix("```").map_or(t, |x| x.trim_end());
    !t.ends_with('}')
}

// ---------------------------------------------------------------------------
// Ollama helpers (you supply the HTTP transport)
// ---------------------------------------------------------------------------
/// Body for `POST {host}/api/chat`. `image_b64` is the base64 image on the vision pass.
pub fn ollama_chat_request_body(
    model: &str,
    text: &str,
    temperature: f32,
    num_ctx: u32,
    num_predict: u32,
    image_b64: Option<&str>,
    json_mode: bool,
) -> String {
    ollama_chat_request_body_with(model, text, temperature, num_ctx, num_predict, image_b64, json_mode, None, None)
}

/// Like `ollama_chat_request_body`, with Ollama's `think` switch (`Some(false)` turns
/// thinking off for models that support it). `None` leaves the field out, which is
/// byte-for-byte what `ollama_chat_request_body` always produced.
#[allow(clippy::too_many_arguments)]
pub fn ollama_chat_request_body_with(
    model: &str,
    text: &str,
    temperature: f32,
    num_ctx: u32,
    num_predict: u32,
    image_b64: Option<&str>,
    json_mode: bool,
    think: Option<bool>,
    repeat_penalty: Option<f32>,
) -> String {
    let images = image_b64
        .map(|b| format!(",\"images\":[\"{}\"]", json::escape(b)))
        .unwrap_or_default();
    let format = if json_mode { ",\"format\":\"json\"" } else { "" };
    let think = think.map(|t| format!("\"think\":{t},")).unwrap_or_default();
    let repeat = repeat_penalty.map(|r| format!(",\"repeat_penalty\":{r}")).unwrap_or_default();
    format!(
        "{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"{}}}],\"stream\":false,{}\"options\":{{\"temperature\":{},\"num_ctx\":{},\"num_predict\":{}{}}}{}}}",
        json::escape(model),
        json::escape(text),
        images,
        think,
        temperature,
        num_ctx,
        num_predict,
        repeat,
        format
    )
}

/// Parse the body returned by `/api/chat` (non-streaming).
pub fn parse_ollama_chat_response(body: &str) -> Result<GemmaReply, String> {
    let v = json::parse(body)?;
    let content = match v.get("message").and_then(|m| m.get("content")) {
        Some(json::Value::Str(s)) => s.clone(),
        _ => String::new(),
    };
    let reason = match v.get("done_reason") {
        Some(json::Value::Str(s)) => Some(s.clone()),
        _ => None,
    };
    Ok(GemmaReply { text: content, finish_reason: reason })
}

/// Is `model` pulled? `tags_body` is the response of `GET {host}/api/tags`.
pub fn ollama_model_pulled(tags_body: &str, model: &str) -> Result<bool, String> {
    let v = json::parse(tags_body)?;
    let mut wanted = vec![model.to_string()];
    if !model.contains(':') {
        wanted.push(format!("{}:latest", model));
    }
    if let Some(json::Value::Arr(models)) = v.get("models") {
        for m in models {
            for key in ["name", "model"] {
                if let Some(json::Value::Str(n)) = m.get(key) {
                    if wanted.contains(n) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

pub fn ollama_not_pulled_message(model: &str) -> String {
    format!("Model {} is not pulled in Ollama; run `ollama pull {}`", model, model)
}

// ---------------------------------------------------------------------------
// VLM retry wrapper
// ---------------------------------------------------------------------------
fn valid_vlm(text: &str) -> bool {
    let t = text.trim();
    if t.chars().count() < 20 {
        return false;
    }
    let head: String = t.chars().take(80).collect::<String>().to_lowercase();
    !["sorry", "cannot", "can't", "unable to"]
        .iter()
        .any(|p| contains_whole_phrase(&head, p))
}

#[inline]
#[allow(dead_code)] // old name kept for the Python-parity API
fn valid_moondream(text: &str) -> bool {
    valid_vlm(text)
}

/// One VLM pass: retry once if it errors, is empty or looks like a refusal.
/// Gives up with "" (that pass then contributes only its WD14 tags) plus a warning.
fn run_vlm(
    vlm: &VlmFn<'_>,
    prompt: &str,
    label: &str,
    cfg: &Config,
    warnings: &mut Vec<String>,
) -> String {
    let mut last_error: Option<String> = None;
    for attempt in 0..=cfg.vlm_retries {
        let temp = cfg.vlm_temperature + if attempt > 0 { cfg.vlm_retry_temp_bump } else { 0.0 };
        match vlm(prompt, temp) {
            Ok(out) if valid_vlm(&out) => return out,
            Ok(_) => {}
            Err(e) => last_error = Some(e),
        }
    }
    let reason = last_error.unwrap_or_else(|| "empty or refusal-like output".to_string());
    warnings.push(format!("VLM {} unavailable ({}); used WD14 tags only for it", label, reason));
    String::new()
}

#[inline]
#[allow(dead_code)] // old name kept for the Python-parity API
fn run_moondream(
    vlm: &VlmFn<'_>,
    prompt: &str,
    label: &str,
    cfg: &Config,
    warnings: &mut Vec<String>,
) -> String {
    run_vlm(vlm, prompt, label, cfg, warnings)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------
/// Runs one VLM pass on the image: `(prompt, temperature) -> text`.
pub type VlmFn<'a> = dyn Fn(&str, f32) -> Result<String, String> + 'a;
pub type MoondreamFn<'a> = VlmFn<'a>;
/// Runs one Gemma request (attach the image when `with_image` is true).
pub type GemmaFn<'a> = dyn Fn(&GemmaRequest) -> Result<GemmaReply, String> + 'a;

#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    pub mode: Mode,
    pub use_llm_synthesis: bool,
    /// Only applies when `use_llm_synthesis` is true.
    pub gemma_vision_pass: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineResult {
    pub scene_prompt: String,
    pub character_prompt: String,
    /// What fell back and why. Show this in the UI: a silent fallback looks like
    /// a normal result otherwise. Empty when nothing went wrong.
    pub warnings: Vec<String>,
}

/// Run the dual-pass pipeline.
///
/// * `vlm(prompt, temperature)` runs one VLM pass on the image.
/// * `gemma` is `None` when no Gemma backend is reachable (e.g. your Ollama health
///   check failed): the mechanical merge is used and a warning recorded.
/// * Never fails: every error becomes a warning plus a fallback.
pub fn run_dual_pass(
    raw_wd14_tags: &str,
    opts: &PipelineOptions,
    cfg: &Config,
    vocab: Option<&HashSet<String>>,
    vlm: &VlmFn<'_>,
    gemma: Option<&GemmaFn<'_>>,
) -> PipelineResult {
    let mode = opts.mode;
    let mut warnings: Vec<String> = Vec::new();
    let pass1 = run_vlm(vlm, mode.pass1_prompt(), "pass 1", cfg, &mut warnings);
    let pass2 = run_vlm(vlm, mode.pass2_prompt(), "pass 2", cfg, &mut warnings);

    let mechanical = |reason: Option<String>, mut warnings: Vec<String>| {
        if let Some(r) = reason {
            warnings.push(r);
        }
        let p = process_direct(raw_wd14_tags, &pass1, &pass2, mode, vocab, cfg);
        PipelineResult { scene_prompt: p.scene_prompt, character_prompt: p.character_prompt, warnings }
    };

    if !opts.use_llm_synthesis {
        let note = opts
            .gemma_vision_pass
            .then(|| "gemma_vision_pass ignored: use_llm_synthesis is off".to_string());
        return mechanical(note, warnings);
    }
    let gemma = match gemma {
        Some(g) => g,
        None => {
            return mechanical(
                Some("No Gemma backend available; used the mechanical merge".to_string()),
                warnings,
            )
        }
    };

    let payload = build_payload(raw_wd14_tags, &pass1, &pass2, mode, vocab, opts.gemma_vision_pass);
    let text = payload_to_text(&payload);
    let call = |max_tokens: u32| {
        gemma(&GemmaRequest {
            text: &text,
            temperature: cfg.gemma_temperature,
            max_tokens,
            with_image: opts.gemma_vision_pass,
        })
    };
    let mut reply = match call(cfg.gemma_max_tokens) {
        Ok(r) => r,
        Err(e) => {
            return mechanical(Some(format!("Gemma call failed ({}); used the mechanical merge", e)), warnings)
        }
    };
    if gemma_truncated(&reply) && !looks_like_a_loop(&reply.text) {
        // One retry with a higher cap (not for a reply that is looping: more room only makes it longer). A still-truncated reply fails to parse below
        // and ends up in the "output unusable" fallback.
        reply = match call(cfg.gemma_max_tokens_retry) {
            Ok(r) => r,
            Err(e) => {
                return mechanical(Some(format!("Gemma call failed ({}); used the mechanical merge", e)), warnings)
            }
        };
    }

    match try_parse_gemma_json(&reply.text) {
        Some((scene, character)) if !character.is_empty() => PipelineResult {
            scene_prompt: style_tags(
                &if scene.is_empty() { cfg.quality_prefix.clone() } else { scene },
                cfg.escape_parentheses,
            ),
            character_prompt: style_tags(&character, cfg.escape_parentheses),
            warnings,
        },
        _ => match salvage_gemma_reply(&reply.text) {
            Some((scene, character)) => {
                warnings.push("Gemma reply was cut off or malformed; kept its complete tags".to_string());
                PipelineResult {
                    scene_prompt: style_tags(
                        &if scene.is_empty() { cfg.quality_prefix.clone() } else { scene },
                        cfg.escape_parentheses,
                    ),
                    character_prompt: style_tags(&character, cfg.escape_parentheses),
                    warnings,
                }
            }
            None => mechanical(Some(gemma_failure_reason(&reply)), warnings),
        },
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Pure Crop Tag Policy & Zone Math
// ---------------------------------------------------------------------------

pub fn is_crop_artifact_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    if CROP_ARTIFACT_TAGS.iter().any(|&c| c == norm) {
        return true;
    }
    if norm.ends_with(" focus") {
        return true;
    }
    if norm.ends_with(" out of frame") {
        return true;
    }
    if norm.starts_with("cropped ") || norm.ends_with(" cropped") {
        return true;
    }
    false
}

pub fn is_rating_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    RATING_TAGS.iter().any(|&r| r == norm)
}

pub fn is_body_size_or_count_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    BODY_SIZE_AND_COUNT_TAGS.iter().any(|&b| b == norm)
}

pub fn is_concrete_scene_object(tag: &str) -> bool {
    let norm = norm_tag(tag);
    CONCRETE_SCENE_OBJECTS.iter().any(|&obj| norm == obj || norm.contains(obj))
}

pub fn is_global_framing_tag(tag: &str) -> bool {
    let norm = norm_tag(tag);
    if is_crop_artifact_tag(&norm) {
        return true;
    }
    if SCENE_KEYWORDS.iter().any(|&k| k == norm) {
        return true;
    }
    norm.ends_with(" shot") || norm.ends_with(" view") || norm.ends_with(" angle") || norm.ends_with(" background")
}

pub fn filter_crop_tags(
    crop_tags: &[(String, f32)],
    crop_type: CropType,
    character_names: &[String],
    min_attribute_threshold: f32,
    scene_object_threshold: f32,
) -> Vec<(String, f32)> {
    let char_set: HashSet<String> = character_names.iter().map(|c| norm_tag(c)).collect();
    let mut accepted = Vec::new();

    for (tag, score) in crop_tags {
        let norm = norm_tag(tag);
        if norm.is_empty() {
            continue;
        }

        // 1. Reject crop artifacts (boundary & zoom artifacts)
        if is_crop_artifact_tag(&norm) {
            continue;
        }

        // 2. Reject ratings
        if is_rating_tag(&norm) {
            continue;
        }

        // 3. Reject character names
        if char_set.contains(&norm) {
            continue;
        }

        // 4. Reject body size & count tags
        if is_body_size_or_count_tag(&norm) {
            continue;
        }

        match crop_type {
            CropType::CharacterZone => {
                // Reject global framing/scene
                if is_global_framing_tag(&norm) {
                    continue;
                }
                // Micro-attributes require confidence >= 0.55
                if *score >= min_attribute_threshold {
                    accepted.push((tag.clone(), *score));
                }
            }
            CropType::Quadrant => {
                // Reject generic backgrounds
                if norm == "simple background"
                    || norm == "white background"
                    || norm == "grey background"
                    || norm == "monochrome"
                    || norm == "blurry background"
                {
                    continue;
                }
                if is_concrete_scene_object(&norm) {
                    if *score >= scene_object_threshold {
                        accepted.push((tag.clone(), *score));
                    }
                } else if !is_global_framing_tag(&norm) && *score >= min_attribute_threshold {
                    accepted.push((tag.clone(), *score));
                }
            }
        }
    }

    accepted
}

pub fn merge_crop_tags(
    base_tags: &[(String, f32)],
    crop_tags: &[(String, f32)],
    crop_type: CropType,
    character_names: &[String],
    min_attribute_threshold: f32,
    scene_object_threshold: f32,
) -> Vec<(String, f32)> {
    let filtered = filter_crop_tags(
        crop_tags,
        crop_type,
        character_names,
        min_attribute_threshold,
        scene_object_threshold,
    );

    let mut out: Vec<(String, f32)> = base_tags.to_vec();
    // first position of each normalized tag (one table instead of a scan of `out` per crop tag)
    let mut index: HashMap<String, usize> = HashMap::with_capacity(out.len());
    for (i, (b_tag, _)) in out.iter().enumerate() {
        index.entry(norm_tag(b_tag)).or_insert(i);
    }

    for (c_tag, c_score) in filtered {
        let norm_c = norm_tag(&c_tag);
        match index.get(&norm_c) {
            Some(&i) => out[i].1 = out[i].1.max(c_score),
            None => {
                index.insert(norm_c, out.len());
                out.push((c_tag, c_score));
            }
        }
    }
    out
}

pub fn compute_zone_boxes(
    img_w: f32,
    img_h: f32,
    box_x0: f32,
    box_y0: f32,
    box_x1: f32,
    box_y1: f32,
) -> Vec<ZoneBox> {
    if img_w <= 0.0 || img_h <= 0.0 {
        return Vec::new();
    }

    let x0 = box_x0.max(0.0).min(img_w);
    let y0 = box_y0.max(0.0).min(img_h);
    let x1 = box_x1.max(x0).min(img_w);
    let y1 = box_y1.max(y0).min(img_h);

    let w = x1 - x0;
    let h = y1 - y0;
    if w <= 1.0 || h <= 1.0 {
        return Vec::new();
    }

    let bottom_edge = y1 >= (0.98 * img_h);
    let height_ratio = h / img_h;

    let mx = w * 0.06;
    let my = h * 0.04;

    let make_box = |name: &'static str, zy0: f32, zy1: f32| -> ZoneBox {
        let cx0 = (x0 - mx).max(0.0).min(img_w - 1.0);
        let cy0 = (zy0 - my).max(0.0).min(img_h - 1.0);
        let cx1 = (x1 + mx).min(img_w).max(cx0 + 1.0);
        let cy1 = (zy1 + my).min(img_h).max(cy0 + 1.0);
        ZoneBox {
            name,
            x0: (cx0 * 100.0).round() / 100.0,
            y0: (cy0 * 100.0).round() / 100.0,
            x1: (cx1 * 100.0).round() / 100.0,
            y1: (cy1 * 100.0).round() / 100.0,
        }
    };

    let mut zones = Vec::new();
    // 1. Upper (0 - 55%): Head, Hair, Face - Always valid
    zones.push(make_box("Upper", y0, y0 + h * 0.55));

    // 2. Middle (30% - 70%): Torso, Arms, Held Items
    if height_ratio >= 0.25 {
        zones.push(make_box("Middle", y0 + h * 0.30, y0 + h * 0.70));
    }

    // 3. Lower (55% - 100%): Hips, Legs, Footwear
    // Skip lower zone if figure touches the bottom frame and height ratio < 0.72 (half-body/cowboy shot cut off)
    let is_cut_off_bottom = bottom_edge && (height_ratio < 0.72);
    if !is_cut_off_bottom && (height_ratio >= 0.40) {
        zones.push(make_box("Lower", y0 + h * 0.55, y1));
    }

    zones
}

// ---------------------------------------------------------------------------
// Tests (ports of test_prompt_condenser.py)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    const WD: &str = "1girl, solo, blue_dress, sitting, from_above, blonde_hair, cleaning, thick_thighs";
    const MD1: &str = "1. Scene & Lighting: a bright room with a couch\n2. Core Pose: she is standing, seen from below";
    const MD2: &str = "1. Hair: long red hair\n2. Clothing: a red dress and black boots";
    const OK_JSON: &str = r#"{"scene_prompt":"s","character_prompt":"c"}"#;

    fn vocab(words: &[&str]) -> HashSet<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    fn md_ok(prompt: &str, _t: f32) -> Result<String, String> {
        Ok(if prompt.contains("space") { MD1 } else { MD2 }.to_string())
    }

    fn opts(mode: Mode, llm: bool, vision: bool) -> PipelineOptions {
        PipelineOptions { mode, use_llm_synthesis: llm, gemma_vision_pass: vision }
    }

    fn reply(text: &str, reason: Option<&str>) -> Result<GemmaReply, String> {
        Ok(GemmaReply { text: text.to_string(), finish_reason: reason.map(String::from) })
    }

    #[test]
    fn router_whole_words_and_underscores() {
        // "bedhead" must NOT match the scene keyword "bed"; "bedroom" is its own keyword.
        let (p1, p2) = sort_tag_lists(
            "cleaning, depressed, sitting, thick_thighs, from_below, bedhead, bedroom",
            Mode::SceneShape,
        );
        assert_eq!(p1.join(", "), "sitting, thick thighs, from below, bedroom");
        assert!(p2.contains(&"cleaning".to_string()));
        assert!(p2.contains(&"depressed".to_string()));
        assert!(p2.contains(&"bedhead".to_string()));
    }

    #[test]
    fn mode_b_sends_body_shape_to_pass2() {
        let (p1, p2) = sort_tag_lists("sitting, thick_thighs, arms up", Mode::ScenePose);
        assert_eq!(p1.join(", "), "sitting, arms up");
        assert_eq!(p2.join(", "), "thick thighs");
    }

    #[test]
    fn unknown_mode_is_an_error_and_known_modes_round_trip() {
        assert!(Mode::parse("bad").unwrap_err().contains("Unknown dual_pass_mode 'bad'"));
        for m in Mode::ALL {
            assert_eq!(Mode::parse(m.as_str()), Ok(m));
        }
    }

    #[test]
    fn prose_to_tags_longest_match() {
        let mut v = vocab(&["red dress", "black boots", "boots"]);
        v.extend(BUILTIN_VOCAB.iter().map(|s| s.to_string()));
        let a = analyze_prose_with_vocab("wearing a red dress and black boots", &v);
        assert_eq!(a.tags, vec!["red dress", "black boots"]);
    }

    #[test]
    fn wd14_wins_colors_moondream_wins_pose() {
        let v = vocab(&["red hair", "red dress", "black boots", "boots"]);
        let m = merge_tags(WD, MD1, MD2, Mode::SceneShape, Some(&v));
        assert_eq!(m.dropped_vlm, vec!["red dress", "red hair"]);
        assert_eq!(m.dropped_moondream, vec!["red dress", "red hair"]);
        assert_eq!(m.dropped_wd14, vec!["from above", "sitting"]);
    }

    #[test]
    fn no_llm_output_is_pure_tags() {
        let v = vocab(&["red hair", "red dress", "black boots", "boots"]);
        let out = process_direct(WD, MD1, MD2, Mode::SceneShape, Some(&v), &Config::default());
        assert_eq!(out.scene_prompt, "masterpiece, best quality, couch, standing, from below, thick thighs");
        assert_eq!(out.character_prompt, "1girl, solo, black boots, blue dress, blonde hair, cleaning");
    }

    #[test]
    fn parser_handles_preamble_lists_and_bad_json() {
        let cfg = Config::default();
        let good = parse_gemma_json_response(
            "Here:\n```json\n{\"scene_prompt\":\"a\",\"character_prompt\":[\"b\",\"c\"]}\n```",
            &cfg,
        );
        assert_eq!(good, Prompts { scene_prompt: "a".into(), character_prompt: "b, c".into() });
        assert_eq!(parse_gemma_json_response("[1,2]", &cfg).scene_prompt, cfg.quality_prefix);
        assert_eq!(parse_gemma_json_response("not json", &cfg).character_prompt, "not json");
    }

    #[test]
    fn llm_path_retries_once_on_truncation_and_requests_image() {
        let calls: RefCell<Vec<(u32, bool)>> = RefCell::new(Vec::new());
        let gemma = |r: &GemmaRequest| {
            calls.borrow_mut().push((r.max_tokens, r.with_image));
            if calls.borrow().len() == 1 {
                reply("{\"scene_prompt\":\"s\",\"char", Some("length"))
            } else {
                reply(OK_JSON, Some("stop"))
            }
        };
        let out = run_dual_pass(WD, &opts(Mode::ScenePose, true, true), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!((out.scene_prompt.as_str(), out.character_prompt.as_str()), ("s", "c"));
        assert!(out.warnings.is_empty());
        assert_eq!(*calls.borrow(), vec![(GEMMA_MAX_TOKENS, true), (GEMMA_MAX_TOKENS_RETRY, true)]);
    }

    #[test]
    fn no_image_requested_without_vision_pass() {
        let seen = Cell::new(true);
        let gemma = |r: &GemmaRequest| {
            seen.set(r.with_image);
            reply(OK_JSON, None)
        };
        run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert!(!seen.get());
    }

    #[test]
    fn vision_instruction_only_added_on_vision_pass() {
        let with = build_payload(WD, MD1, MD2, Mode::SceneShape, None, true);
        let without = build_payload(WD, MD1, MD2, Mode::SceneShape, None, false);
        assert!(with.system_instruction.contains("final check"));
        assert!(!without.system_instruction.contains("final check"));
    }

    #[test]
    fn defaults() {
        assert_eq!(GEMMA_OLLAMA_MODEL, "huihui_ai/gemma-4-abliterated:12b");
        assert_eq!(GEMMA_CONTEXT_WINDOW, 16384);
        assert_eq!(GEMMA_MAX_TOKENS, 768);
        assert_eq!(GEMMA_MAX_TOKENS_RETRY, 1024);
    }

    #[test]
    fn moondream_empty_pass_retries_then_uses_wd14_only() {
        let temps: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        let empty = |_p: &str, t: f32| {
            temps.borrow_mut().push(t);
            Ok(String::new())
        };
        let out = run_dual_pass(
            "1girl, solo, sitting, blue_dress",
            &opts(Mode::SceneShape, false, false),
            &Config::default(),
            None,
            &empty,
            None,
        );
        assert_eq!(temps.borrow().len(), 4); // 2 passes x (1 try + 1 retry)
        assert!((temps.borrow()[1] - (VLM_TEMPERATURE + VLM_RETRY_TEMP_BUMP)).abs() < 1e-6);
        assert_eq!(out.character_prompt, "1girl, solo, blue dress");
        assert!(out.scene_prompt.contains("sitting"));
        assert_eq!(out.warnings.len(), 2);
    }

    #[test]
    fn moondream_refusal_retries_and_recovers() {
        let n = Cell::new(0);
        let flaky = |_p: &str, _t: f32| {
            n.set(n.get() + 1);
            Ok(match n.get() {
                1 => "I'm sorry, I cannot help with that.".to_string(),
                2 => MD1.to_string(),
                _ => MD2.to_string(),
            })
        };
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, false, false), &Config::default(), None, &flaky, None);
        assert!(out.scene_prompt.contains("standing"));
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn attribute_groups_wd14_wins() {
        let m = merge_tags("short_hair, 1girl", "1. Hair: she has long hair and a smile", "", Mode::SceneShape, None);
        assert!(m.dropped_vlm.contains(&"long hair".to_string()));
        assert!(m.dropped_moondream.contains(&"long hair".to_string()));
        let all: Vec<&String> = m.pass1_tags.iter().chain(m.pass2_tags.iter()).collect();
        assert!(all.contains(&&"short hair".to_string()));
    }

    #[test]
    fn unusable_gemma_output_falls_back_to_mechanical_merge() {
        let expected = process_direct(WD, MD1, MD2, Mode::SceneShape, None, &Config::default());
        let gemma = |_r: &GemmaRequest| reply("total garbage", None);
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert_eq!(out.character_prompt, expected.character_prompt);
        assert!(!out.warnings.is_empty());
    }

    #[test]
    fn cut_off_gemma_reply_keeps_its_complete_tags() {
        // cut inside a tag of character_prompt: scene is complete, the unfinished tag is dropped
        let cut = r#"{"scene_prompt":"indoors, bed","character_prompt":"1girl, solo, black hair, long ha"#;
        let (scene, ch) = salvage_gemma_reply(cut).expect("salvaged");
        assert_eq!(scene, "indoors, bed");
        assert_eq!(ch, "1girl, solo, black hair");
        // fenced, escaped quotes, newline in a value
        let fenced = "```json\n{\"scene_prompt\": \"x\", \"character_prompt\": \"1girl, \\\"miu miu\\\", a\\nb, c\"\n";
        assert!(salvage_gemma_reply(fenced).is_some());
        // too little to keep
        assert!(salvage_gemma_reply(r#"{"character_prompt":"1girl, so"#).is_none());
        assert!(salvage_gemma_reply("total garbage").is_none());
        // through the pipeline: the model's tags are used and a warning says so
        let gemma = |_r: &GemmaRequest| reply(cut, Some("length"));
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert!(out.character_prompt.starts_with("1girl, solo, black hair"), "{}", out.character_prompt);
        assert!(out.warnings.iter().any(|w| w.contains("kept its complete tags")), "{:?}", out.warnings);
    }

    #[test]
    fn unusable_reply_warning_says_why() {
        let w = |text: &str, fin: Option<&str>| {
            let gemma = |_r: &GemmaRequest| reply(text, fin);
            run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma)).warnings.join(" | ")
        };
        assert!(w("total garbage", None).contains("not JSON, it began: total garbage"));
        assert!(w(r#"{"scene_prompt":"x","character_prompt":""}"#, None).contains("character_prompt was empty"));
        assert!(w("", None).contains("returned nothing"));
    }

    #[test]
    fn looping_replies_are_detected_and_deduplicated() {
        let looping = r#"{"scene_prompt":"a","character_prompt":"1girl, long hair, short hair, long hair, short hair, long hair, short hair, long hair, short hair, long hair, short hair, long hair, short hair, long ha"#;
        assert!(looks_like_a_loop(looping));
        assert!(!looks_like_a_loop(r#"{"scene_prompt":"a","character_prompt":"1girl, long hair, blue eyes, smile, hat"}"#));
        // a normal list with a couple of repeated words is not a loop
        assert!(!looks_like_a_loop(r#"{"character_prompt":"1girl, hair, eyes, hair ribbon, eyes closed"}"#));
        // the salvaged tags keep each one once
        let (_, ch) = salvage_gemma_reply(looping).expect("salvaged");
        assert_eq!(ch, "1girl, long hair, short hair");
        // a looping, cut-off reply is not retried with a larger cap
        let calls = std::cell::Cell::new(0);
        let gemma = |_r: &GemmaRequest| { calls.set(calls.get() + 1); reply(looping, Some("length")) };
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!(calls.get(), 1);
        assert_eq!(out.character_prompt, "1girl, long hair, short hair");
    }

    #[test]
    fn gemma_empty_character_prompt_falls_back() {
        let expected = process_direct(WD, MD1, MD2, Mode::ScenePose, None, &Config::default());
        let gemma = |_r: &GemmaRequest| reply(r#"{"scene_prompt":"x","character_prompt":""}"#, None);
        let out = run_dual_pass(WD, &opts(Mode::ScenePose, true, false), &Config::default(), None, &md_ok, Some(&gemma));
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert_eq!(out.character_prompt, expected.character_prompt);
        assert!(!out.warnings.is_empty());
    }

    #[test]
    fn parenthesis_escaping_option() {
        let cfg = Config { escape_parentheses: true, ..Config::default() };
        let out = process_direct("kafuu chino (gochiusa), 1girl", MD1, MD2, Mode::SceneShape, None, &cfg);
        assert!(out.character_prompt.contains("kafuu chino \\(gochiusa\\)"));
        assert_eq!(out.character_prompt.matches("\\(").count(), 1); // not double-escaped
        assert_eq!(style_tags("a \\(b\\) (c)", true), "a \\(b\\) \\(c\\)");
    }

    #[test]
    fn scene_keywords_route_common_background_tags() {
        let (p1, p2) = sort_tag_lists(
            "simple_background, white_background, night, cowboy_shot, 1girl, blue_dress",
            Mode::SceneShape,
        );
        assert_eq!(p1.join(", "), "simple background, white background, night, cowboy shot");
        assert_eq!(p2.join(", "), "1girl, blue dress");
    }

    #[test]
    fn negation_drops_tag_and_scope_ends_at_but() {
        let v = vocab(&["glasses", "red hat", "hat", "boots"]);
        let a = analyze_prose_with_vocab("She has no glasses but a red hat, not wearing boots", &v);
        assert!(a.negated.contains(&"glasses".to_string()) && a.negated.contains(&"boots".to_string()));
        assert_eq!(a.tags, vec!["red hat"]);
    }

    #[test]
    fn bare_is_not_a_negator() {
        let v = vocab(&["bare shoulders"]);
        assert_eq!(analyze_prose_with_vocab("bare shoulders", &v).tags, vec!["bare shoulders"]);
    }

    #[test]
    fn hedged_tag_needs_wd14_confirmation() {
        let v = vocab(&["glasses", "smile"]);
        let md1 = "1. Face: she appears to be wearing glasses and has a smile";
        let kept = merge_tags("1girl, glasses", md1, "", Mode::SceneShape, Some(&v));
        let dropped = merge_tags("1girl", md1, "", Mode::SceneShape, Some(&v));
        let all = |m: &Merged| m.pass1_tags.iter().chain(m.pass2_tags.iter()).cloned().collect::<Vec<_>>();
        assert!(all(&kept).contains(&"glasses".to_string()));
        assert!(!all(&dropped).contains(&"glasses".to_string()));
        assert!(dropped.dropped_hedged.contains(&"glasses".to_string()));
        assert!(all(&dropped).contains(&"smile".to_string()));
    }

    #[test]
    fn hedge_and_negation_scope_ends_at_new_statement() {
        let v = vocab(&["glasses", "hat", "smile"]);
        let a = analyze_prose_with_vocab("she appears to be wearing glasses and a hat", &v);
        assert_eq!(a.weak, ["glasses", "hat"].iter().map(|s| s.to_string()).collect());
        assert!(a.strong.is_empty());
        let b = analyze_prose_with_vocab("she appears to be wearing glasses and has a smile", &v);
        assert_eq!(b.weak, ["glasses"].iter().map(|s| s.to_string()).collect());
        assert_eq!(b.strong, ["smile"].iter().map(|s| s.to_string()).collect());
        let c = analyze_prose_with_vocab("she has no glasses and has a hat", &v);
        assert_eq!(c.negated, vec!["glasses"]);
        assert_eq!(c.tags, vec!["hat"]);
    }

    #[test]
    fn gemma_error_falls_back_with_warning() {
        let expected = process_direct(WD, MD1, MD2, Mode::SceneShape, None, &Config::default());
        let boom = |_r: &GemmaRequest| Err("connection refused".to_string());
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, Some(&boom));
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert_eq!(out.character_prompt, expected.character_prompt);
        assert!(out.warnings.iter().any(|w| w.contains("connection refused")));
    }

    #[test]
    fn moondream_error_becomes_wd14_only_pass_with_warning() {
        let boom = |_p: &str, _t: f32| Err("model crashed".to_string());
        let out = run_dual_pass(
            "1girl, solo, sitting, blue_dress",
            &opts(Mode::SceneShape, false, false),
            &Config::default(),
            None,
            &boom,
            None,
        );
        assert_eq!(out.character_prompt, "1girl, solo, blue dress");
        assert_eq!(out.warnings.iter().filter(|w| w.contains("model crashed")).count(), 2);
    }

    #[test]
    fn missing_gemma_backend_uses_mechanical_merge_with_warning() {
        let expected = process_direct(WD, MD1, MD2, Mode::SceneShape, None, &Config::default());
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, true, false), &Config::default(), None, &md_ok, None);
        assert_eq!(out.scene_prompt, expected.scene_prompt);
        assert!(out.warnings.iter().any(|w| w.contains("No Gemma backend")));
    }

    #[test]
    fn vision_toggle_without_llm_is_flagged() {
        let out = run_dual_pass(WD, &opts(Mode::SceneShape, false, true), &Config::default(), None, &md_ok, None);
        assert!(out.warnings.iter().any(|w| w.contains("gemma_vision_pass ignored")));
    }

    #[test]
    fn ollama_helpers() {
        let tags = r#"{"models":[{"name":"other:7b"},{"name":"huihui_ai/gemma-4-abliterated:12b"}]}"#;
        assert_eq!(ollama_model_pulled(tags, GEMMA_OLLAMA_MODEL), Ok(true));
        assert_eq!(ollama_model_pulled(tags, "huihui_ai/gemma-4-abliterated:26b"), Ok(false));
        assert_eq!(ollama_model_pulled(r#"{"models":[{"name":"x:latest"}]}"#, "x"), Ok(true));
        assert!(ollama_model_pulled("nope", "x").is_err());
        assert!(ollama_not_pulled_message("m:1").contains("ollama pull m:1"));

        let body = ollama_chat_request_body("m", "hi \"there\"\n", 0.35, 131072, 768, Some("QUJD"), true);
        let v = json::parse(&body).expect("request body must be valid JSON");
        assert_eq!(v.get("stream"), Some(&json::Value::Bool(false)));
        assert_eq!(v.get("format"), Some(&json::Value::Str("json".into())));
        let opts_v = v.get("options").unwrap();
        assert_eq!(opts_v.get("num_ctx"), Some(&json::Value::Num(131072.0)));
        assert_eq!(opts_v.get("num_predict"), Some(&json::Value::Num(768.0)));
        match v.get("messages") {
            Some(json::Value::Arr(m)) => {
                assert_eq!(m[0].get("content"), Some(&json::Value::Str("hi \"there\"\n".into())));
                assert_eq!(m[0].get("images"), Some(&json::Value::Arr(vec![json::Value::Str("QUJD".into())])));
            }
            other => panic!("bad messages: {:?}", other),
        }
        // `think` is a real JSON field, valid in any position; without it nothing changes
        let off = ollama_chat_request_body_with("m", "t", 0.1, 8192, 64, None, true, Some(false), Some(1.1));
        let v = json::parse(&off).expect("valid JSON");
        assert!(matches!(v.get("think"), Some(json::Value::Bool(false))));
        assert_eq!(
            ollama_chat_request_body_with("m", "t", 0.1, 8192, 64, None, true, None, None),
            ollama_chat_request_body("m", "t", 0.1, 8192, 64, None, true)
        );
        let no_img = ollama_chat_request_body("m", "t", 0.1, 8192, 64, None, false);
        assert!(!no_img.contains("images") && !no_img.contains("\"format\""));

        let r = parse_ollama_chat_response(r#"{"message":{"content":"abc"},"done_reason":"length"}"#).unwrap();
        assert_eq!(r, GemmaReply { text: "abc".into(), finish_reason: Some("length".into()) });
        assert!(gemma_truncated(&r));
    }

    #[test]
    fn truncation_heuristic_without_finish_reason() {
        let mk = |t: &str| GemmaReply { text: t.into(), finish_reason: None };
        assert!(gemma_truncated(&mk("{\"scene_prompt\":\"s\",\"char")));
        assert!(!gemma_truncated(&mk(OK_JSON)));
        assert!(!gemma_truncated(&mk(&format!("```json\n{}\n```", OK_JSON))));
    }

    #[test]
    fn payload_text_is_valid_json_after_the_instruction() {
        let p = build_payload(WD, MD1, MD2, Mode::SceneShape, None, false);
        let text = payload_to_text(&p);
        let (head, body) = text.split_once("\n\nINPUT:\n").unwrap();
        assert_eq!(head, Mode::SceneShape.llm_instruction());
        let v = json::parse(body).expect("INPUT must be valid JSON");
        assert_eq!(v.get("pass_1_tags"), Some(&json::Value::Str(p.pass_1_tags.clone())));
        assert_eq!(v.get("pass_2_tags"), Some(&json::Value::Str(p.pass_2_tags.clone())));
    }

    #[test]
    fn json_parser_basics() {
        use json::Value::*;
        assert_eq!(json::parse(r#"{"a":[1,true,null,"xé😀"]}"#).unwrap().get("a"),
            Some(&Arr(vec![Num(1.0), Bool(true), Null, Str("x\u{e9}\u{1F600}".into())])));
        assert!(json::parse("{\"a\":1} trailing").is_err());
        assert!(json::parse("{\"a\":").is_err());
        assert_eq!(json::parse(r#"{"a":1,"a":2}"#).unwrap().get("a"), Some(&Num(2.0)));
    }

    #[test]
    fn vocab_csv_loading() {
        let csv = "tag_id,name,category,count\n1,1girl,0,100\n2,\"weird,name\",0,5\n3,long_hair,0,9\n";
        let v = load_vocab_from_csv(csv);
        assert!(v.contains("1girl") && v.contains("long hair") && v.contains("weird,name"));
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn zone_header_stripping() {
        assert_eq!(flatten_zones("1. **Scene & Lighting**: soft light\n2) Camera - low angle"), "soft light, low angle");
        assert_eq!(flatten_zones("1. A: x\n\n2. B: y\nno header line"), "x, y, no header line");
    }

    #[test]
    fn subject_tags() {
        for t in ["1girl", "2boys", "6+girls", "solo", "multiple girls", "1other"] {
            assert!(is_subject_tag(t), "{}", t);
        }
        for t in ["girl", "1 girl", "solo focus", "1girls x", ""] {
            assert!(!is_subject_tag(t), "{}", t);
        }
    }

    #[test]
    fn crop_tag_policy_rejects_artifacts_ratings_sizes_counts() {
        let crop = vec![
            ("close-up".to_string(), 0.95),
            ("portrait".to_string(), 0.90),
            ("solo focus".to_string(), 0.88),
            ("breast focus".to_string(), 0.85),
            ("head out of frame".to_string(), 0.80),
            ("cropped legs".to_string(), 0.75),
            ("out of frame".to_string(), 0.90),
            ("feet out of frame".to_string(), 0.65),
            ("partially submerged".to_string(), 0.60),
            ("general".to_string(), 0.99),
            ("sensitive".to_string(), 0.95),
            ("1girl".to_string(), 0.98),
            ("large breasts".to_string(), 0.85),
            ("pyra (xenoblade)".to_string(), 0.95),
            ("silver brooch".to_string(), 0.55),
            ("hair ornament".to_string(), 0.54), // below 0.55
        ];
        let char_names = vec!["pyra (xenoblade)".to_string()];
        let filtered = filter_crop_tags(&crop, CropType::CharacterZone, &char_names, 0.55, 0.40);
        let names: Vec<String> = filtered.into_iter().map(|(t, _)| t).collect();
        assert_eq!(names, vec!["silver brooch"]);
    }

    #[test]
    fn crop_tag_policy_quadrant_concrete_objects() {
        let quad = vec![
            ("potted plant".to_string(), 0.45),
            ("bookshelf".to_string(), 0.40),
            ("simple background".to_string(), 0.90),
            ("white background".to_string(), 0.85),
            ("solo focus".to_string(), 0.70),
            ("close-up".to_string(), 0.75),
            ("wooden chair".to_string(), 0.50),
        ];
        let filtered = filter_crop_tags(&quad, CropType::Quadrant, &[], 0.55, 0.40);
        let names: Vec<String> = filtered.into_iter().map(|(t, _)| t).collect();
        assert!(names.contains(&"potted plant".to_string()));
        assert!(names.contains(&"bookshelf".to_string()));
        assert!(names.contains(&"wooden chair".to_string()));
        assert!(!names.contains(&"simple background".to_string()));
        assert!(!names.contains(&"white background".to_string()));
        assert!(!names.contains(&"solo focus".to_string()));
        assert!(!names.contains(&"close-up".to_string()));
    }

    #[test]
    fn merge_crop_tags_takes_max_score() {
        let base = vec![("blue dress".to_string(), 0.60), ("black boots".to_string(), 0.70)];
        let crop = vec![("blue dress".to_string(), 0.85), ("silver necklace".to_string(), 0.80)];
        let merged = merge_crop_tags(&base, &crop, CropType::CharacterZone, &[], 0.55, 0.40);
        let m_map: HashMap<String, f32> = merged.into_iter().collect();
        assert_eq!(m_map.get("blue dress"), Some(&0.85));
        assert_eq!(m_map.get("black boots"), Some(&0.70));
        assert_eq!(m_map.get("silver necklace"), Some(&0.80));
    }

    #[test]
    fn pure_zone_boxes_geometry() {
        // Full body: 1000x1000 image, box height = 850 (ratio = 0.85)
        let boxes = compute_zone_boxes(1000.0, 1000.0, 200.0, 100.0, 500.0, 950.0);
        let names: Vec<&str> = boxes.iter().map(|b| b.name).collect();
        assert_eq!(names, vec!["Upper", "Middle", "Lower"]);

        // Half body touching bottom: box height = 500 (ratio = 0.50, touches bottom >= 980)
        let half = compute_zone_boxes(1000.0, 1000.0, 200.0, 500.0, 500.0, 1000.0);
        let half_names: Vec<&str> = half.iter().map(|b| b.name).collect();
        assert_eq!(half_names, vec!["Upper", "Middle"]);

        // Head/bust: height = 150 (ratio = 0.15)
        let bust = compute_zone_boxes(1000.0, 1000.0, 300.0, 100.0, 500.0, 250.0);
        let bust_names: Vec<&str> = bust.iter().map(|b| b.name).collect();
        assert_eq!(bust_names, vec!["Upper"]);
    }

    #[test]
    fn two_source_rule_drops_single_mention_vlm_tags_when_enabled() {
        let wd = "1girl, solo, blue_dress";
        let pass1 = "she is standing in a bedroom with a desk.";
        let pass2 = "she has blonde hair and glasses and a blue dress.";
        let vocab: HashSet<String> = ["glasses", "desk", "standing", "bedroom", "blonde hair", "blue dress"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Default: glasses logged in dropped_unconfirmed but kept
        let m_def = merge_tags_with_options(wd, pass1, pass2, Mode::SceneShape, Some(&vocab), false);
        assert!(m_def.pass2_tags.contains(&"glasses".to_string()));
        assert!(m_def.dropped_unconfirmed.contains(&"glasses".to_string()));

        // Strict (require_two_sources=true): glasses pruned
        let m_strict = merge_tags_with_options(wd, pass1, pass2, Mode::SceneShape, Some(&vocab), true);
        assert!(!m_strict.pass2_tags.contains(&"glasses".to_string()));
        assert!(m_strict.dropped_unconfirmed.contains(&"glasses".to_string()));
        assert!(!m_strict.pass2_tags.contains(&"blonde hair".to_string()));
        assert!(m_strict.pass2_tags.contains(&"blue dress".to_string()));
        assert!(m_strict.pass1_tags.contains(&"standing".to_string()));
    }

    #[test]
    fn two_source_rule_retains_when_second_pass_agrees() {
        let wd = "1girl, solo";
        let pass1 = "she is wearing a butterfly hairpin and a red ribbon.";
        let pass2 = "she has a butterfly hairpin in her hair.";
        let vocab: HashSet<String> = ["butterfly hairpin", "red ribbon"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let m = merge_tags_with_options(wd, pass1, pass2, Mode::SceneShape, Some(&vocab), true);
        assert!(m.pass2_tags.contains(&"butterfly hairpin".to_string()));
        assert!(!m.dropped_unconfirmed.contains(&"butterfly hairpin".to_string()));
        assert!(!m.pass2_tags.contains(&"red ribbon".to_string()));
        assert!(m.dropped_unconfirmed.contains(&"red ribbon".to_string()));
    }

    #[test]
    fn layer_conflict_coexist_and_suppression() {
        // 1. can_coexist checks
        assert!(can_coexist("tube top", "cleavage"));
        assert!(can_coexist("boots", "thighhighs"));
        assert!(can_coexist("shoes", "socks"));
        assert!(!can_coexist("turtleneck", "cleavage"));
        assert!(!can_coexist("shoes", "barefoot"));

        // 2. is_occluded_by checks
        assert!(is_occluded_by("turtleneck", "cleavage"));
        assert!(!is_occluded_by("tube top", "cleavage"));
        assert!(!is_occluded_by("boots", "thighhighs"));
        assert!(is_occluded_by("shoes", "barefoot"));
        assert!(is_occluded_by("long pants", "bare legs"));

        // 3. resolve_layer_conflicts
        let tags_tt: Vec<String> = ["1girl", "tube top", "cleavage", "boots", "thighhighs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (kept_tt, sup_tt) = resolve_layer_conflicts(&tags_tt);
        assert!(kept_tt.contains(&"tube top".to_string()));
        assert!(kept_tt.contains(&"cleavage".to_string()));
        assert!(kept_tt.contains(&"boots".to_string()));
        assert!(kept_tt.contains(&"thighhighs".to_string()));
        assert_eq!(sup_tt.len(), 0);

        let tags_tn: Vec<String> = ["1girl", "turtleneck", "cleavage", "boots", "thighhighs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (kept_tn, sup_tn) = resolve_layer_conflicts(&tags_tn);
        assert!(kept_tn.contains(&"turtleneck".to_string()));
        assert!(!kept_tn.contains(&"cleavage".to_string()));
        assert!(kept_tn.contains(&"boots".to_string()));
        assert!(kept_tn.contains(&"thighhighs".to_string()));
        assert!(sup_tn.contains(&("cleavage".to_string(), "turtleneck".to_string())));

        // 4. merge_tags integration
        let m_tt = merge_tags("1girl, tube_top, cleavage, boots, thighhighs", "", "", Mode::SceneShape, None);
        let all_tt: Vec<String> = m_tt.pass1_tags.into_iter().chain(m_tt.pass2_tags).collect();
        assert!(all_tt.contains(&"tube top".to_string()));
        assert!(all_tt.contains(&"cleavage".to_string()));
        assert!(all_tt.contains(&"boots".to_string()));
        assert!(all_tt.contains(&"thighhighs".to_string()));
        assert_eq!(m_tt.dropped_layer_conflicts.len(), 0);

        let m_tn = merge_tags("1girl, turtleneck, cleavage, boots, thighhighs", "", "", Mode::SceneShape, None);
        let all_tn: Vec<String> = m_tn.pass1_tags.into_iter().chain(m_tn.pass2_tags).collect();
        assert!(all_tn.contains(&"turtleneck".to_string()));
        assert!(!all_tn.contains(&"cleavage".to_string()));
        assert!(all_tn.contains(&"boots".to_string()));
        assert!(all_tn.contains(&"thighhighs".to_string()));
        assert!(m_tn.dropped_layer_conflicts.contains(&("cleavage".to_string(), "turtleneck".to_string())));
    }
}
