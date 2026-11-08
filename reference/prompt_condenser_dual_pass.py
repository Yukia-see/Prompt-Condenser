"""
Prompt-Condenser: dual-pass VLM (Qwen-VL / Moondream) + WD14 pipeline with a selectable split mode.

Flow:
  1. WD14 tags the image; VLM describes it in two passes (prose).
  2. VLM prose is converted into WD-style (Danbooru) tags.
  3. WD14 tags adjust the VLM tags (reconcile): WD14 wins on attributes
     (colors on clothing/hair/eyes), VLM wins on pose and camera angle.
  4. Result is one WD-style prompt split into scene_prompt / character_prompt.
  5. Optionally, Gemma polishes the merged tags (use_llm_synthesis).

Modes (dual_pass_mode dropdown):
  "scene_shape": Pass 1 = Scene + Body Shape,  Pass 2 = Face + Clothes
  "scene_pose" : Pass 1 = Scene + Skeletal Pose, Pass 2 = Character + Body Shape

Toggles exposed to the main loop:
  enable_dual_pass   (bool)
  dual_pass_mode     (dropdown: DUAL_PASS_MODES)
  use_llm_synthesis  (bool)
  gemma_vision_pass  (bool)  only applies when use_llm_synthesis is True;
                             Gemma also receives the image and checks the tags
                             against it. Needs a multimodal Gemma model.
"""

import base64
import csv
import io
import json
import logging
import os
import re
import urllib.request
from typing import Callable, Iterable, Optional

log = logging.getLogger("prompt_condenser")

# ---------------------------------------------------------------------------
# PHASE 0: CONFIGURATION CONSTANTS
# Temperatures are passed into the model calls in run_pipeline().
# WD14 thresholds are consumed by the upstream WD14 tagger node
# (pass them to it when you call it; this file only receives the tag string).
# ---------------------------------------------------------------------------
VLM_TEMPERATURE = 0.05
MOONDREAM_TEMPERATURE = VLM_TEMPERATURE
GEMMA_TEMPERATURE = 0.10
GEMMA_OLLAMA_MODEL = "huihui_ai/gemma-4-abliterated:12b"
GEMMA_CONTEXT_WINDOW = 131072   # 128K: the context the model page lists for the 12b tag
GEMMA_MAX_TOKENS = 768          # normal output cap
GEMMA_MAX_TOKENS_RETRY = 1024   # one retry with this cap if the first reply was cut off
VLM_RETRIES = 1                 # re-ask once if a pass comes back empty or looks like a refusal
MOONDREAM_RETRIES = VLM_RETRIES
VLM_RETRY_TEMP_BUMP = 0.2       # retry slightly warmer so it does not repeat the same failure
MOONDREAM_RETRY_TEMP_BUMP = VLM_RETRY_TEMP_BUMP
TAG_ESCAPE_PARENTHESES = False  # True: write "(" as "\(" (A1111/ComfyUI prompt syntax)
WD14_GENERAL_THRESHOLD = 0.22
WD14_CHARACTER_THRESHOLD = 0.85

QUALITY_PREFIX = "masterpiece, best quality"  # configurable; set "" to disable

ZONE_FORMAT_HINT = (
    " Format each zone on its own line exactly as 'N. Zone name: text'. "
    "Plain text only, no markdown."
)

# ---------------------------------------------------------------------------
# PHASE 1: VLM PROMPTS + PER-MODE CONFIG
# ---------------------------------------------------------------------------
VLM_MODE_A_PASS_1 = (
    "Analyze physical space and body proportions and silhouette as visible. "
    "Ignore clothing and face. 6 zones: 1. Scene & Lighting 2. Camera & Depth "
    "3. Core Pose 4. Upper Body Shape 5. Lower Body Shape 6. Physical Anchors"
    + ZONE_FORMAT_HINT
)
MOONDREAM_MODE_A_PASS_1 = VLM_MODE_A_PASS_1

VLM_MODE_A_PASS_2 = (
    "Analyze facial identity and wardrobe. Ignore body weight and environment. "
    "6 zones: 1. Head & Hair 2. Face & Eyes 3. Torso Clothing 4. Leg Clothing "
    "5. Footwear 6. Accessories & Props"
    + ZONE_FORMAT_HINT
)
MOONDREAM_MODE_A_PASS_2 = VLM_MODE_A_PASS_2

VLM_MODE_B_PASS_1 = (
    "Analyze physical space and skeletal pose. Ignore clothing, face, and body "
    "thickness. 6 zones: 1. Scene & Lighting 2. Camera & Depth 3. Center of "
    "Gravity & Stance 4. Torso & Head Alignment 5. Limb Positioning "
    "6. Environmental Anchors"
    + ZONE_FORMAT_HINT
)
MOONDREAM_MODE_B_PASS_1 = VLM_MODE_B_PASS_1

VLM_MODE_B_PASS_2 = (
    "Analyze physical body proportions, facial identity, and wardrobe. Ignore "
    "environment and underlying pose mechanics. 6 zones: 1. Head & Face "
    "2. Upper Body Proportions & Clothing 3. Lower Body Proportions & Clothing "
    "4. Legwear & Footwear 5. Skin & Fabric Physics 6. Accessories & Props"
    + ZONE_FORMAT_HINT
)
MOONDREAM_MODE_B_PASS_2 = VLM_MODE_B_PASS_2

# Scene keywords route WD14 tags to Pass 1 (scene_prompt). Matching is whole-word,
# so "background" catches "simple background" / "white background", and "sky"
# does not match "skyscraper". Words that are also clothing/props are left out
# on purpose ("school", "office", "sun", "light", "cropped", "water").
SCENE_KEYWORDS = {
    # backgrounds and settings
    "background", "indoors", "outdoors", "indoor", "outdoor", "room", "bedroom",
    "bathroom", "kitchen", "classroom", "library", "cafe", "restaurant", "gym",
    "locker room", "balcony", "rooftop", "street", "road", "alley", "bridge",
    "stairs", "city", "cityscape", "building", "skyscraper", "park", "garden",
    "forest", "tree", "trees", "grass", "field", "mountain", "hill", "desert",
    "cave", "beach", "ocean", "sea", "lake", "river", "pool", "poolside",
    "waterfall", "underwater", "castle", "ruins", "temple", "shrine", "stage",
    # sky, time and weather
    "sky", "cloud", "clouds", "starry sky", "moon", "night", "sunset", "sunrise",
    "dusk", "dawn", "evening", "rain", "raining", "snow", "snowing", "fog", "mist",
    # furniture and fixtures
    "window", "door", "wall", "ceiling", "curtains", "carpet", "table", "desk",
    "bed", "chair", "couch", "sofa", "bench", "floor", "bookshelf", "shelf", "lamp",
    # lighting and photographic effects
    "lighting", "backlighting", "sunlight", "moonlight", "sunbeam", "light rays",
    "light particles", "lens flare", "shadow", "silhouette", "bokeh", "blurry",
    "depth of field", "motion blur", "film grain", "vignetting",
    "chromatic aberration", "foreshortening",
    # framing and camera
    "full body", "upper body", "lower body", "cowboy shot", "wide shot", "close-up",
    "portrait", "out of frame", "dutch angle", "from below", "from above",
    "from side", "from behind", "dynamic angle", "pov", "selfie", "fisheye",
    "wide angle",
}

_COMMON_P1 = {
    "sitting", "standing", "kneeling", "squatting", "lying", "leaning",
    "holding", "grasping", "walking", "running", "jumping", "crouching",
    "all fours", "on back", "on stomach", "stretching",
} | SCENE_KEYWORDS

_LLM_TAIL = (
    "The tag lists have already been converted to WD-style tags and reconciled "
    "(WD14 wins on attributes such as colors, Moondream wins on pose and camera): "
    "treat them as ground truth and use the blueprints only for detail the tags "
    "lack. Output Danbooru-style tags only. "
    'Output valid JSON with exactly two keys: "scene_prompt" and '
    '"character_prompt". Both values must be single strings of comma-separated '
    "tags. No markdown, no commentary."
)

_VISION_TAIL = (
    "You are also given the original image. Do a final check of the merged tags "
    "against it: remove tags that clearly contradict the image, add clearly "
    "visible details the tags missed, and keep everything else. Do not invent "
    "anything you cannot see."
)

MODE_CONFIG = {
    "scene_shape": {
        "pass1_prompt": MOONDREAM_MODE_A_PASS_1,
        "pass2_prompt": MOONDREAM_MODE_A_PASS_2,
        # Mass/shape keywords go to Pass 1. Additions vs. original: lighting/furniture.
        "pass1_keywords": _COMMON_P1 | {
            "thick thighs", "wide hips", "breasts", "cleavage",
            "skindentation", "squished", "pressed",
            "backlighting", "couch", "window",
        },
        "llm_instruction": (
            "Merge 'Scene & Body Shape' (Pass 1) with 'Face & Clothes' (Pass 2). "
            "Dress the body described in Pass 1 using the wardrobe from Pass 2. "
            "Do not invent body proportions that are not in the input. "
            "Place environment, lighting and camera in scene_prompt. Place face, "
            "hair, body shape, clothing and props in character_prompt. "
            + _LLM_TAIL
        ),
    },
    "scene_pose": {
        "pass1_prompt": MOONDREAM_MODE_B_PASS_1,
        "pass2_prompt": MOONDREAM_MODE_B_PASS_2,
        # Pose/scene only. Body-shape tags go to Pass 2.
        # Additions vs. original: limb/head keywords so Limb Positioning has tags.
        "pass1_keywords": _COMMON_P1 | {
            "object intersection",
            "arms up", "arms behind", "crossed legs", "legs crossed",
            "spread legs", "head tilt", "looking back",
        },
        "llm_instruction": (
            "Merge 'Scene & Pose' (Pass 1) with 'Character & Body Shape' (Pass 2). "
            "Map the proportions and clothing from Pass 2 onto the pose from "
            "Pass 1 without inventing proportions that are not in the input. "
            "Place environment, lighting, camera and pose in scene_prompt. Place "
            "face, hair, body proportions, clothing and props in character_prompt. "
            + _LLM_TAIL
        ),
    },
}

DUAL_PASS_MODES = tuple(MODE_CONFIG.keys())  # dropdown options


def _get_mode(mode: str) -> dict:
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown dual_pass_mode {mode!r}; expected one of {DUAL_PASS_MODES}")
    return MODE_CONFIG[mode]


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------
def _norm_tag(tag: str) -> str:
    return tag.strip().lower().replace("_", " ")


def _clean_csv(*parts: str) -> str:
    """Join comma-separated chunks, drop empties, de-duplicate, keep order."""
    seen, out = set(), []
    for part in parts:
        for item in (part or "").split(","):
            item = item.strip().strip(".").strip()
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                out.append(item)
    return ", ".join(out)


def _style_tags(csv_str: str) -> str:
    """Final output styling. Currently: optional escaping of parentheses."""
    if TAG_ESCAPE_PARENTHESES:
        csv_str = re.sub(r"(?<!\\)([()])", r"\\\1", csv_str)
    return csv_str


# Matches "1. Zone name:", "1. **Zone name**:", "1) Zone name -" at line start.
_ZONE_HEADER_RE = re.compile(r"(?m)^\s*\d+[.)]\s*\**[^:\n\-–—]+\**\s*[:\-–—]\s*")


def _strip_zone_headers(text: str) -> str:
    return _ZONE_HEADER_RE.sub("", text or "").replace("**", "")


def _flatten_zones(text: str) -> str:
    return _clean_csv(_strip_zone_headers(text).replace("\n", ", "))


_SUBJECT_RE = re.compile(r"^(\d+\+?(girl|boy|girls|boys|other|others)|solo|multiple (girls|boys))$")


def _is_subject_tag(tag: str) -> bool:
    return bool(_SUBJECT_RE.match(tag))


# ---------------------------------------------------------------------------
# PROSE -> WD-STYLE TAGS  (mechanical, needs a tag vocabulary)
# ---------------------------------------------------------------------------
# Best vocabulary: the selected_tags.csv that ships with your WD14 model.
# Without it, only the small built-in set plus this image's own WD14 tags match.
BUILTIN_VOCAB = {
    "standing", "sitting", "kneeling", "squatting", "lying", "leaning",
    "from above", "from below", "from side", "from behind", "dynamic angle",
    "indoors", "outdoors", "indoor", "outdoor", "bed", "chair", "couch", "floor",
    "window", "desk", "table", "backlighting", "depth of field", "foreshortening",
    "looking at viewer", "looking back", "arms up", "crossed legs", "spread legs",
    "thick thighs", "wide hips", "large breasts", "medium breasts", "small breasts",
    "long hair", "short hair", "ponytail", "twintails", "bangs", "braid",
    "blue eyes", "red eyes", "green eyes", "brown eyes",
    "shirt", "t-shirt", "dress", "skirt", "denim shorts", "shorts", "jeans", "pants",
    "jacket", "hoodie", "sweater", "bikini", "swimsuit", "uniform", "school uniform",
    "thighhighs", "socks", "boots", "sneakers", "high heels", "sandals", "barefoot",
    "glasses", "hat", "necklace", "earrings", "gloves", "belt", "bag", "smile",
    "open mouth", "closed mouth", "blush", "holding", "phone", "cup", "umbrella",
    "simple background", "white background", "sky", "night", "sunset", "full body",
    "upper body", "close-up", "portrait", "cowboy shot", "no humans",
}

COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white",
    "grey", "gray", "brown", "blonde", "silver", "aqua", "beige", "gold", "navy",
}

# Mutually exclusive groups: if Moondream names one, WD14 members of that group
# that Moondream did not name are dropped (Moondream wins on pose/camera).
EXCLUSIVE_GROUPS = [
    {"sitting", "standing", "kneeling", "squatting", "lying"},
    {"from above", "from below", "from side", "from behind"},
]

# Attribute groups where WD14 wins: if WD14 names one member, the other members
# Moondream named are dropped (e.g. WD14 "short hair" beats Moondream "long hair").
ATTRIBUTE_GROUPS = [
    {"very long hair", "long hair", "medium hair", "short hair", "very short hair"},
    {"flat chest", "small breasts", "medium breasts", "large breasts", "huge breasts",
     "gigantic breasts"},
]

# ---------------------------------------------------------------------------
# PHASE 2: CROP TAG POLICY & ZONE GEOMETRY
# ---------------------------------------------------------------------------
CROP_ARTIFACT_TAGS = {
    # boundary cut-off tags
    "head out of frame", "face out of frame", "feet out of frame",
    "arm out of frame", "legs out of frame", "hands out of frame",
    "out of frame", "cropped legs", "partially submerged",
    # camera & framing tags
    "full body", "upper body", "lower body", "cowboy shot", "wide shot",
    "close-up", "portrait",
    # zoom focus tags
    "solo focus", "breast focus", "eye focus", "face focus",
    "navel focus", "foot focus", "hip focus", "back focus", "armpit focus",
    "focus",
}

RATING_TAGS = {
    "general", "sensitive", "questionable", "explicit",
}

BODY_SIZE_AND_COUNT_TAGS = {
    "solo", "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls",
    "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple boys",
    "large breasts", "small breasts", "medium breasts", "huge breasts", "gigantic breasts",
    "flat chest", "tall", "short", "petite", "chibi",
}

CONCRETE_SCENE_OBJECTS = {
    "plant", "potted plant", "houseplant", "flower", "flowers", "bouquet", "petal", "rose",
    "sunflower", "lily", "tulip", "cherry blossoms", "tree", "leaf", "leaves", "foliage",
    "fern", "vines", "ivy", "grass", "bush", "hedge", "cactus", "succulent", "vase",
    "flowerpot", "flower pot", "window", "curtain", "curtains", "drapes", "chair", "armchair",
    "couch", "sofa", "table", "desk", "bookshelf", "bookcase", "shelf", "lamp", "lantern",
    "chandelier", "street lamp", "mirror", "clock", "poster", "painting", "picture frame",
    "signboard", "pillow", "cushion", "carpet", "rug", "mat", "bed", "television", "monitor",
    "laptop", "computer", "cup", "mug", "teacup", "bottle", "wine glass", "book", "umbrella",
    "parasol", "weapon", "sword", "gun", "staff", "wand", "fan", "food", "drink", "phone",
    "cellphone", "smartphone", "camera", "plushie", "doll", "instrument", "guitar", "violin",
}


def is_crop_artifact_tag(tag: str) -> bool:
    norm = _norm_tag(tag)
    if norm in CROP_ARTIFACT_TAGS:
        return True
    if norm.endswith(" focus"):
        return True
    if norm.endswith(" out of frame"):
        return True
    if norm.startswith("cropped ") or norm.endswith(" cropped"):
        return True
    return False


def is_rating_tag(tag: str) -> bool:
    return _norm_tag(tag) in RATING_TAGS


def is_body_size_or_count_tag(tag: str) -> bool:
    return _norm_tag(tag) in BODY_SIZE_AND_COUNT_TAGS


def is_concrete_scene_object(norm: str) -> bool:
    n = _norm_tag(norm)
    return any(obj == n or obj in n for obj in CONCRETE_SCENE_OBJECTS)


def is_global_framing_tag(tag: str) -> bool:
    norm = _norm_tag(tag)
    if is_crop_artifact_tag(norm):
        return True
    if norm in SCENE_KEYWORDS:
        return True
    if any(norm.endswith(sfx) for sfx in (" shot", " view", " angle", " background")):
        return True
    return False


def filter_crop_tags(
    crop_tags: Iterable,
    crop_type: str = "character_zone",
    character_names: Optional[Iterable[str]] = None,
    min_attribute_threshold: float = 0.55,
    scene_object_threshold: float = 0.40,
) -> list[tuple[str, float]]:
    """Pure per-crop tag policy filtering.

    Filters out crop-boundary artifacts, ratings, character names, body sizes/counts,
    and framing tags, retaining only confident micro-attributes and concrete scene objects.
    """
    items = []
    if isinstance(crop_tags, dict):
        items = [(k, _norm_tag(k), float(v)) for k, v in crop_tags.items()]
    else:
        for entry in crop_tags:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                items.append((entry[0], _norm_tag(entry[0]), float(entry[1])))
            elif isinstance(entry, str):
                items.append((entry, _norm_tag(entry), 1.0))
            elif hasattr(entry, "name") and hasattr(entry, "score"):
                items.append((entry.name, _norm_tag(entry.name), float(entry.score)))

    char_set = {_norm_tag(c) for c in (character_names or [])}
    accepted = []

    for orig_tag, norm, score in items:
        if not norm:
            continue

        # 1. Reject crop artifacts (boundary & zoom artifacts)
        if is_crop_artifact_tag(norm):
            continue

        # 2. Reject ratings
        if is_rating_tag(norm):
            continue

        # 3. Reject character names
        if norm in char_set:
            continue

        # 4. Reject body size & count tags
        if is_body_size_or_count_tag(norm):
            continue

        if crop_type == "character_zone":
            # 5a. Character zones reject global framing/scene
            if is_global_framing_tag(norm):
                continue
            # Micro-attributes require confidence >= 0.55
            if score >= min_attribute_threshold:
                accepted.append((orig_tag, score))

        elif crop_type == "quadrant":
            # 5b. Quadrants reject generic backgrounds that contradict full frame
            if norm in {"simple background", "white background", "grey background", "monochrome", "blurry background"}:
                continue
            # Concrete scene objects admitted at >= 0.40
            if is_concrete_scene_object(norm):
                if score >= scene_object_threshold:
                    accepted.append((orig_tag, score))
            else:
                # Other props or micro-attributes admitted at >= 0.55
                if not is_global_framing_tag(norm) and score >= min_attribute_threshold:
                    accepted.append((orig_tag, score))

    return accepted


def merge_crop_tags(
    base_tags: Iterable,
    crop_tags: Iterable,
    crop_type: str = "character_zone",
    character_names: Optional[Iterable[str]] = None,
    min_attribute_threshold: float = 0.55,
    scene_object_threshold: float = 0.40,
) -> list[tuple[str, float]]:
    """Merges candidate tags from a crop into base tags using pure tag policy."""
    filtered = filter_crop_tags(
        crop_tags,
        crop_type=crop_type,
        character_names=character_names,
        min_attribute_threshold=min_attribute_threshold,
        scene_object_threshold=scene_object_threshold,
    )

    out = []
    if isinstance(base_tags, dict):
        out = [(k, float(v)) for k, v in base_tags.items()]
    else:
        for entry in base_tags:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                out.append((entry[0], float(entry[1])))
            elif isinstance(entry, str):
                out.append((entry, 1.0))
            elif hasattr(entry, "name") and hasattr(entry, "score"):
                out.append((entry.name, float(entry.score)))

    for c_orig, c_score in filtered:
        norm_c = _norm_tag(c_orig)
        found = False
        for idx, (b_orig, b_score) in enumerate(out):
            if _norm_tag(b_orig) == norm_c:
                out[idx] = (b_orig, max(b_score, c_score))
                found = True
                break
        if not found:
            out.append((c_orig, c_score))
    return out


def compute_zone_boxes(
    img_w: float,
    img_h: float,
    box_x0: float,
    box_y0: float,
    box_x1: float,
    box_y1: float,
) -> list[dict]:
    """Pure zone box function:
    figure box, image size and edge contact in -> valid zones out.
    Skips lower zone for half-body or edge-clipped figures.
    """
    if img_w <= 0.0 or img_h <= 0.0:
        return []

    x0 = max(0.0, min(float(box_x0), float(img_w)))
    y0 = max(0.0, min(float(box_y0), float(img_h)))
    x1 = max(x0, min(float(box_x1), float(img_w)))
    y1 = max(y0, min(float(box_y1), float(img_h)))

    w = x1 - x0
    h = y1 - y0
    if w <= 1.0 or h <= 1.0:
        return []

    bottom_edge = y1 >= (0.98 * img_h)
    height_ratio = h / img_h

    # Margins (6% horiz, 4% vert)
    mx = w * 0.06
    my = h * 0.04

    def make_box(name: str, zy0: float, zy1: float) -> dict:
        cx0 = max(0.0, min(x0 - mx, img_w - 1.0))
        cy0 = max(0.0, min(zy0 - my, img_h - 1.0))
        cx1 = min(img_w, max(x1 + mx, cx0 + 1.0))
        cy1 = min(img_h, max(zy1 + my, cy0 + 1.0))
        return {
            "name": name,
            "x0": round(cx0, 2),
            "y0": round(cy0, 2),
            "x1": round(cx1, 2),
            "y1": round(cy1, 2),
        }

    zones = []
    # 1. Upper (0 - 55%): Head, Hair, Face - Always valid
    zones.append(make_box("Upper", y0, y0 + h * 0.55))

    # 2. Middle (30% - 70%): Torso, Arms, Held Items
    if height_ratio >= 0.25:
        zones.append(make_box("Middle", y0 + h * 0.30, y0 + h * 0.70))

    # 3. Lower (55% - 100%): Hips, Legs, Footwear
    # Skip lower zone if figure touches the bottom frame and height ratio < 0.72 (half-body/cowboy shot cut off)
    is_cut_off_bottom = bottom_edge and (height_ratio < 0.72)
    if not is_cut_off_bottom and (height_ratio >= 0.40):
        zones.append(make_box("Lower", y0 + h * 0.55, y1))

    return zones

_SINGLE_STOP = {
    "light", "hand", "left", "right", "front", "back", "side", "top", "one", "two",
    "air", "dark", "small", "large", "long", "short", "open", "close", "line",
    "edge", "point", "hair", "eyes", "face", "body", "head", "legs", "arms",
}


def load_vocab(path: str) -> set:
    """Load tag names from a WD14 selected_tags.csv (column 'name')."""
    vocab = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = _norm_tag(row.get("name") or "")
            if name:
                vocab.add(name)
    return vocab


def _tokens(clause: str) -> list:
    return re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", clause.lower())


def _variants(phrase_tokens: list) -> list:
    base = " ".join(phrase_tokens)
    out = [base]
    last = phrase_tokens[-1]
    if len(last) > 3 and last.endswith("s"):
        out.append(" ".join(phrase_tokens[:-1] + [last[:-1]]))
    return out


_NEGATORS = {
    "no", "not", "without", "lacks", "lacking", "none", "never", "neither", "nor",
    "missing", "isn't", "aren't", "doesn't", "don't", "hasn't", "haven't", "wasn't",
}
_HEDGES = {
    "appears", "appear", "appearing", "possibly", "maybe", "perhaps", "seems", "seem",
    "seemingly", "might", "probably", "likely", "could", "suggests", "resembles",
    "presumably",
}
# These end a negation/hedge: "no hat but a smile" keeps "smile".
_SCOPE_END = {"but", "with", "while", "though", "although", "however", "yet"}
# "... and has a smile": a finite verb right after "and" starts a new statement.
_NEW_STATEMENT_VERBS = {"has", "have", "is", "are", "wears", "holds", "carries", "shows"}


def analyze_prose(text: str, vocab: set, max_n: int = 4) -> dict:
    """Greedy longest-match of vocabulary tags inside Moondream prose.

    Returns {"tags", "strong", "weak", "negated"}:
      tags    - tags found (negated ones excluded), in order
      strong  - tags stated plainly
      weak    - tags only stated with a hedge ("appears to be", "possibly")
      negated - tags that were explicitly denied ("no glasses", "not wearing shoes")
    A vocabulary tag that itself starts with a negator (e.g. "no humans") is
    matched as a tag, not treated as a negation.
    """
    tags, strong, weak, negated = [], set(), set(), []
    flat = _strip_zone_headers(text).replace("\n", ", ")
    for clause in re.split(r"[,.;:]", flat):
        toks = _tokens(clause)
        neg = hedge = False
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(max_n, len(toks) - i), 0, -1):
                for cand in _variants(toks[i:i + n]):
                    if cand in vocab and not (n == 1 and cand in _SINGLE_STOP):
                        hit = (cand, n)
                        break
                if hit:
                    break
            if hit:
                tag, n = hit
                if neg:
                    if tag not in negated:
                        negated.append(tag)
                else:
                    if tag not in tags:
                        tags.append(tag)
                    if hedge:
                        if tag not in strong:
                            weak.add(tag)
                    else:
                        strong.add(tag)
                        weak.discard(tag)
                i += n
                continue
            tok = toks[i]
            if tok in _NEGATORS:
                neg = True
            elif tok in _HEDGES:
                hedge = True
            elif tok in _SCOPE_END:
                neg = hedge = False
            elif tok in _NEW_STATEMENT_VERBS and i > 0 and toks[i - 1] == "and":
                neg = hedge = False
            i += 1
    return {"tags": tags, "strong": strong, "weak": weak, "negated": negated}


def prose_to_tags(text: str, vocab: set, max_n: int = 4) -> list:
    """Tags found in Moondream prose, with negated mentions removed."""
    return analyze_prose(text, vocab, max_n)["tags"]


# ---------------------------------------------------------------------------
# RECONCILE: WD14 adjusts Moondream
# ---------------------------------------------------------------------------
def _split_color(tag: str):
    parts = tag.split(" ", 1)
    if len(parts) == 2 and parts[0] in COLORS:
        return parts[0], parts[1]
    return None


def reconcile(md_tags: Iterable, wd_tags: Iterable):
    """Return (drop_from_moondream, drop_from_wd14)."""
    md_set, wd_set = set(md_tags), set(wd_tags)
    drop_md, drop_wd = set(), set()

    wd_colors = {}
    for t in wd_set:
        c = _split_color(t)
        if c:
            wd_colors.setdefault(c[1], set()).add(c[0])
    for t in md_set:
        c = _split_color(t)
        if c and c[1] in wd_colors and c[0] not in wd_colors[c[1]]:
            drop_md.add(t)  # WD14 wins on colors
        elif not c and t in wd_colors:
            drop_md.add(t)  # plain "dress" yields to WD14's more specific "blue dress"

    for group in EXCLUSIVE_GROUPS:
        if md_set & group:
            drop_wd |= (wd_set & group) - md_set  # Moondream wins on pose/camera

    for group in ATTRIBUTE_GROUPS:
        if wd_set & group:
            drop_md |= (md_set & group) - wd_set  # WD14 wins on attribute sizes
    return drop_md, drop_wd


# ---------------------------------------------------------------------------
# PHASE 2: DYNAMIC TAG SORTER
# ---------------------------------------------------------------------------
class WD14DualPassRouter:
    def __init__(self):
        self._patterns = {
            mode: re.compile(
                r"\b(?:" + "|".join(re.escape(k) for k in sorted(cfg["pass1_keywords"], key=len, reverse=True)) + r")\b"
            )
            for mode, cfg in MODE_CONFIG.items()
        }

    def normalized_tags(self, raw_wd14_tags: str) -> list:
        return [t for t in (_norm_tag(x) for x in (raw_wd14_tags or "").split(",")) if t]

    def sort_tag_lists(self, raw_wd14_tags: str, mode: str) -> tuple:
        _get_mode(mode)
        pat = self._patterns[mode]
        pass1, pass2 = [], []
        for tag in self.normalized_tags(raw_wd14_tags):
            (pass1 if pat.search(tag) else pass2).append(tag)
        return pass1, pass2

    def sort_tags(self, raw_wd14_tags: str, mode: str) -> dict:
        p1, p2 = self.sort_tag_lists(raw_wd14_tags, mode)
        return {"pass1_tags": ", ".join(p1), "pass2_tags": ", ".join(p2)}


# ---------------------------------------------------------------------------
# MERGE: convert prose to tags, then let WD14 adjust them
# ---------------------------------------------------------------------------
def merge_tags(raw_wd14_tags: str, pass1_txt: str, pass2_txt: str, mode: str,
               vocab: Optional[set] = None, require_two_sources: bool = False) -> dict:
    router = WD14DualPassRouter()
    wd_all = router.normalized_tags(raw_wd14_tags)
    wd1, wd2 = router.sort_tag_lists(raw_wd14_tags, mode)

    full_vocab = set(BUILTIN_VOCAB) | set(wd_all) | (vocab or set())
    a1 = analyze_prose(pass1_txt, full_vocab)
    a2 = analyze_prose(pass2_txt, full_vocab)
    md1, md2 = a1["tags"], a2["tags"]

    # Hedged Moondream tags ("appears to be", "possibly") count only if WD14 agrees.
    wd_set = set(wd_all)
    hedged_unconfirmed = ((a1["weak"] | a2["weak"]) - (a1["strong"] | a2["strong"])) - wd_set
    md1 = [t for t in md1 if t not in hedged_unconfirmed]
    md2 = [t for t in md2 if t not in hedged_unconfirmed]
    negated = sorted(set(a1["negated"]) | set(a2["negated"]))

    # Phase 3: Two-Source Rule
    # A tag reported solely by the VLM is retained only if WD14 confirms it or a second pass/crop agrees.
    # Exclude pose/camera groups where VLM intentionally overrides WD14.
    exclusive_all = {item for group in EXCLUSIVE_GROUPS for item in group}
    all_vlm = set(md1) | set(md2)
    unconfirmed = {
        t for t in all_vlm
        if (t not in wd_set) and (t not in exclusive_all) and not (t in md1 and t in md2)
    }
    if require_two_sources and unconfirmed:
        md1 = [t for t in md1 if t not in unconfirmed]
        md2 = [t for t in md2 if t not in unconfirmed]

    drop_md, drop_wd = reconcile(md1 + md2, wd1 + wd2)
    if drop_md or drop_wd:
        log.info("reconcile dropped moondream=%s wd14=%s", sorted(drop_md), sorted(drop_wd))

    subject = [t for t in wd_all if _is_subject_tag(t)]
    wd1 = [t for t in wd1 if t not in drop_wd and not _is_subject_tag(t)]
    wd2 = [t for t in wd2 if t not in drop_wd and not _is_subject_tag(t)]
    md1 = [t for t in md1 if t not in drop_md]
    md2 = [t for t in md2 if t not in drop_md]
    if negated:
        log.info("Moondream explicitly denied: %s", negated)
    if wd2:
        log.debug("WD14 tags that defaulted to Pass 2 (grow SCENE_KEYWORDS from these): %s", wd2)

    return {
        # VLM is the base; WD14 tags are added/adjusted on top.
        "pass1_tags": md1 + wd1,
        "pass2_tags": md2 + wd2,
        "subject_tags": subject,
        "dropped_vlm": sorted(drop_md),
        "dropped_wd14": sorted(drop_wd),
        "dropped_hedged": sorted(hedged_unconfirmed),
        "dropped_unconfirmed": sorted(unconfirmed),
        "negated_vlm": negated,
        "unrouted_wd14": list(wd2),
        "dropped_moondream": sorted(drop_md),
        "negated_moondream": negated,
    }


# ---------------------------------------------------------------------------
# PHASE 3: LLM PAYLOAD + JSON PARSER
# ---------------------------------------------------------------------------
class AntigravityTagFilter_LLM:
    def build_payload(self, raw_wd14_tags: str, pass1_txt: str, pass2_txt: str, mode: str,
                      vocab: Optional[set] = None, vision: bool = False) -> dict:
        cfg = _get_mode(mode)
        merged = merge_tags(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab)
        instruction = cfg["llm_instruction"] + (" " + _VISION_TAIL if vision else "")
        return {
            "system_instruction": instruction,
            "pass_1_blueprint": _flatten_zones(pass1_txt),
            "pass_1_tags": ", ".join(merged["pass1_tags"]),
            "pass_2_blueprint": _flatten_zones(pass2_txt),
            "pass_2_tags": ", ".join(merged["subject_tags"] + merged["pass2_tags"]),
        }

    @staticmethod
    def payload_to_text(payload: dict) -> str:
        """Gemma has no system role: put the instruction at the top of the user turn."""
        body = {k: v for k, v in payload.items() if k != "system_instruction"}
        return payload["system_instruction"] + "\n\nINPUT:\n" + json.dumps(body, ensure_ascii=False, indent=2)


def _as_str(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return ""


def _strip_fences(raw_llm_text: str) -> str:
    text = (raw_llm_text or "").strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()


def _try_parse_gemma_json(raw_llm_text: str) -> Optional[dict]:
    """Return {"scene_prompt", "character_prompt"} (values may be empty) or None if unparseable."""
    text = _strip_fences(raw_llm_text)

    # Tolerate preamble/postamble: take first '{' to last '}'.
    start, end = text.find("{"), text.rfind("}")
    candidate = text[start:end + 1] if start != -1 and end > start else text

    try:
        data = json.loads(candidate)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("Gemma JSON parse failed: %s", e)
        return None
    return {
        "scene_prompt": _as_str(data.get("scene_prompt")),
        "character_prompt": _as_str(data.get("character_prompt")),
    }


def parse_gemma_json_response(raw_llm_text: str) -> dict:
    """Parse Gemma's reply. Never raises; on failure returns a best-effort fallback."""
    default_scene = QUALITY_PREFIX or ""
    parsed = _try_parse_gemma_json(raw_llm_text)
    if parsed is not None:
        return {
            "scene_prompt": parsed["scene_prompt"] or default_scene,
            "character_prompt": parsed["character_prompt"],
        }
    text = _strip_fences(raw_llm_text)
    return {
        "scene_prompt": default_scene,
        "character_prompt": _clean_csv(re.sub(r"[{}\"]", "", text).replace("\n", ", ")),
    }


# ---------------------------------------------------------------------------
# PHASE 3.5: NO-LLM MECHANICAL PATH (pure WD-style tag prompt)
# ---------------------------------------------------------------------------
class AntigravityTagFilter_NoLLM:
    def process_direct(self, raw_wd14_tags: str, pass1_txt: str, pass2_txt: str, mode: str,
                       vocab: Optional[set] = None, keep_unmatched_prose: bool = False,
                       require_two_sources: bool = False) -> dict:
        merged = merge_tags(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab, require_two_sources=require_two_sources)
        extra1 = _flatten_zones(pass1_txt) if keep_unmatched_prose else ""
        extra2 = _flatten_zones(pass2_txt) if keep_unmatched_prose else ""
        scene = _clean_csv(QUALITY_PREFIX, ", ".join(merged["pass1_tags"]), extra1)
        character = _clean_csv(
            ", ".join(merged["subject_tags"]), ", ".join(merged["pass2_tags"]), extra2
        )
        return {"scene_prompt": _style_tags(scene), "character_prompt": _style_tags(character)}


def _image_to_b64(image) -> str:
    """Accept raw bytes, a file path, or a PIL image; return base64 for Ollama."""
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    elif isinstance(image, str):
        with open(image, "rb") as f:
            data = f.read()
    elif hasattr(image, "save"):  # PIL.Image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()
    else:
        raise TypeError(f"Unsupported image type for Gemma: {type(image).__name__}")
    return base64.b64encode(data).decode("ascii")


def _ollama_base(host: Optional[str] = None) -> str:
    base = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    return base if base.startswith("http") else "http://" + base


def check_ollama(model: str = GEMMA_OLLAMA_MODEL, host: Optional[str] = None,
                 timeout: float = 5) -> tuple:
    """Quick health check: is the server up and the model pulled? Returns (ok, message)."""
    base = _ollama_base(host)
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # server down, wrong host, bad response
        return False, f"Ollama not reachable at {base} ({type(e).__name__}: {e}); used the mechanical merge"
    names = {(m.get("name") or m.get("model") or "") for m in data.get("models", [])}
    wanted = {model} | ({model + ":latest"} if ":" not in model else set())
    if names & wanted:
        return True, "ok"
    return False, f"Model {model} is not pulled in Ollama; run `ollama pull {model}`. Used the mechanical merge"


def make_ollama_gemma_fn(model: str = GEMMA_OLLAMA_MODEL,
                         num_ctx: int = GEMMA_CONTEXT_WINDOW,
                         host: Optional[str] = None,
                         json_mode: bool = True,
                         timeout: float = 300) -> Callable:
    """Build a gemma_fn that talks to a local Ollama server (/api/chat).

    Returns (text, done_reason); done_reason "length" triggers the truncation retry.
    host defaults to $OLLAMA_HOST, then http://localhost:11434.
    """
    base = _ollama_base(host)

    def gemma_fn(text: str, temperature: float, max_tokens: int = GEMMA_MAX_TOKENS, image=None):
        message = {"role": "user", "content": text}
        if image is not None:
            message["images"] = [_image_to_b64(image)]
        body = {
            "model": model,
            "messages": [message],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message") or {}).get("content", ""), data.get("done_reason")

    return gemma_fn


_REFUSAL_RE = re.compile(r"\b(sorry|cannot|can't|unable to)\b", re.IGNORECASE)


def _valid_vlm(text: str) -> bool:
    t = (text or "").strip()
    return len(t) >= 20 and not _REFUSAL_RE.search(t[:80])


_valid_moondream = _valid_vlm


def _run_vlm(vlm_fn: Callable, image, prompt: str,
             label: str = "pass", warnings: Optional[list] = None) -> str:
    """One VLM pass; retry if it fails, is empty, or looks like a refusal.

    Gives up with '' (that pass then contributes only its WD14 tags) and records a
    warning. A TypeError is re-raised: it means the wrapper is wired up wrong.
    """
    last_error = None
    for attempt in range(VLM_RETRIES + 1):
        temp = VLM_TEMPERATURE + (VLM_RETRY_TEMP_BUMP if attempt else 0.0)
        try:
            out = vlm_fn(image, prompt, temp)
        except TypeError:
            raise
        except Exception as e:
            last_error = e
            log.warning("VLM %s attempt %d/%d failed: %s", label, attempt + 1,
                        VLM_RETRIES + 1, e)
            continue
        if _valid_vlm(out):
            return out
        log.warning("VLM %s attempt %d/%d empty or refusal-like", label,
                    attempt + 1, VLM_RETRIES + 1)
    if warnings is not None:
        reason = f"{type(last_error).__name__}: {last_error}" if last_error else "empty or refusal-like output"
        warnings.append(f"VLM {label} unavailable ({reason}); used WD14 tags only for it")
    return ""


_run_moondream = _run_vlm


_TRUNCATION_REASONS = {"length", "max_tokens", "max_length", "max_new_tokens"}


def _call_gemma(gemma_fn: Callable, text: str, image, max_tokens: int) -> tuple:
    """Call gemma_fn; return (raw_text, was_truncated).

    gemma_fn may return either a string, or (string, finish_reason). Without a
    finish reason, a reply that does not end in '}' (ignoring a closing code
    fence) is treated as truncated.
    """
    kwargs = {"max_tokens": max_tokens}
    if image is not None:
        kwargs["image"] = image
    result = gemma_fn(text, GEMMA_TEMPERATURE, **kwargs)
    if isinstance(result, tuple):
        raw, reason = result
    else:
        raw, reason = result, None
    raw = raw or ""
    if reason is not None:
        return raw, str(reason).strip().lower() in _TRUNCATION_REASONS
    tail = re.sub(r"\s*```\s*$", "", raw.strip())
    return raw, not tail.endswith("}")


# ---------------------------------------------------------------------------
# PHASE 4: INTEGRATION
# vlm_fn(image, prompt, temperature) -> str
# gemma_fn(text, temperature, max_tokens=..., image=...) -> str | (str, finish_reason)
#   (image is only passed on the vision pass; finish_reason "length" or
#    "max_tokens" means the reply was cut off and triggers one retry)
# single_pass_fn(image, raw_wd14_tags) -> dict   (your existing non-dual path)
# vocab_path: WD14 selected_tags.csv (strongly recommended for good conversion)
# ---------------------------------------------------------------------------
def run_pipeline(
    image,
    raw_wd14_tags: str,
    enable_dual_pass: bool,
    dual_pass_mode: str,
    use_llm_synthesis: bool,
    vlm_fn: Optional[Callable] = None,
    gemma_fn: Optional[Callable] = None,
    single_pass_fn: Optional[Callable] = None,
    vocab_path: Optional[str] = None,
    keep_unmatched_prose: bool = False,
    gemma_vision_pass: bool = False,
    moondream_fn: Optional[Callable] = None,
) -> dict:
    if vlm_fn is None:
        vlm_fn = moondream_fn
    if vlm_fn is None and enable_dual_pass:
        raise TypeError("run_pipeline() missing required argument: 'vlm_fn'")

    if not enable_dual_pass:
        # dual_pass_mode and use_llm_synthesis are ignored on this path.
        if single_pass_fn is None:
            raise ValueError("enable_dual_pass is False but no single_pass_fn was provided")
        return single_pass_fn(image, raw_wd14_tags)

    cfg = _get_mode(dual_pass_mode)  # raises on an unknown mode: a config error, not a runtime one
    vocab = load_vocab(vocab_path) if vocab_path else None
    warnings: list = []

    pass1_txt = _run_vlm(vlm_fn, image, cfg["pass1_prompt"], "pass 1", warnings)
    pass2_txt = _run_vlm(vlm_fn, image, cfg["pass2_prompt"], "pass 2", warnings)

    def mechanical(reason: Optional[str] = None) -> dict:
        if reason:
            log.warning(reason)
            warnings.append(reason)
        out = AntigravityTagFilter_NoLLM().process_direct(
            raw_wd14_tags, pass1_txt, pass2_txt, dual_pass_mode, vocab, keep_unmatched_prose
        )
        out["warnings"] = warnings
        return out

    if not use_llm_synthesis:
        if gemma_vision_pass:
            log.warning("gemma_vision_pass ignored: use_llm_synthesis is False")
        return mechanical()

    if gemma_fn is None:
        ok, msg = check_ollama()  # default backend: huihui_ai/gemma-4-abliterated:12b via Ollama
        if not ok:
            return mechanical(msg)
        gemma_fn = make_ollama_gemma_fn()

    llm = AntigravityTagFilter_LLM()
    payload = llm.build_payload(raw_wd14_tags, pass1_txt, pass2_txt, dual_pass_mode,
                                vocab, vision=gemma_vision_pass)
    text = llm.payload_to_text(payload)
    gemma_image = image if gemma_vision_pass else None
    try:
        raw, truncated = _call_gemma(gemma_fn, text, gemma_image, GEMMA_MAX_TOKENS)
        if truncated:
            log.warning("Gemma reply cut off at %d tokens; retrying with %d",
                        GEMMA_MAX_TOKENS, GEMMA_MAX_TOKENS_RETRY)
            raw, truncated = _call_gemma(gemma_fn, text, gemma_image, GEMMA_MAX_TOKENS_RETRY)
            if truncated:
                log.warning("Gemma reply still cut off at %d tokens", GEMMA_MAX_TOKENS_RETRY)
    except TypeError:
        raise  # a gemma_fn with the wrong signature is a wiring bug: surface it
    except Exception as e:
        return mechanical(f"Gemma call failed ({type(e).__name__}: {e}); used the mechanical merge")

    parsed = _try_parse_gemma_json(raw)
    if parsed is None or not parsed["character_prompt"]:
        return mechanical("Gemma output unusable; used the mechanical merge")
    return {
        "scene_prompt": _style_tags(parsed["scene_prompt"] or (QUALITY_PREFIX or "")),
        "character_prompt": _style_tags(parsed["character_prompt"]),
        "warnings": warnings,
    }
