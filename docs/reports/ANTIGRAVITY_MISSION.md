# MISSION CONTEXT FOR ANTIGRAVITY

We are finalizing the `Prompt-Condenser` dual-pass Moondream + WD14 pipeline. A complete, tested reference implementation is included below. **Your job is to integrate it into the existing node/main execution loop, not to redesign it.** Keep prompt text, keyword sets, constants and function names exactly as written unless a task below says otherwise.

## 1. What the pipeline does

1. WD14 tags the whole image (Danbooru-style tags). Moondream describes the same image in two passes (prose).
2. Moondream's prose is converted into WD-style tags (`prose_to_tags`), using a tag vocabulary.
3. WD14's tags adjust the Moondream tags (`reconcile`):
   - **WD14 wins on attributes:** colors on clothing, hair, eyes (`blue dress` beats Moondream's `red dress`).
   - **WD14 also wins on attribute sizes:** hair length and breast size (`short hair` from WD14 drops Moondream's `long hair`).
   - **Moondream wins on pose and camera:** if Moondream names `standing` or `from below`, conflicting WD14 tags (`sitting`, `from above`) are dropped.
4. The merged tags are split into `scene_prompt` and `character_prompt` according to the selected mode.
5. Robustness built in: a Moondream pass that comes back empty or refusal-like is retried once (slightly warmer), then that pass contributes only its WD14 tags. If Gemma's output is unparseable or has an empty `character_prompt`, the pipeline falls back to the mechanical merge instead of returning junk.
6. Optionally (`use_llm_synthesis`), Gemma polishes the merged tags into the final JSON. Optionally (`gemma_vision_pass`), Gemma also receives the image and checks the tags against it.

## 2. The two selectable modes (dropdown `dual_pass_mode`)

* Mode A (`"scene_shape"`): Pass 1 = Scene + Body Shape. Pass 2 = Face + Clothes.
* Mode B (`"scene_pose"`): Pass 1 = Scene + Skeletal Pose. Pass 2 = Character + Body Shape.

Both modes live in one `MODE_CONFIG` dict. An unknown mode raises `ValueError`. Never fall through silently to one mode.

## 3. Toggles to expose in the main execution loop

1. `enable_dual_pass` (Boolean). If False, call the existing single-pass path (`single_pass_fn`) and ignore the other toggles.
2. `dual_pass_mode` (Dropdown: `"scene_shape"` OR `"scene_pose"`; use `DUAL_PASS_MODES` for the options).
3. `use_llm_synthesis` (Boolean). If False, use `AntigravityTagFilter_NoLLM` (pure WD-style tag prompt, no LLM).
4. `gemma_vision_pass` (Boolean). Only applies when `use_llm_synthesis` is True; ignored (with a logged warning) otherwise.

Entry point: `run_pipeline(image, raw_wd14_tags, enable_dual_pass, dual_pass_mode, use_llm_synthesis, moondream_fn, gemma_fn=None, single_pass_fn=None, vocab_path=None, keep_unmatched_prose=False, gemma_vision_pass=False)`. It returns `{"scene_prompt": str, "character_prompt": str}`.

## 4. Integration tasks for you

1. **Add the file.** Save the reference implementation (section 7) as `prompt_condenser_dual_pass.py` next to the existing node code. Add the tests (section 8) as `test_prompt_condenser.py`.
2. **Moondream wrapper.** Write `moondream_fn(image, prompt, temperature) -> str` around the existing Moondream call and pass `MOONDREAM_TEMPERATURE` (0.1). First check that the installed Moondream version accepts a temperature setting; if it does not, pass what it does support and note that in a comment.
3. **WD14 thresholds.** `WD14_GENERAL_THRESHOLD` (0.22) and `WD14_CHARACTER_THRESHOLD` (0.85) are NOT used inside the module. Pass them to the existing WD14 tagger node. The module only receives the resulting comma-separated tag string.
4. **WD14 vocabulary (important).** Find the `selected_tags.csv` that ships with the WD14 model in use and pass its path as `vocab_path`. Without it, prose-to-tag conversion only recognizes a small built-in list and loses uncommon phrasing.
5. **Single-pass path.** Provide `single_pass_fn(image, raw_wd14_tags) -> dict` that wraps the existing non-dual behavior.
6. **Gemma via Ollama.** The default model is `huihui_ai/gemma-4-abliterated:12b` (128K context, text and image input), served by Ollama. If `gemma_fn` is None, `run_pipeline` builds one with `make_ollama_gemma_fn()`, which uses `$OLLAMA_HOST` or `http://localhost:11434`. The user must run `ollama pull huihui_ai/gemma-4-abliterated:12b` first.
7. **Context window.** `GEMMA_CONTEXT_WINDOW = 131072` (the model's 128K). Actual input is only about 1.5K tokens, so if the GPU runs out of memory or Ollama offloads to CPU, lower this constant to 8192 or 16384 rather than changing anything else.
8. **Token cap and retry.** `GEMMA_MAX_TOKENS = 768`, with one retry at `GEMMA_MAX_TOKENS_RETRY = 1024` if the reply was cut off. A custom `gemma_fn` has the signature `gemma_fn(text, temperature, max_tokens=..., image=...)` and may return either a string or `(string, finish_reason)`. Do not add more than one retry.
9. **UI wiring.** Expose the four toggles from section 3 as node inputs with the defaults: `enable_dual_pass=True`, `dual_pass_mode="scene_shape"`, `use_llm_synthesis=True`, `gemma_vision_pass=False`.
10. **Run the tests** (`python -m pytest test_prompt_condenser.py`). All 17 must pass before you finish, and again after your integration changes.

## 5. Rules

* Do not rewrite or "improve" the Moondream prompts, keyword sets, reconcile rules or LLM instructions without being asked. If you think something is wrong, say so in your report instead.
* `TAG_ESCAPE_PARENTHESES` (default False) escapes `(` and `)` as `\(` and `\)` for A1111/ComfyUI prompt syntax. Expose it as a setting only if the downstream consumer needs it.
* Do not hardcode `1girl, solo`; subject tags come from WD14.
* Keep `QUALITY_PREFIX` configurable (empty string disables it).
* Use `logging`, not `print`.
* Do not store or log the user's images.

## 6. Known limitations (do not try to fix unless asked)

* The 7-crop region scan (full image, face, upper body, middle, lower body, hands, held items, plus a scene-only inverse) is NOT implemented. Both Moondream passes currently see the whole image. It can be added later as a crop stage before the passes.
* Moondream may ignore "ignore X" instructions, so some content leaks between passes. This is inherent to the model.
* Moondream tiles images automatically and is not configurable; for fine detail, resize the input sensibly before the call.

## 7. Reference implementation: `prompt_condenser_dual_pass.py`

````python
"""
Prompt-Condenser: dual-pass Moondream + WD14 pipeline with a selectable split mode.

Flow:
  1. WD14 tags the image; Moondream describes it in two passes (prose).
  2. Moondream prose is converted into WD-style (Danbooru) tags.
  3. WD14 tags adjust the Moondream tags (reconcile): WD14 wins on attributes
     (colors on clothing/hair/eyes), Moondream wins on pose and camera angle.
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
MOONDREAM_TEMPERATURE = 0.1
GEMMA_TEMPERATURE = 0.35
GEMMA_OLLAMA_MODEL = "huihui_ai/gemma-4-abliterated:12b"
GEMMA_CONTEXT_WINDOW = 131072   # 128K: the context the model page lists for the 12b tag
GEMMA_MAX_TOKENS = 768          # normal output cap
GEMMA_MAX_TOKENS_RETRY = 1024   # one retry with this cap if the first reply was cut off
MOONDREAM_RETRIES = 1           # re-ask once if a pass comes back empty or looks like a refusal
MOONDREAM_RETRY_TEMP_BUMP = 0.2 # retry slightly warmer so it does not repeat the same failure
TAG_ESCAPE_PARENTHESES = False  # True: write "(" as "\(" (A1111/ComfyUI prompt syntax)
WD14_GENERAL_THRESHOLD = 0.22
WD14_CHARACTER_THRESHOLD = 0.85

QUALITY_PREFIX = "masterpiece, best quality"  # configurable; set "" to disable

ZONE_FORMAT_HINT = (
    " Format each zone on its own line exactly as 'N. Zone name: text'. "
    "Plain text only, no markdown."
)

# ---------------------------------------------------------------------------
# PHASE 1: MOONDREAM PROMPTS + PER-MODE CONFIG
# ---------------------------------------------------------------------------
MOONDREAM_MODE_A_PASS_1 = (
    "Analyze physical space and body proportions and silhouette as visible. "
    "Ignore clothing and face. 6 zones: 1. Scene & Lighting 2. Camera & Depth "
    "3. Core Pose 4. Upper Body Shape 5. Lower Body Shape 6. Physical Anchors"
    + ZONE_FORMAT_HINT
)
MOONDREAM_MODE_A_PASS_2 = (
    "Analyze facial identity and wardrobe. Ignore body weight and environment. "
    "6 zones: 1. Head & Hair 2. Face & Eyes 3. Torso Clothing 4. Leg Clothing "
    "5. Footwear 6. Accessories & Props"
    + ZONE_FORMAT_HINT
)

MOONDREAM_MODE_B_PASS_1 = (
    "Analyze physical space and skeletal pose. Ignore clothing, face, and body "
    "thickness. 6 zones: 1. Scene & Lighting 2. Camera & Depth 3. Center of "
    "Gravity & Stance 4. Torso & Head Alignment 5. Limb Positioning "
    "6. Environmental Anchors"
    + ZONE_FORMAT_HINT
)
MOONDREAM_MODE_B_PASS_2 = (
    "Analyze physical body proportions, facial identity, and wardrobe. Ignore "
    "environment and underlying pose mechanics. 6 zones: 1. Head & Face "
    "2. Upper Body Proportions & Clothing 3. Lower Body Proportions & Clothing "
    "4. Legwear & Footwear 5. Skin & Fabric Physics 6. Accessories & Props"
    + ZONE_FORMAT_HINT
)

_COMMON_P1 = {
    "sitting", "standing", "kneeling", "squatting", "lying", "leaning",
    "holding", "grasping", "from below", "from above", "dynamic angle",
    "indoor", "outdoor", "bed", "chair", "floor", "depth of field",
    "foreshortening",
}

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


def prose_to_tags(text: str, vocab: set, max_n: int = 4) -> list:
    """Greedy longest-match of vocabulary tags inside Moondream prose."""
    tags, seen = [], set()
    flat = _strip_zone_headers(text).replace("\n", ", ")
    for clause in re.split(r"[,.;:]", flat):
        toks = _tokens(clause)
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
                if hit[0] not in seen:
                    seen.add(hit[0])
                    tags.append(hit[0])
                i += hit[1]
            else:
                i += 1
    return tags


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
               vocab: Optional[set] = None) -> dict:
    router = WD14DualPassRouter()
    wd_all = router.normalized_tags(raw_wd14_tags)
    wd1, wd2 = router.sort_tag_lists(raw_wd14_tags, mode)

    full_vocab = set(BUILTIN_VOCAB) | set(wd_all) | (vocab or set())
    md1 = prose_to_tags(pass1_txt, full_vocab)
    md2 = prose_to_tags(pass2_txt, full_vocab)

    drop_md, drop_wd = reconcile(md1 + md2, wd1 + wd2)
    if drop_md or drop_wd:
        log.info("reconcile dropped moondream=%s wd14=%s", sorted(drop_md), sorted(drop_wd))

    subject = [t for t in wd_all if _is_subject_tag(t)]
    wd1 = [t for t in wd1 if t not in drop_wd and not _is_subject_tag(t)]
    wd2 = [t for t in wd2 if t not in drop_wd and not _is_subject_tag(t)]
    md1 = [t for t in md1 if t not in drop_md]
    md2 = [t for t in md2 if t not in drop_md]

    return {
        # Moondream is the base; WD14 tags are added/adjusted on top.
        "pass1_tags": md1 + wd1,
        "pass2_tags": md2 + wd2,
        "subject_tags": subject,
        "dropped_moondream": sorted(drop_md),
        "dropped_wd14": sorted(drop_wd),
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
                       vocab: Optional[set] = None, keep_unmatched_prose: bool = False) -> dict:
        merged = merge_tags(raw_wd14_tags, pass1_txt, pass2_txt, mode, vocab)
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


def make_ollama_gemma_fn(model: str = GEMMA_OLLAMA_MODEL,
                         num_ctx: int = GEMMA_CONTEXT_WINDOW,
                         host: Optional[str] = None,
                         json_mode: bool = True,
                         timeout: float = 300) -> Callable:
    """Build a gemma_fn that talks to a local Ollama server (/api/chat).

    Returns (text, done_reason); done_reason "length" triggers the truncation retry.
    host defaults to $OLLAMA_HOST, then http://localhost:11434.
    """
    base = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base

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


def _valid_moondream(text: str) -> bool:
    t = (text or "").strip()
    return len(t) >= 20 and not _REFUSAL_RE.search(t[:80])


def _run_moondream(moondream_fn: Callable, image, prompt: str) -> str:
    """One Moondream pass; retry if empty or refusal-like, else give up with ''.

    An empty result means that pass contributes only its WD14 tags.
    """
    for attempt in range(MOONDREAM_RETRIES + 1):
        temp = MOONDREAM_TEMPERATURE + (MOONDREAM_RETRY_TEMP_BUMP if attempt else 0.0)
        out = moondream_fn(image, prompt, temp)
        if _valid_moondream(out):
            return out
        log.warning("Moondream pass %d/%d empty or refusal-like", attempt + 1, MOONDREAM_RETRIES + 1)
    return ""


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
# moondream_fn(image, prompt, temperature) -> str
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
    moondream_fn: Callable,
    gemma_fn: Optional[Callable] = None,
    single_pass_fn: Optional[Callable] = None,
    vocab_path: Optional[str] = None,
    keep_unmatched_prose: bool = False,
    gemma_vision_pass: bool = False,
) -> dict:
    if not enable_dual_pass:
        # dual_pass_mode and use_llm_synthesis are ignored on this path.
        if single_pass_fn is None:
            raise ValueError("enable_dual_pass is False but no single_pass_fn was provided")
        return single_pass_fn(image, raw_wd14_tags)

    cfg = _get_mode(dual_pass_mode)
    vocab = load_vocab(vocab_path) if vocab_path else None

    pass1_txt = _run_moondream(moondream_fn, image, cfg["pass1_prompt"])
    pass2_txt = _run_moondream(moondream_fn, image, cfg["pass2_prompt"])

    if use_llm_synthesis:
        if gemma_fn is None:
            gemma_fn = make_ollama_gemma_fn()  # huihui_ai/gemma-4-abliterated:12b via Ollama
        llm = AntigravityTagFilter_LLM()
        payload = llm.build_payload(raw_wd14_tags, pass1_txt, pass2_txt, dual_pass_mode,
                                    vocab, vision=gemma_vision_pass)
        text = llm.payload_to_text(payload)
        gemma_image = image if gemma_vision_pass else None
        raw, truncated = _call_gemma(gemma_fn, text, gemma_image, GEMMA_MAX_TOKENS)
        if truncated:
            log.warning("Gemma reply cut off at %d tokens; retrying with %d",
                        GEMMA_MAX_TOKENS, GEMMA_MAX_TOKENS_RETRY)
            raw, truncated = _call_gemma(gemma_fn, text, gemma_image, GEMMA_MAX_TOKENS_RETRY)
            if truncated:
                log.warning("Gemma reply still cut off at %d tokens", GEMMA_MAX_TOKENS_RETRY)
        parsed = _try_parse_gemma_json(raw)
        if parsed is None or not parsed["character_prompt"]:
            log.warning("Gemma output unusable; falling back to the mechanical merge")
            return AntigravityTagFilter_NoLLM().process_direct(
                raw_wd14_tags, pass1_txt, pass2_txt, dual_pass_mode, vocab, keep_unmatched_prose
            )
        return {
            "scene_prompt": _style_tags(parsed["scene_prompt"] or (QUALITY_PREFIX or "")),
            "character_prompt": _style_tags(parsed["character_prompt"]),
        }

    if gemma_vision_pass:
        log.warning("gemma_vision_pass ignored: use_llm_synthesis is False")
    return AntigravityTagFilter_NoLLM().process_direct(
        raw_wd14_tags, pass1_txt, pass2_txt, dual_pass_mode, vocab, keep_unmatched_prose
    )

````

## 8. Acceptance tests: `test_prompt_condenser.py`

````python
"""Acceptance tests for prompt_condenser_dual_pass.py. Run: python -m pytest test_prompt_condenser.py"""
import pytest

import prompt_condenser_dual_pass as p

WD = "1girl, solo, blue_dress, sitting, from_above, blonde_hair, cleaning, thick_thighs"
MD1 = "1. Scene & Lighting: a bright room with a couch\n2. Core Pose: she is standing, seen from below"
MD2 = "1. Hair: long red hair\n2. Clothing: a red dress and black boots"
VOCAB = {"red hair", "red dress", "black boots", "boots"}
OK_JSON = '{"scene_prompt":"s","character_prompt":"c"}'


def md_fn(image, prompt, temperature):
    return MD1 if "space" in prompt else MD2


def test_router_whole_words_and_underscores():
    r = p.WD14DualPassRouter()
    out = r.sort_tags("cleaning, depressed, sitting, thick_thighs, from_below, bedroom", "scene_shape")
    assert out["pass1_tags"] == "sitting, thick thighs, from below"
    assert "cleaning" in out["pass2_tags"] and "depressed" in out["pass2_tags"]


def test_mode_b_sends_body_shape_to_pass2():
    out = p.WD14DualPassRouter().sort_tags("sitting, thick_thighs, arms up", "scene_pose")
    assert out == {"pass1_tags": "sitting, arms up", "pass2_tags": "thick thighs"}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        p.WD14DualPassRouter().sort_tags("a", "bad")


def test_prose_to_tags_longest_match():
    tags = p.prose_to_tags("wearing a red dress and black boots", VOCAB | p.BUILTIN_VOCAB)
    assert tags == ["red dress", "black boots"]


def test_wd14_wins_colors_moondream_wins_pose():
    m = p.merge_tags(WD, MD1, MD2, "scene_shape", VOCAB)
    assert set(m["dropped_moondream"]) == {"red dress", "red hair"}
    assert set(m["dropped_wd14"]) == {"sitting", "from above"}


def test_no_llm_output_is_pure_tags():
    out = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_shape", VOCAB)
    assert out["scene_prompt"] == "masterpiece, best quality, couch, standing, from below, thick thighs"
    assert out["character_prompt"] == "1girl, solo, black boots, blue dress, blonde hair, cleaning"


def test_parser_handles_preamble_lists_and_bad_json():
    good = p.parse_gemma_json_response('Here:\n```json\n{"scene_prompt":"a","character_prompt":["b","c"]}\n```')
    assert good == {"scene_prompt": "a", "character_prompt": "b, c"}
    assert p.parse_gemma_json_response("[1,2]")["scene_prompt"] == p.QUALITY_PREFIX
    assert p.parse_gemma_json_response("not json")["character_prompt"] == "not json"


def test_dual_pass_off_uses_single_pass_fn():
    out = p.run_pipeline("img", WD, False, "scene_shape", True, md_fn,
                         single_pass_fn=lambda i, t: {"scene_prompt": "x", "character_prompt": "y"})
    assert out == {"scene_prompt": "x", "character_prompt": "y"}


def test_llm_path_retries_once_on_truncation_and_sends_image():
    calls = []

    def gemma(text, temp, max_tokens=0, image=None):
        calls.append((max_tokens, image))
        return ('{"scene_prompt":"s","char', "length") if len(calls) == 1 else (OK_JSON, "stop")

    out = p.run_pipeline("IMG", WD, True, "scene_pose", True, md_fn, gemma_fn=gemma, gemma_vision_pass=True)
    assert out == {"scene_prompt": "s", "character_prompt": "c"}
    assert calls == [(p.GEMMA_MAX_TOKENS, "IMG"), (p.GEMMA_MAX_TOKENS_RETRY, "IMG")]


def test_no_image_sent_without_vision_pass():
    seen = {}

    def gemma(text, temp, max_tokens=0, image=None):
        seen["image"] = image
        return OK_JSON

    p.run_pipeline("IMG", WD, True, "scene_shape", True, md_fn, gemma_fn=gemma)
    assert seen["image"] is None


def test_defaults():
    assert p.GEMMA_OLLAMA_MODEL == "huihui_ai/gemma-4-abliterated:12b"
    assert p.GEMMA_CONTEXT_WINDOW == 131072


def test_moondream_empty_pass_retries_then_uses_wd14_only():
    calls = []

    def empty(image, prompt, temperature):
        calls.append(temperature)
        return ""

    out = p.run_pipeline("img", "1girl, solo, sitting, blue_dress", True, "scene_shape", False, empty)
    assert len(calls) == 4  # 2 passes x (1 try + 1 retry)
    assert calls[1] == pytest.approx(p.MOONDREAM_TEMPERATURE + p.MOONDREAM_RETRY_TEMP_BUMP)
    assert out["character_prompt"] == "1girl, solo, blue dress"
    assert "sitting" in out["scene_prompt"]


def test_moondream_refusal_retries_and_recovers():
    seq = iter(["I'm sorry, I cannot help with that.", MD1, MD2])

    def flaky(image, prompt, temperature):
        return next(seq)

    out = p.run_pipeline("img", WD, True, "scene_shape", False, flaky)
    assert "standing" in out["scene_prompt"]


def test_attribute_groups_wd14_wins():
    m = p.merge_tags("short_hair, 1girl", "1. Hair: she has long hair and a smile", "", "scene_shape")
    assert "long hair" in m["dropped_moondream"]
    assert "short hair" in m["pass2_tags"] or "short hair" in m["pass1_tags"]


def test_unusable_gemma_output_falls_back_to_mechanical_merge():
    expected = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_shape")
    out = p.run_pipeline("img", WD, True, "scene_shape", True, md_fn,
                         gemma_fn=lambda text, temp, max_tokens=0, image=None: "total garbage")
    assert out == expected


def test_gemma_empty_character_prompt_falls_back():
    expected = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_pose")
    out = p.run_pipeline("img", WD, True, "scene_pose", True, md_fn,
                         gemma_fn=lambda text, temp, max_tokens=0, image=None:
                         '{"scene_prompt":"x","character_prompt":""}')
    assert out == expected


def test_parenthesis_escaping_option(monkeypatch):
    monkeypatch.setattr(p, "TAG_ESCAPE_PARENTHESES", True)
    out = p.AntigravityTagFilter_NoLLM().process_direct("kafuu chino (gochiusa), 1girl", MD1, MD2, "scene_shape")
    assert r"kafuu chino \(gochiusa\)" in out["character_prompt"]
    assert out["character_prompt"].count("\\(") == 1  # not double-escaped

````

## 9. Report back

When done, report: which files you changed, how the four toggles are exposed, which Moondream temperature argument you used, the `vocab_path` you found, and the test result.
