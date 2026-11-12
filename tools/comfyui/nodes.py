"""ComfyUI nodes for Prompt Condenser integration.

Subprocess-based interface to the high-performance Rust Prompt Condenser engine.
Fast rule-based prompt condensation, auto-negatives, framing conflict resolution,
and LuCiD Visual Regional Prompting (VRP).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("ComfyUI-Prompt-Condenser")


def read_live_prompt_cache(max_age_sec: float = 600.0) -> Tuple[Dict[str, Any] | None, str | None]:
    """Read ~/.cache/prompt-condenser/live_prompt.json if available and fresh (< max_age_sec)."""
    cache_path = os.path.expanduser("~/.cache/prompt-condenser/live_prompt.json")
    if not os.path.isfile(cache_path):
        return (None, "live_prompt.json not found")
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_ts = data.get("timestamp") or os.path.getmtime(cache_path)
        age = time.time() - float(file_ts)
        if age > max_age_sec:
            return (None, f"live_prompt.json is stale ({int(age)}s old > {int(max_age_sec)}s)")
        return (data, None)
    except Exception as e:
        return (None, f"failed to read live_prompt.json: {e}")


@functools.lru_cache(maxsize=1)
def find_binary() -> str:
    """Find the prompt-condenser executable.
    
    Checks environment variable, binary_path.txt, development builds, and system-installed binaries.
    Prioritizes binaries with --vrp and target_view support, sorting by newest mtime.
    Results are cached to avoid repeated probe subprocess overhead.
    """
    env_bin = os.environ.get("PROMPT_CONDENSER_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin

    # Check companion binary_path.txt if installed by prompt-condenser UI/CLI
    bin_txt = os.path.join(os.path.dirname(__file__), "binary_path.txt")
    if os.path.isfile(bin_txt):
        try:
            with open(bin_txt, "r", encoding="utf-8") as f:
                p = f.read().strip()
                if p and os.path.isfile(p) and os.access(p, os.X_OK):
                    probe = subprocess.run(
                        [p, "--prompt", "", "--vrp", "--char", "1girl", "--char", "1boy"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if probe.returncode == 0:
                        return p
        except Exception:
            pass

    potential = [
        os.path.expanduser("~/Documents/Prompt Condensor/target/release/prompt-condenser"),
        os.path.expanduser("~/Documents/Prompt Condensor/target/debug/prompt-condenser"),
        shutil.which("prompt-condenser"),
        os.path.expanduser("~/.local/bin/prompt-condenser"),
        "/usr/local/bin/prompt-condenser",
        "/usr/bin/prompt-condenser",
    ]
    seen = set()
    valid = []
    for p in potential:
        if p:
            real = os.path.realpath(p)
            if real not in seen and os.path.isfile(real) and os.access(real, os.X_OK):
                seen.add(real)
                valid.append(real)

    if not valid:
        raise FileNotFoundError(
            "prompt-condenser binary not found. Build it with 'cargo build --release' or install it to ~/.local/bin/ or /usr/bin/."
        )

    # Sort by mtime descending (most recently compiled or installed wins)
    valid.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Probe candidates for modern CLI flag support (--vrp)
    for candidate in valid:
        try:
            res = subprocess.run(
                [candidate, "--prompt", "", "--vrp", "--char", "1girl", "--char", "1boy"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                return candidate
        except Exception:
            continue

    # Fall back to newest valid candidate if probe fails
    return valid[0]


def resolve_bust_index(val: Any) -> int:
    """Resolve bust choice to a 0-6 index."""
    if not val or val == "none":
        return 0
    if isinstance(val, int) or (isinstance(val, str) and val.isdigit()):
        return max(0, min(6, int(val)))
    val_str = str(val).lower().strip()
    for i, name in enumerate(PromptCondenserText.BUSTS):
        if val_str == name or val_str.startswith(name):
            return i
    return 0


def resolve_frame_index(val: Any) -> int:
    """Resolve frame choice to a 0-6 index."""
    if not val or val == "none":
        return 0
    if isinstance(val, int) or (isinstance(val, str) and val.isdigit()):
        return max(0, min(6, int(val)))
    val_str = str(val).lower().strip()
    for i, name in enumerate(PromptCondenserText.FRAMES):
        if val_str == name or val_str.startswith(name):
            return i
    if "thick" in val_str:
        return 3
    if "pear" in val_str:
        return 4
    return 0


class PromptCondenserText:
    """Cleans, optimizes, and structures prompts for specific checkpoints with auto-negatives."""

    CHECKPOINTS = ["anima", "pony", "illustrious", "sdxl", "sd15", "flux"]
    VIEWS = ["none", "upper-body", "close-up", "front", "from-behind", "side", "full-body", "turnaround"]
    BUSTS = ["none", "flat", "small", "medium", "large", "huge", "gigantic"]
    FRAMES = ["none", "petite", "shortstack", "thick", "pear", "hourglass", "muscular"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "1girl, solo, anime aesthetic",
                    "dynamicPrompts": True
                }),
                "checkpoint": (cls.CHECKPOINTS, {"default": "anima"}),
            },
            "optional": {
                "live_stream": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stream active prompt directly from Prompt Condenser desktop app"
                }),
                "lora_tags": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "dynamicPrompts": True
                }),
                "arranged": ("BOOLEAN", {"default": False}),
                "natural": ("BOOLEAN", {"default": False}),
                "target_view": (cls.VIEWS, {"default": "none"}),
                "bust": (cls.BUSTS, {"default": "none"}),
                "frame": (cls.FRAMES, {"default": "none"}),
                "exaggerated_proportions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow exaggerated/stylized cartoon proportions for shortstack",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "changes_json", "tokens_before", "tokens_after")
    FUNCTION = "condense"
    CATEGORY = "Prompt Condenser"

    @classmethod
    def IS_CHANGED(cls, prompt: str, checkpoint: str, live_stream: bool = False, **kwargs):
        if live_stream:
            cache_path = os.path.expanduser("~/.cache/prompt-condenser/live_prompt.json")
            try:
                if os.path.isfile(cache_path):
                    return os.path.getmtime(cache_path)
            except Exception:
                pass
            return float("nan")
        return False

    def condense(
        self,
        prompt: str,
        checkpoint: str,
        live_stream: bool = False,
        lora_tags: str = "",
        arranged: bool = False,
        natural: bool = False,
        target_view: str = "none",
        bust: str = "none",
        frame: str = "none",
        exaggerated_proportions: bool = False,
    ) -> Tuple[str, str, str, int, int]:
        fallback_msg = None
        if live_stream:
            live_data, err = read_live_prompt_cache(600.0)
            if live_data:
                positive = live_data.get("positive", "")
                negative = live_data.get("negative", "")
                changes = json.dumps(live_data.get("changes", []), indent=2)
                tokens_before = int(live_data.get("tokens_before", 0))
                tokens_after = int(live_data.get("tokens_after", 0))
                return (positive, negative, changes, tokens_before, tokens_after)
            fallback_msg = err
            logger.warning(f"Prompt Condenser live_stream fallback to local input: {fallback_msg}")

        prompt = prompt.strip()
        if not prompt:
            return ("", "", "[]", 0, 0)

        binary = find_binary()
        cmd = [binary, "--prompt", prompt, "--checkpoint", checkpoint.lower()]

        if lora_tags.strip():
            cmd.extend(["--lora", lora_tags.strip()])
        if arranged:
            cmd.append("--arranged")
        if natural:
            cmd.append("--natural")
        if target_view and target_view != "none":
            cmd.extend(["--view", target_view])
        b_idx = resolve_bust_index(bust)
        if b_idx > 0:
            cmd.extend(["--bust", str(b_idx)])
        f_idx = resolve_frame_index(frame)
        if f_idx > 0:
            cmd.extend(["--frame", str(f_idx)])
        if exaggerated_proportions:
            cmd.append("--exaggerated")

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = res.stderr.strip() or res.stdout.strip()
            raise RuntimeError(f"Prompt Condenser failed (binary: {binary}, code {res.returncode}): {err}")

        data = json.loads(res.stdout)
        positive = data.get("positive", "")
        negative = data.get("negative", "")
        changes_list = data.get("changes", [])
        if fallback_msg:
            changes_list.insert(0, {
                "kind": "LiveStreamFallback",
                "description": f"Fallback to node inputs ({fallback_msg})"
            })
        changes = json.dumps(changes_list, indent=2)
        tokens_before = int(data.get("tokens_before", 0))
        tokens_after = int(data.get("tokens_after", 0))

        return (positive, negative, changes, tokens_before, tokens_after)


class PromptCondenserVRP:
    """Multi-Character Visual Regional Prompting (VRP) Generator.

    Decomposes multi-person scenes into clean global scene conditioning and
    character-isolated regional prompts/masks for LuCiD VRP.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "scene_prompt": ("STRING", {
                    "multiline": True,
                    "default": "sunset, beach, warm lighting",
                    "dynamicPrompts": True
                }),
                "char1_prompt": ("STRING", {
                    "multiline": True,
                    "default": "1girl, solo, red hair, blue dress",
                    "dynamicPrompts": True
                }),
                "char2_prompt": ("STRING", {
                    "multiline": True,
                    "default": "1girl, solo, blonde hair, white dress",
                    "dynamicPrompts": True
                }),
                "checkpoint": (PromptCondenserText.CHECKPOINTS, {"default": "anima"}),
            },
            "optional": {
                "live_stream": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stream active VRP regions directly from Prompt Condenser desktop app"
                }),
                "image": ("IMAGE",),
                "char3_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "char4_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "scene_lora": ("STRING", {"multiline": False, "default": ""}),
                "target_view": (PromptCondenserText.VIEWS, {"default": "none"}),
                "custom_regions": ("STRING", {"default": "", "placeholder": "e.g. 0,0,0.5,1;0.5,0,0.5,1"}),
                "bust": (PromptCondenserText.BUSTS, {"default": "none"}),
                "frame": (PromptCondenserText.FRAMES, {"default": "none"}),
                "exaggerated_proportions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow exaggerated/stylized cartoon proportions for shortstack",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_positive", "scene_negative", "vrp_state_json", "summary")
    FUNCTION = "generate_vrp"
    CATEGORY = "Prompt Condenser"

    @classmethod
    def IS_CHANGED(cls, live_stream: bool = False, **kwargs):
        if live_stream:
            cache_path = os.path.expanduser("~/.cache/prompt-condenser/live_prompt.json")
            try:
                if os.path.isfile(cache_path):
                    return os.path.getmtime(cache_path)
            except Exception:
                pass
            return float("nan")
        return False

    def generate_vrp(
        self,
        scene_prompt: str,
        char1_prompt: str,
        char2_prompt: str,
        checkpoint: str = "anima",
        live_stream: bool = False,
        image: torch.Tensor | None = None,
        char3_prompt: str = "",
        char4_prompt: str = "",
        scene_lora: str = "",
        target_view: str = "none",
        custom_regions: str = "",
        bust: str = "none",
        frame: str = "none",
        exaggerated_proportions: bool = False,
    ) -> Tuple[str, str, str, str]:
        fallback_msg = None
        if live_stream:
            live_data, err = read_live_prompt_cache(600.0)
            vrp_state_raw = live_data.get("vrp_state") if live_data else None
            if live_data and vrp_state_raw and vrp_state_raw not in ("{}", '""', ""):
                scene_pos = live_data.get("positive", "")
                scene_neg = live_data.get("negative", "")
                vrp_state = str(vrp_state_raw)
                regions_source = live_data.get("regions_source", "live_stream")
                summary = f"VRP Streamed from Prompt Condenser (source: {regions_source})"
                return (scene_pos, scene_neg, vrp_state, summary)
            elif live_data and not vrp_state_raw:
                fallback_msg = "app is not on a multi-character tab"
            else:
                fallback_msg = err
            logger.warning(f"Prompt Condenser VRP live_stream fallback: {fallback_msg}")
        chars = [c.strip() for c in [char1_prompt, char2_prompt, char3_prompt, char4_prompt] if c.strip()]
        if len(chars) < 2:
            raise ValueError("PromptCondenserVRP requires at least 2 characters (char1 and char2).")

        binary = find_binary()
        cmd = [
            binary,
            "--prompt", scene_prompt.strip(),
            "--checkpoint", checkpoint.lower(),
            "--vrp",
        ]

        for c in chars:
            cmd.extend(["--char", c])

        if scene_lora.strip():
            cmd.extend(["--lora", scene_lora.strip()])
        if target_view and target_view != "none":
            cmd.extend(["--view", target_view])
        if custom_regions.strip():
            cmd.extend(["--regions", custom_regions.strip()])
        b_idx = resolve_bust_index(bust)
        if b_idx > 0:
            cmd.extend(["--bust", str(b_idx)])
        f_idx = resolve_frame_index(frame)
        if f_idx > 0:
            cmd.extend(["--frame", str(f_idx)])
        if exaggerated_proportions:
            cmd.append("--exaggerated")

        temp_img_path = None
        if image is not None:
            # Save frame for person detector bounding boxes
            frame = image[0] if image.ndim == 4 else image
            np_frame = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_frame)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_img_path = f.name
                pil_img.save(temp_img_path)
            cmd.extend(["--image", temp_img_path])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                err = res.stderr.strip() or res.stdout.strip()
                raise RuntimeError(f"Prompt Condenser VRP failed (binary: {binary}): {err}")
        finally:
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass

        data = json.loads(res.stdout)
        scene_pos = data.get("positive", "")
        scene_neg = data.get("negative", "")
        vrp_state = data.get("vrp_state", "{}")
        regions_source = data.get("regions_source", "strips")
        warnings = data.get("warnings", [])

        summary = f"VRP Generated ({len(chars)} characters, source: {regions_source})"
        if fallback_msg:
            summary += f" | LiveStream fallback: {fallback_msg}"
        if warnings:
            summary += f" | Warnings: {'; '.join(warnings)}"

        return (scene_pos, scene_neg, vrp_state, summary)


class PromptCondenserCaption:
    """Fast Danbooru image tagger using Prompt Condenser's WD-EVA02 ONNX + visual rules."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.05}),
            },
            "optional": {
                "checkpoint": (PromptCondenserText.CHECKPOINTS, {"default": "anima"}),
                "auto_condense": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("raw_caption", "condensed_prompt")
    FUNCTION = "caption"
    CATEGORY = "Prompt Condenser"

    def caption(
        self,
        image: torch.Tensor,
        threshold: float = 0.30,
        checkpoint: str = "anima",
        auto_condense: bool = True
    ) -> Tuple[str, str]:
        # Convert first frame from ComfyUI tensor [B, H, W, C] to PIL Image
        if image.ndim == 4:
            frame = image[0]
        else:
            frame = image

        np_frame = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_frame)

        binary = find_binary()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = os.path.join(tmp_dir, "input.png")
            pil_img.save(temp_path)

            cmd = [
                binary,
                "--caption", tmp_dir,
                "--threshold", f"{threshold:.2f}",
                "--overwrite",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Prompt Condenser caption failed (binary: {binary}): {res.stderr}")

            txt_path = os.path.join(tmp_dir, "input.txt")
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"Expected caption file {txt_path} not created.")

            with open(txt_path, "r", encoding="utf-8") as f:
                raw_caption = f.read().strip()

        if not auto_condense or not raw_caption:
            return (raw_caption, raw_caption)

        # Condense the generated caption through text optimizer
        text_node = PromptCondenserText()
        pos, _, _, _, _ = text_node.condense(prompt=raw_caption, checkpoint=checkpoint)
        return (raw_caption, pos)


NODE_CLASS_MAPPINGS = {
    "PromptCondenserText": PromptCondenserText,
    "PromptCondenserVRP": PromptCondenserVRP,
    "PromptCondenserCaption": PromptCondenserCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCondenserText": "Prompt Condenser (Text Optimizer)",
    "PromptCondenserVRP": "Prompt Condenser (Multi-Char VRP)",
    "PromptCondenserCaption": "Prompt Condenser (Fast Tagger & Caption)",
}
