"""ComfyUI nodes for Prompt Condenser integration.

Subprocess-based interface to the high-performance Rust Prompt Condenser engine.
Fast rule-based prompt condensation, auto-negatives, framing conflict resolution,
and LuCiD Visual Regional Prompting (VRP).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("ComfyUI-Prompt-Condenser")


def find_binary() -> str:
    """Find the prompt-condenser executable, prioritizing the newest build."""
    env_bin = os.environ.get("PROMPT_CONDENSER_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin

    candidates = [
        os.path.expanduser("~/Documents/Prompt Condensor/target/debug/prompt-condenser"),
        os.path.expanduser("~/Documents/Prompt Condensor/target/release/prompt-condenser"),
        shutil.which("prompt-condenser"),
        os.path.expanduser("~/.local/bin/prompt-condenser"),
        "/usr/local/bin/prompt-condenser",
        "/usr/bin/prompt-condenser",
    ]
    valid = [c for c in candidates if c and os.path.isfile(c) and os.access(c, os.X_OK)]
    if not valid:
        raise FileNotFoundError(
            "prompt-condenser binary not found. Build it with 'cargo build' or install it to ~/.local/bin/."
        )

    # Sort by mtime descending so newly compiled binaries take precedence over old releases
    valid.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return valid[0]


class PromptCondenserText:
    """Cleans, optimizes, and structures prompts for specific checkpoints with auto-negatives."""

    CHECKPOINTS = ["anima", "pony", "illustrious", "sdxl", "sd15", "flux"]
    VIEWS = ["none", "upper-body", "close-up", "front", "from-behind", "side", "full-body", "turnaround"]

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
                "lora_tags": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "dynamicPrompts": True
                }),
                "arranged": ("BOOLEAN", {"default": False}),
                "natural": ("BOOLEAN", {"default": False}),
                "target_view": (cls.VIEWS, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "changes_json", "tokens_before", "tokens_after")
    FUNCTION = "condense"
    CATEGORY = "Prompt Condenser"

    def condense(
        self,
        prompt: str,
        checkpoint: str,
        lora_tags: str = "",
        arranged: bool = False,
        natural: bool = False,
        target_view: str = "none",
    ) -> Tuple[str, str, str, int, int]:
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

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = res.stderr.strip() or res.stdout.strip()
            raise RuntimeError(f"Prompt Condenser failed (code {res.returncode}): {err}")

        data = json.loads(res.stdout)
        positive = data.get("positive", "")
        negative = data.get("negative", "")
        changes = json.dumps(data.get("changes", []), indent=2)
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
                "image": ("IMAGE",),
                "char3_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "char4_prompt": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": True}),
                "scene_lora": ("STRING", {"multiline": False, "default": ""}),
                "target_view": (PromptCondenserText.VIEWS, {"default": "none"}),
                "custom_regions": ("STRING", {"default": "", "placeholder": "e.g. 0,0,0.5,1;0.5,0,0.5,1"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_positive", "scene_negative", "vrp_state_json", "summary")
    FUNCTION = "generate_vrp"
    CATEGORY = "Prompt Condenser"

    def generate_vrp(
        self,
        scene_prompt: str,
        char1_prompt: str,
        char2_prompt: str,
        checkpoint: str = "anima",
        image: torch.Tensor | None = None,
        char3_prompt: str = "",
        char4_prompt: str = "",
        scene_lora: str = "",
        target_view: str = "none",
        custom_regions: str = "",
    ) -> Tuple[str, str, str, str]:
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
                raise RuntimeError(f"Prompt Condenser VRP failed: {err}")
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
                raise RuntimeError(f"Prompt Condenser caption failed: {res.stderr}")

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
