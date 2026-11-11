"""ComfyUI Prompt Condenser Extension.

Brings high-speed prompt condensation, checkpoint-specific token optimization,
framing conflict resolution, and ONNX anime captioning into ComfyUI.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
