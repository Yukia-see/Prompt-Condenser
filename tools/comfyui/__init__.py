"""ComfyUI Prompt Condenser Extension.

Brings high-speed prompt condensation, checkpoint-specific token optimization,
framing conflict resolution, and ONNX anime captioning into ComfyUI.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Register server route for app -> ComfyUI UI queue trigger
try:
    import logging
    import server
    from aiohttp import web

    prompt_server = getattr(server, "PromptServer", None)
    if prompt_server is not None and hasattr(prompt_server, "instance") and prompt_server.instance is not None:
        @prompt_server.instance.routes.post("/prompt_condenser/queue")
        async def prompt_condenser_queue(request):
            prompt_server.instance.send_sync("prompt_condenser.queue", {})
            return web.json_response({"ok": True}, status=200)
except Exception as e:
    import logging
    logging.warning(f"[Prompt Condenser] Route registration skipped or failed: {e}")
