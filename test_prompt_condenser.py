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
