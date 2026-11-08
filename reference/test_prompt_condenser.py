"""Acceptance tests for prompt_condenser_dual_pass.py.

Run with pytest:           python -m pytest test_prompt_condenser.py
Or with no dependencies:   python3 test_prompt_condenser.py
"""
import contextlib

try:
    import pytest
except ImportError:  # minimal stand-in so the suite runs on a bare Python install
    class _Approx:
        def __init__(self, value, tol=1e-6):
            self.value, self.tol = value, tol

        def __eq__(self, other):
            return abs(other - self.value) <= self.tol

    class pytest:  # noqa: N801
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"did not raise {exc.__name__}")

        @staticmethod
        def approx(value):
            return _Approx(value)

import prompt_condenser_dual_pass as p

WD = "1girl, solo, blue_dress, sitting, from_above, blonde_hair, cleaning, thick_thighs"
MD1 = "1. Scene & Lighting: a bright room with a couch\n2. Core Pose: she is standing, seen from below"
MD2 = "1. Hair: long red hair\n2. Clothing: a red dress and black boots"
VOCAB = {"red hair", "red dress", "black boots", "boots"}
OK_JSON = '{"scene_prompt":"s","character_prompt":"c"}'


def prompts(out):
    return {k: out[k] for k in ("scene_prompt", "character_prompt")}


def md_fn(image, prompt, temperature):
    return MD1 if "space" in prompt else MD2


def test_router_whole_words_and_underscores():
    r = p.WD14DualPassRouter()
    # "bedhead" must NOT match the scene keyword "bed" (whole-word matching);
    # "bedroom" IS a scene keyword in its own right and routes to Pass 1.
    out = r.sort_tags("cleaning, depressed, sitting, thick_thighs, from_below, bedhead, bedroom", "scene_shape")
    assert out["pass1_tags"] == "sitting, thick thighs, from below, bedroom"
    assert "cleaning" in out["pass2_tags"] and "depressed" in out["pass2_tags"]
    assert "bedhead" in out["pass2_tags"]


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
    assert set(m["dropped_vlm"]) == {"red dress", "red hair"}
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
    assert prompts(out) == {"scene_prompt": "s", "character_prompt": "c"}
    assert out["warnings"] == []
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
    assert calls[1] == pytest.approx(p.VLM_TEMPERATURE + p.VLM_RETRY_TEMP_BUMP)
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
    assert "long hair" in m["dropped_vlm"]
    assert "long hair" in m["dropped_moondream"]
    assert "short hair" in m["pass2_tags"] or "short hair" in m["pass1_tags"]


def test_unusable_gemma_output_falls_back_to_mechanical_merge():
    expected = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_shape")
    out = p.run_pipeline("img", WD, True, "scene_shape", True, md_fn,
                         gemma_fn=lambda text, temp, max_tokens=0, image=None: "total garbage")
    assert prompts(out) == expected
    assert out["warnings"]


def test_gemma_empty_character_prompt_falls_back():
    expected = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_pose")
    out = p.run_pipeline("img", WD, True, "scene_pose", True, md_fn,
                         gemma_fn=lambda text, temp, max_tokens=0, image=None:
                         '{"scene_prompt":"x","character_prompt":""}')
    assert prompts(out) == expected
    assert out["warnings"]


def test_parenthesis_escaping_option(monkeypatch):
    monkeypatch.setattr(p, "TAG_ESCAPE_PARENTHESES", True)
    out = p.AntigravityTagFilter_NoLLM().process_direct("kafuu chino (gochiusa), 1girl", MD1, MD2, "scene_shape")
    assert r"kafuu chino \(gochiusa\)" in out["character_prompt"]
    assert out["character_prompt"].count("\\(") == 1  # not double-escaped


def test_scene_keywords_route_common_background_tags():
    out = p.WD14DualPassRouter().sort_tags(
        "simple_background, white_background, night, cowboy_shot, 1girl, blue_dress", "scene_shape")
    assert out["pass1_tags"] == "simple background, white background, night, cowboy shot"
    assert out["pass2_tags"] == "1girl, blue dress"


def test_negation_drops_tag_and_scope_ends_at_but():
    a = p.analyze_prose("She has no glasses but a red hat, not wearing boots",
                        {"glasses", "red hat", "hat", "boots"})
    assert "glasses" in a["negated"] and "boots" in a["negated"]
    assert a["tags"] == ["red hat"]


def test_bare_is_not_a_negator():
    assert p.prose_to_tags("bare shoulders", {"bare shoulders"}) == ["bare shoulders"]


def test_hedged_tag_needs_wd14_confirmation():
    md1 = "1. Face: she appears to be wearing glasses and has a smile"
    kept = p.merge_tags("1girl, glasses", md1, "", "scene_shape", {"glasses", "smile"})
    dropped = p.merge_tags("1girl", md1, "", "scene_shape", {"glasses", "smile"})
    def all_tags(m):
        return m["pass1_tags"] + m["pass2_tags"]

    assert "glasses" in all_tags(kept)
    assert "glasses" not in all_tags(dropped) and "glasses" in dropped["dropped_hedged"]
    assert "smile" in all_tags(dropped)


def test_hedge_and_negation_scope_ends_at_new_statement():
    v = {"glasses", "hat", "smile"}
    # hedge still covers a coordinated noun ("... glasses and a hat")
    a = p.analyze_prose("she appears to be wearing glasses and a hat", v)
    assert a["weak"] == {"glasses", "hat"} and not a["strong"]
    # ... but not a new statement ("and has a smile")
    b = p.analyze_prose("she appears to be wearing glasses and has a smile", v)
    assert b["weak"] == {"glasses"} and b["strong"] == {"smile"}
    # same for negation
    c = p.analyze_prose("she has no glasses and has a hat", v)
    assert c["negated"] == ["glasses"] and c["tags"] == ["hat"]


def test_gemma_exception_falls_back_with_warning():
    def boom(text, temp, max_tokens=0, image=None):
        raise ConnectionRefusedError("connection refused")

    expected = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_shape")
    out = p.run_pipeline("img", WD, True, "scene_shape", True, md_fn, gemma_fn=boom)
    assert prompts(out) == expected
    assert any("connection refused" in w for w in out["warnings"])


def test_moondream_exception_becomes_wd14_only_pass_with_warning():
    def boom(image, prompt, temperature):
        raise RuntimeError("model crashed")

    out = p.run_pipeline("img", "1girl, solo, sitting, blue_dress", True, "scene_shape", False, boom)
    assert out["character_prompt"] == "1girl, solo, blue dress"
    assert len([w for w in out["warnings"] if "model crashed" in w]) == 2


def test_wrong_signature_wrapper_still_raises():
    with pytest.raises(TypeError):
        p.run_pipeline("img", WD, True, "scene_shape", True, md_fn, gemma_fn=lambda text: "x")


def test_ollama_down_uses_mechanical_merge(monkeypatch):
    monkeypatch.setattr(p, "check_ollama", lambda *a, **k: (False, "Ollama not reachable at http://x"))
    expected = p.AntigravityTagFilter_NoLLM().process_direct(WD, MD1, MD2, "scene_shape")
    out = p.run_pipeline("img", WD, True, "scene_shape", True, md_fn)
    assert prompts(out) == expected
    assert out["warnings"] == ["Ollama not reachable at http://x"]


def test_check_ollama_reports_missing_model_and_unreachable():
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"models": [{"name": "other:7b"}]}'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ok, msg = p.check_ollama(host=f"127.0.0.1:{srv.server_port}")
        assert not ok and "ollama pull" in msg
        ok2, _ = p.check_ollama(model="other:7b", host=f"127.0.0.1:{srv.server_port}")
        assert ok2
    finally:
        srv.shutdown()
    ok3, msg3 = p.check_ollama(host="127.0.0.1:1", timeout=1)
    assert not ok3 and "not reachable" in msg3


def test_crop_artifacts_are_strictly_rejected():
    candidate_crop_tags = [
        ("close-up", 0.95),
        ("portrait", 0.90),
        ("solo focus", 0.88),
        ("breast focus", 0.85),
        ("eye focus", 0.70),
        ("head out of frame", 0.80),
        ("cropped legs", 0.75),
        ("out of frame", 0.90),
        ("feet out of frame", 0.65),
        ("partially submerged", 0.60),
        ("full body", 0.92),
        ("cowboy shot", 0.85),
        ("hair ornament", 0.85),
        ("holding spoon", 0.75),
    ]
    accepted = p.filter_crop_tags(candidate_crop_tags, crop_type="character_zone")
    accepted_names = [t[0] for t in accepted]
    assert "hair ornament" in accepted_names
    assert "holding spoon" in accepted_names
    assert "close-up" not in accepted_names
    assert "portrait" not in accepted_names
    assert "solo focus" not in accepted_names
    assert "breast focus" not in accepted_names
    assert "eye focus" not in accepted_names
    assert "head out of frame" not in accepted_names
    assert "cropped legs" not in accepted_names
    assert "out of frame" not in accepted_names
    assert "feet out of frame" not in accepted_names
    assert "partially submerged" not in accepted_names
    assert "full body" not in accepted_names
    assert "cowboy shot" not in accepted_names


def test_ratings_and_characters_rejected_from_crops():
    tags = [
        ("general", 0.99),
        ("sensitive", 0.95),
        ("questionable", 0.90),
        ("explicit", 0.85),
        ("pyra (xenoblade)", 0.95),
        ("mythra (xenoblade)", 0.90),
        ("golden tiara", 0.80),
    ]
    accepted = p.filter_crop_tags(
        tags,
        crop_type="character_zone",
        character_names={"pyra (xenoblade)", "mythra (xenoblade)"},
    )
    accepted_names = [t[0] for t in accepted]
    assert accepted_names == ["golden tiara"]


def test_body_sizes_and_counts_rejected_from_crops():
    tags = [
        ("solo", 0.95),
        ("1girl", 0.98),
        ("large breasts", 0.85),
        ("flat chest", 0.80),
        ("ribbon choker", 0.75),
    ]
    accepted = p.filter_crop_tags(tags, crop_type="character_zone")
    accepted_names = [t[0] for t in accepted]
    assert accepted_names == ["ribbon choker"]


def test_micro_attribute_threshold_enforced():
    tags = [
        ("butterfly hairpin", 0.54),
        ("silver brooch", 0.55),
        ("earrings", 0.60),
    ]
    accepted = p.filter_crop_tags(tags, crop_type="character_zone", min_attribute_threshold=0.55)
    accepted_names = [t[0] for t in accepted]
    assert "butterfly hairpin" not in accepted_names
    assert "silver brooch" in accepted_names
    assert "earrings" in accepted_names


def test_quadrant_crop_concrete_objects_and_generic_backgrounds():
    quad_tags = [
        ("potted plant", 0.45),
        ("bookshelf", 0.40),
        ("simple background", 0.90),
        ("white background", 0.85),
        ("monochrome", 0.80),
        ("solo focus", 0.70),
        ("close-up", 0.75),
        ("wooden chair", 0.50),
    ]
    accepted = p.filter_crop_tags(quad_tags, crop_type="quadrant")
    accepted_names = [t[0] for t in accepted]
    assert "potted plant" in accepted_names
    assert "bookshelf" in accepted_names
    assert "wooden chair" in accepted_names
    assert "simple background" not in accepted_names
    assert "white background" not in accepted_names
    assert "monochrome" not in accepted_names
    assert "solo focus" not in accepted_names
    assert "close-up" not in accepted_names


def test_merge_crop_tags_deduplicates_and_takes_max_score():
    base = [("blue dress", 0.60), ("black boots", 0.70)]
    crop = [("blue dress", 0.85), ("black boots", 0.65), ("silver necklace", 0.80)]
    merged = p.merge_crop_tags(base, crop, crop_type="character_zone")
    merged_dict = dict(merged)
    assert merged_dict["blue dress"] == 0.85
    assert merged_dict["black boots"] == 0.70
    assert merged_dict["silver necklace"] == 0.80
    assert len(merged) == 3


def test_pure_zone_boxes_skips_lower_zone_for_half_body():
    # 1. Full body character: 1000x1000 image, figure box from 100 to 950 (height = 850, ratio = 0.85)
    boxes = p.compute_zone_boxes(1000.0, 1000.0, 200.0, 100.0, 500.0, 950.0)
    names = [b["name"] for b in boxes]
    assert names == ["Upper", "Middle", "Lower"]

    # 2. Half body touching bottom: box from 100 to 1000 (height = 500, ratio = 0.50, touches bottom edge >= 980)
    half_boxes = p.compute_zone_boxes(1000.0, 1000.0, 200.0, 500.0, 500.0, 1000.0)
    half_names = [b["name"] for b in half_boxes]
    assert "Lower" not in half_names
    assert half_names == ["Upper", "Middle"]

    # 3. Small head / bust: height = 150 (ratio = 0.15)
    bust_boxes = p.compute_zone_boxes(1000.0, 1000.0, 300.0, 100.0, 500.0, 250.0)
    bust_names = [b["name"] for b in bust_boxes]
    assert bust_names == ["Upper"]

    # 4. Clipping: coordinates clamped strictly within [0, img_w] x [0, img_h]
    clipped = p.compute_zone_boxes(1000.0, 1000.0, -50.0, -20.0, 1050.0, 1020.0)
    for z in clipped:
        assert z["x0"] >= 0.0 and z["x1"] <= 1000.0
        assert z["y0"] >= 0.0 and z["y1"] <= 1000.0


if __name__ == "__main__":  # zero-dependency runner
    import inspect
    import sys
    import traceback

    class _MonkeyPatch:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)

    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        mp = _MonkeyPatch()
        try:
            fn(*([mp] if "monkeypatch" in inspect.signature(fn).parameters else []))
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
