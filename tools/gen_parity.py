"""Generate golden cases from the Python reference for tests/parity.rs."""
import json, random, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
import prompt_condenser_dual_pass as p

MODES = list(p.DUAL_PASS_MODES)
random.seed(20260107)

wd_pool = ["1girl", "solo", "2boys", "multiple_girls", "blue_dress", "red_dress", "dress", "sitting", "standing",
           "from_above", "from_below", "blonde_hair", "black_hair", "long_hair", "short_hair", "very_long_hair",
           "thick_thighs", "large_breasts", "small_breasts", "cleaning", "depressed", "bedhead", "bedroom", "bed",
           "simple_background", "white_background", "night", "cowboy_shot", "full_body", "close-up", "looking_at_viewer",
           "glasses", "hat", "red_hat", "black_boots", "boots", "barefoot", "smile", "arms_up", "crossed_legs",
           "holding_cup", "cup", "indoors", "window", "sky", "pov", "kafuu_chino_(gochiusa)", "blue_eyes", "red_eyes",
           "denim_shorts", "thighhighs", "white_thighhighs", "skyscraper", "ponytail", "twintails", "from_side"]
prose_pool = [
    "a bright room with a couch", "she is standing, seen from below", "long red hair", "a red dress and black boots",
    "she has no glasses but a red hat", "not wearing boots", "appears to be wearing glasses and has a smile",
    "possibly a blue dress", "bare shoulders and bare legs", "sitting on the bed in a bedroom",
    "she holds a cup in her left hand", "mid-air jump with arms up", "wearing white thighhighs and no shoes",
    "short hair tied in a ponytail", "seems to have twintails, but the hair is long", "a window behind her, night sky",
    "standing barefoot on the floor", "thick thighs and large breasts", "black hair and blue eyes",
    "wearing a hat and glasses and a smile", "lacks a hat; has glasses", "I'm sorry, I cannot describe this",
    "an open mouth", "looking at viewer, from above", "city skyline, cloudy sky", "no humans in this scene",
]
vocab_pool = ["red hair", "black boots", "red dress", "blue dress", "white thighhighs", "blue eyes", "red eyes",
              "black hair", "long hair", "short hair", "twintails", "ponytail", "barefoot", "mid-air", "left hand",
              "kafuu chino (gochiusa)", "denim shorts", "open mouth", "city skyline", "no humans"]


def zone_text(n):
    lines = []
    for i in range(1, n + 1):
        style = random.choice(["plain", "bold", "dash", "paren", "none"])
        body = random.choice(prose_pool) + random.choice(["", ".", ", " + random.choice(prose_pool)])
        name = random.choice(["Scene & Lighting", "Core Pose", "Head & Hair", "Mid-air Pose", "Torso Clothing"])
        if style == "plain":   lines.append(f"{i}. {name}: {body}")
        elif style == "bold":  lines.append(f"{i}. **{name}**: {body}")
        elif style == "dash":  lines.append(f"{i}. {name} - {body}")
        elif style == "paren": lines.append(f"{i}) {name}: {body}")
        else:                  lines.append(body)
    return "\n".join(lines)


cases = {
    "router": [], "analyze": [], "merge": [], "nollm": [], "parse": [],
    "flatten": [], "payload": [], "style": [], "filter_crop": [], "zone_boxes": [],
    "layer_conflicts": []
}

# ---- hand-picked
hand_wd = [WD for WD in [
    "1girl, solo, blue_dress, sitting, from_above, blonde_hair, cleaning, thick_thighs",
    "cleaning, depressed, sitting, thick_thighs, from_below, bedhead, bedroom",
    "", "  ,  , ", "1girl", "kafuu_chino_(gochiusa), 1girl, solo, close-up",
]]
for w in hand_wd:
    for m in MODES:
        cases["router"].append({"wd": w, "mode": m})

hand_raw = [
    '{"scene_prompt":"a","character_prompt":["b","c"]}',
    'Here:\n```json\n{"scene_prompt":"a","character_prompt":["b","c"]}\n```',
    "[1,2]", "not json", "", "{}", '{"scene_prompt":"", "character_prompt":"x, x, y."}',
    '{"scene_prompt":"s","char', '```json\n{"scene_prompt":"s","character_prompt":"c"}', 'prefix {"scene_prompt": 5, "character_prompt": null} suffix',
    '{"scene_prompt":["a",1,true,null,""],"character_prompt":" , "}', '{"scene_prompt":"é \\u00e9 \\ud83d\\ude00","character_prompt":"ok"}',
    '```\n{"scene_prompt":"s","character_prompt":"c"}\n```', 'text with "quotes" and {braces}\nsecond line',
]
for r in hand_raw:
    cases["parse"].append({"raw": r})

for t in ["1. **Scene & Lighting**: soft light\n2) Camera - low angle", "1. A: x\n\n2. B: y\nno header line",
          "1. Mid-air pose: jumping", "   3.   Name  :  text  ", "1.: nothing", "no digits: here", "10. Zone: a, a, b.",
          "1. A: x\r\n2. B: y"]:
    cases["flatten"].append({"text": t})

for s in ["a (b) c", "a \\(b\\) (c)", "((x))", "", "(", "\\)"]:
    cases["style"].append({"text": s})

crop_tag_candidates = [
    "close-up", "portrait", "solo focus", "breast focus", "eye focus", "head out of frame",
    "cropped legs", "out of frame", "feet out of frame", "partially submerged", "full body", "cowboy shot",
    "general", "sensitive", "questionable", "explicit",
    "1girl", "solo", "2boys", "large breasts", "flat chest",
    "hair ornament", "holding spoon", "silver brooch", "blue dress", "black boots",
    "potted plant", "bookshelf", "wooden chair", "simple background", "white background", "monochrome",
    "pyra (xenoblade)", "mythra (xenoblade)",
]

for _ in range(50):
    k = random.randint(1, 10)
    sample_tags = random.sample(crop_tag_candidates, k)
    tags_with_scores = [[t, round(random.uniform(0.3, 0.95), 2)] for t in sample_tags]
    ctype = random.choice(["character_zone", "quadrant"])
    cnames = random.sample(["pyra (xenoblade)", "mythra (xenoblade)"], random.randint(0, 2))
    min_thresh = random.choice([0.55, 0.50])
    sc_thresh = random.choice([0.40, 0.35])
    cases["filter_crop"].append({
        "tags": tags_with_scores,
        "crop_type": ctype,
        "character_names": cnames,
        "min_attribute_threshold": min_thresh,
        "scene_object_threshold": sc_thresh,
    })

for _ in range(50):
    img_w = float(random.choice([800, 1024, 1920]))
    img_h = float(random.choice([600, 1024, 1080]))
    x0 = round(random.uniform(0, img_w * 0.4), 1)
    y0 = round(random.uniform(0, img_h * 0.4), 1)
    x1 = round(random.uniform(x0 + 50, img_w), 1)
    if random.random() < 0.4:
        y1 = img_h
    else:
        y1 = round(random.uniform(y0 + 50, img_h), 1)
    cases["zone_boxes"].append({
        "img_w": img_w, "img_h": img_h,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
    })

# ---- random
for _ in range(250):
    wd = ", ".join(random.sample(wd_pool, random.randint(0, 10)))
    mode = random.choice(MODES)
    md1 = zone_text(random.randint(0, 4)); md2 = zone_text(random.randint(0, 4))
    vocab = random.sample(vocab_pool, random.randint(0, 8))
    keep = random.random() < 0.3
    cases["router"].append({"wd": wd, "mode": mode})
    cases["analyze"].append({"text": md1, "vocab": vocab + random.sample(sorted(p.BUILTIN_VOCAB), 12)})
    cases["merge"].append({"wd": wd, "md1": md1, "md2": md2, "mode": mode, "vocab": vocab})
    cases["nollm"].append({"wd": wd, "md1": md1, "md2": md2, "mode": mode, "vocab": vocab, "keep": keep})
    cases["payload"].append({"wd": wd, "md1": md1, "md2": md2, "mode": mode, "vocab": vocab, "vision": random.random() < 0.5})

conflict_test_samples = [
    ["1girl", "tube top", "cleavage", "boots", "thighhighs"],
    ["1girl", "turtleneck", "cleavage", "boots", "thighhighs"],
    ["1girl", "shoes", "barefoot", "long pants", "bare legs"],
    ["1girl", "open jacket", "cleavage", "midriff", "socks", "shoes"],
    ["1girl", "gloves", "bare hands", "blindfold", "eyes"],
    ["1girl", "turtleneck", "deep cleavage", "collarbone", "bare shoulders"],
]
for sample in conflict_test_samples:
    cases["layer_conflicts"].append({"tags": sample})

# ---- expected values from Python
def vset(v): return set(v) if v else None

for c in cases["router"]:
    r = p.WD14DualPassRouter().sort_tags(c["wd"], c["mode"])
    c["expect"] = r
for c in cases["analyze"]:
    a = p.analyze_prose(c["text"], set(c["vocab"]))
    c["expect"] = {"tags": a["tags"], "strong": sorted(a["strong"]), "weak": sorted(a["weak"]), "negated": a["negated"]}
for c in cases["merge"]:
    m = p.merge_tags(c["wd"], c["md1"], c["md2"], c["mode"], set(c["vocab"]))
    c["expect"] = m
for c in cases["nollm"]:
    old = p.QUALITY_PREFIX
    r = p.AntigravityTagFilter_NoLLM().process_direct(c["wd"], c["md1"], c["md2"], c["mode"], set(c["vocab"]), c["keep"])
    c["expect"] = r
for c in cases["parse"]:
    c["expect"] = p.parse_gemma_json_response(c["raw"])
for c in cases["flatten"]:
    c["expect"] = p._flatten_zones(c["text"])
for c in cases["style"]:
    p.TAG_ESCAPE_PARENTHESES = True
    c["expect"] = p._style_tags(c["text"])
    p.TAG_ESCAPE_PARENTHESES = False
for c in cases["payload"]:
    llm = p.AntigravityTagFilter_LLM()
    pl = llm.build_payload(c["wd"], c["md1"], c["md2"], c["mode"], set(c["vocab"]), vision=c["vision"])
    c["expect"] = llm.payload_to_text(pl)
for c in cases["filter_crop"]:
    res = p.filter_crop_tags(
        c["tags"],
        crop_type=c["crop_type"],
        character_names=c["character_names"],
        min_attribute_threshold=c["min_attribute_threshold"],
        scene_object_threshold=c["scene_object_threshold"],
    )
    c["expect"] = res
for c in cases["zone_boxes"]:
    res = p.compute_zone_boxes(
        c["img_w"], c["img_h"],
        c["x0"], c["y0"], c["x1"], c["y1"],
    )
    c["expect"] = res
for c in cases["layer_conflicts"]:
    kept, sup = p.resolve_layer_conflicts(c["tags"])
    c["expect"] = {"kept": kept, "suppressed": sup}

json.dump(cases, sys.stdout, ensure_ascii=False)
