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


cases = {"router": [], "analyze": [], "merge": [], "nollm": [], "parse": [], "flatten": [], "payload": [], "style": []}

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

json.dump(cases, sys.stdout, ensure_ascii=False)
