"""Emit Rust data tables from the Python reference so nothing is retyped by hand."""
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
import prompt_condenser_dual_pass as p


def rs_str(s):
    out = ['"']
    for ch in s:
        if ch == '"': out.append('\\"')
        elif ch == "\\": out.append("\\\\")
        elif ch == "\n": out.append("\\n")
        elif ch == "\r": out.append("\\r")
        elif ch == "\t": out.append("\\t")
        else: out.append(ch)
    out.append('"')
    return "".join(out)


def rs_arr(name, items, doc=None):
    items = sorted(items)
    lines = []
    if doc: lines.append(f"/// {doc}")
    lines.append(f"pub const {name}: &[&str] = &[")
    for it in items:
        lines.append(f"    {rs_str(it)},")
    lines.append("];")
    return "\n".join(lines)


out = ["// ---- GENERATED from prompt_condenser_dual_pass.py by tools/gen_data.py: do not edit by hand ----", ""]
out.append(f"pub const ZONE_FORMAT_HINT: &str = {rs_str(p.ZONE_FORMAT_HINT)};")
out.append(f"pub const QUALITY_PREFIX_DEFAULT: &str = {rs_str(p.QUALITY_PREFIX)};")
out.append(f"pub const GEMMA_OLLAMA_MODEL: &str = {rs_str(p.GEMMA_OLLAMA_MODEL)};")
out.append(f"pub const VISION_TAIL: &str = {rs_str(p._VISION_TAIL)};")
for mode, key in (("scene_shape", "A"), ("scene_pose", "B")):
    cfg = p.MODE_CONFIG[mode]
    out.append(f"pub const MODE_{key}_PASS1_PROMPT: &str = {rs_str(cfg['pass1_prompt'])};")
    out.append(f"pub const MODE_{key}_PASS2_PROMPT: &str = {rs_str(cfg['pass2_prompt'])};")
    out.append(f"pub const MODE_{key}_LLM_INSTRUCTION: &str = {rs_str(cfg['llm_instruction'])};")
    out.append(rs_arr(f"MODE_{key}_PASS1_KEYWORDS", cfg["pass1_keywords"]))
out.append(rs_arr("BUILTIN_VOCAB", p.BUILTIN_VOCAB))
out.append(rs_arr("COLORS", p.COLORS))
out.append(rs_arr("SINGLE_STOP", p._SINGLE_STOP))
out.append(rs_arr("NEGATORS", p._NEGATORS))
out.append(rs_arr("HEDGES", p._HEDGES))
out.append(rs_arr("SCOPE_END", p._SCOPE_END))
out.append(rs_arr("NEW_STATEMENT_VERBS", p._NEW_STATEMENT_VERBS))
out.append(rs_arr("CROP_ARTIFACT_TAGS", p.CROP_ARTIFACT_TAGS))
out.append(rs_arr("RATING_TAGS", p.RATING_TAGS))
out.append(rs_arr("BODY_SIZE_AND_COUNT_TAGS", p.BODY_SIZE_AND_COUNT_TAGS))
out.append(rs_arr("CONCRETE_SCENE_OBJECTS", p.CONCRETE_SCENE_OBJECTS))
out.append(rs_arr("SCENE_KEYWORDS", p.SCENE_KEYWORDS))
out.append("pub const EXCLUSIVE_GROUPS: &[&[&str]] = &[")
for g in p.EXCLUSIVE_GROUPS:
    out.append("    &[" + ", ".join(rs_str(x) for x in sorted(g)) + "],")
out.append("];")
out.append("pub const ATTRIBUTE_GROUPS: &[&[&str]] = &[")
for g in p.ATTRIBUTE_GROUPS:
    out.append("    &[" + ", ".join(rs_str(x) for x in sorted(g)) + "],")
out.append("];")
out.append("pub const COEXISTING_PAIRS: &[(&str, &str)] = &[")
for a, b in sorted(p.COEXISTING_PAIRS):
    out.append(f"    ({rs_str(a)}, {rs_str(b)}),")
out.append("];")
out.append("pub const OCCLUSION_SUPPRESSIONS: &[(&str, &str)] = &[")
for a, b in sorted(p.SUPPRESSIONS):
    out.append(f"    ({rs_str(a)}, {rs_str(b)}),")
out.append("];")
out.append("pub const CLOTHING_LAYER_DEFS: &[ClothingLayerDef] = &[")
for tag, info in sorted(p.CLOTHING_LAYERS.items()):
    covers = info.get("covers", [])
    covers_str = "&[" + ", ".join(rs_str(c) for c in covers) + "]"
    rank = p.LAYER_RANKS.get(info.get("layer", ""), 0)
    out.append(f"    ClothingLayerDef {{ tag: {rs_str(tag)}, region: {rs_str(info.get('region', ''))}, layer: {rs_str(info.get('layer', ''))}, rank: {rank}, covers: {covers_str} }},")
out.append("];")
out.append("// ---- END GENERATED ----")
print("\n".join(out))
