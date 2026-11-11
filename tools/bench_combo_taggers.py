#!/usr/bin/env python3
"""Comprehensive Benchmark of Combination/Ensemble Vision Taggers.

Evaluates single models and 2-way / 3-way combinations:
- WD-EVA02-v3 (SmilingWolf)
- Redstonexs-v1 (Unfrozen EVA02-Large)
- Camie-v2 (ViT-Base with 2-stage cross-attention)
- ConvNeXt-V2 Huge (Animtimm)
- Combo 1: WD + Redstonexs
- Combo 2: Redstonexs + Camie
- Combo 3: ConvNeXt + Redstonexs
- Combo 4: WD + Camie
- Tri-Combo: WD + Redstonexs + Camie

Run with:
  uv run --with onnxruntime,pillow,numpy python3 tools/bench_combo_taggers.py
"""

import os
import sys
import csv
import json
import time
import re
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

from tools.bench_taggers_shootout import (
    IGNORED, norm, tags_from_text, pad_and_resize, sigmoid,
    WdEva02Runner, RedstoneRunner, CamieRunner, ConvNeXtRunner,
    filter_tags, score_predictions, evaluate_threshold
)

def blend_predictions(preds_list: list[dict[str, float]], weights: list[float] = None) -> list[tuple[str, str, float]]:
    if weights is None:
        weights = [1.0 / len(preds_list)] * len(preds_list)
    
    all_tags = set()
    for p in preds_list:
        all_tags.update(p.keys())
    
    scored = []
    for t in all_tags:
        # Weighted average over models that predict this tag (or zero / model score)
        # Using standard ensemble: average over the models that contain the tag in their vocab
        contributing_scores = [p[t] for p in preds_list if t in p]
        avg_score = sum(contributing_scores) / len(contributing_scores)
        scored.append((t, "general", float(avg_score)))
    return scored


def main():
    repo_root = Path("/home/yukia/Documents/Prompt Condensor")
    eval_dir = repo_root / "tests/eval_local"

    image_paths = sorted(list(eval_dir.glob("*.jpg")) + list(eval_dir.glob("*.png")))
    test_cases = []
    for p in image_paths:
        ans = p.with_suffix(".txt")
        if ans.exists() and "ai_generated" not in str(p):
            key_set = tags_from_text(ans.read_text(encoding="utf-8"))
            test_cases.append((p, key_set))

    print(f"Loaded {len(test_cases)} evaluation test cases from {eval_dir}", flush=True)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 16

    print("\nInitializing model runners...", flush=True)
    runners = {
        "WD": WdEva02Runner(
            "/home/yukia/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/model.onnx",
            "/home/yukia/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/selected_tags.csv",
            opts
        ),
        "RS": RedstoneRunner(
            "tests/tagger_shootout/models/redstonexs_v1/model.onnx",
            "tests/tagger_shootout/models/redstonexs_v1/selected_tags.csv",
            opts
        ),
        "CM": CamieRunner(
            "tests/tagger_shootout/models/camie_v2/model.onnx",
            "tests/tagger_shootout/models/camie_v2/metadata.json",
            opts
        ),
        "CN": ConvNeXtRunner(
            "tests/tagger_shootout/models/convnextv2_huge/model.onnx",
            "tests/tagger_shootout/models/convnextv2_huge/selected_tags.csv",
            opts
        ),
    }

    print("\nRunning inference for all 4 models...", flush=True)
    raw_maps = {k: {} for k in runners.keys()}
    raw_scored_lists = {k: [] for k in runners.keys()}

    for name, runner in runners.items():
        print(f"  Running {name}...", flush=True)
        t_start = time.time()
        for p, key_set in test_cases:
            img = Image.open(p).convert("RGBA")
            scored, lat = runner.predict_raw(img)
            tag_dict = dict([(t, s) for t, c, s in scored])
            raw_maps[name][p.name] = tag_dict
            raw_scored_lists[name].append((p.name, scored, key_set, lat))
        print(f"  {name} completed in {time.time()-t_start:.1f}s", flush=True)

    # Define all ensembles
    ensemble_configs = {
        "WD-EVA02-v3 (Solo)": [("WD", 1.0)],
        "Redstonexs-v1 (Solo)": [("RS", 1.0)],
        "Camie-v2 (Solo)": [("CM", 1.0)],
        "ConvNeXtV2-Huge (Solo)": [("CN", 1.0)],
        "Combo: WD + Redstone": [("WD", 1.0), ("RS", 1.0)],
        "Combo: Redstone + Camie": [("RS", 1.0), ("CM", 1.0)],
        "Combo: ConvNeXt + Redstone": [("CN", 1.0), ("RS", 1.0)],
        "Combo: WD + Camie": [("WD", 1.0), ("CM", 1.0)],
        "Tri-Combo: WD + RS + Camie": [("WD", 1.0), ("RS", 1.0), ("CM", 1.0)],
    }

    thresholds = [0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55]
    summary_results = {}

    print("\n" + "=" * 80, flush=True)
    print("EVALUATING COMBINATION ENSEMBLES ACROSS THRESHOLDS", flush=True)
    print("=" * 80, flush=True)

    for ens_name, components in ensemble_configs.items():
        # Build ensemble predictions
        ens_preds = []
        for p, key_set in test_cases:
            model_dicts = [raw_maps[m_name][p.name] for m_name, _ in components]
            blended = blend_predictions(model_dicts)
            ens_preds.append((p.name, blended, key_set, 0.0))

        best_f1 = -1
        best_thr = None
        best_res = None
        for thr in thresholds:
            res = evaluate_threshold(ens_preds, threshold=thr, char_threshold=0.70)
            if res["avg_f1"] > best_f1:
                best_f1 = res["avg_f1"]
                best_thr = thr
                best_res = res

        summary_results[ens_name] = (best_thr, best_res)
        print(f"{ens_name:<28} Best Thr: {best_thr:.2f} | All F1: {best_res['avg_f1']:.1%} (Rec: {best_res['avg_rec']:.1%}, Prec: {best_res['avg_prec']:.1%}) | Weak F1: {best_res['weak_f1']:.1%} (Rec: {best_res['weak_rec']:.1%})", flush=True)

    # Generate full report table
    report_lines = []
    report_lines.append("# Combination Tagger Benchmark Report\n")
    report_lines.append("**Date**: 2026-11-10  ")
    report_lines.append("**Dataset**: 10 non-AI images in `tests/eval_local/` (including 3 focus weak images)  ")
    report_lines.append("**Metric**: Exact `score_image_prompts.py` logic (excluding quality/solo tags)\n")
    report_lines.append("## Ensemble Scoreboard\n")
    report_lines.append("| Tagger / Combination | Best Thr | All 10 F1 | All Recall | Precision | Weak Images F1 | Weak Recall |")
    report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for ens_name, (best_thr, best_res) in summary_results.items():
        report_lines.append(f"| **{ens_name}** | {best_thr:.2f} | **{best_res['avg_f1']:.1%}** | {best_res['avg_rec']:.1%} | {best_res['avg_prec']:.1%} | **{best_res['weak_f1']:.1%}** | {best_res['weak_rec']:.1%} |")

    report_lines.append("\n## Detailed Per-Image Breakdown for Top 3 Combos\n")
    top_combos = ["Combo: WD + Redstone", "Tri-Combo: WD + RS + Camie", "Combo: ConvNeXt + Redstone"]
    for c_name in top_combos:
        best_thr, res = summary_results[c_name]
        report_lines.append(f"### {c_name} (Optimal Threshold: {best_thr:.2f})\n")
        report_lines.append("| Image | Recall | Precision | F1 | Hit / Key |")
        report_lines.append("|---|:---:|:---:|:---:|:---:|")
        for img_name, rec, prec, f1, n_pred, n_hit, n_key, _ in res["rows"]:
            report_lines.append(f"| `{img_name}` | {rec:.1%} | {prec:.1%} | **{f1:.1%}** | {n_hit}/{n_key} |")
        report_lines.append("")

    report_content = "\n".join(report_lines)
    
    # Save reports
    out_path = repo_root / "docs/reports/combo_tagger_benchmark_report.md"
    out_path.write_text(report_content, encoding="utf-8")
    
    claude_path = Path.home() / ".claude/plans/combo_tagger_benchmark_report.md"
    claude_path.write_text(report_content, encoding="utf-8")
    
    print(f"\nSaved benchmark reports to:")
    print(f"  {out_path}")
    print(f"  {claude_path}", flush=True)

if __name__ == "__main__":
    main()
