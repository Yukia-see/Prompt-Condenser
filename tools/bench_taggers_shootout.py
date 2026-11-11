#!/usr/bin/env python3
"""Standalone Danbooru Tagger Shootout Benchmark.

Compares 4 modern vision taggers (and ensembles) directly against Danbooru ground truth:
1. SmilingWolf/wd-eva02-large-tagger-v3 (10,861 tags, 304M params)
2. Redstonexs/danbooru-tagger-v1 (24,000 tags, unfrozen EVA02-Large)
3. Camais03/camie-tagger-v2 (70,527 tags, ViT-Base 143M params)
4. itterative/convnextv2_huge.dbv4-full-onnx (12,476 tags, ConvNeXt-V2 Huge 692M params)

Run with:
  uv run --with onnxruntime,pillow,numpy python3 tools/bench_taggers_shootout.py
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

IGNORED = {
    "masterpiece", "best quality", "high quality", "amazing quality", "very aesthetic",
    "absurdres", "highres", "solo", "score 9", "score 8 up", "score 7 up"
}

def norm(tag: str) -> str:
    t = tag.strip().lower().replace("_", " ")
    t = re.sub(r"^\((.*?)(:[0-9.]+)?\)$", r"\1", t)
    return t.replace("\\(", "(").replace("\\)", ")").strip()

def tags_from_text(text: str) -> set:
    return {norm(t) for t in re.split(r"[,\n]", text) if norm(t)} - IGNORED

def pad_and_resize(img: Image.Image, target_size: int) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    if img.mode == "RGBA":
        canvas.paste(img, ((side - w) // 2, (side - h) // 2), mask=img.split()[3])
    else:
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((target_size, target_size), Image.Resampling.BICUBIC)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class ModelRunner:
    def __init__(self, name: str):
        self.name = name

    def predict_raw(self, img: Image.Image) -> tuple[list[tuple[str, str, float]], float]:
        """Runs forward pass once, returns ([(normalized_tag, category_kind, probability)], latency_seconds)."""
        raise NotImplementedError


class WdEva02Runner(ModelRunner):
    def __init__(self, model_path: str, tags_path: str, session_options: ort.SessionOptions):
        super().__init__("WD-EVA02-v3")
        self.sess = ort.InferenceSession(model_path, session_options, providers=["CPUExecutionProvider"])
        self.tags = []
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                name = row[1]
                cat = "rating" if row[2] == "9" else ("character" if row[2] == "4" else "general")
                self.tags.append((norm(name), cat))

    def predict_raw(self, img: Image.Image):
        t0 = time.time()
        im448 = pad_and_resize(img, 448)
        arr = np.array(im448, dtype=np.float32)[:, :, ::-1]  # BGR
        arr = np.expand_dims(arr, 0)
        out = self.sess.run(None, {"input": arr})[0][0]
        lat = time.time() - t0
        scored = [(name, cat, float(prob)) for (name, cat), prob in zip(self.tags, out) if cat != "rating"]
        return scored, lat


class RedstoneRunner(ModelRunner):
    def __init__(self, model_path: str, tags_path: str, session_options: ort.SessionOptions):
        super().__init__("Redstonexs-v1")
        self.sess = ort.InferenceSession(model_path, session_options, providers=["CPUExecutionProvider"])
        self.tags = []
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                name = row[1]
                cat = "rating" if row[2] == "9" else ("character" if row[2] == "4" else "general")
                self.tags.append((norm(name), cat))

    def predict_raw(self, img: Image.Image):
        t0 = time.time()
        im448 = pad_and_resize(img, 448)
        arr = (np.array(im448, dtype=np.float32) / 255.0 - 0.5) / 0.5
        arr = arr[:, :, ::-1]  # BGR
        arr = np.transpose(arr, (2, 0, 1))  # NCHW
        arr = np.expand_dims(arr, 0)
        logits = self.sess.run(None, {"input": arr})[0][0]
        probs = sigmoid(logits)
        lat = time.time() - t0
        scored = [(name, cat, float(prob)) for (name, cat), prob in zip(self.tags, probs) if cat != "rating"]
        return scored, lat


class CamieRunner(ModelRunner):
    def __init__(self, model_path: str, meta_path: str, session_options: ort.SessionOptions):
        super().__init__("Camie-v2")
        self.sess = ort.InferenceSession(model_path, session_options, providers=["CPUExecutionProvider"])
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idx_to_tag = meta["dataset_info"]["tag_mapping"]["idx_to_tag"]
        tag_to_cat = meta["dataset_info"]["tag_mapping"].get("tag_to_category", {})
        self.tags = []
        for i in range(len(idx_to_tag)):
            tname = idx_to_tag[str(i)]
            raw_cat = tag_to_cat.get(tname, "general")
            cat = "character" if raw_cat == "character" else ("rating" if raw_cat in ("rating", "year", "meta") else "general")
            self.tags.append((norm(tname), cat))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_raw(self, img: Image.Image):
        t0 = time.time()
        im512 = pad_and_resize(img, 512)
        arr = (np.array(im512, dtype=np.float32) / 255.0 - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # NCHW, RGB
        arr = np.expand_dims(arr, 0)
        res = self.sess.run(None, {"input": arr})
        probs = sigmoid(res[1][0])  # refined_predictions
        lat = time.time() - t0
        scored = [(name, cat, float(prob)) for (name, cat), prob in zip(self.tags, probs) if cat != "rating"]
        return scored, lat


class ConvNeXtRunner(ModelRunner):
    def __init__(self, model_path: str, tags_path: str, session_options: ort.SessionOptions):
        super().__init__("ConvNeXtV2-Huge")
        self.sess = ort.InferenceSession(model_path, session_options, providers=["CPUExecutionProvider"])
        self.tags = []
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                name = row[1]
                cat = "rating" if row[2] == "9" else ("character" if row[2] == "4" else "general")
                self.tags.append((norm(name), cat))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_raw(self, img: Image.Image):
        t0 = time.time()
        im512 = pad_and_resize(img, 512)
        arr = (np.array(im512, dtype=np.float32) / 255.0 - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # NCHW, RGB
        arr = np.expand_dims(arr, 0)
        logits = self.sess.run(None, {"pixel_values": arr})[0][0]
        probs = sigmoid(logits)
        lat = time.time() - t0
        scored = [(name, cat, float(prob)) for (name, cat), prob in zip(self.tags, probs) if cat != "rating"]
        return scored, lat


def filter_tags(raw_scored: list[tuple[str, str, float]], threshold: float, char_threshold: float) -> set[str]:
    selected = set()
    for name, cat, score in raw_scored:
        t = char_threshold if cat == "character" else threshold
        if score >= t and name not in IGNORED:
            selected.add(name)
    return selected


def score_predictions(pred_set: set[str], key_set: set[str]):
    hit = key_set & pred_set
    rec = len(hit) / max(1, len(key_set))
    prec = len(hit) / max(1, len(pred_set))
    f1 = 2 * rec * prec / max(1e-9, rec + prec)
    return rec, prec, f1, len(pred_set), len(hit), len(key_set)


def evaluate_threshold(raw_predictions: list[tuple[str, list, set, float]], threshold: float, char_threshold: float):
    rows = []
    latencies = []
    weak_rows = []
    weak_names = {"chibi_pink.jpg", "dark_armor.jpg", "sakura_wings.jpg"}

    for name, raw_scored, key_set, lat in raw_predictions:
        pred_set = filter_tags(raw_scored, threshold, char_threshold)
        rec, prec, f1, n_pred, n_hit, n_key = score_predictions(pred_set, key_set)
        row = (name, rec, prec, f1, n_pred, n_hit, n_key, lat)
        rows.append(row)
        latencies.append(lat)
        if name in weak_names:
            weak_rows.append(row)

    avg_f1 = sum(r[3] for r in rows) / len(rows)
    avg_rec = sum(r[1] for r in rows) / len(rows)
    avg_prec = sum(r[2] for r in rows) / len(rows)
    avg_lat = sum(latencies) / len(latencies)

    weak_f1 = sum(r[3] for r in weak_rows) / len(weak_rows) if weak_rows else 0.0
    weak_rec = sum(r[1] for r in weak_rows) / len(weak_rows) if weak_rows else 0.0
    weak_prec = sum(r[2] for r in weak_rows) / len(weak_rows) if weak_rows else 0.0

    return {
        "rows": rows,
        "avg_f1": avg_f1,
        "avg_rec": avg_rec,
        "avg_prec": avg_prec,
        "avg_lat": avg_lat,
        "weak_f1": weak_f1,
        "weak_rec": weak_rec,
        "weak_prec": weak_prec,
    }


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

    print(f"Loaded {len(test_cases)} evaluation test cases from {eval_dir}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 16

    print("\nInitializing model runners...")
    runners = [
        WdEva02Runner(
            "/home/yukia/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/model.onnx",
            "/home/yukia/.local/share/prompt-condenser/wd-eva02-large-tagger-v3/selected_tags.csv",
            opts
        ),
        RedstoneRunner(
            "tests/tagger_shootout/models/redstonexs_v1/model.onnx",
            "tests/tagger_shootout/models/redstonexs_v1/selected_tags.csv",
            opts
        ),
        ConvNeXtRunner(
            "tests/tagger_shootout/models/convnextv2_huge/model.onnx",
            "tests/tagger_shootout/models/convnextv2_huge/selected_tags.csv",
            opts
        ),
        CamieRunner(
            "tests/tagger_shootout/models/camie_v2/model.onnx",
            "tests/tagger_shootout/models/camie_v2/metadata.json",
            opts
        ),
    ]

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50]
    all_model_raw = {}

    for runner in runners:
        print(f"\n=======================================================")
        print(f"Running Inferences: {runner.name}")
        print(f"=======================================================")
        raw_preds = []
        for p, key_set in test_cases:
            img = Image.open(p).convert("RGBA")
            scored, lat = runner.predict_raw(img)
            raw_preds.append((p.name, scored, key_set, lat))
            print(f"  {p.name:<28} done in {lat:.2f}s")
        all_model_raw[runner.name] = raw_preds

    # Sweep thresholds for each model
    summary = {}
    for name, raw_preds in all_model_raw.items():
        print(f"\n--- Threshold Sweep for {name} ---")
        best_f1 = -1
        best_thr = None
        best_res = None
        for thr in thresholds:
            res = evaluate_threshold(raw_preds, threshold=thr, char_threshold=0.70)
            print(f"  Thr {thr:.2f} -> All 10: F1={res['avg_f1']:.1%}, Rec={res['avg_rec']:.1%}, Prec={res['avg_prec']:.1%} | 3 Weak: F1={res['weak_f1']:.1%}, Rec={res['weak_rec']:.1%}, Prec={res['weak_prec']:.1%}")
            if res["avg_f1"] > best_f1:
                best_f1 = res["avg_f1"]
                best_thr = thr
                best_res = res
        summary[name] = (best_thr, best_res)

    # Also test Ensemble: WD + Redstonexs (average scores on shared tags, union)
    print(f"\n=======================================================")
    print(f"Testing Ensemble: WD-EVA02 + Redstonexs-v1")
    print(f"=======================================================")
    wd_preds = {name: dict([(t, s) for t, c, s in scored]) for name, scored, _, _ in all_model_raw["WD-EVA02-v3"]}
    rs_preds = {name: dict([(t, s) for t, c, s in scored]) for name, scored, _, _ in all_model_raw["Redstonexs-v1"]}
    ens_preds = []
    for p, key_set in test_cases:
        wd_map = wd_preds[p.name]
        rs_map = rs_preds[p.name]
        all_tags = set(wd_map.keys()) | set(rs_map.keys())
        scored = []
        for t in all_tags:
            if t in wd_map and t in rs_map:
                s = 0.5 * (wd_map[t] + rs_map[t])
            elif t in rs_map:
                s = rs_map[t]
            else:
                s = wd_map[t]
            scored.append((t, "general", s))
        ens_preds.append((p.name, scored, key_set, 0.0))

    best_ens_f1 = -1
    best_ens_thr = None
    best_ens_res = None
    for thr in thresholds:
        res = evaluate_threshold(ens_preds, threshold=thr, char_threshold=0.70)
        print(f"  Ensemble Thr {thr:.2f} -> All 10: F1={res['avg_f1']:.1%}, Rec={res['avg_rec']:.1%}, Prec={res['avg_prec']:.1%} | 3 Weak: F1={res['weak_f1']:.1%}, Rec={res['weak_rec']:.1%}, Prec={res['weak_prec']:.1%}")
        if res["avg_f1"] > best_ens_f1:
            best_ens_f1 = res["avg_f1"]
            best_ens_thr = thr
            best_ens_res = res
    summary["Ensemble(WD+RS)"] = (best_ens_thr, best_ens_res)

    print("\n\n" + "=" * 86)
    print("FINAL STANDALONE SHOOTOUT BENCHMARK RESULTS")
    print("=" * 86)
    print(f"{'Model':<20} {'Best Thr':>8} {'All F1':>8} {'All Rec':>8} {'All Prec':>9} {'Weak F1':>8} {'Weak Rec':>9} {'Avg Lat':>8}")
    print("-" * 86)
    for name, (thr, res) in summary.items():
        print(f"{name:<20} {thr:8.2f} {res['avg_f1']:8.1%} {res['avg_rec']:8.1%} {res['avg_prec']:9.1%} {res['weak_f1']:8.1%} {res['weak_rec']:9.1%} {res['avg_lat']:7.2f}s")
    print("=" * 86)

    # Detailed per-image breakdown
    for name, (thr, res) in summary.items():
        print(f"\nDetailed Breakdown: {name} @ Thr {thr:.2f}:")
        print(f"{'Image':<28} {'Recall':>7} {'Precision':>10} {'F1':>6} {'Hit/Key':>9}")
        for img_name, rec, prec, f1, n_pred, n_hit, n_key, _ in res["rows"]:
            print(f"{img_name:<28} {rec:7.1%} {prec:10.1%} {f1:6.1%} {n_hit:3d}/{n_key:<3d}")


if __name__ == "__main__":
    main()
