#!/usr/bin/env python3
"""Score image-read prompts against hand-written answer files.

Input: the output of `cargo test live_image -- --ignored --nocapture` (its IMAGE / PROMPT lines),
from a file or stdin. Each image needs an answer file next to it with the same name and `.txt`:
the Danbooru tags a good prompt should contain. Prints recall, precision and F1 per image and the
average F1. Quality tags and `solo` are ignored on both sides.

Usage: tools/score_image_prompts.py results.txt [label]
"""
import re
import sys
from pathlib import Path

IGNORED = {"masterpiece", "best quality", "high quality", "amazing quality", "very aesthetic",
           "absurdres", "highres", "solo", "score 9", "score 8 up", "score 7 up"}


def norm(tag: str) -> str:
    t = tag.strip().lower().replace("_", " ")
    t = re.sub(r"^\((.*?)(:[0-9.]+)?\)$", r"\1", t)  # (tag:1.2) -> tag
    return t.replace("\\(", "(").replace("\\)", ")").strip()


def tags(text: str) -> set:
    return {norm(t) for t in re.split(r"[,\n]", text) if norm(t)} - IGNORED


def parse(text: str):
    image = None
    for line in text.splitlines():
        if line.startswith("IMAGE   :"):
            image = line.split(":", 1)[1].strip()
        elif line.startswith("PROMPT  :") and image:
            yield image, line.split(":", 1)[1].strip()
            image = None


def main():
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 and sys.argv[1] != "-" else sys.stdin.read()
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = []
    for image, prompt in parse(text):
        answer = Path(image).with_suffix(".txt")
        if not answer.exists():
            print(f"(no answer file for {image})", file=sys.stderr)
            continue
        key, got = tags(answer.read_text()), tags(prompt)
        hit = key & got
        rec = len(hit) / max(1, len(key))
        prec = len(hit) / max(1, len(got))
        f1 = 2 * rec * prec / max(1e-9, rec + prec)
        rows.append((Path(image).name, rec, prec, f1, len(got)))
    print(f"{'image':<32} {'recall':>7} {'precision':>9} {'F1':>6} {'#tags':>6}")
    for name, rec, prec, f1, n in rows:
        print(f"{name[:32]:<32} {rec:7.0%} {prec:9.0%} {f1:6.0%} {n:6d}")
    if rows:
        print(f"\naverage F1 {label}: {sum(r[3] for r in rows) / len(rows):.0%} over {len(rows)} images")


if __name__ == "__main__":
    main()
