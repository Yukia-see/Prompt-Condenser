#!/usr/bin/env python3
"""Evaluation Harness CLI Wrapper for Phase 1 Baseline & Multi-Arm Ablation

Usage:
    python3 tools/eval_harness.py [--dir DIR] [--arm ARM] [--split SPLIT] [--output REPORT]
"""

import sys
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    cmd = ["cargo", "run", "--", "eval"] + sys.argv[1:]
    
    print(f"Executing evaluation harness via cargo: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
