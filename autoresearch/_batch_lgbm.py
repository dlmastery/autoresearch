"""Batch-run LightGBM 15 experiments covering the SOTA recipe + HP sweep.
Uses the same runner; results go into the standard JSONL + reasoning pipeline.
"""
import subprocess
import sys

PY = "C:/Users/evija/anaconda3/python.exe"
CWD = "C:/Users/evija/autoresearch"

EXPERIMENTS = [
    # (label, extra cli args)
    ("lgbm: Exp1 SOTA baseline (Ke 2017)",
        "--seq-len 10 --seed 42"),
    ("lgbm: Exp2 num_leaves=31 standard",
        "--seq-len 10 --seed 42 --num-leaves 31"),
    ("lgbm: Exp3 num_leaves=127 deeper",
        "--seq-len 10 --seed 42 --num-leaves 127"),
    ("lgbm: Exp4 num_leaves=15 shallow",
        "--seq-len 10 --seed 42 --num-leaves 15"),
    ("lgbm: Exp5 lr=0.01 (XGBoost winning lr)",
        "--seq-len 10 --seed 42 --gbm-lr 0.01"),
    ("lgbm: Exp6 lr=0.01 num_leaves=31",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --num-leaves 31"),
    ("lgbm: Exp7 max_depth=4 (mirror XGB champ)",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --max-depth 4"),
    ("lgbm: Exp8 seq_len=20 (mirror XGB champ)",
        "--seq-len 20 --seed 42 --gbm-lr 0.01 --max-depth 4"),
    ("lgbm: Exp9 min_data_in_leaf=50",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --min-data-in-leaf 50"),
    ("lgbm: Exp10 feature_fraction=0.5 aggressive",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --feature-fraction 0.5"),
    ("lgbm: Exp11 bagging_fraction=0.5",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --bagging-fraction 0.5"),
    ("lgbm: Exp12 reg_lambda=10 strong L2",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --reg-lambda 10"),
    ("lgbm: Exp13 reg_alpha=1 L1",
        "--seq-len 10 --seed 42 --gbm-lr 0.01 --reg-alpha 1"),
    ("lgbm: Exp14 seed=0 variance",
        "--seq-len 10 --seed 0 --gbm-lr 0.01 --max-depth 4"),
    ("lgbm: Exp15 seed=13 variance",
        "--seq-len 10 --seed 13 --gbm-lr 0.01 --max-depth 4"),
]

for label, args in EXPERIMENTS:
    print(f"\n=== {label} ===")
    cmd = (
        f'"{PY}" -m autoresearch.run_autoresearch --backbone lightgbm '
        f'{args} --description "{label}"'
    )
    try:
        result = subprocess.run(cmd, shell=True, cwd=CWD, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 timeout=600)
        out = result.stdout + result.stderr
        # extract composite line
        for line in out.splitlines():
            if "Composite:" in line:
                print(f"  {line.strip()}")
                break
            if "NEW GLOBAL" in line:
                print(f"  {line.strip()}")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")
