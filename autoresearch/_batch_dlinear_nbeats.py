"""Batch-run DLinear + N-BEATS experiments (Tier-2.5 bonus baselines).
~15 experiments total."""
import subprocess
PY = "C:/Users/evija/anaconda3/python.exe"
CWD = "C:/Users/evija/autoresearch"

EXPERIMENTS = [
    # DLinear -- Zeng 2023 AAAI arXiv:2205.13504
    ("dlinear: Exp1 SOTA baseline seq=10",
        "--backbone dlinear --seq-len 10 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.15 --huber-delta 1.0 --seed 42"),
    ("dlinear: Exp2 seq=60 (GBM-winning)",
        "--backbone dlinear --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.15 --huber-delta 1.0 --seed 42"),
    ("dlinear: Exp3 seq=30",
        "--backbone dlinear --seq-len 30 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.15 --huber-delta 1.0 --seed 42"),
    ("dlinear: Exp4 lr=3e-4 slower",
        "--backbone dlinear --seq-len 60 --lr 3e-4 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.15 --huber-delta 1.0 --seed 42"),
    ("dlinear: Exp5 hidden=256",
        "--backbone dlinear --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.15 --huber-delta 1.0 --hidden-size 256 --seed 42"),
    ("dlinear: Exp6 seed=0",
        "--backbone dlinear --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.15 --huber-delta 1.0 --seed 0"),
    # N-BEATS -- Oreshkin 2020 ICLR arXiv:1905.10437
    ("nbeats: Exp1 baseline seq=10",
        "--backbone nbeats --seq-len 10 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("nbeats: Exp2 seq=60 (GBM-winning)",
        "--backbone nbeats --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("nbeats: Exp3 seq=30",
        "--backbone nbeats --seq-len 30 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("nbeats: Exp4 lr=5e-4",
        "--backbone nbeats --seq-len 60 --lr 5e-4 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("nbeats: Exp5 head_dropout=0.25",
        "--backbone nbeats --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.25 --huber-delta 1.0 --seed 42"),
    ("nbeats: Exp6 wd=1e-3",
        "--backbone nbeats --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-3 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("nbeats: Exp7 seed=0",
        "--backbone nbeats --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.1 --huber-delta 1.0 --seed 0"),
    ("nbeats: Exp8 seed=13 variance",
        "--backbone nbeats --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.1 --huber-delta 1.0 --seed 13"),
    ("dlinear: Exp7 seq=60 hd=0.25",
        "--backbone dlinear --seq-len 60 --lr 1e-3 --epochs 100 --patience 20 --batch-size 32 --weight-decay 1e-4 --head-dropout 0.25 --huber-delta 1.0 --seed 42"),
]

for label, args in EXPERIMENTS:
    print(f"\n=== {label} ===")
    cmd = (f'"{PY}" -m autoresearch.run_autoresearch {args} --description "{label}"')
    try:
        r = subprocess.run(cmd, shell=True, cwd=CWD, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
        for line in (r.stdout + r.stderr).splitlines():
            if "Composite:" in line:
                print(f"  {line.strip()}")
                break
            if "NEW GLOBAL" in line:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"  ERROR: {e}")
