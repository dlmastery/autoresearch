"""Batch-run iTransformer + xLSTM experiments."""
import subprocess
PY = "C:/Users/evija/anaconda3/python.exe"
CWD = "C:/Users/evija/autoresearch"

EXPERIMENTS = [
    # iTransformer (Liu 2024 ICLR)
    ("itransformer: Exp1 SOTA baseline seq=10",
        "--backbone itransformer --seq-len 10 --lr 5e-5 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("itransformer: Exp2 seq=60 (GBM-winning)",
        "--backbone itransformer --seq-len 60 --lr 5e-5 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("itransformer: Exp3 lr=1e-4",
        "--backbone itransformer --seq-len 60 --lr 1e-4 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("itransformer: Exp4 seq=30",
        "--backbone itransformer --seq-len 30 --lr 5e-5 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("itransformer: Exp5 hidden=256 num_layers=3",
        "--backbone itransformer --seq-len 60 --lr 5e-5 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.1 --huber-delta 1.0 --hidden-size 256 --num-layers 3 --seed 42"),
    ("itransformer: Exp6 hd=0.25",
        "--backbone itransformer --seq-len 60 --lr 5e-5 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.25 --huber-delta 1.0 --seed 42"),
    ("itransformer: Exp7 seed=0",
        "--backbone itransformer --seq-len 60 --lr 5e-5 --epochs 100 --patience 20 --batch-size 32 --weight-decay 0 --warmup-epochs 10 --head-dropout 0.1 --huber-delta 1.0 --seed 0"),
    # xLSTM (Beck 2024 NeurIPS)
    ("xlstm: Exp1 SOTA baseline seq=10",
        "--backbone xlstm --seq-len 10 --lr 5e-4 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("xlstm: Exp2 seq=60",
        "--backbone xlstm --seq-len 60 --lr 5e-4 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.1 --huber-delta 1.0 --seed 42"),
    ("xlstm: Exp3 lr=1e-3 hd=0.25",
        "--backbone xlstm --seq-len 10 --lr 1e-3 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.25 --huber-delta 1.0 --seed 42"),
    ("xlstm: Exp4 num_layers=3",
        "--backbone xlstm --seq-len 10 --lr 5e-4 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.1 --huber-delta 1.0 --num-layers 3 --seed 42"),
    ("xlstm: Exp5 hidden=256",
        "--backbone xlstm --seq-len 10 --lr 5e-4 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.1 --huber-delta 1.0 --hidden-size 256 --seed 42"),
    ("xlstm: Exp6 seed=0",
        "--backbone xlstm --seq-len 10 --lr 5e-4 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.1 --huber-delta 1.0 --seed 0"),
    ("xlstm: Exp7 seed=13",
        "--backbone xlstm --seq-len 10 --lr 5e-4 --epochs 80 --patience 15 --batch-size 16 --weight-decay 1e-3 --warmup-epochs 5 --head-dropout 0.1 --huber-delta 1.0 --seed 13"),
]

for label, args in EXPERIMENTS:
    print(f"\n=== {label} ===")
    cmd = (f'"{PY}" -m autoresearch.run_autoresearch {args} --description "{label}"')
    try:
        r = subprocess.run(cmd, shell=True, cwd=CWD, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        for line in (r.stdout + r.stderr).splitlines():
            if "Composite:" in line:
                print(f"  {line.strip()}")
                break
    except Exception as e:
        print(f"  ERROR: {e}")
