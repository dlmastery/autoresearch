"""Batch-run CatBoost 15 experiments."""
import subprocess

PY = "C:/Users/evija/anaconda3/python.exe"
CWD = "C:/Users/evija/autoresearch"

EXPERIMENTS = [
    ("catboost: Exp1 SOTA baseline (Prokhorenkova 2018)",
        "--seq-len 10 --seed 42"),
    ("catboost: Exp2 depth=4 (mirror XGB winning depth)",
        "--seq-len 10 --seed 42 --depth 4"),
    ("catboost: Exp3 depth=8 deeper",
        "--seq-len 10 --seed 42 --depth 8"),
    ("catboost: Exp4 depth=4 lr=0.01",
        "--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01"),
    ("catboost: Exp5 depth=4 iter=3000",
        "--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --iterations 3000"),
    ("catboost: Exp6 seq_len=20",
        "--seq-len 20 --seed 42 --depth 4 --gbm-lr 0.01"),
    ("catboost: Exp7 l2_leaf_reg=1",
        "--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --l2-leaf-reg 1"),
    ("catboost: Exp8 l2_leaf_reg=10",
        "--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --l2-leaf-reg 10"),
    ("catboost: Exp9 bagging_temperature=0 (no ordered boosting noise)",
        "--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --bagging-temperature 0"),
    ("catboost: Exp10 random_strength=10",
        "--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --random-strength 10"),
    ("catboost: Exp11 bootstrap_type=Bernoulli",
        '--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --bootstrap-type Bernoulli'),
    ("catboost: Exp12 bootstrap_type=No",
        '--seq-len 10 --seed 42 --depth 4 --gbm-lr 0.01 --bootstrap-type No'),
    ("catboost: Exp13 seed=0 variance",
        "--seq-len 10 --seed 0 --depth 4 --gbm-lr 0.01"),
    ("catboost: Exp14 seed=99 variance",
        "--seq-len 10 --seed 99 --depth 4 --gbm-lr 0.01"),
    ("catboost: Exp15 seed=13 variance",
        "--seq-len 10 --seed 13 --depth 4 --gbm-lr 0.01"),
]

for label, args in EXPERIMENTS:
    print(f"\n=== {label} ===")
    cmd = (
        f'"{PY}" -m autoresearch.run_autoresearch --backbone catboost '
        f'{args} --description "{label}"'
    )
    try:
        result = subprocess.run(cmd, shell=True, cwd=CWD, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 timeout=600)
        out = result.stdout + result.stderr
        for line in out.splitlines():
            if "Composite:" in line:
                print(f"  {line.strip()}")
                break
    except Exception as e:
        print(f"  ERROR: {e}")
