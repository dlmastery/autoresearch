"""Post-XGBoost-batch orchestrator:
 1. Identify current global champion in best_config.json
 2. Update _save_xgb_model.py defaults and re-serialise the new champion
    to a new winners/ archive folder
 3. Run LightGBM batch (_batch_lgbm.py)
 4. Run CatBoost batch (_batch_catboost.py)
 5. Run phase (b) ensemble (_phase_b_ensemble.py)
 6. Sync dashboard
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("C:/Users/evija/autoresearch")
PY = "C:/Users/evija/anaconda3/python.exe"

def run(label, cmd_args, timeout=900):
    print(f"\n>>> {label}")
    print(f"    {' '.join(cmd_args[:2])}...")
    try:
        r = subprocess.run(cmd_args, cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        tail = (r.stdout + r.stderr).strip().splitlines()[-10:]
        for line in tail:
            print(f"    {line}")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return False

# --- 1: quick XGBoost seq=60 to extend the upward trend ---
run("XGBoost seq=60 (trend extension)",
    [PY, "-m", "autoresearch.run_autoresearch",
     "--backbone", "xgboost", "--seq-len", "60",
     "--max-depth", "4", "--gbm-lr", "0.01", "--seed", "42",
     "--description", "xgboost: Exp26 seq=60 (extend upward trend)"],
    timeout=900)

# --- 2: archive current champion ---
best = json.loads((ROOT / "autoresearch" / "autoresearch_results" / "best_config.json").read_text())
print(f"Current champion: {best['backbone']} Exp{best.get('experiment_num')} "
      f"composite +{best.get('composite'):.4f} test Sharpe +{best.get('sharpe'):.4f}")
print(f"  Description: {best.get('description')}")

# --- 3: LightGBM batch ---
run("LightGBM batch (15 experiments)",
    [PY, str(ROOT / "autoresearch" / "_batch_lgbm.py")],
    timeout=3600)

# --- 4: CatBoost batch ---
run("CatBoost batch (15 experiments)",
    [PY, str(ROOT / "autoresearch" / "_batch_catboost.py")],
    timeout=3600)

# --- 5: dashboard sync ---
run("Dashboard sync to docs/",
    [PY, str(ROOT / "autoresearch" / "_sync_dashboard_to_docs.py")],
    timeout=300)

# --- 6: phase (b) ensemble (requires at least 2 GBM pickle bundles) ---
run("Phase (b) ensemble",
    [PY, str(ROOT / "autoresearch" / "_phase_b_ensemble.py")],
    timeout=600)

print("\n>>> Orchestration complete.")
