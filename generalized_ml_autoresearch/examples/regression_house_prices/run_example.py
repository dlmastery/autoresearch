"""Run the California Housing regression example end-to-end.

Usage:
    python run_example.py

This script:
  1. Seeds reasoning_annotations.json with the pre-run entry for Exp 1 (from seed_reasoning.json).
  2. Launches the runner against config.yaml.
  3. Prints the resulting per-fold breakdown.

After running, open the dashboard:
    python -m http.server 8765 --directory autoresearch_results
    # then visit http://localhost:8765/dashboard.html
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from generalized_ml_autoresearch.core.runner import run_experiment, _load_config  # noqa: E402


def main():
    cfg = _load_config(HERE / "config.yaml")
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "trade_logs").mkdir(exist_ok=True)

    # 1. Seed the pre-run reasoning annotation
    seed = json.loads((HERE / "seed_reasoning.json").read_text(encoding="utf-8"))
    ann_path = results_dir / "reasoning_annotations.json"
    if ann_path.exists():
        existing = json.loads(ann_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing.update(seed)
    ann_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # 2. Copy the dashboard
    dashboard_src = REPO / "generalized_ml_autoresearch" / "dashboard" / "dashboard.html"
    shutil.copy2(dashboard_src, results_dir / "dashboard.html")

    # 3. Run
    record = run_experiment(cfg)
    print("\n--- Summary ---")
    print(f"  Exp {record.experiment_num} ({record.backbone})")
    print(f"  Composite: {record.composite:.4f}  Status: {record.status}")
    print(f"  Test RMSE (mean across folds): {record.test_primary:.4f}")
    print(f"  Val RMSE: {record.val_primary:.4f}")
    for i, v in enumerate(record.per_fold_test):
        print(f"  Fold {i+1} test RMSE: {v:.4f}")


if __name__ == "__main__":
    main()
