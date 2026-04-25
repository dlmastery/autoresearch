"""Run the fraud-ecommerce autoresearch baseline.

Mirrors the classification_titanic/run_example.py protocol but uses a real CSV
loader pointed at the FDB-style preprocessed Fraud_Data.csv (151,112 rows,
9.36% fraud rate). Composite primary metric is AUC-ROC since that is what the
FDB paper uses for its leaderboard.
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

    seed = json.loads((HERE / "seed_reasoning.json").read_text(encoding="utf-8"))
    ann_path = results_dir / "reasoning_annotations.json"
    existing = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    existing.update(seed)
    ann_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    dashboard_src = REPO / "generalized_ml_autoresearch" / "dashboard" / "dashboard.html"
    if dashboard_src.exists():
        shutil.copy2(dashboard_src, results_dir / "dashboard.html")

    record = run_experiment(cfg)
    print("\n--- Summary ---")
    print(f"  Exp {record.experiment_num} ({record.backbone})  Composite={record.composite:.4f} Status={record.status}")
    print(f"  Test AUC-ROC: {record.test_primary:.4f}   Val AUC-ROC: {record.val_primary:.4f}")
    for i, v in enumerate(record.per_fold_test):
        print(f"  Fold {i+1} test AUC-ROC: {v:.4f}")
    print(f"  Time elapsed: {record.seconds_elapsed:.1f}s")


if __name__ == "__main__":
    main()
