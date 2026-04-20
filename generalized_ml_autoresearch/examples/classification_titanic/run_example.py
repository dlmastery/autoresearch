"""Run the classification example end-to-end.

NOTE: named "titanic" for demonstration, actually uses sklearn.datasets.load_breast_cancer
as a stable, bundled binary-classification benchmark (no network dependency).
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

    shutil.copy2(REPO / "generalized_ml_autoresearch" / "dashboard" / "dashboard.html",
                  results_dir / "dashboard.html")

    record = run_experiment(cfg)
    print("\n--- Summary ---")
    print(f"  Exp {record.experiment_num} ({record.backbone})  Composite={record.composite:.4f} Status={record.status}")
    print(f"  Test F1: {record.test_primary:.4f}   Val F1: {record.val_primary:.4f}")
    for i, v in enumerate(record.per_fold_test):
        print(f"  Fold {i+1} test F1: {v:.4f}")


if __name__ == "__main__":
    main()
