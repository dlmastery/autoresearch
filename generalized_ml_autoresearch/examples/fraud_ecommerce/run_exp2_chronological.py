"""Run Exp 2 — chronological 80/20 holdout (mirrors FDB protocol)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from generalized_ml_autoresearch.core.runner import run_experiment, _load_config


def main():
    cfg = _load_config(HERE / "config_chronological.yaml")
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "trade_logs").mkdir(exist_ok=True)

    seed = json.loads((HERE / "seed_reasoning_exp2.json").read_text(encoding="utf-8"))
    ann_path = results_dir / "reasoning_annotations.json"
    existing = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    existing.update(seed)
    ann_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    record = run_experiment(cfg)
    print("\n--- Summary (Exp 2 — chronological holdout) ---")
    print(f"  Composite={record.composite:.4f} Status={record.status}")
    print(f"  Test AUC-ROC: {record.test_primary:.4f}   Val AUC-ROC: {record.val_primary:.4f}")
    print(f"  per-fold: {[round(v,4) for v in record.per_fold_test]}")
    print(f"  vs Exp 1: stratified CV test AUC = 0.7738")
    print(f"  vs FDB:   chronological 80/20 ceiling = 0.636")
    print(f"  Time elapsed: {record.seconds_elapsed:.1f}s")


if __name__ == "__main__":
    main()
