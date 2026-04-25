"""Run Exp 5 — chronological holdout + entity-velocity features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from generalized_ml_autoresearch.core.runner import run_experiment, _load_config


def main():
    cfg = _load_config(HERE / "config_exp5.yaml")
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "trade_logs").mkdir(exist_ok=True)

    seed = json.loads((HERE / "seed_reasoning_exp5.json").read_text(encoding="utf-8"))
    ann_path = results_dir / "reasoning_annotations.json"
    existing = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    existing.update(seed)
    ann_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    record = run_experiment(cfg)
    print("\n--- Summary (Exp 5 — velocity features) ---")
    print(f"  Composite={record.composite:.4f} Status={record.status}")
    print(f"  Test AUC-ROC: {record.test_primary:.4f}   Val AUC-ROC: {record.val_primary:.4f}")
    print(f"  Exp 2 baseline:   0.5098   |   Exp 3 +temporal: 0.5116   |   Exp 4 -tss: 0.4960")
    print(f"  vs FDB AFD TFI ceiling: 0.636")
    print(f"  Time elapsed: {record.seconds_elapsed:.1f}s")
    print()
    print("  --- secondary metrics ---")
    for k, v in record.secondary_metrics.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
