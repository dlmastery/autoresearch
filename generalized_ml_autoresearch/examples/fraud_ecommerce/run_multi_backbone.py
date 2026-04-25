"""Run Exp 7 (LightGBM), Exp 8 (CatBoost), Exp 9 (MLP) — multi-backbone phase.

Honors the CLAUDE.md Per-Backbone Mandate. One config per backbone, all on the
same chronological-holdout protocol with the velocity-feature set from Exp 6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from generalized_ml_autoresearch.core.runner import run_experiment, _load_config


def run_one(config_name: str, seed_name: str, label: str):
    cfg = _load_config(HERE / config_name)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "trade_logs").mkdir(exist_ok=True)

    seed = json.loads((HERE / seed_name).read_text(encoding="utf-8"))
    ann_path = results_dir / "reasoning_annotations.json"
    existing = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    existing.update(seed)
    ann_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {label} ===")
    record = run_experiment(cfg)
    print(f"  Composite={record.composite:.4f} Test AUC={record.test_primary:.4f} "
          f"Val AUC={record.val_primary:.4f} Status={record.status} ({record.seconds_elapsed:.1f}s)")
    return record


def main():
    records = []
    for cfg_file, seed_file, label in [
        ("config_exp7_lgbm.yaml", "seed_reasoning_exp7.json", "Exp 7 — LightGBM"),
        ("config_exp8_catboost.yaml", "seed_reasoning_exp8.json", "Exp 8 — CatBoost"),
        ("config_exp9_mlp.yaml", "seed_reasoning_exp9.json", "Exp 9 — MLP"),
    ]:
        try:
            records.append(run_one(cfg_file, seed_file, label))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\n=== Multi-backbone summary ===")
    for r in records:
        print(f"  Exp {r.experiment_num} ({r.backbone}): composite={r.composite:.4f} "
              f"test_auc={r.test_primary:.4f} val_auc={r.val_primary:.4f} {r.status}")


if __name__ == "__main__":
    main()
