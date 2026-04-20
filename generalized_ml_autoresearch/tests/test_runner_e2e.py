"""End-to-end runner test with a tiny synthetic regression task."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip("torch not installed", allow_module_level=True)


def test_runner_e2e(tmp_path):
    from generalized_ml_autoresearch.core.runner import run_experiment
    from generalized_ml_autoresearch.core.reasoning import (
        ReasoningAnnotationsFile, ReasoningEntry,
    )

    # 1. Prepare a tiny synthetic CSV
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 6))
    y = X[:, 0] * 2.0 + X[:, 1] ** 2 - 0.5 * X[:, 2] + rng.standard_normal(n) * 0.3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    results_dir = tmp_path / "results"

    # 2. Author the pre-run reasoning annotation (must pass validation)
    raf = ReasoningAnnotationsFile(results_dir / "reasoning_annotations.json")
    entry = ReasoningEntry(
        experiment_num=1,
        diagnosis=(
            "Baseline synthetic regression experiment for end-to-end plumbing validation. "
            "No prior experiments exist — we establish a linear-plus-nonlinear floor on tiny "
            "synthetic data with 6 features and 200 rows. The target is a known function so "
            "we can verify the runner end-to-end plumbing: reasoning gate, split validator, "
            "backbone build+fit+predict+save, per-prediction CSV, best_model write, and "
            "reasoning post-run fallback all fire correctly in the expected sequence with "
            "valid output files in the expected locations."
        ),
        citations=(
            "Gu, Kelly & Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning' "
            "(arXiv:1802.09003) — motivates MLP-with-dropout as the baseline floor for small "
            "tabular regression problems; we scale the architecture to fit the tiny n=200 by "
            "trimming hidden widths to [16,8] because the paper's full-scale network would overfit."
        ),
        hypothesis=(
            "We hypothesize that a 2-layer MLP (16-8) with dropout 0.1 and AdamW lr=3e-4 "
            "will achieve test RMSE below 0.80 on this synthetic task because the mechanism "
            "is that the target has a quadratic component in feature 1 which an MLP can "
            "approximate via ReLU segments per standard universal approximation results."
        ),
        prediction=(
            "Test RMSE should land in the range 0.50 to 0.80, with all 3 CV folds above the "
            "0.0 RMSE floor. Composite expected between -0.80 and -0.50."
        ),
    )
    raf.commit_pre_run(entry)

    # 3. Run
    config = {
        "paths": {"results_dir": str(results_dir)},
        "task_type": "regression",
        "primary_metric": "rmse",
        "backbone": "mlp",
        "backbone_config": {
            "hidden": [16, 8], "dropout": 0.1, "epochs": 10, "patience": 5,
            "batch_size": 32, "lr": 3e-4, "seed": 0, "force_cpu": True,
            "uncertainty_samples": 3,
        },
        "data": {
            "format": "csv",
            "path": str(csv_path),
            "feature_columns": [f"f{i}" for i in range(6)],
            "target_columns": ["target"],
        },
        "split": {"name": "kfold", "n_splits": 3, "seed": 0},
        "composite": {"higher_is_better": False, "penalty_weight": 0.1, "below_threshold": -10.0},
        "description": "synthetic baseline MLP",
        "seed": 0,
    }
    record = run_experiment(config)

    # 4. Assertions
    assert record.experiment_num == 1
    assert record.status in ("KEEP", "DISCARD")
    assert (results_dir / "experiment_log.jsonl").exists()
    assert (results_dir / "trade_logs" / "exp1_predictions.csv").exists()
    assert (results_dir / "trade_logs" / "exp1_prediction_summary.json").exists()
    if record.status == "KEEP":
        assert (results_dir / "best_config.json").exists()
        assert (results_dir / "best_model.pt").exists()

    # 5. Reasoning post-run should have verdict and learning
    loaded = raf.load()
    assert "1" in loaded
    e = loaded["1"]
    assert e["verdict"], "verdict should be populated by runner"
    assert e["learning"], "learning should be populated by runner"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
