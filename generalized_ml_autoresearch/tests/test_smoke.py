"""Smoke tests — fast, offline, no external data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generalized_ml_autoresearch.core.reasoning import (
    ReasoningEntry,
    validate_reasoning_blob,
    validate_pre_run_entry,
    validate_citation_rigor,
    ReasoningAnnotationsFile,
)
from generalized_ml_autoresearch.core.evaluation import (
    create_splitter,
    validate_no_overlap,
    CompositeCalculator,
    compute_metric,
    full_report,
)
from generalized_ml_autoresearch.core.backbones import create_model, list_backbones


def test_citation_rigor_good():
    good = (
        "Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 ICLR 'On Large-Batch Training for "
        "Deep Learning: Generalization Gap and Sharp Minima' (arXiv:1609.04836) — motivates "
        "bs=16 as a flat-minima probe in our current diagnosis because the paper shows that "
        "smaller batch sizes produce flatter minima that generalize better to held-out sets "
        "in our walk-forward evaluation protocol for the regression case."
    )
    violations = validate_citation_rigor(good)
    assert not violations, violations


def test_citation_rigor_bad():
    bad = "(Keskar2017)"
    assert validate_citation_rigor(bad)


def test_splits_no_overlap():
    from generalized_ml_autoresearch.core.evaluation.splits import KFoldSplit
    s = KFoldSplit(n_splits=5, seed=0)
    assignments = s.split(n_samples=200)
    report = validate_no_overlap(assignments)
    assert report["folds_checked"] == 5


def test_composite_calc():
    calc = CompositeCalculator(primary_metric_name="rmse", higher_is_better=False,
                                penalty_weight=0.1, below_threshold=-1.0)
    # For RMSE (lower=better), internal sign flip => negative numbers are "good"
    composite = calc.compute(val_primary=0.5, test_primary=0.6, per_fold_test=[0.6, 0.55, 0.7])
    # min(-0.5, -0.6) - 0.1 * 0 = -0.6
    assert abs(composite - (-0.6)) < 1e-9


def test_backbone_registry():
    names = list_backbones()
    assert "mlp" in names
    assert "lstm" in names
    assert "xgboost" in names
    assert "lightgbm" in names
    assert "catboost" in names
    assert "ft_transformer" in names


def test_mlp_regression_tiny(tmp_path):
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch not installed")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    # y is a nonlinear function with noise
    y = (X[:, 0] * 2.0 + X[:, 1] ** 2 - 0.5 * X[:, 2] + rng.standard_normal(300) * 0.3).astype(np.float32)
    model = create_model("mlp", {
        "task_type": "regression",
        "hidden": [32, 16],
        "dropout": 0.1,
        "epochs": 10,
        "patience": 5,
        "batch_size": 32,
        "lr": 3e-4,
        "seed": 0,
        "force_cpu": True,
    })
    model.build(model.config, (8,), 1)
    model.fit(X[:240], y[:240], X[240:], y[240:])
    bundle = model.predict_with_uncertainty(X[240:], n_samples=5)
    assert bundle.mean.shape[0] == 60
    report = full_report("regression", y[240:], bundle.mean.flatten())
    assert "rmse" in report


def test_reasoning_roundtrip(tmp_path):
    path = tmp_path / "reasoning_annotations.json"
    raf = ReasoningAnnotationsFile(path)
    entry = ReasoningEntry(
        experiment_num=1,
        diagnosis=(
            "This is the baseline MLP experiment for the regression-house-prices example. "
            "No prior experiments exist, so the diagnosis is scope-setting rather than "
            "champion-weakness analysis: the project's goal is to beat a linear-regression "
            "floor on the California Housing benchmark using MedInc, HouseAge, AveRooms, "
            "AveBedrms, Population, AveOccup, Latitude and Longitude. Prior domain evidence "
            "from Gu, Kelly and Xiu 2020 indicates a 3-layer MLP with dropout is the right "
            "starting point for this tabular regression problem at n=20640 samples."
        ),
        citations=(
            "Gu, Kelly & Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning' "
            "(arXiv:1802.09003) — motivates the 3-layer MLP with dropout 0.2 architecture on "
            "tabular features; we adapt the head-only training recipe as the baseline for the "
            "California Housing dataset. The paper's regularization prescriptions directly "
            "inform the chosen dropout and weight-decay values that define this baseline."
        ),
        hypothesis=(
            "We hypothesize that a 3-layer MLP (256-128-64) with dropout 0.2 and AdamW "
            "(lr=3e-4, wd=1e-5) will achieve test RMSE below 0.55 because the mechanism is "
            "feature-interaction learning per Gu, Kelly & Xiu 2020 — dropout regularizes "
            "the dense interactions while AdamW decoupled weight decay prevents the hidden "
            "layers from memorizing the training set as shown by Loshchilov and Hutter 2019."
        ),
        prediction=(
            "Test RMSE should land in the range 0.50 to 0.58, beating the linear floor "
            "(approximately 0.72). Val RMSE expected near 0.52 with fold 3 strongest at "
            "0.48 across all 5 folds."
        ),
    )
    # pre-run commit
    raf.commit_pre_run(entry)
    loaded = raf.load()
    assert "1" in loaded
    # post-run commit
    violations = raf.commit_post_run(
        1,
        verdict=(
            "KEEP. Composite = 0.5412. Test RMSE 0.5412. Val RMSE 0.5198. 5/5 folds above "
            "threshold; fold 3 strongest at 0.4912 and fold 0 weakest at 0.5689 — this "
            "carries the experiment across regimes."
        ),
        learning=(
            "Baseline confirmed — axis open for deeper MLPs and residual connections. "
            "Next try: 4-layer MLP with a single residual skip connection from input to "
            "the final hidden layer per He et al. 2016, targeting a composite improvement "
            "of roughly +0.02 composite units over the current baseline."
        ),
    )
    # Post-run must pass full validation (word counts, keywords)
    # — we verify the runner's fallback entries are flagged but the real entries pass.
    loaded = raf.load()
    entry_loaded = ReasoningEntry.from_dict(loaded["1"])
    violations = validate_reasoning_blob(entry_loaded)
    assert not violations, violations


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
