"""Tests for evaluation.leakage_check module."""

import numpy as np
import pytest

from autoresearch.data.splits import FOLDS
from autoresearch.evaluation.leakage_check import check_split_gaps, check_scaler_isolation


# ---------------------------------------------------------------------------
# check_split_gaps — actual FOLDS pass
# ---------------------------------------------------------------------------

def test_actual_folds_pass_gap_check():
    """Our real FOLDS configuration must have no gap violations."""
    violations = check_split_gaps(FOLDS)
    assert violations == [], f"Unexpected violations: {violations}"


# ---------------------------------------------------------------------------
# check_split_gaps — bad fold detected
# ---------------------------------------------------------------------------

def test_bad_fold_detected():
    """A fold with only a 14-day gap between val-end and test-start
    must produce at least one violation string."""
    bad_folds = [
        {
            "name": "bad_fold",
            "regime": "test",
            "train": {"start": "2010-01", "end": "2010-06"},
            "val":   {"start": "2010-07", "end": "2010-09"},
            # test starts just 14 days after val end (Sep 30 -> Oct 14)
            "test":  {"start": "2010-10", "end": "2010-12"},
        }
    ]
    violations = check_split_gaps(bad_folds)
    assert len(violations) >= 1
    # The violation message should mention the bad fold
    assert any("bad_fold" in v for v in violations)


# ---------------------------------------------------------------------------
# check_scaler_isolation
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_only():
    """Scaler must learn mean from train (~3.0), not from val/test (~100)."""
    rng = np.random.RandomState(42)
    train = rng.normal(loc=3.0, scale=0.5, size=(200, 2))
    val = rng.normal(loc=100.0, scale=0.5, size=(50, 2))
    test = rng.normal(loc=100.0, scale=0.5, size=(50, 2))

    scaler, val_scaled, test_scaled = check_scaler_isolation(train, val, test)

    # scaler.mean_ should reflect training data (~3.0), not val/test
    assert scaler.mean_ == pytest.approx(3.0, abs=0.5)

    # val_scaled values should be large (shifted far from 0) because
    # 100 - 3 ~ 97, divided by small std ~ 0.5 => ~194
    assert np.abs(val_scaled).mean() > 50.0
    assert np.abs(test_scaled).mean() > 50.0
