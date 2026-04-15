"""Tests for evaluation.metrics module."""

import numpy as np
import pytest

from autoresearch.evaluation.metrics import (
    TRADING_DAYS_PER_YEAR,
    average_sharpe_across_folds,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------

def test_trading_days_per_year():
    assert TRADING_DAYS_PER_YEAR == 252


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

def test_positive_returns_positive_sharpe():
    daily = np.array([0.001] * 100)
    assert sharpe_ratio(daily) > 0.0


def test_zero_returns_zero_sharpe():
    daily = np.zeros(100)
    assert sharpe_ratio(daily) == 0.0


def test_negative_returns_negative_sharpe():
    daily = np.array([-0.001] * 100)
    assert sharpe_ratio(daily) < 0.0


def test_annualized_range_check():
    """mean=0.001, std=0.01 -> Sharpe ~ 0.001/0.01 * sqrt(252) ~ 1.587."""
    rng = np.random.default_rng(42)
    daily = rng.normal(loc=0.001, scale=0.01, size=10_000)
    sr = sharpe_ratio(daily)
    assert 1.3 < sr < 1.9, f"Expected ~1.59, got {sr}"


def test_empty_returns_zero_sharpe():
    assert sharpe_ratio(np.array([])) == 0.0


def test_single_return_zero_std():
    """A single return has zero std; should return 0.0 gracefully."""
    assert sharpe_ratio(np.array([0.005])) == 0.0


# ---------------------------------------------------------------------------
# average_sharpe_across_folds
# ---------------------------------------------------------------------------

def test_average_across_multiple_folds():
    fold1 = np.array([0.001] * 100)
    fold2 = np.array([0.002] * 100)
    avg = average_sharpe_across_folds([fold1, fold2])
    sr1 = sharpe_ratio(fold1)
    sr2 = sharpe_ratio(fold2)
    assert avg == pytest.approx((sr1 + sr2) / 2, rel=1e-6)


def test_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        average_sharpe_across_folds([])
