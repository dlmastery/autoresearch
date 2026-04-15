"""Tests for data.splits module."""

import itertools

import pandas as pd
import pytest

from autoresearch.data.splits import (
    FOLDS, PURGE_DAYS, LABEL_HORIZON_BUFFER,
    get_fold_dates, split_data, _all_held_out_ranges,
)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def test_seven_folds_defined():
    assert len(FOLDS) == 7


def test_purge_days_at_least_63():
    assert PURGE_DAYS >= 63


def test_each_fold_has_regime_label():
    for fold in FOLDS:
        assert "regime" in fold and isinstance(fold["regime"], str) and fold["regime"]


# ---------------------------------------------------------------------------
# No overlap within each fold
# ---------------------------------------------------------------------------

def test_no_overlap_within_folds():
    """Train, val, and test date ranges must not overlap within each fold."""
    for fold in FOLDS:
        dates = get_fold_dates(fold)
        # train must end before val starts
        assert dates["train_end"] < dates["val_start"], (
            f"Fold {fold['name']}: train overlaps val"
        )
        # val must end before test starts
        assert dates["val_end"] < dates["test_start"], (
            f"Fold {fold['name']}: val overlaps test"
        )


# ---------------------------------------------------------------------------
# Purge gap enforcement
# ---------------------------------------------------------------------------

def test_purge_gap_at_least_80_days():
    """Gap between train-end and val-start, and val-end and test-start,
    must be at least 80 calendar days."""
    for fold in FOLDS:
        dates = get_fold_dates(fold)
        train_val_gap = (dates["val_start"] - dates["train_end"]).days
        val_test_gap = (dates["test_start"] - dates["val_end"]).days
        assert train_val_gap >= 80, (
            f"Fold {fold['name']}: train-val gap = {train_val_gap} days"
        )
        assert val_test_gap >= 80, (
            f"Fold {fold['name']}: val-test gap = {val_test_gap} days"
        )


# ---------------------------------------------------------------------------
# Disjoint test sets across folds
# ---------------------------------------------------------------------------

def test_all_test_sets_disjoint():
    """Test date ranges must not overlap across different folds."""
    for a, b in itertools.combinations(FOLDS, 2):
        a_dates = get_fold_dates(a)
        b_dates = get_fold_dates(b)
        # Two intervals overlap iff start_a < end_b AND start_b < end_a
        overlap = (
            a_dates["test_start"] <= b_dates["test_end"]
            and b_dates["test_start"] <= a_dates["test_end"]
        )
        assert not overlap, (
            f"Test sets overlap: fold {a['name']} and fold {b['name']}"
        )


# ---------------------------------------------------------------------------
# get_fold_dates returns pd.Timestamp
# ---------------------------------------------------------------------------

def test_get_fold_dates_returns_timestamps():
    dates = get_fold_dates(FOLDS[0])
    for key in ("train_start", "train_end", "val_start", "val_end",
                "test_start", "test_end"):
        assert isinstance(dates[key], pd.Timestamp), (
            f"{key} is {type(dates[key])}, expected pd.Timestamp"
        )


# ---------------------------------------------------------------------------
# split_data integration test
# ---------------------------------------------------------------------------

def test_split_data_filters_correctly():
    """split_data should partition a DataFrame into train/val/test."""
    # Build a simple daily DataFrame spanning 2005 to 2026
    idx = pd.date_range("2005-01-01", "2026-01-01", freq="B", name="Date")
    df = pd.DataFrame({"Close": range(len(idx))}, index=idx)

    fold = FOLDS[0]
    train_df, val_df, test_df = split_data(df, fold)

    dates = get_fold_dates(fold)
    assert train_df.index.min() >= dates["train_start"]
    assert train_df.index.max() <= dates["train_end"]
    assert val_df.index.min() >= dates["val_start"]
    assert val_df.index.max() <= dates["val_end"]
    assert test_df.index.min() >= dates["test_start"]
    assert test_df.index.max() <= dates["test_end"]
    # All three should be non-empty given the wide date range
    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty


# ---------------------------------------------------------------------------
# Cross-fold contamination: training data must not contain held-out dates
# ---------------------------------------------------------------------------

def test_no_cross_fold_contamination():
    """No fold's training data may contain dates from any fold's val/test window."""
    idx = pd.date_range("2005-01-01", "2026-01-01", freq="B", name="Date")
    df = pd.DataFrame({"Close": range(len(idx))}, index=idx)

    held_out = _all_held_out_ranges()

    for fold in FOLDS:
        train_df, _, _ = split_data(df, fold)
        train_dates = set(train_df.index)
        for ho_start, ho_end in held_out:
            contaminated = {d for d in train_dates if ho_start <= d <= ho_end}
            assert len(contaminated) == 0, (
                f"{fold['name']}: training data contains {len(contaminated)} "
                f"dates from held-out window {ho_start.date()}..{ho_end.date()}"
            )


def test_label_horizon_buffer_enforced():
    """Training data must not include dates within LABEL_HORIZON_BUFFER
    calendar days before any held-out window, preventing fwd_ret_5d
    targets from peeking into excluded periods."""
    idx = pd.date_range("2005-01-01", "2026-01-01", freq="B", name="Date")
    df = pd.DataFrame({"Close": range(len(idx))}, index=idx)

    held_out = _all_held_out_ranges()
    buffer = pd.Timedelta(days=LABEL_HORIZON_BUFFER)

    for fold in FOLDS:
        train_df, _, _ = split_data(df, fold)
        train_dates = set(train_df.index)
        for ho_start, ho_end in held_out:
            buffered_start = ho_start - buffer
            in_buffer = {d for d in train_dates if buffered_start <= d < ho_start}
            assert len(in_buffer) == 0, (
                f"{fold['name']}: training data contains {len(in_buffer)} "
                f"dates in label-horizon buffer before "
                f"{ho_start.date()} (buffer starts {buffered_start.date()})"
            )
