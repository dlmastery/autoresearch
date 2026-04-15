"""Leakage detection utilities for walk-forward cross-validation.

Checks that purge gaps between splits are respected and that
feature scaling is fit on training data only.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..data.splits import get_fold_dates, _all_held_out_ranges

# Minimum calendar-day gap required between adjacent splits.
MIN_GAP_DAYS: int = 80


def check_split_gaps(folds: list[dict]) -> list[str]:
    """Check that purge gaps between train-val and val-test are >= MIN_GAP_DAYS.

    Parameters
    ----------
    folds : list[dict]
        List of fold dictionaries (same schema as ``data.splits.FOLDS``).

    Returns
    -------
    list[str]
        List of violation description strings.  Empty list means all folds
        pass the gap check.
    """
    violations: list[str] = []
    for fold in folds:
        dates = get_fold_dates(fold)
        name = fold.get("name", "<unnamed>")

        train_val_gap = (dates["val_start"] - dates["train_end"]).days
        val_test_gap = (dates["test_start"] - dates["val_end"]).days

        if train_val_gap < MIN_GAP_DAYS:
            violations.append(
                f"{name}: train-val gap is {train_val_gap} days "
                f"(minimum {MIN_GAP_DAYS})"
            )
        if val_test_gap < MIN_GAP_DAYS:
            violations.append(
                f"{name}: val-test gap is {val_test_gap} days "
                f"(minimum {MIN_GAP_DAYS})"
            )

    return violations


def check_cross_fold_contamination(
    folds: list[dict],
    train_indices: dict[str, set],
) -> list[str]:
    """Verify no fold's training data contains dates from any fold's val/test.

    Parameters
    ----------
    folds : list[dict]
        Fold definitions.
    train_indices : dict[str, set]
        Mapping of fold name to the set of dates (as strings or timestamps)
        actually present in that fold's training data after splitting.

    Returns
    -------
    list[str]
        Violation messages. Empty = clean.
    """
    violations = []
    held_out = _all_held_out_ranges()
    for fold in folds:
        name = fold.get("name", "<unnamed>")
        train_dates = train_indices.get(name, set())
        if not train_dates:
            continue
        for ho_start, ho_end in held_out:
            contaminated = {d for d in train_dates if ho_start <= d <= ho_end}
            if contaminated:
                violations.append(
                    f"{name}: training data contains {len(contaminated)} dates "
                    f"from held-out window {ho_start.date()}..{ho_end.date()}"
                )
    return violations


def check_scaler_isolation(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit a StandardScaler on *train* only, then transform val and test.

    This ensures no information from validation or test data leaks into
    the scaling parameters.

    Parameters
    ----------
    train, val, test : np.ndarray
        2-D arrays with shape ``(n_samples, n_features)``.

    Returns
    -------
    (scaler, val_scaled, test_scaled)
        The fitted scaler and the transformed validation / test arrays.
    """
    scaler = StandardScaler()
    scaler.fit(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return scaler, val_scaled, test_scaled
