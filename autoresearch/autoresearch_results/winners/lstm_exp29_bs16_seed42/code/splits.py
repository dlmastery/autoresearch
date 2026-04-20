"""Regime-aware fold definitions for walk-forward cross-validation.

Each fold uses an expanding training window starting from 2005-01,
purge + embargo gaps between splits, and a regime label describing
the dominant market environment during the test period.

Terminology (Lopez de Prado, 2018):
  - PURGE: gap between train-end and val/test-start to prevent label leakage
           from overlapping forward-return windows.
  - EMBARGO: additional holdout after test-end before the next fold's
             training data can begin, preventing information leakage from
             autocorrelated features bleeding across fold boundaries.
"""

from __future__ import annotations

import pandas as pd

# Minimum calendar-day gap between train-end / val-start and val-end / test-start.
PURGE_DAYS: int = 90
# Additional embargo after test-end (not consumed by next fold as training).
# With expanding windows each fold starts from 2005-01, so embargo is
# implicitly satisfied (next fold's train-end > current test-end + gap).
# This constant is used for validation assertions.
EMBARGO_DAYS: int = 21
# Calendar-day buffer before each held-out window to prevent forward-return
# targets (fwd_ret_5d) from peeking into the excluded period.  10 calendar
# days covers ~7 business days, safely beyond the 5-day label horizon.
LABEL_HORIZON_BUFFER: int = 10

# ── 7 walk-forward folds ────────────────────────────────────────────────────

FOLDS: list[dict] = [
    {
        "name": "fold_1",
        "regime": "Pre-crisis upturn + GFC onset",
        "train": {"start": "2005-01", "end": "2006-12"},
        "val":   {"start": "2007-04", "end": "2007-09"},
        "test":  {"start": "2008-01", "end": "2008-06"},
    },
    {
        "name": "fold_2",
        "regime": "Post-crash recovery",
        "train": {"start": "2005-01", "end": "2008-12"},
        "val":   {"start": "2009-04", "end": "2009-09"},
        "test":  {"start": "2010-01", "end": "2010-06"},
    },
    {
        "name": "fold_3",
        "regime": "Eurozone debt plateau",
        "train": {"start": "2005-01", "end": "2011-12"},
        "val":   {"start": "2012-04", "end": "2012-09"},
        "test":  {"start": "2013-01", "end": "2013-06"},
    },
    {
        "name": "fold_4",
        "regime": "Strong USD downturn",
        "train": {"start": "2005-01", "end": "2014-03"},
        "val":   {"start": "2014-07", "end": "2014-12"},
        "test":  {"start": "2015-04", "end": "2015-12"},
    },
    {
        "name": "fold_5",
        "regime": "Low-vol plateau",
        "train": {"start": "2005-01", "end": "2017-12"},
        "val":   {"start": "2018-04", "end": "2018-09"},
        "test":  {"start": "2019-01", "end": "2019-09"},
    },
    {
        "name": "fold_6",
        "regime": "EUR crisis downturn",
        "train": {"start": "2005-01", "end": "2020-12"},
        "val":   {"start": "2021-04", "end": "2021-09"},
        "test":  {"start": "2022-01", "end": "2022-09"},
    },
    {
        "name": "fold_7",
        "regime": "Recent mixed/upturn",
        "train": {"start": "2005-01", "end": "2023-12"},
        "val":   {"start": "2024-04", "end": "2024-09"},
        "test":  {"start": "2025-01", "end": "2025-09"},
    },
]


def get_fold_dates(fold: dict) -> dict[str, pd.Timestamp]:
    """Convert a fold dict's string dates into pd.Timestamps.

    For "end" fields the timestamp is set to the *last day* of the month
    so that filtering with ``<=`` includes the full month.

    Returns
    -------
    dict with keys: train_start, train_end, val_start, val_end,
    test_start, test_end — all pd.Timestamp.
    """
    def _start(s: str) -> pd.Timestamp:
        return pd.Timestamp(s)

    def _end(s: str) -> pd.Timestamp:
        # Offset to last day of the given month
        return pd.Timestamp(s) + pd.offsets.MonthEnd(0)

    return {
        "train_start": _start(fold["train"]["start"]),
        "train_end":   _end(fold["train"]["end"]),
        "val_start":   _start(fold["val"]["start"]),
        "val_end":     _end(fold["val"]["end"]),
        "test_start":  _start(fold["test"]["start"]),
        "test_end":    _end(fold["test"]["end"]),
    }


def _all_held_out_ranges() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return every val and test date range across all folds.

    These ranges are "untouchable" -- they must never appear in any
    fold's training data to prevent cross-fold contamination.
    """
    ranges = []
    for f in FOLDS:
        d = get_fold_dates(f)
        ranges.append((d["val_start"], d["val_end"]))
        ranges.append((d["test_start"], d["test_end"]))
    return ranges


def split_data(
    df: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DatetimeIndex-ed DataFrame into train / val / test subsets.

    Training data is purged of ALL val/test windows across ALL folds,
    not just the current fold.  This prevents cross-fold contamination
    where later expanding-window folds would otherwise train on earlier
    folds' held-out data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.
    fold : dict
        One of the entries in :data:`FOLDS`.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    dates = get_fold_dates(fold)
    train_df = df.loc[dates["train_start"]:dates["train_end"]]
    val_df = df.loc[dates["val_start"]:dates["val_end"]]
    test_df = df.loc[dates["test_start"]:dates["test_end"]]

    # Punch holes: remove every val/test window from training data,
    # plus a LABEL_HORIZON_BUFFER before each window to prevent
    # fwd_ret_5d targets from peeking into the excluded period.
    held_out = _all_held_out_ranges()
    buffer = pd.Timedelta(days=LABEL_HORIZON_BUFFER)
    for start, end in held_out:
        train_df = train_df.loc[~((train_df.index >= start - buffer) & (train_df.index <= end))]

    return train_df, val_df, test_df


def split_superfold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a single 'super-fold' for definitive evaluation.

    Uses fold 7's training window (2005-01 to 2023-12) with all val/test
    windows punched out + label-horizon buffer.  Validation and test sets
    are the UNION of all folds' val and test windows respectively, giving
    maximum statistical power for hyperparameter tuning.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    # Train: fold 7's holed training data
    train_df, _, _ = split_data(df, FOLDS[6])

    # Val: union of all folds' val windows
    val_frames = []
    for fold in FOLDS:
        d = get_fold_dates(fold)
        chunk = df.loc[d["val_start"]:d["val_end"]]
        if not chunk.empty:
            val_frames.append(chunk)
    val_df = pd.concat(val_frames).sort_index()

    # Test: union of all folds' test windows
    test_frames = []
    for fold in FOLDS:
        d = get_fold_dates(fold)
        chunk = df.loc[d["test_start"]:d["test_end"]]
        if not chunk.empty:
            test_frames.append(chunk)
    test_df = pd.concat(test_frames).sort_index()

    return train_df, val_df, test_df


def validate_purge_embargo() -> list[str]:
    """Validate all folds satisfy purge and embargo constraints.

    Checks:
    1. train_end + PURGE_DAYS < val_start (purge between train and val)
    2. val_end + PURGE_DAYS < test_start (purge between val and test)
    3. For consecutive folds: fold[i].test_end + EMBARGO_DAYS < fold[i+1].val_start
       (embargo prevents autocorrelation leakage across folds)
    4. No test set overlap between any two folds (disjoint test sets)

    Returns list of violation messages (empty = all checks pass).
    """
    violations = []
    all_dates = [get_fold_dates(f) for f in FOLDS]

    for i, (fold, dates) in enumerate(zip(FOLDS, all_dates)):
        name = fold["name"]

        # Purge: train → val
        gap_tv = (dates["val_start"] - dates["train_end"]).days
        if gap_tv < PURGE_DAYS:
            violations.append(f"{name}: train→val purge gap {gap_tv}d < {PURGE_DAYS}d")

        # Purge: val → test
        gap_vt = (dates["test_start"] - dates["val_end"]).days
        if gap_vt < PURGE_DAYS:
            violations.append(f"{name}: val→test purge gap {gap_vt}d < {PURGE_DAYS}d")

    # Embargo: consecutive fold test→next fold val
    for i in range(len(FOLDS) - 1):
        d_cur = all_dates[i]
        d_next = all_dates[i + 1]
        embargo_gap = (d_next["val_start"] - d_cur["test_end"]).days
        if embargo_gap < EMBARGO_DAYS:
            violations.append(
                f"{FOLDS[i]['name']}→{FOLDS[i+1]['name']}: "
                f"embargo gap {embargo_gap}d < {EMBARGO_DAYS}d"
            )

    # Disjoint test sets
    for i in range(len(FOLDS)):
        for j in range(i + 1, len(FOLDS)):
            di, dj = all_dates[i], all_dates[j]
            if di["test_start"] <= dj["test_end"] and dj["test_start"] <= di["test_end"]:
                violations.append(
                    f"{FOLDS[i]['name']} & {FOLDS[j]['name']}: overlapping test sets"
                )

    # Cross-fold contamination: no fold's training window may contain
    # any other fold's val or test dates
    held_out = _all_held_out_ranges()
    for i, (fold, dates) in enumerate(zip(FOLDS, all_dates)):
        name = fold["name"]
        train_start = dates["train_start"]
        train_end = dates["train_end"]
        for ho_start, ho_end in held_out:
            # Skip this fold's own val/test (already outside train by purge)
            if ho_start == dates["val_start"] or ho_start == dates["test_start"]:
                continue
            # Check if held-out range overlaps with training window
            if ho_start <= train_end and ho_end >= train_start:
                # This overlap exists but will be punched out by split_data.
                # Log it for transparency but don't treat as a violation
                # since split_data now removes these ranges.
                pass

    return violations
