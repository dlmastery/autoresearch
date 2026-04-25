"""Split protocols — all return a list of FoldAssignment objects.

Generalized from the FX SuperFoldSplit. Every splitter exposes the same
interface so the runner can call them uniformly:

    splitter = create_splitter(name, **kwargs)
    fold_assignments = splitter.split(indices_or_df)
    validate_no_overlap(fold_assignments)
    for fa in fold_assignments:
        X_train = X[fa.train_idx]; X_val = X[fa.val_idx]; X_test = X[fa.test_idx]

Invariant enforced by `validate_no_overlap`:
  for every fold, train ∩ val = ∅; train ∩ test = ∅; val ∩ test = ∅.
Cross-fold invariants depend on the splitter (e.g. SuperFold: train excludes
all folds' val+test; KFold: only per-fold disjointness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from sklearn.model_selection import (
        KFold as _SkKFold,
        StratifiedKFold as _SkStratifiedKFold,
        GroupKFold as _SkGroupKFold,
        TimeSeriesSplit as _SkTimeSeriesSplit,
    )
except ImportError:  # pragma: no cover
    _SkKFold = None  # type: ignore[assignment]


@dataclass
class FoldAssignment:
    """One (train, val, test) index triple."""

    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    regime_label: str = ""  # optional human-readable label (e.g. "GFC recovery")

    def n_train(self) -> int:
        return len(self.train_idx)

    def n_val(self) -> int:
        return len(self.val_idx)

    def n_test(self) -> int:
        return len(self.test_idx)


class _BaseSplitter:
    """Abstract splitter. Subclasses implement `split(n_samples)` -> list[FoldAssignment]."""

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:  # pragma: no cover
        raise NotImplementedError


class HoldoutSplit(_BaseSplitter):
    """Single train/val/test split by fraction (or by contiguous slice for time-series).

    Parameters
    ----------
    val_fraction, test_fraction: floats in (0, 1). train gets the remainder.
    order: "shuffle" (default, random) or "time" (contiguous slices — test is last block).
    seed: RNG seed for shuffle order.
    """

    def __init__(self, val_fraction: float = 0.15, test_fraction: float = 0.15,
                 order: str = "shuffle", seed: int = 0,
                 min_train_idx: int = 0):
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.order = order
        self.seed = seed
        # min_train_idx: drop training rows with index < this value. Lets you test
        # recency hypotheses WITHOUT changing val_idx or test_idx (anti-reward-hacking).
        self.min_train_idx = int(min_train_idx)

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        n_test = int(round(n_samples * self.test_fraction))
        n_val = int(round(n_samples * self.val_fraction))
        indices = np.arange(n_samples)
        if self.order == "shuffle":
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(indices)
            test_idx = indices[:n_test]
            val_idx = indices[n_test : n_test + n_val]
            train_idx = indices[n_test + n_val :]
        elif self.order == "time":
            train_idx = indices[: n_samples - n_val - n_test]
            val_idx = indices[n_samples - n_val - n_test : n_samples - n_test]
            test_idx = indices[n_samples - n_test :]
        else:
            raise ValueError(f"HoldoutSplit unknown order={self.order!r}")
        if self.min_train_idx > 0:
            train_idx = train_idx[train_idx >= self.min_train_idx]
        return [FoldAssignment(0, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), "holdout")]


class KFoldSplit(_BaseSplitter):
    """Standard k-fold. Each fold's test is the held-out slice; val is a portion of the
    remaining training data (default 10% of train as val, random)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 0,
                 val_fraction_of_train: float = 0.1):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.val_fraction_of_train = val_fraction_of_train

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        if _SkKFold is None:
            raise ImportError("scikit-learn not installed; KFoldSplit requires it.")
        kf = _SkKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        out: list[FoldAssignment] = []
        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(indices)):
            shuffled = rng.permutation(trainval_idx)
            n_val = max(1, int(round(len(shuffled) * self.val_fraction_of_train)))
            val_idx = shuffled[:n_val]
            train_idx = shuffled[n_val:]
            out.append(
                FoldAssignment(
                    fold_id, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx),
                    f"kfold-{fold_id}"
                )
            )
        return out


class StratifiedKFoldSplit(_BaseSplitter):
    """Stratified k-fold (for classification). Requires y on `split()`."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 0,
                 val_fraction_of_train: float = 0.1):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.val_fraction_of_train = val_fraction_of_train

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        if _SkStratifiedKFold is None:
            raise ImportError("scikit-learn required.")
        if y is None:
            raise ValueError("StratifiedKFoldSplit requires y")
        skf = _SkStratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        out: list[FoldAssignment] = []
        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)
        for fold_id, (trainval_idx, test_idx) in enumerate(skf.split(indices, y)):
            shuffled = rng.permutation(trainval_idx)
            n_val = max(1, int(round(len(shuffled) * self.val_fraction_of_train)))
            val_idx = shuffled[:n_val]
            train_idx = shuffled[n_val:]
            out.append(
                FoldAssignment(fold_id, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx),
                               f"stratified-{fold_id}")
            )
        return out


class GroupKFoldSplit(_BaseSplitter):
    """Grouped k-fold — samples with the same group never cross splits."""

    def __init__(self, n_splits: int = 5, val_fraction_of_train: float = 0.1, seed: int = 0):
        self.n_splits = n_splits
        self.val_fraction_of_train = val_fraction_of_train
        self.seed = seed

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        if _SkGroupKFold is None:
            raise ImportError("scikit-learn required.")
        if groups is None:
            raise ValueError("GroupKFoldSplit requires groups")
        gkf = _SkGroupKFold(n_splits=self.n_splits)
        out: list[FoldAssignment] = []
        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_samples)
        for fold_id, (trainval_idx, test_idx) in enumerate(gkf.split(indices, y, groups)):
            shuffled = rng.permutation(trainval_idx)
            n_val = max(1, int(round(len(shuffled) * self.val_fraction_of_train)))
            val_idx = shuffled[:n_val]
            train_idx = shuffled[n_val:]
            out.append(
                FoldAssignment(fold_id, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx),
                               f"group-{fold_id}")
            )
        return out


class TimeSeriesSplit(_BaseSplitter):
    """Expanding-window time-series split. Test is always future of train/val."""

    def __init__(self, n_splits: int = 5, val_fraction_of_train: float = 0.1,
                 gap: int = 0):
        self.n_splits = n_splits
        self.val_fraction_of_train = val_fraction_of_train
        self.gap = gap

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        if _SkTimeSeriesSplit is None:
            raise ImportError("scikit-learn required.")
        tss = _SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        out: list[FoldAssignment] = []
        indices = np.arange(n_samples)
        for fold_id, (trainval_idx, test_idx) in enumerate(tss.split(indices)):
            trainval_idx = np.sort(trainval_idx)
            n_val = max(1, int(round(len(trainval_idx) * self.val_fraction_of_train)))
            val_idx = trainval_idx[-n_val:]
            train_idx = trainval_idx[:-n_val]
            out.append(
                FoldAssignment(fold_id, train_idx, val_idx, test_idx, f"ts-fold-{fold_id}")
            )
        return out


class WalkForwardSplit(_BaseSplitter):
    """Fixed-window walk-forward with purge gap and embargo (generalized from FX splits.py).

    Parameters
    ----------
    n_splits: number of walk-forward folds
    train_window: rows
    val_window: rows
    test_window: rows
    purge: rows between train and val (for look-ahead protection)
    embargo: rows between test and the next fold's train
    label_horizon_buffer: additional rows of label-leakage buffer
    """

    def __init__(self, n_splits: int, train_window: int, val_window: int, test_window: int,
                 purge: int = 0, embargo: int = 0, label_horizon_buffer: int = 0):
        self.n_splits = n_splits
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.purge = purge
        self.embargo = embargo
        self.label_horizon_buffer = label_horizon_buffer

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        out: list[FoldAssignment] = []
        stride = self.val_window + self.test_window + self.embargo
        for fold_id in range(self.n_splits):
            # fold_id's test end must fit within n_samples
            test_end = n_samples - fold_id * stride
            test_start = test_end - self.test_window
            val_end = test_start - self.purge
            val_start = val_end - self.val_window
            train_end = val_start - self.label_horizon_buffer
            train_start = train_end - self.train_window
            if train_start < 0:
                break
            fa = FoldAssignment(
                fold_id,
                np.arange(train_start, train_end),
                np.arange(val_start, val_end),
                np.arange(test_start, test_end),
                f"walk-forward-{fold_id}",
            )
            out.append(fa)
        return list(reversed(out))  # oldest fold first


class SuperFoldSplit(_BaseSplitter):
    """Generalized super-fold (FX-style): train = all_data − union(val+test+buffers).

    For a list of pre-specified (val, test) windows (each a `slice(start, stop)`):
      - val_set = concat of all val windows
      - test_set = concat of all test windows
      - train_set = everything except val, test, and buffer zones before each
    This produces ONE fold assignment, not K — the "super-fold" pattern.
    """

    def __init__(self, val_windows: Sequence[tuple[int, int]],
                 test_windows: Sequence[tuple[int, int]],
                 label_horizon_buffer: int = 0):
        if len(val_windows) != len(test_windows):
            raise ValueError("val_windows and test_windows must have equal length")
        self.val_windows = list(val_windows)
        self.test_windows = list(test_windows)
        self.label_horizon_buffer = label_horizon_buffer

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        all_idx = np.arange(n_samples)
        excluded = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)

        for (v_start, v_end), (t_start, t_end) in zip(self.val_windows, self.test_windows):
            val_mask[v_start:v_end] = True
            test_mask[t_start:t_end] = True
            excluded[v_start:v_end] = True
            excluded[t_start:t_end] = True
            # Buffer before each excluded window
            if self.label_horizon_buffer > 0:
                excluded[max(0, v_start - self.label_horizon_buffer) : v_start] = True
                excluded[max(0, t_start - self.label_horizon_buffer) : t_start] = True

        train_idx = all_idx[~excluded]
        val_idx = all_idx[val_mask]
        test_idx = all_idx[test_mask]
        return [FoldAssignment(0, train_idx, val_idx, test_idx, "super-fold")]


SPLIT_REGISTRY = {
    "holdout": HoldoutSplit,
    "kfold": KFoldSplit,
    "stratified_kfold": StratifiedKFoldSplit,
    "group_kfold": GroupKFoldSplit,
    "time_series_split": TimeSeriesSplit,
    "walk_forward": WalkForwardSplit,
    "super_fold": SuperFoldSplit,
}


def create_splitter(name: str, **kwargs) -> _BaseSplitter:
    if name not in SPLIT_REGISTRY:
        raise ValueError(f"Unknown split protocol {name!r}; must be one of {list(SPLIT_REGISTRY)}")
    return SPLIT_REGISTRY[name](**kwargs)


def validate_no_overlap(fold_assignments: Iterable[FoldAssignment]) -> dict[str, int]:
    """Assert per-fold disjointness. Returns a report dict. Raises AssertionError on violation.

    This is the programmatic enforcement of CLAUDE.md Data Integrity rule
    "NEVER include any fold's val or test dates in any fold's training data."
    """
    report = {"folds_checked": 0, "total_overlaps": 0}
    for fa in fold_assignments:
        train = set(int(x) for x in fa.train_idx)
        val = set(int(x) for x in fa.val_idx)
        test = set(int(x) for x in fa.test_idx)
        tv = train & val
        tt = train & test
        vt = val & test
        if tv or tt or vt:
            report["total_overlaps"] += len(tv) + len(tt) + len(vt)
            raise AssertionError(
                f"Fold {fa.fold_id} has overlap: train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
            )
        report["folds_checked"] += 1
    return report
