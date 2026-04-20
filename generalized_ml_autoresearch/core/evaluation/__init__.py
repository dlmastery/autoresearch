"""Evaluation subsystem — splits, metrics, composite, uncertainty."""

from .splits import (
    HoldoutSplit,
    KFoldSplit,
    StratifiedKFoldSplit,
    GroupKFoldSplit,
    TimeSeriesSplit,
    WalkForwardSplit,
    SuperFoldSplit,
    FoldAssignment,
    create_splitter,
    validate_no_overlap,
)
from .metrics import METRIC_REGISTRY, compute_metric, full_report
from .composite import CompositeCalculator
from .uncertainty import mc_dropout_predict, ensemble_predict, tree_quantile_predict

__all__ = [
    "HoldoutSplit",
    "KFoldSplit",
    "StratifiedKFoldSplit",
    "GroupKFoldSplit",
    "TimeSeriesSplit",
    "WalkForwardSplit",
    "SuperFoldSplit",
    "FoldAssignment",
    "create_splitter",
    "validate_no_overlap",
    "METRIC_REGISTRY",
    "compute_metric",
    "full_report",
    "CompositeCalculator",
    "mc_dropout_predict",
    "ensemble_predict",
    "tree_quantile_predict",
]
