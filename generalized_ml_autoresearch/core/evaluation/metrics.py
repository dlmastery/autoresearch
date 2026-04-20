"""Metric registry — pluggable named metrics for regression, classification, ranking, etc.

Each metric is a callable `(y_true, y_pred, sample_weight=None, **kwargs) -> float`.
Register user-defined metrics via `register_metric`.

The full_report helper computes task-specific metric bundles per the CLAUDE.md
"Traditional ML Metrics" section.
"""

from __future__ import annotations

from typing import Callable, Any
import math

import numpy as np

try:
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
        matthews_corrcoef, roc_auc_score, average_precision_score, log_loss,
        confusion_matrix,
    )
except ImportError:  # pragma: no cover
    mean_squared_error = None  # type: ignore[assignment]


MetricFn = Callable[..., float]
METRIC_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str):
    def _wrap(fn: MetricFn):
        METRIC_REGISTRY[name] = fn
        return fn
    return _wrap


# -------- regression --------

def _flatten_pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return y_true, y_pred


@register_metric("rmse")
def rmse(y_true, y_pred, sample_weight=None, **_):
    y_true, y_pred = _flatten_pair(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))


@register_metric("mae")
def mae(y_true, y_pred, sample_weight=None, **_):
    y_true, y_pred = _flatten_pair(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))


@register_metric("mape")
def mape(y_true, y_pred, sample_weight=None, **_):
    y_true, y_pred = _flatten_pair(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


@register_metric("r2")
def r2(y_true, y_pred, sample_weight=None, **_):
    y_true, y_pred = _flatten_pair(y_true, y_pred)
    return float(r2_score(y_true, y_pred, sample_weight=sample_weight))


@register_metric("ic")
def information_coefficient(y_true, y_pred, **_):
    """Spearman rank correlation — scale-free linear association."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) < 2:
        return float("nan")
    r_true = y_true.argsort().argsort().astype(float)
    r_pred = y_pred.argsort().argsort().astype(float)
    if r_true.std() == 0 or r_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(r_true, r_pred)[0, 1])


# -------- binary classification --------

@register_metric("accuracy")
def accuracy(y_true, y_pred, **_):
    return float(accuracy_score(y_true, y_pred))


@register_metric("precision")
def precision(y_true, y_pred, **_):
    return float(precision_score(y_true, y_pred, average="binary", zero_division=0))


@register_metric("recall")
def recall(y_true, y_pred, **_):
    return float(recall_score(y_true, y_pred, average="binary", zero_division=0))


@register_metric("f1")
def f1(y_true, y_pred, **_):
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))


@register_metric("f2")
def f2(y_true, y_pred, **_):
    return float(fbeta_score(y_true, y_pred, beta=2.0, average="binary", zero_division=0))


@register_metric("mcc")
def mcc(y_true, y_pred, **_):
    return float(matthews_corrcoef(y_true, y_pred))


@register_metric("auc_roc")
def auc_roc(y_true, y_score, **_):
    return float(roc_auc_score(y_true, y_score))


@register_metric("auc_pr")
def auc_pr(y_true, y_score, **_):
    return float(average_precision_score(y_true, y_score))


@register_metric("log_loss")
def log_loss_metric(y_true, y_score, **_):
    return float(log_loss(y_true, y_score, labels=[0, 1]))


# -------- multiclass --------

@register_metric("accuracy_multi")
def accuracy_multi(y_true, y_pred, **_):
    return float(accuracy_score(y_true, y_pred))


@register_metric("f1_macro")
def f1_macro(y_true, y_pred, **_):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("mcc_multi")
def mcc_multi(y_true, y_pred, **_):
    return float(matthews_corrcoef(y_true, y_pred))


# -------- time-series / finance (optional) --------

@register_metric("sharpe")
def sharpe(y_true, y_pred, annualization: float = 252.0, **_):
    """Strategy Sharpe using sign(pred) as the directional bet.
    Generalized from CLAUDE.md Trade-Level Win/Loss Logging section.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    strat = np.sign(y_pred) * y_true
    mu = strat.mean()
    sd = strat.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float(mu / sd * math.sqrt(annualization))


@register_metric("hit_rate")
def hit_rate(y_true, y_pred, **_):
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    nonzero = y_true != 0
    if not nonzero.any():
        return 0.5
    return float((np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])).mean())


# -------- dispatcher --------

def compute_metric(name: str, y_true, y_pred, **kwargs) -> float:
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric {name!r}. Registered: {list(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[name](y_true, y_pred, **kwargs)


# -------- task-specific full report --------

def full_report(task_type: str, y_true, y_pred, y_score=None, **kwargs) -> dict[str, float]:
    """Compute the canonical metric bundle for a given task type.

    For classification tasks, `y_pred` is the hard prediction and `y_score` is
    the predicted probability (for AUC / log-loss).

    This is the generalized version of CLAUDE.md's `trading_report()`.
    """
    report: dict[str, Any] = {"task_type": task_type, "n": int(len(y_true))}

    if task_type == "regression":
        report.update({
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "ic": information_coefficient(y_true, y_pred),
        })
    elif task_type == "binary_classification":
        report.update({
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1": f1(y_true, y_pred),
            "f2": f2(y_true, y_pred),
            "mcc": mcc(y_true, y_pred),
        })
        if y_score is not None:
            try:
                report["auc_roc"] = auc_roc(y_true, y_score)
                report["auc_pr"] = auc_pr(y_true, y_score)
                report["log_loss"] = log_loss_metric(y_true, y_score)
            except Exception as e:  # pragma: no cover
                report["auc_roc_error"] = str(e)
        # Confusion matrix counts
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel().tolist()
        report.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    elif task_type == "multiclass_classification":
        report.update({
            "accuracy": accuracy_multi(y_true, y_pred),
            "f1_macro": f1_macro(y_true, y_pred),
            "mcc": mcc_multi(y_true, y_pred),
        })
    elif task_type == "time_series_forecasting":
        report.update({
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "ic": information_coefficient(y_true, y_pred),
            "sharpe": sharpe(y_true, y_pred),
            "hit_rate": hit_rate(y_true, y_pred),
        })
    elif task_type == "ranking":
        # Fallback: Spearman IC + RMSE
        report.update({"rmse": rmse(y_true, y_pred), "ic": information_coefficient(y_true, y_pred)})
    elif task_type == "survival":
        # Users must add concordance index via a custom metric
        report.update({"rmse": rmse(y_true, y_pred)})
    else:
        # Unknown task: safe-default to every registered metric we can compute.
        for k, fn in METRIC_REGISTRY.items():
            try:
                report[k] = fn(y_true, y_pred)
            except Exception:
                continue
    return report
