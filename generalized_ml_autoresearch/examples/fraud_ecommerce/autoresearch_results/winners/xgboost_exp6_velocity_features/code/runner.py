"""Generic experiment runner — one experiment per invocation.

Invocation:

    python -m generalized_ml_autoresearch.core.runner \
        --config path/to/config.yaml \
        --description "baseline MLP with dropout=0.2" \
        --seed 0

Or programmatically:

    from generalized_ml_autoresearch.core.runner import run_experiment
    record = run_experiment(config_dict, reasoning_dir=...)

The runner is the ONLY place where the ownership contract from CLAUDE.md's
"Dashboard Files Update Mandate" is executed:

  - experiment_log.jsonl       (append)
  - best_config.json           (overwrite if new global champion)
  - best_model.pt              (overwrite if new global champion)
  - trade_logs/exp<N>_predictions.csv    (create)
  - trade_logs/exp<N>_prediction_summary.json (create)
  - reasoning_annotations.json (fallback only — pre-run entry must already exist)

It does NOT update: research_journal.md, experiment_summary.md, checkpoint.md,
winners/*, or dashboard.html. Those are Claude's responsibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Absolute package imports so this also works when run as a script.
# Support both `python -m generalized_ml_autoresearch.core.runner` and
# `python path/to/runner.py` invocations.
if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generalized_ml_autoresearch.core.backbones import create_model, list_backbones
from generalized_ml_autoresearch.core.checkpoint import ExperimentRecord
from generalized_ml_autoresearch.core.evaluation import (
    create_splitter,
    validate_no_overlap,
    full_report,
    compute_metric,
    CompositeCalculator,
)
from generalized_ml_autoresearch.core.reasoning import (
    ReasoningAnnotationsFile,
    ReasoningEntry,
    validate_pre_run_entry,
)


# ---------------------------------------------------------------- utility

def _pin_to_safe_cores(cores: list[int] | None, n_threads: int | None) -> None:
    """Optional CPU-affinity pinning (from CLAUDE.md Hardware Constraints).

    Respects env overrides:
      AUTORESEARCH_USE_ALL_CORES=1 → skip affinity
      AUTORESEARCH_N_THREADS=N → override thread count
    """
    if os.environ.get("AUTORESEARCH_USE_ALL_CORES") == "1":
        return
    try:
        import torch
        env_threads = os.environ.get("AUTORESEARCH_N_THREADS")
        threads = int(env_threads) if env_threads else (n_threads or 4)
        torch.set_num_threads(threads)
    except ImportError:
        pass
    if cores:
        try:
            import psutil  # type: ignore[import-not-found]
            p = psutil.Process()
            p.cpu_affinity(cores)
        except Exception:
            pass


def _standard_scale(X_train: np.ndarray, X_val: np.ndarray | None, X_test: np.ndarray | None):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mu) / sigma
    X_val_s = (X_val - mu) / sigma if X_val is not None else None
    X_test_s = (X_test - mu) / sigma if X_test is not None else None
    return X_train_s, X_val_s, X_test_s, mu, sigma


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover
            raise ImportError("PyYAML required to load .yaml configs") from e
        return yaml.safe_load(text)
    return json.loads(text)


def _load_dataset(cfg: dict[str, Any]):
    """Load dataset per config. Supports:
      - "csv": pandas.read_csv
      - "parquet": pandas.read_parquet
      - "numpy": np.load
      - "sklearn": sklearn.datasets.fetch_* or load_*
      - "callable": import path "module:function" returning (X, y, feature_columns, target_columns, groups_or_None)
    """
    import pandas as pd

    data_cfg = cfg["data"]
    fmt = data_cfg["format"]
    feature_columns = data_cfg.get("feature_columns")
    target_columns = data_cfg.get("target_columns", ["target"])
    group_column = data_cfg.get("group_column")
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None = None

    if fmt in ("csv", "parquet"):
        df = pd.read_csv(data_cfg["path"]) if fmt == "csv" else pd.read_parquet(data_cfg["path"])
        if not feature_columns:
            feature_columns = [c for c in df.columns if c not in target_columns and c != group_column]
        X = df[feature_columns].to_numpy(dtype=float)
        y = df[target_columns].to_numpy(dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if group_column:
            groups = df[group_column].to_numpy()
    elif fmt == "numpy":
        payload = np.load(data_cfg["path"], allow_pickle=False)
        X = payload["X"]; y = payload["y"]
        feature_columns = feature_columns or [f"f{i}" for i in range(X.shape[1])]
    elif fmt == "sklearn":
        import sklearn.datasets as skd
        loader = getattr(skd, data_cfg["loader"])
        bunch = loader()
        X = np.asarray(bunch.data, dtype=float)
        y = np.asarray(bunch.target)
        feature_columns = feature_columns or list(getattr(bunch, "feature_names", [f"f{i}" for i in range(X.shape[1])]))
    elif fmt == "callable":
        module_path, fn_name = data_cfg["loader"].split(":")
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
        X, y, feature_columns, target_columns, groups = fn(**data_cfg.get("loader_kwargs", {}))
    else:
        raise ValueError(f"Unknown data.format={fmt!r}")
    return X, y, feature_columns, target_columns, groups


# ---------------------------------------------------------------- runner

def _evaluate_fold(model, X_test, y_test, task_type: str, n_samples: int):
    bundle = model.predict_with_uncertainty(X_test, n_samples=n_samples)
    mean_pred = np.asarray(bundle.mean).flatten() if np.asarray(bundle.mean).ndim <= 1 else np.asarray(bundle.mean)
    y_true = np.asarray(y_test).flatten() if np.asarray(y_test).ndim <= 1 else np.asarray(y_test)
    if task_type in ("binary_classification", "multiclass_classification"):
        y_pred = np.asarray(bundle.mean).astype(int).flatten()
        y_score = bundle.probabilities if bundle.probabilities is not None else None
        if y_score is not None and y_score.ndim > 1 and task_type == "binary_classification":
            y_score = y_score.ravel() if y_score.shape[-1] == 1 else y_score[:, 1]
        metrics = full_report(task_type, y_true.astype(int), y_pred, y_score=y_score)
    else:
        metrics = full_report(task_type, y_true, mean_pred)
    metrics["_bundle_mean"] = np.asarray(bundle.mean).tolist()
    metrics["_bundle_aleatoric"] = np.asarray(bundle.aleatoric).tolist()
    metrics["_bundle_epistemic"] = np.asarray(bundle.epistemic).tolist()
    metrics["_bundle_confidence"] = np.asarray(bundle.confidence).tolist()
    metrics["_y_true"] = y_true.tolist()
    return metrics


def run_experiment(config: dict[str, Any]) -> ExperimentRecord:
    """Run one experiment. Writes all runner-owned files. Returns the ExperimentRecord."""
    t0 = time.time()

    results_dir = Path(config["paths"]["results_dir"])
    trade_logs_dir = results_dir / "trade_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    trade_logs_dir.mkdir(parents=True, exist_ok=True)

    # Hardware pinning (optional)
    hw = config.get("hardware", {})
    _pin_to_safe_cores(hw.get("cpu_affinity"), hw.get("n_threads"))

    # Experiment numbering
    log_path = results_dir / "experiment_log.jsonl"
    exp_num = _next_experiment_num(log_path)

    # Reasoning annotation pre-check
    annotations = ReasoningAnnotationsFile(results_dir / "reasoning_annotations.json")
    existing_entries = annotations.load()
    key = str(exp_num)
    pre_run_ok = False
    if key in existing_entries:
        e = ReasoningEntry.from_dict(existing_entries[key])
        pre_run_violations = validate_pre_run_entry(e)
        if not pre_run_violations:
            pre_run_ok = True
        else:
            print(f"[runner] Pre-run reasoning validation failed for Exp{exp_num}:", file=sys.stderr)
            for v in pre_run_violations:
                print(f"  - {v}", file=sys.stderr)
            if not config.get("bypass_reasoning_gate", False):
                raise ValueError(
                    f"Experiment {exp_num} cannot launch — fix pre-run reasoning annotation first.\n"
                    f"See {annotations.path}"
                )
    else:
        if not config.get("bypass_reasoning_gate", False):
            raise ValueError(
                f"Experiment {exp_num} has no pre-run reasoning annotation in {annotations.path}.\n"
                f"Author `diagnosis`, `citations`, `hypothesis`, `prediction` with `_manual: true` before launching."
            )

    # Data
    X, y, feature_columns, target_columns, groups = _load_dataset(config)
    task_type = config["task_type"]

    # Split
    split_cfg = dict(config["split"])
    split_name = split_cfg.pop("name")
    splitter = create_splitter(split_name, **split_cfg)
    fold_assignments = splitter.split(len(X), y=y, groups=groups)
    validate_no_overlap(fold_assignments)

    # Per-fold training
    backbone_name = config["backbone"]
    if backbone_name not in list_backbones():
        raise ValueError(f"Unknown backbone {backbone_name}. Available: {list_backbones()}")
    bb_config = dict(config.get("backbone_config", {}))
    bb_config.setdefault("task_type", task_type)
    bb_config.setdefault("seed", config.get("seed", 0))

    primary_metric = config["primary_metric"]
    composite_cfg = config.get("composite", {})
    calc = CompositeCalculator(
        primary_metric_name=primary_metric,
        higher_is_better=composite_cfg.get("higher_is_better", True),
        penalty_weight=float(composite_cfg.get("penalty_weight", 0.1)),
        below_threshold=float(composite_cfg.get("below_threshold", 0.0)),
        formula=composite_cfg.get("formula"),
    )

    per_fold_val_primary: list[float] = []
    per_fold_test_primary: list[float] = []
    per_fold_test_reports: list[dict[str, Any]] = []
    all_rows_csv: list[dict[str, Any]] = []

    # Determine n_outputs
    if task_type == "multiclass_classification":
        n_outputs = int(np.max(y)) + 1
    elif task_type == "binary_classification":
        n_outputs = 1
    else:
        n_outputs = 1

    trained_model = None
    saved_model_path = None

    for fa in fold_assignments:
        X_train = X[fa.train_idx]
        y_train = y[fa.train_idx]
        X_val = X[fa.val_idx] if fa.val_idx is not None else None
        y_val = y[fa.val_idx] if fa.val_idx is not None else None
        X_test = X[fa.test_idx]
        y_test = y[fa.test_idx]

        X_train_s, X_val_s, X_test_s, mu, sigma = _standard_scale(X_train, X_val, X_test)

        model = create_model(backbone_name, bb_config)
        model.feature_columns = list(feature_columns or [])
        model.target_columns = list(target_columns or [])
        model.scaler_mean = mu
        model.scaler_scale = sigma
        model.build(bb_config, (X_train_s.shape[1],), n_outputs)
        model.fit(X_train_s, y_train, X_val_s, y_val)

        # Val metric
        if X_val_s is not None:
            val_report = _evaluate_fold(model, X_val_s, y_val, task_type, bb_config.get("uncertainty_samples", 30))
            per_fold_val_primary.append(float(val_report.get(primary_metric, float("nan"))))

        test_report = _evaluate_fold(model, X_test_s, y_test, task_type, bb_config.get("uncertainty_samples", 30))
        per_fold_test_primary.append(float(test_report.get(primary_metric, float("nan"))))
        # strip bundles from published report, keep aggregate metrics
        published = {k: v for k, v in test_report.items() if not k.startswith("_")}
        per_fold_test_reports.append({"fold_id": fa.fold_id, "regime": fa.regime_label, **published})

        # per-prediction rows
        bundle_mean = test_report["_bundle_mean"]
        bundle_ale = test_report["_bundle_aleatoric"]
        bundle_epi = test_report["_bundle_epistemic"]
        bundle_conf = test_report["_bundle_confidence"]
        y_true_list = test_report["_y_true"]
        for i, (pred, actual, ale, epi, conf) in enumerate(zip(bundle_mean, y_true_list, bundle_ale, bundle_epi, bundle_conf)):
            pred_scalar = pred[0] if isinstance(pred, list) and len(pred) == 1 else (pred if not isinstance(pred, list) else pred)
            row = {
                "index": int(fa.test_idx[i]),
                "fold": fa.fold_id,
                "regime": fa.regime_label,
                "prediction": float(np.asarray(pred_scalar).flatten()[0]) if task_type != "multiclass_classification" else int(np.asarray(pred_scalar).flatten()[0]),
                "actual": float(actual) if not isinstance(actual, (list, tuple)) else float(actual[0]),
                "confidence": float(np.asarray(conf).flatten()[0]),
                "aleatoric": float(np.asarray(ale).flatten()[0]),
                "epistemic": float(np.asarray(epi).flatten()[0]),
            }
            if task_type in ("regression", "time_series_forecasting"):
                row["abs_error"] = abs(row["prediction"] - row["actual"])
                if task_type == "time_series_forecasting":
                    row["pred_direction"] = int(np.sign(row["prediction"])) or 1
                    row["actual_direction"] = int(np.sign(row["actual"])) or 1
                    row["correct"] = int(row["pred_direction"] == row["actual_direction"])
                    row["strategy_return"] = row["pred_direction"] * row["actual"]
            else:
                row["correct"] = int(row["prediction"] == row["actual"])
            all_rows_csv.append(row)

        # Keep last-fold model for the saved checkpoint (or supply a seed-0 single-fold
        # variant in simple protocols like holdout where there's only one fold)
        trained_model = model

    # Aggregate
    val_primary_agg = float(np.mean(per_fold_val_primary)) if per_fold_val_primary else float("nan")
    test_primary_agg = float(np.mean(per_fold_test_primary))
    composite = calc.compute(val_primary_agg, test_primary_agg, per_fold_test_primary)

    # Did we beat the global best?
    best_config_path = results_dir / "best_config.json"
    status = "DISCARD"
    is_new_champion = False
    if best_config_path.exists():
        try:
            prev = json.loads(best_config_path.read_text(encoding="utf-8"))
            prev_composite = float(prev.get("composite", float("-inf")))
        except Exception:
            prev_composite = float("-inf")
    else:
        prev_composite = float("-inf")
    if composite > prev_composite:
        status = "KEEP"
        is_new_champion = True

    # Secondary metrics (aggregate across folds)
    agg_secondary: dict[str, Any] = {}
    for k in per_fold_test_reports[0].keys() if per_fold_test_reports else []:
        if k in ("fold_id", "regime", "task_type", "n") or k == primary_metric:
            continue
        vals = [r.get(k) for r in per_fold_test_reports if isinstance(r.get(k), (int, float))]
        if vals:
            agg_secondary[k] = float(np.mean(vals))

    # Write per-prediction CSV
    preds_csv_path = trade_logs_dir / f"exp{exp_num}_predictions.csv"
    if all_rows_csv:
        with open(preds_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows_csv[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows_csv)
    # Per-fold prediction summary
    summary: dict[str, Any] = {}
    for r in per_fold_test_reports:
        fold_rows = [row for row in all_rows_csv if row["fold"] == r["fold_id"]]
        n_total = len(fold_rows)
        n_correct = sum(int(row.get("correct", 0)) for row in fold_rows)
        summary[str(r["fold_id"])] = {
            "regime": r["regime"],
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": n_correct / n_total if n_total else 0.0,
            "primary_metric": r.get(primary_metric),
        }
    (trade_logs_dir / f"exp{exp_num}_prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Runner-fallback reasoning entry (if pre-run was missing and user bypassed)
    if not pre_run_ok:
        runner_verdict = (
            f"{'KEEP' if status == 'KEEP' else 'DISCARD'} (runner-fallback, rewrite manually). "
            f"Composite={composite:.4f}; per-fold primary={per_fold_test_primary}; "
            f"delta vs prev global best={(composite - prev_composite):.4f}."
        )
        runner_learning = (
            "TODO-REWRITE (runner-fallback entry). "
            "Axis state unknown — Claude MUST rewrite this learning with mechanism and next-try language. "
            "Next try: see checkpoint."
        )
        annotations.commit_post_run(exp_num, runner_verdict, runner_learning)
    else:
        # Only fill verdict/learning if Claude didn't pre-author them
        existing = annotations.load().get(str(exp_num), {})
        if not existing.get("verdict") or not existing.get("learning"):
            runner_verdict = (
                f"{'KEEP' if status == 'KEEP' else 'DISCARD'} (runner-fallback, rewrite manually). "
                f"Composite={composite:.4f}; per-fold={per_fold_test_primary}."
            )
            runner_learning = (
                "TODO-REWRITE (runner-fallback entry) — axis state pending human analysis. Next try: pending."
            )
            annotations.commit_post_run(exp_num, runner_verdict, runner_learning)

    # Save model + checkpoint if champion
    if is_new_champion and trained_model is not None:
        saved_model_path = results_dir / "best_model.pt"
        try:
            trained_model.save(saved_model_path)
        except Exception as e:  # pragma: no cover
            print(f"[runner] failed to save model: {e}", file=sys.stderr)
        best_config_path.write_text(json.dumps({
            "experiment_num": exp_num,
            "backbone": backbone_name,
            "composite": composite,
            "test_primary": test_primary_agg,
            "val_primary": val_primary_agg,
            "config": config,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, indent=2, default=str), encoding="utf-8")

    # Append to experiment_log.jsonl
    record = ExperimentRecord(
        experiment_num=exp_num,
        backbone=backbone_name,
        description=config.get("description", ""),
        config=config,
        composite=composite,
        val_primary=val_primary_agg,
        test_primary=test_primary_agg,
        per_fold_test=per_fold_test_primary,
        per_fold_val=per_fold_val_primary,
        status=status,
        seconds_elapsed=time.time() - t0,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        secondary_metrics=agg_secondary,
    )
    log_entry = asdict(record)
    log_entry["per_fold_test_reports"] = per_fold_test_reports
    log_entry["composite_fingerprint"] = calc.fingerprint()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")

    print(
        f"[runner] Exp{exp_num} ({backbone_name}) done in {record.seconds_elapsed:.1f}s — "
        f"composite={composite:.4f} status={status}"
    )
    return record


def _next_experiment_num(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    max_num = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                n = int(entry.get("experiment_num", 0))
                max_num = max(max_num, n)
            except Exception:
                continue
    return max_num + 1


# ---------------------------------------------------------------- CLI

def _cli():
    ap = argparse.ArgumentParser(description="Generalized ML AutoResearch runner")
    ap.add_argument("--config", required=True, help="Path to YAML or JSON config file")
    ap.add_argument("--description", default="", help="Experiment description (overrides config)")
    ap.add_argument("--seed", type=int, help="Override random seed")
    ap.add_argument("--backbone", help="Override backbone name")
    ap.add_argument("--bypass-reasoning-gate", action="store_true",
                     help="Allow experiment to launch without a valid pre-run reasoning annotation "
                          "(runner inserts TODO-REWRITE sentinels — not for regular use).")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    if args.description:
        cfg["description"] = args.description
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg.setdefault("backbone_config", {})["seed"] = args.seed
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.bypass_reasoning_gate:
        cfg["bypass_reasoning_gate"] = True

    run_experiment(cfg)


if __name__ == "__main__":
    _cli()
