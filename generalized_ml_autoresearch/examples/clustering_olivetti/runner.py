"""Unified clustering experiment runner.

Mirrors the supervised runner's contract:
- reads pre-run reasoning entry (validates Citation Rigor + Reasoning Blob Completeness)
- runs ONE clustering experiment
- writes experiment_log.jsonl, best_config.json, trade_logs/exp<N>_predictions.csv
- writes post-run reasoning fallback (Claude rewrites later)

Different from supervised:
- No train/val/test split — entire dataset is clustered, evaluated against ground truth
- Primary metric: ARI (adjusted_rand_score). Secondary: NMI, FMI, Silhouette, Homogeneity, Completeness, V-measure
- Per-fold = single fold (no split)
- "test set hash" = SHA-256 of sorted indices (always 0..399 for Olivetti)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,
    silhouette_score, homogeneity_score, completeness_score, v_measure_score,
    adjusted_mutual_info_score,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "autoresearch_results"
RESULTS.mkdir(exist_ok=True)
(RESULTS / "trade_logs").mkdir(exist_ok=True)


# Repo-import for the reasoning validator
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.load(DATA / "X.npy")
    y = np.load(DATA / "y.npy")
    assert X.shape == (400, 4096) and y.shape == (400,), f"unexpected shapes {X.shape}, {y.shape}"
    return X, y


def data_hash() -> str:
    X, y = load_data()
    return hashlib.sha256(np.concatenate([X.tobytes(), y.tobytes()]).encode() if isinstance(X.tobytes(), str)
                          else X.tobytes() + y.tobytes()).hexdigest()[:16]


def next_exp_num() -> int:
    log_path = RESULTS / "experiment_log.jsonl"
    if not log_path.exists():
        return 1
    n = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n = max(n, int(json.loads(line).get("experiment_num", 0)))
    return n + 1


def run_clustering(
    backbone: str,
    description: str,
    cluster_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    config: dict[str, Any],
    pre_run_reasoning: dict[str, Any],
    bypass_gate: bool = False,
) -> dict[str, Any]:
    """Run one clustering experiment.

    cluster_fn(X, y_true) -> y_pred  (the algorithm receives X and may use n_clusters from y_true.max()+1)

    pre_run_reasoning must have all 5 pre-run fields (diagnosis, citations, hypothesis, prediction, _manual).
    """
    t0 = time.time()
    X, y = load_data()
    exp_num = next_exp_num()

    # Persist + validate the reasoning entry BEFORE running
    ann_path = RESULTS / "reasoning_annotations.json"
    data = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    entry = dict(pre_run_reasoning)
    entry["experiment_num"] = exp_num
    entry.setdefault("verdict", "")
    entry.setdefault("learning", "")
    entry.setdefault("_manual", True)
    entry.setdefault("_needs_rewrite", False)
    data[str(exp_num)] = entry
    ann_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    if not bypass_gate:
        from generalized_ml_autoresearch.core.reasoning import ReasoningEntry, validate_pre_run_entry
        v = validate_pre_run_entry(ReasoningEntry.from_dict(entry))
        if v:
            raise ValueError(f"Exp {exp_num} pre-run validation failed: {v}")

    # Run the clustering algorithm
    y_pred = cluster_fn(X, y)
    elapsed = time.time() - t0

    # Compute metrics
    n_clusters_pred = len(set(y_pred[y_pred >= 0]))
    n_noise = int((y_pred == -1).sum()) if -1 in y_pred else 0
    metrics = {
        "ari": float(adjusted_rand_score(y, y_pred)),
        "nmi": float(normalized_mutual_info_score(y, y_pred)),
        "ami": float(adjusted_mutual_info_score(y, y_pred)),
        "fmi": float(fowlkes_mallows_score(y, y_pred)),
        "homogeneity": float(homogeneity_score(y, y_pred)),
        "completeness": float(completeness_score(y, y_pred)),
        "v_measure": float(v_measure_score(y, y_pred)),
        "n_clusters_pred": n_clusters_pred,
        "n_noise_points": n_noise,
    }
    # Silhouette only if at least 2 clusters and no all-noise
    valid_mask = y_pred >= 0
    if valid_mask.sum() > 10 and len(set(y_pred[valid_mask])) >= 2:
        try:
            metrics["silhouette"] = float(silhouette_score(X[valid_mask], y_pred[valid_mask]))
        except Exception:
            metrics["silhouette"] = None
    else:
        metrics["silhouette"] = None

    print(f"Exp {exp_num} ({backbone}): ARI={metrics['ari']:.4f}  NMI={metrics['nmi']:.4f}  "
          f"FMI={metrics['fmi']:.4f}  V={metrics['v_measure']:.4f}  "
          f"k_pred={n_clusters_pred}  noise={n_noise}  ({elapsed:.1f}s)")

    # Composite metric: ARI is primary; floor at 0.50 (above-random for Olivetti)
    floor = config.get("composite", {}).get("below_threshold", 0.50)
    composite = metrics["ari"] - 0.05 * (1 if metrics["ari"] < floor else 0)
    status = "KEEP" if composite > floor else "DISCARD"

    # Per-prediction CSV
    preds_csv = RESULTS / "trade_logs" / f"exp{exp_num}_predictions.csv"
    import pandas as pd
    pred_df = pd.DataFrame({
        "index": np.arange(len(y)),
        "true_subject_id": y,
        "predicted_cluster": y_pred,
        "is_noise": (y_pred == -1).astype(int),
    })
    pred_df.to_csv(preds_csv, index=False)

    # Append record to experiment_log.jsonl
    record = {
        "experiment_num": exp_num,
        "backbone": backbone,
        "description": description,
        "config": config,
        "composite": composite,
        "val_primary": metrics["ari"],  # no val for clustering — use same as test for compat
        "test_primary": metrics["ari"],
        "per_fold_test": [metrics["ari"]],
        "per_fold_val": [],
        "status": status,
        "seconds_elapsed": elapsed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "secondary_metrics": metrics,
        "per_fold_test_reports": [{"fold_id": 0, "regime": "all-data-clustering", **metrics, "n": int(len(y))}],
        "composite_fingerprint": "clustering-ari-floor-0.50",
        "data_hash_first16": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:16],
    }
    log_path = RESULTS / "experiment_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # Update best_config.json if new champion
    best_path = RESULTS / "best_config.json"
    prev_best = -1.0
    if best_path.exists():
        try:
            prev_best = float(json.loads(best_path.read_text(encoding="utf-8")).get("composite", -1.0))
        except Exception:
            prev_best = -1.0
    if composite > prev_best:
        best_path.write_text(json.dumps({
            "champion_experiment_num": exp_num,
            "champion_backbone": backbone,
            "composite": composite,
            "test_primary": metrics["ari"],
            "val_primary": metrics["ari"],
            "config": config,
            "metrics": metrics,
            "description": description,
            "timestamp": record["timestamp"],
        }, indent=2, default=str), encoding="utf-8")

    return record


if __name__ == "__main__":
    print("This module provides run_clustering(); use run_*.py scripts to launch experiments.")
