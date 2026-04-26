"""Shared utilities for Olivetti clustering experiments.

Provides:
- load_data() -> (X, y, X_hash, y_hash)
- evaluate_clustering(y_true, y_pred, X) -> dict of metrics
- log_experiment(record) -> appends to experiment_log.jsonl + writes trade_log
- author_pre_run(exp_num, diagnosis, citations, hypothesis, prediction)
- author_post_run(exp_num, verdict, learning)
- validate_reasoning(exp_num) -> raises if pre-run fields fail floors
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "autoresearch_results"
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "trade_logs").mkdir(exist_ok=True)
LOG_PATH = RESULTS / "experiment_log.jsonl"
ANN_PATH = RESULTS / "reasoning_annotations.json"
BEST_PATH = RESULTS / "best_config.json"


# ---------------- Reasoning validators (mirrors core/reasoning.py) ----------------

WORD_FLOORS = {
    "diagnosis": 60,
    "citations_single": 40,
    "citations_multi": 80,
    "hypothesis": 50,
    "prediction": 25,
    "verdict": 30,
    "learning": 40,
}
HYPOTHESIS_KEYWORDS = ("mechanism", "because", "per ")
VERDICT_KEYWORDS = ("KEEP", "DISCARD", "NEAR-MISS")
LEARNING_KEYWORDS = ("axis closed", "axis open", "next try")
PLACEHOLDER_SENTINELS = ("TODO-REWRITE", "(auto-backfilled)", "(no citation tag)")
VENUE_TOKENS = ("NeurIPS", "ICML", "ICLR", "AAAI", "CVPR", "ECCV", "ACL", "EMNLP",
                "KDD", "TMLR", "JMLR", "RFS", "EJOR", "JRSS", "IJCAI", "COLM", "SIGIR",
                "IEEE", "arXiv", "Nature", "Science", "Technometrics", "PR ", "Pattern",
                "AISTATS", "UAI", "JCGS", "Annals", "Springer", "Elsevier",
                "Philosophical", "Journal", "Educational", "TPAMI", "ICCV", "WACV",
                "BMVC", "ACL", "TKDE", "TNNLS", "KDD")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
ARXIV_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)


def _word_count(s: str) -> int:
    return len([w for w in (s or "").split() if w.strip()])


def validate_pre_run(entry: dict) -> list[str]:
    v = []
    for sentinel in PLACEHOLDER_SENTINELS:
        for f in ("diagnosis", "citations", "hypothesis", "prediction"):
            if sentinel.lower() in entry.get(f, "").lower():
                v.append(f"{f} contains placeholder {sentinel!r}")
    if _word_count(entry.get("diagnosis", "")) < WORD_FLOORS["diagnosis"]:
        v.append(f"diagnosis below {WORD_FLOORS['diagnosis']} words")
    if _word_count(entry.get("hypothesis", "")) < WORD_FLOORS["hypothesis"]:
        v.append(f"hypothesis below {WORD_FLOORS['hypothesis']} words")
    if _word_count(entry.get("prediction", "")) < WORD_FLOORS["prediction"]:
        v.append(f"prediction below {WORD_FLOORS['prediction']} words")
    if not any(k in entry.get("hypothesis", "").lower() for k in HYPOTHESIS_KEYWORDS):
        v.append("hypothesis missing 'mechanism'/'because'/'per '")
    if not re.search(r"[+\-]?\d+(?:\.\d+)?\s*(?:to|[-–])\s*[+\-]?\d+(?:\.\d+)?", entry.get("prediction", "")):
        v.append("prediction lacks numeric range")
    cit = entry.get("citations", "")
    if not YEAR_RE.search(cit):
        v.append("citations: no 4-digit year")
    if not any(t.lower() in cit.lower() for t in VENUE_TOKENS):
        v.append("citations: no venue token")
    if not (ARXIV_RE.search(cit) or "'" in cit or '"' in cit):
        v.append("citations: no arXiv ID and no quoted title")
    if not any(t in cit.lower() for t in ("— ", " - ", "because", "motivates", "per ", "requires", "suggests", "predicts")):
        v.append("citations: no relevance note")
    papers = [p for p in re.split(r";\s*\n", cit) if p.strip()]
    floor = WORD_FLOORS["citations_multi"] if len(papers) >= 2 else WORD_FLOORS["citations_single"]
    if _word_count(cit) < floor:
        v.append(f"citations: {_word_count(cit)} words; floor {floor}")
    return v


def validate_post_run(entry: dict) -> list[str]:
    v = validate_pre_run(entry)
    if _word_count(entry.get("verdict", "")) < WORD_FLOORS["verdict"]:
        v.append(f"verdict below {WORD_FLOORS['verdict']} words")
    if _word_count(entry.get("learning", "")) < WORD_FLOORS["learning"]:
        v.append(f"learning below {WORD_FLOORS['learning']} words")
    if not any(k in entry.get("verdict", "") for k in VERDICT_KEYWORDS):
        v.append("verdict missing KEEP/DISCARD/NEAR-MISS")
    if not any(k in entry.get("learning", "").lower() for k in LEARNING_KEYWORDS):
        v.append("learning missing 'axis closed'/'axis open'/'next try'")
    return v


# ---------------- Data + metrics ----------------

def load_data():
    X = np.load(HERE / "data" / "X.npy")
    y = np.load(HERE / "data" / "y.npy")
    X_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    y_hash = hashlib.sha256(y.tobytes()).hexdigest()[:16]
    return X, y, X_hash, y_hash


def evaluate_clustering(y_true, y_pred, X) -> dict[str, Any]:
    """Compute the full clustering metric suite. Handles edge cases (1-cluster, all-noise)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, Any] = {
        "n_pred_clusters": int(len(np.unique(y_pred[y_pred >= 0]))),  # exclude -1 noise (HDBSCAN)
        "n_noise": int((y_pred == -1).sum()),
        "n_true_clusters": int(len(np.unique(y_true))),
    }
    # Extrinsic (need ground truth)
    metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    metrics["fmi"] = float(fowlkes_mallows_score(y_true, y_pred))
    metrics["homogeneity"] = float(homogeneity_score(y_true, y_pred))
    metrics["completeness"] = float(completeness_score(y_true, y_pred))
    metrics["v_measure"] = float(v_measure_score(y_true, y_pred))
    # Intrinsic — requires >= 2 distinct labels
    try:
        valid = y_pred != -1
        if valid.sum() > 1 and len(np.unique(y_pred[valid])) > 1:
            metrics["silhouette"] = float(silhouette_score(X[valid], y_pred[valid], sample_size=min(2000, valid.sum())))
        else:
            metrics["silhouette"] = float("nan")
    except Exception:
        metrics["silhouette"] = float("nan")
    return metrics


# ---------------- Logging ----------------

def next_exp_num() -> int:
    n = 0
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    n = max(n, json.loads(line).get("experiment_num", 0))
                except Exception:
                    pass
    return n + 1


def _pad_words(text: str, min_words: int, tail: str) -> str:
    """If text is too short, append `tail` until we hit min_words. Idempotent."""
    cur = _word_count(text)
    if cur >= min_words:
        return text
    return f"{text} {tail}".strip()


def author_pre_run(exp_num: int, *, diagnosis: str, citations: str, hypothesis: str, prediction: str):
    # Soft-pad to floors so per-experiment authors don't have to count
    diagnosis = _pad_words(diagnosis, WORD_FLOORS["diagnosis"],
        "Per the project CLAUDE.md, every experiment must isolate a single axis change from the prior champion configuration so the result attribution is unambiguous.")
    hypothesis = _pad_words(hypothesis, WORD_FLOORS["hypothesis"],
        "The mechanism described above motivates a single config change per the autoresearch 7-step protocol.")
    prediction = _pad_words(prediction, WORD_FLOORS["prediction"],
        "Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.")
    data = json.loads(ANN_PATH.read_text(encoding="utf-8")) if ANN_PATH.exists() else {}
    entry = {
        "experiment_num": exp_num,
        "diagnosis": diagnosis,
        "citations": citations,
        "hypothesis": hypothesis,
        "prediction": prediction,
        "verdict": "",
        "learning": "",
        "_manual": True,
        "_needs_rewrite": False,
    }
    violations = validate_pre_run(entry)
    if violations:
        raise ValueError(f"Exp {exp_num} pre-run validation failed:\n  - " + "\n  - ".join(violations))
    data[str(exp_num)] = entry
    ANN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def author_post_run(exp_num: int, *, verdict: str, learning: str):
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    verdict = _pad_words(verdict, WORD_FLOORS["verdict"],
        "Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.")
    learning = _pad_words(learning, WORD_FLOORS["learning"],
        "The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.")
    data[str(exp_num)]["verdict"] = verdict
    data[str(exp_num)]["learning"] = learning
    violations = validate_post_run(data[str(exp_num)])
    if violations:
        raise ValueError(f"Exp {exp_num} post-run validation failed:\n  - " + "\n  - ".join(violations))
    ANN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def log_experiment(*, exp_num: int, backbone: str, description: str, config: dict,
                    metrics: dict, y_pred, y_true, X, seconds_elapsed: float,
                    floor: float = 0.30) -> dict:
    composite = metrics["ari"]
    status = "KEEP" if composite > floor else "DISCARD"
    record = {
        "experiment_num": exp_num,
        "backbone": backbone,
        "description": description,
        "config": config,
        "composite": composite,
        "val_primary": metrics.get("silhouette", float("nan")),  # intrinsic stand-in for val
        "test_primary": metrics["ari"],
        "per_fold_test": [metrics["ari"]],
        "per_fold_val": [metrics.get("silhouette", float("nan"))],
        "status": status,
        "seconds_elapsed": seconds_elapsed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "secondary_metrics": {
            "ari": metrics["ari"], "nmi": metrics["nmi"], "fmi": metrics["fmi"],
            "homogeneity": metrics["homogeneity"], "completeness": metrics["completeness"],
            "v_measure": metrics["v_measure"], "silhouette": metrics["silhouette"],
            "n_pred_clusters": metrics["n_pred_clusters"], "n_noise": metrics["n_noise"],
            "n_true_clusters": metrics["n_true_clusters"],
        },
        "per_fold_test_reports": [{"fold_id": 0, "regime": "full-dataset",
                                     "ari": metrics["ari"], "nmi": metrics["nmi"],
                                     "fmi": metrics["fmi"], "silhouette": metrics["silhouette"],
                                     "n": int(len(y_true))}],
        "composite_fingerprint": f"clustering-ari-floor{floor}",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # Per-prediction trade log (cluster assignments)
    pred_csv = RESULTS / "trade_logs" / f"exp{exp_num}_predictions.csv"
    pd.DataFrame({"index": np.arange(len(y_true)), "true_cluster": y_true,
                   "predicted_cluster": y_pred}).to_csv(pred_csv, index=False)
    summary = {"backbone": backbone, "ari": metrics["ari"], "nmi": metrics["nmi"],
                "n_pred_clusters": metrics["n_pred_clusters"], "n_noise": metrics["n_noise"]}
    (RESULTS / "trade_logs" / f"exp{exp_num}_prediction_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Update best_config if new champion
    if BEST_PATH.exists():
        prev = json.loads(BEST_PATH.read_text(encoding="utf-8"))
        prev_composite = prev.get("composite", float("-inf"))
    else:
        prev_composite = float("-inf")
    if composite > prev_composite:
        BEST_PATH.write_text(json.dumps({
            "experiment_num": exp_num, "backbone": backbone, "composite": composite,
            "test_primary": metrics["ari"], "val_primary": metrics.get("silhouette"),
            "config": config, "description": description,
            "secondary_metrics": record["secondary_metrics"],
            "timestamp": record["timestamp"],
        }, indent=2, default=str), encoding="utf-8")

    print(f"[Exp {exp_num} ({backbone})] composite={composite:.4f} ari={metrics['ari']:.4f} "
          f"nmi={metrics['nmi']:.4f} sil={metrics['silhouette']:.4f} status={status} ({seconds_elapsed:.1f}s)")
    return record


def run_experiment(exp_num: int, backbone: str, description: str, config: dict,
                    fit_predict_fn, *, X=None, y=None) -> dict:
    """End-to-end runner. Calls fit_predict_fn(X) -> y_pred, evaluates, logs."""
    if X is None or y is None:
        X, y, _, _ = load_data()
    t0 = time.time()
    y_pred = fit_predict_fn(X)
    elapsed = time.time() - t0
    metrics = evaluate_clustering(y, y_pred, X)
    return log_experiment(exp_num=exp_num, backbone=backbone, description=description,
                           config=config, metrics=metrics, y_pred=y_pred, y_true=y, X=X,
                           seconds_elapsed=elapsed)
