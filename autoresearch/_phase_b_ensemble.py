"""Phase (b) — cross-backbone ensemble experiments.
Loads champion checkpoints from multiple backbones, averages their predictions
on the same test set, evaluates as a 'meta-model' with no additional training.

Three ensemble strategies tried:
  1. Simple-avg: mean of normalized predictions
  2. Rank-avg: mean of prediction ranks (most robust to scale mismatches)
  3. Weighted: weighted by per-backbone composite score

Run AFTER xgboost+lightgbm+catboost champions are archived.
"""
import pickle
import sys
from pathlib import Path
sys.path.insert(0, "C:/Users/evija/autoresearch")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from autoresearch.run_autoresearch import compute_all_features, compute_targets
from autoresearch.data.download import download_all_pairs, download_macro_signals
from autoresearch.data.splits import split_superfold, FOLDS, get_fold_dates
from autoresearch.model.train import find_contiguous_segments
from autoresearch.evaluation.metrics import (
    trading_report, sharpe_ratio, information_coefficient, classification_metrics,
)

RESULTS = Path("C:/Users/evija/autoresearch/autoresearch/autoresearch_results")
WINNERS = RESULTS / "winners"

# --- Load data ---
pairs = download_all_pairs()
macro = download_macro_signals()
feats = compute_all_features(pairs, macro)
targets = compute_targets(pairs["EURUSD=X"])
common = feats.index.intersection(targets.index)
feats = feats.loc[common]
targets = targets.loc[common]
train_feat, val_feat, test_feat = split_superfold(feats)
train_tgt, val_tgt, test_tgt = split_superfold(targets)

# --- Collect GBM champion pickles ---
def find_gbm_champions() -> list[Path]:
    """Discover pickle checkpoints in winners/ for GBM backbones."""
    return sorted(WINNERS.rglob("*_model.pkl"))

champ_paths = find_gbm_champions()
print(f"Found {len(champ_paths)} GBM champion pickle(s):")
for p in champ_paths:
    print(f"  {p.relative_to(WINNERS)}")

# --- For each champion, get test predictions per fold ---
def predict_fold_windows(bundle, feat_df, tgt_df, fold_list):
    """Run a bundle (scaler + GBMWrapper) over fold windows, return per-fold
    (preds, targets) pairs and aggregated return/Sharpe."""
    scaler_mean = bundle["scaler_mean"]
    scaler_scale = bundle["scaler_scale"]
    model = bundle["gbm_wrapper"]
    seq_len = bundle["seq_len"]
    per_fold = []
    for fold in fold_list:
        d = get_fold_dates(fold)
        w_start, w_end = d["test_start"], d["test_end"]
        wf = feat_df.loc[w_start:w_end]
        wt = tgt_df.loc[w_start:w_end]
        if len(wf) < seq_len + 1:
            continue
        ws = (wf.values - scaler_mean) / scaler_scale
        X = np.array([ws[i:i+seq_len].ravel() for i in range(len(ws) - seq_len + 1)])
        y = wt.values[seq_len-1:][:len(X), 0]  # ret_1d only
        preds = model.predict(X)[:, 0]  # ret_1d
        per_fold.append({
            "fold": fold["name"], "regime": fold["regime"],
            "preds": preds, "actuals": y,
        })
    return per_fold

bundles = []
for p in champ_paths:
    try:
        with open(p, "rb") as f:
            bundle = pickle.load(f)
        bundle["_path"] = p
        bundles.append(bundle)
        print(f"  Loaded {p.name}: backbone={bundle.get('backbone')} "
              f"composite={bundle.get('composite')}")
    except Exception as e:
        print(f"  Skip {p.name}: {e}")

if len(bundles) < 2:
    print("Need at least 2 GBM champions to ensemble. Exiting.")
    sys.exit(0)

# --- Group bundles by seq_len (ensemble requires matching prediction lengths) ---
from collections import defaultdict
groups = defaultdict(list)
for b in bundles:
    groups[b["seq_len"]].append(b)
print(f"\nGrouped {len(bundles)} pickles by seq_len: "
      f"{', '.join(f'seq={sl}:{len(bs)}' for sl, bs in groups.items())}")

# Run ensembling for EVERY seq_len group with 2+ members, so we don't
# miss the higher-composite minority group.
print(f"\nGroups to ensemble: "
      f"{', '.join(f'seq={sl}({len(bs)})' for sl, bs in sorted(groups.items()) if len(bs) >= 2)}")


def ensemble_group_report(ensemble_group, label):
    print(f"\n--- seq_len={label} group ({len(ensemble_group)} bundles) ---")
    per_bb = [predict_fold_windows(b, test_feat, test_tgt, FOLDS) for b in ensemble_group]
    # Individual sharpes
    for b, pf in zip(ensemble_group, per_bb):
        ret = np.concatenate([np.sign(f["preds"]) * f["actuals"] for f in pf])
        rpt = trading_report(ret)
        ic = information_coefficient(
            np.concatenate([f["preds"] for f in pf]),
            np.concatenate([f["actuals"] for f in pf]))
        print(f"  [{b['backbone']:<8}] {b['_path'].parent.name[:38]:<38}  "
              f"Sharpe={sharpe_ratio(ret):+.4f}  "
              f"Ret={rpt['total_return_pct']:+.2f}%  IC={ic['ic_spearman']:+.3f}")
    # Ensemble three ways
    ensembled = [{"actuals": f["actuals"]} for f in per_bb[0]]
    for fi in range(len(ensembled)):
        raw = np.column_stack([pf[fi]["preds"] for pf in per_bb])
        ensembled[fi]["simple"] = raw.mean(axis=1)
        z = np.column_stack([(pf[fi]["preds"] - pf[fi]["preds"].mean()) / (pf[fi]["preds"].std() + 1e-12) for pf in per_bb])
        ensembled[fi]["zscore"] = z.mean(axis=1)
        r = np.column_stack([rankdata(pf[fi]["preds"]) for pf in per_bb])
        ensembled[fi]["rank"] = r.mean(axis=1) - (len(raw) + 1) / 2
    for key in ("simple", "zscore", "rank"):
        rets = np.concatenate([np.sign(f[key]) * f["actuals"] for f in ensembled])
        rpt = trading_report(rets)
        ic = information_coefficient(
            np.concatenate([f[key] for f in ensembled]),
            np.concatenate([f["actuals"] for f in ensembled]))
        sh = sharpe_ratio(rets)
        print(f"  ENSEMBLE {key:<7} Sharpe={sh:+.4f}  Ret={rpt['total_return_pct']:+.2f}%  "
              f"IC={ic['ic_spearman']:+.3f}  Hit={ic['hit_rate']:.1f}%")


for sl in sorted(groups.keys()):
    if len(groups[sl]) >= 2:
        ensemble_group_report(groups[sl], sl)
sys.exit(0)  # skip the legacy code below

# --- Per-backbone fold predictions ---
per_backbone = [predict_fold_windows(b, test_feat, test_tgt, FOLDS) for b in list(groups.values())[0]]
bundles = list(groups.values())[0]

# --- Ensemble strategies ---
def composite_metric(per_fold):
    agg = np.concatenate([f["preds"] * np.sign(f["preds"]) for f in per_fold])  # dummy
    returns = np.concatenate([np.sign(f["preds"]) * f["actuals"] for f in per_fold])
    test_sharpe = sharpe_ratio(returns)
    n_neg = sum(1 for f in per_fold if sharpe_ratio(np.sign(f["preds"]) * f["actuals"]) < 0)
    return test_sharpe - 0.1 * n_neg

print("\n=== Individual champion Test Sharpes ===")
for b, pf in zip(bundles, per_backbone):
    ret = np.concatenate([np.sign(f["preds"]) * f["actuals"] for f in pf])
    print(f"  {b['_path'].parent.name:<40} n_folds={len(pf)}  "
          f"Sharpe={sharpe_ratio(ret):+.4f}  "
          f"total_ret={((1 + ret).prod() - 1) * 100:+.2f}%")

print("\n=== Ensemble strategies ===")
# Normalize predictions per-backbone (z-score) then average
def zscore(x):
    m, s = x.mean(), x.std() + 1e-12
    return (x - m) / s

# Stack predictions per-fold and average across backbones
ensembled = [{"fold": f["fold"], "regime": f["regime"], "actuals": f["actuals"]}
             for f in per_backbone[0]]
for fi in range(len(ensembled)):
    # Simple avg of raw predictions
    raw = np.column_stack([pf[fi]["preds"] for pf in per_backbone])
    ensembled[fi]["simple_avg"] = raw.mean(axis=1)
    # Z-score then avg
    z = np.column_stack([zscore(pf[fi]["preds"]) for pf in per_backbone])
    ensembled[fi]["zscore_avg"] = z.mean(axis=1)
    # Rank then avg
    r = np.column_stack([rankdata(pf[fi]["preds"]) for pf in per_backbone])
    ensembled[fi]["rank_avg"] = r.mean(axis=1) - (len(raw) + 1) / 2  # centered

for strategy in ("simple_avg", "zscore_avg", "rank_avg"):
    rets = np.concatenate([
        np.sign(f[strategy]) * f["actuals"] for f in ensembled
    ])
    rpt = trading_report(rets)
    ic = information_coefficient(
        np.concatenate([f[strategy] for f in ensembled]),
        np.concatenate([f["actuals"] for f in ensembled])
    )
    print(f"  {strategy:<14} Sharpe={sharpe_ratio(rets):+.4f}  "
          f"Ret={rpt['total_return_pct']:+.2f}%  IC={ic['ic_spearman']:+.3f}  "
          f"Hit={ic['hit_rate']:.1f}%")
