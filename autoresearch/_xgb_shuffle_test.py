"""Sanity-check XGBoost pipeline: train on permuted targets, evaluate on real test.
If test Sharpe > 1, there's a leak (target identity transmitted through features
or evaluator). If Sharpe ~ 0, the +7.85 result is real."""
import sys
sys.path.insert(0, "C:/Users/evija/autoresearch")
import numpy as np
from autoresearch.run_autoresearch import _run_experiment_inner
from autoresearch.data.download import download_all_pairs, download_macro_signals
from autoresearch.run_autoresearch import compute_all_features, compute_targets
from autoresearch.data.splits import split_superfold
from autoresearch.model.backbone import GBMWrapper
from autoresearch.model.train import find_contiguous_segments
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

# --- Build the same features/targets the runner uses ---
np.random.seed(42)
torch.manual_seed(42)

pairs = download_all_pairs()
macro = download_macro_signals()
feats = compute_all_features(pairs, macro)
targets = compute_targets(pairs["EURUSD=X"])
common = feats.index.intersection(targets.index)
feats = feats.loc[common]
targets = targets.loc[common]

train_feat, val_feat, test_feat = split_superfold(feats)
train_tgt, val_tgt, test_tgt = split_superfold(targets)
print(f"train n={len(train_feat)}, val n={len(val_feat)}, test n={len(test_feat)}")

scaler = StandardScaler()
scaler.fit(train_feat.values)

# --- Train XGBoost on PERMUTED targets ---
np.random.seed(0)
train_tgt_shuffled = train_tgt.copy()
permuted = np.random.permutation(train_tgt_shuffled.values)
train_tgt_shuffled.loc[:, :] = permuted

train_s = scaler.transform(train_feat.values)
seq_len = 10
segments = find_contiguous_segments(train_feat.index)
X_parts, y_parts = [], []
for seg_start, seg_end in segments:
    seg = train_s[seg_start:seg_end]
    seg_tgt = train_tgt_shuffled.iloc[seg_start:seg_end]
    if len(seg) <= seq_len:
        continue
    X = np.array([seg[i:i+seq_len].ravel() for i in range(len(seg) - seq_len + 1)])
    y = seg_tgt.values[seq_len-1:][:len(X)]
    X_parts.append(X[:len(y)])
    y_parts.append(y)

X = np.concatenate(X_parts)
y = np.concatenate(y_parts)

print(f"Train X shape: {X.shape}, y shape: {y.shape}")

model = GBMWrapper("xgboost", n_targets=2, hp_overrides={
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
})
model.fit(X, y)

# --- Evaluate on REAL test ---
# For each fold window, build test dataset and predict
from autoresearch.data.splits import FOLDS, get_fold_dates
from autoresearch.evaluation.metrics import trading_report, sharpe_ratio, information_coefficient

test_returns_all = []
print(f"\nEvaluating on REAL test (actual targets, model trained on SHUFFLED):")
for fold in FOLDS:
    d = get_fold_dates(fold)
    w_start, w_end = d["test_start"], d["test_end"]
    wf = test_feat.loc[w_start:w_end]
    wt = test_tgt.loc[w_start:w_end]
    if len(wf) < seq_len + 1:
        continue
    ws = scaler.transform(wf.values)
    # Build windowed X (match evaluator convention: window [idx..idx+L-1], target at idx+L-1)
    X_test = np.array([ws[i:i+seq_len].ravel() for i in range(len(ws) - seq_len + 1)])
    y_test = wt.values[seq_len-1:][:len(X_test)]
    preds = model.predict(X_test)[:, 0]  # ret_1d
    returns = np.sign(preds) * y_test[:, 0]
    rpt = trading_report(returns)
    ic = information_coefficient(preds, y_test[:, 0])
    print(f"  {fold['name']} {fold['regime'][:30]:30} "
          f"Sharpe={sharpe_ratio(returns):+.3f} Ret={rpt['total_return_pct']:+.2f}% "
          f"IC={ic['ic_spearman']:+.3f} Hit={ic['hit_rate']:.1f}%")
    test_returns_all.append(returns)

all_rets = np.concatenate(test_returns_all)
print(f"\nAGGREGATE TEST Sharpe on real y, model trained on SHUFFLED y: "
      f"{sharpe_ratio(all_rets):+.4f}")
print("If this is near 0, the +7.85 result is LEGIT (features genuinely predict targets).")
print("If this is > +1, there's a data leak somewhere.")
