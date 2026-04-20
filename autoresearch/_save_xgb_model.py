"""Retrain + serialise the XGBoost champion to the winner archive (pickle).
The runner's best_model.pt save path is torch-only; GBMs need a separate
serialisation. Runs the champion config and dumps the fitted GBMWrapper."""
import pickle
import sys
sys.path.insert(0, "C:/Users/evija/autoresearch")
import numpy as np
from sklearn.preprocessing import StandardScaler
from autoresearch.run_autoresearch import compute_all_features, compute_targets
from autoresearch.data.download import download_all_pairs, download_macro_signals
from autoresearch.data.splits import split_superfold
from autoresearch.model.backbone import GBMWrapper
from autoresearch.model.train import find_contiguous_segments

np.random.seed(42)
pairs = download_all_pairs()
macro = download_macro_signals()
feats = compute_all_features(pairs, macro)
targets = compute_targets(pairs["EURUSD=X"])
common = feats.index.intersection(targets.index)
feats = feats.loc[common]; targets = targets.loc[common]
train_feat, _, _ = split_superfold(feats)
train_tgt, _, _ = split_superfold(targets)

scaler = StandardScaler()
scaler.fit(train_feat.values)
train_s = scaler.transform(train_feat.values)
seq_len = 10
X_parts, y_parts = [], []
for seg_start, seg_end in find_contiguous_segments(train_feat.index):
    seg = train_s[seg_start:seg_end]
    seg_tgt = train_tgt.iloc[seg_start:seg_end]
    if len(seg) <= seq_len: continue
    X = np.array([seg[i:i+seq_len].ravel() for i in range(len(seg) - seq_len + 1)])
    y = seg_tgt.values[seq_len-1:][:len(X)]
    X_parts.append(X[:len(y)]); y_parts.append(y)

X = np.concatenate(X_parts); y = np.concatenate(y_parts)
model = GBMWrapper("xgboost", n_targets=2, hp_overrides={
    "n_estimators": 1500, "max_depth": 6, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 1, "gamma": 0, "reg_alpha": 0, "reg_lambda": 1.0,
    "tree_method": "hist",
})
model.fit(X, y)

# Save the fitted GBMWrapper + scaler + feature columns + provenance
bundle = {
    "gbm_wrapper": model,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "feature_columns": list(train_feat.columns),
    "target_columns": list(train_tgt.columns),
    "seq_len": seq_len,
    "backbone": "xgboost",
    "composite": 7.1686,
    "notes": "XGBoost SOTA recipe champion. See winners/xgboost_exp1_sota_seed42/README.md",
}

out = "C:/Users/evija/autoresearch/autoresearch/autoresearch_results/winners/xgboost_exp1_sota_seed42/xgboost_model.pkl"
with open(out, "wb") as f:
    pickle.dump(bundle, f)
import os
print(f"Saved {out} ({os.path.getsize(out) / 1024 / 1024:.2f} MB)")
