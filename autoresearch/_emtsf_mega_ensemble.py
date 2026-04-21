"""EMTSF-style mega-ensemble: combine GBM 3-way seq=60 ensemble with the
best neural champions (LSTM, Mamba) via rank-averaged predictions.

Requires loading:
 - 3 GBM pickles (xgboost/lightgbm/catboost) at seq=60
 - LSTM champion checkpoint (best_model.pt from winners/lstm_exp35_wd7e4_bs16_seed42/)
 - Mamba champion checkpoint (winners/) -- if saved

Neural checkpoints are torch.nn.Module state_dicts; they need the
original architecture rebuilt + scaler applied + windowing at their
training seq_len (10 for LSTM, 10 for Mamba).

Strategy: predict on test set with each model at its own seq_len, align
predictions to a common timestamp index (the LATEST seq_len-start that
all models have), rank-average.
"""
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, "C:/Users/evija/autoresearch")

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from autoresearch.run_autoresearch import compute_all_features, compute_targets
from autoresearch.data.download import download_all_pairs, download_macro_signals
from autoresearch.data.splits import split_superfold, FOLDS, get_fold_dates
from autoresearch.model.backbone import GBMWrapper, create_model
from autoresearch.model.train import find_contiguous_segments, create_dataset
from autoresearch.evaluation.metrics import (
    trading_report, sharpe_ratio, information_coefficient, classification_metrics,
)

ROOT = Path("C:/Users/evija/autoresearch")
RESULTS = ROOT / "autoresearch" / "autoresearch_results"
WINNERS = RESULTS / "winners"

# Load data
pairs = download_all_pairs()
macro = download_macro_signals()
feats = compute_all_features(pairs, macro)
targets = compute_targets(pairs["EURUSD=X"])
common = feats.index.intersection(targets.index)
feats = feats.loc[common]; targets = targets.loc[common]
train_feat, _, test_feat = split_superfold(feats)
_, _, test_tgt = split_superfold(targets)

# Fit scaler on train
scaler = StandardScaler()
scaler.fit(train_feat.values)
n_features = train_feat.shape[1]

# === 1. GBM predictions (seq=60) ===
gbm_pickles = [
    WINNERS / "xgboost_exp203_maxdepth4_gbmlr0.01_seq60" / "xgboost_model.pkl",
    WINNERS / "lightgbm_exp235_maxdepth4_gbmlr0.01_seq60" / "lightgbm_model.pkl",
    WINNERS / "catboost_exp236_gbmlr0.01_depth4_seq60" / "catboost_model.pkl",
]
gbm_bundles = []
for p in gbm_pickles:
    if p.exists():
        gbm_bundles.append(pickle.load(open(p, "rb")))

def gbm_predict_fold(bundle, wf, wt):
    seq = bundle["seq_len"]
    ws = (wf.values - bundle["scaler_mean"]) / bundle["scaler_scale"]
    if len(ws) < seq + 1:
        return None, None, None
    X = np.array([ws[i:i+seq].ravel() for i in range(len(ws) - seq + 1)])
    y = wt.values[seq-1:][:len(X), 0]
    dates = wt.index[seq-1:][:len(X)]
    preds = bundle["gbm_wrapper"].predict(X)[:, 0]
    return dates, preds, y

# === 2. Neural predictions ===
# Load LSTM champion
lstm_ckpt_path = WINNERS / "lstm_exp35_wd7e4_bs16_seed42" / "model_checkpoint.pt"
neural_models = []

def load_neural(ckpt_path, backbone_name, seq_len):
    if not ckpt_path.exists(): return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = create_model(
        backbone=backbone_name, n_input_features=n_features, seq_len=seq_len,
        freeze_backbone=True,
        head_dropout=cfg.get("head_dropout", 0.25),
        het_loss=cfg.get("het_loss", False),
        hidden_size=cfg.get("hidden_size"),
        bidirectional=cfg.get("bidirectional"),
        num_layers=cfg.get("num_layers"),
        rnn_cell=cfg.get("rnn_cell"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler_mean = np.asarray(ckpt.get("scaler_mean"))
    scaler_scale = np.asarray(ckpt.get("scaler_scale"))
    return {"model": model, "seq_len": seq_len, "backbone": backbone_name,
            "scaler_mean": scaler_mean, "scaler_scale": scaler_scale}

lstm = load_neural(lstm_ckpt_path, "lstm", 10)
if lstm: neural_models.append(lstm)

def neural_predict_fold(bundle, wf, wt):
    seq = bundle["seq_len"]
    ws = (wf.values - bundle["scaler_mean"]) / bundle["scaler_scale"]
    if len(ws) < seq + 1:
        return None, None, None
    from torch.utils.data import DataLoader
    class _DS(torch.utils.data.Dataset):
        def __init__(self, f, t, L):
            self.f = torch.tensor(f, dtype=torch.float32)
            self.t = torch.tensor(t, dtype=torch.float32)
            self.L = L
        def __len__(self): return len(self.f) - self.L + 1
        def __getitem__(self, i):
            return self.f[i:i+self.L], self.t[i+self.L-1]
    ds = _DS(ws, wt.values, seq)
    loader = DataLoader(ds, batch_size=256)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            out = bundle["model"](x)
            preds.append(out["ret_1d"][:, 0].numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    y = wt.values[seq-1:][:len(preds), 0]
    dates = wt.index[seq-1:][:len(preds)]
    return dates, preds, y

# === 3. Per-fold, build aligned prediction matrix ===
print(f"GBM bundles loaded: {len(gbm_bundles)}")
print(f"Neural bundles loaded: {len(neural_models)}")
print()

all_returns = {"gbm_rank": [], "gbm_zscore": [], "mega_rank": [], "mega_zscore": []}
for fold in FOLDS:
    d = get_fold_dates(fold)
    wf = test_feat.loc[d["test_start"]:d["test_end"]]
    wt = test_tgt.loc[d["test_start"]:d["test_end"]]
    if len(wf) < 61:  # need at least seq=60 + 1
        continue

    # Collect (dates, preds) per model
    per_model = []
    for b in gbm_bundles:
        dt, p, y = gbm_predict_fold(b, wf, wt)
        if p is not None: per_model.append(("gbm_"+b["backbone"], dt, p, y))
    for b in neural_models:
        dt, p, y = neural_predict_fold(b, wf, wt)
        if p is not None: per_model.append(("nn_"+b["backbone"], dt, p, y))

    if not per_model: continue

    # Align on the LATEST seq_start -- i.e. the earliest common first-date
    latest_start = max(m[1][0] for m in per_model)
    aligned = []
    for name, dt, p, y in per_model:
        mask = dt >= latest_start
        aligned.append((name, dt[mask], p[mask], y[mask]))
    # Trim to shortest
    min_n = min(len(a[1]) for a in aligned)
    aligned = [(n, dt[:min_n], p[:min_n], y[:min_n]) for n, dt, p, y in aligned]
    y_true = aligned[0][3]  # all should match

    gbm_arr = np.column_stack([a[2] for a in aligned if a[0].startswith("gbm_")])
    mega_arr = np.column_stack([a[2] for a in aligned])

    # Rank-avg
    def rank_avg(arr):
        ranks = np.column_stack([rankdata(arr[:, c]) for c in range(arr.shape[1])])
        return ranks.mean(axis=1) - (len(arr) + 1) / 2

    def zscore_avg(arr):
        z = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-12)
        return z.mean(axis=1)

    all_returns["gbm_rank"].append(np.sign(rank_avg(gbm_arr)) * y_true)
    all_returns["gbm_zscore"].append(np.sign(zscore_avg(gbm_arr)) * y_true)
    all_returns["mega_rank"].append(np.sign(rank_avg(mega_arr)) * y_true)
    all_returns["mega_zscore"].append(np.sign(zscore_avg(mega_arr)) * y_true)

# Aggregate + report
print("="*72)
print("ENSEMBLE COMPARISON  (aligned on latest common start across models)")
print("="*72)
for name, fold_rets in all_returns.items():
    rets = np.concatenate(fold_rets)
    rpt = trading_report(rets)
    sh = sharpe_ratio(rets)
    print(f"{name:<14} n={len(rets):<5} Sharpe={sh:+.4f}  "
          f"Ret={rpt['total_return_pct']:+.2f}%  WinRate={rpt['win_rate']:.1f}%")
