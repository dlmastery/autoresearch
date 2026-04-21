"""After all GBM batches run, serialise each backbone's best experiment
to a pickle bundle in winners/, ready for cross-backbone ensembling."""
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, "C:/Users/evija/autoresearch")

import numpy as np
from sklearn.preprocessing import StandardScaler

from autoresearch.run_autoresearch import compute_all_features, compute_targets
from autoresearch.data.download import download_all_pairs, download_macro_signals
from autoresearch.data.splits import split_superfold
from autoresearch.model.backbone import GBMWrapper
from autoresearch.model.train import find_contiguous_segments

ROOT = Path("C:/Users/evija/autoresearch")
RESULTS = ROOT / "autoresearch" / "autoresearch_results"
WINNERS = RESULTS / "winners"

# --- Load data once ---
pairs = download_all_pairs()
macro = download_macro_signals()
feats = compute_all_features(pairs, macro)
targets = compute_targets(pairs["EURUSD=X"])
common = feats.index.intersection(targets.index)
feats = feats.loc[common]; targets = targets.loc[common]
train_feat, _, _ = split_superfold(feats)
train_tgt, _, _ = split_superfold(targets)

# --- Read experiment log, find best per-backbone ---
entries = [json.loads(l) for l in (RESULTS / "experiment_log.jsonl").read_text().splitlines() if l.strip()]
by_backbone = {}
for e in entries:
    b = e.get("backbone")
    if b not in {"xgboost", "lightgbm", "catboost"}: continue
    if b not in by_backbone or (e.get("composite") or -1e9) > by_backbone[b].get("composite", -1e9):
        by_backbone[b] = e

def extract_hp(entry, backbone):
    """Reconstruct GBM hp_overrides from stored config."""
    cfg = entry.get("config", {})
    hp = {}
    # Map stored keys back to GBM kwarg names
    alias = {
        "n_estimators": "n_estimators",
        "max_depth": "max_depth",
        "gbm_lr": "learning_rate",
        "subsample": "subsample",
        "colsample_bytree": "colsample_bytree",
        "reg_lambda": "reg_lambda",
        "reg_alpha": "reg_alpha",
        "min_child_weight": "min_child_weight",
        "gamma": "gamma",
        "num_leaves": "num_leaves",
        "feature_fraction": "feature_fraction",
        "bagging_fraction": "bagging_fraction",
        "min_data_in_leaf": "min_data_in_leaf",
        "iterations": "iterations",
        "depth": "depth",
        "l2_leaf_reg": "l2_leaf_reg",
        "random_strength": "random_strength",
        "bagging_temperature": "bagging_temperature",
        "bootstrap_type": "bootstrap_type",
        "seed": "random_state",
    }
    for k, v in cfg.items():
        if v is not None and k in alias:
            hp[alias[k]] = v
    return hp

for backbone, entry in by_backbone.items():
    exp_num = entry.get("experiment_num")
    composite = entry.get("composite")
    seq_len = entry.get("config", {}).get("seq_len", 10)
    hp = extract_hp(entry, backbone)
    # xgboost needs tree_method=hist default if not set
    if backbone == "xgboost" and "tree_method" not in hp:
        hp["tree_method"] = "hist"

    print(f"\n>>> {backbone} champion: Exp{exp_num} composite {composite:+.4f}")
    print(f"    seq_len={seq_len}, hp={hp}")

    # Fit scaler on training features
    scaler = StandardScaler()
    scaler.fit(train_feat.values)
    train_s = scaler.transform(train_feat.values)

    # Build training windows with champion seq_len
    X_parts, y_parts = [], []
    for seg_start, seg_end in find_contiguous_segments(train_feat.index):
        seg = train_s[seg_start:seg_end]
        seg_tgt = train_tgt.iloc[seg_start:seg_end]
        if len(seg) <= seq_len: continue
        X = np.array([seg[i:i+seq_len].ravel() for i in range(len(seg) - seq_len + 1)])
        y = seg_tgt.values[seq_len-1:][:len(X)]
        X_parts.append(X[:len(y)]); y_parts.append(y)
    X = np.concatenate(X_parts); y = np.concatenate(y_parts)

    model = GBMWrapper(backbone, n_targets=2, hp_overrides=hp)
    model.fit(X, y)

    # Determine a short-name slug
    slug_bits = [backbone, f"exp{exp_num}"]
    for k in ("max_depth", "gbm_lr", "depth", "num_leaves", "iterations"):
        v = entry.get("config", {}).get(k)
        if v is not None: slug_bits.append(f"{k.replace('_','')}{v}")
    if seq_len != 10: slug_bits.append(f"seq{seq_len}")
    slug = "_".join(str(s) for s in slug_bits)

    out_dir = WINNERS / slug
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "code").mkdir(exist_ok=True)
    bundle = {
        "gbm_wrapper": model,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "feature_columns": list(train_feat.columns),
        "target_columns": list(train_tgt.columns),
        "seq_len": seq_len, "backbone": backbone,
        "composite": composite,
        "hp": hp,
        "description": entry.get("description"),
        "experiment_num": exp_num,
    }
    out_path = out_dir / f"{backbone}_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"    Saved {out_path.relative_to(ROOT)} ({size_mb:.2f} MB)")

print("\n[OK] All available GBM champions serialised.")
