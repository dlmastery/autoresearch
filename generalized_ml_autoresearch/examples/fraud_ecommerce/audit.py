"""Audit script — verify the autoresearch result is real and not a leakage artifact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent

# --- 1. Load raw and processed datasets ---------------------------------------
raw_train = pd.read_csv(HERE / "data" / "raw_train.csv")
raw_test = pd.read_csv(HERE / "data" / "raw_test.csv")
raw = pd.concat([raw_train, raw_test], ignore_index=True)
proc = pd.read_csv(HERE / "data" / "features.csv")

print("=" * 78)
print("1. RAW DATASET SANITY")
print("=" * 78)
print(f"raw rows: {len(raw):,}")
print(f"raw columns: {list(raw.columns)}")
print(f"FDB documented:        151,112 rows × 14 columns (combined train+test)")
print(f"class balance: {raw['class'].value_counts().to_dict()}")
print(f"fraud rate: {raw['class'].mean():.4f}  (FDB: 0.0936-0.106)")

# --- 2. Duplicate-row check ---------------------------------------------------
print("\n" + "=" * 78)
print("2. DUPLICATE ROW CHECK")
print("=" * 78)
n_dup = raw.duplicated().sum()
print(f"exact duplicate rows in raw: {n_dup}")
n_dup_user = raw["user_id"].duplicated().sum()
print(f"duplicate user_id values:    {n_dup_user}  (each user_id should be unique)")

# --- 3. Per-fold entity overlap (the BIG one) ---------------------------------
print("\n" + "=" * 78)
print("3. ENTITY-LEVEL LEAKAGE ACROSS STRATIFIED 3-FOLD CV")
print("=" * 78)
print("If the same device_id / ip_address appears in train and test folds,")
print("the model can memorize entity-level fraud rather than learn transaction signal.")
print()

# Reproduce the runner's split exactly
y = proc["class"].to_numpy()
X_idx = np.arange(len(proc))
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Match raw row order to processed order. We concatenated train+test, so they line up.
device_id_raw = raw["device_id"].to_numpy()
ip_raw = raw["ip_address"].to_numpy()

print(f"unique device_id values: {pd.Series(device_id_raw).nunique():,}  (n_rows = {len(raw):,})")
print(f"unique ip_address values: {pd.Series(ip_raw).nunique():,}")
print()

def overlap_pct(train_set, test_vals):
    test_unique = set(test_vals)
    overlap = len(test_unique & train_set)
    return overlap, len(test_unique), 100 * overlap / max(len(test_unique), 1)

for fold, (tr, te) in enumerate(skf.split(X_idx, y)):
    tr_dev = set(device_id_raw[tr])
    tr_ip = set(ip_raw[tr])
    dev_ovl, dev_total, dev_pct = overlap_pct(tr_dev, device_id_raw[te])
    ip_ovl, ip_total, ip_pct = overlap_pct(tr_ip, ip_raw[te])
    print(f"fold {fold}: device_id overlap {dev_ovl}/{dev_total} ({dev_pct:.1f}% of test devices seen in train)")
    print(f"        ip_address overlap {ip_ovl}/{ip_total} ({ip_pct:.1f}% of test IPs seen in train)")

# --- 4. Device-id repeated-use signal ----------------------------------------
print("\n" + "=" * 78)
print("4. DEVICE-ID FRAUD SIGNAL (entity-level shortcut?)")
print("=" * 78)
device_counts = raw.groupby("device_id").agg(n=("class", "size"), n_fraud=("class", "sum"))
device_counts["fraud_rate"] = device_counts["n_fraud"] / device_counts["n"]
multi = device_counts[device_counts["n"] > 1]
print(f"devices with >1 transaction: {len(multi):,}  ({100*len(multi)/len(device_counts):.1f}% of unique devices)")
if len(multi) > 0:
    print(f"  among multi-tx devices, mean fraud rate: {multi['fraud_rate'].mean():.4f}")
    print(f"  among multi-tx devices, devices that are 100% fraud: "
          f"{(multi['fraud_rate'] == 1.0).sum():,} ({100*(multi['fraud_rate']==1.0).mean():.1f}%)")
    print(f"  among multi-tx devices, devices with 0% fraud: "
          f"{(multi['fraud_rate'] == 0.0).sum():,} ({100*(multi['fraud_rate']==0.0).mean():.1f}%)")
print(f"  --> if devices are bimodal (always-fraud OR always-clean) then a model that")
print(f"      memorizes device_id codes during training will recognize them at test")
print(f"      time across stratified-CV folds. This is real-world leakage.")

# --- 5. time_since_signup discrimination -------------------------------------
print("\n" + "=" * 78)
print("5. time_since_signup AS A SHORTCUT FEATURE")
print("=" * 78)
proc["time_since_signup"] = pd.to_numeric(proc["time_since_signup"])
fraud = proc.loc[proc["class"] == 1, "time_since_signup"]
clean = proc.loc[proc["class"] == 0, "time_since_signup"]
print(f"time_since_signup (seconds):")
print(f"  fraud: median={fraud.median():.1f}  mean={fraud.mean():.1f}  std={fraud.std():.1f}")
print(f"  clean: median={clean.median():.1f}  mean={clean.mean():.1f}  std={clean.std():.1f}")
# How separable on this feature alone?
from sklearn.metrics import roc_auc_score
auc_tss = roc_auc_score(proc["class"], -proc["time_since_signup"])  # smaller = more fraud
print(f"  AUC of -time_since_signup alone vs class: {auc_tss:.4f}")
print(f"  --> if AUC ~ 0.77 then time_since_signup IS the entire signal and the model")
print(f"      has no choice but to learn it. That's a real predictive feature, not leakage.")

# --- 6. Permutation-importance proxy: zero out features one at a time --------
print("\n" + "=" * 78)
print("6. SINGLE-FEATURE AUC RANKING (does any one feature carry the model?)")
print("=" * 78)
for col in [c for c in proc.columns if c != "class"]:
    try:
        x = proc[col].to_numpy()
        # try both directions
        a = roc_auc_score(proc["class"], x)
        a = max(a, 1 - a)
        print(f"  {col:25s} single-feature AUC = {a:.4f}")
    except Exception as e:
        print(f"  {col:25s} skipped ({e})")

# --- 7. Final verdict --------------------------------------------------------
print("\n" + "=" * 78)
print("AUDIT COMPLETE — read above for evidence")
print("=" * 78)
