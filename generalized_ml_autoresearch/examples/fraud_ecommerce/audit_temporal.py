"""Diagnose why the chronological holdout collapsed to AUC ~ 0.51."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "features.csv")
print(f"rows: {len(df):,}  cols: {list(df.columns)}")

n = len(df)
n_test = int(round(n * 0.2))
n_val = int(round(n * 0.1))
train = df.iloc[: n - n_val - n_test]
val = df.iloc[n - n_val - n_test : n - n_test]
test = df.iloc[n - n_test :]
print(f"train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

print("\n--- class balance per split ---")
for name, d in [("train", train), ("val", val), ("test", test)]:
    print(f"  {name}: fraud_rate={d['class'].mean():.4f}")

print("\n--- time_since_signup distribution per split ---")
for name, d in [("train", train), ("val", val), ("test", test)]:
    s = d["time_since_signup"]
    f = d.loc[d["class"] == 1, "time_since_signup"]
    c = d.loc[d["class"] == 0, "time_since_signup"]
    auc_alone = roc_auc_score(d["class"], -d["time_since_signup"])
    print(f"  {name}: median={s.median():.0f}  fraud_med={f.median():.0f}  clean_med={c.median():.0f}  "
          f"single-feat AUC={auc_alone:.4f}")

print("\n--- per-feature single-feature AUC on TEST set only ---")
for col in [c for c in df.columns if c != "class"]:
    a = roc_auc_score(test["class"], test[col])
    a = max(a, 1 - a)
    print(f"  {col:25s} AUC={a:.4f}")

print("\n--- KEY: is the chronological split CORRELATED with class? ---")
# If purchase_time correlates with class, the temporal split itself biases the test set.
print(f"  train fraud rate: {train['class'].mean():.4f}")
print(f"  val   fraud rate: {val['class'].mean():.4f}")
print(f"  test  fraud rate: {test['class'].mean():.4f}")

# Are device_id and country distributions stable over time?
print("\n--- categorical distribution shift (test-train Jensen-Shannon-ish) ---")
for col in ["country", "source", "browser"]:
    train_dist = train[col].value_counts(normalize=True).sort_index()
    test_dist = test[col].value_counts(normalize=True).reindex(train_dist.index, fill_value=0)
    new_in_test = set(test[col].unique()) - set(train[col].unique())
    print(f"  {col}: {len(new_in_test)} categories in test never seen in train  "
          f"(test top freq: {test[col].value_counts().iloc[0]/len(test):.3f}, "
          f"train top freq: {train[col].value_counts().iloc[0]/len(train):.3f})")

# device_id is super-high cardinality so it's almost-always-new; that's expected
new_devs = set(test["device_id"].unique()) - set(train["device_id"].unique())
print(f"  device_id: {len(new_devs):,} of {test['device_id'].nunique():,} test devices unseen "
      f"({100*len(new_devs)/test['device_id'].nunique():.1f}%)")

# Quick sanity model: random forest on numeric subset, no device_id
print("\n--- sanity model: XGBoost without device_id, holding chronological split ---")
import xgboost as xgb
features = [c for c in df.columns if c not in ("class", "device_id", "ip_address")]
print(f"  using features: {features}")
clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.85, colsample_bytree=0.85,
                        random_state=0, tree_method="hist", verbosity=0)
clf.fit(train[features].to_numpy(), train["class"].to_numpy(),
        eval_set=[(val[features].to_numpy(), val["class"].to_numpy())], verbose=False)
test_proba = clf.predict_proba(test[features].to_numpy())[:, 1]
print(f"  test AUC: {roc_auc_score(test['class'], test_proba):.4f}")

print("\n--- final sanity: check time_since_signup direction is consistent in TEST ---")
# In train, fraud has SMALL time_since_signup (median 1s)
# Verify same is true in test
tr_fraud_med = train.loc[train['class']==1, 'time_since_signup'].median()
te_fraud_med = test.loc[test['class']==1, 'time_since_signup'].median()
tr_clean_med = train.loc[train['class']==0, 'time_since_signup'].median()
te_clean_med = test.loc[test['class']==0, 'time_since_signup'].median()
print(f"  train: fraud median={tr_fraud_med:.0f}s  clean median={tr_clean_med:.0f}s  "
      f"(separation={tr_clean_med - tr_fraud_med:.0f})")
print(f"  test:  fraud median={te_fraud_med:.0f}s  clean median={te_clean_med:.0f}s  "
      f"(separation={te_clean_med - te_fraud_med:.0f})")
