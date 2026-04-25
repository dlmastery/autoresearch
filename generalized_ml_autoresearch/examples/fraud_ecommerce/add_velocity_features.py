"""Compute entity-velocity / frequency features using TRAIN-ONLY counts to avoid leakage.

AFD TFI's documented strength is entity-level aggregation. We mirror that with:
  device_id_freq        — count of each device_id in the training period
  ip_address_freq       — count of each ip_address in the training period
  country_freq          — count of each country in the training period
  source_freq           — count of each source in the training period
  browser_freq          — count of each browser in the training period
  device_id_fraud_rate  — train-only fraud rate per device_id (smoothed)

For chronological 80/20 evaluation, "training period" = first 80% chronologically.
Test rows look up these maps; unseen entities get 0 freq and global fraud rate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "features.csv"
OUT = HERE / "data" / "features_velocity.csv"


def main():
    df = pd.read_csv(SRC)
    print(f"loaded {len(df):,} rows, {len(df.columns)} cols")
    # The runner's HoldoutSplit with order=time slices: train=first 70%, val=next 10%, test=last 20%.
    # To avoid leaking val/test rows into the velocity feature computation, the "training period"
    # for these counts MUST be exactly the first 70% — not the first 80%.
    n = len(df)
    n_test = int(round(n * 0.2))
    n_val = int(round(n * 0.1))
    n_train = n - n_test - n_val
    train_view = df.iloc[:n_train]
    print(f"train view (chronological first 70%, n_train={n_train:,}; "
          f"val={n_val}, test={n_test} held out from feature computation)")
    global_fraud_rate = train_view["class"].mean()
    print(f"global train fraud rate: {global_fraud_rate:.4f}")

    smoothing = 5.0  # additive smoothing for fraud rates

    for col in ["device_id", "ip_address", "country", "source", "browser"]:
        freq_map = train_view[col].value_counts().to_dict()
        df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype(np.int32)

    # Per-device train-only fraud rate (Bayesian smoothed toward global)
    grp = train_view.groupby("device_id")["class"].agg(["sum", "count"])
    grp["smoothed"] = (grp["sum"] + smoothing * global_fraud_rate) / (grp["count"] + smoothing)
    fraud_rate_map = grp["smoothed"].to_dict()
    df["device_fraud_rate_train"] = df["device_id"].map(fraud_rate_map).fillna(global_fraud_rate).astype(float)

    # Per-country train-only fraud rate
    grp = train_view.groupby("country")["class"].agg(["sum", "count"])
    grp["smoothed"] = (grp["sum"] + smoothing * global_fraud_rate) / (grp["count"] + smoothing)
    country_fr_map = grp["smoothed"].to_dict()
    df["country_fraud_rate_train"] = df["country"].map(country_fr_map).fillna(global_fraud_rate).astype(float)

    print(f"new feature stats:")
    for c in [c for c in df.columns if c.endswith("_freq") or c.endswith("_train")]:
        print(f"  {c:30s} mean={df[c].mean():.4f} std={df[c].std():.4f}")

    print(f"final cols ({len(df.columns)}): {list(df.columns)}")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
