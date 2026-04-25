"""Add rolling time-windowed velocity features (the documented AFD-TFI class).

For each row at time t, compute:
  device_id_count_1d  — # of training-period transactions of this device in [t-1d, t)
  device_id_count_7d  — same for last 7 days
  device_id_count_30d — same for last 30 days
  ip_count_1d / ip_count_7d / ip_count_30d
  device_fraud_count_7d — # of training-period FRAUD transactions of this device in last 7 days

Train-period window is the first 70% chronologically (n - n_val - n_test).
Test rows look up training-period counts; new entities get 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_TRAIN = HERE / "data" / "raw_train.csv"
RAW_TEST = HERE / "data" / "raw_test.csv"
SRC = HERE / "data" / "features_velocity.csv"
OUT = HERE / "data" / "features_rolling.csv"


def main():
    raw_train = pd.read_csv(RAW_TRAIN)
    raw_test = pd.read_csv(RAW_TEST)
    raw = pd.concat([raw_train, raw_test], ignore_index=True)
    raw["purchase_time"] = pd.to_datetime(raw["purchase_time"])
    raw = raw.sort_values("purchase_time", kind="stable").reset_index(drop=True)

    feats = pd.read_csv(SRC)
    assert len(feats) == len(raw), f"length mismatch: feats={len(feats)} raw={len(raw)}"
    feats["purchase_time"] = raw["purchase_time"].values
    feats["device_id_raw"] = raw["device_id"].values
    feats["ip_address_raw"] = raw["ip_address"].values

    # Train-period boundary aligned with HoldoutSplit's first 70%
    n = len(feats)
    n_test = int(round(n * 0.2))
    n_val = int(round(n * 0.1))
    n_train = n - n_test - n_val
    feats["_in_train_window"] = (np.arange(n) < n_train).astype(int)
    print(f"train window: rows 0 to {n_train-1} ({n_train:,} rows)")

    # Helper: rolling time-windowed count of training-period rows where the entity matched
    # We use a fast pandas merge_asof approach: for each row, find prior rows with same entity
    # within the time window, restricted to training-period rows.
    train_view = feats[feats["_in_train_window"] == 1].copy()
    train_view = train_view.sort_values("purchase_time")

    def add_rolling(entity_col: str, label: str, windows: list):
        for d in windows:
            print(f"  computing {label}_count_{d}d ...")
            # Use Pandas rolling on time index per entity; then map back to all rows.
            # Approach: for each entity in train view, build a sorted timestamp series and
            # count how many fall in the window before each row's purchase_time.
            tv = train_view.sort_values([entity_col, "purchase_time"]).copy()
            tv["one"] = 1
            tv["dt"] = tv["purchase_time"]
            tv = tv.set_index("dt")
            # rolling count over a time delta, grouped by entity
            tv[f"_train_count_{d}"] = (
                tv.groupby(entity_col)["one"]
                .rolling(f"{d}D", closed="left").sum()
                .reset_index(level=0, drop=True)
                .fillna(0)
                .astype(int)
            )
            tv = tv.reset_index()
            # Now we have count for each TRAINING-PERIOD row. For test/val rows we need
            # to do a merge_asof: for each non-train row at time t and entity e, find the
            # latest training-period row with same entity and time <= t, then look up its
            # count_d to approximate the count over the preceding d days. Plus 1 for the
            # row itself if its time is within d days of that latest training row.
            # Simpler & exact: sort all rows by time, scan and count in-window.
            del tv

            # Exact computation via group + searchsorted (vectorized, fast)
            entity_to_train_times: dict = {}
            mask = feats["_in_train_window"] == 1
            for ent, ts in zip(feats.loc[mask, entity_col].values, feats.loc[mask, "purchase_time"].values):
                entity_to_train_times.setdefault(ent, []).append(ts)
            # convert to numpy datetime arrays once
            entity_to_train_times = {k: np.sort(np.array(v)) for k, v in entity_to_train_times.items()}

            window_td = np.timedelta64(d, "D")
            counts = np.zeros(len(feats), dtype=np.int32)
            ents = feats[entity_col].values
            ts_all = feats["purchase_time"].values
            for i in range(len(feats)):
                arr = entity_to_train_times.get(ents[i])
                if arr is None or len(arr) == 0:
                    continue
                t = ts_all[i]
                # count training rows with time in [t - d days, t)
                lo = np.searchsorted(arr, t - window_td, side="left")
                hi = np.searchsorted(arr, t, side="left")
                counts[i] = hi - lo
            feats[f"{label}_count_{d}d"] = counts

    add_rolling("device_id_raw", "device_id", [1, 7, 30])
    add_rolling("ip_address_raw", "ip_address", [1, 7, 30])

    # Fraud-count rolling: only training-period rows that are class=1
    print("  computing device_id_fraud_count_7d ...")
    fraud_mask = (feats["_in_train_window"] == 1) & (feats["class"] == 1)
    entity_to_fraud_times: dict = {}
    for ent, ts in zip(feats.loc[fraud_mask, "device_id_raw"].values,
                         feats.loc[fraud_mask, "purchase_time"].values):
        entity_to_fraud_times.setdefault(ent, []).append(ts)
    entity_to_fraud_times = {k: np.sort(np.array(v)) for k, v in entity_to_fraud_times.items()}
    window_td = np.timedelta64(7, "D")
    counts = np.zeros(len(feats), dtype=np.int32)
    ents = feats["device_id_raw"].values
    ts_all = feats["purchase_time"].values
    for i in range(len(feats)):
        arr = entity_to_fraud_times.get(ents[i])
        if arr is None or len(arr) == 0:
            continue
        t = ts_all[i]
        lo = np.searchsorted(arr, t - window_td, side="left")
        hi = np.searchsorted(arr, t, side="left")
        counts[i] = hi - lo
    feats["device_id_fraud_count_7d"] = counts

    # Drop helper columns; keep only the new numeric features
    feats = feats.drop(columns=["purchase_time", "device_id_raw", "ip_address_raw", "_in_train_window"])

    new_cols = [c for c in feats.columns if "_count_" in c]
    print(f"\nnew rolling features ({len(new_cols)}):")
    for c in new_cols:
        print(f"  {c:30s} mean={feats[c].mean():.3f} std={feats[c].std():.3f} max={feats[c].max()}")

    print(f"\ntotal columns: {len(feats.columns)}")
    feats.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
