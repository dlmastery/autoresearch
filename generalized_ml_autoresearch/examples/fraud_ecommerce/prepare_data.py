"""Prepare the FDB-style Fraud ecommerce dataset for autoresearch.

Mirrors the FDB FraudecomPreProcessor logic:
  - Merges raw train + test (autoresearch does its own splits)
  - Engineers `time_since_signup` (purchase_time - signup_time, in seconds)
  - Drops high-cardinality / leakage columns: signup_time, purchase_time, sex, user_id
  - Label-encodes categorical strings (device_id, source, browser, country)
  - Keeps purchase_value, age, ip_address as numeric
  - Writes a single features.csv suitable for the runner's `format: csv` loader
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    raw_train = pd.read_csv(DATA / "raw_train.csv")
    raw_test = pd.read_csv(DATA / "raw_test.csv")
    df = pd.concat([raw_train, raw_test], ignore_index=True)
    print(f"raw rows: {len(df):,}  cols: {list(df.columns)}")

    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().astype(float)

    df = df.sort_values("purchase_time", kind="stable").reset_index(drop=True)
    print(f"sorted chronologically by purchase_time; range "
          f"{df['purchase_time'].min()} to {df['purchase_time'].max()}")

    df["purchase_hour"] = df["purchase_time"].dt.hour.astype(np.int32)
    df["purchase_dayofweek"] = df["purchase_time"].dt.dayofweek.astype(np.int32)
    df["signup_hour"] = df["signup_time"].dt.hour.astype(np.int32)

    df = df.drop(columns=["signup_time", "purchase_time", "sex", "user_id"])

    for col in ["device_id", "source", "browser", "country"]:
        df[col] = df[col].astype(str).fillna("UNK")
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes.astype(np.int32)

    df["ip_address"] = df["ip_address"].astype(float)
    df["purchase_value"] = df["purchase_value"].astype(float)
    df["age"] = df["age"].astype(float)
    df["class"] = df["class"].astype(int)

    fraud_rate = df["class"].mean()
    print(f"processed rows: {len(df):,}  fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
    print(f"final columns: {list(df.columns)}")

    out = DATA / "features.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
