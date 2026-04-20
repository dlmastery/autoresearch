"""Generate the airline-passengers-style time-series dataset (bundled, no network).

We synthesize a 144-month monthly series with trend + seasonality + noise, then
turn it into a supervised time-series-forecasting problem via lag features.
This mirrors the classic Box-Jenkins airline dataset without needing network
access (the dataset is tiny and well-behaved — perfect for a smoke example).

Output: writes `airline_features.csv` with columns:
  lag_1 .. lag_12  — previous 12 months' passenger counts (standardized context)
  month_sin, month_cos — cyclical month encoding
  trend — months since series start
  target — next month's passenger count
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build():
    rng = np.random.default_rng(0)
    n_months = 144
    t = np.arange(n_months)
    trend = 100 + 2.5 * t
    season = 30 * np.sin(2 * np.pi * t / 12) + 15 * np.cos(2 * np.pi * t / 6)
    noise = rng.standard_normal(n_months) * 8.0
    y = trend + season + noise

    rows = []
    for i in range(12, n_months - 1):
        row = {f"lag_{k}": y[i - k] for k in range(1, 13)}
        row["month_sin"] = np.sin(2 * np.pi * (i % 12) / 12)
        row["month_cos"] = np.cos(2 * np.pi * (i % 12) / 12)
        row["trend"] = i
        row["target"] = y[i + 1]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "airline_features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows, {df.shape[1]} columns.")


if __name__ == "__main__":
    build()
