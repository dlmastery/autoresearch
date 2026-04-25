"""Standalone inference for the Exp 6 XGBoost champion.

Usage:
    python predict.py path/to/features.csv > predictions.csv

Loads the saved model_checkpoint.pt, applies the saved StandardScaler params,
and emits per-row probability + binary prediction + confidence.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "model_checkpoint.pt"


def load_model():
    with open(CHECKPOINT, "rb") as f:
        payload = pickle.load(f)
    return payload


def predict(features_csv: str, output_csv: str | None = None):
    payload = load_model()
    df = pd.read_csv(features_csv)
    feat_cols = payload["feature_columns"]
    X = df[feat_cols].to_numpy(dtype=float)
    mu = payload["scaler_mean"]
    sigma = payload["scaler_scale"]
    Xs = (X - mu) / sigma
    model = payload["model"]
    probs = model.predict_proba(Xs)[:, 1]
    out = df.copy()
    out["pred_prob_fraud"] = probs
    out["pred_class"] = (probs > 0.5).astype(int)
    out["confidence"] = np.abs(probs - 0.5) * 2.0
    if output_csv:
        out.to_csv(output_csv, index=False)
    else:
        print(out.head(20).to_string())
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py input_features.csv [output.csv]")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
