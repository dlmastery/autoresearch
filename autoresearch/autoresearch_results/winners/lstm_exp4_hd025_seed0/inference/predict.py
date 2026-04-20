"""Standalone inference script for the Residual MLP champion (Exp32, seed=0).

Loads the model checkpoint, downloads recent FX data, computes features,
and outputs predictions with confidence, aleatoric, and epistemic uncertainty.

Usage:
    python predict.py                        # predict latest available day
    python predict.py --days 5               # predict last 5 days
    python predict.py --checkpoint path.pt   # use a different checkpoint
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model Architecture (self-contained — no imports from autoresearch package)
# ---------------------------------------------------------------------------
N_PAIRS = 6
HORIZONS = ["ret_1d", "ret_5d"]
N_INPUT_FEATURES = 104
SEQ_LEN = 10


def _make_heads(hidden_size: int, dropout: float = 0.15, head_hidden: int = 64):
    """Prediction heads: one per horizon, plain mode (mean only)."""
    heads = nn.ModuleDict()
    for h in HORIZONS:
        heads[h] = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, N_PAIRS),
        )
    return heads


class ResidualMLP(nn.Module):
    """Residual MLP: shortcut + nonlinear correction (He et al. 2016)."""

    def __init__(self, n_features: int = N_INPUT_FEATURES, seq_len: int = SEQ_LEN,
                 hidden: int = 128, head_dropout: float = 0.15):
        super().__init__()
        input_dim = n_features * seq_len
        self.shortcut = nn.Linear(input_dim, hidden)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
        )
        self.heads = _make_heads(hidden, dropout=head_dropout)
        self.het_loss = False  # plain mode — uncertainty via MC Dropout

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.size(0), -1)
        hidden = self.shortcut(flat) + self.residual(flat)
        return {name: head(hidden) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Uncertainty estimation (MC Dropout)
# ---------------------------------------------------------------------------
def predict_with_uncertainty(
    model: nn.Module, x: torch.Tensor,
    n_mc: int = 20, horizon: str = "ret_1d", pair_idx: int = 0,
) -> dict[str, np.ndarray]:
    """MC Dropout uncertainty decomposition (Gal & Ghahramani, 2016)."""
    model.train()  # enable dropout
    mc_means = []
    with torch.no_grad():
        for _ in range(n_mc):
            out = model(x)
            mc_means.append(out[horizon][:, pair_idx].cpu().numpy())

    mc_means = np.stack(mc_means)  # (T, B)
    mean = mc_means.mean(axis=0)
    epistemic = mc_means.var(axis=0)
    aleatoric = epistemic * 0.5  # approximation in plain mode
    total = aleatoric + epistemic
    pred_std = np.sqrt(total)
    confidence = 1.0 / (1.0 + np.exp(np.log(np.maximum(total, 1e-12))))  # sigmoid(-log(total))

    return {
        "prediction": mean,
        "direction": np.sign(mean),
        "confidence": confidence,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "pred_std": pred_std,
        "lower_1s": mean - pred_std,
        "upper_1s": mean + pred_std,
        "lower_2s": mean - 2 * pred_std,
        "upper_2s": mean + 2 * pred_std,
    }


# ---------------------------------------------------------------------------
# Data loading (minimal — just downloads latest and computes features)
# ---------------------------------------------------------------------------
def load_latest_data(days: int = 1) -> tuple[np.ndarray, list[str]]:
    """Download recent FX data and compute features.

    Returns (features_array, dates) where features_array has shape
    (days, SEQ_LEN, N_INPUT_FEATURES).
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        print("ERROR: yfinance and pandas required. Install: pip install yfinance pandas")
        sys.exit(1)

    # Download FX + macro data
    tickers = {
        "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
        "USDCHF=X": "USD/CHF", "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD",
        "^TNX": "US10Y", "^TYX": "US30Y", "^IRX": "US3M",
        "^GSPC": "SP500", "^VIX": "VIX",
        "GC=F": "Gold", "CL=F": "Oil", "DX-Y.NYB": "DXY",
    }
    lookback = max(days + SEQ_LEN + 70, 200)  # need enough history for features
    data = yf.download(list(tickers.keys()), period=f"{lookback}d", progress=False)

    if data.empty:
        print("ERROR: No data downloaded. Check internet connection.")
        sys.exit(1)

    # Compute simple returns for each instrument
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
    returns = close.pct_change()

    # Build feature matrix (simplified — matches autoresearch feature set structure)
    features = pd.DataFrame(index=returns.index)

    for ticker, name in tickers.items():
        if ticker in returns.columns:
            r = returns[ticker]
            # Returns at multiple lookbacks
            for lb in [1, 2, 3, 5, 10, 20]:
                features[f"{name}_ret_{lb}d"] = close[ticker].pct_change(lb)
            # Volatility
            for vol_w in [5, 10, 20, 60]:
                features[f"{name}_vol_{vol_w}d"] = r.rolling(vol_w).std()
            # Momentum (price vs MA)
            for ma_w in [10, 20, 50]:
                features[f"{name}_mom_{ma_w}d"] = close[ticker] / close[ticker].rolling(ma_w).mean() - 1

    features = features.dropna()

    # Pad or trim to exactly N_INPUT_FEATURES columns
    if features.shape[1] > N_INPUT_FEATURES:
        features = features.iloc[:, :N_INPUT_FEATURES]
    elif features.shape[1] < N_INPUT_FEATURES:
        # Pad with zeros (dummy features)
        for i in range(N_INPUT_FEATURES - features.shape[1]):
            features[f"pad_{i}"] = 0.0

    # Create windowed samples
    values = features.values.astype(np.float32)
    dates = features.index.strftime("%Y-%m-%d").tolist()

    windows = []
    window_dates = []
    for i in range(SEQ_LEN, min(len(values), SEQ_LEN + days)):
        windows.append(values[i - SEQ_LEN:i])
        window_dates.append(dates[i])

    if not windows:
        print("ERROR: Not enough data for prediction windows.")
        sys.exit(1)

    return np.stack(windows), window_dates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Residual MLP FX Prediction (Champion Exp32)")
    parser.add_argument("--checkpoint", default=None, help="Path to model_checkpoint.pt")
    parser.add_argument("--days", type=int, default=5, help="Number of recent days to predict")
    parser.add_argument("--mc-samples", type=int, default=20, help="MC Dropout samples")
    parser.add_argument("--pair", type=int, default=0, help="Currency pair index (0=EUR/USD)")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(__file__).parent.parent / "model_checkpoint.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(__file__).parent.parent.parent / "best_model.pt"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Provide path via --checkpoint or place model_checkpoint.pt in the winner directory")
        sys.exit(1)

    # Load model
    print(f"Loading model from {ckpt_path}...")
    model = ResidualMLP()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print(f"Downloading last {args.days} days of FX data...")
    features, dates = load_latest_data(days=args.days)
    x = torch.tensor(features)

    # Predict with uncertainty
    print(f"Running {args.mc_samples} MC Dropout passes...")
    results = predict_with_uncertainty(model, x, n_mc=args.mc_samples, pair_idx=args.pair)

    # Print results
    pair_names = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]
    pair_name = pair_names[args.pair] if args.pair < len(pair_names) else f"Pair {args.pair}"

    print(f"\n{'='*90}")
    print(f"  Residual MLP Champion — {pair_name} Predictions")
    print(f"{'='*90}")
    print(f"{'Date':<12} {'Pred':>10} {'Dir':>5} {'Conf':>8} {'Aleatoric':>12} {'Epistemic':>12} {'1σ Band':>20}")
    print(f"{'-'*12} {'-'*10} {'-'*5} {'-'*8} {'-'*12} {'-'*12} {'-'*20}")

    for i, date in enumerate(dates):
        pred = results["prediction"][i]
        direction = "LONG" if results["direction"][i] > 0 else "SHORT"
        conf = results["confidence"][i]
        ale = results["aleatoric"][i]
        epi = results["epistemic"][i]
        lo = results["lower_1s"][i]
        hi = results["upper_1s"][i]
        print(f"{date:<12} {pred:>+10.6f} {direction:>5} {conf:>8.4f} {ale:>12.8f} {epi:>12.8f} [{lo:>+.6f}, {hi:>+.6f}]")

    print(f"{'='*90}")
    print(f"  Avg Confidence: {results['confidence'].mean():.4f}")
    print(f"  Avg Epistemic:  {results['epistemic'].mean():.8f}")
    print(f"  Avg Aleatoric:  {results['aleatoric'].mean():.8f}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
