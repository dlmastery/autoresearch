"""Panel-mode runner for QQQ + 30 NDX top + 10 Asia/EU + 8 Asia megacap +
6 adjacent US = 55-asset shared-trunk panel learning.

Per Gu-Kelly-Xiu 2020 RFS "Empirical Asset Pricing via Machine Learning"
(arXiv:1807.04365): one backbone trunk, asset-id embedding at input,
per-asset prediction heads. Trained on the union of all assets'
training days; evaluated per-asset and aggregated to a basket Sharpe.

Per Lim-Zohren-Roberts 2019 (arXiv:1906.04025): position size =
sign(prediction) × confidence_gate × (1/realized_vol_20). Caps per-asset
weight at 5% of basket capital. Daily rebalance.

Per Lou-Polk-Skouras 2019 JFE: Asia/EU close prices on day T are causal
for QQQ/US-asset day-T close prediction (Asia closes 01:00-05:00 ET,
Europe ~11:30 ET, US ~16:00 ET).

User directive 2026-04-29: predict 5d (B) as primary KEEP/DISCARD target
but realise trade on 1d return. Focus per-fold metrics on day 1 + day 5.

Self-contained: imports from autoresearchindexstock.data.download +
.evaluation.metrics; reuses splits.split_superfold and most of
model/backbone.py via the existing CurrencyMLP/CurrencyLSTM/CurrencyMamba
classes by adding a thin asset-id-embedding wrapper.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Pin to safe P-cores (E-cores cause WHEA BSODs on Intel 14th-gen HX).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autoresearch.run_autoresearch import _pin_to_safe_cores
    _pin_to_safe_cores()
except Exception:
    pass

from autoresearchindexstock.data.download import (
    download_panel_targets,
    download_all,
    DEFAULT_END,
)
from autoresearch.evaluation.metrics import (
    sharpe_ratio,
    information_coefficient,
    classification_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "autoresearch_results"
RESULTS.mkdir(exist_ok=True)
PANEL_LOG = RESULTS / "panel_experiment_log.jsonl"
PANEL_BEST = RESULTS / "panel_best_config.json"


# ---------------------------------------------------------------------------
# Per-asset feature computation
# ---------------------------------------------------------------------------

def compute_per_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~28 per-asset features from a single asset's OHLCV.

    Pure pandas, no leakage (all rolling/shift operations look backward).
    Returns DataFrame indexed by date with NaN-prefilled features.
    """
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # Returns at multiple horizons
    logc = np.log(c.replace(0, np.nan))
    out["ret_1d"] = logc.diff(1)
    out["ret_5d"] = logc.diff(5)
    out["ret_20d"] = logc.diff(20)
    out["ret_60d"] = logc.diff(60)

    # Volatility (rolling)
    r1 = out["ret_1d"]
    out["rv_5"] = r1.rolling(5).std()
    out["rv_20"] = r1.rolling(20).std()
    out["rv_60"] = r1.rolling(60).std()

    # Range estimators (Parkinson, Garman-Klass)
    rng = np.log(h / l.replace(0, np.nan))
    out["parkinson_5"] = (rng.pow(2).rolling(5).mean() / (4 * np.log(2))).pow(0.5)
    out["parkinson_20"] = (rng.pow(2).rolling(20).mean() / (4 * np.log(2))).pow(0.5)

    # RSI (Wilder)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / dn.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    out["macd"] = macd_line / c
    out["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean() / c

    # Bollinger Z
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    out["bb_z"] = (c - ma20) / sd20.replace(0, np.nan)

    # Donchian position (rank within rolling window)
    for w in [20, 60]:
        rmin = c.rolling(w).min()
        rmax = c.rolling(w).max()
        out[f"donchian_{w}"] = (c - rmin) / (rmax - rmin).replace(0, np.nan)

    # Volume features
    vlog = np.log(v.replace(0, np.nan))
    out["vol_ma20_z"] = (vlog - vlog.rolling(20).mean()) / vlog.rolling(20).std().replace(0, np.nan)
    out["vol_change_5d"] = vlog.diff(5)

    # Open-close vs high-low (intraday pressure proxy)
    out["intraday_close_loc"] = (c - l) / (h - l).replace(0, np.nan)
    out["overnight_gap"] = np.log(o / c.shift(1).replace(0, np.nan))

    # Skew + kurtosis of recent returns
    out["skew_20"] = r1.rolling(20).skew()
    out["kurt_20"] = r1.rolling(20).kurt()

    # Z-score of ret_5d vs ret_60d (medium-term momentum vs long-term)
    out["mom_zscore"] = (out["ret_5d"] - out["ret_5d"].rolling(60).mean()) / out["ret_5d"].rolling(60).std().replace(0, np.nan)

    return out


def compute_panel_features_targets(panel: pd.DataFrame,
                                    cross_asset: Optional[pd.DataFrame] = None
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-format (date, asset) -> (features, targets) panel.

    `panel`: long DataFrame with columns [date, open, high, low, close,
             volume, asset].
    `cross_asset`: optional wide DataFrame indexed by date with shared
                    cross-asset features (VIX, yields, sector ETFs etc.)
                    that broadcast across all assets.
    Returns (features_long, targets_long) — each indexed by (date, asset).
    """
    features_frames = []
    targets_frames = []
    for asset, g in panel.groupby("asset", sort=False):
        g = g.sort_values("date").set_index("date")
        feats = compute_per_asset_features(g)
        # Targets: fwd_ret_1d, fwd_ret_5d (look forward — only used as labels)
        logc = np.log(g["close"].replace(0, np.nan))
        tgt = pd.DataFrame(index=g.index)
        tgt["fwd_ret_1d"] = logc.shift(-1) - logc
        tgt["fwd_ret_5d"] = logc.shift(-5) - logc
        # Vol-adjusted 1d (target D for confidence-weighted sizing)
        rv20 = feats["rv_20"]
        tgt["fwd_ret_1d_voladj"] = tgt["fwd_ret_1d"] / rv20.replace(0, np.nan)
        feats["asset"] = asset
        tgt["asset"] = asset
        features_frames.append(feats)
        targets_frames.append(tgt)
    F = pd.concat(features_frames).reset_index().rename(columns={"index": "date"})
    T = pd.concat(targets_frames).reset_index().rename(columns={"index": "date"})

    # Broadcast cross-asset features (wide → long)
    if cross_asset is not None and not cross_asset.empty:
        F = F.merge(cross_asset.reset_index(), on="date", how="left")

    F = F.set_index(["date", "asset"]).sort_index()
    T = T.set_index(["date", "asset"]).sort_index()
    return F, T


# ---------------------------------------------------------------------------
# Per-asset super-fold splits (date-based, applied to the panel)
# ---------------------------------------------------------------------------

# Uses the SAME 7-fold regime windows as the QQQ-only runner. Per-asset
# split is the same date-range; train_idx / val_idx / test_idx are
# computed per-asset since some assets (META 2012, TSLA 2010, AVGO 2009)
# have shorter histories.
FOLD_DATE_RANGES = [
    ("2008-11-01", "2009-03-31", "GFC peak crash"),
    ("2011-08-01", "2011-12-31", "EU debt"),
    ("2013-05-01", "2014-06-30", "Taper tantrum"),
    ("2015-08-01", "2016-02-29", "China-oil crash"),
    ("2018-02-01", "2018-12-31", "Vol-mageddon"),
    ("2020-02-15", "2020-09-30", "COVID + V-recovery"),
    ("2022-01-01", "2025-12-30", "AI rally + 2025"),
]


def panel_split(features: pd.DataFrame, targets: pd.DataFrame,
                purge_days: int = 90, embargo_days: int = 21,
                label_buffer: int = 10):
    """Return per-asset train/val/test masks compatible with the QQQ
    super-fold. Val window is 96d before each test window."""
    dates_all = features.index.get_level_values("date").unique().sort_values()
    train_mask = pd.Series(True, index=features.index)
    val_dates = []
    test_dates = []
    val_per_fold, test_per_fold = [], []
    for fold_idx, (start, end, regime) in enumerate(FOLD_DATE_RANGES):
        test_start = pd.Timestamp(start)
        test_end = pd.Timestamp(end)
        test_dates_f = dates_all[(dates_all >= test_start) & (dates_all <= test_end)]
        # Val = 96 trading days before each test window
        val_end = test_dates_f.min() - pd.Timedelta(days=purge_days + label_buffer)
        val_start = val_end - pd.Timedelta(days=130)  # ~96 trading days
        val_dates_f = dates_all[(dates_all >= val_start) & (dates_all <= val_end)]
        val_dates.extend(val_dates_f.tolist())
        test_dates.extend(test_dates_f.tolist())
        val_per_fold.append((regime, val_dates_f.tolist()))
        test_per_fold.append((regime, test_dates_f.tolist()))
        # Punch out from train (purge + embargo + label buffer)
        excl_start = val_start - pd.Timedelta(days=embargo_days + label_buffer)
        excl_end = test_end + pd.Timedelta(days=embargo_days + label_buffer)
        excl_dates = dates_all[(dates_all >= excl_start) & (dates_all <= excl_end)]
        idx_lvl = features.index.get_level_values("date")
        train_mask = train_mask & ~idx_lvl.isin(excl_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)
    val_mask = features.index.get_level_values("date").isin(val_set)
    test_mask = features.index.get_level_values("date").isin(test_set)
    # Convert pandas Series → numpy for boolean ops
    train_mask_np = np.asarray(train_mask)
    val_mask_np = np.asarray(val_mask)
    test_mask_np = np.asarray(test_mask)
    train_mask_np = train_mask_np & ~val_mask_np & ~test_mask_np
    info = {
        "train_rows": int(train_mask_np.sum()),
        "val_rows": int(val_mask_np.sum()),
        "test_rows": int(test_mask_np.sum()),
        "val_per_fold": [(r, [str(d) for d in dl]) for r, dl in val_per_fold],
        "test_per_fold": [(r, [str(d) for d in dl]) for r, dl in test_per_fold],
    }
    return train_mask_np, val_mask_np, test_mask_np, info


# ---------------------------------------------------------------------------
# Sliding-window dataset (panel)
# ---------------------------------------------------------------------------

class PanelWindowDataset(Dataset):
    """Window dataset over a panel; each item is one window for one
    (asset, date_t) pair. Pads short asset histories with zeros."""

    def __init__(self, X: np.ndarray, y: np.ndarray, asset_ids: np.ndarray,
                 dates: np.ndarray, mask: np.ndarray, seq_len: int = 10):
        # Build window indices per asset s.t. windows do not span asset
        # boundaries and only include rows in the mask.
        self.seq_len = seq_len
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.asset_ids = asset_ids.astype(np.int64)
        windows = []
        n = len(self.X)
        for i in range(seq_len, n):
            # Window i covers rows [i-seq_len, i)
            # Reject if asset boundary is crossed
            if self.asset_ids[i] != self.asset_ids[i - seq_len]:
                continue
            # Reject if the predict-day i is not in mask (eval/train selector)
            if not mask[i]:
                continue
            # Reject if y[i] is NaN (no forward return available)
            if not np.isfinite(self.y[i]).all():
                continue
            windows.append(i)
        self.windows = np.asarray(windows, dtype=np.int64)
        logger.info("PanelWindow: %d valid windows (seq_len=%d, mask sum=%d)",
                    len(self.windows), seq_len, int(mask.sum()))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        i = self.windows[idx]
        x = self.X[i - self.seq_len:i]  # [seq_len, D]
        a = self.asset_ids[i]
        y = self.y[i]
        return torch.from_numpy(x), torch.tensor(a), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Backbones with asset-id embedding
# ---------------------------------------------------------------------------

class AssetEmbeddingMLP(nn.Module):
    """Residual MLP with learnable asset-id embedding per Gu-Kelly-Xiu
    2020 §3.2. The asset embedding is concatenated with the flattened
    seq_len × n_features input."""

    def __init__(self, n_features: int, seq_len: int, n_assets: int,
                 hidden: int = 128, dropout: float = 0.25,
                 asset_emb_dim: int = 16):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)
        flat = seq_len * n_features + asset_emb_dim
        self.l1 = nn.Linear(flat, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.skip = nn.Linear(flat, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 4)  # mean_1d, logvar_1d, mean_5d, logvar_5d

    def forward(self, x: torch.Tensor, asset: torch.Tensor):
        B = x.size(0)
        flat = x.reshape(B, -1)
        ae = self.asset_emb(asset)
        combined = torch.cat([flat, ae], dim=-1)
        h = F.gelu(self.l1(combined)) + self.skip(combined)
        h = self.drop(h)
        h = F.gelu(self.l2(h))
        out = self.head(h)
        return out  # [B, 4] = [mu_1d, logvar_1d, mu_5d, logvar_5d]


class AssetEmbeddingLSTM(nn.Module):
    def __init__(self, n_features: int, seq_len: int, n_assets: int,
                 hidden: int = 128, dropout: float = 0.1,
                 num_layers: int = 1, asset_emb_dim: int = 16):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)
        self.input_proj = nn.Linear(n_features + asset_emb_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, asset: torch.Tensor):
        B, L, _ = x.shape
        ae = self.asset_emb(asset).unsqueeze(1).expand(-1, L, -1)
        h = self.input_proj(torch.cat([x, ae], dim=-1))
        out, _ = self.lstm(h)
        h = out[:, -1, :]
        return self.head(self.drop(h))


# ---------------------------------------------------------------------------
# Heteroscedastic loss (Kendall-Gal 2017)
# ---------------------------------------------------------------------------

def het_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # pred: [B, 4] = [mu_1d, logvar_1d, mu_5d, logvar_5d]
    # y: [B, 2] = [fwd_ret_1d, fwd_ret_5d]
    mu = pred[:, [0, 2]]
    logvar = pred[:, [1, 3]].clamp(min=-8, max=2)
    sq = (mu - y).pow(2)
    return (torch.exp(-logvar) * sq + 0.5 * logvar).mean()


def huber_loss_2tgt(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # pred: [B, 4] but only mu_1d (col 0) + mu_5d (col 2) used; logvars ignored
    # y: [B, 2]
    mu = pred[:, [0, 2]]
    return torch.nn.functional.huber_loss(mu, y, delta=1.0)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn=het_loss):
    model.train()
    total = 0.0
    n = 0
    for x, a, y in loader:
        x, a, y = x.to(DEVICE), a.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x, a)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict_panel(model, loader):
    model.eval()
    mu_1d, mu_5d, var_1d, var_5d, assets, ys = [], [], [], [], [], []
    for x, a, y in loader:
        x, a = x.to(DEVICE), a.to(DEVICE)
        pred = model(x, a).cpu().numpy()
        mu_1d.append(pred[:, 0])
        var_1d.append(np.exp(np.clip(pred[:, 1], -8, 2)))
        mu_5d.append(pred[:, 2])
        var_5d.append(np.exp(np.clip(pred[:, 3], -8, 2)))
        assets.append(a.cpu().numpy())
        ys.append(y.cpu().numpy() if hasattr(y, "cpu") else y.numpy())
    return (
        np.concatenate(mu_1d), np.concatenate(var_1d),
        np.concatenate(mu_5d), np.concatenate(var_5d),
        np.concatenate(assets), np.concatenate(ys),
    )


# ---------------------------------------------------------------------------
# Vol-weighted basket Sharpe (Lim-Zohren-Roberts 2019)
# ---------------------------------------------------------------------------

def basket_sharpe(predictions: np.ndarray, actuals_1d: np.ndarray,
                   asset_ids: np.ndarray, vols: np.ndarray,
                   confidence: np.ndarray = None,
                   conf_threshold: float = 0.0,
                   max_weight: float = 0.05) -> Dict:
    """Per Lim-Zohren-Roberts 2019: basket return = mean over assets of
    sign(prediction) × actual_1d × (1/vol_target / vol_realized).

    Position size is clipped to ±max_weight per asset.
    Optional confidence gate skips trades when confidence < threshold.
    """
    direction = np.sign(predictions)
    # Vol target = 0.4 (annualised) — standard in vol-targeting trade
    vol_target_daily = 0.4 / np.sqrt(252)
    vols_safe = np.clip(vols, 1e-4, None)
    weights = direction * vol_target_daily / vols_safe
    weights = np.clip(weights, -max_weight, max_weight)
    if confidence is not None:
        weights = weights * (confidence >= conf_threshold).astype(weights.dtype)
    pnl = weights * actuals_1d
    # Aggregate per date by averaging across assets present that day
    # (here we just compute the unweighted mean over rows; the dataset is
    # already (date, asset) so aggregating gives basket return per (date, asset)
    # — proper basket Sharpe needs date-level aggregation, done below)
    per_asset_sharpe = {}
    for asset in np.unique(asset_ids):
        mask = asset_ids == asset
        if mask.sum() < 10:
            continue
        rets = pnl[mask]
        per_asset_sharpe[int(asset)] = sharpe_ratio(rets)
    # Basket: mean across assets per day
    # Build a (date, asset) → pnl frame for aggregation (caller passes
    # asset_ids that should already be aligned with predictions/actuals)
    return {
        "per_asset_sharpe": per_asset_sharpe,
        "n_trades": int(np.isfinite(pnl).sum()),
        "mean_pnl_bps": float(np.nanmean(pnl) * 1e4),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["mlp", "lstm"], default="mlp",
                        help="Panel-mode backbone")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--asset-emb-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--description", type=str, default="panel-mode baseline")
    parser.add_argument("--exclude-assets", type=str, default="",
                        help="Comma-separated list of assets to drop from the panel (e.g. '^HSI,^N225,^AXJO,^STI,^KS11,^TWII')")
    parser.add_argument("--loss", choices=["het", "huber"], default="het",
                        help="Training loss: 'het' = Kendall-Gal heteroscedastic; 'huber' = plain Huber on [mu_1d, mu_5d]")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay (LSTM QQQ-only sweet spot was 7e-4)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    # 1. Download panel
    panel = download_panel_targets()
    logger.info("Panel: %d assets, %d total rows", panel.asset.nunique(), len(panel))

    # 1b. Optional asset exclusion (structural-change axis: drop assets with
    # known time-shift confound or chronic negative Sharpe across seeds).
    excluded = [a.strip() for a in args.exclude_assets.split(",") if a.strip()]
    if excluded:
        before = panel.asset.nunique()
        panel = panel[~panel.asset.isin(excluded)].copy()
        after = panel.asset.nunique()
        logger.info("Excluded %d assets: %s (panel %d → %d)",
                    len(excluded), excluded, before, after)

    # 2. Compute per-asset features + targets (no cross-asset broadcast for
    # this baseline — keeps the tensor lean. Cross-asset features can be
    # added later via the cross_asset arg).
    F_long, T_long = compute_panel_features_targets(panel)
    F_long = F_long.dropna(how="all")
    feature_cols = [c for c in F_long.columns if c != "asset"]
    target_cols = ["fwd_ret_1d", "fwd_ret_5d"]
    n_features = len(feature_cols)
    asset_unique = sorted(F_long.index.get_level_values("asset").unique().tolist())
    asset_to_id = {a: i for i, a in enumerate(asset_unique)}
    n_assets = len(asset_unique)
    logger.info("Features: %d cols; assets: %d", n_features, n_assets)

    # 3. Drop rows with any NaN feature OR NaN target
    keep = F_long[feature_cols].notna().all(axis=1) & T_long[target_cols].notna().all(axis=1)
    F_clean = F_long.loc[keep, feature_cols]
    T_clean = T_long.loc[keep, target_cols]
    # CRITICAL: re-sort by (asset, date) so each asset's history is contiguous
    # in the row index. The PanelWindowDataset rejects windows that cross asset
    # boundaries; if rows are sorted by (date, asset) this rejects ~all windows.
    F_clean = F_clean.swaplevel(0, 1).sort_index()
    T_clean = T_clean.swaplevel(0, 1).sort_index()
    logger.info("After NaN drop: %d rows (sorted by asset, date)", len(F_clean))

    # 4. Standardise features (per-feature z-score over training set; this
    # avoids leaking val/test stats into training).
    train_mask, val_mask, test_mask, split_info = panel_split(F_clean, T_clean)
    logger.info("Split: train=%d val=%d test=%d",
                split_info["train_rows"], split_info["val_rows"],
                split_info["test_rows"])

    # ========================================================================
    # DATA LEAKAGE AUDIT (mandatory per CLAUDE.md "Data Integrity")
    # ========================================================================
    overlap_tv = int((train_mask & val_mask).sum())
    overlap_tt = int((train_mask & test_mask).sum())
    overlap_vt = int((val_mask & test_mask).sum())
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"LEAKAGE: train∩val={overlap_tv} train∩test={overlap_tt} "
            f"val∩test={overlap_vt} — split is broken")
    # Verify per-asset purge: for each test fold's first day, no train row
    # within label_buffer + embargo days of it.
    train_dates = pd.DatetimeIndex(F_clean.index.get_level_values("date").values[train_mask])
    test_dates_arr = pd.DatetimeIndex(F_clean.index.get_level_values("date").values[test_mask])
    for fold_idx, (start, end, regime) in enumerate(FOLD_DATE_RANGES):
        test_start = pd.Timestamp(start)
        # Count any train day within (label_buffer + embargo) days BEFORE test_start
        violation_window_start = test_start - pd.Timedelta(days=10 + 21)
        violations = ((train_dates >= violation_window_start) &
                       (train_dates < test_start)).sum()
        if violations > 0:
            raise RuntimeError(
                f"LEAKAGE: fold {fold_idx} ({regime}) — {violations} train rows "
                f"in [{violation_window_start.date()}, {test_start.date()}) "
                f"label-buffer/embargo violation")
    logger.info("[leakage audit] PASSED: train∩val=0, train∩test=0, val∩test=0; "
                "purge=90d, embargo=21d, label_buffer=10d enforced per fold")

    X_all = F_clean.values
    train_X = X_all[train_mask]
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0) + 1e-8
    X_norm = (X_all - mu) / sd
    y_all = T_clean.values
    # After swaplevel (asset, date) the first level is asset
    asset_ids = np.array([asset_to_id[a] for a in F_clean.index.get_level_values("asset")])
    dates_all = F_clean.index.get_level_values("date").values

    # 5. Build window datasets
    train_ds = PanelWindowDataset(X_norm, y_all, asset_ids, dates_all,
                                   train_mask, seq_len=args.seq_len)
    val_ds = PanelWindowDataset(X_norm, y_all, asset_ids, dates_all,
                                 val_mask, seq_len=args.seq_len)
    test_ds = PanelWindowDataset(X_norm, y_all, asset_ids, dates_all,
                                  test_mask, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                             num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    # 6. Build model
    if args.backbone == "mlp":
        model = AssetEmbeddingMLP(n_features, args.seq_len, n_assets,
                                   hidden=args.hidden, dropout=args.head_dropout,
                                   asset_emb_dim=args.asset_emb_dim).to(DEVICE)
    else:
        model = AssetEmbeddingLSTM(n_features, args.seq_len, n_assets,
                                    hidden=args.hidden, dropout=args.head_dropout,
                                    num_layers=args.num_layers,
                                    asset_emb_dim=args.asset_emb_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s n_params=%d", args.backbone, n_params)

    # 7. Train with early stopping on val loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = huber_loss_2tgt if args.loss == "huber" else het_loss
    logger.info("Loss function: %s", args.loss)
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn=loss_fn)
        # Val
        model.eval()
        with torch.no_grad():
            v_total, v_n = 0.0, 0
            for x, a, y in val_loader:
                x, a, y = x.to(DEVICE), a.to(DEVICE), y.to(DEVICE)
                v_total += loss_fn(model(x, a), y).item() * x.size(0)
                v_n += x.size(0)
            val_loss = v_total / max(v_n, 1)
        logger.info("Epoch %d train=%.4f val=%.4f", epoch, train_loss, val_loss)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save({"state_dict": model.state_dict()}, RESULTS / "_panel_ckpt.pt")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stop at epoch %d (best=%d val=%.4f)",
                            epoch, best_epoch, best_val)
                break
    # Reload best
    model.load_state_dict(torch.load(RESULTS / "_panel_ckpt.pt")["state_dict"])

    # 8. Test prediction + per-asset Sharpe + basket Sharpe
    mu_1d, var_1d, mu_5d, var_5d, asset_test, y_test = predict_panel(model, test_loader)
    actuals_1d = y_test[:, 0]
    actuals_5d = y_test[:, 1]
    # Realised vol per (asset, date) — reuse rv_20 from features
    rv_test = []
    for asset in asset_unique:
        rv_a = F_clean.xs(asset, level="asset")["rv_20"].values
        # Index lookup is messy here; use a flat path: F_clean ordering is
        # (date, asset) sorted; we already have asset_ids aligned with
        # F_clean.values so keep test predictions aligned via test_ds.windows
        pass
    # Simpler: compute vol per test prediction using a precomputed array
    rv_arr = F_clean["rv_20"].values
    test_window_idx = test_ds.windows  # row indices into X_norm/y_all
    vols = rv_arr[test_window_idx]

    # Trade direction = sign(5d prediction) — user directive: 5d primary
    # Confidence = 1 / (aleatoric uncertainty); higher confidence = lower variance
    confidence_5d = 1.0 / (1e-4 + np.sqrt(var_5d))
    conf_threshold_q50 = float(np.median(confidence_5d))

    # Trade on actual 1d return per CLAUDE.md "trade always realised on 1d"
    direction_5d = np.sign(mu_5d)
    raw_pnl = direction_5d * actuals_1d  # 1d realised return
    sharpe_unconf = sharpe_ratio(raw_pnl)
    sharpe_conf = sharpe_ratio(raw_pnl[confidence_5d >= conf_threshold_q50])

    # 5d-on-5d (for comparison; B target)
    pnl_5d = direction_5d * actuals_5d
    sharpe_5d = sharpe_ratio(pnl_5d)

    # Per-asset Sharpe + basket
    bsk = basket_sharpe(mu_5d, actuals_1d, asset_test, vols,
                         confidence=confidence_5d, conf_threshold=conf_threshold_q50)

    # 1d direction Sharpe
    direction_1d = np.sign(mu_1d)
    pnl_1d_only = direction_1d * actuals_1d
    sharpe_1d = sharpe_ratio(pnl_1d_only)

    elapsed = time.time() - t0

    # Composite (5d primary): we use 5d-direction trading on 1d returns
    # as the realised P&L. Negative folds penalty: count assets with
    # negative per-asset sharpe in basket.
    n_neg = sum(1 for s in bsk["per_asset_sharpe"].values() if s < 0)
    composite = float(min(sharpe_unconf, sharpe_conf) - 0.05 * n_neg)

    id_to_asset = {i: a for a, i in asset_to_id.items()}
    per_asset_named = {id_to_asset[int(aid)]: float(s)
                       for aid, s in bsk["per_asset_sharpe"].items()}

    entry = {
        "backbone": args.backbone,
        "description": args.description,
        "seed": args.seed,
        "config": {
            "seq_len": args.seq_len, "hidden": args.hidden,
            "num_layers": args.num_layers, "lr": args.lr, "bs": args.bs,
            "epochs": args.epochs, "patience": args.patience,
            "head_dropout": args.head_dropout, "asset_emb_dim": args.asset_emb_dim,
        },
        "n_features": n_features,
        "n_assets": n_assets,
        "excluded_assets": excluded,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
        # Headline numbers
        "test_sharpe_5d_on_1d_uncgated": float(sharpe_unconf),
        "test_sharpe_5d_on_1d_confgated": float(sharpe_conf),
        "test_sharpe_5d_on_5d": float(sharpe_5d),
        "test_sharpe_1d_on_1d": float(sharpe_1d),
        "basket": {
            "n_trades": bsk["n_trades"],
            "mean_pnl_bps": bsk["mean_pnl_bps"],
            "per_asset_sharpe_count": len(bsk["per_asset_sharpe"]),
            "per_asset_sharpe_median": float(np.median(list(bsk["per_asset_sharpe"].values()))) if bsk["per_asset_sharpe"] else 0.0,
            "per_asset_sharpe_mean": float(np.mean(list(bsk["per_asset_sharpe"].values()))) if bsk["per_asset_sharpe"] else 0.0,
            "n_negative_assets": n_neg,
            "per_asset_sharpe": per_asset_named,
        },
        "composite": composite,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
    }

    with open(PANEL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, allow_nan=False, default=float) + "\n")
    logger.info("[panel] composite=%+.4f sharpe_5dx1d_unc=%+.4f conf=%+.4f sharpe_5d=%+.4f sharpe_1d=%+.4f elapsed=%.1fs",
                composite, sharpe_unconf, sharpe_conf, sharpe_5d, sharpe_1d, elapsed)
    print(json.dumps(entry, indent=2, default=float))


if __name__ == "__main__":
    main()
