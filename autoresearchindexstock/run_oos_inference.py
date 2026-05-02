"""Out-of-sample (OOS) inference on the global QQQ champion.

User directive 2026-05-02: download data Nov 2025 - Apr 2026, run pure
inference (no training, no peeking) on the current best_model.pt, write
a CSV of predictions vs actuals.

Loads `autoresearch_results/best_model.pt` (currently exp 276 dmamba
bs=16 single-seed=42 +1.5094, the composite-leader). Note the
multi-seed lucky-basin caveat (3-seed median +0.14) — this script runs
inference on whatever checkpoint best_model.pt points to.

Usage:
    python -m autoresearchindexstock.run_oos_inference \
        --start 2025-11-01 --end 2026-04-30 \
        --out autoresearch_results/oos_predictions_nov25_apr26.csv
"""
from __future__ import annotations

# pin CPU first, before torch import — same as run_autoresearch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.run_autoresearch import _pin_to_safe_cores  # noqa
_pin_to_safe_cores()

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from autoresearchindexstock.data.features import compute_qqq_features, compute_qqq_targets
from autoresearchindexstock.data.download import ALL_SIGNALS
from autoresearchindexstock.data.splits import (
    FOLDS, get_fold_dates, LABEL_HORIZON_BUFFER, EMBARGO_DAYS, PURGE_DAYS,
)
from autoresearch.model.backbone import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos")


def _download_no_cap(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a ticker WITHOUT the 2026-cap. For OOS inference only."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # features.py expects capitalized OHLCV column names
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    return df


def download_for_oos(start: str, end: str) -> dict:
    """Download every signal in ALL_SIGNALS over [start, end] without the
    2025-12-31 cap. Bypasses cache to ensure fresh post-Dec-2025 data."""
    out = {}
    total = sum(len(g) for g in ALL_SIGNALS.values())
    fetched = 0
    for group_name, group in ALL_SIGNALS.items():
        for ticker in group:
            try:
                df = _download_no_cap(ticker, start, end)
            except Exception as e:
                log.warning("Skip %s (%s): %s", ticker, group_name, e)
                continue
            if df is None or df.empty:
                log.warning("Empty %s (%s)", ticker, group_name)
                continue
            out[ticker] = df
            fetched += 1
    log.info("[download oos] %d / %d tickers fetched (%s -> %s, NO 2026 cap)",
             fetched, total, start, end)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="autoresearch_results/best_model.pt",
                        help="Path to model checkpoint (default: best_model.pt)")
    parser.add_argument("--start", default="2004-01-01",
                        help="Download start. Default 2004-01-01 because compute_qqq_features drops any column whose first_valid_index > 2007-01-01 (sector-ETFs etc) — must download full training history so filter retains all features. OOS dates are sliced from this on the predict side.")
    parser.add_argument("--inference-start", default="2007-01-01",
                        help="First date to consider for inference. Default 2007 = right after warmup; we punch holes in train/val/embargo windows so we only emit predictions on (a) the 7 historical TEST fold windows the model never saw during training, and (b) every date after the last fold's test_end (true OOS).")
    parser.add_argument("--end", default="2026-04-30",
                        help="Download + inference end date")
    parser.add_argument("--out", default="autoresearch_results/oos_predictions_holepunch_2007_apr26.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # 1) Load checkpoint — get the exact config + scaler + feature_columns
    log.info("[load] %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    config = ckpt["config"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    feature_columns = ckpt["feature_columns"]
    backbone = ckpt["backbone"]
    n_features = ckpt["n_features"]
    composite = ckpt.get("composite", "?")
    exp_num = ckpt.get("experiment_num", "?")
    description = ckpt.get("description", "")
    log.info("[ckpt] exp%s backbone=%s composite=%s n_features=%d", exp_num, backbone, composite, n_features)
    log.info("[ckpt] desc: %s", description[:80])

    # 2) Download fresh OOS data WITHOUT the 2025-12-31 cap
    log.info("[download] %s -> %s (NO 2026 cap, bypass cache for fresh data)", args.start, args.end)
    raw = download_for_oos(args.start, args.end)

    # 3) Compute features + targets using the same pipeline as training
    log.info("[features] computing 104 features")
    feats_full = compute_qqq_features(raw)
    targets_full = compute_qqq_targets(raw)
    log.info("[features] feats shape=%s targets shape=%s", feats_full.shape, targets_full.shape)

    # 4) Align feature columns with training (best_model has feature_columns saved)
    missing = set(feature_columns) - set(feats_full.columns)
    extra = set(feats_full.columns) - set(feature_columns)
    if missing:
        log.warning("[features] %d cols missing in OOS data: %s",
                    len(missing), sorted(missing)[:5])
    if extra:
        log.warning("[features] %d cols extra in OOS (will drop): %s",
                    len(extra), sorted(extra)[:5])
    feats = feats_full.reindex(columns=feature_columns).dropna(how="any")
    log.info("[features] aligned shape=%s; dates %s -> %s",
             feats.shape, feats.index.min(), feats.index.max())

    # 5) Scale features using TRAINING scaler (no peeking)
    arr = (feats.values - scaler_mean) / scaler_scale
    log.info("[scale] applied training scaler (mean shape=%s, scale shape=%s)",
             scaler_mean.shape, scaler_scale.shape)

    # 6) Build sliding windows and predict
    seq_len = config.get("seq_len", 60)
    log.info("[infer] seq_len=%d, building sliding windows", seq_len)

    # Filter to the inference window: predict-day must be >= inference-start
    inf_start = pd.Timestamp(args.inference_start)
    end_date = pd.Timestamp(args.end)

    # Assemble model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[device] %s", device)
    # The training-time mamba HPs (variant/d_state/expand) are not stored in
    # the saved config dict — infer them from state_dict shapes.
    sd = ckpt["model_state_dict"]
    mamba_d_state = None
    mamba_variant = None
    mamba_expand = None
    if backbone == "mamba":
        if "blocks.0.A_log" in sd:
            mamba_d_state = sd["blocks.0.A_log"].shape[1]  # (d_inner, d_state)
        if "trend_mlp.0.weight" in sd:
            mamba_variant = "dmamba"  # decomposition head present
        # expand = d_inner / d_model; d_inner = blocks.0.A_log.shape[0]
        # d_model is harder to recover; infer from in_proj if present, else default
        if "blocks.0.in_proj.weight" in sd:
            d_inner_x2 = sd["blocks.0.in_proj.weight"].shape[0]
            d_model = sd["blocks.0.in_proj.weight"].shape[1]
            d_inner = d_inner_x2 // 2
            mamba_expand = d_inner // d_model if d_model > 0 else 2
        else:
            mamba_expand = 2  # default
        log.info("[mamba] inferred from state_dict: variant=%s d_state=%s expand=%s",
                 mamba_variant, mamba_d_state, mamba_expand)

    model = create_model(
        backbone=backbone,
        n_input_features=n_features,
        seq_len=seq_len,
        head_dropout=config.get("head_dropout", 0.1),
        het_loss=True,
        mamba_variant=mamba_variant,
        mamba_d_state=mamba_d_state,
        mamba_expand=mamba_expand,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # 7) PUNCH HOLES: build the mask of "predictable" dates by EXCLUDING all
    # train/val + embargo + label-horizon buffer ranges per fold. The dates
    # that REMAIN are exactly: (a) the 7 test fold windows + (b) anything
    # after the last fold's test_end (true OOS = Nov 2025 onward).
    held_out_ranges = []  # list of (lo_date, hi_date, kind)
    train_blocked_ranges = []
    test_keep_ranges = []
    last_test_end = None
    for f in FOLDS:
        d = get_fold_dates(f)
        # train ranges per fold = train_start..train_end (we must NOT predict here)
        train_blocked_ranges.append((d["train_start"], d["train_end"]))
        # val window also held out (model used it for early-stop/composite)
        held_out_ranges.append((d["val_start"], d["val_end"], "val"))
        # test window — KEEP for prediction (and also exclude from train block)
        test_keep_ranges.append((d["test_start"], d["test_end"], "test"))
        # embargo: PURGE_DAYS before val and after test
        held_out_ranges.append(
            (d["val_start"] - pd.Timedelta(days=LABEL_HORIZON_BUFFER + EMBARGO_DAYS),
             d["val_start"] - pd.Timedelta(days=1), "embargo_pre_val"))
        held_out_ranges.append(
            (d["test_end"] + pd.Timedelta(days=1),
             d["test_end"] + pd.Timedelta(days=PURGE_DAYS), "embargo_post_test"))
        if last_test_end is None or d["test_end"] > last_test_end:
            last_test_end = d["test_end"]

    log.info("[hole-punch] %d FOLDS, last test_end=%s, post-fold OOS = (%s, %s]",
             len(FOLDS), last_test_end.date(), last_test_end.date(), end_date.date())

    def in_test_window(dt):
        for lo, hi, _ in test_keep_ranges:
            if lo <= dt <= hi:
                return True
        return False

    def in_held_out(dt):
        for lo, hi, _ in held_out_ranges:
            if lo <= dt <= hi:
                return True
        return False

    def in_train_block(dt):
        for lo, hi in train_blocked_ranges:
            if lo <= dt <= hi:
                return True
        return False

    def is_post_fold_oos(dt):
        return dt > last_test_end and dt <= end_date

    def keep_predict(dt):
        # Keep if either inside a test fold window OR post-last-test OOS.
        # Skip if in val/embargo (held_out) explicitly OR train-only block AND not test fold.
        if dt < inf_start or dt > end_date:
            return False, ""
        if in_held_out(dt):
            return False, "held_out_val_or_embargo"
        if in_test_window(dt):
            return True, "test_fold"
        if is_post_fold_oos(dt):
            return True, "post_fold_oos"
        # Otherwise it's a training date or pre-fold-1 date — skip
        return False, "train_or_pre_fold"

    rows = []
    dates_arr = feats.index
    n = len(arr)
    log.info("[infer] %d days in feature window; running hole-punched inference", n)

    n_kept_test = n_kept_oos = n_skipped = 0
    with torch.no_grad():
        for i in range(seq_len - 1, n):
            predict_date = dates_arr[i]
            keep, reason = keep_predict(predict_date)
            if not keep:
                n_skipped += 1
                continue
            if reason == "test_fold":
                n_kept_test += 1
            else:
                n_kept_oos += 1
            window = arr[i - seq_len + 1 : i + 1]  # shape (seq_len, n_features)
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)  # (1, seq_len, n_features)
            out = model(x)
            mu_a = float(out["ret_1d"][:, 0].cpu().numpy()[0])
            sigma_a = float(out["ret_1d"][:, 1].cpu().numpy()[0]) if out["ret_1d"].shape[1] > 1 else float("nan")
            mu_b = float(out["ret_5d"][:, 0].cpu().numpy()[0]) if "ret_5d" in out else float("nan")
            sigma_b = float(out["ret_5d"][:, 1].cpu().numpy()[0]) if "ret_5d" in out and out["ret_5d"].shape[1] > 1 else float("nan")

            # Pull actual returns from targets_full if available (for backtest only — NOT used in prediction)
            actual_1d = float(targets_full["fwd_ret_1d"].get(predict_date, float("nan"))) if "fwd_ret_1d" in targets_full.columns else float("nan")
            actual_5d = float(targets_full["fwd_ret_5d"].get(predict_date, float("nan"))) if "fwd_ret_5d" in targets_full.columns else float("nan")

            direction = int(np.sign(mu_a)) if mu_a != 0 else 0
            strategy_pnl_1d = direction * actual_1d if not np.isnan(actual_1d) else float("nan")
            confidence = float(np.exp(-abs(sigma_a))) if not np.isnan(sigma_a) else float("nan")  # rough proxy

            # Tag fold membership for analysis
            fold_tag = "post_fold_oos"
            for f in FOLDS:
                d = get_fold_dates(f)
                if d["test_start"] <= predict_date <= d["test_end"]:
                    fold_tag = f["name"]
                    break

            rows.append({
                "date": predict_date.strftime("%Y-%m-%d"),
                "fold": fold_tag,
                "pred_ret_1d": mu_a,
                "pred_ret_5d": mu_b,
                "logvar_1d": sigma_a,
                "logvar_5d": sigma_b,
                "confidence": confidence,
                "pred_direction": direction,
                "actual_ret_1d": actual_1d,
                "actual_ret_5d": actual_5d,
                "actual_direction_1d": int(np.sign(actual_1d)) if not np.isnan(actual_1d) else 0,
                "correct_1d": int(direction == int(np.sign(actual_1d))) if not np.isnan(actual_1d) and direction != 0 else 0,
                "strategy_pnl_1d": strategy_pnl_1d,
            })

    log.info("[infer] %d predictions generated  test=%d  post-fold-oos=%d  skipped=%d",
             len(rows), n_kept_test, n_kept_oos, n_skipped)
    if not rows:
        log.error("No predictions generated! Check date ranges + feature alignment.")
        return

    df = pd.DataFrame(rows)
    df["cumulative_pnl"] = df["strategy_pnl_1d"].fillna(0).cumsum()
    df["cumulative_buy_hold"] = df["actual_ret_1d"].fillna(0).cumsum()

    # Save CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    log.info("[out] wrote %d rows to %s", len(df), out_path)

    # Per-fold + post-OOS summary
    valid = df.dropna(subset=["actual_ret_1d"])
    log.info("[summary] %d total predictions, %d with actuals", len(df), len(valid))
    log.info("[summary] %s | %-8s | %-5s | %-7s | %-9s | %-9s | %-9s",
             "fold", "n", "hit%", "ret%", "bh%", "Shp", "bh_Shp")
    for fold_name in valid["fold"].unique():
        f = valid[valid["fold"] == fold_name]
        if len(f) == 0:
            continue
        hit = (f["correct_1d"] == 1).mean() * 100
        strat_ret = f["strategy_pnl_1d"].sum() * 100
        bh_ret = f["actual_ret_1d"].sum() * 100
        sh = (f["strategy_pnl_1d"].mean() / f["strategy_pnl_1d"].std() * np.sqrt(252)) if f["strategy_pnl_1d"].std() > 0 else 0
        bsh = (f["actual_ret_1d"].mean() / f["actual_ret_1d"].std() * np.sqrt(252)) if f["actual_ret_1d"].std() > 0 else 0
        log.info("[summary] %-25s | %4d | %5.2f | %+7.2f | %+8.2f | %+8.4f | %+8.4f",
                 fold_name[:25], len(f), hit, strat_ret, bh_ret, sh, bsh)

    if len(valid) > 0:
        hit_rate = (valid["correct_1d"] == 1).mean() * 100
        strat_total = valid["strategy_pnl_1d"].sum()
        bh_total = valid["actual_ret_1d"].sum()
        log.info("[summary OVERALL] %d days  hit=%.2f%%  strat_total=%.4f  bh_total=%.4f  excess=%+.4f",
                 len(valid), hit_rate, strat_total, bh_total, strat_total - bh_total)
        if valid["strategy_pnl_1d"].std() > 0:
            sharpe = (valid["strategy_pnl_1d"].mean() / valid["strategy_pnl_1d"].std()) * np.sqrt(252)
            log.info("[summary OVERALL] strategy annualised Sharpe = %+.4f", sharpe)
        if valid["actual_ret_1d"].std() > 0:
            bh_sharpe = (valid["actual_ret_1d"].mean() / valid["actual_ret_1d"].std()) * np.sqrt(252)
            log.info("[summary OVERALL] buy-and-hold Sharpe = %+.4f", bh_sharpe)


if __name__ == "__main__":
    main()
