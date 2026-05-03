"""Retrain top-30 winners under production split (train through 2025-09-30,
OOS test 2025-10-01 → 2026-04-30) and run OOS inference on each fresh
checkpoint. Per CLAUDE.md Directive 42, every retrain is auto-archived.

User directive 2026-05-02: original checkpoints lost (best_model.pt was
overwritten); model trained with fold-7 train_end at 2023-09-30 is missing
2 years of recent regime. Retrain top-30 with extended training data, then
update OOS dashboard.

Output:
- winners/exp{N}_prod_retrain/model_checkpoint.pt + config.json (per exp)
- oos_exp{N}_prod.csv (per-experiment OOS predictions)
- oos_top30_table.json (overall table; rows updated as retrains land)
- prod_retrain_progress.json (live progress for dashboard polling)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.run_autoresearch import _pin_to_safe_cores
_pin_to_safe_cores()

import json
import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from autoresearchindexstock.data.features import compute_qqq_features, compute_qqq_targets
from autoresearchindexstock.data.download import ALL_SIGNALS
from autoresearch.model.backbone import create_model, get_seq_len
from autoresearch.model.train import find_contiguous_segments, create_contiguous_datasets, train_one_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prod_retrain")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "autoresearch_results"
WINNERS = RESULTS / "winners"
JSONL = RESULTS / "experiment_log.jsonl"
PROGRESS_PATH = RESULTS / "prod_retrain_progress.json"
TABLE_PATH = RESULTS / "oos_top30_table.json"

# Production split — train uses ALL data 2004 → train_end; tiny val tail; OOS test = the new window
TRAIN_END = pd.Timestamp("2025-09-30")
VAL_END = pd.Timestamp("2025-09-30")
VAL_START = pd.Timestamp("2025-07-01")  # last 90 days as val for early-stop
OOS_START = pd.Timestamp("2025-10-01")
OOS_END = pd.Timestamp("2026-04-30")


def _download_no_cap(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    return df


def download_data(start="2004-01-01", end="2026-04-30") -> dict:
    out = {}
    for group in ALL_SIGNALS.values():
        for ticker in group:
            try:
                df = _download_no_cap(ticker, start, end)
            except Exception as e:
                log.warning("Skip %s: %s", ticker, e); continue
            if df is None or df.empty:
                continue
            out[ticker] = df
    log.info("[download] %d tickers fetched", len(out))
    return out


def split_prod(features: pd.DataFrame, targets: pd.DataFrame):
    """Production split (operates on the intersection of features.index and targets.index
    so date-based masks align correctly even though raw frames have different lengths)."""
    common = features.index.intersection(targets.index)
    feats = features.loc[common]
    tars = targets.loc[common]
    tr_mask = feats.index < VAL_START
    vl_mask = (feats.index >= VAL_START) & (feats.index <= VAL_END)
    te_mask = (feats.index >= OOS_START) & (feats.index <= OOS_END)
    return (feats.loc[tr_mask], feats.loc[vl_mask], feats.loc[te_mask],
            tars.loc[tr_mask], tars.loc[vl_mask], tars.loc[te_mask])


def update_progress(state: dict):
    PROGRESS_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def retrain_one(config_row: dict, raw_data: dict, feats_full: pd.DataFrame, targets_full: pd.DataFrame) -> dict:
    """Retrain one experiment under production split, run OOS, return metrics."""
    exp_num = config_row["experiment_num"]
    backbone = config_row["backbone"]
    cfg = config_row["config"]
    seed = cfg.get("seed", config_row.get("seed", 42))
    seq_len = cfg.get("seq_len") or get_seq_len(backbone)
    log.info("[exp%s] backbone=%s seq=%d seed=%s", exp_num, backbone, seq_len, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split
    tr_f, vl_f, te_f, tr_t, vl_t, te_t = split_prod(feats_full, targets_full)
    log.info("[exp%s] split: train=%d val=%d oos=%d", exp_num, len(tr_f), len(vl_f), len(te_f))
    if len(tr_f) < seq_len + 10 or len(te_f) < seq_len + 1:
        return {"error": "insufficient data after split"}

    # Scale
    scaler = StandardScaler().fit(tr_f.values)
    n_features = tr_f.shape[1]

    # Mamba HPs from history (some are saved separately in the JSONL fields, not in config)
    mamba_variant = cfg.get("mamba_variant") or config_row.get("mamba_variant")
    mamba_d_state = cfg.get("d_state") or cfg.get("mamba_d_state")
    mamba_expand = cfg.get("expand") or cfg.get("mamba_expand")
    # Heuristic from description if missing
    desc = config_row.get("description", "") or ""
    if backbone == "mamba" and mamba_variant is None:
        if "dmamba" in desc.lower(): mamba_variant = "dmamba"
        elif "mambats" in desc.lower(): mamba_variant = "mambats"
        elif "mambastock" in desc.lower(): mamba_variant = "mambastock"
        elif "samba" in desc.lower(): mamba_variant = "samba"
        elif "hybrid_mamba" in desc.lower(): mamba_variant = "hybrid_mamba"
        elif "crossmamba" in desc.lower(): mamba_variant = "crossmamba"
        else: mamba_variant = "vanilla"
    if backbone == "mamba" and mamba_d_state is None:
        if "d_state=32" in desc: mamba_d_state = 32
        elif "d_state=64" in desc: mamba_d_state = 64
        else: mamba_d_state = 16
    if backbone == "mamba" and mamba_expand is None:
        if "expand=1" in desc: mamba_expand = 1
        elif "expand=4" in desc: mamba_expand = 4
        else: mamba_expand = 2

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        backbone=backbone,
        n_input_features=n_features,
        seq_len=seq_len,
        head_dropout=cfg.get("head_dropout", 0.1),
        het_loss=True,
        hidden_size=cfg.get("hidden_size") if cfg.get("hidden_size") else None,
        num_layers=cfg.get("num_layers") if cfg.get("num_layers") else None,
        mamba_variant=mamba_variant,
        mamba_d_state=mamba_d_state,
        mamba_expand=mamba_expand,
    )
    if hasattr(model, "to"):
        model = model.to(device)

    # Build datasets via the shared training helper (handles segmentation + scaling)
    ts = scaler.transform(tr_f.values)
    vs = scaler.transform(vl_f.values)
    train_df_scaled = pd.DataFrame(ts, index=tr_f.index, columns=tr_f.columns)
    val_df_scaled = pd.DataFrame(vs, index=vl_f.index, columns=vl_f.columns)
    train_ds = create_contiguous_datasets(train_df_scaled, tr_t, seq_len)
    val_ds = create_contiguous_datasets(val_df_scaled, vl_t, seq_len)
    if len(train_ds) == 0:
        return {"error": "empty train dataset"}

    # Train
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False, num_workers=0) if len(val_ds) > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                                   weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("epochs", 50)
    patience = cfg.get("patience", 10)
    warmup_epochs = cfg.get("warmup_epochs", 0)

    best_val_loss = float("inf"); patience_ctr = 0
    for epoch in range(1, epochs + 1):
        # Warmup LR
        if warmup_epochs and epoch <= warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = cfg.get("lr", 3e-4) * epoch / warmup_epochs
        model.train()
        train_loss = 0.0; n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            mu_a = pred["ret_1d"][:, 0]
            lv_a = pred["ret_1d"][:, 1] if pred["ret_1d"].shape[1] > 1 else torch.zeros_like(mu_a)
            mu_b = pred["ret_5d"][:, 0] if "ret_5d" in pred else mu_a
            lv_b = pred["ret_5d"][:, 1] if "ret_5d" in pred and pred["ret_5d"].shape[1] > 1 else torch.zeros_like(mu_b)
            loss_a = (torch.exp(-lv_a.clamp(-8, 2)) * (mu_a - y[:, 0]).pow(2) + 0.5 * lv_a.clamp(-8, 2)).mean()
            loss_b = (torch.exp(-lv_b.clamp(-8, 2)) * (mu_b - y[:, 1]).pow(2) + 0.5 * lv_b.clamp(-8, 2)).mean() if y.shape[1] > 1 else loss_a * 0
            loss = loss_a + 0.5 * loss_b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()
            train_loss += loss.item() * x.size(0); n += x.size(0)
        train_loss /= max(n, 1)

        if val_loader is not None:
            model.eval()
            v_loss = 0.0; v_n = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    mu_a = pred["ret_1d"][:, 0]
                    lv_a = pred["ret_1d"][:, 1] if pred["ret_1d"].shape[1] > 1 else torch.zeros_like(mu_a)
                    loss_a = (torch.exp(-lv_a.clamp(-8, 2)) * (mu_a - y[:, 0]).pow(2) + 0.5 * lv_a.clamp(-8, 2)).mean()
                    v_loss += loss_a.item() * x.size(0); v_n += x.size(0)
            v_loss /= max(v_n, 1)
            if v_loss < best_val_loss - 1e-4:
                best_val_loss = v_loss
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs:
            log.info("[exp%s] ep%d train=%.4f val=%.4f patience=%d/%d", exp_num, epoch, train_loss,
                     v_loss if val_loader else 0, patience_ctr, patience)

    # Load best
    if "best_state" in dir():
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_dir = WINNERS / f"exp{exp_num}_prod_retrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_columns": list(tr_f.columns),
        "target_columns": ["fwd_ret_1d", "fwd_ret_5d"],
        "n_features": n_features,
        "backbone": backbone,
        "experiment_num": exp_num,
        "description": f"PROD-RETRAIN exp{exp_num}: " + (config_row.get("description", "")[:80]),
        "prod_train_end": str(TRAIN_END.date()),
        "prod_oos_window": (str(OOS_START.date()), str(OOS_END.date())),
    }
    torch.save(ckpt, ckpt_dir / "model_checkpoint.pt")
    log.info("[exp%s] checkpoint saved -> %s", exp_num, ckpt_dir.relative_to(RESULTS))

    # OOS inference
    arr_te = scaler.transform(te_f.values)
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(seq_len - 1, len(arr_te)):
            predict_date = te_f.index[i]
            window = arr_te[i - seq_len + 1: i + 1]
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)
            out = model(x)
            mu_a = float(out["ret_1d"][:, 0].cpu().numpy()[0])
            actual_1d = float(te_t["fwd_ret_1d"].iloc[i]) if i < len(te_t) and not pd.isna(te_t["fwd_ret_1d"].iloc[i]) else float("nan")
            direction = int(np.sign(mu_a)) if mu_a != 0 else 0
            rows.append({
                "date": predict_date.strftime("%Y-%m-%d"),
                "pred_ret_1d": mu_a,
                "pred_direction": direction,
                "actual_ret_1d": actual_1d,
                "correct": int(direction == int(np.sign(actual_1d))) if not np.isnan(actual_1d) and direction != 0 else 0,
                "strategy_pnl": direction * actual_1d if not np.isnan(actual_1d) else float("nan"),
            })
    df = pd.DataFrame(rows)
    df["cumulative_pnl"] = df["strategy_pnl"].fillna(0).cumsum()
    df["cumulative_buy_hold"] = df["actual_ret_1d"].fillna(0).cumsum()
    df.to_csv(RESULTS / f"oos_exp{exp_num}_prod.csv", index=False, float_format="%.6f")

    valid = df.dropna(subset=["actual_ret_1d"])
    metrics = {"oos_n_predictions": len(df), "oos_n_with_actuals": len(valid)}
    if len(valid) > 0 and valid["strategy_pnl"].std() > 0:
        sh = valid["strategy_pnl"].mean() / valid["strategy_pnl"].std() * np.sqrt(252)
        bh = valid["actual_ret_1d"].mean() / valid["actual_ret_1d"].std() * np.sqrt(252) if valid["actual_ret_1d"].std() > 0 else 0
        metrics.update({
            "oos_strategy_annual_sharpe": round(sh, 4),
            "oos_buy_hold_annual_sharpe": round(bh, 4),
            "oos_excess_sharpe": round(sh - bh, 4),
            "oos_strategy_total_return_pct": round(valid["strategy_pnl"].sum() * 100, 4),
            "oos_buy_hold_total_return_pct": round(valid["actual_ret_1d"].sum() * 100, 4),
            "oos_excess_return_pct": round((valid["strategy_pnl"].sum() - valid["actual_ret_1d"].sum()) * 100, 4),
            "oos_hit_rate_pct": round((valid["correct"] == 1).mean() * 100, 2),
            "oos_max_drawdown_pct": round((valid["cumulative_pnl"] - valid["cumulative_pnl"].cummax()).min() * 100, 4),
            "oos_csv": f"oos_exp{exp_num}_prod.csv",
            "oos_status": "completed",
            "oos_checkpoint_source": str((ckpt_dir / "model_checkpoint.pt").relative_to(RESULTS)),
            "oos_equity_curve": {
                "dates": valid["date"].tolist(),
                "strategy_pct": [round(v * 100, 4) for v in valid["cumulative_pnl"].tolist()],
                "buy_hold_pct": [round(v * 100, 4) for v in valid["cumulative_buy_hold"].tolist()],
            },
        })
    log.info("[exp%s] OOS Sharpe=%.4f BH=%.4f excess=%.4f",
             exp_num, metrics.get("oos_strategy_annual_sharpe", 0),
             metrics.get("oos_buy_hold_annual_sharpe", 0),
             metrics.get("oos_excess_sharpe", 0))
    return metrics


def main():
    # 1) Top 30 by composite
    with open(JSONL, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    top30 = sorted(lines, key=lambda e: -e.get("composite", -99))[:30]

    # 2) Download data once
    log.info("[setup] downloading market data 2004 → 2026-04-30")
    raw = download_data()
    feats_full = compute_qqq_features(raw)
    targets_full = compute_qqq_targets(raw)
    log.info("[setup] features %s targets %s", feats_full.shape, targets_full.shape)

    # 3) Load existing top-30 table to update in-place (preserves original train metrics)
    if TABLE_PATH.exists():
        with open(TABLE_PATH, encoding="utf-8") as f:
            table_data = json.load(f)
    else:
        table_data = {"oos_window": {"start": str(OOS_START.date()), "end": str(OOS_END.date())},
                      "table": [], "n_total_top30": len(top30), "n_completed": 0}

    # 4) Retrain each top-30 in sequence
    state = {
        "started_at": pd.Timestamp.utcnow().isoformat(),
        "n_total": len(top30),
        "n_completed": 0,
        "n_failed": 0,
        "current_exp": None,
        "results_so_far": [],
    }
    update_progress(state)
    table_data["oos_run_at"] = state["started_at"]
    table_data["prod_train_end"] = str(TRAIN_END.date())

    for rank, e in enumerate(top30, 1):
        exp_num = e.get("experiment_num")
        state["current_exp"] = exp_num
        state["current_rank"] = rank
        update_progress(state)
        t0 = time.time()
        try:
            metrics = retrain_one(e, raw, feats_full, targets_full)
            elapsed = time.time() - t0
            metrics["retrain_elapsed_sec"] = round(elapsed, 1)
            # Update the matching row in table_data (or append if missing)
            row_idx = next((i for i, r in enumerate(table_data["table"]) if r.get("experiment_num") == exp_num), None)
            if row_idx is None:
                base_row = {
                    "rank": rank, "experiment_num": exp_num, "backbone": e.get("backbone"),
                    "seed": e.get("seed", e.get("config", {}).get("seed")),
                    "description": e.get("description", "")[:80],
                    "train_composite": round(e.get("composite", 0), 4),
                    "train_test_pos_folds": e.get("test_pos_folds"),
                    "train_val_sharpe": round(e.get("val_sharpe", 0), 4) if e.get("val_sharpe") is not None else None,
                    "checkpoint_status": "available",
                }
                base_row.update(metrics)
                table_data["table"].append(base_row)
            else:
                table_data["table"][row_idx].update(metrics)
                table_data["table"][row_idx]["checkpoint_status"] = "available"
            state["n_completed"] += 1
            state["results_so_far"].append({
                "exp": exp_num, "rank": rank,
                "oos_sharpe": metrics.get("oos_strategy_annual_sharpe"),
                "elapsed_sec": elapsed,
            })
            log.info("[rank %d/%d] exp%s completed in %.0fs", rank, len(top30), exp_num, elapsed)
        except Exception as ex:
            state["n_failed"] += 1
            log.error("[rank %d/%d] exp%s FAILED: %s", rank, len(top30), exp_num, ex)
            import traceback; traceback.print_exc()

        # Live update: rewrite table + progress after each retrain
        table_data["n_completed"] = state["n_completed"]
        TABLE_PATH.write_text(json.dumps(table_data, indent=2, default=str), encoding="utf-8")
        update_progress(state)

    state["finished_at"] = pd.Timestamp.utcnow().isoformat()
    state["current_exp"] = None
    update_progress(state)
    log.info("[done] %d/%d retrains completed, %d failed", state["n_completed"], state["n_total"], state["n_failed"])


if __name__ == "__main__":
    main()
