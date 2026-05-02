"""Run OOS inference for the top-30 experiments by composite. For each
experiment with an archived checkpoint, run pure inference Dec 2025-Apr
2026 and save (a) per-experiment CSV and (b) a row in the top-30 JSON.

For experiments WITHOUT archived checkpoints, the row is filled with
"checkpoint missing — retrain to enable" so the user knows what's gated.

User directive 2026-05-02:
- Run OOS on top 30 winners
- Inference only, no training
- Provide table similar to original dashboard table but only OOS metrics
- Each row gets full metrics + CSV link
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.run_autoresearch import _pin_to_safe_cores
_pin_to_safe_cores()

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from autoresearchindexstock.data.features import compute_qqq_features, compute_qqq_targets
from autoresearchindexstock.data.download import ALL_SIGNALS
from autoresearch.model.backbone import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos30")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "autoresearch_results"
WINNERS = RESULTS / "winners"
JSONL = RESULTS / "experiment_log.jsonl"

OOS_START = "2025-12-01"
OOS_END = "2026-04-30"


def find_checkpoint_for_exp(exp_num: int) -> Path | None:
    """Find a model checkpoint file for the given experiment number.

    Sources (in priority order):
    - winners/*/model_checkpoint.pt where the dir name contains 'exp{N}_'
    - best_model.pt (only matches the very latest experiment, exp 276 currently)
    """
    if not WINNERS.exists():
        return None
    for d in WINNERS.iterdir():
        if d.is_dir() and f"exp{exp_num}_" in d.name:
            for pt in d.glob("model_checkpoint.pt"):
                return pt
    # Best model — check if it's this experiment
    best_pt = RESULTS / "best_model.pt"
    if best_pt.exists():
        try:
            ckpt = torch.load(best_pt, weights_only=False, map_location="cpu")
            if ckpt.get("experiment_num") == exp_num:
                return best_pt
        except Exception:
            pass
    return None


def _download_no_cap(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    return df


def download_for_oos(start="2004-01-01", end=OOS_END) -> dict:
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


def run_oos_for_checkpoint(ckpt_path: Path, raw_data: dict, exp_num: int) -> dict:
    """Run full OOS inference Dec 2025-Apr 2026 for one checkpoint.
    Returns dict with metrics + csv_filename + per-day predictions."""
    log.info("[oos exp%s] loading %s", exp_num, ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    config = ckpt["config"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    feature_columns = ckpt["feature_columns"]
    backbone = ckpt["backbone"]
    n_features = ckpt["n_features"]

    # Feature build
    feats_full = compute_qqq_features(raw_data)
    targets_full = compute_qqq_targets(raw_data)
    feats = feats_full.reindex(columns=feature_columns).dropna(how="any")
    if len(feats) == 0:
        return {"experiment_num": exp_num, "error": "feature_alignment_failed",
                "n_predictions": 0}

    arr = (feats.values - scaler_mean) / scaler_scale
    seq_len = config.get("seq_len", 60)

    # Infer mamba HPs from state_dict
    sd = ckpt["model_state_dict"]
    mamba_d_state = mamba_variant = mamba_expand = None
    if backbone == "mamba":
        if "blocks.0.A_log" in sd:
            mamba_d_state = sd["blocks.0.A_log"].shape[1]
        if "trend_mlp.0.weight" in sd:
            mamba_variant = "dmamba"
        if "blocks.0.in_proj.weight" in sd:
            d_inner = sd["blocks.0.in_proj.weight"].shape[0] // 2
            d_model = sd["blocks.0.in_proj.weight"].shape[1]
            mamba_expand = d_inner // d_model if d_model > 0 else 2
        else:
            mamba_expand = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        backbone=backbone,
        n_input_features=n_features,
        seq_len=seq_len,
        head_dropout=config.get("head_dropout", 0.1),
        het_loss=True,
        mamba_variant=mamba_variant,
        mamba_d_state=mamba_d_state,
        mamba_expand=mamba_expand,
        hidden_size=config.get("hidden_size") if config.get("hidden_size") else None,
        num_layers=config.get("num_layers") if config.get("num_layers") else None,
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        return {"experiment_num": exp_num, "error": f"state_dict_mismatch: {str(e)[:200]}",
                "n_predictions": 0}
    model = model.to(device).eval()

    # Sliding-window prediction
    inf_start = pd.Timestamp(OOS_START)
    inf_end = pd.Timestamp(OOS_END)
    rows = []
    with torch.no_grad():
        for i in range(seq_len - 1, len(arr)):
            predict_date = feats.index[i]
            if predict_date < inf_start or predict_date > inf_end:
                continue
            window = arr[i - seq_len + 1: i + 1]
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)
            out = model(x)
            mu_a = float(out["ret_1d"][:, 0].cpu().numpy()[0])
            sigma_a = float(out["ret_1d"][:, 1].cpu().numpy()[0]) if out["ret_1d"].shape[1] > 1 else float("nan")
            mu_b = float(out["ret_5d"][:, 0].cpu().numpy()[0]) if "ret_5d" in out else float("nan")
            actual_1d = float(targets_full["fwd_ret_1d"].get(predict_date, float("nan"))) if "fwd_ret_1d" in targets_full else float("nan")
            direction = int(np.sign(mu_a)) if mu_a != 0 else 0
            strategy = direction * actual_1d if not np.isnan(actual_1d) else float("nan")
            rows.append({
                "date": predict_date.strftime("%Y-%m-%d"),
                "pred_ret_1d": mu_a, "pred_ret_5d": mu_b, "logvar_1d": sigma_a,
                "pred_direction": direction,
                "actual_ret_1d": actual_1d,
                "actual_direction": int(np.sign(actual_1d)) if not np.isnan(actual_1d) else 0,
                "correct": int(direction == int(np.sign(actual_1d))) if not np.isnan(actual_1d) and direction != 0 else 0,
                "strategy_pnl": strategy,
            })

    df = pd.DataFrame(rows)
    df["cumulative_pnl"] = df["strategy_pnl"].fillna(0).cumsum()
    df["cumulative_buy_hold"] = df["actual_ret_1d"].fillna(0).cumsum()
    csv_name = f"oos_exp{exp_num}.csv"
    csv_path = RESULTS / csv_name
    df.to_csv(csv_path, index=False, float_format="%.6f")

    valid = df.dropna(subset=["actual_ret_1d"])
    metrics = {"experiment_num": exp_num, "n_predictions": len(df), "n_with_actuals": len(valid),
               "csv": csv_name, "checkpoint_source": str(ckpt_path.relative_to(RESULTS))}
    if len(valid) > 0:
        metrics["hit_rate_pct"] = round((valid["correct"] == 1).mean() * 100, 2)
        metrics["strategy_total_return_pct"] = round(valid["strategy_pnl"].sum() * 100, 4)
        metrics["buy_hold_total_return_pct"] = round(valid["actual_ret_1d"].sum() * 100, 4)
        metrics["excess_return_pct"] = round((valid["strategy_pnl"].sum() - valid["actual_ret_1d"].sum()) * 100, 4)
        if valid["strategy_pnl"].std() > 0:
            metrics["strategy_annual_sharpe"] = round((valid["strategy_pnl"].mean() / valid["strategy_pnl"].std()) * np.sqrt(252), 4)
        if valid["actual_ret_1d"].std() > 0:
            metrics["buy_hold_annual_sharpe"] = round((valid["actual_ret_1d"].mean() / valid["actual_ret_1d"].std()) * np.sqrt(252), 4)
        metrics["excess_sharpe"] = round(metrics.get("strategy_annual_sharpe", 0) - metrics.get("buy_hold_annual_sharpe", 0), 4)
        metrics["max_drawdown_pct"] = round((valid["cumulative_pnl"] - valid["cumulative_pnl"].cummax()).min() * 100, 4)
        # PSR proxy: 1 - tail risk approximation
        n = len(valid); sr = metrics.get("strategy_annual_sharpe", 0) / np.sqrt(252)  # daily
        if n > 1:
            from scipy.stats import norm
            try:
                # Simple PSR approximation (Bailey-López de Prado 2012)
                skew = valid["strategy_pnl"].skew()
                kurt = valid["strategy_pnl"].kurtosis()
                psr_z = sr * np.sqrt(n - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr * sr) if 1 - skew * sr + (kurt - 1) / 4 * sr * sr > 0 else 0
                metrics["psr"] = round(float(norm.cdf(psr_z)), 4)
            except Exception:
                metrics["psr"] = None
    return metrics


def main():
    # 1) Top 30 experiments by composite
    with open(JSONL, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    top30 = sorted(lines, key=lambda e: -e.get("composite", -99))[:30]

    # 2) Download data once
    raw = download_for_oos()

    # 3) For each top-30 row, find checkpoint & run inference
    table = []
    for rank, e in enumerate(top30, 1):
        exp_num = e.get("experiment_num")
        ckpt_path = find_checkpoint_for_exp(exp_num)
        row = {
            "rank": rank,
            "experiment_num": exp_num,
            "backbone": e.get("backbone"),
            "seed": e.get("seed", e.get("config", {}).get("seed")),
            "description": e.get("description", "")[:80],
            "train_composite": round(e.get("composite", 0), 4),
            "train_test_pos_folds": e.get("test_pos_folds"),
            "train_val_sharpe": round(e.get("val_sharpe", 0), 4) if e.get("val_sharpe") is not None else None,
            "train_return_pct": round(e.get("return_pct", 0), 4) if e.get("return_pct") is not None else None,
            "checkpoint_status": "available" if ckpt_path else "missing",
        }
        if ckpt_path is None:
            row["oos_status"] = "skipped — checkpoint not archived (would require retrain)"
        else:
            log.info("[rank %d] exp%s — running OOS", rank, exp_num)
            try:
                oos = run_oos_for_checkpoint(ckpt_path, raw, exp_num)
                row.update({f"oos_{k}": v for k, v in oos.items() if k != "experiment_num"})
                row["oos_status"] = "completed" if oos.get("n_with_actuals", 0) > 0 else f"failed: {oos.get('error', 'unknown')}"
            except Exception as ex:
                row["oos_status"] = f"error: {str(ex)[:150]}"
                log.error("[rank %d] exp%s FAILED: %s", rank, exp_num, ex)
        table.append(row)

    # 4) Save the top-30 OOS summary
    out = {
        "oos_run_at": pd.Timestamp.utcnow().isoformat(),
        "oos_window": {"start": OOS_START, "end": OOS_END},
        "n_total_top30": len(table),
        "n_with_checkpoint": sum(1 for r in table if r["checkpoint_status"] == "available"),
        "n_completed": sum(1 for r in table if r.get("oos_status") == "completed"),
        "table": table,
    }
    out_path = RESULTS / "oos_top30_table.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("[done] wrote %s — %d completed / %d top30",
             out_path.name, out["n_completed"], out["n_total_top30"])


if __name__ == "__main__":
    main()
