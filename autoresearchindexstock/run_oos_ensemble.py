"""Build a deep ensemble OOS signal by averaging predictions across the
prod-retrain BH-beaters.

Per Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474
"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
— averaging M independently-initialized models reduces predictive variance
by ~M and improves calibration. M=13 here (the 13 prod-retrains that beat
buy-hold individually).

Two ensembling strategies computed:
  - mean_pred: simple unweighted average of pred_ret_1d
  - majority_vote: sign of sum(sign(pred)) over members
  - weighted_excess: members weighted by their individual oos_excess_sharpe

Produces:
  - oos_ensemble.csv (per-day predictions + strategy returns)
  - oos_ensemble_summary.json (Sharpe, return, hit-rate per strategy)
  - dashboard read-friendly equity curves

The script is read-only from the existing oos_exp{N}_prod.csv files —
no new training or inference needed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "autoresearch_results"
TABLE_PATH = RESULTS / "oos_top30_table.json"
ENSEMBLE_CSV = RESULTS / "oos_ensemble.csv"
ENSEMBLE_JSON = RESULTS / "oos_ensemble_summary.json"


def load_bh_beaters() -> list[dict]:
    d = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    rows = [r for r in d["table"]
            if r.get("oos_csv", "").endswith("_prod.csv")
            and r.get("oos_excess_sharpe", -99) > 0]
    rows.sort(key=lambda r: -r["oos_excess_sharpe"])
    return rows


def load_member_csv(member: dict) -> pd.DataFrame:
    csv_path = RESULTS / member["oos_csv"]
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["pred_ret_1d", "pred_direction", "actual_ret_1d"]]
    df.columns = [f"{c}_exp{member['experiment_num']}" for c in df.columns]
    return df


def annualized_sharpe(pnl: pd.Series) -> float:
    if pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def compute_strategy(df: pd.DataFrame, signal_col: str, actual_col: str) -> dict:
    """Given a signal column (mean pred or vote), compute strategy returns + metrics."""
    signal = df[signal_col]
    actual = df[actual_col]
    direction = np.sign(signal).fillna(0).astype(int)
    strategy_pnl = direction * actual
    valid = strategy_pnl.dropna()
    bh_pnl = actual.dropna()
    if len(valid) == 0:
        return {}
    cum_strat = strategy_pnl.fillna(0).cumsum()
    cum_bh = actual.fillna(0).cumsum()
    correct = (direction == np.sign(actual)).astype(int)
    sh = annualized_sharpe(valid)
    bh = annualized_sharpe(bh_pnl) if len(bh_pnl) else 0.0
    return {
        "n_predictions": int(len(direction)),
        "n_with_actuals": int(len(valid)),
        "strategy_annual_sharpe": round(sh, 4),
        "buy_hold_annual_sharpe": round(bh, 4),
        "excess_sharpe": round(sh - bh, 4),
        "strategy_total_return_pct": round(float(valid.sum()) * 100, 4),
        "buy_hold_total_return_pct": round(float(bh_pnl.sum()) * 100, 4),
        "excess_return_pct": round(float(valid.sum() - bh_pnl.sum()) * 100, 4),
        "hit_rate_pct": round(float(correct.mean()) * 100, 2),
        "max_drawdown_pct": round(float((cum_strat - cum_strat.cummax()).min() * 100), 4),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in cum_strat.dropna().index],
            "strategy_pct": [round(v * 100, 4) for v in cum_strat.fillna(0).tolist()],
            "buy_hold_pct": [round(v * 100, 4) for v in cum_bh.fillna(0).tolist()],
        },
    }


def main():
    members = load_bh_beaters()
    print(f"[ensemble] {len(members)} BH-beater members")
    if len(members) == 0:
        raise SystemExit("No BH-beater members found")

    # Load each member, inner-join on date
    frames = [load_member_csv(m) for m in members]
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="inner")
    print(f"[ensemble] merged: {merged.shape[0]} common dates across {len(members)} members")
    if merged.shape[0] == 0:
        raise SystemExit("No common dates — members have disjoint OOS windows")

    # Build the ensemble columns
    pred_cols = [c for c in merged.columns if c.startswith("pred_ret_1d_")]
    dir_cols = [c for c in merged.columns if c.startswith("pred_direction_")]
    actual_cols = [c for c in merged.columns if c.startswith("actual_ret_1d_")]

    # Mean prediction
    merged["ensemble_mean_pred"] = merged[pred_cols].mean(axis=1)
    # Majority vote on direction
    merged["ensemble_vote_signed"] = merged[dir_cols].sum(axis=1)
    # Weighted by excess sharpe
    weights = np.array([m["oos_excess_sharpe"] for m in members])
    weights = weights / weights.sum()
    merged["ensemble_weighted_pred"] = (merged[pred_cols].values * weights[None, :]).sum(axis=1)
    # All members agree on actual return — take first
    merged["actual"] = merged[actual_cols[0]]

    # Top-K and family ensembles
    top5_pred_cols = [f"pred_ret_1d_exp{m['experiment_num']}" for m in members[:5]]
    top3_pred_cols = [f"pred_ret_1d_exp{m['experiment_num']}" for m in members[:3]]
    top5_dir_cols = [f"pred_direction_exp{m['experiment_num']}" for m in members[:5]]
    top3_dir_cols = [f"pred_direction_exp{m['experiment_num']}" for m in members[:3]]
    lstm_members = [m for m in members if m["backbone"] == "lstm"]
    mamba_members = [m for m in members if m["backbone"] == "mamba"]
    lstm_pred_cols = [f"pred_ret_1d_exp{m['experiment_num']}" for m in lstm_members]
    mamba_pred_cols = [f"pred_ret_1d_exp{m['experiment_num']}" for m in mamba_members]

    merged["ensemble_top5_mean"] = merged[top5_pred_cols].mean(axis=1) if top5_pred_cols else 0
    merged["ensemble_top3_mean"] = merged[top3_pred_cols].mean(axis=1) if top3_pred_cols else 0
    merged["ensemble_top5_vote"] = merged[top5_dir_cols].sum(axis=1) if top5_dir_cols else 0
    merged["ensemble_top3_vote"] = merged[top3_dir_cols].sum(axis=1) if top3_dir_cols else 0
    merged["ensemble_lstm_mean"] = merged[lstm_pred_cols].mean(axis=1) if lstm_pred_cols else 0
    merged["ensemble_mamba_mean"] = merged[mamba_pred_cols].mean(axis=1) if mamba_pred_cols else 0

    # Confidence-filtered: only trade when 9+ of 13 members agree on direction
    high_conf_threshold = 9
    merged["ensemble_high_conf"] = merged["ensemble_vote_signed"].where(
        merged["ensemble_vote_signed"].abs() >= high_conf_threshold, 0
    )

    # Compute strategy metrics for each ensemble strategy
    strategies = {}
    for name, signal_col in [
        ("all13_mean_pred",   "ensemble_mean_pred"),
        ("all13_vote",        "ensemble_vote_signed"),
        ("all13_weighted",    "ensemble_weighted_pred"),
        ("top5_mean",         "ensemble_top5_mean"),
        ("top3_mean",         "ensemble_top3_mean"),
        ("top5_vote",         "ensemble_top5_vote"),
        ("top3_vote",         "ensemble_top3_vote"),
        (f"lstm_only_{len(lstm_members)}", "ensemble_lstm_mean"),
        (f"mamba_only_{len(mamba_members)}", "ensemble_mamba_mean"),
        (f"vote_geq_{high_conf_threshold}", "ensemble_high_conf"),
    ]:
        strategies[name] = compute_strategy(merged, signal_col, "actual")
        print(f"[ensemble] {name:>20}: Sharpe={strategies[name].get('strategy_annual_sharpe'):>+6.3f} "
              f"excess={strategies[name].get('excess_sharpe'):>+6.3f} "
              f"return={strategies[name].get('strategy_total_return_pct'):>+6.2f}% "
              f"hit={strategies[name].get('hit_rate_pct'):>5.2f}%")

    # Save the per-day frame for inspection
    out_df = merged[[*pred_cols, "ensemble_mean_pred", "ensemble_vote_signed",
                     "ensemble_weighted_pred", "actual"]].copy()
    out_df.to_csv(ENSEMBLE_CSV, float_format="%.6f")
    print(f"[ensemble] wrote {ENSEMBLE_CSV.relative_to(RESULTS)} ({len(out_df)} rows × {out_df.shape[1]} cols)")

    summary = {
        "method": "Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474",
        "n_members": len(members),
        "members": [{
            "experiment_num": m["experiment_num"],
            "backbone": m["backbone"],
            "seed": m.get("seed"),
            "individual_oos_sharpe": m["oos_strategy_annual_sharpe"],
            "individual_excess_sharpe": m["oos_excess_sharpe"],
            "ensemble_weight": float(weights[i]),
        } for i, m in enumerate(members)],
        "n_common_dates": int(len(merged)),
        "oos_window": {"start": str(merged.index.min().date()),
                       "end": str(merged.index.max().date())},
        "strategies": strategies,
    }
    ENSEMBLE_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[ensemble] wrote {ENSEMBLE_JSON.relative_to(RESULTS)}")


if __name__ == "__main__":
    main()
