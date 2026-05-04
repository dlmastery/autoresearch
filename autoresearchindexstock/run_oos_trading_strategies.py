"""Trading strategy laboratory on the ensemble-winner OOS predictions.

Builds 8 distinct trading strategies on top of the top-5 vote ensemble
and ranks them by OOS Sharpe.

Each strategy uses the SAME underlying signal (votes from the 5 prod-retrain
BH-beaters by individual excess Sharpe) but applies a different exposure /
sizing / filtering rule. This is the right way to compare strategies — keep
the alpha source constant, vary only the execution overlay.

Strategies (all on the 81 OOS dates 2025-12-24 → 2026-04-22):
  1. vanilla_long_short — daily rebalance, +1 / -1 / 0
  2. long_only          — long when vote > 0, cash otherwise
  3. high_confidence_5  — only trade when all 5 agree (+1 or -1, else 0)
  4. confidence_weighted — position = vote_sum / 5 (continuous −1..+1)
  5. vol_target_15      — confidence_weighted scaled to 15% annual vol
  6. trend_filter_50d   — vanilla only when QQQ above 50d SMA
  7. stop_loss_2pct     — vanilla + intraday stop at -2%
  8. two_tier_sizing    — 5/5 → 1.5x; 3-4/5 → 1.0x; <3 → 0

For each:
  - daily P&L series
  - cumulative equity curve
  - annualized Sharpe + Sortino
  - PSR (Bailey-López de Prado 2012)
  - max drawdown
  - hit rate, exposure ratio (% of days with non-zero position)
  - average position size
  - turnover (sum of |position[t] - position[t-1]|)

Outputs:
  - oos_strategy_<name>.csv (daily)
  - oos_trading_strategies_summary.json (top-5 ranked + detailed methodology)

References:
  - Vanilla long-short:        Sharpe 1966 J. Business
  - Long-only filter:          Daly 2008 J. Asset Management
  - Confidence-weighted size:  Lim, Zohren & Roberts 2019 J. Financial Data Science
                               arXiv:1906.04572 'Enhancing Time Series Momentum...'
  - Vol-targeting:             Moskowitz, Ooi & Pedersen 2012 JFE 'Time series momentum'
  - Kelly fractional:          Kelly 1956 Bell Sys. Tech. J.
  - Trend filter:              Faber 2007 J. Wealth Management 'A Quantitative Approach
                               to Tactical Asset Allocation'
  - Two-tier conviction sizing: López de Prado 2018 'Advances in Financial ML' §10
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "autoresearch_results"
TABLE_PATH = RESULTS / "oos_top30_table.json"
ENSEMBLE_SUMMARY = RESULTS / "oos_ensemble_summary.json"
SUMMARY_PATH = RESULTS / "oos_trading_strategies_summary.json"


def load_top5_member_directions() -> pd.DataFrame:
    """Returns DataFrame indexed by date with 5 columns (one per top-5 member),
    each value = pred_direction (-1, 0, +1) and an 'actual_ret_1d' column."""
    es = json.loads(ENSEMBLE_SUMMARY.read_text(encoding="utf-8"))
    members = sorted(es["members"], key=lambda m: -m["individual_excess_sharpe"])[:5]
    frames = []
    for m in members:
        csv_path = RESULTS / f"oos_exp{m['experiment_num']}_prod.csv"
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
        col = f"dir_exp{m['experiment_num']}"
        frames.append(df[["pred_direction"]].rename(columns={"pred_direction": col}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="inner")
    actual = pd.read_csv(RESULTS / f"oos_exp{members[0]['experiment_num']}_prod.csv",
                         parse_dates=["date"]).set_index("date")["actual_ret_1d"]
    merged["actual_ret_1d"] = actual
    return merged.dropna(subset=["actual_ret_1d"]), members


def annualized_sharpe(pnl: pd.Series) -> float:
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def annualized_sortino(pnl: pd.Series) -> float:
    downside = pnl[pnl < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(pnl.mean() / downside.std() * np.sqrt(252))


def probabilistic_sharpe_ratio(pnl: pd.Series) -> float:
    n = len(pnl)
    if n < 2 or pnl.std() == 0:
        return 0.0
    sh = annualized_sharpe(pnl) / np.sqrt(252)  # daily Sharpe
    skew = float(pnl.skew()) if n > 2 else 0.0
    kurt = float(pnl.kurtosis()) if n > 3 else 0.0
    sigma_sh = np.sqrt(max(1e-9, (1 - skew * sh + ((kurt - 1) / 4) * sh ** 2) / (n - 1)))
    return float(stats.norm.cdf(sh / sigma_sh))


def metrics_from_position(position: pd.Series, actual: pd.Series, name: str) -> dict:
    pnl = position * actual
    cum_pnl = pnl.fillna(0).cumsum()
    cum_bh = actual.fillna(0).cumsum()
    correct = ((np.sign(position) == np.sign(actual)) & (position != 0)).astype(int)
    n_trades = int((position != 0).sum())
    exposure = float((position != 0).mean())
    avg_pos = float(position.abs().mean())
    turnover = float(position.diff().abs().sum())

    sharpe = annualized_sharpe(pnl.dropna())
    bh_sharpe = annualized_sharpe(actual.dropna())
    sortino = annualized_sortino(pnl.dropna())
    psr = probabilistic_sharpe_ratio(pnl.dropna())
    maxdd = float((cum_pnl - cum_pnl.cummax()).min() * 100) if len(cum_pnl) else 0.0

    out = {
        "strategy": name,
        "n_predictions": int(len(position)),
        "n_trades": n_trades,
        "exposure_pct": round(exposure * 100, 2),
        "avg_position": round(avg_pos, 4),
        "turnover": round(turnover, 4),
        "strategy_annual_sharpe": round(sharpe, 4),
        "buy_hold_annual_sharpe": round(bh_sharpe, 4),
        "excess_sharpe": round(sharpe - bh_sharpe, 4),
        "annual_sortino": round(sortino, 4),
        "psr": round(psr, 4),
        "strategy_total_return_pct": round(float(pnl.sum()) * 100, 4),
        "buy_hold_total_return_pct": round(float(actual.sum()) * 100, 4),
        "excess_return_pct": round(float(pnl.sum() - actual.sum()) * 100, 4),
        "hit_rate_when_traded_pct": round(float(correct.sum() / max(n_trades, 1)) * 100, 2),
        "max_drawdown_pct": round(maxdd, 4),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in cum_pnl.index],
            "strategy_pct": [round(v * 100, 4) for v in cum_pnl.tolist()],
            "buy_hold_pct": [round(v * 100, 4) for v in cum_bh.tolist()],
        },
    }
    return out


def main():
    df, members = load_top5_member_directions()
    print(f"[strategies] {len(df)} OOS dates, top-5 members:")
    for m in members:
        print(f"  exp {m['experiment_num']:>3} {m['backbone']:>10} seed={m['seed']} "
              f"indv={m['individual_oos_sharpe']:>+5.2f}")

    dir_cols = [c for c in df.columns if c.startswith("dir_exp")]
    actual = df["actual_ret_1d"]

    # Vote sum across 5 members (range -5..+5)
    vote_sum = df[dir_cols].sum(axis=1)
    vote_sign = np.sign(vote_sum).fillna(0).astype(int)

    # QQQ 50d SMA — use the QQQ_logret_1d cumulative path proxied from actuals
    # (actuals are 1-day forward log returns; build a price index from them)
    qqq_logprice = actual.cumsum()
    sma50 = qqq_logprice.rolling(50, min_periods=10).mean()
    above_50d_sma = (qqq_logprice > sma50).astype(int)

    # === STRATEGY 1: Vanilla long-short (the current production champion) ===
    pos1 = vote_sign.copy().astype(float)

    # === STRATEGY 2: Long-only (skip shorts) ===
    pos2 = vote_sign.where(vote_sign > 0, 0).astype(float)

    # === STRATEGY 3: High-confidence (5/5 agree, else flat) ===
    pos3 = vote_sign.where(vote_sum.abs() == 5, 0).astype(float)

    # === STRATEGY 4: Confidence-weighted (continuous -1..+1) ===
    pos4 = (vote_sum / 5.0).astype(float)

    # === STRATEGY 5: Vol-targeting (15% annual on confidence-weighted) ===
    target_vol = 0.15  # 15% annualized
    rolling_vol = (actual.rolling(20, min_periods=5).std() * np.sqrt(252)).fillna(target_vol)
    vol_scale = (target_vol / rolling_vol).clip(upper=2.0)  # cap leverage at 2x
    pos5 = (pos4 * vol_scale).clip(-2.0, 2.0)

    # === STRATEGY 6: Trend-filter (vanilla only when QQQ above 50d SMA) ===
    # Default is to follow ensemble; if below 50d SMA, no LONG (stay neutral or SHORT only)
    pos6 = vote_sign.copy().astype(float)
    pos6 = pos6.where(~((above_50d_sma == 0) & (vote_sign > 0)), 0)

    # === STRATEGY 7: Stop-loss 2% (modeled as ex-post per-day return floor) ===
    # Approximation: if intraday loss exceeds 2%, exit. Daily-bar approximation:
    # cap each daily strategy P&L at -0.02 (2% loss). This is an UPPER-BOUND
    # estimate since real intraday execution might trigger earlier.
    pos7 = vote_sign.copy().astype(float)
    # Compute pnl normally then floor at -2% as if a stop triggered
    pnl7_uncapped = pos7 * actual
    pnl7 = pnl7_uncapped.clip(lower=-0.02)
    # We'll handle this as a custom case below

    # === STRATEGY 8: Two-tier conviction sizing ===
    pos8 = pd.Series(0.0, index=vote_sum.index)
    pos8 = pos8.where(vote_sum.abs() < 3, np.sign(vote_sum) * 1.0)   # 3-4/5 → 1x
    pos8 = pos8.where(vote_sum.abs() != 5, np.sign(vote_sum) * 1.5)  # 5/5 → 1.5x

    # Compute metrics per strategy
    strategies = {}
    strategies["vanilla_long_short"] = metrics_from_position(pos1, actual, "vanilla_long_short")
    strategies["long_only"] = metrics_from_position(pos2, actual, "long_only")
    strategies["high_confidence_5_5"] = metrics_from_position(pos3, actual, "high_confidence_5_5")
    strategies["confidence_weighted"] = metrics_from_position(pos4, actual, "confidence_weighted")
    strategies["vol_target_15pct"] = metrics_from_position(pos5, actual, "vol_target_15pct")
    strategies["trend_filter_50d"] = metrics_from_position(pos6, actual, "trend_filter_50d")
    # Stop-loss: handle via custom pnl
    cum7 = pnl7.fillna(0).cumsum()
    correct7 = ((np.sign(pos7) == np.sign(actual)) & (pos7 != 0)).astype(int)
    n_tr7 = int((pos7 != 0).sum())
    sh7 = annualized_sharpe(pnl7.dropna())
    bh = annualized_sharpe(actual.dropna())
    strategies["stop_loss_2pct"] = {
        "strategy": "stop_loss_2pct",
        "n_predictions": int(len(pos7)),
        "n_trades": n_tr7,
        "exposure_pct": round(float((pos7 != 0).mean()) * 100, 2),
        "avg_position": round(float(pos7.abs().mean()), 4),
        "turnover": round(float(pos7.diff().abs().sum()), 4),
        "strategy_annual_sharpe": round(sh7, 4),
        "buy_hold_annual_sharpe": round(bh, 4),
        "excess_sharpe": round(sh7 - bh, 4),
        "annual_sortino": round(annualized_sortino(pnl7.dropna()), 4),
        "psr": round(probabilistic_sharpe_ratio(pnl7.dropna()), 4),
        "strategy_total_return_pct": round(float(pnl7.sum()) * 100, 4),
        "buy_hold_total_return_pct": round(float(actual.sum()) * 100, 4),
        "excess_return_pct": round(float(pnl7.sum() - actual.sum()) * 100, 4),
        "hit_rate_when_traded_pct": round(float(correct7.sum() / max(n_tr7, 1)) * 100, 2),
        "max_drawdown_pct": round(float((cum7 - cum7.cummax()).min() * 100), 4),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in cum7.index],
            "strategy_pct": [round(v * 100, 4) for v in cum7.tolist()],
            "buy_hold_pct": [round(v * 100, 4) for v in actual.fillna(0).cumsum().tolist()],
        },
    }
    strategies["two_tier_sizing"] = metrics_from_position(pos8, actual, "two_tier_sizing")

    # Save per-strategy CSVs
    csv_data = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in df.index],
        "vote_sum": vote_sum.values,
        "vote_sign": vote_sign.values,
        "actual_ret_1d": actual.values,
        "pos_vanilla_long_short": pos1.values,
        "pos_long_only": pos2.values,
        "pos_high_conf_5": pos3.values,
        "pos_conf_weighted": pos4.values,
        "pos_vol_target": pos5.values,
        "pos_trend_filter": pos6.values,
        "pos_stop_loss": pos7.values,
        "pos_two_tier": pos8.values,
        "pnl_vanilla": (pos1 * actual).values,
        "pnl_long_only": (pos2 * actual).values,
        "pnl_high_conf": (pos3 * actual).values,
        "pnl_conf_weighted": (pos4 * actual).values,
        "pnl_vol_target": (pos5 * actual).values,
        "pnl_trend_filter": (pos6 * actual).values,
        "pnl_stop_loss": pnl7.values,
        "pnl_two_tier": (pos8 * actual).values,
    })
    csv_data.to_csv(RESULTS / "oos_trading_strategies.csv", index=False, float_format="%.6f")

    # Per-strategy individual CSVs (for dashboard download)
    for name, s in strategies.items():
        per = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in df.index],
            "vote_sum": vote_sum.values,
            "actual_ret_1d": actual.values,
            "position": csv_data[f"pos_{name.split('_')[0] if name == 'vanilla_long_short' else name.replace('vanilla_long_short','vanilla').replace('long_only','long_only').replace('high_confidence_5_5','high_conf_5').replace('confidence_weighted','conf_weighted').replace('vol_target_15pct','vol_target').replace('trend_filter_50d','trend_filter').replace('stop_loss_2pct','stop_loss').replace('two_tier_sizing','two_tier')}"].values
                if False else
                csv_data[{
                    "vanilla_long_short": "pos_vanilla_long_short",
                    "long_only": "pos_long_only",
                    "high_confidence_5_5": "pos_high_conf_5",
                    "confidence_weighted": "pos_conf_weighted",
                    "vol_target_15pct": "pos_vol_target",
                    "trend_filter_50d": "pos_trend_filter",
                    "stop_loss_2pct": "pos_stop_loss",
                    "two_tier_sizing": "pos_two_tier",
                }[name]].values,
            "pnl": csv_data[{
                "vanilla_long_short": "pnl_vanilla",
                "long_only": "pnl_long_only",
                "high_confidence_5_5": "pnl_high_conf",
                "confidence_weighted": "pnl_conf_weighted",
                "vol_target_15pct": "pnl_vol_target",
                "trend_filter_50d": "pnl_trend_filter",
                "stop_loss_2pct": "pnl_stop_loss",
                "two_tier_sizing": "pnl_two_tier",
            }[name]].values,
        })
        per["cumulative_pnl"] = per["pnl"].fillna(0).cumsum()
        per["cumulative_buy_hold"] = per["actual_ret_1d"].fillna(0).cumsum()
        per.to_csv(RESULTS / f"oos_strategy_{name}.csv", index=False, float_format="%.6f")
        s["csv"] = f"oos_strategy_{name}.csv"

    # Rank by Sharpe
    ranked = sorted(strategies.values(), key=lambda s: -s["strategy_annual_sharpe"])
    print("\n=== TOP-5 TRADING STRATEGIES (by OOS Sharpe) ===")
    print(f"{'#':>2} {'Strategy':<22} {'Sharpe':>7} {'Excess':>7} {'Return':>8} {'Hit%':>6} {'Exp%':>6}")
    for i, s in enumerate(ranked[:8], 1):
        print(f"{i:>2} {s['strategy']:<22} {s['strategy_annual_sharpe']:>+7.3f} {s['excess_sharpe']:>+7.3f} "
              f"{s['strategy_total_return_pct']:>+7.2f}% {s['hit_rate_when_traded_pct']:>5.1f}% {s['exposure_pct']:>5.1f}%")

    summary = {
        "method": "Lakshminarayanan-Pritzel-Blundell 2017 ensemble + various execution overlays",
        "ensemble_members": members,
        "n_oos_dates": len(df),
        "oos_window": {"start": str(df.index.min().date()), "end": str(df.index.max().date())},
        "strategies": strategies,
        "ranked_by_sharpe": [s["strategy"] for s in ranked],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] wrote {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
