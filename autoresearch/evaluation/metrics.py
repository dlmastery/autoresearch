"""Evaluation metrics for trading strategy performance.

Provides annualized Sharpe ratio computation, fold-level aggregation,
and ML-finance best practices including PSR, deflated Sharpe, and
information coefficient (Lopez de Prado, 2018).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sp_stats

# Standard US equity/FX calendar assumption.
TRADING_DAYS_PER_YEAR: int = 252


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from an array of daily returns.

    Sharpe = (mean / std) * sqrt(252)

    Edge cases
    ----------
    * Empty array  -> 0.0
    * Zero std     -> 0.0  (constant or single-element returns)
    """
    if len(daily_returns) == 0:
        return 0.0
    std = float(np.std(daily_returns, ddof=0))
    if std == 0.0:
        return 0.0
    mean = float(np.mean(daily_returns))
    return (mean / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    daily_returns: np.ndarray,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR: probability that the true Sharpe exceeds sr_benchmark.

    Accounts for skewness and kurtosis of returns.
    Returns a value in [0, 1]; > 0.95 is conventionally significant.
    """
    n = len(daily_returns)
    if n < 3:
        return 0.0
    sr = sharpe_ratio(daily_returns)
    sr_daily = sr / math.sqrt(TRADING_DAYS_PER_YEAR)
    sr_bench_daily = sr_benchmark / math.sqrt(TRADING_DAYS_PER_YEAR)

    mu = float(np.mean(daily_returns))
    std = float(np.std(daily_returns, ddof=1))
    if std < 1e-12:
        return 0.0
    z = (daily_returns - mu) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0  # excess kurtosis

    # Standard error of the Sharpe ratio (Lo, 2002 + skew/kurtosis correction).
    # The bracketed term can go negative at extreme skew/kurt combinations
    # (e.g. nearly-deterministic strategies on short windows) -- clamp to
    # zero to avoid sqrt domain errors. PSR returned as 1.0 if the strategy
    # dominates under the Lo/BLP framework (se collapses to 0).
    bracket = 1.0 - skew * sr_daily + (kurt / 4.0) * sr_daily ** 2
    if bracket <= 0.0:
        return 1.0 if sr_daily > sr_bench_daily else 0.0
    se = math.sqrt(bracket / max(n - 1, 1))
    if se < 1e-12:
        return 0.0

    z_score = (sr_daily - sr_bench_daily) / se
    # CDF of standard normal
    psr = float(sp_stats.norm.cdf(z_score))
    return psr


def deflated_sharpe_ratio(
    daily_returns: np.ndarray,
    n_trials: int,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts for multiple testing by computing PSR against the expected
    maximum Sharpe under the null hypothesis of n_trials independent trials.

    Parameters
    ----------
    daily_returns : array of daily strategy returns
    n_trials : number of backtest trials / experiments run

    Returns probability that the observed Sharpe exceeds what you'd expect
    from the best of n_trials random strategies. > 0.95 = significant.
    """
    if n_trials < 1:
        n_trials = 1
    n = len(daily_returns)
    if n < 3:
        return 0.0

    # Expected max Sharpe under null: E[max(Z_1,...,Z_k)] ≈ approximation
    # Using the Euler-Mascheroni approximation for Gumbel extreme value
    euler_mascheroni = 0.5772156649
    e_max_z = (
        (1.0 - euler_mascheroni) * sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    ) if n_trials > 1 else 0.0

    # Convert daily benchmark Sharpe to annualized
    sr_benchmark = e_max_z * math.sqrt(TRADING_DAYS_PER_YEAR)

    return probabilistic_sharpe_ratio(daily_returns, sr_benchmark=sr_benchmark)


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def information_coefficient(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """Spearman rank IC and Pearson IC between predictions and actuals.

    Returns dict with ic_spearman, ic_pearson, ic_pvalue, and hit_rate.
    """
    if len(predictions) < 3 or len(actuals) < 3:
        return {"ic_spearman": 0.0, "ic_pearson": 0.0, "ic_pvalue": 1.0, "hit_rate": 0.0}

    spearman_corr, spearman_p = sp_stats.spearmanr(predictions, actuals)
    pearson_corr, pearson_p = sp_stats.pearsonr(predictions, actuals)

    # Directional hit rate: how often sign(pred) == sign(actual)
    hits = np.sum(np.sign(predictions) == np.sign(actuals))
    hit_rate = float(hits / len(predictions)) * 100.0

    return {
        "ic_spearman": round(float(spearman_corr), 4),
        "ic_pearson": round(float(pearson_corr), 4),
        "ic_pvalue": round(float(min(spearman_p, pearson_p)), 6),
        "hit_rate": round(hit_rate, 2),
    }


def classification_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """Direction-classification metrics for sign(prediction) strategy.

    Treats prediction as a binary classifier on direction:
      TP: pred UP and actual UP
      TN: pred DOWN and actual DOWN
      FP: pred UP but actual DOWN (false positive)
      FN: pred DOWN but actual UP (missed positive)

    Returns precision, recall, F1, F2, accuracy, MCC, and confusion counts.
    Per CLAUDE.md traditional ML metrics requirement.
    """
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0,
                "accuracy": 0.0, "mcc": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    pred_up = predictions > 0
    actual_up = actuals > 0
    tp = int(np.sum(pred_up & actual_up))
    tn = int(np.sum(~pred_up & ~actual_up))
    fp = int(np.sum(pred_up & ~actual_up))
    fn = int(np.sum(~pred_up & actual_up))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0

    # F_beta: (1 + beta^2) * P * R / (beta^2 * P + R)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0

    # Matthews Correlation Coefficient
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "accuracy": round(accuracy, 4),
        "mcc": round(mcc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def average_sharpe_across_folds(fold_returns: list[np.ndarray]) -> float:
    """Mean of per-fold Sharpe ratios.

    Parameters
    ----------
    fold_returns : list[np.ndarray]
        Each element is a 1-D array of daily returns for one fold.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If *fold_returns* is empty.
    """
    if not fold_returns:
        raise ValueError("fold_returns must be a non-empty list")
    sharpes = [sharpe_ratio(r) for r in fold_returns]
    return float(np.mean(sharpes))


def weighted_sharpe_across_folds(
    fold_returns: list[np.ndarray],
    train_sizes: list[int],
) -> float:
    """Training-size-weighted mean of per-fold Sharpe ratios.

    Folds with more training data get proportionally more weight,
    reflecting greater confidence in those estimates.
    """
    if not fold_returns:
        raise ValueError("fold_returns must be a non-empty list")
    sharpes = np.array([sharpe_ratio(r) for r in fold_returns])
    weights = np.array(train_sizes, dtype=float)
    weights /= weights.sum()
    return float(np.dot(sharpes, weights))


# ---------------------------------------------------------------------------
# Extended trading metrics
# ---------------------------------------------------------------------------


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a fraction (0.0 to 1.0) from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > worst:
            worst = dd
    return float(worst)


def trading_report(
    daily_returns: np.ndarray,
    initial_capital: float = 1000.0,
) -> dict:
    """Compute comprehensive trading metrics for a strategy.

    Parameters
    ----------
    daily_returns : np.ndarray
        1-D array of daily strategy returns (e.g. sign(pred) * actual).
    initial_capital : float
        Starting capital in dollars.

    Returns
    -------
    dict with keys:
        sharpe, total_return_pct, final_equity, max_drawdown_pct,
        profit, win_rate, n_trades, avg_win, avg_loss,
        profit_factor, calmar_ratio, annualized_return_pct
    """
    if len(daily_returns) == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0,
            "total_return_pct": 0.0, "final_equity": initial_capital,
            "max_drawdown_pct": 0.0, "profit": 0.0, "win_rate": 0.0,
            "n_trades": 0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "calmar_ratio": 0.0, "annualized_return_pct": 0.0,
            "mean_daily_bps": 0.0, "median_daily_bps": 0.0, "daily_vol_bps": 0.0,
            "skewness": 0.0, "excess_kurtosis": 0.0,
            "var_95": 0.0, "cvar_95": 0.0, "var_99": 0.0, "cvar_99": 0.0,
            "omega_ratio": 0.0, "tail_ratio": 0.0,
            "max_consec_wins": 0, "max_consec_losses": 0,
            "recovery_factor": 0.0,
            "psr": 0.0,
        }

    # Equity curve
    equity = initial_capital * np.cumprod(1.0 + daily_returns)
    final = float(equity[-1])
    profit = final - initial_capital
    total_return_pct = (final / initial_capital - 1.0) * 100.0

    # Annualized return
    n_days = len(daily_returns)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    annualized = ((final / initial_capital) ** (1.0 / max(n_years, 1e-9)) - 1.0) * 100.0

    # Drawdown
    dd = max_drawdown(equity)
    dd_pct = dd * 100.0

    # Win/loss
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    n_trades = int(np.sum(daily_returns != 0))
    win_rate = float(len(wins) / max(n_trades, 1)) * 100.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    # Profit factor = gross wins / gross losses
    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    profit_factor = gross_wins / gross_losses

    # Calmar ratio = annualized return / max drawdown
    calmar = annualized / (dd_pct if dd_pct > 0 else 1e-9)

    # Sortino ratio (downside deviation only)
    downside = daily_returns[daily_returns < 0]
    downside_std = float(np.std(downside, ddof=0)) if len(downside) > 0 else 1e-9
    sortino = (float(np.mean(daily_returns)) / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Return distribution stats
    skewness = float(np.mean(((daily_returns - np.mean(daily_returns)) / max(np.std(daily_returns), 1e-9)) ** 3)) if len(daily_returns) > 2 else 0.0
    kurtosis = float(np.mean(((daily_returns - np.mean(daily_returns)) / max(np.std(daily_returns), 1e-9)) ** 4)) - 3.0 if len(daily_returns) > 3 else 0.0

    # Value at Risk and Conditional VaR (Expected Shortfall)
    var_95 = float(np.percentile(daily_returns, 5))
    var_99 = float(np.percentile(daily_returns, 1))
    tail_5 = daily_returns[daily_returns <= var_95]
    cvar_95 = float(np.mean(tail_5)) if len(tail_5) > 0 else var_95
    tail_1 = daily_returns[daily_returns <= var_99]
    cvar_99 = float(np.mean(tail_1)) if len(tail_1) > 0 else var_99

    # Omega ratio: sum(max(r - 0, 0)) / sum(max(0 - r, 0))
    gains_sum = float(np.sum(np.maximum(daily_returns, 0)))
    losses_sum = float(np.sum(np.maximum(-daily_returns, 0)))
    omega = gains_sum / max(losses_sum, 1e-9)

    # Tail ratio: 95th percentile / abs(5th percentile)
    p95 = float(np.percentile(daily_returns, 95))
    p5_abs = abs(var_95) if abs(var_95) > 1e-12 else 1e-12
    tail_ratio = p95 / p5_abs

    # Max consecutive wins/losses
    signs = np.sign(daily_returns)
    max_consec_wins, max_consec_losses = 0, 0
    cur_wins, cur_losses = 0, 0
    for s in signs:
        if s > 0:
            cur_wins += 1
            cur_losses = 0
        elif s < 0:
            cur_losses += 1
            cur_wins = 0
        else:
            cur_wins = cur_losses = 0
        max_consec_wins = max(max_consec_wins, cur_wins)
        max_consec_losses = max(max_consec_losses, cur_losses)

    # Recovery factor: total return / max drawdown
    recovery_factor = total_return_pct / dd_pct if dd_pct > 0 else 0.0

    # Mean daily return in bps
    mean_daily_bps = float(np.mean(daily_returns)) * 10000
    median_daily_bps = float(np.median(daily_returns)) * 10000
    daily_vol_bps = float(np.std(daily_returns, ddof=0)) * 10000

    # Probabilistic Sharpe Ratio (vs benchmark of 0)
    psr = probabilistic_sharpe_ratio(daily_returns, sr_benchmark=0.0)

    return {
        "sharpe": sharpe_ratio(daily_returns),
        "sortino": round(sortino, 4),
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final, 2),
        "max_drawdown_pct": round(dd_pct, 2),
        "profit": round(profit, 2),
        "win_rate": round(win_rate, 2),
        "n_trades": n_trades,
        "avg_win": round(avg_win * 100, 4),  # as bps
        "avg_loss": round(avg_loss * 100, 4),  # as bps
        "profit_factor": round(profit_factor, 3),
        "calmar_ratio": round(calmar, 3),
        "annualized_return_pct": round(annualized, 2),
        "mean_daily_bps": round(mean_daily_bps, 2),
        "median_daily_bps": round(median_daily_bps, 2),
        "daily_vol_bps": round(daily_vol_bps, 2),
        "skewness": round(skewness, 4),
        "excess_kurtosis": round(kurtosis, 4),
        "var_95": round(var_95 * 100, 4),  # as pct
        "cvar_95": round(cvar_95 * 100, 4),  # as pct
        "var_99": round(var_99 * 100, 4),  # as pct
        "cvar_99": round(cvar_99 * 100, 4),  # as pct
        "omega_ratio": round(omega, 3),
        "tail_ratio": round(tail_ratio, 3),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "recovery_factor": round(recovery_factor, 3),
        "psr": round(psr, 4),
    }
