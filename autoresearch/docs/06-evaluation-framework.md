# 06 - Evaluation Framework

**SWEBoK Knowledge Area:** KA5 — Software Testing (Verification)
**Google SWE Reference:** Ch. 14 — "Larger Testing" (system-level evaluation)

---

## 1. Evaluation Philosophy

The evaluation framework is designed around three principles from Lopez de Prado (2018) "Advances in Financial Machine Learning":

1. **No single-split evaluation:** 7 disjoint test sets prevent overfitting to one market regime
2. **Risk-adjusted metrics:** Sharpe ratio penalizes volatile returns; raw returns can be gamed
3. **Statistical significance:** PSR and DSR quantify whether observed Sharpe is distinguishable from luck

## 2. Primary Metric: Average Sharpe Ratio

```
Sharpe_i = mean(strategy_returns_i) / std(strategy_returns_i) * sqrt(252)
Average_Sharpe = (1/7) * sum(Sharpe_1 ... Sharpe_7)
```

**Strategy returns = `sign(predicted_return) * actual_return`**

This directional trading strategy:
- Goes long when model predicts positive return
- Goes short when model predicts negative return
- Return magnitude = actual market return (no position sizing)
- Measures pure directional forecasting ability

### Why Average Sharpe (not concatenated)

Averaging per-fold Sharpe ratios prevents later folds (with more test data) from dominating. Each regime contributes equally, forcing the model to generalize across all market conditions.

### Weighted Sharpe

A secondary metric: `weighted_sharpe = dot(per_fold_sharpes, train_sizes / sum(train_sizes))`

Gives more weight to folds with more training data (higher confidence estimates).

## 3. Complete Metrics Suite

### 3.1 Risk-Adjusted Performance

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Sharpe Ratio** | (mean/std) * sqrt(252) | Risk-adjusted return; >1.0 = good |
| **Sortino Ratio** | (mean/downside_std) * sqrt(252) | Penalizes only downside volatility |
| **Calmar Ratio** | annualized_return / max_drawdown | Return per unit drawdown |
| **Omega Ratio** | sum(max(r,0)) / sum(max(-r,0)) | Gain/loss ratio; >1.0 = net positive |

### 3.2 Statistical Significance (Lopez de Prado)

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **PSR** | CDF(z_score) where z = (SR - bench) / SE(SR) | Probability observed Sharpe is real; >0.95 = significant |
| **DSR** | PSR(SR, benchmark=expected_max_null_SR) | Sharpe adjusted for multiple testing (n_trials) |

**PSR Standard Error** accounts for:
- Sample size (n observations)
- Skewness of returns
- Kurtosis of returns
- Based on Lo (2002) with corrections

**DSR Benchmark** uses Gumbel extreme value distribution:
```
E[max(SR)] = (1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))
```
where N = number of trials, gamma = Euler-Mascheroni constant

### 3.3 Predictive Quality

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **IC (Spearman)** | rank_corr(predictions, actuals) | Monotonic prediction quality; >0.05 = meaningful |
| **IC (Pearson)** | corr(predictions, actuals) | Linear prediction quality |
| **Hit Rate** | % where sign(pred) == sign(actual) | Directional accuracy; >50% = better than coin flip |

### 3.4 Return Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| Total Return | prod(1 + r_i) - 1 | % |
| Annualized Return | (1 + total)^(252/n) - 1 | % |
| Final Equity | initial * prod(1 + r_i) | $ |
| Profit | final_equity - initial | $ |
| Mean Daily Return | mean(r_i) * 10000 | bps |
| Median Daily Return | median(r_i) * 10000 | bps |
| Daily Volatility | std(r_i) * 10000 | bps |

### 3.5 Risk & Tail Statistics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Max Drawdown** | max peak-to-trough decline | Worst loss from peak |
| **VaR 95%** | 5th percentile of returns | 95% of days lose less than this |
| **CVaR 95%** | mean of returns below VaR 95% | Expected loss on worst 5% of days |
| **VaR 99%** | 1st percentile of returns | Extreme tail risk |
| **CVaR 99%** | mean of returns below VaR 99% | Expected extreme loss |
| **Skewness** | scipy.stats.skew(returns) | <0 = left tail heavier (more large losses) |
| **Excess Kurtosis** | scipy.stats.kurtosis(returns) | >0 = fatter tails than normal |
| **Tail Ratio** | |percentile_95| / |percentile_5| | >1 = right tail fatter (more big wins) |

### 3.6 Trade Statistics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Win Rate | % positive returns | >50% = directional edge |
| N Trades | count of non-zero returns | Activity level |
| Avg Win | mean of positive returns (bps) | Average gain per winning day |
| Avg Loss | mean of negative returns (bps) | Average loss per losing day |
| Profit Factor | sum(wins) / |sum(losses)| | >1 = net profitable |
| Max Consec Wins | longest winning streak | Streak analysis |
| Max Consec Losses | longest losing streak | Drawdown duration proxy |
| Recovery Factor | total_return / max_drawdown | How quickly drawdowns recover |

## 4. Walk-Forward Protocol

```
For each fold i in [1..7]:
    1. Split: train_i, val_i, test_i = split_data(data, FOLDS[i])
       - split_data punches out ALL val/test windows from ALL folds
         from the training data (cross-fold contamination prevention)
       - A 10-calendar-day label-horizon buffer before each excluded
         window is also removed to prevent fwd_ret_5d targets from
         peeking into held-out periods
    2. Validate: len(train) >= seq_len, len(val) >= seq_len, len(test) >= seq_len
    3. Scale: scaler.fit(train) → transform(train, val, test)
    4. Train: model = train_one_fold(backbone, train, val, scaler, epochs)
    5. Predict: preds = model(test)
    6. Evaluate:
       - strategy_returns = sign(preds) * actuals
       - report = trading_report(strategy_returns)
       - ic = information_coefficient(preds, actuals)
    7. Checkpoint: save fold results to JSON

After all folds:
    avg_sharpe = mean(per_fold_sharpes)
    wtd_sharpe = weighted_mean(per_fold_sharpes, train_sizes)
    overall = trading_report(concatenate(all_fold_returns))
    psr = probabilistic_sharpe_ratio(all_returns)
    dsr = deflated_sharpe_ratio(all_returns, n_trials)
```

### Cross-Fold Contamination Prevention

With expanding training windows, later folds' training data would naturally
include earlier folds' val/test date ranges. This is prevented by:

1. **Hole-punching:** `split_data()` collects all 14 held-out ranges (7 folds x 2
   windows each) and removes them from every fold's training data.
2. **Label-horizon buffer:** An additional 10 calendar days (~7 business days) before
   each held-out window are excluded, because `fwd_ret_5d` targets at those dates
   reference prices inside the excluded period.
3. **Runtime validation:** `validate_purge_embargo()` checks all constraints at startup;
   the run aborts if any violation is detected.
4. **Test coverage:** `test_no_cross_fold_contamination` and
   `test_label_horizon_buffer_enforced` verify no training date falls in or near
   any held-out window.

## 5. Ablation Study Protocol

```
For each backbone in BACKBONE_REGISTRY:
    1. Run full walk-forward evaluation (7 folds)
    2. Record: avg_sharpe, wtd_sharpe, overall_report, per_fold_details
    3. Save to ablation_results/{backbone}.json
    4. Handle errors gracefully (record error, continue to next backbone)

After all backbones:
    1. Generate summary table (sorted by avg_sharpe descending)
    2. Generate per-backbone detail sections
    3. Identify best/worst backbones
    4. Check statistical significance (PSR > 0.95)
    5. Analyze IC across backbones
    6. Save Markdown report to reports/
```

## 6. Interpretation Guide

| Scenario | Avg Sharpe | PSR | IC | Interpretation |
|----------|-----------|-----|-----|---------------|
| Strong signal | > 0.5 | > 0.95 | > 0.05 | Statistically significant predictive edge |
| Weak signal | 0.0 - 0.5 | 0.5 - 0.95 | 0.01 - 0.05 | Some edge, but not statistically significant |
| No signal | ~ 0.0 | ~ 0.5 | ~ 0.0 | Model no better than random |
| Negative signal | < 0.0 | < 0.5 | < 0.0 | Model is contrarian indicator (inverse useful) |
| Overfitting | High on some folds, negative on others | Low | Variable | Regime-specific signal, not generalizable |

## 7. Known Limitations

1. **Strategy simplicity:** Pure directional (`sign(pred) * actual`) ignores position sizing, transaction costs, and slippage
2. **No transaction costs:** Real FX spreads are ~1-2 pips; not modeled
3. **No position sizing:** Equal-weight positions regardless of conviction
4. **Single-pair evaluation:** Only EUR/USD returns in strategy; other 5 pairs predicted but not traded
5. **Daily rebalancing assumed:** Implicitly assumes daily position changes at close
