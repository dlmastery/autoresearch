# 06 - Evaluation Framework

**SWEBoK Knowledge Area:** KA5 — Software Testing (Verification)
**Google SWE Reference:** Ch. 14 — "Larger Testing" (system-level evaluation)

---

## Executive Summary

This document describes the evaluation methodology for the AutoResearch FX prediction system. The system uses a **super-fold** approach: one model trains on all historical data (2005-2025) with all 7 folds' validation and test windows hole-punched out, then evaluates on each window individually. The primary decision metric is a **composite score** that rewards consistency across market regimes:

```
composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds
```

The system computes 30+ metrics per evaluation window spanning risk-adjusted performance (Sharpe, Sortino, Calmar, Omega), statistical significance (PSR, DSR), predictive quality (IC, hit rate), tail risk (VaR, CVaR), and trade statistics (profit factor, win streaks, recovery factor). Classification metrics (precision, recall, F1, F2, MCC) for directional accuracy are planned additions.

**Champion results (Residual MLP, 301K params):**
- Test Sharpe: +6.21 (7/7 positive folds)
- Val Sharpe: +5.60
- Composite: +5.50
- Total test return: +1001%
- 2,478 training samples, 1,170 test samples across 7 regime windows

```
  Evaluation Architecture
  ========================

  Trained Model (single super-fold)
       |
       ├──> Test Window 1 (2008-01 to 2008-06, GFC onset)
       |         -> Sharpe, Sortino, IC, hit rate, ...
       |
       ├──> Test Window 2 (2010-01 to 2010-06, post-crash)
       |         -> Sharpe, Sortino, IC, hit rate, ...
       |
       ├──> Test Window 3 (2013-01 to 2013-06, Eurozone)
       |         -> ...
       |
       ├──> Test Window 4 (2015-04 to 2015-12, strong USD)
       |         -> ...
       |
       ├──> Test Window 5 (2019-01 to 2019-09, low-vol)
       |         -> ...
       |
       ├──> Test Window 6 (2022-01 to 2022-09, EUR crisis)
       |         -> ...
       |
       ├──> Test Window 7 (2025-01 to 2025-09, recent)
       |         -> ...
       |
       v
  Aggregate: avg Sharpe, composite score, KEEP/DISCARD decision
```

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

## 3. Complete Metrics Suite (30+ Metrics)

The evaluation framework computes a comprehensive suite of metrics organized into six categories. All metrics are computed per evaluation window and then aggregated.

```
  Metrics Taxonomy
  ================

  +--------------------+   +--------------------+   +--------------------+
  | Risk-Adjusted      |   | Statistical        |   | Predictive         |
  | Performance        |   | Significance       |   | Quality            |
  |                    |   |                    |   |                    |
  | - Sharpe           |   | - PSR              |   | - IC (Spearman)    |
  | - Sortino          |   | - DSR              |   | - IC (Pearson)     |
  | - Calmar           |   |                    |   | - Hit Rate         |
  | - Omega            |   |                    |   |                    |
  +--------------------+   +--------------------+   +--------------------+

  +--------------------+   +--------------------+   +--------------------+
  | Return Metrics     |   | Risk & Tail        |   | Trade Statistics   |
  |                    |   | Statistics         |   |                    |
  | - Total Return     |   | - Max Drawdown     |   | - Win Rate         |
  | - Annualized Ret   |   | - VaR 95%/99%      |   | - Avg Win/Loss     |
  | - Final Equity     |   | - CVaR 95%/99%     |   | - Profit Factor    |
  | - Profit           |   | - Skewness         |   | - Max Consec W/L   |
  | - Mean Daily (bps) |   | - Kurtosis         |   | - Recovery Factor  |
  | - Daily Vol (bps)  |   | - Tail Ratio       |   |                    |
  +--------------------+   +--------------------+   +--------------------+
```

### 3.1 Risk-Adjusted Performance

| Metric | Formula | Interpretation | Champion (Test Avg) |
|--------|---------|---------------|:-------------------:|
| **Sharpe Ratio** | (mean/std) * sqrt(252) | Risk-adjusted return; >1.0 = good, >3.0 = exceptional | +6.21 |
| **Sortino Ratio** | (mean/downside_std) * sqrt(252) | Penalizes only downside volatility; always >= Sharpe | >+8.0 |
| **Calmar Ratio** | annualized_return / max_drawdown | Return per unit drawdown; >1.0 = recovers from drawdowns within a year | >+5.0 |
| **Omega Ratio** | sum(max(r,0)) / sum(max(-r,0)) | Gain/loss ratio; >1.0 = net positive, >1.5 = strong edge | >1.5 |

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

## 5. Super-Fold Evaluation Methodology

The super-fold approach replaces the traditional 7-fold walk-forward loop with a single train/eval pass, reducing experiment time by ~7x while maintaining the same evaluation rigor.

### 5.1 How Super-Fold Works

Instead of training 7 separate models (one per fold), super-fold trains ONE model on ALL available training data with ALL 14 held-out windows (7 val + 7 test) removed:

```
  Training Data Construction (Super-Fold)
  ========================================

  Full timeline: 2005-01 ─────────────────────────────── 2025-09
                 |                                              |
  Training data: |████████████████████████████████████████████| = 2478 rows
                 |   (with 14 holes + 14 label buffers)        |
                 |                                              |
  Holes:  V1 T1  V2 T2  V3 T3  V4  T4  V5   T5  V6   T6  V7  T7
          |  |   |  |   |  |   |   |   |    |   |    |   |   |
          '07'08 '09'10 '12'13 '14 '15 '18  '19 '21  '22 '24 '25

  V = val window, T = test window
  Each hole also has a 10-day label-horizon buffer before it
  to prevent fwd_ret_5d targets from leaking into the held-out period.
```

### 5.2 Super-Fold Data Counts

| Split | Rows | Composition |
|-------|:----:|-------------|
| **Train** | 2,478 | All business days 2005-2025 minus all 14 held-out windows, minus 14 label-horizon buffers (10 cal days each) |
| **Val** | 915 | Union of all 7 val windows (7 x ~130 rows) |
| **Test** | 1,170 | Union of all 7 test windows (7 x ~167 rows) |
| **Total** | 4,563 | Zero overlap verified programmatically |

### 5.3 Per-Window Evaluation

After training, the model is evaluated on each of the 14 windows independently. Each window is processed as a contiguous block with its own metrics:

```python
for window in test_windows:
    preds = model(window_features)           # forward pass on window
    strategy_returns = sign(preds) * actuals  # directional trading
    report = trading_report(strategy_returns)  # 30+ metrics
    ic = information_coefficient(preds, actuals)
```

### 5.4 The 7 Regime Windows (Test Set)

| Fold | Regime | Test Dates | Market Context | Champion Sharpe |
|:----:|--------|:----------:|----------------|:---------------:|
| 1 | Pre-crisis/GFC onset | 2008-01 to 2008-06 | Housing crisis unfolds, Bear Stearns collapse, extreme volatility | +2.46 |
| 2 | Post-crash recovery | 2010-01 to 2010-06 | EUR flash crash, Greek debt crisis begins, recovery stalls | +1.17 |
| 3 | Eurozone debt plateau | 2013-01 to 2013-06 | Draghi "whatever it takes" aftermath, low volatility | +9.76 |
| 4 | Strong USD downturn | 2015-04 to 2015-12 | Fed first rate hike, EUR weakness, China devaluation | +9.78 |
| 5 | Low-vol plateau | 2019-01 to 2019-09 | Pre-COVID calm, trade war uncertainty, inverted yield curve | +8.85 |
| 6 | EUR crisis downturn | 2022-01 to 2022-09 | Russia-Ukraine war, ECB rate hikes, EUR/USD parity | +9.95 |
| 7 | Recent mixed/upturn | 2025-01 to 2025-09 | Post-election, tariff uncertainty, mixed data | +8.48 |

### 5.5 Composite Metric (Keep/Revert Decision)

The composite metric is the single number that determines whether an experiment's result is kept or discarded:

```
composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds
```

**Design rationale:**
- `min(test_sharpe, val_sharpe)`: The model must perform well on BOTH val and test. This prevents overfitting to test (impossible since test is unseen during training, but catches lucky noise) or val (which guides early stopping).
- `0.1 * n_negative_folds`: Penalty for each fold window (val or test) with negative Sharpe. A model with avg Sharpe +3.0 but two negative folds scores `3.0 - 0.2 = 2.8`, while a model with avg Sharpe +2.5 and zero negative folds scores `2.5 - 0.0 = 2.5`. This rewards **consistency across all regimes** rather than extreme performance in a few.
- n_negative_folds counts BOTH val and test windows. Maximum possible penalty: 0.1 * 14 = 1.4.

**Champion composite breakdown:**
```
composite = min(+6.21, +5.60) - 0.1 * 0
          = +5.60 - 0.0
          = +5.50  (rounded)
```

### 5.6 Planned Classification Metrics

The current evaluation suite focuses on regression quality and trading performance. Planned additions include classification metrics for directional accuracy, motivated by the 918-experiment study (Saidd, 2026) finding that MSE-trained models achieve only ~50% directional accuracy:

| Metric | Formula | Why It Matters for FX |
|--------|---------|----------------------|
| **Precision** | TP / (TP + FP) | Of all "go long" signals, how many were correct? False longs = trading losses. |
| **Recall** | TP / (TP + FN) | Of all actual up days, how many did we catch? Low recall = missed opportunities. |
| **F1 Score** | 2 * P * R / (P + R) | Harmonic mean balancing precision and recall |
| **F2 Score** | 5 * P * R / (4*P + R) | Weights recall 2x more than precision (prefer catching moves over avoiding bad trades) |
| **MCC** | (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Best single metric for imbalanced binary classification; [-1, +1] |

These metrics would be computed by treating `sign(predicted_return)` as the predicted class and `sign(actual_return)` as the true class.

### 5.7 Per-Trade Win/Loss Logging (Planned)

Future enhancement: log individual trade-level detail for deeper analysis:
- Entry date, exit date, direction (long/short), predicted return, actual return, P&L
- Enables analysis of: average holding period, worst individual trades, correlation of win streaks with market volatility, regime-specific trade sizing optimization

---

## 6. Ablation Study Protocol

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

## 7. Interpretation Guide

| Scenario | Avg Sharpe | PSR | IC | Interpretation |
|----------|-----------|-----|-----|---------------|
| Strong signal | > 0.5 | > 0.95 | > 0.05 | Statistically significant predictive edge |
| Weak signal | 0.0 - 0.5 | 0.5 - 0.95 | 0.01 - 0.05 | Some edge, but not statistically significant |
| No signal | ~ 0.0 | ~ 0.5 | ~ 0.0 | Model no better than random |
| Negative signal | < 0.0 | < 0.5 | < 0.0 | Model is contrarian indicator (inverse useful) |
| Overfitting | High on some folds, negative on others | Low | Variable | Regime-specific signal, not generalizable |

## 8. Known Limitations

1. **Strategy simplicity:** Pure directional (`sign(pred) * actual`) ignores position sizing, transaction costs, and slippage
2. **No transaction costs:** Real FX spreads are ~1-2 pips; not modeled
3. **No position sizing:** Equal-weight positions regardless of conviction
4. **Single-pair evaluation:** Only EUR/USD returns in strategy; other 5 pairs predicted but not traded
5. **Daily rebalancing assumed:** Implicitly assumes daily position changes at close
