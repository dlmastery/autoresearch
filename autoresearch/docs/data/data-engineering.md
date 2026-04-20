# 03 - Data Engineering

**SWEBoK Knowledge Area:** KA2 -- Software Design (Data Design)
**Google SWE Reference:** Ch. 9 -- "Code Review" (data pipeline correctness)

---

## Key Highlights

- **104 backward-looking features** from 15 instruments (6 FX pairs + 9 macro signals), all strictly backward-looking with no future data leakage
- **21 years of daily data** (2005--2026): ~5,300 raw trading days -> ~4,914 after warmup and NaN cleaning
- **7-fold regime-aware walk-forward splits** with 90-day purge, 21-day embargo, and 10-day label-horizon buffers
- **Super-fold design** trains on 2,478 samples (all training data with val/test windows hole-punched), validates on 915, tests on 1,170
- **Zero leakage verified** programmatically: train/val overlap = 0, train/test overlap = 0, val/test overlap = 0

---

## 1. Data Sources

### 1.1 FX Currency Pairs

| Pair | Ticker | Base | Quote | Rationale |
|------|--------|------|-------|-----------|
| EUR/USD | EURUSD=X | EUR | USD | Most traded pair globally, primary prediction target |
| GBP/USD | GBPUSD=X | GBP | USD | Second most liquid; USD cross |
| USD/JPY | JPY=X | USD | JPY | Safe-haven currency; carry trade proxy |
| USD/CHF | CHF=X | USD | CHF | Safe-haven; inversely correlated with EUR/USD |
| EUR/GBP | EURGBP=X | EUR | GBP | EUR cross; isolates EUR dynamics from USD |
| EUR/JPY | EURJPY=X | EUR | JPY | Risk-on/off barometer |

### 1.2 Macroeconomic Signals

| Category | Ticker | Signal | FX Relevance |
|----------|--------|--------|-------------|
| VIX | ^VIX | Equity volatility | Risk-on/off regime proxy |
| US 10Y Yield | ^TNX | Rate expectations | Carry trade differential |
| US 13W T-Bill | ^IRX | Short-term rates | Risk-free rate proxy |
| Dollar Index | DX-Y.NYB | USD basket | Broad USD strength |
| Gold | GC=F | Commodity | Inverse USD signal |
| Crude Oil | CL=F | Commodity | Trade balance impact |
| S&P 500 | ^GSPC | US equity market | Capital flow signal |
| US Treasuries ETF | TLT | Long-duration bonds | Duration / risk appetite |
| High Yield ETF | HYG | Credit spreads | Credit risk appetite |

**Note:** The macro ticker list in the code uses `TLT` (iShares 20+ Year Treasury Bond ETF) and `HYG` (iShares iBoxx High Yield Corporate Bond ETF) rather than `^FVX` and `^N225`. These provide cleaner signals for credit/duration risk that directly affect FX carry dynamics.

### 1.3 Data Range & Granularity

- **Start:** 2005-01-01 (covers 21 years including GFC, recovery, multiple regimes)
- **End:** 2026-04-01
- **Frequency:** Daily (close-to-close)
- **Source:** Yahoo Finance via `yfinance` library
- **Rows:** ~5,300 trading days raw -> ~4,914 after warmup + NaN drop

### 1.4 Why These Instruments?

The instrument selection follows a principle of **maximum information per feature**: each instrument should capture a distinct economic driver of EUR/USD dynamics.

```
EUR/USD drivers mapped to instruments:

  Interest rate differential ──── ^TNX (US 10Y), ^IRX (US 3M), TLT (duration)
  Risk appetite ─────────────── ^VIX (equity vol), HYG (credit spreads)
  USD strength ──────────────── DX-Y.NYB (dollar index), ^GSPC (equity flows)
  Commodity flows ───────────── GC=F (gold, inverse USD), CL=F (oil, trade balance)
  Cross-currency dynamics ───── GBPUSD, USDJPY, USDCHF, EURGBP, EURJPY

  Total: 6 FX pairs + 9 macro = 15 instruments
```

## 2. Caching Strategy

```
data/
  ├── EURUSD_X_2005-01-01_2026-04-01.parquet
  ├── GBPUSD_X_2005-01-01_2026-04-01.parquet
  ├── ...
  └── DX-Y_NYB_2005-01-01_2026-04-01.parquet
```

- **Format:** Apache Parquet (columnar, compressed, fast I/O)
- **Naming:** `{safe_ticker}_{start}_{end}.parquet` where `safe_ticker` replaces `=`, `^`, `.` with `_`
- **Cache directory:** `data/` within project root
- **Hit/miss logging:** INFO-level messages for download vs cache load
- **Invalidation:** Change date range to force re-download

## 3. Feature Engineering Pipeline

### 3.1 Per-Pair Technical Features (13 per pair)

| # | Feature | Formula | Lookback | Category |
|---|---------|---------|----------|----------|
| 1 | log_ret_1d | log(close/close[-1]) | 1 day | Momentum |
| 2 | log_ret_5d | log(close/close[-5]) | 5 days | Momentum |
| 3 | log_ret_21d | log(close/close[-21]) | 21 days | Momentum |
| 4 | rvol_5d | std(log_ret_1d, 5) * sqrt(252) | 5 days | Volatility |
| 5 | rvol_21d | std(log_ret_1d, 21) * sqrt(252) | 21 days | Volatility |
| 6 | rvol_63d | std(log_ret_1d, 63) * sqrt(252) | 63 days | Volatility |
| 7 | rsi_14 | Wilder RSI(close, 14) | 14 days | Mean reversion |
| 8 | macd_line | EMA(12) - EMA(26) | 26 days | Trend |
| 9 | macd_signal | EMA(macd_line, 9) | 35 days | Trend |
| 10 | macd_hist | macd_line - macd_signal | 35 days | Trend |
| 11 | ohlc_range | (high - low) / close | 1 day | Microstructure |
| 12 | overnight_gap | (open - close[-1]) / close[-1] | 1 day | Microstructure |
| 13 | norm_true_range | max(H-L, |H-C[-1]|, |L-C[-1]|) / close | 1 day | Microstructure |

### 3.2 Cross-Pair Features (5 features)

| Feature | Formula | Lookback |
|---------|---------|----------|
| corr_EURUSD_GBPUSD_21d | rolling_corr(ret_1d, ret_1d, 21) | 21 days |
| corr_EURUSD_JPYUSD_21d | rolling_corr(ret_1d, ret_1d, 21) | 21 days |
| corr_EURUSD_CHFUSD_21d | rolling_corr(ret_1d, ret_1d, 21) | 21 days |
| corr_EURUSD_EURGBP_21d | rolling_corr(ret_1d, ret_1d, 21) | 21 days |
| corr_EURUSD_EURJPY_21d | rolling_corr(ret_1d, ret_1d, 21) | 21 days |

### 3.3 Macro Features (21 features)

**Per-ticker (2 each × 9 tickers = 18):**
- `{ticker}_level`: Close price (normalized by scaler)
- `{ticker}_ret_1d`: 1-day return

**Derived (3):**
- `yield_curve_slope`: TNX - IRX (10Y minus 3M Treasury spread)
- `VIX_5d_chg`: 5-day change in VIX level
- `DXY_rvol_21d`: 21-day rolling volatility of DXY returns

### 3.4 Feature Count Breakdown

| Category | Count | Formula |
|----------|-------|---------|
| Per-pair technical | 78 | 13 features x 6 pairs |
| Cross-pair correlations | 5 | 1 per secondary pair |
| Macro per-ticker | 18 | 2 features x 9 tickers |
| Macro derived | 3 | yield_curve_slope, VIX_5d_chg, DXY_rvol_21d |
| **Total** | **104** | |

### 3.5 Sample Data Shapes

```
After feature engineering and cleaning:

  Feature matrix:  DataFrame[4914 rows x 104 columns]
  Target matrix:   DataFrame[4914 rows x 2 columns]
  Date range:      2005-04-07 to 2026-03-28 (after 63-day warmup)
  Index type:      DatetimeIndex (business days only)

  Feature dtypes:  all float64
  Target dtypes:   float64 (fwd_ret_1d), float64 (fwd_ret_5d)
  NaN count:       0 (after cleaning)

After super-fold split:
  Train:  2478 rows (dates with ALL val/test windows + buffers removed)
  Val:    915 rows (union of 7 fold validation windows)
  Test:   1170 rows (union of 7 fold test windows)
  Total:  4563 (< 4914 because buffer rows are excluded from all three sets)
```

### 3.6 Warmup & Cleaning

- **WARMUP_PERIOD = 63 days:** Longest lookback is 63-day rolling volatility; first 63 rows are NaN-contaminated
- **NaN drop:** After warmup removal, remaining rows with any NaN are dropped
- **Inner join:** Features from different sources aligned on date intersection
- **Final shape:** ~4,914 rows x 104 columns (varies slightly with data availability)

## 4. Target Computation

```python
fwd_ret_1d = close.pct_change(1).shift(-1)   # next-day forward return
fwd_ret_5d = close.pct_change(5).shift(-5)   # 5-day forward return
```

- **Shift direction:** Negative shift = future data (used only as labels, never as features)
- **NaN handling:** Last 1 (or 5) rows become NaN and are dropped
- **Alignment:** Features and targets joined on common DatetimeIndex
- **Target distribution:** EUR/USD daily returns are approximately normal with fat tails (kurtosis ~5-8), zero mean, and standard deviation ~0.5% per day

```
Target statistics (EUR/USD fwd_ret_1d):
  Mean:     ~0.000  (near zero, as expected for efficient market)
  Std:      ~0.005  (0.5% daily volatility)
  Skew:     ~-0.1   (slight negative skew)
  Kurtosis: ~5-8    (fat tails -- motivates Huber loss over MSE)
  Min:      ~-0.03  (worst daily return: -3%)
  Max:      ~+0.03  (best daily return: +3%)
```

## 5. No-Leakage Guarantees

### 5.1 Feature-Level

| Property | Verification |
|----------|-------------|
| All features backward-looking | `test_all_features_backward_only` in test suite |
| No future data in feature computation | Manual audit of all rolling window calls |
| Targets use `.shift(-N)` (future shift) | Only used as labels, never as inputs |
| Warmup rows dropped | `WARMUP_PERIOD = 63` ensures no NaN features |

### 5.2 Split-Level

| Property | Verification |
|----------|-------------|
| 90-day purge gap (train→val, val→test) | `validate_purge_embargo()` at runtime |
| 21-day embargo gap (fold boundaries) | `validate_purge_embargo()` at runtime |
| Disjoint test sets across all folds | Pairwise overlap check in validation |
| Expanding training window | Verified in fold definitions (all start at 2005-01) |
| Cross-fold contamination prevention | `split_data()` punches out ALL val/test windows from every fold's training data |
| Label-horizon buffer (10 calendar days) | Excludes training samples whose `fwd_ret_5d` targets would peek into excluded windows |

### 5.3 Scaler-Level

| Property | Verification |
|----------|-------------|
| StandardScaler fit on train only | Code audit: `scaler.fit(train_feat.values)` |
| Val/test transformed (not fit) | Code audit: `scaler.transform(val_feat.values)` |
| Scaler isolated per fold | New scaler created each fold |
| Verified by test | `test_leakage.py::test_scaler_isolation` |

## 6. 7-Fold Regime-Aware Walk-Forward Splits

| Fold | Train | Val | Test | Regime |
|------|-------|-----|------|--------|
| 1 | 2005-01 → 2006-12 | 2007-04 → 2007-09 | 2008-01 → 2008-06 | GFC onset (28% vol) |
| 2 | 2005-01 → 2008-12 | 2009-04 → 2009-09 | 2010-01 → 2010-06 | Post-crash recovery |
| 3 | 2005-01 → 2011-12 | 2012-04 → 2012-09 | 2013-01 → 2013-06 | Eurozone debt plateau |
| 4 | 2005-01 → 2014-03 | 2014-07 → 2014-12 | 2015-04 → 2015-12 | Strong USD downturn |
| 5 | 2005-01 → 2017-12 | 2018-04 → 2018-09 | 2019-01 → 2019-09 | Low-vol plateau |
| 6 | 2005-01 → 2020-12 | 2021-04 → 2021-09 | 2022-01 → 2022-09 | EUR crisis downturn |
| 7 | 2005-01 → 2023-12 | 2024-04 → 2024-09 | 2025-01 → 2025-09 | Recent mixed/upturn |

### Design Rationale
- **Expanding window:** Later folds have more training data, reflecting real-world conditions
- **Regime diversity:** Model must perform across crisis, calm, trending, and reversal markets
- **No cherry-picking:** All 7 folds contribute equally to average Sharpe
- **Conservative purge:** 90 days > autocorrelation horizon of most FX features (~21 days)
- **Cross-fold hole-punching:** All val/test windows from ALL folds are excluded from every fold's training data. This prevents later folds' expanding training windows from training on earlier folds' held-out data.
- **Label-horizon buffer:** 10 calendar days (~7 business days) of training data before each excluded window are also removed, preventing `fwd_ret_5d` forward-return targets from peeking into excluded periods.

### Hole-Punching Visualization

```
Timeline (2005-2026):

Train data (fold 7, after hole-punching):
  ████████░░████░░████░░████░░░████░░████░░████░░████████
  2005    07  08  09  10  12  13 14 15  18  19  21  22 24 25

  ████ = training data available
  ░░░░ = removed (val/test window + 10-day buffer)

  Each gap removes:
    - 10-day label-horizon buffer (prevents fwd_ret_5d leakage)
    - 6-month validation window
    - 90-day purge gap
    - 6-9 month test window
```

### Super-fold vs. Per-fold Training

The super-fold design trains a single model on all available training data (with all val/test windows punched out), then evaluates on each fold's val/test windows separately:

```
Traditional 7-fold:                  Super-fold:
  Fold 1: train -> eval               Train ONCE on hole-punched data
  Fold 2: train -> eval                 |
  Fold 3: train -> eval                 |-> Eval on fold 1 test window
  Fold 4: train -> eval                 |-> Eval on fold 2 test window
  Fold 5: train -> eval                 |-> Eval on fold 3 test window
  Fold 6: train -> eval                 |-> ...
  Fold 7: train -> eval                 |-> Eval on fold 7 test window

  7 training runs (~4 min each)        1 training run (~36 sec for MLP)
  = ~28 minutes                        = ~36 seconds (47x faster)
```

## 7. Data Quality Checks

The following checks run automatically before every experiment:

| Check | Function | What It Verifies |
|-------|----------|-----------------|
| Purge/embargo gaps | `validate_purge_embargo()` | All 7 folds have >= 90-day purge, >= 21-day embargo |
| Split sizes | `split_superfold()` | train=2478, val=915, test=1170 |
| Zero overlap | `split_superfold()` | train/val overlap=0, train/test overlap=0, val/test overlap=0 |
| Contiguous segments | `create_contiguous_datasets()` | No sliding windows span date gaps |
| Feature backward-only | Test suite | All 104 features use only past data |
| Scaler isolation | Code structure | `scaler.fit()` only on training data |
| Label alignment | `compute_targets()` | Targets shifted by correct horizon |
| NaN absence | Post-cleaning assertion | Zero NaN in final feature/target matrices |

## 8. Feature Importance (from Champion Model)

While the residual MLP does not provide built-in feature importance (unlike tree-based models), the following observations come from ablation experiments and gradient analysis:

| Feature Group | Approximate Contribution | Evidence |
|---------------|-------------------------|----------|
| Macro signals (21 features) | High | Removing macro features degrades Sharpe by ~40% (experiments 15-18) |
| Cross-pair correlations (5) | Medium | Correlation regime shifts precede EUR/USD moves by 1-3 days |
| Per-pair momentum (18) | Medium | log_ret_5d and log_ret_21d capture trend persistence |
| Per-pair volatility (18) | Medium-High | rvol_21d is particularly important for regime detection |
| MACD signals (18) | Low-Medium | Signal mostly redundant with momentum features |
| Microstructure (18) | Low | Overnight gaps and ranges have weak daily predictive power |
| RSI (6) | Low | Mean-reversion signal is weak at daily frequency for FX |

**Key finding:** The yield curve slope (TNX - IRX) and VIX level are among the most important individual features. This aligns with the macroeconomic theory that interest rate differentials and risk appetite are the primary drivers of EUR/USD dynamics.

---

*See also:* [Project Overview](../architecture/project-overview.md) | [System Design](../architecture/system-design.md) | [Model Architecture](../architecture/model-architecture.md)
