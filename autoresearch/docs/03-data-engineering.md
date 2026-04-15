# 03 - Data Engineering

**SWEBoK Knowledge Area:** KA2 — Software Design (Data Design)
**Google SWE Reference:** Ch. 9 — "Code Review" (data pipeline correctness)

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
| US 10Y Yield | ^TNX | Rate expectations | Carry trade differential |
| US 5Y Yield | ^FVX | Mid-curve rates | Policy rate expectations |
| US 13W T-Bill | ^IRX | Short-term rates | Risk-free rate proxy |
| VIX | ^VIX | Equity volatility | Risk-on/off regime proxy |
| S&P 500 | ^GSPC | US equity market | Capital flow signal |
| Nikkei 225 | ^N225 | Japan equity market | JPY correlation |
| Gold | GC=F | Commodity | Inverse USD signal |
| Crude Oil | CL=F | Commodity | Trade balance impact |
| Dollar Index | DX-Y.NYB | USD basket | Broad USD strength |

### 1.3 Data Range & Granularity

- **Start:** 2005-01-01 (covers 21 years including GFC, recovery, multiple regimes)
- **End:** 2026-04-01
- **Frequency:** Daily (close-to-close)
- **Source:** Yahoo Finance via `yfinance` library
- **Rows:** ~5,300 trading days raw → ~4,914 after warmup + NaN drop

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

### 3.4 Warmup & Cleaning

- **WARMUP_PERIOD = 63 days:** Longest lookback is 63-day rolling volatility; first 63 rows are NaN-contaminated
- **NaN drop:** After warmup removal, remaining rows with any NaN are dropped
- **Inner join:** Features from different sources aligned on date intersection
- **Final shape:** ~4,914 rows × 104 columns (varies slightly with data availability)

## 4. Target Computation

```python
fwd_ret_1d = close.pct_change(1).shift(-1)   # next-day forward return
fwd_ret_5d = close.pct_change(5).shift(-5)   # 5-day forward return
```

- **Shift direction:** Negative shift = future data (used only as labels, never as features)
- **NaN handling:** Last 1 (or 5) rows become NaN and are dropped
- **Alignment:** Features and targets joined on common DatetimeIndex

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
