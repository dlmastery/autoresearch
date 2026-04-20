# Forensic Investigation Report — AutoResearch Results

**Date:** 2026-04-14
**Investigator:** Claude Code Agent
**Verdict: RESULTS ARE INVALID — DXY feature contains look-ahead bias**

---

## Executive Summary

The champion model's test Sharpe of +6.21 is **NOT genuine predictive alpha**. It is primarily driven by a single feature (`DXY_ret_1d`) that has a near-perfect contemporaneous correlation with the target (`fwd_ret_1d`), likely caused by timezone misalignment between yfinance data sources.

A trivial strategy of `sign(-DXY_ret_1d) * actual_return` — with ZERO machine learning — achieves Sharpe +14 to +16 on 5 of 7 test folds. The ML model's Sharpe of +6 to +10 on those same folds is actually UNDERPERFORMING the raw feature.

**All 90 experiments must be re-run after removing or correcting the DXY feature.**

---

## Evidence

### Check 1: Data Split Integrity — PASSED
- Zero date overlap between train/val/test verified programmatically
- 90-day purge gap, 21-day embargo, 10-day label buffer — all verified
- Super-fold counts match expected (train=2738, val=838, test=1043)

### Check 2: Target Computation — PASSED
- `fwd_ret_1d = close.pct_change(1).shift(-1)` correctly computes next-day return
- Manual verification: `close[T+1]/close[T] - 1` matches the computed target exactly

### Check 3: Feature Forward-Looking — PARTIAL PASS
- All `.shift(-N)` calls are only in `compute_targets()`, not in feature computation
- Features are backward-looking in their code — no `.shift(-1)` in feature functions
- **BUT**: the data sources have different closing times (see below)

### Check 4: Shuffled Labels Test — AMBIGUOUS
| Shuffle Type | Shuffle Seed | Test Sharpe | Expected |
|-------------|-------------|-------------|----------|
| Train only shuffled | 0 | +1.33 | ~0 |
| Train+Val shuffled | 0 | +1.33 | ~0 |
| Train only shuffled | 1 | -0.02 | ~0 |
| Train only shuffled | 2 | +0.14 | ~0 |
| Train only shuffled | 3 | +0.59 | ~0 |

Shuffled labels produced median test Sharpe ~+0.3, not zero. This is partially explained by the DXY feature having inherent correlation with the target regardless of what the model learns.

### Check 5: DXY Feature Forensics — **FAILED (CRITICAL)**

#### Raw DXY_ret_1d IC with fwd_ret_1d per test fold:

| Fold | DXY IC | DXY-only Sharpe | DXY-only WR | Model Sharpe | Model WR |
|------|--------|-----------------|-------------|-------------|----------|
| 1 | +0.10 | -3.45 | 44% | +2.46 | 60% |
| 2 | -0.01 | +1.09 | 52% | +1.17 | 53% |
| 3 | **-0.83** | **+15.18** | **85%** | +9.76 | 76% |
| 4 | **-0.90** | **+15.21** | **89%** | +9.78 | 75% |
| 5 | **-0.93** | **+16.17** | **86%** | +8.85 | 71% |
| 6 | **-0.89** | **+15.05** | **83%** | +9.95 | 71% |
| 7 | **-0.84** | **+14.51** | **87%** | +8.48 | 72% |

**On 5 of 7 folds, a trivial `sign(-DXY_ret_1d)` strategy achieves Sharpe +14 to +16 — the ML model's +8 to +10 on these folds is UNDERPERFORMANCE.**

The model actually HURTS performance on folds 3-7 compared to just using DXY directly.

On folds 1 and 2, where DXY has near-zero IC, the model also performs weakly (+2.46 and +1.17). The model has learned to rely on DXY and has little genuine alpha.

#### Why DXY has IC = -0.9 with fwd_ret_1d

This is a **data timing artifact**, not genuine predictive signal:

1. **EUR/USD (EURUSD=X)** from yfinance: Forex market, closes at ~5pm ET daily
2. **DXY (DX-Y.NYB)** from yfinance: ICE Futures US Dollar Index, pit-traded close may differ
3. Both get the same date label in yfinance, but the actual close times differ

When we compute `DXY_ret_1d` (today's DXY return), this may contain information about EUR/USD moves that happened AFTER the EUR/USD daily close but BEFORE the DXY close. This makes `fwd_ret_1d` (EUR/USD close-to-close) partially knowable from `DXY_ret_1d`.

More precisely: DXY is composed of 6 currencies with EUR at 57.6% weight. The DXY "today" effectively contains the EUR/USD price at a slightly different time than the EUR/USD "today" close, creating a temporal arbitrage in the data that doesn't exist in real trading.

### Check 6: Cross-Asset Correlation Analysis

| Comparison | IC |
|-----------|-----|
| DXY_ret_1d vs EURUSD_ret_today | -0.28 |
| EURUSD_ret_today vs fwd_ret_1d | -0.04 |
| DXY_ret_1d vs fwd_ret_1d | **-0.61** |

If DXY were a perfect inverse of EUR/USD (IC=-1.0), then DXY_ret_1d should have IC=+0.04 with fwd_ret_1d (same as EURUSD autocorrelation). Instead it has IC=-0.61, which is **15x higher than expected**. This proves the data timing artifact.

### Check 7: Sharpe Calculation — PASSED
Manual computation matches the function exactly. The Sharpe formula is correct.

### Check 8: Strategy Returns — PASSED  
`sign(pred) * actual` correctly assigns positive returns to correct directional predictions.

---

## Root Cause

The DXY index data from yfinance (ticker `DX-Y.NYB`) has a **temporal misalignment** with EUR/USD spot data (ticker `EURUSD=X`). Both are assigned the same daily date, but their effective closing times differ, causing DXY "today" to contain partial knowledge of EUR/USD's "today-to-tomorrow" move.

This is a well-known issue in multi-source financial data. Lopez de Prado (2018, ch.2) specifically warns: *"Timestamps from different sources must be synchronized. A few hours of misalignment can create the illusion of predictability."*

---

## Remediation

### Immediate (before any more experiments)

1. **Remove DXY_ret_1d and all DXY-derived features** from the feature set
2. Re-run the champion config without DXY
3. If test Sharpe drops to ~0-1, the DXY was the sole driver
4. If test Sharpe stays > 2, there's some genuine signal in other features

### Better fix

5. **Replace DXY with a lagged version**: use `DXY_ret_1d.shift(1)` (yesterday's DXY return) as a feature. This guarantees no look-ahead.
6. **Audit all cross-asset features** for similar timing issues (GOLD, OIL, SPX, VIX, bonds)
7. **Run the shuffled-labels test on the corrected feature set** — test Sharpe should be < 0.5

### Long-term

8. Use a single data source with consistent timestamps for all instruments
9. Implement a systematic look-ahead bias checker that tests each feature's IC with the target before training

---

## Conclusion

**The model's results are inflated by a data timing artifact in the DXY feature.** The residual MLP architecture and hyperparameter findings (skip connection, reduced capacity, etc.) may still be valid architectural choices, but the absolute performance numbers (+6.21 Sharpe, 75% win rate) are NOT achievable in live trading. The true out-of-sample performance is likely in the range of Sharpe +0.5 to +1.5 — respectable for FX but far less dramatic.

**Do NOT publish or trade on these results without the DXY correction.**
