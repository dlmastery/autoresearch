# Forensic Investigation Checkpoint

## Status: INCONCLUSIVE — needs hourly data verification

## What we know so far

### Daily data timing (from hourly matching)
- **DXY daily close** matches hourly bar at ~18:00 UTC (2pm ET) on the labeled date
- **EUR/USD daily close** matches hourly bar at ~00:00 UTC next day (8pm ET on labeled date - 1)
- There is a 6-hour gap on same date label: DXY closes at 2pm ET, EUR/USD closes at 8pm ET

### Correlation structure (daily data)
| Test | IC | Interpretation |
|------|-----|---------------|
| DXY_ret(T) vs EUR_ret(T) | -0.29 | Partial overlap (~18h shared) |
| DXY_ret(T) vs EUR_fwd(T) | -0.60 | Either leakage OR momentum |
| DXY_ret(T-1) vs EUR_ret(T) | -0.59 | Lagged = true same-day correlation |
| DXY_ret(T-1) vs EUR_fwd(T) | +0.01 | Lagged = no forward prediction |

### Two competing hypotheses

**H1: Date labeling offset (leakage)**
yfinance labels DXY one day ahead of EUR/USD. DXY "date T" close actually corresponds to the same physical time as EUR/USD "date T+1" open. If so, DXY_ret(T) contains information about the period that includes EUR/USD's "date T+1" window. The -0.60 IC is mechanical, not predictive.

Evidence FOR: lagging DXY by 1 day perfectly restores expected IC (-0.59 same-day, +0.01 forward).

**H2: Genuine momentum (alpha)**  
DXY closes 6 hours before EUR/USD daily close. Dollar moves from 2pm ET carry into EUR/USD moves over the next 6-30 hours. The -0.60 IC is real predictive signal from dollar momentum. Tradeable: enter at 2pm ET after observing DXY move, hold 24h.

Evidence FOR: DXY physically closes before EUR/USD. No actual future data is used.

### What would settle this
Hourly resolution analysis: compute IC(DXY_ret_hour, EUR_USD_ret_next_hours) at every lag from 0h to 48h. If the IC peaks at lag=0 (same hour) and decays smoothly, it's momentum. If the IC peaks at a specific offset (e.g., 6h or 24h) that matches the date-label shift, it's an artifact.

## Old methodology (preserved)
All 90 experiments used DXY_ret_1d as a feature WITHOUT any lag. If H1 is correct, all results are inflated. If H2 is correct, results may be genuine but overly reliant on a single macro feature.

## Champion config (preserved regardless of forensic outcome)
Residual MLP, hidden=128, head=64, lr=5e-4, bs=32, seq=10, ep=50, wd=1e-5, pat=10, hd=0.15, huber=0.5, seed=0
Composite +5.50, Test Sharpe +6.21, 7/7 positive folds
