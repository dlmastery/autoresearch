# Currency Prediction with LFM2.5 + Autoresearch Optimizer

**Date:** 2026-04-04
**Status:** Approved
**Origin:** Pivot from Colab-based GPT autoresearch to locally runnable CLI currency prediction

## Problem Statement

Build an ML system that predicts multi-horizon FX returns using Liquid AI's LFM2.5-350M foundation model as backbone, with an autoresearch optimizer loop that autonomously improves the model.

The system must have **zero data leakage**, **regime-diverse evaluation**, and a **clean metric** (average Sharpe across disjoint test sets) that is resistant to overfitting.

## Currency Pairs

Major + cross pairs (6 total):

| Pair | Ticker (yfinance) |
|------|-------------------|
| EUR/USD | EURUSD=X |
| GBP/USD | GBPUSD=X |
| USD/JPY | JPY=X |
| USD/CHF | CHF=X |
| EUR/GBP | EURGBP=X |
| EUR/JPY | EURJPY=X |

## Data Source

- **Primary:** Yahoo Finance via `yfinance` (daily OHLCV, no API key required)
- **Data range:** 2005-01-01 to present (~5500 trading days per pair)
- **Interface:** Abstracted so we can swap to OANDA/Alpha Vantage for intraday later

## Architecture

```
Raw OHLCV (6 pairs x 5 features)
        |
  +-------------+
  | Feature Eng  |  log returns, rolling vol, RSI, MACD
  | + Projection |  Linear(n_features -> 1024)
  +------+------+
         |  inputs_embeds (batch, seq_len, 1024)
  +------+------+
  |   LFM2.5    |  Lfm2Model from HuggingFace (350M params)
  |   Backbone  |  10 LIV conv blocks + 6 GQA blocks, frozen initially
  +------+------+
         |  last_hidden_state (batch, seq_len, 1024)
  +------+------+
  | Multi-Scale  |  Linear heads for 1d and 5d (1w) horizons
  | Pred Heads   |  Each outputs predicted returns per pair
  +-------------+
```

### Model Details

- **Backbone:** `LiquidAI/LFM2.5-350M-Base` loaded via `transformers.Lfm2Model`
  - hidden_size: 1024
  - num_hidden_layers: 16 (10 double-gated LIV convolution + 6 grouped query attention)
  - 354.5M parameters
  - Accepts `inputs_embeds` directly (bypasses text tokenizer)
  - Runs on CPU (float32) — no discrete GPU on target machine
- **Input projection:** Linear layer mapping feature vector to 1024-dim
- **Prediction heads:** Separate lightweight MLP per horizon (1d, 5d)
- **Prediction targets:** Forward returns at 1-day and 5-day horizons
  - 1h and 4h targets deferred until intraday data source added

### Why LFM2.5

- Released March 31, 2026 — latest state-of-the-art foundation model
- Built on Liquid Neural Networks with adaptive time constants — inherently suited for non-stationary sequential data like FX
- Hybrid architecture (LIV convolutions + GQA attention) captures both local patterns and long-range dependencies
- 350M small enough for CPU fine-tuning; `inputs_embeds` pathway verified working
- The autoresearch optimizer can explore: unfreezing layers, swapping to 1.2B, adding ms-Mamba blocks, etc.

## Feature Engineering (~55 features)

All features are **strictly backward-looking** (no future data leakage).

### Per-Pair Technical Features (13 features x 6 pairs = 78 raw, deduplicated ~70)

| Feature | Lookback | Signal |
|---------|----------|--------|
| Log returns | 1d, 5d, 21d | Momentum at multiple scales |
| Rolling volatility | 5d, 21d, 63d | Volatility regimes |
| RSI | 14d | Overbought/oversold |
| MACD, signal, histogram | 12/26/9 EMA | Trend direction + acceleration |
| OHLC range ratio | 1d | Intraday volatility |
| Overnight gap | 1d | Session open vs prior close |
| True range (normalized) | 1d | Volatility inclusive of gaps |

### Cross-Pair Features (5 features)

- Rolling 21d correlation between EUR/USD and each of the other 5 pairs
- Captures regime structure (e.g., EUR/USD and EUR/GBP correlation shifts during EUR-specific events)

### Macro Signal Features (~15 features)

| Source | Ticker | Features |
|--------|--------|----------|
| US 10Y Yield | ^TNX | Level, 1d return |
| US 5Y Yield | ^FVX | Level, 1d return |
| US 13W T-Bill | ^IRX | Level, 1d return |
| Yield curve slope | ^TNX - ^IRX | 10Y minus 3M spread |
| VIX | ^VIX | Level, 1d return, 5d change |
| S&P 500 | ^GSPC | 1d return, level |
| Nikkei 225 | ^N225 | 1d return, level |
| Gold | GC=F | 1d return, level |
| Crude Oil | CL=F | 1d return, level |
| US Dollar Index | DX-Y.NYB | 1d return, level, 21d vol |

### Why These Macro Signals

- **Yield differentials** are the #1 macro driver of FX (carry trade)
- **VIX** signals risk-on/off regimes — safe-haven currencies (USD, JPY, CHF) strengthen when VIX spikes
- **DXY** summarizes USD strength against a basket including pairs we don't have (CAD, SEK)
- **Gold** moves inversely to USD; oil impacts trade-balance-sensitive currencies
- **Equities** (S&P, Nikkei) proxy for risk appetite and capital flows

Normalization: fit StandardScaler on training data only, apply to val/test.

## 7-Fold Regime-Aware Data Splits

Designed from empirical EUR/USD regime analysis (2005-2026). Each test set targets a distinct market regime. **3-month (63 trading day) purge gaps** between every adjacent train/val/test boundary.

| Fold | Train Period | Val Period | Test Period | Market Regime Tested |
|------|-------------|------------|-------------|---------------------|
| 1 | 2005-01 to 2006-12 | 2007-04 to 2007-09 | 2008-01 to 2008-06 | Pre-crisis upturn + GFC onset (vol 28%) |
| 2 | 2005-01 to 2008-12 | 2009-04 to 2009-09 | 2010-01 to 2010-06 | Post-crash recovery (-10% then +9%) |
| 3 | 2005-01 to 2011-12 | 2012-04 to 2012-09 | 2013-01 to 2013-06 | Eurozone debt plateau (low vol) |
| 4 | 2005-01 to 2014-03 | 2014-07 to 2014-12 | 2015-04 to 2015-12 | Strong USD downturn (-10.5% qtr) |
| 5 | 2005-01 to 2017-12 | 2018-04 to 2018-09 | 2019-01 to 2019-09 | Low-vol plateau (vol 4-6%) |
| 6 | 2005-01 to 2020-12 | 2021-04 to 2021-09 | 2022-01 to 2022-09 | EUR crisis downturn (-12% cumulative) |
| 7 | 2005-01 to 2023-12 | 2024-04 to 2024-09 | 2025-01 to 2025-09 | Recent mixed/upturn (+8.4% Q2) |

### No-Leakage Guarantees

1. **Temporal purge:** 3-month gap between every train/val and val/test boundary
2. **Feature isolation:** All rolling windows computed backward-only; no feature uses future data
3. **Normalization isolation:** Scaler fit on train set only, applied (not refit) on val/test
4. **Disjoint test sets:** No overlap between any two test periods
5. **Walk-forward:** Each fold's train set includes all data before the purge gap
6. **Metric aggregation:** Optimizer sees average Sharpe across ALL 7 test folds — cannot overfit to one regime

## Evaluation Metric

**Average annualized Sharpe ratio** across all 7 test folds.

```
Sharpe_fold_i = mean(daily_returns_i) / std(daily_returns_i) * sqrt(252)
Final_metric = mean(Sharpe_fold_1, ..., Sharpe_fold_7)
```

Using Sharpe rather than raw return or MSE because:
- Risk-adjusted — penalizes volatile predictions
- Hard to game — can't inflate by increasing position sizes
- Industry-standard for trading strategy evaluation

## Project Structure

```
autoresearch/
  data/
    download.py         # yfinance download + caching
    features.py         # feature engineering (backward-only)
    splits.py           # 7-fold regime-aware split definitions
  model/
    backbone.py         # LFM2.5 wrapper + input projection
    heads.py            # multi-scale prediction heads
    train.py            # training loop
  evaluation/
    metrics.py          # Sharpe calculation per fold
    leakage_check.py    # automated leakage detection tests
  optimizer/
    agent_loop.py       # autoresearch Claude API agent
    prompts.py          # brainstorm + modification prompts
  tests/
    test_features.py    # feature backward-only verification
    test_splits.py      # split disjointness + gap verification
    test_leakage.py     # end-to-end leakage detection
    test_model.py       # model forward pass verification
  baseline.py           # train + evaluate full baseline
  run_optimizer.py      # launch autoresearch loop
  requirements.txt
```

## Optimizer Loop (Autoresearch)

Ported from the Colab notebook, adapted for CLI:

1. Run baseline: train on fold train sets, evaluate average Sharpe across 7 test sets
2. Claude API agent brainstorms 3 experiment ideas given current code + past results
3. Agent selects and implements one modification (to model/, data/, or heads)
4. Syntax check the modified code
5. Train and evaluate on all 7 folds
6. If average Sharpe improves: keep changes. Else: revert to previous best.
7. Log result to experiment tracker. Repeat.

### What the Optimizer Can Explore

- Unfreeze LFM2.5 layers progressively
- Swap backbone to LFM2.5-1.2B or add ms-Mamba multi-scale blocks
- Add ChromaDB-based regime retrieval (store historical pattern embeddings)
- Modify feature engineering (add/remove features, change windows)
- Change head architecture (MLP depth, dropout, skip connections)
- Learning rate schedules, optimizers, regularization
- Ensemble methods across folds

## Verified Dependencies

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.11.9 | Installed |
| torch | 2.5.1+cu121 | Installed (CPU mode) |
| transformers | 5.5.0 | Installed |
| Lfm2Model | from transformers | Verified: loads, forward pass works, inputs_embeds works |
| yfinance | latest | Installed, data download verified |
| pandas | latest | Installed |
| numpy | latest | Installed |

## Hardware Constraints

- **CPU only:** Intel Iris Xe Graphics (no discrete GPU)
- **Implication:** Keep batch sizes small, sequence lengths manageable (~60-120 days)
- **LFM2.5-350M on CPU:** Forward pass works in float32, training will be slow but feasible
- **Optimizer consideration:** Each experiment iteration will take minutes, not seconds
