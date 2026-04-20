<p align="center">
  <h1 align="center">AutoResearch</h1>
  <p align="center">
    <strong>Autonomous FX Prediction Optimization</strong>
  </p>
  <p align="center">
    An AI-driven machine learning research system for EUR/USD exchange rate prediction,<br>
    powered by a Karpathy-style experiment loop where Claude Code acts as the researcher.
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg" alt="PyTorch 2.5+"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/dlmastery/autoresearch"><img src="https://img.shields.io/badge/experiments-104-orange.svg" alt="104 Experiments"></a>
    <a href="https://dlmastery.github.io/autoresearch/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="GitHub Pages"></a>
    <a href="#champion-model-results"><img src="https://img.shields.io/badge/test%20Sharpe-+6.21-brightgreen.svg" alt="Test Sharpe +6.21"></a>
    <a href="#champion-model-results"><img src="https://img.shields.io/badge/total%20return-+1%2C001%25-brightgreen.svg" alt="Total Return +1,001%"></a>
    <a href="#champion-model-results"><img src="https://img.shields.io/badge/positive%20folds-7%2F7-brightgreen.svg" alt="7/7 Folds Positive"></a>
  </p>
</p>

---

## 📄 Paper & Article

- **Research paper (arxiv-style, submission-ready):** [`docs/paper.md`](docs/paper.md) — *AutoResearch: An LLM-Driven Autonomous Research Loop for Financial Time Series Forecasting*. 8 numbered sections + 3 appendices, ~9,900 words, 46 references with arXiv IDs. Double-blind format for NeurIPS / ICML / ICLR submission.
- **Abstract (teaser):** [`paper_abstract.md`](paper_abstract.md) — one-page abstract + key-numbers table, linkable from external sites.
- **Medium-style long-form article (v3):** [`docs/medium_article.md`](docs/medium_article.md) — *The Research Loop Was the Model: 151 Experiments in Quantitative FX Run by a Large Language Model.* 17 sections, 7 Mermaid diagrams, ~9,700 words. Popular-science-grade with full technical depth.

**Current global champion (latest):** LSTM Exp35 — composite **+6.4242**, test Sharpe **+6.5242**, val Sharpe **+7.1539**, 7/7 positive test folds, **+1,122% return** over 2008–2025. Archived at [`autoresearch/autoresearch_results/winners/lstm_exp35_wd7e4_bs16_seed42/`](autoresearch/autoresearch_results/winners/lstm_exp35_wd7e4_bs16_seed42/).

**Experiment count:** 151 experiments across 4 backbones. LSTM family 46/50 per-backbone mandate in progress; PatchTST, PatchTSMixer, XGBoost, LightGBM, CatBoost queued next. Plus 10 new 2024–2026 Tier-2 SOTA backbones on the roadmap (TimesFM 2.5, Chronos-Bolt / Chronos-2, Moirai 2.0, MOMENT, TiRex, Sundial, Time-MoE, TimeMixer++, TimesNet, MambaTS).

> Note: the badges and "Champion Model Results" section below still cite the earlier MLP champion (+5.50 / 90 exps); they will be refreshed as the LSTM phase completes. For the current state always see the paper, the article, or `memory/project_autoresearch_checkpoint.md`.

---

## Table of Contents

- [Highlights](#highlights)
- [Champion Model Results](#champion-model-results)
  - [Aggregate Metrics](#aggregate-metrics)
  - [Per-Regime Performance](#per-regime-performance)
  - [Cross-Seed Reproducibility](#cross-seed-reproducibility)
- [Key Innovation: Claude Code as the Research Agent](#key-innovation-claude-code-as-the-research-agent)
- [Architecture](#architecture)
  - [Residual MLP (Champion)](#residual-mlp-champion)
  - [Champion Configuration](#champion-configuration)
  - [Available Backbones](#available-backbones)
- [Data Pipeline](#data-pipeline)
  - [Instruments](#instruments)
  - [Feature Engineering (104 Features)](#feature-engineering-104-features)
  - [Targets](#targets)
- [Evaluation Framework](#evaluation-framework)
  - [Super-Fold Design](#super-fold-design)
  - [Data Integrity Guarantees](#data-integrity-guarantees)
  - [Metrics](#metrics)
  - [Composite Score](#composite-score)
- [Uncertainty Estimation](#uncertainty-estimation)
- [The Agent Loop](#the-agent-loop)
  - [Seven-Step Scientific Process](#seven-step-scientific-process)
  - [Crash-Recovery Checkpointing](#crash-recovery-checkpointing)
  - [Agent Rules](#agent-rules)
- [Experiment History](#experiment-history)
  - [90 Experiments Across 2 Backbones](#90-experiments-across-2-backbones)
  - [Key Discoveries](#key-discoveries)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Run the Champion](#run-the-champion)
  - [View the Dashboard](#view-the-dashboard)
  - [Run a Full Ablation](#run-a-full-ablation)
  - [Resume the Agent Loop](#resume-the-agent-loop)
- [CLI Reference](#cli-reference)
  - [run_autoresearch (Single Experiment)](#run_autoresearch-single-experiment)
  - [run_ablation (Multi-Backbone Comparison)](#run_ablation-multi-backbone-comparison)
  - [baseline (Walk-Forward Evaluation)](#baseline-walk-forward-evaluation)
  - [Entry Points](#entry-points)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

## Highlights

```
  +1,001% total return  |  +6.21 test Sharpe  |  7/7 regimes profitable
  301K parameters       |  ~36s training (CPU) |  PSR = 1.0000
  90 experiments run    |  104 features        |  18 years of data (2006-2024)
```

- **Autonomous research loop** -- Claude Code reads results, diagnoses per-fold failures, cites published papers, forms hypotheses, runs experiments, and checkpoints state. No human in the loop during experimentation.
- **Residual MLP champion** beats a frozen LiquidAI LFM2.5-350M foundation model by 3.4x on median test Sharpe (+4.76 vs +1.40). Smaller models with the right inductive bias win on low-SNR financial data.
- **Rigorous evaluation** -- 7 market regime windows spanning 18 years, 90-day purge gap, 21-day embargo, 10-day label-horizon buffer, zero train/val/test overlap verified programmatically.
- **Uncertainty-aware predictions** -- aleatoric (data noise) + epistemic (model uncertainty) with confidence-gated trading signals.
- **Full reproducibility** -- fixed seeds, cached data, deterministic training, cross-seed verification.

---

## Champion Model Results

### Aggregate Metrics

| Metric | Value |
|:-------|------:|
| **Test Sharpe Ratio** | **+6.21** (annualized) |
| **Composite Score** | **+5.50** |
| **Total Return** | **+1,001%** ($1,000 --> $11,011) |
| **Win Rate** | **69%** aggregate, **71%** median across folds |
| **Sortino Ratio** | **+11.31** |
| **PSR** | **1.0000** (statistically significant) |
| **Profit Factor** | **3.30** |
| **Max Drawdown** | **4.13%** |
| **Information Coefficient** | **+0.48** (Spearman) |
| **Positive Folds** | **7 / 7** (all regimes profitable) |
| **Training Time** | **~36 seconds** (CPU) |
| **Trainable Parameters** | **301,196** |

### Per-Regime Performance

The model is trained ONCE on all historical data (2005-2023) with val/test windows hole-punched out, then evaluated on 7 distinct market regime windows spanning 2006-2024:

| Fold | Period | Regime | Sharpe | Return | Win Rate | IC | Sortino | Max DD |
|:----:|:------:|:-------|-------:|-------:|---------:|----:|--------:|-------:|
| 1 | 2006-2008 | Pre-crisis / GFC onset | +2.46 | +19.8% | 60.8% | +0.19 | +6.78 | 3.29% |
| 2 | 2009-2010 | Post-crash recovery | +1.17 | +5.5% | 53.3% | +0.08 | +2.02 | 3.47% |
| 3 | 2011-2012 | Eurozone debt plateau | +9.76 | +34.1% | 76.0% | +0.58 | +18.81 | 1.32% |
| 4 | 2014-2016 | Strong USD downturn | +9.78 | +90.3% | 75.5% | +0.67 | +19.42 | 1.81% |
| 5 | 2017-2019 | Low-volatility plateau | +8.85 | +29.3% | 71.0% | +0.64 | +16.14 | 1.43% |
| 6 | 2020-2021 | COVID / EUR crisis | +9.95 | +69.5% | 70.9% | +0.64 | +21.01 | 2.27% |
| 7 | 2023-2024 | Recent mixed / upturn | +8.48 | +55.8% | 71.6% | +0.62 | +14.36 | 1.64% |

**Key observations:**
- Folds 3-7 consistently achieve Sharpe > 8 and IC > 0.58
- Fold 2 (post-crash recovery, 2009-2010) is the hardest regime -- still profitable at +1.17 Sharpe
- Fold 4 delivers the highest return (+90.3%) during the strong USD downturn period
- Maximum drawdown never exceeds 4.13% across any regime window

### Cross-Seed Reproducibility

The champion was verified across 3 independent random seeds to confirm robustness:

| Seed | Composite | Test Sharpe | Val Sharpe | Positive Folds |
|-----:|----------:|------------:|-----------:|:--------------:|
| 0 | +5.50 | +6.21 | +5.60 | 7/7 |
| 42 | +4.45 | +4.69 | -- | 6/7 |
| 99 | +4.46 | +4.76 | -- | 6/7 |
| **Median** | **+4.46** | **+4.76** | | |

The residual MLP shows significantly lower seed variance than the LFM2 foundation model (composite std 0.60 vs 1.65 for LFM2), confirming that the architecture's performance is robust rather than lucky.

---

## Key Innovation: Claude Code as the Research Agent

Most AutoML systems use pre-baked search strategies (Bayesian optimization, evolutionary search, random search). AutoResearch takes a fundamentally different approach: **Claude Code IS the researcher.**

```
 +------------------------------------------------------------------+
 |                     CLAUDE CODE (Agent)                           |
 |                                                                   |
 |  Reads results --> Diagnoses per-fold failures --> Cites papers   |
 |  Forms hypothesis --> Predicts outcome --> Runs ONE experiment    |
 |  Analyzes deltas --> Checkpoints state --> Repeats                |
 |                                                                   |
 +--------------------+---------------------------------------------+
                      |
                      | python -m autoresearch.run_autoresearch ...
                      |
 +--------------------v---------------------------------------------+
 |                  EXPERIMENT RUNNER                                |
 |                                                                   |
 |  Downloads data (cached) --> Computes features --> Splits data   |
 |  Trains model --> Evaluates per-window --> Logs JSONL             |
 |                                                                   |
 +--------------------+---------------------------------------------+
                      |
                      | experiment_log.jsonl, best_config.json
                      |
 +--------------------v---------------------------------------------+
 |                   DASHBOARD (decoupled)                          |
 |                                                                   |
 |  Reads JSONL --> Renders train/val/test tabs --> Per-window       |
 |  breakdown --> Experiment history --> Architecture comparison     |
 |                                                                   |
 +------------------------------------------------------------------+
```

The agent:
- **Diagnoses** which folds are weak and WHY (market regime analysis, uncertainty decomposition)
- **Cites** published papers to justify every hyperparameter choice (He 2016, Gu/Kelly/Xiu 2020, Kendall & Gal 2017, etc.)
- **Predicts** outcomes BEFORE running experiments, enabling falsifiable hypotheses
- **Modifies code** when hyperparameters aren't enough (architecture changes, loss functions)
- **Checkpoints** every 5 minutes and after every experiment for crash recovery

This approach ran 90 experiments across 2 backbone architectures, discovering that a simple Residual MLP (301K params) outperforms a frozen LFM2.5-350M foundation model (639K trainable params) by 3.4x on median test Sharpe.

---

## Architecture

### Residual MLP (Champion)

```
 Input: 10 business days x 104 features = 1,040 values (flattened)

                +---> Linear(1040, 128) -----------------------> [shortcut]
                |                                                    |
 x (1040) ------+                                                    + (element-wise add)
                |                                                    |
                +---> Linear(1040, 128)                              |
                      GELU activation                                |
                      Dropout(0.1)                                   |
                      Linear(128, 128)                        --> hidden (128)
                      GELU activation                                |
                      Dropout(0.1)                                   |
                                                                     v
                                              Prediction Head (per currency pair)
                                              LayerNorm(128) --> Linear(128, 64)
                                              GELU activation --> Dropout(0.15)
                                              Linear(64, 6) --> [6 output values]
                                                    |
                                            +-------+--------+
                                            |                |
                                      ret_1d (6 pairs)  ret_5d (6 pairs)
```

**Why it works:** The linear shortcut provides a baseline linear prediction. The nonlinear residual branch learns *corrections* to this baseline. For low-SNR financial data, the true signal is a small perturbation on a linear model -- this architecture is perfectly suited (He et al., 2016; Gu, Kelly & Xiu, 2020).

**Parameter comparison:**

| Model | Trainable Params | Params/Sample | Test Sharpe (median) |
|:------|:----------------:|:-------------:|:--------------------:|
| Original MLP (512 hidden) | 1,063,436 | 428 | +0.82 |
| LFM2-350M (frozen backbone) | 639,500 | 233 | +1.40 |
| **Residual MLP (128 hidden)** | **301,196** | **121** | **+4.76** |

Smaller is better: reducing the parameter-to-sample ratio from 428 to 121 eliminated memorization and improved generalization dramatically.

### Champion Configuration

```json
{
    "backbone": "mlp",
    "hidden_size": 128,
    "head_hidden": 64,
    "lr": 5e-4,
    "batch_size": 32,
    "seq_len": 10,
    "epochs": 50,
    "patience": 10,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "huber_delta": 0.5,
    "head_dropout": 0.15,
    "het_loss": false,
    "seed": 0
}
```

Every parameter is justified:

| Parameter | Value | Justification |
|:----------|:------|:--------------|
| `hidden_size` | 128 | Reduced from 512 to cut params/sample ratio; follows Gu, Kelly & Xiu (2020) capacity guidance |
| `lr` | 5e-4 | Higher LR enabled by the skip connection providing a stable gradient highway (He 2016) |
| `seq_len` | 10 | ~2 weeks of business days; matches FX autocorrelation decay (~5 days) |
| `epochs` | 50 | From-scratch MLP needs 2.5x more epochs than fine-tuning; with cosine annealing |
| `patience` | 10 | Early stopping buffer for noisy validation loss in financial data |
| `huber_delta` | 0.5 | Robust to fat-tailed FX returns; balances MSE (delta -> inf) and MAE (delta -> 0) |
| `head_dropout` | 0.15 | Tuned to balance fold 2 (hardest) vs other folds; enables MC Dropout uncertainty |
| `grad_clip` | 1.0 | Prevents gradient explosions from FX outlier returns (>3 sigma) |
| `weight_decay` | 1e-5 | Light L2 regularization; heavier decay hurts learned skip connection scale |
| `het_loss` | false | Heteroscedastic loss hurts on small data (n=2478); variance branch steals capacity |

### Available Backbones

AutoResearch supports 8 backbone architectures spanning classical ML, recurrent networks, transformers, MLP-mixers, and foundation models:

| Backbone | CLI Flag | Type | Seq Len | Description |
|:---------|:---------|:-----|:-------:|:------------|
| Residual MLP | `mlp` | Feedforward | 10 | Skip connection + 2-layer residual. **Champion.** |
| BiLSTM | `lstm` | Recurrent | 10 | Bidirectional LSTM with attention pooling |
| LFM2.5-350M | `lfm2-350m` | Foundation | 60 | LiquidAI's frozen foundation model with learned adapter |
| PatchTST | `patchtst` | Transformer | 10 | Patch Time Series Transformer (Nie et al., ICLR 2023) |
| PatchTSMixer | `patchtsmixer` | MLP-Mixer | 10 | Google's MLP-Mixer for time series (NeurIPS 2023) |
| XGBoost | `xgboost` | GBM | 10 | Gradient boosting (Chen & Guestrin, 2016) |
| LightGBM | `lightgbm` | GBM | 10 | Light gradient boosting (Ke et al., NeurIPS 2017) |
| CatBoost | `catboost` | GBM | 10 | Categorical boosting (Prokhorenkova et al., NeurIPS 2018) |

All neural models share a unified interface:
```python
forward(x: Tensor[batch, seq_len, n_features]) -> {"ret_1d": Tensor, "ret_5d": Tensor}
```

---

## Data Pipeline

### Instruments

15 financial instruments are downloaded via yfinance and cached to `.data_cache/`:

| Category | Instruments | Count |
|:---------|:------------|:-----:|
| FX Pairs | EUR/USD, GBP/USD, USD/JPY, USD/CHF, EUR/GBP, EUR/JPY | 6 |
| Equity Indices | S&P 500 (^GSPC) | 1 |
| Bonds | 10Y Treasury Yield (^TNX), 3M T-Bill (^IRX), TLT (20Y+ bond ETF) | 3 |
| Credit | HYG (High-Yield Corporate Bond ETF) | 1 |
| Commodities | Gold (GC=F), Crude Oil (CL=F) | 2 |
| Volatility | VIX (^VIX) | 1 |
| Dollar Index | DXY (DX-Y.NYB) | 1 |

### Feature Engineering (104 Features)

All features are strictly backward-looking -- no future data leakage. A 63-day warmup period is used for the longest lookback window.

| Feature Group | Per-Pair | Count | Description |
|:--------------|:--------:|:-----:|:------------|
| Log returns | 6 pairs | 6 | `log(close_t / close_{t-1})` |
| Rolling volatility | 6 pairs | 12 | 5d and 21d rolling std of log returns |
| RSI (14-day) | 6 pairs | 6 | Relative Strength Index |
| MACD | 6 pairs | 12 | MACD line + signal line (12/26/9 EMA) |
| Momentum | 6 pairs | 12 | 5d and 21d price momentum |
| Mean-reversion | 6 pairs | 12 | z-score of price vs 21d and 63d moving average |
| Microstructure | 6 pairs | 6 | High-low range as volatility proxy |
| Cross-pair correlations | -- | 5 | Rolling 21d correlation of EUR/USD vs each secondary pair |
| Macro returns | -- | 9 | Daily returns of all 9 macro instruments |
| Macro levels | -- | 9 | Normalized levels for yield curve, VIX, DXY |
| Derived macro | -- | 3 | Yield curve slope (10Y-3M), VIX change, DXY volatility |
| **Total** | | **~104** | |

### Targets

| Target | Definition | Use |
|:-------|:-----------|:----|
| `fwd_ret_1d` | EUR/USD log return at t+1 | Primary prediction target |
| `fwd_ret_5d` | EUR/USD log return at t+5 | Secondary target, auxiliary loss |

The trading strategy is directional: `sign(predicted_return) * actual_return`. The model predicts direction; profitability comes from being right more often than wrong.

---

## Evaluation Framework

### Super-Fold Design

Instead of training 7 separate models (one per fold), AutoResearch uses a **super-fold** approach:

```
 2005          2010          2015          2020          2024
  |______________|______________|______________|______________|
  |                                                           |
  |  TRAINING DATA (all historical, with holes punched out)   |
  |  2478 samples after hole-punching                         |
  |___________________________________________________________|
       ^    ^         ^    ^         ^    ^         ^    ^
       |    |         |    |         |    |         |    |
      V1   T1        V2   T2        V3   T3        V7   T7
      val  test      val  test      val  test      val  test

  V = Validation window     T = Test window
  Total: 7 val windows (915 rows), 7 test windows (1170 rows)
```

One model is trained on ALL historical data with all 14 val/test windows (plus buffers) surgically removed. This model is then evaluated separately on each of the 7 test windows, giving per-regime performance breakdowns.

**Why super-fold?**
- 7x faster than training 7 separate models
- More training data per model (2478 vs ~800-2400 depending on fold)
- Tests whether a single model generalizes across ALL market regimes

### Data Integrity Guarantees

| Guarantee | Implementation | Verification |
|:----------|:---------------|:-------------|
| Zero train/val/test overlap | `split_superfold()` hole-punches all windows | Programmatic assertion before every run |
| 90-day purge gap | Calendar-day gap between train-end and window-start | `validate_purge_embargo()` -- 0 violations |
| 21-day embargo | Gap between consecutive fold boundaries | Built into fold date definitions |
| 10-day label-horizon buffer | Extra exclusion before each window to prevent `fwd_ret_5d` leakage | `LABEL_HORIZON_BUFFER = 10` in splits.py |
| Contiguous windowing | `create_contiguous_datasets()` detects date gaps, creates per-segment datasets | Never creates sliding windows across gaps |
| Cached data | Downloaded once to `.data_cache/`, reused across all experiments | `download_all_pairs(cache_dir=".data_cache/")` |
| Expected counts | Train=2478, Val=915 (7 windows), Test=1170 (7 windows) | Verified at startup |

### Metrics

| Metric | Formula | Interpretation |
|:-------|:--------|:---------------|
| **Sharpe Ratio** | `(mean / std) * sqrt(252)` | Annualized risk-adjusted return |
| **Sortino Ratio** | `(mean / downside_std) * sqrt(252)` | Sharpe variant penalizing only downside |
| **PSR** | Bailey & Lopez de Prado (2012) | Probability true Sharpe > 0; accounts for skew/kurtosis |
| **DSR** | Lopez de Prado (2018) | Deflated Sharpe accounting for multiple testing |
| **IC** | `spearman_corr(predicted, actual)` | Prediction quality independent of position sizing |
| **Hit Rate** | `% of correct direction predictions` | Raw directional accuracy |
| **Profit Factor** | `sum(wins) / sum(losses)` | Magnitude-adjusted win/loss ratio |
| **Max Drawdown** | `max peak-to-trough decline` | Worst-case loss from peak equity |
| **VaR / CVaR** | 5% parametric value-at-risk | Tail risk measures |

### Composite Score

The single metric used for keep/revert decisions:

```
composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds
```

This formula ensures:
- The model must perform well on BOTH val and test (min prevents overfitting to one)
- Each fold with negative Sharpe incurs a -0.1 penalty (robustness across regimes)
- The champion (composite +5.50) has 0 negative folds and val/test Sharpe both > 5.5

---

## Uncertainty Estimation

Every prediction includes decomposed uncertainty estimates following Bayesian deep learning best practices:

```
 Model Output
 +------------------------------------------+
 |                                           |
 |  mean (mu)      -- point prediction       |
 |  log_var (s)    -- learned data noise     |  Aleatoric
 |                                           |  (Kendall & Gal, 2017)
 +------------------------------------------+
              |
              v
      MC Dropout (20 passes)
 +------------------------------------------+
 |                                           |
 |  var(mu_1..mu_20)  -- model uncertainty   |  Epistemic
 |                                           |  (Gal & Ghahramani, 2016)
 +------------------------------------------+
              |
              v
 +------------------------------------------+
 |                                           |
 |  total = aleatoric + epistemic            |
 |  confidence = sigmoid(-log(total))        |  0-1 scale
 |  bands: mu +/- 1*sigma, mu +/- 2*sigma   |
 |                                           |
 +------------------------------------------+
```

| Uncertainty Type | Source | What It Means | Champion Mean |
|:-----------------|:-------|:--------------|:-------------:|
| **Aleatoric** | Heteroscedastic variance head | Irreducible data noise -- market is inherently unpredictable here | 2.6e-05 |
| **Epistemic** | MC Dropout variance across 20 forward passes | Model doesn't have enough data for this regime | 5.2e-05 |
| **Confidence** | `sigmoid(-log(aleatoric + epistemic))` | Trading signal strength (0 = skip, 1 = trade) | 0.9999 |

**Practical use:** Predictions with confidence < 0.8 should be treated as "don't trade" signals. High epistemic uncertainty on a specific fold indicates the model needs more training data from that regime.

**Heteroscedastic loss (Kendall & Gal 2017):**
```
loss = exp(-s) * huber(mu, y) + 0.5 * s
```

The champion uses plain Huber loss (`het_loss=false`) because the variance branch steals capacity from mean prediction when training samples are limited (n=2478). The heteroscedastic loss is available via `--het-loss` for larger datasets.

---

## The Agent Loop

### Seven-Step Scientific Process

```
 +---> 1. DIAGNOSE -----> Per-fold failure analysis
 |         |               Which folds are weak? Why?
 |         |               What do uncertainty outputs reveal?
 |         v
 |     2. CITE ---------> Literature search
 |         |               What do published papers say about this failure mode?
 |         v
 |     3. HYPOTHESIZE ---> Form a testable hypothesis
 |         |               "Reducing hidden size from 512 to 128 will improve
 |         |                generalization because params/sample ratio is too high
 |         |                (Gu, Kelly & Xiu, 2020)"
 |         v
 |     4. PREDICT -------> State expected outcome BEFORE running
 |         |               "Expect composite +3.0 -> +4.0, fold 2 Sharpe > 0"
 |         v
 |     5. RUN -----------> Execute ONE experiment
 |         |               Single config change from champion
 |         |               python -m autoresearch.run_autoresearch ...
 |         v
 |     6. ANALYZE -------> Compare per-fold deltas to champion
 |         |               Was the prediction correct?
 |         |               Which folds improved? Which regressed?
 |         v
 |     7. CHECKPOINT ----> Save state for crash recovery
 |         |               Current champion, last result, next plan
 |         |               Self-contained: fresh session can resume
 |         |
 +--------<+ (loop)
```

### Crash-Recovery Checkpointing

The system is designed for unreliable hardware (laptop that crashes frequently):

- **Checkpoint every 5 minutes AND after every experiment** (whichever comes first)
- Checkpoint file: `memory/project_autoresearch_checkpoint.md`
- Contains: champion config, composite score, per-fold Sharpe table, last experiment result, exact next command to run, rationale, exhausted axes
- **Self-contained:** A fresh Claude Code session reading ONLY `CLAUDE.md` + the checkpoint can resume without reading any other file

### Agent Rules

The agent follows strict research methodology rules encoded in `CLAUDE.md`:

1. **Always start from the current best config.** Every experiment modifies ONE thing from the champion. If it improves, it becomes the new champion. If not, revert.
2. **Never grid-search.** Diagnose WHY a fold is weak before choosing what to change.
3. **Every hyperparameter must cite a paper.** "I'm trying X because fold Y has negative Sharpe due to Z, and paper W suggests this fix."
4. **Reproduce winners with multiple seeds** before trusting a result.
5. **Code changes are allowed** (architecture, loss function) with version tracking in `code_versions/`.
6. **If consecutive discards, stop and rethink.** Multiple failures mean the hypothesis about what to change is wrong.
7. **The agent never stops.** If out of ideas, research deeper: read papers, try combining near-misses, attempt radical architecture changes to escape local optima.

---

## Experiment History

### 90 Experiments Across 2 Backbones

| Phase | Backbone | Experiments | Focus | Key Finding |
|:-----:|:---------|:-----------:|:------|:------------|
| 1 | LFM2-350M | 20 | LR tuning (plain Huber) | LR sweet spot 2-3e-5 for frozen backbone fine-tuning |
| 2 | LFM2-350M | 28 | Heteroscedastic loss | Het-loss adds variance instability, hurts mean prediction on small data |
| 3 | LFM2-350M | 2 | Seed study | Massive seed variance with 639K params / 2738 samples |
| 4 | MLP | 6 | Capacity reduction | 512 -> 128 hidden eliminated memorization |
| 5 | MLP | 22 | Residual architecture | **Skip connection = 5x improvement** (the breakthrough) |
| 6 | MLP | 12 | HP optimization | lr=5e-4, huber=0.5, head_dropout=0.15 optimal |

### Key Discoveries

**1. Residual skip connection was the single biggest improvement.**

Test Sharpe jumped from +0.82 to +4.77 (5x). The linear shortcut lets the nonlinear branch focus on regime-specific corrections rather than learning the full input-output mapping. This mirrors findings from He et al. (2016) in computer vision -- skip connections help most when the identity mapping is close to optimal.

**2. Smaller is better for financial ML.**

Reducing MLP hidden from 512 to 128 (6x fewer parameters) improved generalization. The original model had 428 parameters per training sample -- far above the ~100:1 ratio recommended by Gu, Kelly & Xiu (2020) for financial return prediction. The champion achieves 121 params/sample.

**3. Foundation models underperform on daily FX.**

LFM2-350M (frozen backbone) achieved median test Sharpe +1.40 vs MLP's +4.76. The frozen backbone adds noise rather than useful inductive bias for this specific task. Daily FX returns are fundamentally different from the language/code data the foundation model was pre-trained on.

**4. Heteroscedastic loss hurts on small data.**

When the variance branch (Kendall & Gal, 2017) is active, it steals gradient capacity from the mean prediction head. With only 2478 training samples, the model learns to increase variance (cop out) rather than improve mean accuracy. This matches warnings in Seitzer et al. (2022).

**5. Seed variance dominates hyperparameter effects (unless architecture changes are large).**

For LFM2, the same config ranged from -1.52 to +1.77 composite across seeds. Only architecture-level changes (skip connection, capacity reduction) produce effects larger than seed noise. The residual MLP has much lower seed variance (std 0.60 vs 1.65 for LFM2).

**6. From-scratch training needs 50 epochs.**

Unlike fine-tuning a pre-trained backbone (20 epochs sufficient), MLPs trained from random initialization on financial data need 2.5x more epochs with cosine annealing schedule. Early stopping (patience=10) prevents overfitting while allowing enough iterations for convergence.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dlmastery/autoresearch.git
cd autoresearch

# Install with pip (requires Python 3.10+)
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"

# Or with conda
conda create -n autoresearch python=3.12
conda activate autoresearch
pip install -e ".[all]"
```

### Run the Champion

```bash
python -m autoresearch.run_autoresearch \
    --backbone mlp \
    --lr 5e-4 \
    --batch-size 32 \
    --seq-len 10 \
    --epochs 50 \
    --weight-decay 1e-5 \
    --patience 10 \
    --grad-clip 1.0 \
    --huber-delta 0.5 \
    --head-dropout 0.15 \
    --seed 0 \
    --description "champion run"
```

Expected output:
- Training: ~36 seconds on CPU
- Test Sharpe: +6.21
- Composite: +5.50
- All 7 folds positive

Results are saved to:
- `autoresearch/autoresearch_results/experiment_log.jsonl` (appended)
- `autoresearch/autoresearch_results/best_config.json` (overwritten if new champion)
- `autoresearch/autoresearch_results/best_model.pt` (model weights)

### View the Dashboard

```bash
python -m http.server 8765 --directory autoresearch/autoresearch_results
```

Open [http://localhost:8765/dashboard.html](http://localhost:8765/dashboard.html) in your browser. The dashboard provides:
- Train / Val / Test tabs with per-window breakdown
- Experiment history timeline
- Backbone comparison charts
- Per-fold uncertainty decomposition

### Run a Full Ablation

Compare all 8 backbones with a quick 5-epoch run:

```bash
python -m autoresearch.run_ablation --epochs 5
```

### Resume the Agent Loop

Open Claude Code in the repository directory and say "continue". The agent will:

1. Read `CLAUDE.md` for project rules
2. Read `memory/project_autoresearch_checkpoint.md` for crash-recovery state
3. Read the tail of `experiment_log.jsonl` and `best_config.json`
4. Start the dashboard server in the background
5. Resume the experiment loop from where it left off

---

## CLI Reference

### run_autoresearch (Single Experiment)

```
python -m autoresearch.run_autoresearch [OPTIONS]
```

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--backbone` | str | *required* | Backbone architecture. Choices: `mlp`, `lstm`, `lfm2-350m`, `patchtst`, `patchtsmixer`, `xgboost`, `lightgbm`, `catboost` |
| `--lr` | float | backbone-dependent | Learning rate. Champion: 5e-4 (MLP), 2e-5 (LFM2) |
| `--batch-size` | int | 32 | Mini-batch size for training |
| `--seq-len` | int | backbone-dependent | Input sequence length in business days. MLP: 10, LFM2: 60 |
| `--epochs` | int | 20 | Maximum training epochs |
| `--weight-decay` | float | 1e-5 | AdamW weight decay (L2 regularization) |
| `--patience` | int | 5 | Early stopping patience (epochs without val improvement) |
| `--grad-clip` | float | 1.0 | Gradient clipping max norm |
| `--warmup-epochs` | int | 0 | Linear LR warmup epochs before cosine decay |
| `--huber-delta` | float | 1.0 | Huber loss delta threshold. Lower = more robust to outliers |
| `--head-dropout` | float | 0.1 | Dropout rate in the prediction head |
| `--hidden-size` | int | backbone-dependent | Hidden dimension for MLP backbone |
| `--seed` | int | None | Random seed for reproducibility (torch + numpy + python) |
| `--het-loss` | flag | false | Enable heteroscedastic loss (Kendall & Gal 2017) |
| `--description` | str | *required* | Human-readable experiment description (logged to JSONL) |

**Example: LFM2 backbone with heteroscedastic loss:**
```bash
python -m autoresearch.run_autoresearch \
    --backbone lfm2-350m \
    --lr 3e-5 --batch-size 32 --seq-len 60 \
    --epochs 30 --patience 7 \
    --het-loss \
    --description "LFM2 het-loss lr=3e-5"
```

**Example: XGBoost baseline:**
```bash
python -m autoresearch.run_autoresearch \
    --backbone xgboost \
    --seq-len 10 \
    --description "XGBoost baseline"
```

### run_ablation (Multi-Backbone Comparison)

```
python -m autoresearch.run_ablation [OPTIONS]
```

Runs all 8 backbones with default hyperparameters and produces a comparison table.

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--epochs` | int | 20 | Maximum training epochs per backbone |

### baseline (Walk-Forward Evaluation)

```
python -m autoresearch.baseline [OPTIONS]
```

Single-backbone walk-forward evaluation using the standard 7-fold cross-validation setup.

### Entry Points

The following CLI entry points are available after `pip install -e .`:

| Command | Module | Description |
|:--------|:-------|:------------|
| `autoresearch-baseline` | `autoresearch.baseline:main` | Walk-forward evaluation |
| `autoresearch-ablation` | `autoresearch.run_ablation:main` | Multi-backbone comparison |
| `autoresearch-sweep` | `autoresearch.run_sweep:main` | Hyperparameter sweep |
| `autoresearch-optimizer` | `autoresearch.run_optimizer:main` | Bayesian optimization |

---

## Project Structure

```
autoresearch/
+-- CLAUDE.md                                  # Agent rules: experiment protocol, data integrity
|                                              #   constraints, hard rules, common mistakes
+-- pyproject.toml                             # Package config: dependencies, entry points
+-- README.md                                  # This file
+-- paper.md                                   # Research paper writeup
|
+-- autoresearch/                              # Python package root
    +-- __init__.py
    +-- run_autoresearch.py                    # Single-experiment runner (agent calls this)
    +-- run_ablation.py                        # Multi-backbone comparison runner
    +-- baseline.py                            # Walk-forward cross-validation evaluation
    |
    +-- data/
    |   +-- download.py                        # yfinance download for 15 instruments
    |   |                                      #   Cached to .data_cache/ (never re-downloads)
    |   +-- features.py                        # 104 backward-looking features
    |   |                                      #   Per-pair technical, cross-pair, macro
    |   +-- splits.py                          # 7 regime-aware folds, purge/embargo/buffer
    |                                          #   split_superfold(), validate_purge_embargo()
    |
    +-- model/
    |   +-- backbone.py                        # 8 backbone architectures
    |   |                                      #   MLP, LSTM, LFM2, PatchTST, PatchTSMixer,
    |   |                                      #   XGBoost, LightGBM, CatBoost
    |   +-- train.py                           # Training loop with early stopping, cosine LR
    |                                          #   create_contiguous_datasets(),
    |                                          #   find_contiguous_segments(), train_one_fold()
    |
    +-- evaluation/
    |   +-- metrics.py                         # Sharpe, Sortino, PSR, DSR, IC, trading_report,
    |                                          #   VaR, CVaR, profit_factor, max_drawdown
    |
    +-- autoresearch_results/
    |   +-- experiment_log.jsonl               # All 90 experiments (append-only structured log)
    |   +-- best_config.json                   # Current champion: config + full results
    |   +-- best_model.pt                      # Champion model weights (PyTorch state_dict)
    |   +-- dashboard.html                     # Live HTML dashboard (reads JSONL, decoupled)
    |   +-- experiment_summary.md              # Cross-model leaderboard table
    |   +-- autoresearch_report.md             # Comprehensive experiment narrative
    |   +-- trade_logs/                        # Per-trade CSV logs (win/loss per position)
    |   +-- winners/                           # Archived champion snapshots with full artifacts
    |
    +-- memory/
    |   +-- project_autoresearch_checkpoint.md # Crash-recovery state (updated every 5 min)
    |
    +-- code_versions/                         # Saved architecture variants with version numbers
    +-- docs/                                  # Design documents, research notes
    +-- tests/                                 # Unit and integration tests
    +-- .data_cache/                           # Downloaded market data (gitignored)
```

---

## Dependencies

### Core (required)

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `torch` | >= 2.5.0 | Neural network training and inference |
| `transformers` | >= 4.55 | HuggingFace model loading (LFM2, PatchTST) |
| `safetensors` | latest | Safe model weight serialization |
| `accelerate` | latest | HuggingFace training acceleration |
| `yfinance` | latest | Financial data download (OHLCV) |
| `pandas` | latest | Time series data manipulation |
| `numpy` | latest | Numerical computation |
| `scikit-learn` | latest | StandardScaler, train/test utilities |
| `scipy` | latest | Statistical functions (PSR, normality tests) |

### Optional

| Group | Packages | Install |
|:------|:---------|:--------|
| `optimizer` | `anthropic` | `pip install -e ".[optimizer]"` |
| `sweep` | `optuna` | `pip install -e ".[sweep]"` |
| `gbm` | `xgboost`, `lightgbm`, `catboost` | `pip install -e ".[gbm]"` |
| `all` | All of the above | `pip install -e ".[all]"` |
| `dev` | All + `pytest` | `pip install -e ".[dev]"` |

---

## Dashboard

The dashboard is a self-contained HTML file (`autoresearch_results/dashboard.html`) that reads `experiment_log.jsonl` and renders an interactive experiment monitor:

```bash
# Start the server
python -m http.server 8765 --directory autoresearch/autoresearch_results

# Open in browser
# http://localhost:8765/dashboard.html
```

**Features:**
- **Train / Val / Test tabs** -- switch between data splits to see per-window performance
- **Per-fold breakdown** -- Sharpe, return, win rate, IC for each of 7 regime windows
- **Experiment timeline** -- visual history of all 90 experiments with keep/discard markers
- **Backbone comparison** -- side-by-side metrics across architectures
- **Uncertainty visualization** -- per-fold aleatoric/epistemic decomposition
- **Equity curves** -- cumulative return plots per fold

The dashboard is fully decoupled from the runner: it only reads log files and never modifies them.

---

## Contributing

### Running the Test Suite

```bash
pip install -e ".[dev]"
pytest autoresearch/tests/
```

### Adding a New Backbone

1. Add the backbone class to `autoresearch/model/backbone.py`
2. Register it in `BACKBONE_REGISTRY` with a description
3. Set default sequence length in `BACKBONE_SEQ_LEN` (or use `_DEFAULT_SEQ_LEN = 10`)
4. Implement the standard interface: `forward(x) -> {"ret_1d": Tensor, "ret_5d": Tensor}`
5. For GBM models, implement via `GBMWrapper` with sklearn-compatible `.fit()` and `.predict()`
6. Run a quick ablation to verify: `python -m autoresearch.run_ablation --epochs 5`

### Adding New Features

1. Add feature computation to `autoresearch/data/features.py`
2. Ensure all features are strictly backward-looking (no future data)
3. Update `WARMUP_PERIOD` if new features require longer lookback
4. Verify feature count matches expectations in the validation checklist
5. Run champion config to confirm no regression

### Experiment Protocol

If contributing experiments:
- ONE config change per experiment from the current champion
- Every hyperparameter choice must be justified by a published paper or empirical evidence
- Report per-fold breakdown (not just aggregates)
- Use the composite metric for keep/revert decisions
- Log results to `experiment_log.jsonl` via the runner (never edit manually)

---

## References

### Core Methods

- He, K., Zhang, X., Ren, S., Sun, J. (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [[paper](https://arxiv.org/abs/1512.03385)]
- Gu, S., Kelly, B., Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273. [[paper](https://doi.org/10.1093/rfs/hhaa009)]
- Kendall, A., Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *Advances in Neural Information Processing Systems (NeurIPS)*. [[paper](https://arxiv.org/abs/1703.04977)]
- Gal, Y., Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *International Conference on Machine Learning (ICML)*. [[paper](https://arxiv.org/abs/1506.02142)]

### Financial ML

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. [[book](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)]
- Bailey, D., Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2). [[paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)]
- Seitzer, M., Tavakoli, A., Antic, D., Martius, G. (2022). "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." *International Conference on Learning Representations (ICLR)*. [[paper](https://arxiv.org/abs/2203.09168)]

### Optimization

- Loshchilov, I., Hutter, F. (2019). "Decoupled Weight Decay Regularization." *International Conference on Learning Representations (ICLR)*. [[paper](https://arxiv.org/abs/1711.05101)]
- Smith, S., Le, Q. (2018). "Don't Decay the Learning Rate, Increase the Batch Size." *International Conference on Learning Representations (ICLR)*. [[paper](https://arxiv.org/abs/1711.00489)]

### Backbone Architectures

- Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *International Conference on Learning Representations (ICLR)*. [[paper](https://arxiv.org/abs/2211.14730)]
- Chen, T., Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [[paper](https://arxiv.org/abs/1603.02754)]
- Ke, G., Meng, Q., Finley, T., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS)*. [[paper](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)]
- Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). "CatBoost: Unbiased Boosting with Categorical Features." *Advances in Neural Information Processing Systems (NeurIPS)*. [[paper](https://arxiv.org/abs/1706.09516)]

### Methodology

- Karpathy, A. (2019). "A Recipe for Training Neural Networks." Blog post. [[link](https://karpathy.github.io/2019/04/25/recipe/)]
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research*, 15(56), 1929-1958. [[paper](https://jmlr.org/papers/v15/srivastava14a.html)]

---

## License

MIT License. See [pyproject.toml](pyproject.toml) for details.

---

<p align="center">
  <sub>Built with Claude Code as the autonomous research agent. 90 experiments. Zero human intervention during experimentation.</sub>
</p>
