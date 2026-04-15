# AutoResearch Implementation Status

**Last updated:** 2026-04-06

## Architecture Overview

Multi-horizon FX return prediction using walk-forward evaluation across 7 regime-aware folds with 11 model backbones spanning classical ML, deep learning, and foundation models.

## Completed Implementation

### 1. Data Pipeline (`data/`)
- **download.py**: Yahoo Finance data for 6 FX pairs + 9 macro tickers (2005-2026), with parquet caching
- **features.py**: 104 backward-looking features (per-pair technical, cross-pair correlations, macro signals)
- **splits.py**: 7 walk-forward folds with 90-day purge + 21-day embargo, validated by `validate_purge_embargo()`

### 2. Model Backbones (`model/backbone.py`)

| # | Backbone | Type | Reference | Status |
|---|----------|------|-----------|--------|
| 1 | `mlp` | Feedforward | Baseline | Tested |
| 2 | `lstm` | Recurrent (BiLSTM) | Baseline | Tested |
| 3 | `lfm2-350m` | Foundation (frozen) | Liquid AI, Mar 2026 | Tested |
| 4 | `lfm2-1.2b` | Foundation (frozen) | Liquid AI, Mar 2026 | Tested |
| 5 | `patchtst` | Transformer | Nie et al., ICLR 2023 | Tested |
| 6 | `patchtsmixer` | MLP-Mixer | Google, NeurIPS 2023 | Tested |
| 7 | `mamba2` | State-space | Gu & Dao, ICML 2024 | Tested |
| 8 | `informer` | Sparse Transformer | Zhou et al., AAAI 2021 | Tested |
| 9 | `xgboost` | Gradient Boosting | Chen & Guestrin, 2016 | Tested |
| 10 | `lightgbm` | Gradient Boosting | Ke et al., NeurIPS 2017 | Tested |
| 11 | `catboost` | Gradient Boosting | Prokhorenkova et al., 2018 | Tested |

All neural models share interface: `forward(x: [B, T, F]) -> {"ret_1d": [B, 6], "ret_5d": [B, 6]}`
GBM models use `GBMWrapper` with `fit(X, y)` / `predict(X)` interface.

### 3. Training (`model/train.py`)
- **Huber loss** (robust to fat-tailed FX returns)
- **AdamW optimizer** with cosine annealing LR schedule
- **Gradient clipping** (max norm 1.0)
- **Early stopping** (patience 3 epochs, restores best model)
- **GPU-optimized** DataLoaders with pin_memory
- **Batch size 64** (tuned for RTX 4090)

### 4. Evaluation (`evaluation/metrics.py`)

**Standard Metrics:**
- Sharpe, Sortino, Calmar, Omega ratios
- Total/annualized return, max drawdown
- Win rate, profit factor, avg win/loss

**ML Finance Best Practices (Lopez de Prado, 2018):**
- **PSR** (Probabilistic Sharpe Ratio): statistical significance of Sharpe
- **DSR** (Deflated Sharpe Ratio): adjusted for multiple testing
- **IC** (Information Coefficient): Spearman rank correlation pred vs actual
- **VaR/CVaR** at 95% and 99%: tail risk
- Skewness, excess kurtosis, tail ratio

**Weighted Sharpe:** training-size-weighted average across folds

### 5. Walk-Forward Validation (`data/splits.py`)
- **Purge**: 90-day gap train→val and val→test (prevents label leakage)
- **Embargo**: 21-day gap between consecutive fold boundaries
- **Disjoint test sets**: no overlap between any two folds
- **Expanding window**: each fold trains on all prior data
- **Regime labels**: GFC, recovery, debt crisis, strong USD, low-vol, EUR crisis, recent

### 6. Infrastructure
- **Checkpointing**: per-fold JSON checkpoints, automatic resume on crash
- **GPU support**: automatic CUDA detection, model + data on RTX 4090
- **Ablation runner**: `run_ablation.py` evaluates all backbones and generates Markdown report
- **CLI arguments**: `--backbone`, `--epochs`, `--seq-len`

## Entry Points

```bash
# Single backbone
python baseline.py --backbone lfm2-350m --epochs 5

# Full ablation (all 11 models)
python run_ablation.py --epochs 5

# Subset ablation
python run_ablation.py --backbones mlp lstm patchtst mamba2 xgboost

# Optimizer (Claude-powered iterative improvement)
python run_optimizer.py --max-experiments 10
```

## Recovery from Crash

If the process crashes mid-ablation:
1. Per-backbone results are saved to `ablation_results/<backbone>.json` as each completes
2. The baseline checkpoint (`baseline_checkpoint.json`) saves per-fold progress
3. Re-running will skip completed folds automatically
4. Delete `baseline_checkpoint.json` to force a clean re-run of the current backbone

## File Structure

```
autoresearch/
├── data/
│   ├── download.py        # Yahoo Finance data + caching
│   ├── features.py        # 104 features, all backward-looking
│   └── splits.py          # 7-fold walk-forward + purge/embargo validation
├── model/
│   ├── backbone.py        # 11 backbones (MLP, LSTM, LFM2.5, PatchTST, ...)
│   └── train.py           # Huber loss, AdamW, cosine LR, early stopping
├── evaluation/
│   ├── metrics.py         # 25+ metrics (Sharpe, PSR, DSR, IC, VaR, ...)
│   └── leakage_check.py   # No-leakage verification
├── optimizer/
│   ├── agent_loop.py      # Claude API autonomous experiment loop
│   └── prompts.py         # Brainstorm + code generation prompts
├── tests/                 # Pytest suite
├── baseline.py            # Walk-forward evaluation (neural + GBM)
├── run_ablation.py        # Multi-backbone ablation with Markdown report
├── run_optimizer.py       # CLI for autonomous optimizer
└── run_overnight.py       # Full overnight pipeline
```
