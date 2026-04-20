# 00 - Project Overview & Charter

**Project:** AutoResearch -- Autonomous FX Return Prediction System
**Date:** 2026-04-04 (inception) -- 2026-04-19 (current)
**Classification:** Research prototype / Quantitative finance ML system

---

## Key Highlights

- **Champion model:** Residual MLP with 300K parameters achieves test Sharpe +6.21, 7/7 positive folds, 1001% cumulative return across 7 regime windows (2008--2025)
- **Evaluation rigor:** 7-fold regime-aware walk-forward CV with 90-day purge gaps, 21-day embargo, 10-day label-horizon buffers -- zero data leakage verified programmatically
- **Autonomous research loop:** 90 experiments executed by Claude Code agent using Karpathy-style one-change-at-a-time methodology -- no human hyperparameter tuning
- **Data coverage:** 21 years of daily data (2005--2026) from 6 FX pairs + 9 macro instruments, yielding 104 backward-looking features
- **Simplicity wins:** A 301K-parameter residual MLP outperforms 354M-parameter foundation models (LFM2.5), PatchTST, LSTM, and gradient boosting ensembles on risk-adjusted returns

---

## 1. Executive Summary

AutoResearch is a multi-horizon foreign exchange (FX) return prediction system that combines state-of-the-art deep learning backbones with an autonomous optimization loop powered by Claude Code. The system predicts 1-day and 5-day forward returns for 6 currency pairs using 104 backward-looking features derived from price action and macroeconomic signals spanning 2005--2026.

The evaluation framework implements 7-fold regime-aware walk-forward cross-validation with 90-day purge gaps and 21-day embargo windows -- following Lopez de Prado (2018) best practices to prevent data leakage and multiple-testing bias. The primary metric is annualized Sharpe ratio averaged across all folds, penalized by the number of negative-fold windows.

An autonomous optimizer loop (Claude Code as the outer agent) iteratively diagnoses per-fold failure modes, formulates literature-backed hypotheses, runs single-variable experiments, and either keeps or reverts changes -- treating the entire ML pipeline (architecture, loss, features, hyperparameters, training schedule) as a search space for Sharpe improvement. After 90 experiments, the champion is a residual MLP with test Sharpe +6.21 and 7/7 positive test folds.

## 2. Why This Project Exists

### 2.1 The FX Prediction Challenge

Foreign exchange markets are the largest and most liquid financial markets in the world, with daily turnover exceeding $7.5 trillion (BIS Triennial Survey, 2022). Despite this liquidity, consistently profitable FX prediction remains one of the hardest problems in quantitative finance:

- **Low signal-to-noise ratio (SNR):** Daily FX returns have SNR on the order of 0.05--0.1, compared to 0.5+ for many equity factors. The signal is a tiny perturbation on near-random walk dynamics.
- **Non-stationarity:** FX dynamics are driven by macroeconomic regimes (interest rate differentials, risk appetite, central bank policy) that shift over multi-year cycles. A model trained on crisis data may fail completely in low-volatility environments.
- **The efficient market hypothesis (EMH):** Major FX pairs are among the most informationally efficient instruments. Any exploitable pattern is quickly arbitraged away by institutional market makers.
- **Academic skepticism:** Meese & Rogoff (1983) famously showed that no structural model could outperform a random walk for FX forecasting. Decades of subsequent research have produced only marginal improvements (Rossi, 2013).

### 2.2 Why AutoResearch Can Succeed

AutoResearch addresses these challenges through several key innovations:

1. **Multi-signal feature engineering:** 104 features from 15 instruments capture not just price momentum but cross-currency correlations, yield curve dynamics, volatility regimes, and commodity flows. This breadth of signal is difficult for humans to synthesize manually.

2. **Regime-aware evaluation:** The 7-fold CV design deliberately spans crisis (2008 GFC), recovery (2009--2010), low-volatility plateaus (2018--2019), and trend reversals (2022 EUR crisis). A model must perform across all regimes to achieve a high composite score.

3. **Autonomous experimentation at scale:** Claude Code acts as the research loop, running 90+ experiments with disciplined one-variable-at-a-time methodology. This is equivalent to months of manual research compressed into days.

4. **Simplicity bias via competitive selection:** The Karpathy-style keep/discard protocol naturally favors simpler models that generalize. The 301K-parameter residual MLP beat 354M-parameter foundation models -- a strong empirical confirmation that for low-SNR financial data, overparameterization hurts.

### 2.3 Competitive Landscape

| Approach | Typical Sharpe (daily, OOS) | Weakness vs. AutoResearch |
|----------|----------------------------|---------------------------|
| Random walk (baseline) | 0.0 | No signal at all |
| Traditional econometric (ARIMA, VAR) | 0.1--0.3 | Linear, no regime awareness |
| Academic ML (LSTM, RNN) | 0.3--0.8 | In-sample leakage common, single-fold eval |
| Proprietary quant (e.g., Man AHL, Citadel) | 1.0--3.0 (portfolio) | Not comparable (portfolio-level, multi-asset) |
| Foundation model fine-tuning (TimesFM, LFM2) | 0.5--1.5 | Overparameterized for daily FX SNR |
| **AutoResearch (this project)** | **6.21 (test, 7-fold avg)** | Research prototype, not yet live-tested |

**Important caveat:** The Sharpe of 6.21 is computed from a sign-based strategy (go long if predicted return > 0, else short) applied to each of 7 test windows separately and then averaged. This is not a single continuous track record. Live trading performance would depend on transaction costs, slippage, and capital constraints not modeled here.

## 3. Problem Statement

Build an ML system that:
1. Predicts multi-horizon FX returns using foundation models as backbone
2. Evaluates with zero data leakage across 7 regime-diverse test sets
3. Uses a clean, risk-adjusted metric (average Sharpe) resistant to overfitting
4. Supports autonomous iterative improvement via an AI agent loop

### 3.1 Formal Objective

Given a feature matrix X(t) constructed from 104 backward-looking signals at time t, predict:

```
y_hat_1d(t) = E[r(t+1) | X(t)]       -- 1-day forward return
y_hat_5d(t) = E[r(t+1:t+5) | X(t)]   -- 5-day forward return
```

for 6 currency pairs simultaneously. The trading strategy is:

```
position(t) = sign(y_hat(t))           -- long if predicted positive, short otherwise
strategy_return(t) = position(t) * actual_return(t+1)
```

The primary evaluation metric is the annualized Sharpe ratio of strategy returns, averaged across 7 regime-diverse test windows.

## 4. Stakeholders & Roles

| Role | Description |
|------|-------------|
| **User/Researcher** | Defines requirements, selects architecture, reviews results |
| **Claude Agent (Optimizer)** | Proposes experiments, generates code modifications, evaluates |
| **System** | Executes training, evaluation, checkpointing, reporting |

## 5. Scope

### In Scope
- Daily-frequency FX return prediction (1-day, 5-day horizons)
- 6 major + cross currency pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, EUR/GBP, EUR/JPY)
- 8 model backbones (MLP, LSTM, LFM2.5-350M, PatchTST, PatchTSMixer, XGBoost, LightGBM, CatBoost)
- Walk-forward backtesting with purge/embargo (7 folds, regime-aware)
- Autonomous optimization via Claude Code agent (Karpathy-style loop)
- Per-fold regime analysis with uncertainty quantification (aleatoric + epistemic)

### Out of Scope (Deferred)
- Intraday (1h, 4h) prediction horizons
- Live trading / execution engine
- Portfolio-level position sizing and risk management
- Retrieval-augmented generation (ChromaDB regime retrieval)
- Ensemble methods across backbones
- Distributed training / multi-GPU
- Transaction cost modeling and slippage estimation

## 6. Success Criteria

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Zero data leakage | All purge/embargo validations pass | PASSED |
| Baseline completes all 7 folds | No errors, all folds evaluated | PASSED |
| Ablation covers all 8 backbones | Per-backbone results generated | PASSED |
| Average test Sharpe > 0 (any backbone) | Positive risk-adjusted return | PASSED (+6.21) |
| PSR > 0.95 (statistical significance) | At least one backbone | PASSED (PSR = 1.0) |
| All test folds positive | 7/7 positive Sharpe folds | PASSED (7/7) |
| Optimizer improves over baseline | Sharpe increase after experiments | PASSED (90 experiments, major improvement) |

### 6.1 Champion Performance Summary

| Metric | Test (7 windows) | Validation (7 windows) | Training |
|--------|-------------------|------------------------|----------|
| Sharpe ratio | 6.21 | 5.60 | 7.94 |
| Sortino ratio | 11.31 | 8.45 | 23.26 |
| Cumulative return | 1001% | 247% | 566,915% |
| Max drawdown | 4.13% | 7.43% | 4.92% |
| Win rate | 69.4% | 66.2% | 78.0% |
| Profit factor | 3.30 | 2.61 | 7.92 |
| IC (rank corr) | 0.485 | 0.458 | 0.756 |
| PSR | 1.00 | 1.00 | 1.00 |
| Positive folds | 7/7 | 6/7 | 1/1 |

## 7. Key Design Decisions

| # | Decision | Rationale | Source |
|---|----------|-----------|--------|
| 1 | LFM2.5 over LTC | Production foundation model (354M params) vs academic toy network | User feedback |
| 2 | Multi-horizon targets | Prevents single-timescale overfitting, more realistic | User choice |
| 3 | Sharpe metric over MSE | Risk-adjusted, industry standard, harder to game | User choice |
| 4 | 7 regime-aware folds | Tests across crisis, recovery, plateau, downturn, upturn | User feedback |
| 5 | 90-day purge gaps | Conservative leakage prevention (literature uses 30-60 days) | User requirement |
| 6 | Multi-pair input | Cross-currency relationships provide additional signal | User choice |
| 7 | Huber loss over MSE | Robust to fat-tailed FX returns | Best practice |
| 8 | 8 backbones | Comprehensive comparison: classical ML to foundation models | Ablation design |
| 9 | Residual MLP architecture | Skip connection helps in low-SNR regime (He et al., 2016) | Experiment results |
| 10 | Super-fold evaluation | Single train/eval pass with hole-punching, much faster than 7x | Performance optimization |

## 8. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11.9 |
| Deep Learning | PyTorch | >=2.5.0 |
| Foundation Models | HuggingFace Transformers | >=4.55 |
| Data Source | Yahoo Finance (yfinance) | Latest |
| Data Processing | pandas, numpy | Latest |
| Preprocessing | scikit-learn (StandardScaler) | Latest |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | Latest |
| AI Agent | Claude Code (Claude Opus 4.6) | 1M context |
| Testing | pytest | Latest |
| Hardware | CPU (Intel Iris Xe) | float32 mode |

## 9. Project Timeline

```
2026-04-04  Project inception, requirements, initial data pipeline
2026-04-05  Feature engineering (104 features), 7-fold split design
2026-04-06  8-backbone ablation study, baseline evaluation
2026-04-07  Autonomous experiment loop (run_autoresearch.py)
2026-04-08  Super-fold evaluation, heteroscedastic loss experiments
     ...    90 experiments: architecture search, hyperparameter tuning
2026-04-14  Champion identified: Residual MLP, test Sharpe +6.21
2026-04-19  Documentation consolidation (current)
```

## 10. How to Get Started

If you just joined the project, follow this reading order:

1. **This document** -- understand the goals and scope
2. **[System Design](system-design.md)** -- understand how the components connect
3. **[Data Engineering](../data/data-engineering.md)** -- understand the 104 features and split design
4. **[Model Architecture](model-architecture.md)** -- understand the 8 backbones, especially the residual MLP champion
5. **[Training Infrastructure](../training/training-infrastructure.md)** -- understand the training loop and loss functions
6. **[Evaluation Framework](../evaluation/evaluation-framework.md)** -- understand metrics and the composite score
7. **[Autonomous Optimization](../operations/autonomous-optimization.md)** -- understand the Claude Code experiment loop

To run the champion model:

```bash
cd C:/Users/evija/autoresearch
"C:/Users/evija/anaconda3/python.exe" -m autoresearch.run_autoresearch \
    --backbone mlp --lr 5e-4 --batch-size 32 --seq-len 10 --epochs 50 \
    --weight-decay 1e-5 --patience 10 --grad-clip 1.0 \
    --huber-delta 0.5 --head-dropout 0.15 --seed 0 --no-het-loss \
    --description "champion baseline"
```

## 11. Document Index

| # | Document | SWEBoK KA | Contents |
|---|----------|-----------|----------|
| 00 | Project Overview (this) | — | Charter, scope, decisions |
| 01 | Requirements Specification | KA1: Requirements | Functional/non-functional requirements |
| 02 | System Design & Architecture | KA2: Design | Architecture, patterns, data flow |
| 03 | Data Engineering | KA2: Design | Pipeline, features, leakage prevention |
| 04 | Model Architecture | KA2: Design | 11 backbones, interfaces, registry |
| 05 | Training Infrastructure | KA3: Construction | Training loop, hyperparameters |
| 06 | Evaluation Framework | KA5: Testing | Metrics, walk-forward, regime analysis |
| 07 | Testing Strategy | KA5: Testing | Test suite, coverage, verification |
| 08 | Configuration Management | KA6: Config Mgmt | Dependencies, environment, versioning |
| 09 | Autonomous Optimization | KA3: Construction | Claude API agent loop |
| 10 | Quality & Best Practices | KA11: Quality | Practices followed + improvements |
| 11 | Operations & Deployment | KA9: Maintenance | CLI, entry points, crash recovery |

---

## 12. References

- Bailey, D.H. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*.
- He, K. et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. (Residual connections rationale)
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Purge/embargo, combinatorial CV)
- Meese, R. & Rogoff, K. (1983). "Empirical Exchange Rate Models of the Seventies." *Journal of International Economics*.
- Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR*.
- Rossi, B. (2013). "Exchange Rate Predictability." *Journal of Economic Literature*.

---

*See also:* [System Design](system-design.md) | [Data Engineering](../data/data-engineering.md) | [Model Architecture](model-architecture.md)
