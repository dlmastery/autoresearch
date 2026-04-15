# 00 - Project Overview & Charter

**Project:** AutoResearch — Autonomous FX Return Prediction System
**Date:** 2026-04-04 (inception) — 2026-04-06 (current)
**Classification:** Research prototype / Quantitative finance ML system

---

## 1. Executive Summary

AutoResearch is a multi-horizon foreign exchange (FX) return prediction system that combines state-of-the-art deep learning backbones with an autonomous optimization loop powered by Claude API. The system predicts 1-day and 5-day forward returns for 6 currency pairs using 104 backward-looking features derived from price action and macroeconomic signals spanning 2005-2026.

The evaluation framework implements 7-fold regime-aware walk-forward cross-validation with 90-day purge gaps and 21-day embargo windows — following Lopez de Prado (2018) best practices to prevent data leakage and multiple-testing bias. The primary metric is annualized Sharpe ratio averaged across all folds.

An autonomous optimizer loop (Claude API) iteratively proposes, implements, evaluates, and either keeps or reverts code modifications — treating the entire ML pipeline as a search space for Sharpe improvement.

## 2. Problem Statement

Build an ML system that:
1. Predicts multi-horizon FX returns using foundation models as backbone
2. Evaluates with zero data leakage across 7 regime-diverse test sets
3. Uses a clean, risk-adjusted metric (average Sharpe) resistant to overfitting
4. Supports autonomous iterative improvement via an AI agent loop

## 3. Stakeholders & Roles

| Role | Description |
|------|-------------|
| **User/Researcher** | Defines requirements, selects architecture, reviews results |
| **Claude Agent (Optimizer)** | Proposes experiments, generates code modifications, evaluates |
| **System** | Executes training, evaluation, checkpointing, reporting |

## 4. Scope

### In Scope
- Daily-frequency FX return prediction (1-day, 5-day horizons)
- 6 major + cross currency pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, EUR/GBP, EUR/JPY)
- 11 model backbones (MLP, LSTM, LFM2.5, PatchTST, PatchTSMixer, Mamba2, Informer, XGBoost, LightGBM, CatBoost)
- Walk-forward backtesting with purge/embargo
- Autonomous optimization via Claude API
- Ablation study with Markdown reporting

### Out of Scope (Deferred)
- Intraday (1h, 4h) prediction horizons
- Live trading / execution engine
- Portfolio-level position sizing and risk management
- Retrieval-augmented generation (ChromaDB regime retrieval)
- Ensemble methods across backbones
- Distributed training / multi-GPU

## 5. Success Criteria

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Zero data leakage | All purge/embargo validations pass | PASSED |
| Baseline completes all 7 folds | No errors, all folds evaluated | PASSED |
| Ablation covers all 11 backbones | Per-backbone results generated | IN PROGRESS |
| Average Sharpe > 0 (any backbone) | Positive risk-adjusted return | PASSED (fold-level) |
| PSR > 0.95 (statistical significance) | At least one backbone | PENDING |
| Optimizer improves over baseline | Sharpe increase after experiments | PENDING |

## 6. Key Design Decisions

| # | Decision | Rationale | Source |
|---|----------|-----------|--------|
| 1 | LFM2.5 over LTC | Production foundation model (354M params) vs academic toy network | User feedback |
| 2 | Multi-horizon targets | Prevents single-timescale overfitting, more realistic | User choice |
| 3 | Sharpe metric over MSE | Risk-adjusted, industry standard, harder to game | User choice |
| 4 | 7 regime-aware folds | Tests across crisis, recovery, plateau, downturn, upturn | User feedback |
| 5 | 90-day purge gaps | Conservative leakage prevention (literature uses 30-60 days) | User requirement |
| 6 | Multi-pair input | Cross-currency relationships provide additional signal | User choice |
| 7 | Huber loss over MSE | Robust to fat-tailed FX returns | Best practice |
| 8 | 11 backbones | Comprehensive comparison: classical ML to foundation models | Ablation design |

## 7. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11.9 |
| Deep Learning | PyTorch | >=2.5.0 |
| Foundation Models | HuggingFace Transformers | >=4.55 |
| Data Source | Yahoo Finance (yfinance) | Latest |
| Data Processing | pandas, numpy | Latest |
| Preprocessing | scikit-learn (StandardScaler) | Latest |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | Latest |
| AI Agent | Anthropic Claude API | claude-sonnet-4 |
| Testing | pytest | Latest |
| Hardware | CPU (Intel Iris Xe) | float32 mode |

## 8. Document Index

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
