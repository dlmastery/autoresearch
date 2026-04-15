# 12 - ArXiv State-of-the-Art Survey: Time Series & FX Prediction

**Date:** 2026-04-06
**Scope:** 2024-2026 papers on time series foundation models, financial forecasting, and methods relevant to the AutoResearch FX prediction system
**Current Backbones:** MLP, BiLSTM, LFM2.5, PatchTST, PatchTSMixer, Mamba2, Informer, XGBoost, LightGBM, CatBoost

---

## Table of Contents

1. [Time Series Foundation Models](#1-time-series-foundation-models)
2. [Mamba / SSM Variants for Finance](#2-mamba--ssm-variants-for-finance)
3. [Transformer Variants for Time Series](#3-transformer-variants-for-time-series)
4. [Multi-Scale / Multi-Horizon Architectures](#4-multi-scale--multi-horizon-architectures)
5. [Regime-Aware / Non-Stationary Methods](#5-regime-aware--non-stationary-methods)
6. [FX-Specific Papers](#6-fx-specific-papers)
7. [LLM + Financial Data Combination](#7-llm--financial-data-combination)
8. [Retrieval-Augmented Approaches for Time Series](#8-retrieval-augmented-approaches-for-time-series)
9. [Graph Neural Networks for Multi-Asset Prediction](#9-graph-neural-networks-for-multi-asset-prediction)
10. [Walk-Forward Evaluation & Lopez de Prado Methods](#10-walk-forward-evaluation--lopez-de-prado-methods)
11. [Ensemble & Stacking Methods](#11-ensemble--stacking-methods)
12. [Critical Perspectives](#12-critical-perspectives)
13. [Comparison Matrix](#13-comparison-matrix)
14. [Recommendations for AutoResearch](#14-recommendations-for-autoresearch)

---

## 1. Time Series Foundation Models

### 1.1 Chronos-2 (Amazon, 2025)

| Field | Detail |
|-------|--------|
| **Title** | Chronos-2: From Univariate to Universal Forecasting |
| **ArXiv** | [2510.15821](https://arxiv.org/abs/2510.15821) |
| **Year** | October 2025 |
| **Authors** | Amazon Science team |
| **Parameters** | 120M (encoder-only) |
| **Key Contribution** | First TSFM to handle univariate, multivariate, and covariate-informed forecasting in zero-shot. Uses T5 encoder with group attention mechanism for efficient in-context learning across related series and covariates. Produces multi-step-ahead quantile forecasts. |
| **Performance** | SOTA zero-shot on fev-bench, GIFT-Eval, and Chronos Benchmark II. Surpasses TiRex and TimesFM-2.5 by statistically significant margins. Over 300 forecasts/second on A10G GPU. |
| **Relevance to AutoResearch** | **High.** Supports multivariate inputs (our 104 features) and covariate-informed prediction. Could replace frozen LFM2.5 backbone as a pretrained feature extractor. Zero-shot capability means no fine-tuning needed initially. |

### 1.2 TimesFM 2.5 (Google, 2025)

| Field | Detail |
|-------|--------|
| **Title** | A Decoder-Only Foundation Model for Time-Series Forecasting |
| **ArXiv** | [2310.10688](https://arxiv.org/abs/2310.10688) (original); v2.5 released September 2025 |
| **Year** | 2023 (original), 2025 (v2.5) |
| **Venue** | ICML 2024 (v1.0) |
| **Parameters** | 200M |
| **Key Contribution** | Decoder-only transformer pretrained on 100B real-world time points. Patch-based tokenization with continuous embeddings. TimesFM 2.0 reached #1 on GIFT-Eval in early 2025; v2.5 retook #1 in September 2025 (subsequently surpassed by Chronos-2). |
| **Performance** | Zero-shot performance competitive with supervised approaches. Available in BigQuery. |
| **Relevance to AutoResearch** | **Medium.** Strong general forecaster, but Re(Visiting) paper (Section 7) shows off-the-shelf TSFMs perform poorly on financial data without domain adaptation. Would need fine-tuning on FX data. |

### 1.3 Moirai 2.0 & Moirai-MoE (Salesforce, 2025)

| Field | Detail |
|-------|--------|
| **Title** | Moirai 2.0: When Less Is More for Time Series Forecasting |
| **ArXiv** | [2511.11698](https://arxiv.org/abs/2511.11698) (Moirai 2.0); [2410.10469](https://arxiv.org/abs/2410.10469) (Moirai-MoE) |
| **Year** | November 2025 (2.0), October 2024 (MoE) |
| **Parameters** | 2.0 is 30x smaller than 1.0-Large while performing better |
| **Key Contribution** | Moirai 2.0 replaces masked-encoder with decoder-only architecture, uses quantile loss and multi-token prediction. 2x faster, 30x smaller than Moirai 1.0-Large. Moirai-MoE is first MoE time series foundation model, achieving token-level specialization with up to 65x fewer activated parameters than Chronos/TimesFM. |
| **Performance** | Moirai-MoE: 17% improvement over Moirai 1.0 at same size. Moirai 2.0: top-tier on GIFT-Eval with excellent speed/accuracy/size trade-off. |
| **Relevance to AutoResearch** | **Medium-High.** MoE approach interesting for multi-regime FX data where different "experts" could specialize in different market conditions. Decoder-only quantile forecasts provide built-in uncertainty quantification. |

### 1.4 TiRex (NX-AI, 2025)

| Field | Detail |
|-------|--------|
| **Title** | TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning |
| **ArXiv** | [2505.23719](https://arxiv.org/abs/2505.23719) |
| **Year** | May 2025 |
| **Venue** | NeurIPS 2025 |
| **Parameters** | 35M (xLSTM-based) |
| **Key Contribution** | Bridges gap between LSTMs and in-context learning using xLSTM architecture. Introduces Contiguous Patch Masking (CPM) training strategy to prevent degradation in long-horizon autoregressive predictions. SOTA zero-shot on GIFT-Eval and Chronos-ZS benchmarks. |
| **Performance** | Outperforms TabPFN-TS, Chronos Bolt, TimesFM despite being significantly smaller. |
| **Relevance to AutoResearch** | **Very High.** xLSTM architecture is a natural upgrade path from our BiLSTM backbone. 35M params makes it feasible on CPU (our Intel Iris Xe). Could be fine-tuned on FX data. |

### 1.5 FlowState (IBM, 2025)

| Field | Detail |
|-------|--------|
| **Title** | FlowState: Sampling Rate Invariant Time Series Forecasting |
| **ArXiv** | [2508.05287](https://arxiv.org/abs/2508.05287) |
| **Year** | August 2025 |
| **Venue** | NeurIPS 2025 Workshop |
| **Parameters** | 9.1M |
| **Key Contribution** | SSM-based encoder with functional basis decoder enables continuous-time modeling and dynamic time-scale adjustment. Can train at one sampling rate and predict at another. Smallest model in GIFT-Eval top 10, outperforming models 20x its size. |
| **Performance** | SOTA on GIFT-ZS and Chronos-ZS benchmarks. |
| **Relevance to AutoResearch** | **Very High.** SSM encoder aligns with our Mamba2 backbone but adds continuous-time modeling. At 9.1M params, extremely lightweight. Multi-scale capability relevant for our 1d/5d multi-horizon prediction. |

### 1.6 TabPFN-TS (Prior Labs, 2025)

| Field | Detail |
|-------|--------|
| **Title** | From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting |
| **ArXiv** | [2501.02945](https://arxiv.org/abs/2501.02945) |
| **Year** | January 2025 (revised January 2026) |
| **Venue** | NeurIPS 2024 TRL & TSALM Workshops |
| **Parameters** | 11M |
| **Key Contribution** | Treats forecasting as tabular regression via temporal featurization + pretrained TabPFN-v2. No time-series-specific pretraining. Only model supporting direct covariate inputs. Non-autoregressive: predicts full horizon in single forward pass. |
| **Performance** | Top rank on GIFT-Eval for covariate-informed forecasting. Outperforms all multivariate approaches on covariate benchmarks. |
| **Relevance to AutoResearch** | **High.** Our 104-feature input with covariates (macro, technical) is a perfect fit for TabPFN-TS's covariate-informed design. Lightweight (11M) and does not require sequential processing. Conceptually similar to our GBM approach but with neural backbone. |

### 1.7 Kronos (Tsinghua, 2025)

| Field | Detail |
|-------|--------|
| **Title** | Kronos: A Foundation Model for the Language of Financial Markets |
| **ArXiv** | [2508.02739](https://arxiv.org/abs/2508.02739) |
| **Year** | August 2025 |
| **Venue** | AAAI 2026 |
| **Parameters** | Not specified (multiple sizes) |
| **Key Contribution** | First TSFM pretrained specifically on financial K-line (OHLCV) data. Specialized tokenizer quantizes candlestick data into hierarchical discrete tokens. Autoregressive transformer pretrained on 12B+ K-line records from 45 global exchanges (equities, futures, forex, crypto). |
| **Performance** | Price series forecasting RankIC: +93% over leading TSFM, +87% over best non-pretrained baseline. Volatility forecasting MAE: 9% lower. Synthetic K-line fidelity: +22%. |
| **Relevance to AutoResearch** | **Very High.** Only TSFM pretrained specifically on financial data including forex. OHLCV tokenizer designed for market data. Zero-shot financial capability directly applicable. Strong validation of domain-specific pretraining for finance. |

### 1.8 Lag-Llama (2023-2024)

| Field | Detail |
|-------|--------|
| **Title** | Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting |
| **ArXiv** | [2310.08278](https://arxiv.org/abs/2310.08278) |
| **Year** | October 2023 (v3: February 2024) |
| **Parameters** | Based on LLaMA architecture with RMSNorm and SwiGLU |
| **Key Contribution** | First open-source TSFM. Decoder-only transformer using lags as covariates for univariate probabilistic forecasting. Strong zero-shot generalization; SOTA when fine-tuned on small dataset fractions. |
| **Relevance to AutoResearch** | **Medium.** Univariate only, which limits direct applicability to our 104-feature multivariate setup. However, fine-tuning approach on small data fractions is relevant given our fold sizes. |

### 1.9 MOMENT (CMU, 2024)

| Field | Detail |
|-------|--------|
| **Title** | MOMENT: A Family of Open Time-series Foundation Models |
| **ArXiv** | [2402.03885](https://arxiv.org/abs/2402.03885) |
| **Year** | February 2024 |
| **Key Contribution** | Open-source multi-task foundation model using patch embedding and masked prediction. Constructs "Time Series Pile" with 1.23B timestamps from 13 domains. |
| **Relevance to AutoResearch** | **Low-Medium.** Univariate only. Masked prediction approach less suited for our directional forecasting task. |

### 1.10 UniTS (Harvard/MIT, 2024)

| Field | Detail |
|-------|--------|
| **Title** | UniTS: A Unified Multi-Task Time Series Model |
| **ArXiv** | [2403.00131](https://arxiv.org/abs/2403.00131) |
| **Year** | March 2024 |
| **Venue** | NeurIPS 2024 |
| **Key Contribution** | Task tokenization to express predictive and generative tasks within single model. Modified transformer for universal time series representations. |
| **Relevance to AutoResearch** | **Medium.** Multi-task capability could unify our ret_1d and ret_5d predictions. Not finance-specific. |

---

## 2. Mamba / SSM Variants for Finance

### 2.1 Mamba-3 (2026)

| Field | Detail |
|-------|--------|
| **Title** | Mamba-3: Improved Sequence Modeling using State Space Principles |
| **ArXiv** | [2603.15569](https://arxiv.org/abs/2603.15569) |
| **Year** | March 2026 |
| **Authors** | Lahoti, Li, Chen, Wang et al. |
| **Key Contribution** | Three core improvements: (1) more expressive recurrence from SSM discretization, (2) complex-valued state update for richer state tracking, (3) multi-input architecture. Direct successor to Mamba2 which we already use. |
| **Relevance to AutoResearch** | **Very High.** Direct upgrade path for our Mamba2 backbone. Complex-valued states may better capture oscillatory FX patterns. Multi-input architecture aligns with our 104-feature input. |

### 2.2 FinMamba (2025)

| Field | Detail |
|-------|--------|
| **Title** | FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement Prediction |
| **ArXiv** | [2502.06707](https://arxiv.org/abs/2502.06707) |
| **Year** | February 2025 |
| **Key Contribution** | Combines Mamba with GNN via: (1) Market-Aware Graph (MAG) module capturing inter-asset relationships conditioned on macro market dynamics, (2) Multi-Level Mamba (MLM) modeling micro- and macro-time dependencies. Adapts to evolving market conditions. |
| **Performance** | Effective on US and Chinese markets with low computational complexity. |
| **Relevance to AutoResearch** | **High.** MAG module could model cross-pair dependencies (EUR/USD, GBP/USD, etc.). MLM addresses our multi-horizon (1d, 5d) needs. Market-condition awareness aligns with our regime-aware evaluation. |

### 2.3 SAMBA: Graph-Mamba for Finance (2024-2025)

| Field | Detail |
|-------|--------|
| **Title** | Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction |
| **ArXiv** | [2410.03707](https://arxiv.org/abs/2410.03707) |
| **Year** | October 2024 (revised January 2025) |
| **Key Contribution** | Bidirectional Mamba block for long-term dependencies + adaptive graph convolution for inter-feature dependencies. Near-linear computational complexity. |
| **Relevance to AutoResearch** | **Medium-High.** Bidirectional Mamba aligns with our BiLSTM philosophy. Graph convolution could model feature interactions among our 104 features. |

### 2.4 xLSTM-Mixer (2024-2025)

| Field | Detail |
|-------|--------|
| **Title** | xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories |
| **ArXiv** | [2410.16928](https://arxiv.org/abs/2410.16928) |
| **Year** | October 2024 |
| **Venue** | NeurIPS 2025 |
| **Key Contribution** | Combines xLSTM (extended LSTM with exponential gating + revised memory) with mixing architecture. Three stages: (1) NLinear channel-independent forecast, (2) joint time-variate mixing via sLSTM, (3) view mixing to reconcile. Effectively uses longer lookback windows than baselines. |
| **Performance** | SOTA long-term forecasting while requiring very little memory. |
| **Relevance to AutoResearch** | **Very High.** Direct upgrade path from BiLSTM. The mixing architecture handles multivariate data natively. Longer lookback benefit relevant for our 60-step sequences. Very memory efficient for CPU deployment. |

### 2.5 xLSTMTime (2024)

| Field | Detail |
|-------|--------|
| **Title** | xLSTMTime: Long-term Time Series Forecasting With xLSTM |
| **ArXiv** | [2407.10240](https://arxiv.org/abs/2407.10240) |
| **Year** | July 2024 |
| **Key Contribution** | Adapts xLSTM for LTSF with exponential gating and higher-capacity memory structure. Surpasses SOTA models across multiple real-world datasets. |
| **Relevance to AutoResearch** | **High.** Simpler xLSTM adaptation than xLSTM-Mixer; good intermediate upgrade from BiLSTM. |

### 2.6 StoxLSTM (2025)

| Field | Detail |
|-------|--------|
| **Title** | StoxLSTM: A Stochastic Extended Long Short-Term Memory Network for Time Series Forecasting |
| **ArXiv** | [2509.01187](https://arxiv.org/abs/2509.01187) |
| **Year** | September 2025 |
| **Key Contribution** | Stochastic xLSTM within SSM framework. Integrates latent stochastic variables into recurrent units for deep latent temporal dynamics and uncertainty modeling. |
| **Performance** | Consistently outperforms SOTA baselines. |
| **Relevance to AutoResearch** | **High.** Built-in uncertainty quantification addresses our need for confidence-weighted trading signals. SSM + xLSTM hybrid directly relevant to our architecture. |

---

## 3. Transformer Variants for Time Series

### 3.1 iTransformer (Tsinghua, 2024)

| Field | Detail |
|-------|--------|
| **Title** | iTransformer: Inverted Transformers Are Effective for Time Series Forecasting |
| **ArXiv** | [2310.06625](https://arxiv.org/abs/2310.06625) |
| **Year** | October 2023 |
| **Venue** | ICLR 2024 Spotlight |
| **Key Contribution** | Inverts the standard transformer: applies attention across variates (channels) instead of time steps. Each variate is a token, enabling cross-variate dependency capture. Simple yet effective reformulation. |
| **Relevance to AutoResearch** | **High.** Our 104 features would each become a token, enabling the model to learn feature interactions (e.g., interest rate differentials influencing carry trade returns). Directly comparable to our PatchTST backbone. |

### 3.2 TimeXer (Tsinghua, 2024)

| Field | Detail |
|-------|--------|
| **Title** | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables |
| **ArXiv** | [2402.19072](https://arxiv.org/abs/2402.19072) |
| **Year** | February 2024 |
| **Venue** | NeurIPS 2024 |
| **Key Contribution** | Reconciles endogenous and exogenous information through patch-wise self-attention (endogenous) and variate-wise cross-attention (exogenous). Consistent SOTA on 12 benchmarks. Notable generality and scalability. |
| **Relevance to AutoResearch** | **Very High.** Our setup has clear endogenous (price returns) and exogenous (macro indicators, volatility, interest rates) variables. TimeXer's dual-attention design is tailor-made for this separation. Tested in the 918-experiment study (Section 6.3). |

### 3.3 TimeMixer++ (Tsinghua, 2025)

| Field | Detail |
|-------|--------|
| **Title** | TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis |
| **ArXiv** | [2410.16032](https://arxiv.org/abs/2410.16032) |
| **Year** | October 2024 (updated May 2025) |
| **Key Contribution** | Multi-resolution time imaging (MRTI) + time image decomposition (TID) + multi-scale mixing (MCM) + multi-resolution mixing (MRM). Dual-axis attention for seasonal/trend decomposition. SOTA across 8 time series tasks. |
| **Relevance to AutoResearch** | **High.** Multi-scale decomposition directly applicable to our multi-horizon (1d, 5d) prediction. Seasonal/trend separation useful for FX markets with carry/momentum components. |

### 3.4 TimeFormer (2025)

| Field | Detail |
|-------|--------|
| **Title** | TimeFormer: Transformer with Attention Modulation Empowered by Temporal Characteristics |
| **ArXiv** | [2510.06680](https://arxiv.org/abs/2510.06680) |
| **Year** | October 2025 |
| **Key Contribution** | Attention modulation driven by temporal characteristics. Improves over TimeMixer++, PatchMLP, TimeMixer, iTransformer, and PatchTST. |
| **Relevance to AutoResearch** | **Medium.** Incremental improvement over existing transformers we already benchmark. |

### 3.5 ModernTCN (2024)

| Field | Detail |
|-------|--------|
| **Title** | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis |
| **Venue** | ICLR 2024 |
| **Key Contribution** | Modernized TCN with much larger effective receptive fields. Pure convolution achieving SOTA across 5 time series tasks. Best mean rank (1.333) in the 918-experiment financial comparison study. |
| **Performance** | 75% first-place rate in multi-horizon financial forecasting across crypto, FX, equities. |
| **Relevance to AutoResearch** | **Very High.** Top performer in the only controlled multi-horizon financial forecasting comparison. Pure convolution is efficient on CPU. Strong inductive bias for sequential financial data. Not currently in our backbone registry. |

---

## 4. Multi-Scale / Multi-Horizon Architectures

### 4.1 MDMixer (2025)

| Field | Detail |
|-------|--------|
| **Title** | A Multi-scale Representation Learning Framework for Long-Term Time Series Forecasting |
| **ArXiv** | [2505.08199](https://arxiv.org/abs/2505.08199) |
| **Year** | May 2025 |
| **Key Contribution** | Multi-granularity Parallel Predictor (MPP) + Multi-granularity Iterative Mixing (MIM). Models trend and seasonal elements independently across multiple temporal granularities. 4.64% MAE improvement over TimeMixer. |
| **Relevance to AutoResearch** | **High.** Multi-granularity design maps directly to our 1-day/5-day horizons. Could extend to intraday horizons (deferred scope). |

### 4.2 N-HiTS (Nixtla, 2022)

| Field | Detail |
|-------|--------|
| **Title** | N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting |
| **ArXiv** | [2201.12886](https://arxiv.org/abs/2201.12886) |
| **Year** | January 2022 |
| **Venue** | AAAI 2023 |
| **Key Contribution** | Multi-rate data sampling + hierarchical interpolation. Each block specializes in a frequency band. Outperforms all transformers with less compute. |
| **Performance** | SOTA long-term forecasting; best results among all architectures in 918-experiment study (3rd tier though). |
| **Relevance to AutoResearch** | **Medium-High.** Hierarchical multi-rate design natural for our 1d/5d multi-horizon setup. Tested in controlled financial comparison. Lighter than transformers. |

---

## 5. Regime-Aware / Non-Stationary Methods

### 5.1 DTAF: Dual-Branch Temporal-Frequency Framework (2025)

| Field | Detail |
|-------|--------|
| **Title** | Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing |
| **ArXiv** | [2511.08229](https://arxiv.org/abs/2511.08229) |
| **Year** | November 2025 |
| **Key Contribution** | Dual-branch framework addressing non-stationarity in both temporal and frequency domains simultaneously. Temporal stabilization + frequency differencing. |
| **Relevance to AutoResearch** | **High.** FX returns are highly non-stationary with regime changes. Dual-domain approach could improve generalization across our 7 regime-diverse folds. |

### 5.2 DERITS: Deep Frequency Derivative Learning (2024)

| Field | Detail |
|-------|--------|
| **Title** | Deep Frequency Derivative Learning for Non-stationary Time Series Forecasting |
| **ArXiv** | [2407.00502](https://arxiv.org/abs/2407.00502) |
| **Year** | June 2024 |
| **Venue** | IJCAI 2024 |
| **Key Contribution** | Frequency Derivative Transformation (FDT) to create stationary frequency representations. Order-adaptive Fourier Convolution Network for frequency filtering. Parallel-stacked multi-order derivation and fusion. |
| **Relevance to AutoResearch** | **Medium-High.** Addresses the core challenge of non-stationarity in FX markets. Could be applied as a preprocessing module before any backbone. |

### 5.3 Frequency Adaptive Normalization (2024)

| Field | Detail |
|-------|--------|
| **Title** | Frequency Adaptive Normalization For Non-stationary Time Series Forecasting |
| **ArXiv** | [2409.20371](https://arxiv.org/abs/2409.20371) |
| **Year** | September 2024 |
| **Key Contribution** | Adaptive normalization in frequency domain for non-stationary series. Plug-in module compatible with any backbone. |
| **Relevance to AutoResearch** | **High.** Could be added to our existing backbones as a preprocessing step without architecture changes. |

---

## 6. FX-Specific Papers

### 6.1 EUR/USD Information Fusion with LLMs (2024)

| Field | Detail |
|-------|--------|
| **Title** | EUR-USD Exchange Rate Forecasting Based on Information Fusion with Large Language Models and Deep Learning Methods |
| **ArXiv** | [2408.13214](https://arxiv.org/abs/2408.13214) |
| **Year** | August 2024 |
| **Key Contribution** | IUS framework integrating unstructured text (news/analysis) with structured data (rates, indicators). LLMs for sentiment scoring and movement classification. Optuna-optimized Bi-LSTM for prediction. |
| **Performance** | MAE reduced by 10.69%, RMSE by 9.56% vs. best baseline. |
| **Relevance to AutoResearch** | **Very High.** Directly addresses EUR/USD prediction. Our BiLSTM backbone + sentiment features from LLMs could replicate this approach. Information fusion with our 104 features + text could add significant alpha. |

### 6.2 EUR/USD Text Mining with Pre-trained LMs (2024)

| Field | Detail |
|-------|--------|
| **Title** | EUR/USD Exchange Rate Forecasting incorporating Text Mining Based on Pre-trained Language Models and Deep Learning Methods |
| **ArXiv** | [2411.07560](https://arxiv.org/abs/2411.07560) |
| **Year** | November 2024 |
| **Key Contribution** | RoBERTa-Large for sentiment analysis + LDA topic modeling + PSO-optimized LSTM. Combines qualitative news data with quantitative features. |
| **Performance** | PSO-LSTM outperforms traditional econometric and ML baselines. |
| **Relevance to AutoResearch** | **High.** PSO hyperparameter optimization + text features applicable to our framework. |

### 6.3 Controlled Comparison of DL for Financial Forecasting (2026)

| Field | Detail |
|-------|--------|
| **Title** | A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments |
| **ArXiv** | [2603.16886](https://arxiv.org/abs/2603.16886) |
| **Year** | February 2026 |
| **Author** | Nabeel Ahmad Saidd |
| **Key Contribution** | Most rigorous controlled comparison to date: 9 architectures (Autoformer, DLinear, iTransformer, LSTM, ModernTCN, N-HiTS, PatchTST, TimesNet, TimeXer) across crypto/forex/equity at 4h and 24h horizons. 918 experiments with 5-stage protocol: fixed-seed Bayesian HPO, config freezing, multi-seed retraining, uncertainty aggregation, statistical validation. |
| **Key Results** | **ModernTCN: best mean rank 1.333, 75% first-place rate.** PatchTST: rank 2.000. Architecture explains nearly all performance variance. **Directional accuracy near 50% for all MSE-trained models** -- indicating MSE training insufficient for directional trading. Rankings stable across horizons despite 2-2.5x error amplification. |
| **Relevance to AutoResearch** | **Critical.** Directly applicable: same assets (forex), same horizons (daily-scale), same evaluation rigor. Key insight: MSE-trained models lack directional skill. Our Huber loss may partially address this, but we should consider directional/classification losses (cross-entropy on sign). ModernTCN should be added to our backbone registry. |

### 6.4 Hybrid Framework for Exchange Rate Prediction (2025)

| Field | Detail |
|-------|--------|
| **Title** | A hybrid framework of deep learning and traditional time series models for exchange rate prediction |
| **Venue** | ScienceDirect, 2025 |
| **Key Contribution** | SARIMA-LSTM hybrid outperforms standalone GRU, RNN, LSTM, ARIMA, SARIMA. Signal decomposition + deep learning combination. |
| **Relevance to AutoResearch** | **Medium.** Hybrid statistical+DL approach could complement our pure DL/GBM backbones. |

### 6.5 Spatio-Temporal GNNs for FX Markets (2025)

| Field | Detail |
|-------|--------|
| **Title** | Financial asset price prediction with graph neural network-based temporal deep learning models |
| **Venue** | Neural Computing and Applications, September 2025 |
| **Key Contribution** | Evaluates MTGNN, StemGNN, FourierGNN specifically on forex and crypto markets. First systematic comparison of spatio-temporal GNNs on FX. MTGNN noted to fail on exchange-rate data due to smaller graph size. |
| **Relevance to AutoResearch** | **High.** Direct evaluation on FX markets. Our 6 currency pairs form a natural graph. StemGNN or FourierGNN could model inter-pair dependencies. Caveat: small FX graph may limit GNN effectiveness (per MTGNN finding). |

### 6.6 Cross-Asset Hybrid ML Ensemble for Market Risk (2025)

| Field | Detail |
|-------|--------|
| **Title** | Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha Generation Using Hybrid Machine Learning Ensembles |
| **ArXiv** | [2510.22348](https://arxiv.org/abs/2510.22348) |
| **Year** | October 2025 |
| **Key Contribution** | Hybrid ML ensemble (neural networks + tree-based voting) predicting 5-day drawdowns across equities, fixed income, FX, commodities, volatility. |
| **Performance** | **Sharpe ratio: 0.51 over 2005-2025 backtest period.** |
| **Relevance to AutoResearch** | **High.** Cross-asset approach with FX component. 5-day horizon matches our ret_5d target. Sharpe 0.51 provides a realistic benchmark. Tree+neural hybrid aligns with our backbone diversity. |

---

## 7. LLM + Financial Data Combination

### 7.1 Re(Visiting) Time Series Foundation Models in Finance (2025)

| Field | Detail |
|-------|--------|
| **Title** | Re(Visiting) Time Series Foundation Models in Finance |
| **ArXiv** | [2511.18578](https://arxiv.org/abs/2511.18578) |
| **Year** | November 2025 |
| **Authors** | Eghbal Rahimikia, Hao Ni, Weiguan Wang |
| **Key Contribution** | **First comprehensive empirical study of TSFMs in global financial markets.** Tests zero-shot, fine-tuning, and pre-training from scratch. Uses daily excess returns across diverse markets. |
| **Key Finding** | **Off-the-shelf pretrained TSFMs perform poorly in zero-shot AND fine-tuning settings for finance. Models pretrained from scratch on financial data achieve substantial forecasting and economic improvements.** Dataset size, synthetic augmentation, and HPO further enhance performance. |
| **Relevance to AutoResearch** | **Critical.** Validates our approach of training from scratch on FX data rather than relying on pretrained TSFMs. Suggests we should (1) increase training data size, (2) use synthetic data augmentation, (3) intensive HPO. Our LFM2.5 frozen backbone approach may underperform scratch-trained smaller models. |

### 7.2 LoFT-LLM (2025-2026)

| Field | Detail |
|-------|--------|
| **Title** | LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models |
| **ArXiv** | [2512.20002](https://arxiv.org/abs/2512.20002) |
| **Year** | December 2025 (revised January 2026) |
| **Key Contribution** | Three-stage pipeline: (1) Patch Low-Frequency Forecasting Module (PLFM) for stable trends, (2) residual learner for high-frequency variations, (3) fine-tuned LLM refining predictions via structured natural language prompts with domain knowledge. |
| **Performance** | Outperforms strong baselines in full-data and few-shot regimes on financial and energy datasets. Superior accuracy, robustness, interpretability. |
| **Relevance to AutoResearch** | **High.** Low-frequency / high-frequency decomposition relevant for FX (carry = low-freq, momentum = high-freq). LLM refinement adds contextual intelligence. Our Claude API optimizer loop could incorporate LLM-based prediction refinement. |

### 7.3 LLM4FTS / GPT4FTS (2025-2026)

| Field | Detail |
|-------|--------|
| **Title** | Beyond Fixed Patches: Enhancing GPTs for Financial Prediction with Adaptive Segmentation and Learnable Wavelets |
| **ArXiv** | [2505.02880](https://arxiv.org/abs/2505.02880) |
| **Year** | May 2025 (revised January 2026) |
| **Key Contribution** | GPT4FTS: Dynamic patch segmentation via K-means++ on DTW distance. Learnable wavelet transform replacing traditional DWT with adaptive parameter matrix convolution. Two-stage training: Next-Patch Pre-Training + Multi-Resolution Fine-Tuning. First to enhance LLMs for scale-invariant financial patterns. |
| **Performance** | Validated on CSI 300, CSI 500, S&P 500, NASDAQ 100. Higher annualized returns and better risk-adjusted metrics vs. SOTA. |
| **Relevance to AutoResearch** | **High.** Adaptive segmentation better than fixed patches (our PatchTST uses fixed patch_len=12). Learnable wavelets could capture FX regime transitions. Scale-invariant patterns relevant across our 1d/5d horizons. |

### 7.4 WaveLSFormer (2026)

| Field | Detail |
|-------|--------|
| **Title** | A Learnable Wavelet Transformer for Long-Short Equity Trading and Risk-Adjusted Return Optimization |
| **ArXiv** | [2601.13435](https://arxiv.org/abs/2601.13435) |
| **Year** | January 2026 |
| **Authors** | Shuozhe Li, Du Cheng, Leqi Liu |
| **Key Contribution** | End-to-end learnable wavelet filter bank front-end + transformer + direct portfolio output. Low-Guided High-Frequency Injection (LGHI) module. Trained on trading objective with risk-aware regularization (not MSE). |
| **Performance** | **Sharpe ratio: 2.157 +/- 0.166** across 6 industry groups, 5 years hourly data, 10 random seeds. Cumulative return: 0.607 +/- 0.045. |
| **Relevance to AutoResearch** | **Very High.** Sharpe 2.157 is an exceptional result. Key insight: training on trading objective (not MSE) is critical -- directly validates adding a directional/Sharpe loss to our training. Wavelet decomposition + transformer is a powerful combination for FX. |

---

## 8. Retrieval-Augmented Approaches for Time Series

### 8.1 TS-RAG (2025)

| Field | Detail |
|-------|--------|
| **Title** | TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster |
| **ArXiv** | [2503.07649](https://arxiv.org/abs/2503.07649) |
| **Year** | March 2025 |
| **Venue** | NeurIPS 2025 |
| **Key Contribution** | Retrieval-augmented framework for TSFMs. Pre-trained time series encoders retrieve semantically relevant segments from knowledge base. Adaptive Retrieval Mixer (ARM) dynamically fuses retrieved patterns with TSFM's internal representation. No task-specific fine-tuning required. |
| **Performance** | Up to 6.84% improvement over existing TSFMs across diverse domains with built-in interpretability. |
| **Relevance to AutoResearch** | **High.** Our deferred "ChromaDB regime retrieval" feature (listed in out-of-scope) aligns perfectly with TS-RAG. Could retrieve similar historical market regimes to enhance predictions. ARM module is the key innovation for fusion. |

### 8.2 Retrieval Augmented Forecasting (RAF) (2024)

| Field | Detail |
|-------|--------|
| **Title** | Retrieval Augmented Time Series Forecasting |
| **ArXiv** | [2411.08249](https://arxiv.org/abs/2411.08249) |
| **Year** | November 2024 |
| **Key Contribution** | RAG framework specifically for TSFMs. Substantially enhances accuracy on out-of-domain datasets where TSFMs lack domain-specific information. |
| **Relevance to AutoResearch** | **Medium-High.** Financial data is "out-of-domain" for most general TSFMs, making RAG particularly valuable. |

### 8.3 FinSrag: RAG for Financial Time Series (2025)

| Field | Detail |
|-------|--------|
| **Title** | Retrieval-augmented Large Language Models for Financial Time Series Forecasting |
| **ArXiv** | [2502.05878](https://arxiv.org/abs/2502.05878) |
| **Year** | February 2025 (revised June 2025) |
| **Authors** | Mengxi Xiao et al. |
| **Key Contribution** | First RAG framework with domain-specific retriever (FinSeer) for financial time series. Candidate selection refined by LLM feedback + similarity-driven training. StockLLM (1B params) backbone for prediction. Expanded retrieval dataset with additional financial indicators. |
| **Performance** | FinSeer outperforms textual retrievers and distance-based retrieval for financial prediction. |
| **Relevance to AutoResearch** | **High.** Domain-specific retrieval for finance is directly applicable. FinSeer's LLM-refined retrieval could identify relevant historical FX regimes for our walk-forward folds. |

### 8.4 TimeART: Agentic Time Series Reasoning (2026)

| Field | Detail |
|-------|--------|
| **Title** | TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation |
| **ArXiv** | [2601.13653](https://arxiv.org/abs/2601.13653) |
| **Year** | January 2026 |
| **Key Contribution** | Agentic framework fusing 21 analysis tools (statistical methods + TSFMs) with LLM reasoning. TimeToolBench: 100k ReAct-style tool-use trajectories. Four-stage training strategy with self-reflection. |
| **Relevance to AutoResearch** | **Medium.** Conceptually similar to our Claude API autonomous optimizer loop. Could inform how we structure the agent's tool selection across our 11 backbones. |

---

## 9. Graph Neural Networks for Multi-Asset Prediction

### 9.1 Temporal GAT for Volatility Spillovers (2024-2025)

| Field | Detail |
|-------|--------|
| **Title** | Dynamic graph neural networks for enhanced volatility prediction in financial markets |
| **ArXiv** | [2410.16858](https://arxiv.org/abs/2410.16858) |
| **Year** | October 2024 |
| **Key Contribution** | Temporal Graph Attention Network combining GCN and GAT for temporal + structural dynamics of volatility spillovers across 8 market indices. Outperforms GARCH models. |
| **Relevance to AutoResearch** | **Medium-High.** Volatility spillovers between currency pairs are well-documented. Could model USD/JPY volatility feeding into EUR/USD predictions. |

### 9.2 Graph Attention RL for Portfolio Optimization (2026)

| Field | Detail |
|-------|--------|
| **Title** | Graph attention-based heterogeneous multi-agent deep reinforcement learning for adaptive portfolio optimization |
| **Venue** | Scientific Reports, 2026 |
| **Key Contribution** | Graph attention models time-varying asset correlations. Three heterogeneous agents: risk assessment, return prediction, market environment perception. Adaptive optimization strategy. |
| **Relevance to AutoResearch** | **Medium.** Multi-agent approach with market environment perception relevant for our regime-aware evaluation. Currently out of scope (portfolio optimization). |

### 9.3 MTGNN / StemGNN / FourierGNN on FX (2025)

| Field | Detail |
|-------|--------|
| **Title** | Financial asset price prediction with graph neural network-based temporal deep learning models |
| **Venue** | Neural Computing and Applications, 2025 |
| **Key Contribution** | Systematic comparison of three spatio-temporal GNN architectures (MTGNN, StemGNN, FourierGNN) on forex and crypto. StemGNN uses Graph Fourier Transform + DFT. FourierGNN uses Fourier Graph Operator. **MTGNN fails to improve on exchange-rate data due to small graph size.** |
| **Relevance to AutoResearch** | **High but cautionary.** Direct FX evaluation shows GNNs struggle with small currency graphs (our 6 pairs). FourierGNN and StemGNN may work better than MTGNN for small graphs. |

### 9.4 Equity Correlation Forecasting with Hybrid Transformer-GNN (2026)

| Field | Detail |
|-------|--------|
| **Title** | Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network |
| **ArXiv** | [2601.04602](https://arxiv.org/abs/2601.04602) |
| **Year** | January 2026 |
| **Key Contribution** | Hybrid Transformer-GNN for correlation forecasting. Relevant for cross-asset dependency modeling. |
| **Relevance to AutoResearch** | **Medium.** Cross-currency correlation forecasting could enhance our multi-pair prediction. |

---

## 10. Walk-Forward Evaluation & Lopez de Prado Methods

### 10.1 Interpretable Hypothesis-Driven Trading with Walk-Forward Validation (2025)

| Field | Detail |
|-------|--------|
| **Title** | Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals |
| **ArXiv** | [2512.12924](https://arxiv.org/abs/2512.12924) |
| **Year** | December 2025 |
| **Authors** | Gagan Deep, Akash Deep, William Lamptey |
| **Key Contribution** | Rigorous walk-forward framework with: strict information set discipline, rolling window validation across 34 independent test periods, complete interpretability via natural language hypothesis explanations, realistic transaction costs and position constraints. Combined with RL for signal exploitation. |
| **Performance** | Annualized return: 0.55%. **Sharpe ratio: 0.33.** Maximum drawdown: -2.76%. Beta: 0.058 (market-neutral). Strong regime dependence: +0.60% quarterly in high-vol (2020-2024), -0.16% in stable (2015-2019). |
| **Relevance to AutoResearch** | **Very High.** Most methodologically similar to our approach. Their 34 rolling windows vs. our 7 regime-aware folds. Sharpe 0.33 is a realistic benchmark for rigorous walk-forward on OHLCV data. Key finding: regime dependence of performance validates our regime-aware fold design. |

### 10.2 Backtest Overfitting Comparison (2024-2025)

| Field | Detail |
|-------|--------|
| **Title** | Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment |
| **Venue** | Knowledge-Based Systems, 2024 |
| **Authors** | Hamid R. Arian, Daniel Norouzi M., Luis A. Seco |
| **Key Contribution** | Comprehensive comparison of CPCV, walk-forward, and standard CV in synthetic controlled environment. **CPCV shows marked superiority in mitigating overfitting** with lower Probability of Backtest Overfitting (PBO) and superior Deflated Sharpe Ratio (DSR). |
| **Relevance to AutoResearch** | **Very High.** Directly validates our use of DSR and PSR. Suggests upgrading from walk-forward to CPCV for more robust evaluation. Our 7-fold design could be augmented with combinatorial splits. |

### 10.3 skfolio: Combinatorial Purged CV Implementation

| Field | Detail |
|-------|--------|
| **Source** | [skfolio](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html) |
| **Key Contribution** | Production-ready Python implementation of CPCV with purge and embargo. Drop-in replacement for sklearn cross-validators. |
| **Relevance to AutoResearch** | **High.** Could augment our existing 7-fold walk-forward with CPCV to generate a distribution of Sharpe ratios rather than point estimates. |

---

## 11. Ensemble & Stacking Methods

### 11.1 Multi-layer Stack Ensembles (2025)

| Field | Detail |
|-------|--------|
| **Title** | Multi-layer Stack Ensembles for Time Series Forecasting |
| **ArXiv** | [2511.15350](https://arxiv.org/abs/2511.15350) |
| **Year** | November 2025 |
| **Key Contribution** | Evaluates 33 ensemble models across 50 real-world datasets. Stacking consistently improves accuracy over simple linear combinations. No single stacker performs best universally. |
| **Relevance to AutoResearch** | **High.** Our 11 backbones are perfect candidates for stacking. Currently out of scope but should be prioritized -- meta-learning across backbone predictions could capture regime-dependent strengths. |

### 11.2 Bagging/Boosting/Stacking with Foundation Models (2025)

| Field | Detail |
|-------|--------|
| **Title** | Enhancing Transformer-Based Foundation Models for Time Series Forecasting via Bagging, Boosting and Statistical Ensembles |
| **ArXiv** | [2508.16641](https://arxiv.org/abs/2508.16641) |
| **Year** | August 2025 |
| **Key Contribution** | Unified methodology for integrating bagging, stacking, prediction intervals, and residual boosting with Transformer-based TSFMs. Linear stacking of Lag-Llama + AutoGluon achieves lowest MSE. |
| **Relevance to AutoResearch** | **High.** Could stack our PatchTST, Mamba2, LFM2.5 predictions with GBM meta-learner. Residual boosting: train GBM on transformer residuals. |

---

## 12. Critical Perspectives

### 12.1 Universal TSFMs Rest on a Category Error (2026)

| Field | Detail |
|-------|--------|
| **Title** | Position: Universal Time Series Foundation Models Rest on a Category Error |
| **ArXiv** | [2602.05287](https://arxiv.org/abs/2602.05287) |
| **Year** | February 2026 |
| **Key Contribution** | Argues that "universal" TSFMs confuse a structural container (time series) with a semantic modality. Introduces "Autoregressive Blindness Bound" proving history-only models cannot predict intervention-driven regime shifts. Advocates Causal Control Agent paradigm with specialized solvers. |
| **Relevance to AutoResearch** | **Very High (conceptual).** Theoretically validates why off-the-shelf TSFMs fail on finance (per Section 7.1). Supports our approach of training specialized models on FX data. The "Causal Control Agent" concept aligns with our Claude API autonomous optimizer. Regime shifts in FX (central bank interventions, policy changes) are exactly the "intervention-driven" shifts this paper identifies as fundamentally unpredictable by autoregressive models. |

### 12.2 How Foundational Are Foundation Models? (2025)

| Field | Detail |
|-------|--------|
| **Title** | How Foundational are Foundation Models for Time Series Forecasting? |
| **ArXiv** | [2510.00742](https://arxiv.org/abs/2510.00742) |
| **Year** | October 2025 |
| **Key Contribution** | Critical examination of TSFM claims. Questions whether pretrained models truly generalize or are simply large capacity models that benefit from scale. |
| **Relevance to AutoResearch** | **Medium.** Tempers expectations for zero-shot TSFM approaches. Supports our fine-tuning/scratch-training strategy. |

---

## 13. Comparison Matrix

### 13.1 Foundation Models Comparison

| Model | Params | Architecture | Multivariate | Covariates | Finance-Specific | Zero-Shot FX | Fine-Tune | CPU Feasible | Year |
|-------|--------|-------------|-------------|-----------|-----------------|-------------|-----------|-------------|------|
| **Chronos-2** | 120M | Encoder (T5) | Yes | Yes | No | Possible | Yes | Marginal | 2025 |
| **TimesFM 2.5** | 200M | Decoder-only | Limited | No | No | Poor* | Yes | No | 2025 |
| **Moirai 2.0** | Small | Decoder-only | Yes | Yes | No | Possible | Yes | Yes | 2025 |
| **TiRex** | 35M | xLSTM | Limited | No | No | Possible | Yes | **Yes** | 2025 |
| **FlowState** | 9.1M | SSM+FuncBasis | Yes | Yes | No | Possible | Yes | **Yes** | 2025 |
| **TabPFN-TS** | 11M | TabPFN | Yes | **Yes** | No | Yes | N/A | **Yes** | 2025 |
| **Kronos** | Variable | Decoder (OHLCV) | Yes | OHLCV | **Yes** | **Yes** | Yes | Possible | 2025 |
| **Lag-Llama** | Medium | Decoder (LLaMA) | No | Lags | No | Moderate | Yes | No | 2024 |
| **MOMENT** | Medium | Encoder (masked) | No | No | No | Moderate | Yes | Possible | 2024 |

*Per Re(Visiting) paper: off-the-shelf TSFMs perform poorly on financial data.

### 13.2 Architecture Comparison for Financial Forecasting

| Architecture | Mean Rank (918-exp) | Dir. Accuracy | Multi-Horizon | Regime-Aware | Complexity | In Our Registry |
|-------------|--------------------:|:-------------:|:-------------:|:------------:|:----------:|:---------------:|
| **ModernTCN** | **1.333** | ~50% | Yes | No | O(n) | **No** |
| **PatchTST** | 2.000 | ~50% | Yes | No | O(n*p) | Yes |
| **TimeXer** | ~3 (est.) | ~50% | Yes | Partial | O(n*p) | **No** |
| **iTransformer** | ~4 (est.) | ~50% | Yes | No | O(d^2) | **No** |
| **N-HiTS** | ~4 (est.) | ~50% | **Native** | No | O(n) | **No** |
| **DLinear** | ~5 (est.) | ~50% | Yes | No | O(1) | **No** |
| **LSTM** | ~5 (est.) | ~50% | Yes | No | O(n) | Yes (Bi) |
| **Informer** | Not tested | ~50% | Yes | No | O(n log n) | Yes |
| **Mamba2** | Not tested | ~50% | Yes | No | O(n) | Yes |

### 13.3 Relevance-Priority Matrix for AutoResearch

| Approach | Relevance | Expected Effort | Expected Sharpe Lift | Priority |
|----------|:---------:|:---------------:|:-------------------:|:--------:|
| Add ModernTCN backbone | Very High | Low (pure conv) | Moderate | **P0** |
| Add xLSTM-Mixer backbone | Very High | Medium | Moderate-High | **P0** |
| Directional loss (not MSE/Huber) | Critical | Low (loss swap) | **High** | **P0** |
| Add TimeXer backbone | Very High | Medium | Moderate | **P1** |
| Add iTransformer backbone | High | Medium | Moderate | **P1** |
| Kronos zero-shot baseline | Very High | Low (inference) | Unknown | **P1** |
| Ensemble/stacking across backbones | High | Medium | Moderate-High | **P1** |
| Wavelet decomposition preprocessing | High | Medium | Moderate | **P2** |
| TS-RAG regime retrieval | High | High | Unknown | **P2** |
| Non-stationary normalization (FAN/DTAF) | High | Low (plug-in) | Low-Moderate | **P2** |
| Synthetic data augmentation | High | Medium | Moderate | **P2** |
| Text/sentiment feature fusion | High | High | Moderate | **P3** |
| GNN for cross-pair modeling | Medium-High | High | Low-Moderate | **P3** |
| CPCV evaluation upgrade | Very High | Medium | N/A (eval) | **P2** |
| FinMamba (Mamba+GNN) | High | High | Unknown | **P3** |

---

## 14. Recommendations for AutoResearch

### 14.1 Immediate Actions (P0)

**1. Add a directional/classification loss component.**
The 918-experiment study (Section 6.3) conclusively shows that MSE-trained models achieve ~50% directional accuracy on financial data. Since our primary metric is Sharpe ratio from a sign(pred)*actual strategy, we are fundamentally misaligned: we optimize Huber loss but evaluate directional accuracy. Options:
- Add cross-entropy loss on sign(return) as auxiliary loss
- Use a Sharpe-ratio-aware loss (cf. WaveLSFormer achieving Sharpe 2.157 with risk-aware training objective)
- Weighted combination: `loss = alpha * huber(pred, actual) + beta * bce(sign(pred), sign(actual))`

**2. Add ModernTCN backbone.**
Best-performing architecture in the only rigorous controlled multi-horizon financial forecasting comparison. Pure convolution is efficient on CPU. Large effective receptive field captures long-range dependencies without attention. Implementation available in the Time-Series-Library (thuml/Time-Series-Library).

**3. Add xLSTM-Mixer backbone.**
Natural upgrade from BiLSTM with exponential gating and revised memory structure. NeurIPS 2025. Handles multivariate data natively through mixing. Very memory efficient. Code at mauricekraus/xlstm-mixer.

### 14.2 High-Priority Additions (P1)

**4. Add TimeXer backbone.**
Explicitly designed for exogenous variables (our macro indicators, volatility features). Patch-wise self-attention for endogenous + variate-wise cross-attention for exogenous. NeurIPS 2024.

**5. Add iTransformer backbone.**
Inverted attention across variates enables learning feature interactions (interest rate differentials, carry trade signals). ICLR 2024 Spotlight. Direct comparison to our PatchTST.

**6. Test Kronos zero-shot on our FX data.**
Only TSFM pretrained on financial data including forex. AAAI 2026. +93% RankIC over generic TSFMs. Free zero-shot evaluation to establish a pretrained-model baseline. Code at shiyu-coder/Kronos.

**7. Implement backbone ensembling.**
Our 11 backbones produce diverse predictions. Stack with a lightweight meta-learner (logistic regression or small GBM on backbone predictions). The multi-layer stack ensemble paper shows consistent improvement over any single model.

### 14.3 Medium-Priority Improvements (P2)

**8. Upgrade evaluation with CPCV.**
Combinatorial Purged Cross-Validation generates a distribution of Sharpe ratios rather than 7 point estimates. Shown to have lower Probability of Backtest Overfitting. Use skfolio's CombinatorialPurgedCV alongside our existing walk-forward.

**9. Add frequency-domain preprocessing.**
- Frequency Adaptive Normalization: plug-in module, no architecture changes
- DERITS-style frequency derivative transformation for stationarization
- Learnable wavelet decomposition (from WaveLSFormer / GPT4FTS)

**10. Synthetic data augmentation.**
Re(Visiting) paper shows synthetic data improves TSFM performance on financial data. Generate synthetic FX returns preserving statistical properties (fat tails, volatility clustering, mean-reversion).

**11. Implement TS-RAG regime retrieval.**
Currently in our deferred scope (ChromaDB regime retrieval). TS-RAG provides the architecture: encode historical regime embeddings, retrieve similar regimes during prediction, fuse via Adaptive Retrieval Mixer.

### 14.4 Longer-Term Research (P3)

**12. Text/sentiment feature fusion.**
EUR/USD-specific papers show 10%+ improvement from LLM sentiment + news integration. Add RoBERTa-based sentiment scores from FX news as additional features.

**13. Cross-pair GNN modeling.**
While GNNs struggle on small currency graphs (MTGNN finding), FourierGNN or StemGNN may capture cross-pair dependencies (e.g., EUR/GBP and GBP/USD jointly influencing EUR/USD). Our 6 pairs + correlations form a small but dense graph.

**14. Explore Mamba-3 as Mamba2 upgrade.**
Complex-valued state updates may capture oscillatory FX patterns. Multi-input architecture for our 104 features. Direct drop-in replacement for our Mamba2 backbone.

### 14.5 Key Strategic Insights from the Literature

| Insight | Source | Implication for AutoResearch |
|---------|--------|------------------------------|
| Off-the-shelf TSFMs fail on finance | Re(Visiting), 2025 | Our scratch-trained approach is correct. LFM2.5 frozen backbone may underperform. |
| MSE training yields ~50% directional accuracy | 918-experiment study, 2026 | **Switching to directional loss is the single highest-impact change.** |
| Architecture matters more than scale | 918-experiment study, 2026 | Adding ModernTCN/xLSTM-Mixer more valuable than scaling existing models. |
| Domain-specific pretraining dominates | Kronos, 2025 | Financial-specific pretraining (Kronos) >> general pretraining (TimesFM). |
| Regime dependence is fundamental | Walk-forward paper, 2025; Position paper, 2026 | Our regime-aware 7-fold design is well-justified. Consider regime-conditioned models. |
| Ensembling consistently helps | Stack ensemble paper, 2025 | Stacking our 11 backbones should be prioritized. |
| Wavelet/frequency methods + trading objectives work | WaveLSFormer, 2026 | Sharpe 2.157 achieved with wavelet+transformer+trading loss. |
| Small models can beat large ones | FlowState (9.1M), TiRex (35M), TabPFN-TS (11M) | Our CPU constraint is not a fundamental limitation. |
| CPCV superior to walk-forward for overfitting control | Backtest overfitting paper, 2024 | Augment our evaluation with CPCV. |

---

## References (Chronological)

1. N-HiTS (2022). arXiv:2201.12886. AAAI 2023.
2. Lag-Llama (2023). arXiv:2310.08278.
3. iTransformer (2023). arXiv:2310.06625. ICLR 2024 Spotlight.
4. TimesFM (2023). arXiv:2310.10688. ICML 2024.
5. MOMENT (2024). arXiv:2402.03885.
6. TimeXer (2024). arXiv:2402.19072. NeurIPS 2024.
7. UniTS (2024). arXiv:2403.00131. NeurIPS 2024.
8. ModernTCN (2024). ICLR 2024.
9. DERITS (2024). arXiv:2407.00502. IJCAI 2024.
10. xLSTMTime (2024). arXiv:2407.10240.
11. EUR/USD Info Fusion (2024). arXiv:2408.13214.
12. Frequency Adaptive Normalization (2024). arXiv:2409.20371.
13. Moirai-MoE (2024). arXiv:2410.10469.
14. GIFT-Eval Benchmark (2024). arXiv:2410.10393.
15. TimeMixer++ (2024). arXiv:2410.16032.
16. xLSTM-Mixer (2024). arXiv:2410.16928. NeurIPS 2025.
17. Temporal GAT for Volatility (2024). arXiv:2410.16858.
18. EUR/USD Text Mining (2024). arXiv:2411.07560.
19. RAF: Retrieval Augmented Forecasting (2024). arXiv:2411.08249.
20. Backtest Overfitting Comparison (2024). Knowledge-Based Systems.
21. TabPFN-TS (2025). arXiv:2501.02945.
22. FinSrag (2025). arXiv:2502.05878.
23. FinMamba (2025). arXiv:2502.06707.
24. TS-RAG (2025). arXiv:2503.07649. NeurIPS 2025.
25. Deep Learning for TS Survey (2025). arXiv:2503.10198.
26. LLM4FTS / GPT4FTS (2025). arXiv:2505.02880.
27. MDMixer (2025). arXiv:2505.08199.
28. TiRex (2025). arXiv:2505.23719. NeurIPS 2025.
29. Kronos (2025). arXiv:2508.02739. AAAI 2026.
30. FlowState (2025). arXiv:2508.05287. NeurIPS 2025 Workshop.
31. StoxLSTM (2025). arXiv:2509.01187.
32. Ensemble with Foundation Models (2025). arXiv:2508.16641.
33. TimeFormer (2025). arXiv:2510.06680.
34. How Foundational Are TSFMs? (2025). arXiv:2510.00742.
35. TSFM Benchmarking Challenges (2025). arXiv:2510.13654.
36. Chronos-2 (2025). arXiv:2510.15821.
37. Cross-Asset ML Ensemble (2025). arXiv:2510.22348.
38. DTAF (2025). arXiv:2511.08229.
39. Multi-layer Stack Ensembles (2025). arXiv:2511.15350.
40. Moirai 2.0 (2025). arXiv:2511.11698.
41. Re(Visiting) TSFMs in Finance (2025). arXiv:2511.18578.
42. Walk-Forward Validation Framework (2025). arXiv:2512.12924.
43. LoFT-LLM (2025). arXiv:2512.20002.
44. WaveLSFormer (2026). arXiv:2601.13435.
45. TimeART (2026). arXiv:2601.13653.
46. Position: Category Error (2026). arXiv:2602.05287.
47. 918-Experiment Comparison (2026). arXiv:2603.16886.
48. Mamba-3 (2026). arXiv:2603.15569.
49. Spatio-Temporal GNNs on FX (2025). Neural Computing and Applications.
50. Graph Attention RL Portfolio (2026). Scientific Reports.

---

*Survey compiled 2026-04-06 for AutoResearch FX prediction project. 50 papers reviewed across 10 categories.*
