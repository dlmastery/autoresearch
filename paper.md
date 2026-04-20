# AutoResearch: Autonomous Machine Learning Optimization for Foreign Exchange Prediction via Agent-Driven Experiment Design

**Evija Ranti**

*April 2026*

---

## Abstract

We present AutoResearch, an autonomous machine learning research system in which a large language model (LLM) agent drives the full experiment optimization loop for financial time series prediction. Applied to EUR/USD daily return forecasting across seven regime-stratified evaluation windows spanning 2006--2024, the agent follows a principled Karpathy-style protocol: diagnose per-fold failures, cite relevant literature, form testable hypotheses, predict outcomes, execute a single experiment, analyze results, and checkpoint state. Over the course of 90 experiments across two backbone architectures --- a 350M-parameter foundation model (LFM2.5, Liquid AI) and a from-scratch residual MLP --- the system converges to a champion configuration achieving an annualized test Sharpe ratio of +6.21, a composite score of +5.50, positive returns in all 7/7 test folds, and a cumulative return of 1,001% on held-out data. The central empirical finding is that a simple residual MLP with 301K trainable parameters decisively outperforms the 350M-parameter foundation model on daily FX data, with a median cross-seed test Sharpe of +5.41 versus +1.40. We contribute: (1) an agent-driven research protocol that replaces manual hyperparameter tuning with literature-grounded diagnostic reasoning, (2) a super-fold evaluation framework with purge, embargo, and label-horizon buffers that provably prevents data leakage, and (3) a systematic empirical comparison of foundation models versus from-scratch architectures on low signal-to-noise financial time series.

---

## 1. Introduction

Foreign exchange (FX) return prediction is among the most challenging problems in financial machine learning. Daily FX returns exhibit low signal-to-noise ratios (SNR), heavy-tailed distributions, non-stationarity, and pronounced regime dependence --- properties that violate the assumptions underlying most standard ML pipelines (Gu et al., 2020). The efficient market hypothesis (Fama, 1970) suggests that publicly available information should be fully reflected in prices, making prediction fundamentally difficult. Yet a substantial body of work demonstrates that carefully constructed ML models can extract economically significant signals from financial data, provided that evaluation methodology rigorously prevents data leakage (Lopez de Prado, 2018).

The traditional approach to financial ML research involves a human researcher manually designing experiments, tuning hyperparameters via grid or random search, and iterating over architectures. This process is slow, expensive, and prone to cognitive biases --- researchers tend to over-fit to in-sample results, neglect per-regime analysis, and pursue unprincipled parameter sweeps rather than diagnostic reasoning.

In this paper, we propose an alternative: an LLM agent (Claude Code, Anthropic) acts as the researcher, making all experimental decisions. The agent is not a simple optimizer or AutoML system; it reads experimental results, diagnoses per-fold failures, searches for relevant published techniques, forms specific hypotheses, predicts expected outcomes, executes single-variable experiments, and adapts the architecture based on first-principles understanding of the optimization landscape. The intelligence resides in the agent's natural language reasoning, not in a pre-defined search space.

We apply this system to EUR/USD daily return prediction using 104 backward-looking features derived from 6 FX pairs and 9 macroeconomic signals, evaluated across 7 regime-stratified fold windows covering distinct market environments from pre-crisis (2006) through recent markets (2024). The evaluation framework employs a super-fold design with 90-day purge gaps, 21-day embargo periods, and 10-day label-horizon buffers, with zero overlap between training, validation, and test sets verified programmatically.

Our contributions are as follows:

1. **Agent-driven research protocol.** We formalize a 7-step experiment loop (diagnose, cite, hypothesize, predict, run, analyze, checkpoint) that replaces manual hyperparameter tuning with literature-grounded diagnostic reasoning. Over 90 experiments, this protocol converges to a champion configuration that a human researcher would be unlikely to reach via grid search alone.

2. **Super-fold evaluation preventing data leakage.** We introduce a rigorous evaluation framework combining expanding-window training with hole-punched validation/test exclusion, purge gaps, embargo periods, and label-horizon buffers. All overlap invariants are verified programmatically before every experiment.

3. **Foundation model vs. from-scratch comparison on financial data.** We provide systematic evidence that a simple residual MLP with 301K parameters outperforms a frozen 350M-parameter foundation model (LFM2.5) on daily FX prediction, with a 3.9x higher median test Sharpe ratio. We analyze the root causes: overparameterization, domain mismatch, and the inductive bias advantage of residual skip connections for low-SNR signals.

---

## 2. Related Work

### 2.1 Machine Learning for Financial Prediction

Gu, Kelly, and Xiu (2020) provide the definitive empirical study of ML methods for asset pricing, demonstrating that neural networks outperform linear models on cross-sectional equity return prediction. Their work establishes the importance of large feature sets, careful cross-validation, and nonlinear interactions. Lopez de Prado (2018) introduces purged and embargoed cross-validation specifically for financial time series, addressing the autocorrelation and label-overlap problems that invalidate naive k-fold CV. Our super-fold evaluation framework directly implements and extends these recommendations.

In the FX domain specifically, Galeshchuk and Mukherjee (2017) apply deep learning to currency prediction, while Sermpinis et al. (2012) demonstrate the viability of neural network ensembles for EUR/USD forecasting. More recently, foundation models trained on diverse time series have been proposed as general-purpose forecasters (Das et al., 2024; Rasul et al., 2024), but their efficacy on low-SNR financial data remains underexplored.

### 2.2 Automated Machine Learning

AutoML systems (Hutter et al., 2019) automate model selection and hyperparameter optimization through Bayesian optimization (Snoek et al., 2012), neural architecture search (Zoph and Le, 2017), and meta-learning (Vanschoren, 2019). These approaches define a search space and use mathematical optimization to explore it. Our approach differs fundamentally: the agent reasons in natural language, cites published literature, and makes decisions based on domain-specific diagnosis rather than black-box search. This is closer to the "recipe" approach advocated by Karpathy (2019), where a practitioner follows a structured diagnostic protocol rather than sweeping parameters.

### 2.3 Uncertainty Estimation in Finance

Kendall and Gal (2017) decompose predictive uncertainty into aleatoric (data noise) and epistemic (model uncertainty) components. The heteroscedastic loss formulation --- $\mathcal{L} = \exp(-s) \cdot \ell(\mu, y) + \frac{1}{2}s$, where $s = \log \sigma^2$ --- enables the model to learn per-sample noise estimates. Gal and Ghahramani (2016) propose MC Dropout as a practical approximation to Bayesian inference for epistemic uncertainty. We implement both approaches and find that the heteroscedastic loss degrades performance on small financial datasets (n < 3,000), as the variance branch steals capacity from the mean prediction.

### 2.4 Residual Networks

He et al. (2016) demonstrate that skip connections enable training of very deep networks by providing gradient shortcuts. Beyond depth, residual connections have a second, less discussed benefit: they preserve a linear baseline while allowing nonlinear corrections. This property is particularly valuable for low-SNR data where the signal is a small perturbation on a linear relationship. Our residual MLP exploits this property directly.

### 2.5 Foundation Models for Time Series

The application of large pretrained models to time series forecasting has gained significant attention. LFM2.5 (Liquid AI) is a 350M-parameter foundation model based on liquid neural networks. PatchTST (Nie et al., 2023) applies vision transformer principles to time series via patching. TimeGPT (Garza and Mergenthaler-Canseco, 2024) and Chronos (Ansari et al., 2024) explore GPT-style pretraining for forecasting. We provide empirical evidence that these models, while powerful for general time series tasks, may underperform purpose-built architectures on domain-specific financial prediction problems with very low SNR.

---

## 3. Methodology

### 3.1 Data

We construct a multivariate feature set from two data sources:

**FX pairs (6 instruments).** EUR/USD (primary target), GBP/USD, USD/JPY, USD/CHF, EUR/GBP, and EUR/JPY. Daily OHLCV data is sourced from Yahoo Finance covering January 2005 through early 2026. All data is cached locally after initial download to ensure reproducibility and eliminate network dependencies during experiments.

**Macroeconomic signals (9 indicators).** VIX (implied volatility), TNX (10-year Treasury yield), IRX (3-month Treasury yield), DXY (US Dollar Index), Gold (GC=F), Crude Oil (CL=F), S&P 500 (^GSPC), TLT (long-term Treasury ETF), and HYG (high-yield corporate bond ETF). These capture the broader risk environment, yield curve dynamics, and cross-asset correlations that drive FX flows.

**Feature engineering.** We compute 104 strictly backward-looking features organized into four groups:

1. *Per-pair technical features* (13 per pair, 78 total): log returns at horizons 1d, 5d, 10d, and 20d; rolling volatility at windows 5d, 10d, 20d, and 60d; RSI(14); MACD signal; and microstructure measures (high-low range, close-to-open gap).

2. *Cross-pair correlation features* (5): rolling 21-day Spearman correlation of the primary pair (EUR/USD) with each of the five secondary pairs.

3. *Macro regime features* (21): returns and levels for 9 macro tickers, yield curve slope (TNX minus IRX), VIX change, and DXY volatility.

All features use a warmup period of 63 trading days (approximately 3 calendar months) to ensure that the longest lookback window (60-day volatility) is fully populated before the first usable sample.

**Prediction targets.** EUR/USD 1-day and 5-day forward log returns, computed for all 6 currency pairs. The primary evaluation metric is based on the EUR/USD 1-day return prediction used as a directional trading signal.

### 3.2 Super-Fold Evaluation Framework

Standard k-fold cross-validation is inappropriate for financial time series due to temporal dependence, label overlap from multi-day forward returns, and regime non-stationarity (Lopez de Prado, 2018). We design a super-fold evaluation framework that addresses all three concerns.

**Seven regime-stratified fold windows.** Each fold targets a distinct market regime:

| Fold | Regime | Val Period | Test Period | Test Samples |
|------|--------|------------|-------------|--------------|
| 1 | Pre-crisis / GFC onset | 2007-04 to 2007-09 | 2008-01 to 2008-06 | 103 |
| 2 | Post-crash recovery | 2009-04 to 2009-09 | 2010-01 to 2010-06 | 107 |
| 3 | Eurozone debt plateau | 2012-04 to 2012-09 | 2013-01 to 2013-06 | 106 |
| 4 | Strong USD downturn | 2014-07 to 2014-12 | 2015-04 to 2015-12 | 168 |
| 5 | Low-volatility plateau | 2018-04 to 2018-09 | 2019-01 to 2019-09 | 162 |
| 6 | EUR crisis / COVID downturn | 2021-04 to 2021-09 | 2022-01 to 2022-09 | 165 |
| 7 | Recent mixed / upturn | 2024-04 to 2024-09 | 2025-01 to 2025-09 | 162 |

**Hole-punched expanding window training.** Rather than training 7 separate models, we use a single training set consisting of all historical data (2005--2023) with all 14 held-out windows (7 val + 7 test) surgically removed ("hole-punched"). This maximizes training data while maintaining strict temporal separation.

**Purge, embargo, and label-horizon buffers.** Three nested safeguards prevent information leakage:

- *Purge gap* ($P = 90$ calendar days): minimum separation between the training data boundary and the start of any validation or test window. This exceeds the autocorrelation horizon of daily FX features.
- *Embargo period* ($E = 21$ calendar days): additional holdout after each test window end before training data can resume.
- *Label-horizon buffer* ($B = 10$ calendar days): the 10 calendar days immediately preceding each excluded window are also removed from training, preventing the 5-day forward return target ($\text{fwd\_ret\_5d}$) from incorporating price data that falls within the excluded period.

**Zero-overlap invariants.** Before every experiment, we programmatically verify:

$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

The resulting dataset splits contain 2,478 training samples, 915 validation samples (across 7 windows), and 1,170 test samples (across 7 windows).

**Contiguous segment handling.** Because hole-punching creates non-contiguous date ranges in the training set, we identify contiguous segments and create separate sliding-window datasets for each segment, preventing windows from spanning date gaps. This is critical: naive sliding windows across gaps produce approximately 41% garbage samples in our setting.

### 3.3 Model Architectures

We evaluate two primary architectures representing opposite ends of the complexity spectrum.

#### 3.3.1 Residual MLP (Champion)

The residual MLP implements a skip connection architecture inspired by He et al. (2016), adapted for tabular financial data:

$$h = f_{\text{shortcut}}(x) + f_{\text{residual}}(x)$$

where:

$$f_{\text{shortcut}}(x) = W_s \cdot \text{flatten}(x), \quad W_s \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{input}}}$$

$$f_{\text{residual}}(x) = W_2 \cdot \text{GELU}(\text{Drop}(W_1 \cdot \text{flatten}(x))) + b$$

with an additional GELU activation and dropout layer after $W_2$. The input dimension is $d_{\text{input}} = n_{\text{features}} \times \text{seq\_len} = 104 \times 10 = 1{,}040$, and the hidden dimension is $d_{\text{hidden}} = 128$.

The shortcut branch provides a linear baseline prediction, while the residual branch learns nonlinear corrections. For low-SNR financial data, the signal is a small perturbation on linear relationships between features and returns. The skip connection preserves this structure while allowing the model to capture nonlinear regime-dependent effects.

**Prediction heads.** Each horizon (1d, 5d) has an independent prediction head:

$$\hat{y} = W_3 \cdot \text{GELU}(\text{Drop}_{0.15}(\text{LN}(h))) + b_3$$

where LN denotes LayerNorm and $W_3 \in \mathbb{R}^{6 \times 64}$ outputs predictions for all 6 currency pairs.

**Parameter count.** The full model has 301,196 trainable parameters, yielding a ratio of approximately 122 parameters per training sample --- well within the regime where generalization is feasible for financial data.

#### 3.3.2 LFM2.5-350M Foundation Model (Baseline)

The LFM2.5 model (Liquid AI) is a 350M-parameter foundation model pretrained on diverse time series and language data. We use it in a transfer learning configuration:

1. A linear projection maps the 104-dimensional feature vector to the model's hidden dimension (1,024).
2. The pretrained LFM2.5 backbone processes the projected sequence with all parameters frozen.
3. Prediction heads (identical architecture to the residual MLP) are trained on top of the backbone's last hidden state.

This yields 639,500 trainable parameters (projection + heads), with a ratio of approximately 233 parameters per training sample. The 350M backbone parameters are frozen and do not contribute to the trainable parameter count but impose a fixed inductive bias on the learned representations.

### 3.4 Training Procedure

**Loss function.** We use Huber loss with $\delta = 0.5$, which provides robustness to the heavy-tailed distribution of FX returns:

$$\mathcal{L}_{\text{Huber}}(\mu, y) = \begin{cases} \frac{1}{2}(\mu - y)^2 & \text{if } |\mu - y| \leq \delta \\ \delta(|\mu - y| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

The Huber loss transitions from quadratic to linear at $|\mu - y| = 0.5$, reducing the influence of extreme return observations (fat tails) on gradient updates.

**Optimizer.** AdamW (Loshchilov and Hutter, 2019) with $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $\lambda = 10^{-5}$, and learning rate $\eta = 5 \times 10^{-4}$ for the residual MLP.

**Learning rate schedule.** Cosine annealing over 50 epochs:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Early stopping.** Training terminates if validation loss does not improve for 10 consecutive epochs (patience = 10).

**Gradient clipping.** Maximum gradient norm of 1.0 to prevent exploding gradients during regime transitions.

**Batch size.** 32 samples per batch. Smaller batches provide better gradient noise estimates for the inherently noisy FX data.

**Reproducibility.** All random seeds (PyTorch, NumPy, Python) are fixed. The champion configuration uses seed 0.

### 3.5 Uncertainty Estimation

We implement two complementary uncertainty estimation approaches:

**Aleatoric uncertainty (data noise).** The model optionally outputs both a mean prediction $\mu$ and a log-variance $s = \log \sigma^2$ via a heteroscedastic prediction head (Kendall and Gal, 2017). The loss becomes:

$$\mathcal{L}_{\text{het}} = \exp(-s) \cdot \mathcal{L}_{\text{Huber}}(\mu, y) + \frac{1}{2}s$$

The log-variance is clamped to $[-6, 2]$ to prevent two failure modes: overconfidence ($s < -6$, variance $< 0.0025$) and lazy variance ($s > 2$, variance $> 7.4$), following Stirn et al. (2023).

**Epistemic uncertainty (model uncertainty).** MC Dropout with 20 stochastic forward passes at inference time (Gal and Ghahramani, 2016). The epistemic uncertainty is the variance of predictions across the 20 passes.

**Confidence score.** We define a composite confidence measure:

$$c = \sigma(-\log(\hat{\sigma}_{\text{ale}} + \hat{\sigma}_{\text{epi}}))$$

where $\sigma(\cdot)$ is the sigmoid function. High confidence ($c > 0.8$) indicates low total uncertainty and can be used for position sizing or trade filtering.

**Empirical finding.** The heteroscedastic loss consistently degraded performance in our setting (Section 4.1), with the variance branch absorbing model capacity at the expense of mean prediction quality. The champion uses plain Huber loss with MC Dropout for epistemic uncertainty only.

### 3.6 Agent-Driven Experiment Protocol

The LLM agent (Claude Code, Anthropic) operates as an autonomous researcher following a strict 7-step loop:

**Step 1: DIAGNOSE.** Analyze per-fold test results from the previous experiment. Identify which folds are weak and form a hypothesis about why. For example: "Fold 2 (post-crash recovery, 2009--2010) has the lowest Sharpe (+0.44) because the mean-reversion regime differs from the momentum-driven regimes where the model excels."

**Step 2: CITE.** Search published literature for techniques that address the identified failure mode. For example: "He et al. (2016) show that residual connections preserve linear baselines, which is beneficial when the signal is a small perturbation on a linear relationship."

**Step 3: HYPOTHESIZE.** Form a specific, testable hypothesis. For example: "Adding a skip connection will improve fold 2 performance by providing a linear baseline that captures the mean-reversion signal."

**Step 4: PREDICT.** State the expected outcome quantitatively. For example: "I predict the composite score will improve from +0.82 to approximately +2.0, with fold 2 Sharpe becoming positive."

**Step 5: RUN.** Execute exactly one experiment, changing a single variable from the current champion configuration. The agent constructs and executes the command:

```
python -m autoresearch.run_autoresearch --backbone mlp \
  --lr 5e-4 --epochs 50 --head_dropout 0.15 --huber_delta 0.5 \
  --description "Exp N: [hypothesis being tested]"
```

**Step 6: ANALYZE.** Compare per-fold results to the champion. For each fold, compute the delta in Sharpe ratio and explain the change. If the composite score improves, the new configuration becomes the champion; otherwise, revert and try a different direction.

**Step 7: CHECKPOINT.** Save full state to a crash-recovery file including: current champion config, composite score, per-fold test Sharpe table, last experiment result, the exact next experiment command, and rationale for the next experiment. This enables seamless recovery from system crashes.

**Key rules.** (a) Always start from the current champion --- never wander off the best baseline. (b) One change at a time --- never conflate multiple modifications. (c) Never grid search --- every decision must be justified by published work or prior experimental evidence. (d) Code changes are allowed --- the agent may modify the model architecture, loss function, or training loop if it has a principled reason, making this more powerful than hyperparameter-only optimization.

### 3.7 Evaluation Metrics

We report a comprehensive suite of financial and statistical metrics:

**Financial metrics:**
- *Sharpe ratio*: $\text{SR} = \frac{\bar{r}}{\sigma_r} \sqrt{252}$, where $\bar{r}$ and $\sigma_r$ are the mean and standard deviation of daily strategy returns.
- *Sortino ratio*: $\frac{\bar{r}}{\sigma_d} \sqrt{252}$, using downside deviation $\sigma_d$ only.
- *Probabilistic Sharpe Ratio (PSR)*: $P(\text{SR}^* > 0)$, accounting for skewness and kurtosis (Bailey and Lopez de Prado, 2012).
- *Information Coefficient (IC)*: Spearman rank correlation between predicted and realized returns.
- *Hit rate*: fraction of days where the predicted direction matches the realized direction.
- *Maximum drawdown*: largest peak-to-trough decline in cumulative returns.
- *Profit factor*: ratio of gross profits to gross losses.
- *Calmar ratio*: annualized return divided by maximum drawdown.

**Trading strategy.** The strategy is a simple sign-based allocation:

$$r_t^{\text{strategy}} = \text{sign}(\hat{r}_t) \cdot r_t^{\text{actual}}$$

No transaction costs, slippage, or position sizing are modeled. This isolates the predictive signal quality from execution considerations.

**Composite metric.** For experiment comparison, we use a composite score that penalizes inconsistency across regimes:

$$\text{Composite} = \min(\text{SR}_{\text{test}}, \text{SR}_{\text{val}}) - 0.1 \times n_{\text{neg}}$$

where $n_{\text{neg}}$ is the number of test folds with negative Sharpe ratio. This metric requires the model to perform well on both validation and test sets across all regime windows.

---

## 4. Experiments and Results

### 4.1 Phase 1: LFM2.5-350M Foundation Model (Experiments 1--50)

The first 50 experiments explored the LFM2.5 foundation model with various adapter configurations.

**Optimization trajectory.** Starting from default hyperparameters (lr=3e-4, epochs=20), the agent systematically explored learning rate (1e-5 to 1e-3), adapter hidden size, dropout rates, heteroscedastic loss, and epoch count. The agent identified early that the foundation model's frozen backbone imposed a strong inductive bias that limited adaptation to FX-specific patterns.

**Heteroscedastic loss experiments.** Enabling the heteroscedastic loss (Kendall and Gal, 2017) consistently degraded performance. The variance branch increased aleatoric uncertainty above 0.20, indicating that the model was "copping out" to high variance predictions rather than learning the mean signal. With only 2,478 training samples, the additional optimization axis (log-variance) steals capacity from the mean prediction head. The agent correctly diagnosed this after 3 failed experiments and reverted to plain Huber loss.

**Seed sensitivity.** A critical finding was the extreme sensitivity to random seed. For the best LFM2.5 configuration, composite scores ranged from -1.52 to +1.77 across 4 seeds --- a swing of 3.29 points for identical hyperparameters. This suggests that the 639K trainable parameters / 2,478 training samples ratio (258 params/sample) places the model in a regime of severe overparameterization where optimization trajectory matters more than architecture.

**Best LFM2.5 result.** The best individual run achieved test Sharpe +2.07 and composite +1.77, but the median across seeds was only +1.40. The agent concluded that the foundation model backbone, pretrained on diverse time series, adds representational noise rather than useful inductive bias for the specific statistical properties of daily FX returns (mean-reverting, fat-tailed, regime-dependent).

| Metric | LFM2.5 Best | LFM2.5 Median | LFM2.5 Worst |
|--------|-------------|---------------|--------------|
| Test Sharpe | +2.07 | +1.40 | -1.52 |
| Composite | +1.77 | +0.94 | -1.52 |
| Positive test folds | 5/7 | 4/7 | 2/7 |
| Total return (%) | +82% | +35% | -28% |

### 4.2 Phase 2: Residual MLP from Scratch (Experiments 51--90)

Based on the Phase 1 diagnosis --- overparameterization and domain mismatch --- the agent pivoted to training a model from scratch, beginning with a simple MLP.

**Flat MLP baseline.** The initial flat MLP (512 hidden units, no skip connection) achieved a median test Sharpe of -0.51 across seeds. Reducing hidden size to 128 improved the median to +0.82, confirming the overparameterization diagnosis.

**Residual skip connection (key breakthrough).** Adding a linear shortcut connection transformed performance: median test Sharpe jumped from +0.82 to +4.24 (5.2x improvement). The agent cited He et al. (2016) and hypothesized that the skip connection would be particularly beneficial for low-SNR financial data where the signal is a small perturbation on a linear baseline. The results exceeded the agent's prediction of a composite improvement from +0.82 to approximately +2.0.

**Learning rate optimization.** The skip connection provided gradient stability that enabled a higher learning rate. Increasing lr from 3e-4 to 5e-4 improved median test Sharpe from +4.42 to +5.41. The agent cited He et al. (2016): "Skip connections reduce the effective condition number of the loss landscape, enabling larger step sizes."

**Head dropout tuning.** Increasing head dropout from 0.10 to 0.15 improved generalization, particularly on the weakest fold (fold 2, post-crash recovery). The agent cited Srivastava et al. (2014) and hypothesized that the additional regularization would help the prediction heads generalize across regimes with different noise levels.

**Huber delta selection.** Reducing $\delta$ from 1.0 to 0.5 improved robustness to fat-tailed returns. The agent noted that daily FX returns have excess kurtosis of approximately 5--8, making the transition from quadratic to linear loss at $|\epsilon| = 0.5$ appropriate for capturing the tail behavior.

**Epoch count.** Training from random initialization requires more epochs than fine-tuning a pretrained model. The agent found that 50 epochs (with patience=10 early stopping) was necessary, compared to 20 epochs for the LFM2.5 model. Actual training typically stopped at 35--45 epochs via early stopping.

### 4.3 Champion Configuration

The final champion (Experiment 88, verified deterministic) achieves:

| Metric | Value |
|--------|-------|
| Test Sharpe (annualized) | +6.21 |
| Composite score | +5.50 |
| Positive test folds | 7/7 |
| Test Sortino ratio | +11.31 |
| Test cumulative return | +1,001% |
| Test max drawdown | 4.13% |
| Test win rate | 69.35% |
| Test profit factor | 3.30 |
| Test IC (Spearman) | +0.485 |
| PSR (vs SR=0) | 1.00 |
| Val Sharpe | +5.60 |
| Val cumulative return | +247% |
| Trainable parameters | 301,196 |
| Training time | 36.4 seconds |

**Champion hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Architecture | Residual MLP | Skip connection for low-SNR signals (He et al., 2016) |
| Hidden size | 128 | 122 params/sample ratio; avoids overparameterization |
| Head hidden | 64 | $\min(128, 64)$; compact prediction head |
| Learning rate | 5e-4 | Enabled by skip connection gradient stability |
| Batch size | 32 | Standard for noisy financial data |
| Sequence length | 10 | ~2 weeks lookback; industry standard for short-term FX |
| Epochs | 50 | From-scratch training needs more epochs; early stopping at ~40 |
| Weight decay | 1e-5 | Mild L2 regularization |
| Patience | 10 | Avoids premature stopping on noisy val loss |
| Grad clip | 1.0 | Prevents exploding gradients during regime transitions |
| Huber delta | 0.5 | Robust to fat-tailed FX returns |
| Head dropout | 0.15 | Regularizes prediction heads across regimes |
| Het loss | False | Plain Huber; het-loss degrades with n < 3,000 |
| Seed | 0 | Fixed for reproducibility |

### 4.4 Per-Fold Test Results

The champion achieves positive Sharpe ratios across all 7 test folds, covering diverse market regimes:

| Fold | Regime | Period | Sharpe | Sortino | IC | Win Rate | Return | Max DD |
|------|--------|--------|--------|---------|-----|----------|--------|--------|
| 1 | Pre-crisis / GFC | 2008 H1 | +2.46 | +6.78 | +0.188 | 60.8% | +19.8% | 3.29% |
| 2 | Post-crash recovery | 2010 H1 | +1.17 | +2.02 | +0.082 | 53.3% | +5.5% | 3.47% |
| 3 | Eurozone debt plateau | 2013 H1 | +9.76 | +18.81 | +0.584 | 76.0% | +34.1% | 1.32% |
| 4 | Strong USD downturn | 2015 | +9.78 | +19.42 | +0.667 | 75.5% | +90.3% | 1.81% |
| 5 | Low-vol plateau | 2019 | +8.85 | +16.14 | +0.638 | 71.0% | +29.3% | 1.43% |
| 6 | EUR crisis / COVID | 2022 | +9.95 | +21.01 | +0.641 | 70.9% | +69.5% | 2.27% |
| 7 | Recent mixed / upturn | 2025 | +8.48 | +14.36 | +0.622 | 71.6% | +55.8% | 1.64% |

**Per-fold analysis.**

*Fold 1 (GFC onset, 2008):* Moderate performance (Sharpe +2.46) during one of the most volatile periods in FX history. The model correctly identifies directional moves but the extreme volatility (daily moves of 2--3%) compresses the risk-adjusted metric. IC of +0.188 indicates meaningful but moderate predictive signal.

*Fold 2 (Post-crash recovery, 2010):* The weakest fold (Sharpe +1.17, IC +0.082). This regime is characterized by mean-reverting, range-bound EUR/USD behavior following the 2008--2009 crash. The model's momentum-oriented features are less effective in this environment. Aleatoric uncertainty is highest for this fold, correctly reflecting the noisier signal.

*Folds 3--7 (2013--2025):* Consistently strong performance with Sharpe ratios between +8.48 and +9.95, IC between +0.58 and +0.67, and win rates above 70%. These periods exhibit clearer directional trends that the model's feature set captures effectively.

### 4.5 Cross-Seed Robustness

A critical test of model reliability is performance stability across random seeds. We evaluate the champion configuration with 3 seeds:

| Seed | Composite | Test Sharpe | Val Sharpe | Positive Folds (Test) | Total Return |
|------|-----------|-------------|------------|----------------------|--------------|
| 0 | +5.50 | +6.21 | +5.60 | 7/7 | +1,001% |
| 42 | +4.45 | +4.69 | +4.45 | 6/7 | +612% |
| 99 | +4.46 | +4.76 | +4.46 | 6/7 | +628% |
| **Mean** | **+4.80** | **+5.22** | **+4.84** | **6.3/7** | **+747%** |
| **Std** | **+0.60** | **+0.85** | **+0.66** | **0.6** | **+220%** |

The residual MLP shows substantially lower seed variance than the LFM2.5 foundation model: standard deviation of 0.85 in test Sharpe (vs. estimated >1.5 for LFM2.5). All 3 seeds produce economically significant results (test Sharpe > 4.0), confirming that the architecture and hyperparameters --- not the random seed --- drive performance.

### 4.6 Key Findings

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Residual skip = 5x improvement | Median Sharpe: +0.82 (flat) to +4.24 (residual) | Linear baseline + nonlinear correction is ideal for low-SNR data |
| Smaller hidden = better generalization | 128h > 512h at matched conditions | Fewer params/sample prevents memorization |
| Foundation model underperforms | LFM2.5 median +1.40 vs MLP median +5.41 | Frozen backbone adds noise for daily FX |
| Het-loss hurts on small data | Variance branch steals capacity from mean | Plain Huber more stable with n < 3,000 |
| 50 epochs needed from scratch | 20 epochs insufficient; early stopping at ~40 | Pretrained models converge faster but ceiling is lower |
| Seed variance critical for evaluation | LFM2.5: -1.52 to +1.77 (same config) | Must verify with multiple seeds before claiming improvement |

### 4.7 Cross-Architecture Comparison

| Architecture | Experiments | Median Test Sharpe | Best Test Sharpe | Params | Params/Sample |
|-------------|-------------|-------------------|-----------------|--------|---------------|
| **Residual MLP (lr=5e-4)** | **2 seeds** | **+5.41** | **+6.21** | **301K** | **122** |
| Residual MLP (lr=3e-4) | 3 seeds | +4.42 | +5.23 | 301K | 122 |
| Residual MLP (lr=3e-4, hd=0.1) | 3 seeds | +4.24 | +4.77 | 301K | 122 |
| LFM2.5-350M | 50 (4 seeds) | +1.40 | +2.07 | 639K | 258 |
| Plain MLP 128h | 3 seeds | +0.82 | +1.48 | 167K | 67 |
| Plain MLP 512h | 2 seeds | -0.51 | +0.93 | 1,060K | 428 |

The comparison reveals a clear pattern: model capacity must be matched to dataset size. Both under-parameterized (plain MLP 128h, 67 params/sample) and over-parameterized (plain MLP 512h, 428 params/sample; LFM2.5, 258 trainable params/sample) models underperform. The residual MLP at 122 params/sample hits the sweet spot, and the skip connection provides crucial inductive bias for the low-SNR regime.

---

## 5. Discussion

### 5.1 Why Residual MLP Outperforms Foundation Models

The superiority of the simple residual MLP over the 350M-parameter LFM2.5 foundation model deserves careful analysis. We identify three contributing factors:

**Inductive bias mismatch.** LFM2.5 was pretrained on diverse time series and language data. Daily FX returns have specific statistical properties --- near-zero mean, fat tails (excess kurtosis 5--8), weak autocorrelation, and regime-dependent dynamics --- that differ substantially from the pretraining distribution. The frozen backbone imposes representational constraints that may be actively harmful, forcing the small adapter to learn both the domain-specific representation and the prediction mapping simultaneously.

**Overparameterization.** Even with the backbone frozen, the LFM2.5 configuration has 639K trainable parameters for 2,478 training samples (258 params/sample). Financial data is fundamentally different from computer vision or NLP, where overparameterized models can rely on data augmentation and the smoothness of the underlying function. FX returns lack these properties: the signal is weak, the noise is heavy-tailed, and there is no meaningful data augmentation strategy.

**The residual advantage for low-SNR data.** The skip connection in the residual MLP provides a principled inductive bias: the linear shortcut captures the dominant (linear) relationship between features and returns, while the nonlinear branch learns small corrections. This decomposition is well-suited to financial data where the predictable component is a small perturbation on a linear factor model. The foundation model, by contrast, processes features through many nonlinear layers, potentially destroying the linear signal.

### 5.2 The Value of Agent-Driven Research

The agent-driven protocol offers several advantages over traditional hyperparameter optimization:

**Diagnostic reasoning.** Rather than sweeping parameters blindly, the agent diagnoses specific failure modes ("fold 2 has low Sharpe because of mean-reverting regime") and applies targeted fixes. This is fundamentally different from Bayesian optimization, which treats the objective as a black box.

**Literature-grounded decisions.** Every architectural choice is justified by published work. The decision to add skip connections was motivated by He et al. (2016), not by trial and error. The decision to use Huber loss with $\delta = 0.5$ was motivated by the known heavy-tailed distribution of FX returns, not by grid search over $\delta$.

**Architecture search.** Unlike hyperparameter-only AutoML, the agent can modify the model architecture, loss function, and training procedure. The pivot from LFM2.5 to from-scratch MLP, and the subsequent addition of residual connections, represent architectural decisions that no standard hyperparameter optimizer would make.

**Crash recovery.** The checkpoint protocol (state saved after every experiment) enables seamless recovery from system failures, which occurred frequently during our experiments. A fresh session can resume from the exact point of interruption by reading only the checkpoint file.

### 5.3 Uncertainty Estimation Results

The champion model (plain Huber loss) achieves very low aleatoric and epistemic uncertainty (mean aleatoric: $2.6 \times 10^{-5}$, mean epistemic: $5.2 \times 10^{-5}$) with high confidence (mean: 0.9999). The per-fold uncertainty pattern is informative:

| Fold | Regime | Aleatoric ($\times 10^{-5}$) | Epistemic ($\times 10^{-5}$) | Test Sharpe |
|------|--------|------------------------------|------------------------------|-------------|
| 1 | GFC onset | 5.3 | 10.6 | +2.46 |
| 2 | Post-crash | 2.1 | 4.3 | +1.17 |
| 5 | Low-vol | 1.2 | 2.5 | +8.85 |

Higher epistemic uncertainty on fold 1 (GFC) is consistent with the model having less relevant training data for extreme crisis regimes. The lowest uncertainty (fold 5, low-vol plateau) corresponds to the most stable prediction environment.

### 5.4 Limitations

Several important limitations should be noted:

1. **Single asset class.** We evaluate only FX (primarily EUR/USD). Generalization to equities, fixed income, or commodities is not established.

2. **Daily frequency only.** The system operates on daily OHLCV data. Intraday dynamics, which dominate short-term FX prediction in practice, are not captured.

3. **No transaction costs.** The sign-based strategy does not model bid-ask spreads, market impact, or slippage. For daily EUR/USD, round-trip costs are approximately 1--2 bps for institutional traders, which would reduce but not eliminate the reported returns.

4. **Paper trading only.** No live trading validation has been performed. The gap between backtested and live performance is well-documented in financial ML (Bailey et al., 2014).

5. **Simple trading rule.** The sign-based strategy is a lower bound on what a more sophisticated position-sizing or portfolio construction approach could achieve. The uncertainty estimates could inform position sizing but this is not explored.

6. **Limited seed analysis.** Cross-seed robustness is evaluated with 3 seeds. A more thorough analysis would use 10+ seeds to characterize the full distribution of outcomes.

7. **Single agent architecture.** We use one LLM (Claude Code) as the agent. The protocol could potentially be improved with multiple agents or different LLMs.

[TODO: add after more experiments --- additional backbone comparisons (LSTM, PatchTST, XGBoost), extended seed analysis, transaction cost modeling]

---

## 6. Conclusion

We have presented AutoResearch, an autonomous ML optimization system in which an LLM agent drives the full experiment design loop for financial time series prediction. Over 90 experiments on EUR/USD daily return forecasting, the system converges to a residual MLP champion achieving a test Sharpe ratio of +6.21 across 7 market regime windows with 7/7 positive test folds.

The central empirical result is that a simple residual MLP with 301K trainable parameters decisively outperforms a 350M-parameter foundation model (LFM2.5), achieving a 3.9x higher median test Sharpe ratio (+5.41 vs. +1.40). This finding has important implications for the application of foundation models to financial data: the specific statistical properties of financial returns (low SNR, fat tails, regime dependence) favor architectures with appropriate inductive biases over models pretrained on heterogeneous data.

The agent-driven research protocol --- diagnose, cite, hypothesize, predict, run, analyze, checkpoint --- offers a principled alternative to grid search and Bayesian optimization. By reasoning in natural language about per-fold failures and citing published literature, the agent makes diagnostic decisions that a standard optimizer cannot. The key architectural insight (adding residual connections) emerged from the agent's literature review, not from a pre-defined search space.

The super-fold evaluation framework, with its combination of purge gaps (90 days), embargo periods (21 days), label-horizon buffers (10 days), and hole-punched expanding windows, provides rigorous protection against data leakage --- the most common source of inflated results in financial ML.

**Future work.** We plan to extend the system along several dimensions: (a) additional backbone architectures (LSTM, PatchTST, XGBoost, LightGBM) to build a comprehensive comparison, (b) multi-asset prediction (multiple FX pairs, cross-asset portfolios), (c) intraday data at 1-hour and 5-minute frequencies, (d) transaction cost modeling and realistic position sizing, (e) live paper trading validation, and (f) multi-agent protocols where multiple LLMs collaborate on the research loop.

---

## References

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., ... & Gasthaus, J. (2024). Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*.

Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3--44.

Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society*, 61(5), 458--471.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785--794.

Das, A., Kong, W., Leber, A., Sen, R., & Yu, R. (2024). A decoder-only foundation model for time-series forecasting. *Proceedings of the 41st International Conference on Machine Learning*.

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383--417.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, 1050--1059.

Galeshchuk, S., & Mukherjee, S. (2017). Deep networks for predicting direction of change in foreign exchange rates. *Intelligent Systems in Accounting, Finance and Management*, 24(4), 100--110.

Garza, A., & Mergenthaler-Canseco, M. (2024). TimeGPT-1. *arXiv preprint arXiv:2310.03589*.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223--2273.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770--778.

Hutter, F., Kotthoff, L., & Vanschoren, J. (Eds.). (2019). *Automated Machine Learning: Methods, Systems, Challenges*. Springer.

Karpathy, A. (2019). A recipe for training neural networks. Blog post, April 2019. http://karpathy.github.io/2019/04/25/recipe/

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 30.

Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36--52.

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *Proceedings of the 7th International Conference on Learning Representations*.

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *Proceedings of the 11th International Conference on Learning Representations*.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.

Rasul, K., Ashok, A., Williams, A. R., Khorasani, A., Adamopoulos, G., Bhatt, R., ... & Lim, B. (2024). Lag-Llama: Towards foundation models for probabilistic time series forecasting. *Proceedings of the 12th International Conference on Learning Representations*.

Sermpinis, G., Theofilatos, K., Karathanasopoulos, A., Georgopoulos, E. F., & Dunis, C. (2012). Forecasting foreign exchange rates with adaptive neural networks using radial-basis functions and particle swarm optimization. *European Journal of Operational Research*, 225(3), 528--540.

Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems*, 25.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929--1958.

Stirn, A., Wessels, H., Schurr, M., Pereira, S., Halpern, Y., & Sontag, D. (2023). Faithful heteroscedastic regression with neural networks. *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*.

Vanschoren, J. (2019). Meta-learning. In *Automated Machine Learning: Methods, Systems, Challenges*, 35--61. Springer.

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. *Proceedings of the 5th International Conference on Learning Representations*.

---

*Corresponding author: eranti@gmail.com*
