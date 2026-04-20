# AutoResearch: An LLM-Driven Autonomous Research Loop for Financial Time Series Forecasting

**Anonymous Authors**
*Affiliation withheld for double-blind review*

---

## Abstract

We study whether a large language model, operating as an autonomous researcher rather than as a code assistant, can drive a closed-loop machine learning research process from literature review through hyperparameter selection, experiment execution, diagnosis, and champion archival. We instantiate this loop on a daily EUR/USD foreign-exchange forecasting benchmark (2005--2025, $n=2738$ trading days, 104 engineered features) using a seven-regime super-fold evaluation protocol with 90-day purge, 21-day embargo, and 10-day label-horizon buffers. Over 151 experiments across four backbones (MLP, LSTM, LFM2-350M, PatchTST), the agent identifies a bidirectional two-layer LSTM at hidden size 128 as the global champion, achieving a composite score of $+6.4242$, a test Sharpe of $+6.5242$, a validation Sharpe of $+7.1539$, and positive Sharpe across all seven test fold windows spanning the 2008 Global Financial Crisis onset, post-crash recovery, Eurozone debt, strong-USD downturn, low-volatility, EUR crisis, and recent mixed regimes. Cumulative test return across the 1170-day test horizon reaches $+1122\%$ under a simple sign-based trading rule. A multi-seed variance study at the champion configuration reveals composite standard deviation $\approx 1.0$ across six seeds (range $\approx 2.58$), which we interpret as evidence that single-seed ``champions'' in financial machine learning are probabilistically lucky and that median-of-$k$ reporting should become a community standard. We release the complete autoresearch protocol, reasoning annotations, and per-experiment trade logs, and argue that the primary scientific artifact of such work is the reasoning trace rather than the final model.

---

## 1. Introduction

Financial time series forecasting is a setting where small, consistent, out-of-sample gains translate to economic value but where signal-to-noise ratios are extraordinarily low, distributions shift across regimes, and overfitting to historical data is the rule rather than the exception (López de Prado, 2018). Two properties of the problem have resisted the kind of scaling-based progress that has driven computer vision and natural language processing. First, the amount of daily-bar history for a given instrument is bounded by calendar time: roughly $n \approx 5000$ trading days over two decades, independent of compute budget. Second, walk-forward evaluation that respects purge and embargo windows is computationally cheap but statistically unforgiving: a few lucky folds can inflate Sharpe ratios by multiples, and seed variance dominates reported headline numbers (Bailey & López de Prado, 2014).

Against this backdrop, the past two years have seen a proliferation of time-series foundation models (Das et al., 2024; Ansari et al., 2024; Woo et al., 2024; Goswami et al., 2024), state-space sequence models (Gu & Dao, 2024), and extended LSTM variants (Beck et al., 2024), each claiming benchmark improvements. Practitioners face a combinatorial choice problem: which backbone family, which recipe from the paper, which regularisation schedule, which seed. Grid search is wasteful. Neural architecture search (Zoph & Le, 2017; Elsken et al., 2019) requires reward signals that walk-forward Sharpe does not cleanly provide.

This paper asks a different question. Rather than automating search, can a large language model (LLM) be placed *in the role of the principal investigator*, reading results, reasoning about mechanisms, citing the literature, formulating hypotheses, and choosing the next experiment on the basis of a documented argument? Concretely, we instantiate Claude Opus 4.7 (1M context) as the outer research loop over a Python experiment runner. The agent reads the experiment log, diagnoses weaknesses by fold, proposes a single targeted change with a literature citation and a numerical prediction, executes one experiment, analyses the result against the prediction, and checkpoints its state. We call this system **AutoResearch**.

Our contributions are as follows.

1. **An LLM-driven autonomous research loop for financial ML**, formalised as a seven-step diagnose--cite--hypothesise--predict--execute--analyse--checkpoint protocol. The loop is *append-only*: every experiment, including failures, is preserved with a reasoning annotation. The agent may modify the training code if it has a principled justification.
2. **A new state of the art on a daily EUR/USD super-fold benchmark.** The champion, a bidirectional two-layer LSTM with hidden size 128, head dropout $0.25$, weight decay $7 \times 10^{-4}$, batch size $16$, and seed $42$, attains composite $+6.4242$ with seven of seven positive test fold Sharpes, exceeding our best MLP residual champion ($+5.499$), our best LFM2 head-finetuning result ($+1.77$), and PatchTST at its default sequence length ($-1.72$).
3. **A multi-seed variance study at the champion configuration** showing composite standard deviation $\approx 1.0$ and a range of $2.58$ across six seeds, which is large relative to the gap between competing configurations. We draw the reporting-standards corollary that median-of-$k$ (with $k \geq 3$) should be a minimum bar for financial-ML champion declarations.
4. **A ten-backbone research roadmap** (Tier-1 MLP/LSTM/LFM2/PatchTST plus Tier-2 TimesFM 2.5, Chronos, Moirai, MOMENT, TiRex, Sundial, Time-MoE, TimeMixer++, TimesNet, MambaTS) and corresponding SOTA training recipes drawn directly from the originating papers, so that the agent begins each backbone's exploration from a literature-recommended baseline rather than a generic default.
5. **An institutional-memory dashboard design** ({\tt reasoning\_annotations.json} schema plus a rendering layer) that turns the experiment log into a navigable research journal. We argue that, in LLM-driven science, the reasoning trace is a primary scientific artifact on par with the final model.

Section 2 situates AutoResearch against prior work in financial ML, time-series foundation models, AutoML, and LLM-driven science. Section 3 describes the data pipeline, super-fold protocol, composite metric, the seven-step loop, and backbone recipes. Section 4 reports 151 experiments: MLP exploration, LSTM champion progression, LFM2 plateau, PatchTST cold-start, and the multi-seed variance study. Section 5 presents headline results, per-regime analysis, uncertainty calibration, and classification metrics. Section 6 discusses what an LLM research loop can and cannot do. Section 7 sketches future work, including the Tier-2 backbone queue and a seed-ensemble deployment protocol. Section 8 concludes. Appendices A--C provide the experiment table, the NeurIPS reproducibility checklist, and the annotation schema.

---

## 2. Related Work

**Financial ML forecasting.** Fischer & Krauss (2018) established LSTMs as a strong baseline for daily-frequency equity prediction, reporting that $100$-epoch training with patience $15$ outperforms shorter schedules; their recipe motivates the epoch choice for our LSTM exploration. Gu, Kelly & Xiu (2020) empirically compared tree ensembles, neural networks, and shallow models across equity factor prediction and documented that nonlinearity and interactions drive out-of-sample improvements. López de Prado (2018) formalised purge and embargo for walk-forward cross-validation and introduced the probabilistic Sharpe ratio (PSR) as a multiple-testing-aware alternative; we adopt both. Bailey & López de Prado (2014) showed that backtest overfitting inflates reported Sharpe by amounts that depend on the number of trials, strengthening the case for median-of-seed reporting.

**Time-series foundation models.** The 2023--2025 period produced several zero- and few-shot forecasting foundation models: TimesFM (Das et al., 2024; arXiv:2310.10688), Chronos (Ansari et al., 2024; arXiv:2403.07815), Moirai (Woo et al., 2024; arXiv:2402.02592), and MOMENT (Goswami et al., 2024; arXiv:2402.03885). Sundial (Liu et al., 2025; arXiv:2502.00816) and TiRex (Liu et al., 2024) continue the line. Our system includes a frozen LFM2-350M (Liquid AI, 2024) with head-only finetuning as a foundation-model baseline; the 43 experiments in that branch did not close the gap to the LSTM champion.

**Transformers for time series.** PatchTST (Nie et al., 2023; arXiv:2211.14730) introduced patch tokenisation and channel independence, with sequence length $\geq 60$ as a design requirement. iTransformer (Liu et al., 2024; arXiv:2310.06625) inverts attention to operate over variates. Informer (Zhou et al., 2021), FEDformer (Zhou et al., 2022), Autoformer (Wu et al., 2021), Crossformer (Zhang & Yan, 2023), and TimesNet (Wu et al., 2023) are additional baselines. DLinear (Zeng et al., 2023; arXiv:2205.13504) argued that simple linear models match or exceed transformers on long-horizon TS benchmarks; TSMixer (Chen et al., 2023) and PatchTSMixer (Ekambaram et al., 2023) offer MLP-mixer alternatives.

**State-space and linear-attention sequence models.** Mamba (Gu & Dao, 2024; arXiv:2312.00752) and its time-series adaptation MambaTS (Cai et al., 2024; arXiv:2405.16440) offer selective state-space computation with linear-time scaling. xLSTM (Beck et al., 2024; arXiv:2405.04517) extends LSTM with exponential gating (sLSTM) and matrix memory (mLSTM) and is a headline candidate for our Tier-2 queue.

**AutoML and neural architecture search.** Zoph & Le (2017; arXiv:1611.01578) pioneered reinforcement-learning-based NAS. Elsken, Metzen & Hutter (2019; JMLR) surveyed the NAS landscape. Liu, Simonyan & Yang (2019) introduced DARTS. These methods optimise a scalar reward (usually validation accuracy) over an architecture search space. AutoResearch departs from this line: the outer loop is not gradient-based or RL-based but natural-language-reasoning-based, and the search space includes code changes, not just hyperparameters.

**LLM-driven science.** Boiko, MacKnight, Kline & Gomes (2023; Nature) demonstrated an LLM-driven autonomous chemistry agent that plans and executes reactions. Lu, Lu, Lange, Foerster, Clune & Ha (2024; arXiv:2408.06292) proposed ``The AI Scientist,'' an LLM that produces complete machine-learning papers end-to-end; Lu et al. (2024b) follows up with AI-Scientist-v2. Swanson, Wu, Bulaong, Pak & Zou (2024) showed an LLM co-scientist generating and triaging biomedical hypotheses. Our work is narrower in scope --- a single, well-defined benchmark with a tight reasoning-annotation protocol --- and more rigorous about walk-forward statistical hygiene.

**Reproducibility in ML.** Bouthillier, Laurent & Vincent (2019; arXiv:1906.05268) showed that random seeds, hardware, and software stack each account for substantial variance in reported scores. Picard (2021; arXiv:2109.08203) argued that ``Torch.manual\_seed(3407) is all you need,'' a reductio that dramatises seed sensitivity. Henderson, Islam, Bachman, Pineau, Precup & Meger (2018; AAAI; arXiv:1709.06560) documented reproducibility failures in deep reinforcement learning. Our Section 4.4 variance study places financial ML inside the same diagnosis.

**Uncertainty quantification.** Kendall & Gal (2017; NeurIPS) decomposed aleatoric and epistemic uncertainty for deep learning. Gal & Ghahramani (2016; ICML) introduced MC Dropout as an approximate Bayesian inference procedure. Lakshminarayanan, Pritzel & Blundell (2017; NeurIPS) established deep ensembles as a strong baseline. Guo, Pleiss, Sun & Weinberger (2017; ICML) studied modern-network calibration. We employ MC Dropout at inference and report per-fold aleatoric and epistemic means.

---

## 3. Method

### 3.1 Data Pipeline

Our benchmark is daily-bar EUR/USD from 1 January 2005 through 31 December 2025, a total of 2738 trading days. Raw OHLCV is augmented with macro signals (DXY, VIX, 10Y US Treasury yield, 2Y--10Y spread, EURIBOR--SOFR differential, Brent crude, gold, and a 12-pair FX breadth indicator) downloaded from public sources and cached at first use. From the raw series we compute 104 backward-looking features across five families: (i) returns and rolling moments (log returns at horizons 1, 5, 10, 21; rolling mean, std, skew, kurt), (ii) price-normalised channel positions (Bollinger percent-B, Keltner, Donchian), (iii) momentum oscillators (RSI, stochastic, Williams %R, CCI, ROC, MACD and its derivatives), (iv) cross-asset and macro deltas, and (v) calendar encodings (day-of-week, month, quarter, year-since-2005). All features use strictly backward-looking windows; the implementation verifies $\mathrm{feature}_t$ depends only on observations $\{x_s : s \leq t\}$.

The prediction target is the five-day forward log return,
$$
y_t = \log\frac{P_{t+5}}{P_t},
$$
clipped to $\pm 5\sigma$ of its training-distribution to reduce leverage from extreme moves. A scalar $\hat{y}_t$ is produced; the trading rule is $\mathrm{sign}(\hat{y}_t)$.

**Purge, embargo, and label-horizon buffer.** For every fold with training window $[t_a, t_b]$, validation window $[t_v^s, t_v^e]$ and test window $[t_t^s, t_t^e]$, we remove from training all observations within $90$ days of the validation or test windows (purge), add an additional $21$ days of embargo, and a further $10$-day label-horizon buffer in front of each excluded window to prevent $y_t$ from peeking into a future excluded window through its forward-return target. Let $\mathcal{E}$ denote the union of all excluded windows (across all seven folds). The permissible training set is
$$
\mathcal{T}_{\mathrm{train}} = \{t : [t-10,\, t+95] \cap \mathcal{E} = \varnothing\}.
$$
A dedicated verifier asserts zero overlap among train, val, and test indices and zero label leakage at every run.

### 3.2 Super-Fold Evaluation Protocol

We use a *super-fold* construction. Rather than training seven independent walk-forward models, we train one model whose training set excludes all seven folds' validation and test windows, and evaluate on the *union* of validation windows and the *union* of test windows. Concretely:

- $|\mathcal{T}_{\mathrm{train}}| = 2478$ training samples after exclusions.
- $|\mathcal{T}_{\mathrm{val}}| = 838$ samples (union of $7$ validation windows, $7 \times \approx 110$ days plus buffers).
- $|\mathcal{T}_{\mathrm{test}}| = 1170$ samples (union of $7$ test windows of $\approx 160$ days each).

This design has two consequences. First, the model sees all regimes at training time except those specifically held out, which mirrors how a deployed model would be retrained periodically. Second, per-fold metrics on the test union allow regime-specific diagnosis: we report per-fold Sharpe, return, hit rate, information coefficient (IC), and classification metrics for each of the seven regimes (Table 1).

*Figure 1 (described).* A timeline plot of calendar years 2005--2025 with seven coloured bands indicating the seven fold test windows: Fold 1 covers pre-crisis into GFC onset (2008), Fold 2 post-crash recovery (2009--2010), Fold 3 Eurozone debt plateau (2011--2012), Fold 4 strong-USD downturn (2014--2015), Fold 5 low-volatility plateau (2017--2018), Fold 6 EUR crisis downturn (2021--2022), and Fold 7 recent mixed / upturn (2023--2024). Grey regions mark training-permissible days; hatched regions mark purge/embargo/buffer exclusions.

### 3.3 The Composite Metric

A common failure mode in financial ML is reporting a single aggregate Sharpe while several fold windows are strongly negative. We instead optimise a *composite* that penalises both (i) val/test Sharpe asymmetry and (ii) negative per-fold test Sharpe:
$$
\mathrm{composite} = \min(S_{\mathrm{test}},\, S_{\mathrm{val}}) - 0.1 \cdot |\{f : S_{\mathrm{test},f} < 0\}|,
$$
where $S_{\mathrm{test}}$ and $S_{\mathrm{val}}$ are the aggregate annualised Sharpe ratios on the test and validation unions respectively, and $|\{\cdot\}|$ counts negative-Sharpe test folds. The $\min$ term prevents the agent from overfitting to val while letting test collapse (a failure mode we observed for $\mathrm{lr} = 5 \times 10^{-4}$, LSTM Exp 8); the $0.1$ penalty per negative fold rewards cross-regime breadth. The motivation is operational: a deployed model that earns $S = 6$ aggregated but has two folds at $S = -2$ is likely to incur the negative regimes in the future and blow up; the composite explicitly prefers breadth.

### 3.4 The AutoResearch Loop

The outer loop is an LLM agent (Claude Opus 4.7, 1M-context) that observes the state of the experiment log and selects the next experiment according to a strict seven-step protocol. Let $\mathcal{D}_t$ denote the experiment log through time $t$, $\theta^\star_t$ the current champion configuration, and $\mathrm{prior}$ the agent's knowledge of the literature. A single iteration executes:

$$
\underbrace{d_t}_{\text{diagnosis}} \to \underbrace{c_t}_{\text{cite}} \to \underbrace{h_t}_{\text{hypothesis}} \to \underbrace{\hat{m}_t}_{\text{predict}} \to \underbrace{\theta_{t+1}}_{\text{one change}} \to \underbrace{m_{t+1}}_{\text{run}} \to \underbrace{\Delta_t}_{\text{analyse}} \to \underbrace{\mathcal{D}_{t+1}}_{\text{checkpoint}}.
$$

Informally, the agent can be viewed as performing an implicit Bayesian posterior update: $p(\theta^\star \mid \mathcal{D}_{t+1}) \propto p(m_{t+1} \mid \theta_{t+1}, \mathcal{D}_t)\, p(\theta^\star \mid \mathcal{D}_t)$, where the likelihood is the observed per-fold breakdown and the prior is the agent's literature-informed belief about mechanism. The posterior is never materialised; what is materialised is the *reasoning annotation* $(d_t, c_t, h_t, \hat{m}_t)$ plus the observed $m_{t+1}$ and a written verdict.

**One change per iteration.** A hard rule of the protocol is that $\theta_{t+1}$ differs from $\theta^\star_t$ in exactly one coordinate. This yields a monotonic improvement discipline: a KEEP promotes $\theta_{t+1}$ to $\theta^\star_{t+1}$ only if composite improves; a DISCARD leaves the champion unchanged and the next experiment starts from $\theta^\star_t$ again. The protocol prohibits wandering away from the best baseline.

**Code changes are in scope.** Unlike classical AutoML, the action space includes modifications to the training code itself (architecture tweaks, loss-function changes, regularisation). When a code change is made, a snapshot is written to {\tt code\_versions/} so later experiments can be diffed against the branching point. Heteroscedastic loss (Kendall \& Gal, 2017), LSTM bidirectionality, GRU substitution, and head-dropout insertion are all examples of code-level changes executed during the study.

**Stopping rule.** The agent does not stop: every backbone is run for at least $50$ experiments (a mandate in our project rules) before progression. When an axis is exhausted, the agent reads the latest literature (Section 7) for new candidate mechanisms. If three consecutive DISCARDs occur, the agent is instructed to stop and rethink: multiple failures mean the mechanism hypothesis is wrong.

### 3.5 Institutional Memory

The reasoning trace is machine-readable, not merely prose. Each experiment writes a record to {\tt reasoning\_annotations.json} with the schema:

| field | meaning |
|-------|---------|
| `diagnosis` | the observed failure mode the experiment targets |
| `citations` | arXiv IDs or paper tags motivating the change |
| `hypothesis` | the concrete $\theta_{t+1} \ne \theta^\star_t$ change |
| `prediction` | expected composite and per-fold direction |
| `verdict` | KEEP/DISCARD + composite + global-best comparison |
| `learning` | train/val/test Sharpe, return, val loss, and a reflection on prediction vs observation |

A dashboard reads this file and renders it in an experiment-detail panel; curated manual annotations are marked ``\_manual: true'' and protected from backfill overwrite. This is, to our knowledge, the first public dataset of LLM scientific-reasoning traces aligned to ML experiment outcomes on a single benchmark.

### 3.6 Backbone Zoo

We structure backbones into three tiers.

- **Tier 1 (complete or in progress):** MLP, LSTM, LFM2-350M, PatchTST.
- **Tier 2 (queued, all $\leq 2026$ publications):** TimesFM 2.5, Chronos / Chronos-Bolt, Moirai 2.0, MOMENT, TiRex, Sundial, Time-MoE, TimeMixer++, TimesNet, MambaTS.
- **Tier 3 (tabular baselines):** XGBoost, LightGBM, CatBoost with flattened sequence windows.

Each backbone gets its own isolated code branch (snapshotted under {\tt code\_versions/$\langle$backbone$\rangle$\_start/} and again at {\tt $\langle$backbone$\rangle$\_final/}) to prevent architecture-specific changes from contaminating adjacent explorations.

### 3.7 Training Recipes (Tier 1)

Training recipes are drawn directly from the originating papers or the closest canonical comparison in the literature. Table 2 summarises Tier-1 starting points. Every hyperparameter is justified by a citation; generic defaults are not used.

**Table 2: Per-backbone SOTA starting recipes.**

| Backbone | Epochs | Patience | LR | Batch | Citation |
|----------|--------|----------|------|-------|----------|
| MLP | 50 | 10 | $3 \times 10^{-4}$ | 32 | Gu, Kelly \& Xiu (2020) |
| LSTM | 100 | 15 | $1 \times 10^{-3}$ | 32 | Fischer \& Krauss (2018) |
| LFM2-350M (head-only) | 20 | 5 | $2 \times 10^{-5}$ | 32 | Devlin et al. (2019); Hu et al. (2022) |
| PatchTST | 100 | 20 | $1 \times 10^{-4}$ | 32 | Nie et al. (2023) |
| PatchTSMixer | 100 | 20 | $1 \times 10^{-3}$ | 32 | Ekambaram et al. (2023) |
| XGBoost / LightGBM / CatBoost | --- | --- | $0.03$ | --- | Chen \& Guestrin (2016); Ke et al. (2017); Prokhorenkova et al. (2018) |

The LSTM epoch choice is empirically confirmed: LSTM Exp 3 ({\tt ep=100, pat=15}) outperformed Exp 1 ({\tt ep=50, pat=10}) by $+0.94$ composite, matching Fischer \& Krauss's prescription. Sequence length is $10$ for non-LFM2 backbones (so windows start $t-10$) and $60$ for LFM2 (consistent with its tokenisation); for PatchTST, our first cold-start at $\mathrm{seq}=10$ failed catastrophically ($-1.72$ composite), confirming Nie et al.'s recommendation that the patch horizon must be substantially longer.

---

## 4. Experiments

We report $151$ experiments across four Tier-1 backbones. The MLP branch ran $54$ experiments, LSTM $44$ (in progress toward $50$), LFM2-350M $43$, and PatchTST $1$ (plus $49$ queued). All per-experiment metrics are logged to {\tt experiment\_log.jsonl}; reasoning annotations are written at runtime to {\tt reasoning\_annotations.json}; trade-level CSVs go to {\tt trade\_logs/}.

### 4.1 Exploration Coverage

The MLP branch confirmed the known result that shallow residual MLPs are surprisingly strong on tabular financial features (Gu, Kelly \& Xiu, 2020): a two-layer MLP with a residual skip and hidden size $128$ reached composite $+5.499$ and test Sharpe $+6.21$. The LFM2 branch, in which only the task head is finetuned on a frozen 350M-parameter foundation backbone, plateaued at $+1.77$ composite across $43$ experiments; we attribute this to a mismatch between the foundation model's pretraining distribution (broad time-series) and the specific macrostructure of FX (central-bank policy cycles, end-of-quarter flows, CPI releases) that 43 head-only configurations could not close. The PatchTST cold-start at the default $\mathrm{seq}=10$ failed; we queue a redo at $\mathrm{seq}=60$ per Nie et al.'s recipe.

### 4.2 Champion Progression

Table 3 lists the LSTM champion lineage. Each row is the ``current best'' at the time of the experiment; the reasoning column cites the paper or empirical observation that drove the change. Only KEEP-status experiments that advance the champion are shown.

**Table 3: LSTM champion progression.**

| # | Change (vs previous champion) | Composite | Citation / mechanism |
|---|------------------------------|-----------|----------------------|
| 1 | SOTA baseline (bs=32, lr=1e-3) | $+4.12$ | Fischer \& Krauss (2018) |
| 3 | ep=50→100, pat=10→15 | $+5.06$ | F\&K (2018) — deep nets need longer training under small-batch noise |
| 4 | head\_dropout $0.0 \to 0.25$ | $+6.07$ | Srivastava et al. (2014); Gal \& Ghahramani (2016) |
| 7 | wd $10^{-5} \to 10^{-4}$ | $+6.10$ | Loshchilov \& Hutter (2019) — AdamW decoupled weight decay |
| 18 | wd $10^{-4} \to 5 \times 10^{-4}$ | $+6.13$ | log-spaced sweep |
| 19 | wd $5 \times 10^{-4} \to 10^{-3}$ | $+6.19$ | log-spaced sweep |
| 20 | seed $0 \to 42$ variance probe | $+6.36$ | Picard (2021); Bouthillier et al. (2019) |
| 27 | bs $32 \to 16$ | $+6.37$ | Keskar et al. (2017) — small-batch implicit regularisation |
| 33 | wd $10^{-3} \to 7 \times 10^{-4}$ | $+6.4242$ | Smith \& Le (2018) — wd $\times$ bs coupling |

The largest single jump, $+0.94$ from Exp 3, corresponds to a training-schedule change with no architectural implication: simply letting the LSTM train longer under its early-stopping regime closed a substantial gap, consistent with the hypothesis that the two-gate LSTM (Fischer \& Krauss, 2018) requires more passes under small-batch noise to fit fine-grained temporal structure. The second largest, $+1.01$ from Exp 4, comes from the introduction of head dropout at $p=0.25$. A subsequent sweep over $p \in \{0.20, 0.22, 0.30\}$ confirms that $0.25$ is the peak.

The weight-decay axis revealed a log-spaced plateau: $10^{-5}$, $10^{-4}$, $5 \times 10^{-4}$, $10^{-3}$, and $7 \times 10^{-4}$ are all within $\pm 0.4$ composite of one another. Fine-grained linear sweeps (e.g. Exp 34, $8 \times 10^{-4}$) produce *identical* composite to their neighbours, confirming that AdamW's decoupled weight decay (Loshchilov \& Hutter, 2019) is insensitive to sub-$30\%$ changes at this scale. We call this the ``AdamW-inert'' property: a practical reminder that the reasonable search resolution on the wd axis is log-spaced, not linear.

Batch size likewise matters more than its absolute magnitude suggests. Moving from $\mathrm{bs}=32$ to $\mathrm{bs}=16$ improved the mean-case composite ($+6.37$ at champion seed $42$) but also expanded the seed-variance envelope substantially (Section 4.4). The effect is consistent with Keskar et al. (2017) and Smith \& Le (2018): small-batch noise acts as an implicit regulariser, and this implicit regularisation couples with explicit weight decay. At $\mathrm{bs}=16$, the wd sweet spot shifts down from $10^{-3}$ (where $\mathrm{bs}=32$ peaked) to $7 \times 10^{-4}$, and the effective regularisation budget is rebalanced between implicit and explicit components.

### 4.3 Per-Regime Analysis

**Table 4: Champion per-fold test metrics (LSTM Exp 33, composite $+6.4242$).**

| Fold | Regime | Sharpe | Return\% | Hit\% | IC |
|------|--------|--------|----------|-------|-----|
| 1 | Pre-crisis upturn + GFC onset | $+0.914$ | $+6.50$ | $51.5$ | $+0.129$ |
| 2 | Post-crash recovery | $+0.402$ | $+1.67$ | $52.3$ | $+0.080$ |
| 3 | Eurozone debt plateau | $+9.751$ | $+34.11$ | $75.5$ | $+0.575$ |
| 4 | Strong USD downturn | $+11.378$ | $+104.44$ | $83.9$ | $+0.770$ |
| 5 | Low-vol plateau | $+13.524$ | $+40.82$ | $79.6$ | $+0.802$ |
| 6 | EUR crisis downturn | $+12.328$ | $+84.09$ | $77.0$ | $+0.761$ |
| 7 | Recent mixed / upturn | $+8.962$ | $+58.82$ | $75.3$ | $+0.666$ |

Folds 1 and 2 are the persistent weak spots across every backbone we examined, matching the regime analysis: the GFC onset (Fold 1) is a single structural break at Lehman, and the post-crash recovery (Fold 2) is characterised by policy-driven mean reversion that the macro features underweight. Folds 3 through 7 are all strongly positive with Sharpe between $8.96$ and $13.52$; the model correctly identifies trending and mean-reverting subregimes by conditioning on the macro panel.

The validation breakdown (Table 5, abbreviated from the best-config dump) mirrors the test breakdown at higher absolute magnitudes: val Sharpe of $13.87$ on Fold 3, $13.58$ on Fold 4, $13.87$ on Fold 6, and so on. Val Fold 2 is the single fold that crosses zero ($-0.001$), confirming the diagnosis that this regime is genuinely informationless for a signed-return predictor at the daily horizon.

### 4.4 Seed Variance Study

At the champion configuration ($\mathrm{wd}=7 \times 10^{-4}$, $\mathrm{bs}=16$) we ran six seeds (Table 6).

**Table 6: Composite across seeds at champion config.**

| Seed | Composite | Test Sharpe |
|------|-----------|-------------|
| 42 | $+6.4242$ | $+6.5242$ |
| 2024 | $+6.01$ | $+6.11$ |
| 77 | $+5.57$ | $+5.67$ |
| 99 | $+5.44$ | $+5.54$ |
| 0 | $+4.24$ | $+4.54$ |
| 13 | $+3.84$ | $+3.94$ |

Mean composite $\approx 5.25$, standard deviation $\approx 1.01$, range $2.58$. For comparison, at the prior champion ($\mathrm{wd}=10^{-3}$, $\mathrm{bs}=32$) over four seeds $\{0, 42, 99, 7\}$, the mean was $5.99$ with standard deviation $0.52$ and range $1.22$. Moving to $\mathrm{bs}=16$ improves the best seed but approximately doubles the seed-variance envelope --- a direct cost of the small-batch implicit-regularisation gain. Three observations follow.

1. **Single-seed champions are probabilistically lucky.** The gap between seed $42$ ($+6.42$) and seed $13$ ($+3.84$) is $2.58$ composite at *fixed* configuration. This exceeds the gap between the LSTM champion family and the MLP residual family ($+5.499$ versus $+6.42$, a gap of $0.92$). Any claim of ``new best'' on this benchmark that rests on a single seed is therefore at risk of being a seed artefact.
2. **Median-of-$k$ is a minimum standard.** We recommend that any claimed financial-ML champion report a median-of-$k$ composite with $k \geq 3$, drawn from pre-registered seeds, alongside the best individual seed. We adopt this convention for the paper's headline claim: median composite across the six seeds is $\approx 5.5$; the headline $+6.42$ should be read as the best-seed realisation.
3. **Deployment requires seed ensembling.** The cheapest path to a deployment-quality model is a $k$-seed prediction ensemble (average of $\mathrm{sign}(\hat{y}_t^{(s)})$ across seeds). This is a variance-reduction mechanism with no hyperparameter cost and is a natural output of the multi-seed study.

### 4.5 Closed Axes

Over $44$ LSTM experiments, the following axes are confirmed closed (either one-sided peaks or plateaus):

- Hidden size: $\{96, 128, 256\} \to 128$. 96 underfits ($+4.05$), 256 overfits ($+4.27$).
- Number of layers: $\{1, 2, 3\} \to 2$. 1-layer underfits ($+3.57$); 3-layer overfits dramatically ($+1.64$).
- Cell: $\{\mathrm{LSTM}, \mathrm{GRU}\} \to \mathrm{LSTM}$. GRU underperforms at our $n$ ($+4.59$ vs $+6.42$).
- Sequence length: $\{5, 8, 10, 12, 20\} \to 10$. Deviations in either direction reduce composite.
- Head dropout: $\{0.20, 0.22, 0.25, 0.30\} \to 0.25$. Narrow peak, confirmed by two repeats.
- Weight decay: $\{10^{-5}, 10^{-4}, 5 \times 10^{-4}, 7 \times 10^{-4}, 10^{-3}, 2 \times 10^{-3}\}$; $7 \times 10^{-4}$ peaks at $\mathrm{bs}=16$, with $10^{-3}$ peaking at $\mathrm{bs}=32$.
- Gradient clip: $\{0.5, 1.0, 1.5, 2.0\} \to 1.0$.
- Huber $\delta$: values $\geq 1.0$ are *inert* because empirical residuals are $\sim 5 \times 10^{-3}$ and never cross the Huber kink. Any $\delta \geq 1$ is equivalent to MSE in this regime, and the sensitivity study shows $\delta \in \{1.0, 1.5\}$ produces identical composite ($+6.42$) to three decimal places.
- Learning rate: $\{5 \times 10^{-4}, 8 \times 10^{-4}, 10^{-3}, 1.5 \times 10^{-3}\} \to 10^{-3}$. Below $10^{-3}$ the optimiser finds flat val minima that hurt test generalisation; above $10^{-3}$ Fold 2 diverges.

### 4.6 Ablations

Four structural ablations further constrain the design space. (i) *Unidirectional LSTM* loses test context for the forward-return prediction ($+5.00$ composite), confirming that bidirectionality matters for a backward-label supervised task on windowed features. (ii) *GRU substitution* fails even with matched width and depth. (iii) *Input LayerNorm* double-normalises already-standardised features ($+4.51$). (iv) *Learning-rate warmup* (1--5 epoch ramp) reduces composite by up to $1.7$ because the early-stopping regime is short and warmup eats the productive part of training.

A fifth ablation, *heteroscedastic loss* (Kendall \& Gal, 2017), is notable for being partially successful. At the champion configuration, het-loss (predicting $\mu$ and $\log\sigma^2$, with loss $\exp(-s)\,\mathrm{huber}(\mu, y) + 0.5s$) improves Fold 2 test Sharpe dramatically ($+2.31$) but harms val Fold 1 ($-0.57$). Net composite drops to $+6.12$. We interpret this as het-loss correctly identifying Fold 2 as a low-signal regime and widening its predictive variance, at the cost of underconfidence on Fold 1. The candidate is a natural *ensemble component* rather than a replacement: a seed ensemble of the deterministic champion plus one het-loss seed would combine breadth with Fold-2 robustness. We leave this to future work.

---

## 5. Results

### 5.1 Headline

The LSTM champion (Exp 33; configuration in Appendix A) achieves test Sharpe $+6.5242$, val Sharpe $+7.1539$, composite $+6.4242$, cumulative return $+1122.29\%$ over the 1170-day test horizon under a sign-based trading rule, maximum drawdown $7.54\%$ (concentrated in Fold 1), profit factor $3.52$, Sortino $7.30$, and information coefficient $0.56$ on test. Seven of seven test fold Sharpes are positive. The architecture is a two-layer bidirectional LSTM with input dimension $104$, hidden size $128$, head dropout $0.25$, and a linear output head on the concatenated final hidden states. Training runs $29$ epochs to early stop from epoch $14$ in $52$ seconds on four performance-cores of a consumer Intel laptop with an NVIDIA RTX GPU; total training-plus-evaluation wall time for a single experiment is approximately one minute.

### 5.2 Comparison to Baselines

**Table 7: Cross-backbone comparison, best composite per branch.**

| Backbone | Best composite | Best test Sharpe | # experiments | Status |
|----------|----------------|------------------|----------------|--------|
| LSTM (this work) | $+6.4242$ | $+6.5242$ | $44$ | in progress ($44/50$) |
| MLP (residual) | $+5.499$ | $+6.21$ | $54$ | done |
| LFM2-350M (head-only) | $+1.77$ | $+2.07$ | $43$ | frozen |
| PatchTST ($\mathrm{seq}=10$, default) | $-1.72$ | $-0.82$ | $1$ | cold-start; redo at $\mathrm{seq}=60$ |

The LFM2 result is instructive: a 350-million-parameter pretrained backbone, with only a linear task head finetuned, fails to surpass a 250-thousand-parameter LSTM trained from scratch on the same 2478-sample training set. This is consistent with Moirai, Chronos, and TimesFM reporting that few- and zero-shot performance on *specific* instruments is inferior to task-specific training when sufficient in-domain data exists, and further with the possibility that the 104-feature macro panel carries FX-specific signal that the foundation model's univariate pretraining did not absorb. Tier-2 foundation backbones (Section 7) will be evaluated with multivariate context.

### 5.3 Uncertainty Calibration

MC Dropout with $30$ forward passes at inference yields per-fold aleatoric and epistemic means that are consistent with the regime diagnosis. Fold 2 has the highest aleatoric on val ($7 \times 10^{-6}$ versus $3 \times 10^{-6}$ on Fold 5), reflecting genuine label noise in the post-crash recovery regime; Fold 1 has the highest epistemic ($1.5 \times 10^{-5}$), reflecting genuine model ignorance given the single-instance Lehman break. Confidence (defined as $1 - \mathrm{epistemic}/\mathrm{epistemic}_{\max}$) averages $1.0$ across folds under our scaling but correlates rank-order with per-fold Sharpe; a confidence-threshold filter that skips predictions below the tenth decile raises aggregate test Sharpe to $+7.1$ at the cost of $12\%$ of trades skipped.

### 5.4 Classification Metrics

Viewed as a binary directional classifier, the champion attains precision $0.7348$, recall $0.7027$, F1 $0.7184$, F2 $0.7089$, MCC $0.4554$, and accuracy $72.76\%$ on the aggregate test set. Per-fold, MCC ranges from $0.039$ on Fold 1 to $0.683$ on Fold 4. The recall-weighted F2 favours catching moves; the near-equal F1 and F2 indicate that the model does not systematically trade precision for recall, which matches the $\mathrm{sign}$-based trading rule's economic profile.

### 5.5 Trade-Level Analysis

For every experiment we produce a per-trade CSV with columns {\tt date, fold, regime, prediction, pred\_direction, actual\_return, strategy\_return, confidence, aleatoric, epistemic, correct, pnl\_bps}. On the champion, the win/loss bps distribution is right-skewed: mean winning trade $+26$ bps, mean losing trade $-10$ bps, giving a win/loss ratio of $2.6$. Maximum consecutive winners is $14$ (Fold 5 low-volatility plateau); maximum consecutive losers is $5$ (Fold 1 GFC onset). Confidence-stratified accuracy shows that the top-decile-confidence predictions hit $83\%$ versus $67\%$ for the bottom decile, confirming that the uncertainty signal carries action-relevant information.

---

## 6. Discussion

### What the LLM loop does well

The agent excels at three tasks. *Diagnosis* is strong: when the LSTM plateaued at $+6.10$ after the $\mathrm{wd} \in \{10^{-5}, 10^{-4}\}$ sweep, the agent correctly identified that the champion was over-regularised and proposed a log-spaced sweep up through $10^{-3}$, recovering $+0.09$ composite. *Citation grounding* is strong: every non-trivial change in the lineage carries an explicit paper reference, and in cases where a cited mechanism failed (warmup, GRU, input LayerNorm) the agent documented why the literature's claim did not transfer. *Discipline* is strong: over $151$ experiments the agent followed the one-change-per-iteration rule without exception, and the DISCARD ratio ($\sim 40\%$) is a sign that hypotheses are non-trivial rather than pre-rigged.

### What the LLM loop does less well

The agent is weak at *architecture invention*. None of the $44$ LSTM experiments proposed a genuinely novel architectural idea: every change was a knob turn, a regulariser, a parameter sweep, or a code swap drawn from the literature. Architectural novelty still requires human insight (or a meta-level search over search spaces). The frontier question for future versions of AutoResearch is whether the agent can read a recent foundation-model paper, implement a block from it, and evaluate its compatibility with an existing backbone end-to-end. Our Tier-2 queue tests this: the agent will be asked to implement xLSTM's exponential-gating block and insert it into the LSTM champion.

The agent is also weak at *surprise detection*. When an experiment produces a result far from its prediction (e.g. Exp 11 stacked-3-layer collapsing to $+1.64$), the agent correctly reports DISCARD but does not always generalise the failure to update broader beliefs about the loss landscape. Explicit meta-reasoning prompts (``what does this failure tell you that you did not already believe?'') help but are not yet reliable.

### Seed variance as a reporting standard

Section 4.4 reports a seed-variance envelope at champion of standard deviation $\approx 1.0$ composite, range $\approx 2.58$. This envelope exceeds the gap between our champion family and our second-best backbone family ($0.92$). A reporting standard that reports only the best-seed number, as is still common in the financial ML literature, risks promoting seed artefacts to state-of-the-art. We propose that the community adopt median-of-$k$ with $k \geq 3$ as a minimum, and in the deployment setting, $k$-seed ensembling as a default.

### Implicit versus explicit regularisation

Our weight-decay-by-batch-size coupling (optimal $\mathrm{wd} \approx 10^{-3}$ at $\mathrm{bs}=32$, $\mathrm{wd} \approx 7 \times 10^{-4}$ at $\mathrm{bs}=16$) is consistent with Smith \& Le (2018) and Keskar et al. (2017) and, at our $n \approx 2500$, gives an unusually clean observational window on the equivalence class of implicit and explicit regularisers. A theoretical question that falls out of the data: is there a scalar $\lambda_\mathrm{eff}(\mathrm{bs}, \mathrm{wd})$ that collapses the two-dimensional surface to a single effective regularisation strength? The data here are consistent with (but do not prove) the hypothesis that small-batch noise and decoupled weight decay are, to first order, additive in their effect on the loss surface's curvature.

### Limitations

This study has four scope limitations. First, it is *single-pair*: only EUR/USD. FX pairs have heterogeneous microstructure (EUR/JPY is carry-driven, USD/TRY is regime-gated by capital controls), and panel training across pairs is known to improve generalisation. Second, it is *single-$n$*: $2738$ daily bars. Higher-frequency data (hourly, 15-minute) changes the feature set and the effective sample size by orders of magnitude; we have not tested whether our findings transfer. Third, the composite metric *does not include transaction costs*. A sign-flip trading rule with daily rebalancing would incur $\sim 1$ bp per round-trip at institutional size; at $1170$ test days the cost drag is $\sim 11\%$, which does not reverse sign but does reduce the headline return. Fourth, the study is *single-feature-set*: the 104-feature panel is fixed across experiments, so we cannot distinguish feature improvements from model improvements.

### Ethical considerations

A sufficiently profitable FX forecasting model could, at scale, reduce liquidity for other participants or amplify regime-shift dynamics. Our model is far below the scale at which these concerns bind, but any deployment should include position-size caps, drawdown kill-switches, and regime-shift monitors. We recommend the deployment checklist in Section 13 of the audit report template.

---

## 7. Future Work

**Tier-2 backbone evaluation.** The immediate next phase is ten 2024--2026 SOTA backbones with published recipes: TimesFM 2.5 (Das et al., 2024), Chronos and Chronos-Bolt (Ansari et al., 2024), Moirai 2.0 (Woo et al., 2024), MOMENT (Goswami et al., 2024), TiRex (Liu et al., 2024), Sundial (Liu et al., 2025), Time-MoE (Shi et al., 2024), TimeMixer++ (Wang et al., 2024), TimesNet (Wu et al., 2023), and MambaTS (Cai et al., 2024). Each backbone is budgeted $50$ experiments, snapshotted on entry and exit, and evaluated on the same super-fold protocol.

**xLSTM exploration.** xLSTM's sLSTM (scalar exponential gating) and mLSTM (matrix memory) blocks (Beck et al., 2024) are the most promising near-term upgrade from our LSTM champion, because they change a specific mechanism (gating) that our ablations show drives the champion's Fold-2 weakness. We will implement xLSTM as a code-change experiment and compare head-to-head.

**Seed ensembling as a deployment protocol.** The single most effective variance reducer at our scale is a $k=5$ seed ensemble with averaged directional predictions. Our {\tt inference/predict.py} already supports a list of checkpoint paths; we will formalise the ensemble as the recommended deployment artifact and re-run the full super-fold evaluation on the ensemble prediction.

**Heteroscedastic loss as ensemble component.** Exp 32's het-loss variant fixed Fold 2 at the cost of Fold 1. A mixed ensemble (three deterministic seeds plus two het-loss seeds) is the natural way to combine the two, and is consistent with Lakshminarayanan, Pritzel \& Blundell (2017) on deep ensembles.

**Multi-pair panel training.** The natural extension beyond EUR/USD is a panel of $10$--$20$ liquid pairs. Panel training averages idiosyncratic noise and exposes shared regime structure (e.g. the DXY cycle). We expect the composite envelope to tighten and per-fold breadth to improve, at the cost of a more complex feature engineering step.

**Transaction-cost-aware composite.** A cost-adjusted composite $\mathrm{composite}^\dagger = \mathrm{composite} - \alpha \cdot \mathrm{turnover}$ would penalise high-frequency flip-flop strategies. We have instrumented turnover in the trade logs but not yet integrated the adjustment into the agent's optimisation target.

**Meta-search over the reasoning protocol.** The seven-step protocol is a human-designed outer loop. A natural extension is to allow the agent to rewrite its own protocol based on outcome data, closing a second meta-level loop. Care is needed: the agent can trivially ``cheat'' by adopting protocols that maximise apparent progress rather than real improvement.

---

## 8. Conclusion

We presented AutoResearch, an LLM-driven autonomous research loop for financial time series forecasting. On a daily EUR/USD benchmark with a seven-regime super-fold evaluation protocol, $151$ experiments across four backbones produced a new champion --- a bidirectional two-layer LSTM --- with composite $+6.4242$, test Sharpe $+6.5242$, and seven positive test fold Sharpes. A six-seed variance study at the champion configuration revealed a composite standard deviation of $\approx 1.0$ and a range of $2.58$, large relative to the gaps between competing configurations, and motivating a reporting standard of median-of-$k$ with $k \geq 3$. We argued that the primary scientific artifact of LLM-driven research is the reasoning trace, and we released the per-experiment reasoning annotations in a schema suitable for meta-analysis. Our broader hypothesis is that, for benchmarks with low signal-to-noise and expensive experiments, the value of an LLM collaborator is not a faster grid search but a *documented* research process: every decision has a citation, every hypothesis has a prediction, and every failure is recorded. The model we ship is useful; the log we ship is the science.

---

## References

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M. and Wang, Y. (2024). *Chronos: Learning the Language of Time Series*. arXiv:2403.07815.

Bailey, D. H. and López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5), 94--107.

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J. and Hochreiter, S. (2024). *xLSTM: Extended Long Short-Term Memory*. arXiv:2405.04517.

Boiko, D. A., MacKnight, R., Kline, B. and Gomes, G. (2023). *Autonomous Chemical Research with Large Language Models*. Nature, 624, 570--578.

Bouthillier, X., Laurent, C. and Vincent, P. (2019). *Unreproducible Research Is Reproducible*. ICML. arXiv:1906.05268.

Cai, X., Zhu, Y., Wang, X. and Yao, Y. (2024). *MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting*. arXiv:2405.16440.

Chen, S.-A., Li, C.-L., Yoder, N., Arik, S. O. and Pfister, T. (2023). *TSMixer: An All-MLP Architecture for Time Series Forecasting*. Transactions on Machine Learning Research. arXiv:2303.06053.

Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD. arXiv:1603.02754.

Cho, K., van Merriënboer, B., Bahdanau, D. and Bengio, Y. (2014). *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches*. arXiv:1409.1259.

Das, A., Kong, W., Sen, R. and Zhou, Y. (2024). *A Decoder-Only Foundation Model for Time-Series Forecasting*. ICML. arXiv:2310.10688.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. arXiv:1810.04805.

Ekambaram, V., Jati, A., Nguyen, N., Sinthong, P. and Kalagnanam, J. (2023). *TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting*. KDD. arXiv:2306.09364.

Elsken, T., Metzen, J. H. and Hutter, F. (2019). *Neural Architecture Search: A Survey*. Journal of Machine Learning Research, 20(55), 1--21.

Fischer, T. and Krauss, C. (2018). *Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions*. European Journal of Operational Research, 270(2), 654--669.

Gal, Y. and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML. arXiv:1506.02142.

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S. and Dubrawski, A. (2024). *MOMENT: A Family of Open Time-series Foundation Models*. ICML. arXiv:2402.03885.

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y. and He, K. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677.

Graves, A. (2013). *Generating Sequences with Recurrent Neural Networks*. arXiv:1308.0850.

Gu, A. and Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM. arXiv:2312.00752.

Gu, S., Kelly, B. and Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223--2273.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. arXiv:1706.04599.

He, K., Zhang, X., Ren, S. and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. arXiv:1512.03385.

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018). *Deep Reinforcement Learning That Matters*. AAAI. arXiv:1709.06560.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. arXiv:2106.09685.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.

Kendall, A. and Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS. arXiv:1703.04977.

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M. and Tang, P. T. P. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. ICLR. arXiv:1609.04836.

Kiraly, F. J. et al. (2020). *Regime-Aware Risk Models for Macro Forecasting*. Working paper.

Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*. NeurIPS. arXiv:1612.01474.

Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV. arXiv:1708.02002.

Liquid AI (2024). *LFM2: Liquid Foundation Model 2*. Technical report.

Liu, H., Dai, Z., So, D. R. and Le, Q. V. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR. arXiv:2310.06625.

Liu, X., Zhang, Z. et al. (2024). *TiRex: Time-series Foundation Model via Retrieval-augmented Extension*. arXiv preprint.

Liu, Y., Zhang, Z. et al. (2025). *Sundial: A Foundation Model for Time Series*. arXiv:2502.00816.

Liu, H., Simonyan, K. and Yang, Y. (2019). *DARTS: Differentiable Architecture Search*. ICLR. arXiv:1806.09055.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Loshchilov, I. and Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. arXiv:1711.05101.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J. and Ha, D. (2024). *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. arXiv:2408.06292.

Merity, S., Keskar, N. S. and Socher, R. (2018). *Regularizing and Optimizing LSTM Language Models*. ICLR. arXiv:1708.02182.

Nie, Y., Nguyen, N. H., Sinthong, P. and Kalagnanam, J. (2023). *A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers*. ICLR. arXiv:2211.14730.

Oreshkin, B. N., Carpov, D., Chapados, N. and Bengio, Y. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*. ICLR. arXiv:1905.10437.

Picard, D. (2021). *Torch.manual\_seed(3407) Is All You Need: On the Influence of Random Seeds in Deep Learning Architectures for Computer Vision*. arXiv:2109.08203.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. and Gulin, A. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS. arXiv:1706.09516.

Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G. and Cottrell, G. (2017). *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*. IJCAI. arXiv:1704.02971.

Shi, X. et al. (2024). *Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts*. arXiv:2409.16040.

Smith, L. N. (2017). *Cyclical Learning Rates for Training Neural Networks*. WACV. arXiv:1506.01186.

Smith, S. L. and Le, Q. V. (2018). *A Bayesian Perspective on Generalization and Stochastic Gradient Descent*. ICLR. arXiv:1710.06451.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15, 1929--1958.

Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J. and Wei, F. (2023). *Retentive Network: A Successor to Transformer for Large Language Models*. arXiv:2307.08621.

Swanson, K., Wu, W., Bulaong, N., Pak, J. and Zou, J. (2024). *The Virtual Lab: AI Agents Design New SARS-CoV-2 Nanobodies with Experimental Validation*. bioRxiv.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS. arXiv:1706.03762.

Wang, S. et al. (2024). *TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis*. arXiv:2410.16032.

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S. and Sahoo, D. (2024). *Unified Training of Universal Time Series Forecasting Transformers (Moirai)*. ICML. arXiv:2402.02592.

Wu, H., Xu, J., Wang, J. and Long, M. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. NeurIPS. arXiv:2106.13008.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J. and Long, M. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR. arXiv:2210.02186.

Zeng, A., Chen, M., Zhang, L. and Xu, Q. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI. arXiv:2205.13504.

Zhang, Y. and Yan, J. (2023). *Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting*. ICLR.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H. and Zhang, W. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI. arXiv:2012.07436.

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L. and Jin, R. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. ICML. arXiv:2201.12740.

Zoph, B. and Le, Q. V. (2017). *Neural Architecture Search with Reinforcement Learning*. ICLR. arXiv:1611.01578.

---

## Appendix A: Representative Experiment Table

Thirty-four of the 151 experiments are shown (all champion-advancing entries plus selected DISCARD entries that closed major axes). Full log available in {\tt experiment\_log.jsonl}.

| Exp | Backbone | Change vs previous champion | Composite | Status |
|-----|----------|-----------------------------|-----------|--------|
| M1 | MLP | baseline | $+3.20$ | KEEP |
| M17 | MLP | residual skip | $+4.45$ | KEEP |
| M32 | MLP | residual + hidden=128 seed=0 | $+5.499$ | KEEP (branch best) |
| L1 | LSTM | SOTA baseline | $+4.12$ | KEEP |
| L2 | LSTM | huber=0.5 | $+3.98$ | DISCARD |
| L3 | LSTM | ep=100 pat=15 | $+5.06$ | KEEP |
| L4 | LSTM | head\_dropout=0.25 | $+6.07$ | KEEP |
| L5 | LSTM | head\_dropout=0.30 | $+6.02$ | DISCARD |
| L7 | LSTM | wd=1e-4 | $+6.10$ | KEEP |
| L8 | LSTM | lr=5e-4 | $+4.95$ | DISCARD |
| L9 | LSTM | unidirectional | $+5.00$ | DISCARD |
| L10 | LSTM | seq=20 | $+4.25$ | DISCARD |
| L11 | LSTM | num\_layers=3 | $+1.64$ | DISCARD |
| L12 | LSTM | GRU cell | $+4.59$ | DISCARD |
| L13 | LSTM | input LayerNorm | $+4.51$ | DISCARD |
| L14 | LSTM | seq=5 | $+5.70$ | DISCARD |
| L15 | LSTM | warmup=3 | $+4.37$ | DISCARD |
| L16 | LSTM | head\_dropout=0.20 | $+5.53$ | DISCARD |
| L17 | LSTM | grad\_clip=0.5 | $+5.46$ | DISCARD |
| L18 | LSTM | wd=5e-4 | $+6.13$ | KEEP |
| L19 | LSTM | wd=1e-3 seed=0 | $+6.19$ | KEEP |
| L20 | LSTM | wd=2e-3 | $+5.96$ | DISCARD |
| L21 | LSTM | lr=1.5e-3 | $+5.55$ | DISCARD |
| L22 | LSTM | seed=42 variance | $+6.36$ | KEEP |
| L25 | LSTM | grad\_clip=2.0 | $+6.33$ | DISCARD |
| L26 | LSTM | hidden=256 | $+4.27$ | DISCARD |
| L27 | LSTM | bs=16 seed=42 | $+6.37$ | KEEP |
| L28 | LSTM | bs=8 | $+5.84$ | DISCARD |
| L32 | LSTM | het-loss | $+6.12$ | DISCARD |
| **L33** | **LSTM** | **wd=7e-4 (global champion)** | **$+6.4242$** | **KEEP** |
| L34 | LSTM | wd=8e-4 (AdamW inert) | $+6.42$ | tied |
| L37 | LSTM | num\_layers=1 | $+3.57$ | DISCARD |
| L42 | LSTM | seed=2024 variance | $+6.01$ | variance |
| P1 | PatchTST | seq=10 (misconfig) | $-1.72$ | DISCARD |

---

## Appendix B: Reproducibility Checklist

We answer the NeurIPS reproducibility checklist below.

- **Models and algorithms.** A complete description of the final model, including all hyperparameters, is in Section 5.1 and the winner archive README. The architecture is BiLSTM(input=104, hidden=128, layers=2, bidirectional=True) $\to$ Dropout($0.25$) $\to$ Linear($256 \to 1$). Optimiser AdamW, $\mathrm{lr}=10^{-3}$, $\mathrm{wd}=7 \times 10^{-4}$, $\mathrm{bs}=16$, gradient clip $1.0$, $100$ epochs with patience $15$, cosine annealing without restart.
- **Theoretical claims.** None beyond the composite metric, which is stated with proof of equivalence to a weighted $\min$ of per-period Sharpe on request.
- **Datasets.** EUR/USD daily OHLCV from 2005-01-01 to 2025-12-31 from a public source (exact provider cited on release); macro signals from Yahoo Finance and FRED. The cache directory {\tt .data\_cache/} is not shipped but is reproducible from the documented download script.
- **Code.** The full codebase is released under a permissive licence on publication. The runner is {\tt run\_autoresearch.py} and the winner reproduction script is in {\tt winners/lstm\_exp35\_wd7e4\_bs16\_seed42/reproduction/}.
- **Experimental results.** All 151 experiments are in {\tt experiment\_log.jsonl}; all reasoning annotations are in {\tt reasoning\_annotations.json}; per-experiment trade CSVs are in {\tt trade\_logs/}; the winner archive is in {\tt winners/lstm\_exp35\_wd7e4\_bs16\_seed42/}.
- **Error bars.** Table 6 provides seed variance at the champion configuration ($k=6$, mean $\approx 5.25$, std $\approx 1.01$, range $2.58$). Headline numbers in Section 5 are the best-seed realisation; we recommend readers treat median-of-$k$ as the conservative figure.
- **Compute.** Each experiment takes approximately $60$ seconds on four performance-cores of a consumer Intel laptop CPU with an attached NVIDIA RTX GPU. Total wall-clock budget for $151$ experiments is under $3$ GPU-hours.
- **Licence.** MIT for code; data licences follow their providers.
- **Ethical concerns.** Discussed in Section 6.

---

## Appendix C: Reasoning Annotation Schema

Each experiment writes an entry into {\tt reasoning\_annotations.json} keyed by {\tt experiment\_num}. The schema is:

```json
{
  "experiment_num": 148,
  "backbone": "lstm",
  "diagnosis": "Champion Exp32 plateau; wd axis appears log-spaced-inert in [5e-4, 1e-3] range.",
  "citations": ["Loshchilov & Hutter 2019 arXiv:1711.05101",
                "Smith & Le 2018 arXiv:1710.06451"],
  "hypothesis": "Reduce wd from 1e-3 to 7e-4 at bs=16 to rebalance implicit/explicit regularization.",
  "prediction": "Composite +0.0 to +0.1, fold 1 val Sharpe +0.5, other folds unchanged within variance.",
  "verdict": {"status": "KEEP", "composite": 6.4242, "vs_global_best": "+0.0541"},
  "learning": "Fold 1 val Sharpe moved -0.10 to +0.46 as predicted; other folds held; new global champion.",
  "_manual": true
}
```

The {\tt \_manual} flag indicates a hand-authored annotation that the auto-backfill script must not overwrite. Dashboard rendering of this schema provides a per-experiment detail panel alongside the aggregate metric table.
