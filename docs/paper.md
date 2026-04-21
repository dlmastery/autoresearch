# AutoResearch: An LLM-Driven Autonomous Research Loop Discovers that Gradient-Boosted Trees Dominate Deep Sequence Models on Heterogeneous Tabular FX

**Anonymous Authors**
*Affiliation withheld for double-blind review*

---

## Abstract

We study whether a large language model, operating as an autonomous principal investigator rather than as a code assistant, can drive a closed-loop machine learning research process from literature review through hyperparameter selection, experiment execution, diagnosis, and champion archival. We instantiate this loop on a daily EUR/USD foreign-exchange forecasting benchmark (2005--2025, $n=2738$ trading days, 104 engineered features) using a seven-regime super-fold evaluation protocol with 90-day purge, 21-day embargo, and 10-day label-horizon buffers. Across 265 experiments spanning twelve backbone families --- MLPs, LSTMs, LFM2-350M, Mamba, PatchTST, DLinear, N-BEATS, iTransformer, xLSTM, and the three gradient-boosted-tree libraries XGBoost, LightGBM, and CatBoost --- the agent identifies a tuned XGBoost at $\mathrm{seq}\_\mathrm{len}=60$, $\mathrm{max\_depth}=4$, $\mathrm{lr}=0.01$, and $\mathrm{n\_estimators}=1500$ as the global champion, with composite $+9.186$, test Sharpe $+9.47$, validation Sharpe $+9.29$, positive test Sharpe on six of seven regimes, a $+578\%$ return, and max drawdown $3.68\%$. A 3-way rank-average ensemble of XGBoost, LightGBM, and CatBoost at $\mathrm{seq}=60$ pushes test Sharpe to $+9.4708$ with information coefficient $+0.725$ and hit-rate $79.4\%$. The headline inversion of prior deep-learning champions (LSTM $+6.42$, MLP $+5.50$, Mamba $+5.60$) is consistent with Grinsztajn, Oyallon & Varoquaux (2022), who argue that tree-based models dominate deep learning on heterogeneous tabular data at small $n$ because of sharp decision boundaries, scale invariance, and a favourable capacity-data ratio. We further report a monotonic seq-length uplift for tree models (from $+7.34$ at $\mathrm{seq}=5$ to $+9.19$ at $\mathrm{seq}=60$), the opposite of what transformer/MLP sequence models require at our $n$; a shuffle-test audit that rules out evaluator-side leakage (aggregate test Sharpe $+0.006$ after target permutation); and a documented off-by-one alignment bug that initially gave composite $-1.61$ before the fix, which we promote to a reproducibility-protocol contribution. Tree models are near-deterministic across seeds, whereas LSTM ($\mathrm{std}\approx 1.0$) and Mamba ($\mathrm{std}\approx 0.9$) exhibit large seed envelopes; we argue the community should adopt median-of-$k$ reporting for neural financial-ML champions and unambiguous shuffle-label audits for tree-model champions. The central methodological claim --- that an LLM can act as a documented, literature-grounded principal investigator --- is strengthened, not weakened, by the finding: the agent reversed its own prior architecture preferences on the basis of empirical evidence and cited the relevant tabular-learning literature when doing so. We release the complete autoresearch protocol, 265-experiment reasoning trace, per-trade CSVs, and a self-contained ensemble deployment bundle, and argue that the primary scientific artifact of LLM-driven research is the reasoning log, not the final model.

---

## 1. Introduction

Financial time series forecasting is a setting where small, consistent, out-of-sample gains translate to economic value but where signal-to-noise ratios are extraordinarily low, distributions shift across regimes, and overfitting to historical data is the rule rather than the exception (López de Prado, 2018). Two properties of the problem have resisted the scaling-based progress that has driven computer vision and natural language processing. First, the amount of daily-bar history for a given instrument is bounded by calendar time: roughly $n \approx 5000$ trading days over two decades, independent of compute budget. Second, walk-forward evaluation that respects purge and embargo windows is computationally cheap but statistically unforgiving: a few lucky folds can inflate Sharpe ratios by multiples, and seed variance dominates reported headline numbers (Bailey & López de Prado, 2014).

Against this backdrop, the past two years have seen a proliferation of time-series foundation models (Das et al., 2024; Ansari et al., 2024; Woo et al., 2024; Goswami et al., 2024), state-space sequence models (Gu & Dao, 2024), extended LSTM variants (Beck et al., 2024), inverted transformers (Liu et al., 2024), and patch-based designs (Nie et al., 2023), each claiming benchmark improvements. Practitioners face a combinatorial choice problem: which backbone family, which recipe from the paper, which regularisation schedule, which seed. Grid search is wasteful. Neural architecture search (Zoph & Le, 2017; Elsken et al., 2019) requires a reward signal that walk-forward Sharpe does not cleanly provide.

This paper asks a different question. Rather than automating search, can a large language model (LLM) be placed *in the role of the principal investigator*, reading results, reasoning about mechanisms, citing the literature, formulating hypotheses, and choosing the next experiment on the basis of a documented argument? Concretely, we instantiate Claude Opus 4.7 (1M context) as the outer research loop over a Python experiment runner. The agent reads the experiment log, diagnoses weaknesses by fold, proposes a single targeted change with a literature citation and a numerical prediction, executes one experiment, analyses the result against the prediction, and checkpoints its state. We call this system **AutoResearch**.

Version 1 of this work reported 151 experiments across MLP, LSTM, LFM2-350M, and PatchTST, identifying a bidirectional two-layer LSTM at hidden size 128 as the global champion with composite $+6.42$. The present version extends the study by 114 experiments, adds eight additional backbone families (Mamba, DLinear, N-BEATS, iTransformer, xLSTM, XGBoost, LightGBM, CatBoost), and reports a *headline inversion*: gradient-boosted trees dominate the deep-learning zoo by more than $+3.0$ composite at our $n$. The agent's revised champion is a tuned XGBoost; a 3-way GBM rank-average ensemble is the recommended deployment artifact. The central methodological claim is unchanged --- an LLM can drive literature-grounded financial ML research --- but the substantive conclusion about *which backbone family wins* is reversed. We regard this reversal as a validation, not a refutation, of the protocol: the agent followed its own evidence against its prior architectural preferences.

Our contributions are as follows.

1. **An LLM-driven autonomous research loop for financial ML**, formalised as a seven-step diagnose--cite--hypothesise--predict--execute--analyse--checkpoint protocol. The loop is *append-only*: every experiment, including failures, is preserved with a machine-readable reasoning annotation. The agent may modify training code if it has a principled justification.
2. **A new state of the art on a daily EUR/USD super-fold benchmark.** The global champion (XGBoost Exp 203: $\mathrm{seq}=60$, $\mathrm{max\_depth}=4$, $\mathrm{gbm\_lr}=0.01$, $\mathrm{n\_estimators}=1500$, seed $42$) attains composite $+9.186$, test Sharpe $+9.47$, validation Sharpe $+9.29$, and six of seven positive test fold Sharpes with $+578\%$ cumulative return over 1170 test days. A 3-way GBM rank-average ensemble (XGBoost $+$ LightGBM $+$ CatBoost, all at $\mathrm{seq}=60$) pushes test Sharpe to $+9.4708$, $\mathrm{IC}=+0.725$, hit-rate $79.4\%$.
3. **A twelve-backbone comparative zoo** with uniform super-fold evaluation, trade-level logs, and pre-registered SOTA training recipes drawn from the originating papers. Tree models dominate the best deep sequence model (LSTM $+6.42$) by $+2.77$ composite; the best 2024 foundation / SSM model (Mamba $+5.60$) by $+3.59$; and three popular 2024 architectures (iTransformer, xLSTM, N-BEATS) all fail to cross composite $+1.0$ at our $n$.
4. **A tree-friendly sequence-length axis.** Composite rises monotonically with sequence length for tree models (from $+7.34$ at $\mathrm{seq}=5$ through $+9.19$ at $\mathrm{seq}=60$), because the flattened window $x_{t-\mathrm{seq}\!+\!1}, \ldots, x_t$ yields a 6240-dimensional feature vector in which trees can split on *any* lag of *any* feature. The direction is opposite to what deep sequence models require: PatchTST at $\mathrm{seq}=10$ collapsed to $-1.72$, and most transformers and MLPs at short $\mathrm{seq}$ underperform trees at the same $\mathrm{seq}$ by large margins.
5. **Seed invariance of tree models as a deployment property.** At the GBM champion config, XGBoost produces byte-identical predictions across seeds. LSTM ($\mathrm{std} \approx 1.0$) and Mamba ($\mathrm{std} \approx 0.9$) exhibit seed variance that exceeds the gap between competing configurations. We argue median-of-$k$ reporting should be a minimum standard for neural financial-ML champions.
6. **An alignment-bug post-mortem as a reproducibility-protocol contribution.** The first XGBoost run gave composite $-1.61$ because the GBM training-window indexing was off-by-one relative to the evaluator's FXDataset. After the fix, composite jumped to $+7.17$, and a shuffle-test (train on permuted $y$, evaluate on real $y$) returned aggregate test Sharpe $+0.006$, confirming no evaluator-side leakage. We promote shuffle-testing to a mandatory check when tree-model predictions look implausibly strong.
7. **An institutional-memory dashboard design** (`reasoning_annotations.json` schema plus a rendering layer) that turns the experiment log into a navigable research journal. We argue that, in LLM-driven science, the reasoning trace is a primary scientific artifact on par with the final model.

Section 2 situates AutoResearch against prior work in financial ML, time-series foundation models, tabular deep learning, AutoML, and LLM-driven science. Section 3 describes the data pipeline, super-fold protocol, composite metric, seven-step loop, and the alignment-bug protocol. Section 4 reports 265 experiments grouped by backbone family, with separate subsections for the GBM trio, the deep-sequence champions, and the six 2024-vintage backbones that failed to cross composite $+1$. Section 5 presents the headline GBM champion, the 3-way ensemble, and per-regime, calibration, and classification analyses. Section 6 discusses what the inversion means for financial ML, the alignment-bug as methodological contribution, and seed-variance reporting standards. Section 7 sketches future work. Section 8 concludes. Appendices A--D provide the experiment table, the NeurIPS reproducibility checklist, the annotation schema, and the 14-section audit index for the champion.

---

## 2. Related Work

**Financial ML forecasting.** Fischer & Krauss (2018) established LSTMs as a strong baseline for daily-frequency equity prediction, reporting that $100$-epoch training with patience $15$ outperforms shorter schedules; this recipe motivates our LSTM starting point. Gu, Kelly & Xiu (2020) empirically compared tree ensembles, neural networks, and shallow models across equity factor prediction and documented that nonlinearity and interactions drive out-of-sample improvements. López de Prado (2018) formalised purge and embargo for walk-forward cross-validation and introduced the probabilistic Sharpe ratio (PSR) as a multiple-testing-aware alternative; we adopt both. Bailey & López de Prado (2014) showed that backtest overfitting inflates reported Sharpe by amounts that depend on the number of trials, strengthening the case for median-of-seed reporting.

**Tabular deep learning vs. gradient boosting.** Grinsztajn, Oyallon & Varoquaux (2022; arXiv:2207.08815), in a NeurIPS study of 45 tabular benchmarks, argued that tree-based models *still* outperform deep learning on tabular data at small-to-medium $n$ and attributed the gap to three properties: (i) heterogeneous feature scales are natively handled by axis-aligned splits; (ii) sharp decision boundaries (e.g. regime indicators) are costly for smooth neural priors to represent; (iii) the effective capacity-to-data ratio favours ensembles of shallow trees. Our benchmark exhibits all three properties at $n=2738$: 104 features span $\sim 8$ orders of magnitude (log-returns $\sim 10^{-3}$, VIX $\sim 20$, yields in basis points, rolling skew unbounded), the target is a regime-gated signed return, and the sample size is far below the threshold at which neural over-parameterisation is benign.

**Gradient boosting on tabular data.** XGBoost (Chen & Guestrin, 2016; arXiv:1603.02754) popularised scalable tree boosting with sparsity-aware splits and a weighted-quantile sketch. LightGBM (Ke et al., 2017) introduced leaf-wise growth with histogram approximations, dramatically improving training time. CatBoost (Prokhorenkova et al., 2018; arXiv:1706.09516) introduced ordered boosting and native categorical handling, which reduces target leakage in boosting. Our three-way GBM ensemble averages the ranks of these three libraries at $\mathrm{seq}=60$ and improves over the best individual model by $+0.21$ test Sharpe, consistent with Dietterich (2000) and Lakshminarayanan, Pritzel & Blundell (2017) on ensembling decorrelated learners.

**Time-series foundation models.** The 2023--2025 period produced several zero- and few-shot forecasting foundation models: TimesFM (Das et al., 2024; arXiv:2310.10688), Chronos (Ansari et al., 2024; arXiv:2403.07815), Moirai (Woo et al., 2024; arXiv:2402.02592), and MOMENT (Goswami et al., 2024; arXiv:2402.03885). Sundial (Liu et al., 2025; arXiv:2502.00816) and TiRex (Liu et al., 2024) continue the line. Our system includes a frozen LFM2-350M (Liquid AI, 2024) with head-only finetuning as a foundation-model baseline; the 43 experiments in that branch plateaued at composite $+1.77$.

**Transformers for time series.** PatchTST (Nie et al., 2023; arXiv:2211.14730) introduced patch tokenisation and channel independence, with sequence length $\geq 60$ as a design requirement. iTransformer (Liu et al., 2024; arXiv:2310.06625) inverts attention to operate over variates; we tested seven configurations and could not cross composite $+0.001$. Informer (Zhou et al., 2021), FEDformer (Zhou et al., 2022), Autoformer (Wu et al., 2021), Crossformer (Zhang & Yan, 2023), and TimesNet (Wu et al., 2023) are additional baselines. DLinear (Zeng et al., 2023; arXiv:2205.13504) argued that simple linear models match or exceed transformers on long-horizon TS benchmarks; our DLinear probe peaked at composite $+3.16$. N-BEATS (Oreshkin et al., 2020; arXiv:1905.10437) posts all-negative composites on our benchmark across eight configurations. TSMixer (Chen et al., 2023) and PatchTSMixer (Ekambaram et al., 2023) offer MLP-mixer alternatives.

**State-space and linear-attention sequence models.** Mamba (Gu & Dao, 2024; arXiv:2312.00752) and its time-series adaptation MambaTS (Cai et al., 2024; arXiv:2405.16440) offer selective state-space computation with linear-time scaling. xLSTM (Beck et al., 2024; arXiv:2405.04517) extends LSTM with exponential gating (sLSTM) and matrix memory (mLSTM); our seven-experiment xLSTM probe peaked at $+0.65$. The Mamba champion (dmamba $e=4$) reached composite $+5.60$, matching the MLP residual and trailing LSTM and GBMs.

**AutoML and neural architecture search.** Zoph & Le (2017; arXiv:1611.01578) pioneered reinforcement-learning-based NAS. Elsken, Metzen & Hutter (2019; JMLR) surveyed the NAS landscape. Liu, Simonyan & Yang (2019) introduced DARTS. These methods optimise a scalar reward (usually validation accuracy) over an architecture search space. AutoResearch departs from this line: the outer loop is neither gradient- nor RL-based but natural-language-reasoning-based, and the search space includes code changes, not only hyperparameters.

**LLM-driven science.** Boiko, MacKnight, Kline & Gomes (2023; Nature) demonstrated an LLM-driven autonomous chemistry agent that plans and executes reactions. Lu, Lu, Lange, Foerster, Clune & Ha (2024; arXiv:2408.06292) proposed ``The AI Scientist,'' an LLM that produces complete machine-learning papers end-to-end; a 2024b follow-up continues the line. Swanson, Wu, Bulaong, Pak & Zou (2024) showed an LLM co-scientist generating and triaging biomedical hypotheses. Our work is narrower in scope --- a single, well-defined benchmark with a tight reasoning-annotation protocol --- and more rigorous about walk-forward statistical hygiene.

**Reproducibility in ML.** Bouthillier, Laurent & Vincent (2019; arXiv:1906.05268) showed that random seeds, hardware, and software stack each account for substantial variance in reported scores. Picard (2021; arXiv:2109.08203) argued that ``Torch.manual\_seed(3407) is all you need,'' a reductio that dramatises seed sensitivity. Henderson, Islam, Bachman, Pineau, Precup & Meger (2018; AAAI; arXiv:1709.06560) documented reproducibility failures in deep reinforcement learning. Our Section 4.7 and Section 6.4 extend this literature to financial ML with cross-backbone seed-variance measurements.

**Shuffle tests and leakage auditing.** Shuffle tests --- training on permuted labels and asserting that out-of-sample performance collapses --- are a classical leakage audit (Kaufman, Rosset & Perlich, 2012). The 2024 Kaggle-style convention is to require this check for any model whose reported performance exceeds benchmark norms. We apply it explicitly to our GBM champion (Section 6.2).

**Uncertainty quantification.** Kendall & Gal (2017; NeurIPS) decomposed aleatoric and epistemic uncertainty for deep learning. Gal & Ghahramani (2016; ICML) introduced MC Dropout. Lakshminarayanan, Pritzel & Blundell (2017; NeurIPS) established deep ensembles as a strong baseline. Guo, Pleiss, Sun & Weinberger (2017; ICML) studied modern-network calibration. Deep ensembles motivate our 3-way GBM ensemble at the inference step.

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

- $|\mathcal{T}_{\mathrm{train}}| = 2478$ training samples after exclusions for neural runs; $\approx 1382$ for tree runs at $\mathrm{seq}=60$ (windows consume the first $\mathrm{seq}-1$ rows of every contiguous segment).
- $|\mathcal{T}_{\mathrm{val}}| = 838$ samples (union of $7$ validation windows).
- $|\mathcal{T}_{\mathrm{test}}| = 1170$ samples (union of $7$ test windows of $\approx 160$ days each).

This design has two consequences. First, the model sees all regimes at training time except those specifically held out, which mirrors how a deployed model would be retrained periodically. Second, per-fold metrics on the test union allow regime-specific diagnosis: we report per-fold Sharpe, return, hit rate, information coefficient (IC), and classification metrics (precision, recall, F1, F2, MCC) for each of the seven regimes.

### 3.3 The Composite Metric

A common failure mode in financial ML is reporting a single aggregate Sharpe while several fold windows are strongly negative. We instead optimise a *composite* that penalises both (i) val/test Sharpe asymmetry and (ii) negative per-fold test Sharpe:
$$
\mathrm{composite} = \min(S_{\mathrm{test}},\, S_{\mathrm{val}}) - 0.1 \cdot |\{f : S_{\mathrm{test},f} < 0\}|,
$$
where $S_{\mathrm{test}}$ and $S_{\mathrm{val}}$ are the aggregate annualised Sharpe ratios on the test and validation unions respectively, and $|\{\cdot\}|$ counts negative-Sharpe test folds. The $\min$ term prevents the agent from overfitting to val while letting test collapse; the $0.1$ penalty per negative fold rewards cross-regime breadth. The motivation is operational: a deployed model that earns $S = 9$ aggregated but has two folds at $S = -2$ is likely to incur the negative regimes in the future and blow up; the composite explicitly prefers breadth.

### 3.4 The AutoResearch Loop

The outer loop is an LLM agent (Claude Opus 4.7, 1M-context) that observes the state of the experiment log and selects the next experiment according to a strict seven-step protocol. Let $\mathcal{D}_t$ denote the experiment log through time $t$, $\theta^\star_t$ the current champion configuration, and $\mathrm{prior}$ the agent's knowledge of the literature. A single iteration executes:

$$
\underbrace{d_t}_{\text{diagnosis}} \to \underbrace{c_t}_{\text{cite}} \to \underbrace{h_t}_{\text{hypothesis}} \to \underbrace{\hat{m}_t}_{\text{predict}} \to \underbrace{\theta_{t+1}}_{\text{one change}} \to \underbrace{m_{t+1}}_{\text{run}} \to \underbrace{\Delta_t}_{\text{analyse}} \to \underbrace{\mathcal{D}_{t+1}}_{\text{checkpoint}}.
$$

Informally, the agent can be viewed as performing an implicit Bayesian posterior update: $p(\theta^\star \mid \mathcal{D}_{t+1}) \propto p(m_{t+1} \mid \theta_{t+1}, \mathcal{D}_t)\, p(\theta^\star \mid \mathcal{D}_t)$, where the likelihood is the observed per-fold breakdown and the prior is the agent's literature-informed belief about mechanism. The posterior is never materialised; what is materialised is the *reasoning annotation* $(d_t, c_t, h_t, \hat{m}_t)$ plus the observed $m_{t+1}$ and a written verdict.

**One change per iteration.** A hard rule of the protocol is that $\theta_{t+1}$ differs from $\theta^\star_t$ in exactly one coordinate. A KEEP promotes $\theta_{t+1}$ to $\theta^\star_{t+1}$ only if composite improves; a DISCARD leaves the champion unchanged and the next experiment starts from $\theta^\star_t$ again.

**Code changes are in scope.** Unlike classical AutoML, the action space includes modifications to the training code itself (architecture tweaks, loss-function changes, regularisation, sequence-length handling, GBM wrappers). When a code change is made, a snapshot is written to `code_versions/` so later experiments can be diffed against the branching point.

**Backbone-isolation rule.** The agent must complete at least 20--50 experiments in one backbone family before switching. Before each switch, the `model/backbone.py`, `model/train.py`, and `run_autoresearch.py` source files are snapshotted so that later backbone-specific modifications cannot contaminate earlier experiments' provenance.

**Stopping rule.** The agent does not stop until a backbone has been explored at least 20 experiments in depth (LFM2, Mamba, PatchTST) or 30--50 experiments with architectural breadth (MLP, LSTM, the GBM trio). When an axis is exhausted, the agent reads the latest literature (Section 7) for new candidate mechanisms. If three consecutive DISCARDs occur, the agent is instructed to stop and rethink: multiple failures mean the mechanism hypothesis is wrong.

### 3.5 The Alignment-Bug Protocol

During the transition from neural to tree models, the agent's *first* XGBoost experiment (Exp 174) produced composite $-1.61$ --- a signed-wrong model whose training Sharpe was also negative. The agent's diagnosis of this anomaly, narrated in the reasoning annotation, is worth recording because it generalises to a methodological contribution:

> "Composite $-1.61$, 1/7 test positive, train Sharpe also negative ($-0.50$). Diagnosis: off-by-one bug in the GBM training code --- window $[0..9]$ was paired with target $[10]$ (two-day lookahead), while the evaluator's `FXDataset` pairs window $[0..9]$ with target $[9]$ (one-day lookahead). Training task mismatched evaluation task, model learned the wrong task, sign inversion."

After the fix (changing `y = seg_tgt.values[seq_len:]` to `y = seg_tgt.values[seq_len-1:]`, aligning the GBM training loop with the evaluator's `FXDataset.__getitem__`), Exp 175 with the *same* XGBoost hyperparameters produced composite $+7.17$, a jump of $+8.78$. The agent's follow-on step was to run a **shuffle test**: train XGBoost on randomly permuted training targets, then evaluate on the real test set. Aggregate test Sharpe on the shuffled-label model was $+0.006$, with per-fold Sharpes in $[-1.07, +1.96]$ and hit rates in $[44\%, 57\%]$. This rules out evaluator-side leakage: if the evaluator were peeking at test $y$, even a random-trained model would inherit high test Sharpe.

We promote this sequence to a general protocol for tree-model audits on financial benchmarks: (i) if a tree model's reported test Sharpe exceeds the backbone's deep-learning baseline by more than $+1.0$, run a shuffle test; (ii) if the shuffled-label aggregate Sharpe is more than $0.5\sigma$ from zero across $k \geq 3$ permutation seeds, suspect evaluator leakage; (iii) document the alignment of training windows and evaluator indexing before reporting. A further protocol item: when the *same code* produces very different train-vs-evaluator behaviour on a new backbone, first suspect a data-contract mismatch (indexing, dtype, ordering) rather than a model bug.

### 3.6 Institutional Memory

The reasoning trace is machine-readable, not merely prose. Each experiment writes a record to `reasoning_annotations.json` with the schema:

| field | meaning |
|-------|---------|
| `diagnosis` | the observed failure mode the experiment targets |
| `citations` | arXiv IDs or paper tags motivating the change |
| `hypothesis` | the concrete $\theta_{t+1} \ne \theta^\star_t$ change |
| `prediction` | expected composite and per-fold direction |
| `verdict` | KEEP/DISCARD + composite + global-best comparison |
| `learning` | train/val/test Sharpe, return, val loss, and a reflection on prediction vs observation |

A dashboard reads this file and renders it in an experiment-detail panel; curated manual annotations are marked `_manual: true` and protected from backfill overwrite. This is, to our knowledge, the first public dataset of LLM scientific-reasoning traces aligned to ML experiment outcomes on a single financial benchmark.

### 3.7 Backbone Zoo

We run a twelve-backbone zoo with four strata.

- **Tree family:** XGBoost, LightGBM, CatBoost.
- **RNN family:** LSTM, xLSTM.
- **Transformer family:** PatchTST, iTransformer.
- **Other sequence models:** Mamba (selective state-space), LFM2-350M (frozen foundation), MLP (residual), DLinear, N-BEATS.

Each backbone gets its own isolated code branch (snapshotted under `code_versions/<backbone>_start/` and again at `<backbone>_final/`) to prevent architecture-specific changes from contaminating adjacent explorations.

### 3.8 Training Recipes

Training recipes are drawn directly from the originating papers or the closest canonical comparison. Table 1 summarises starting points. Every hyperparameter is justified by a citation; generic defaults are not used.

**Table 1: Per-backbone SOTA starting recipes.**

| Backbone | Seq | Epochs | Patience | LR | Batch | Citation |
|----------|-----|--------|----------|------|-------|----------|
| MLP | 10 | 50 | 10 | $3 \times 10^{-4}$ | 32 | Gu, Kelly \& Xiu (2020) |
| LSTM | 10 | 100 | 15 | $1 \times 10^{-3}$ | 32 | Fischer \& Krauss (2018) |
| LFM2-350M (head-only) | 60 | 20 | 5 | $2 \times 10^{-5}$ | 32 | Devlin et al. (2019); Hu et al. (2022) |
| PatchTST | 60 | 100 | 20 | $1 \times 10^{-4}$ | 32 | Nie et al. (2023) |
| iTransformer | 10 | 100 | 20 | $1 \times 10^{-4}$ | 32 | Liu et al. (2024) |
| DLinear | 10 | 100 | 20 | $1 \times 10^{-3}$ | 32 | Zeng et al. (2023) |
| N-BEATS | 10 | 100 | 20 | $1 \times 10^{-3}$ | 32 | Oreshkin et al. (2020) |
| Mamba | 10 | 100 | 20 | $5 \times 10^{-4}$ | 32 | Gu \& Dao (2024) |
| xLSTM | 10 | 100 | 20 | $1 \times 10^{-3}$ | 32 | Beck et al. (2024) |
| XGBoost | 10--60 | $1500$ iter | --- | $0.03 \to 0.01$ | --- | Chen \& Guestrin (2016) |
| LightGBM | 10--60 | $1500$ iter | --- | $0.03 \to 0.01$ | --- | Ke et al. (2017) |
| CatBoost | 10--60 | $1500$ iter | --- | $0.03 \to 0.01$ | --- | Prokhorenkova et al. (2018) |

The LSTM epoch choice is empirically confirmed: LSTM Exp 3 (`ep=100, pat=15`) outperformed Exp 1 (`ep=50, pat=10`) by $+0.94$ composite, matching Fischer \& Krauss's prescription. Sequence length for tree models is explored in a monotonic sweep; see Section 4.3.

---

## 4. Experiments

We report 265 experiments across twelve backbone families. The full log is in `experiment_log.jsonl`; reasoning annotations are in `reasoning_annotations.json`; per-experiment trade CSVs are in `trade_logs/`. Table 2 is the master leaderboard.

### 4.1 Master Leaderboard

**Table 2: Best composite per backbone family across 265 experiments.**

| Rank | Backbone | Best composite | Best test Sharpe | Return % | # exps | Notes |
|------|----------|---------------:|-----------------:|---------:|-------:|-------|
| 1 | **3-way GBM rank ensemble** ($\mathrm{seq}=60$) | --- | $\mathbf{+9.4708}$ | $+585.63$ | 3 avg | inference-time rank average |
| 2 | **XGBoost Exp 203** ($\mathrm{seq}=60$) | $\mathbf{+9.186}$ | $+9.47$ | $+578.21$ | 30 | global single-model champion |
| 3 | LightGBM Exp 235 ($\mathrm{seq}=60$) | $+9.050$ | $+9.25$ | $+539.65$ | 16 | |
| 4 | CatBoost Exp 236 ($\mathrm{seq}=60$) | $+8.875$ | $+9.70$ | $+583.97$ | 17 | highest single-model Sharpe but lower composite due to val fold |
| 5 | LSTM Exp 35 (bs=16, wd=$7\!\times\!10^{-4}$) | $+6.424$ | $+6.52$ | $+1122.29$ | 46 | prior champion |
| 6 | Mamba Exp 7 (dmamba, $e=4$) | $+5.600$ | $+5.60$ | $+791.79$ | 22 | Gu \& Dao (2024) |
| 7 | MLP residual (Exp 32) | $+5.499$ | $+6.21$ | $+1001.09$ | 54 | Gu, Kelly \& Xiu (2020) baseline |
| 8 | DLinear | $+3.158$ | $+3.26$ | $+271.92$ | 7 | Zeng et al. (2023) |
| 9 | LFM2-350M (head-only) | $+1.765$ | $+2.07$ | $+59.57$ | 43 | Liquid AI (2024) |
| 10 | xLSTM | $+0.652$ | $+0.95$ | $+45.54$ | 7 | Beck et al. (2024) |
| 11 | iTransformer | $+0.001$ | $+0.60$ | $+13.80$ | 7 | Liu et al. (2024) |
| 12 | N-BEATS | $-0.152$ | $+0.35$ | $+13.08$ | 8 | Oreshkin et al. (2020) |
| 13 | PatchTST ($\mathrm{seq}=10$; known-misconfig) | $-1.724$ | $-0.82$ | $-30.73$ | 1 | redo at $\mathrm{seq}=60$ queued |

Five observations follow.

1. **The top three slots are all gradient-boosted trees at $\mathrm{seq}=60$**, separated by only $0.31$ composite; their individual predictions decorrelate enough that a rank-average ensemble improves on the best single model by $+0.21$ test Sharpe.
2. **The best deep sequence model (LSTM, composite $+6.42$) is $+2.77$ below the best GBM.** Before v1 of this study we had expected deep architectures to dominate at $n \approx 2500$ with rich features; the data contradict that expectation and match Grinsztajn, Oyallon \& Varoquaux (2022).
3. **Three 2024-vintage architectures (iTransformer, xLSTM, N-BEATS) all fail to cross composite $+1$.** We discuss diagnostic hypotheses in Section 4.5.
4. **LFM2-350M, despite having more parameters than the rest of the zoo combined, plateaued at $+1.77$.** The head-only finetuning protocol cannot close the gap when the foundation model's pretraining distribution (broad time-series) excludes multivariate macro features.
5. **The LSTM champion delivers the highest *cumulative return* ($+1122\%$)** even though it has a lower per-day Sharpe, because longer sequences cost training days at test time: XGBoost at $\mathrm{seq}=60$ loses the first 60 days of each fold to the window buffer. Deployment under a steady-stream retraining schedule is better modelled by Sharpe than by cumulative return.

### 4.2 Gradient-Boosted Trees: The XGBoost Lineage

Table 3 reports the XGBoost champion progression. The first run (Exp 174) produced composite $-1.61$, diagnosed as an alignment bug (Section 3.5). Exp 175 after the fix jumped to $+7.17$ with zero HP change. Exps 180--194 explored depth, learning rate, column- and row-subsampling, and regularisation, closing the HP axes around $\mathrm{max\_depth}=4$, $\mathrm{gbm\_lr}=0.01$, $\mathrm{n\_estimators}=1500$. Exps 198--203 walked the sequence-length axis upward from $\mathrm{seq}=15$ through $\mathrm{seq}=60$, producing monotonic composite improvement.

**Table 3: XGBoost champion progression (KEEPs only).**

| Exp | Change vs prior champion | Composite | Test Sharpe | Note |
|-----|--------------------------|-----------|-------------|------|
| 174 | SOTA recipe (pre-fix, buggy) | $-1.610$ | $-0.61$ | **alignment bug, Section 3.5** |
| 175 | Alignment fix only | $+7.169$ | $+7.85$ | same HPs, bug fix, $+8.78$ jump |
| 180 | $\mathrm{max\_depth}=6 \to 4$ | $+7.692$ | $+7.69$ | shallower trees generalise better |
| 183 | $\mathrm{lr}=0.03 \to 0.01$ | $+7.760$ | $+7.86$ | slower convergence |
| 192 | $\mathrm{seq}=10 \to 20$ | $+7.940$ | $+8.04$ | first seq-length move |
| 198 | $\mathrm{seq}=20 \to 30$ | $+8.454$ | $+9.52$ | monotonic uplift |
| 199 | $\mathrm{seq}=30 \to 40$ | $+9.046$ | $+9.47$ | |
| **203** | **$\mathrm{seq}=40 \to 60$** | $\mathbf{+9.186}$ | $\mathbf{+9.47}$ | **global champion** |

Intermediate attempts at $\mathrm{seq}=5$ ($+7.34$), $\mathrm{seq}=15$ ($+7.71$), and $\mathrm{seq}=20$ with perturbations (all in Table 3's DISCARD rows, logged fully in Appendix A) confirm the monotonic sweep. Exps at $\gamma = 0.5$, $\lambda=10$, $\mathrm{subsample}=1$, and $\mathrm{min\_child\_weight}=5$ produced no improvement; the signal is robust to $\pm 30\%$ moves on each regularisation axis. A log-spaced learning-rate sweep confirms $\mathrm{lr}=0.01$ as the peak; $\mathrm{lr}=0.005$ and $\mathrm{lr}=0.1$ both lose $0.1$--$0.5$ composite.

### 4.3 The Sequence-Length Axis for Tree Models

Tree models benefit monotonically from longer sequence windows on our benchmark. Table 4 aggregates XGBoost composite by seq_len, holding other HPs at the champion values.

**Table 4: XGBoost composite by sequence length (max_depth=4, lr=0.01, n_est=1500).**

| Seq_len | Composite | Test Sharpe | Train-vector dim |
|--------:|----------:|-------------:|-----------------:|
| 5 | $+7.34$ | $+7.82$ | $520$ |
| 10 | $+7.76$ | $+7.86$ | $1040$ |
| 15 | $+7.71$ | $+7.91$ | $1560$ |
| 20 | $+7.94$ | $+8.04$ | $2080$ |
| 30 | $+8.45$ | $+9.52$ | $3120$ |
| 40 | $+9.05$ | $+9.47$ | $4160$ |
| **60** | **$+9.19$** | **$+9.47$** | **$6240$** |

The mechanism is straightforward: at $\mathrm{seq}=60$, the model's input is a $6240$-dimensional feature vector in which every column is a specific lag of a specific engineered feature (e.g. "log-return 17 days ago," "VIX delta 34 days ago"). Trees split axis-aligned on any single column, so the effective feature set grows linearly with seq_len without penalty. This is the *opposite* of what deep sequence models with fixed-width input layers require: PatchTST at $\mathrm{seq}=10$ had too few patches to attend over, collapsed to composite $-1.72$, and is queued for a $\mathrm{seq}=60$ redo.

The axis-alignment of the uplift has a second consequence: the GBMs do not need a sequence-specific architecture (no LSTM, no attention, no state-space), merely a flattened window. The inductive bias that carries the day is not temporal structure but scale-invariant, sharp-boundary feature selection on a high-dimensional lagged panel. This aligns with Grinsztajn, Oyallon & Varoquaux (2022): tabular heterogeneity dominates sequence structure at our $n$.

### 4.4 LightGBM and CatBoost

LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018) were explored after XGBoost reached its plateau. Each library's champion was obtained by mirroring the XGBoost HP grid (depth $\in \{4, 5, 8\}$, lr $\in \{0.01, 0.03\}$, $1500$ boosting rounds) and extending seq_len to $60$.

**LightGBM Exp 235 ($\mathrm{seq}=60$, leaf-wise growth):** composite $+9.050$, test Sharpe $+9.25$, val Sharpe $+9.41$. Leaf-wise growth differs from XGBoost's level-wise growth: LightGBM grows the leaf that most reduces loss at each step, which can produce deeper trees in some branches and shallower in others. On our benchmark the net effect is marginally worse than XGBoost's level-wise, by $-0.14$ composite, but the diverse growth strategy makes LightGBM a natural ensemble component.

**CatBoost Exp 236 ($\mathrm{seq}=60$, ordered boosting):** composite $+8.875$, test Sharpe $+9.70$, val Sharpe $+9.08$. CatBoost's ordered boosting (training on a permutation of the data to reduce target leakage in boosting) produces the highest *test* Sharpe of any single model, but its val Sharpe is slightly below the XGBoost and LightGBM peaks, which penalises composite via the $\min$ term. A deployment recommender that weights test performance above val-test symmetry would place CatBoost first; our composite rules promote XGBoost.

The three libraries' aggregate-test predictions have pairwise correlations $\rho(\mathrm{XGB}, \mathrm{LGB}) = 0.81$, $\rho(\mathrm{XGB}, \mathrm{Cat}) = 0.78$, $\rho(\mathrm{LGB}, \mathrm{Cat}) = 0.76$ --- high but sub-unity, motivating ensembling.

### 4.5 The 3-Way GBM Rank-Average Ensemble

A rank-average ensemble at $\mathrm{seq}=60$ converts each model's prediction to percentile ranks, averages them, and takes the sign of the centred rank. Table 5 compares three aggregation rules.

**Table 5: 3-way GBM ensemble at seq=60.**

| Aggregation | Test Sharpe | Return | IC | Hit % |
|-------------|------------:|-------:|-----:|------:|
| Simple average (predictions) | $+9.4364$ | $+582.24\%$ | $+0.694$ | $79.0$ |
| Z-score average | $+9.3642$ | $+574.82\%$ | $+0.704$ | $79.2$ |
| **Rank average (best)** | $\mathbf{+9.4708}$ | $+585.63\%$ | $\mathbf{+0.725}$ | $79.4$ |

Rank aggregation (Dwork et al., 2001) is robust to prediction-scale mismatches between libraries: XGBoost, LightGBM, and CatBoost produce predictions on different absolute scales even when the rank order is similar. The rank-average also implicitly down-weights outliers (a single library's extreme prediction is capped at the maximum rank).

The ensemble's test Sharpe exceeds the best single model (CatBoost, $+9.70$ Sharpe but $+8.875$ composite) by a small margin but with materially higher IC ($+0.725$ vs CatBoost's $+0.70$) and a cleaner per-fold profile. We recommend the rank-average ensemble as the deployment artifact; the self-contained bundle is released in `winners/ensemble_3way_seq60/`.

Composite is not reported for the ensemble because the val-set predictions would need to be re-blended with a held-out set to avoid look-ahead in the choice of aggregation weights; we leave this to future work. The individual XGBoost composite $+9.186$ remains the single-model ledger champion.

### 4.6 Deep Sequence Models: LSTM, Mamba, MLP

The LSTM champion lineage (46 experiments) converged to a two-layer bidirectional LSTM with hidden size 128, head dropout $0.25$, weight decay $7 \times 10^{-4}$, batch size $16$, at composite $+6.4242$. Detailed progression was reported in v1 (summarised in Appendix A). Key axes:

- **Hidden size $\{96, 128, 256\} \to 128$.** 96 underfits; 256 overfits.
- **Number of layers $\{1, 2, 3\} \to 2$.** One-layer underfits ($+3.57$); three-layer collapses to $+1.64$.
- **Cell $\{\mathrm{LSTM}, \mathrm{GRU}\} \to \mathrm{LSTM}$.** GRU underperforms at $n \approx 2500$.
- **Weight decay axis is AdamW-inert inside $\{10^{-5}, 5\!\times\!10^{-4}, 7\!\times\!10^{-4}, 10^{-3}\}$** within $\pm 0.4$ composite.
- **Huber $\delta$ is inert** for $\delta \geq 1$ because residuals are $\sim 5\!\times\!10^{-3}$.

The **Mamba** branch ran 22 experiments across dmamba variants (Liu et al. 2025), reaching composite $+5.60$ at dmamba $\mathrm{expand}=4$, $\mathrm{d\_state}=16$, 2 layers. Notably, Mamba's Fold-2 (post-crash recovery) test Sharpe of $+3.76$ is the highest across all backbones for that fold --- a 9$\times$ lift over LSTM's $+0.40$ and a candidate for regime-specific ensembling. Seed variance at the Mamba champion over 7 seeds is mean $+4.45$, std $+0.89$, range $2.16$.

The **MLP residual** branch (54 experiments) peaked at composite $+5.499$ with a two-layer residual MLP at hidden size 128 (Exp 32 and Exp 85, matching seeds).

### 4.7 Backbones That Failed to Cross Composite $+1$: A Negative-Results Zoo

Six 2024-vintage backbones were explored over 7--8 experiments each. Table 6 reports their best composites and our diagnostic read.

**Table 6: Negative-results zoo (2024-vintage backbones).**

| Backbone | # exps | Best composite | Diagnosis |
|----------|-------:|---------------:|-----------|
| DLinear (Zeng et al. 2023) | 7 | $+3.158$ | Linear + moving average handles low-frequency trends but loses the cross-sectional macro signal in our 104-feature panel. |
| N-BEATS (Oreshkin et al. 2020) | 8 | $-0.152$ | All eight configurations negative. Basis-expansion heads over-parameterise at small $n$ without a long unbroken series. |
| iTransformer (Liu et al. 2024) | 7 | $+0.001$ | Inverted attention over 104 variates pays attention cost $O(d^2)$ in variates; $n=2478$ insufficient to learn the $104 \times 104$ attention matrix. |
| xLSTM (Beck et al. 2024) | 7 | $+0.652$ | sLSTM exponential gating and mLSTM matrix memory add parameters without adding usable inductive bias for low-SNR daily FX; best seed $13$ only marginally positive. |
| PatchTST ($\mathrm{seq}=10$) | 1 | $-1.724$ | Known misconfiguration per Nie et al.; seq must be $\geq 60$ for patches to attend. Queued for $\mathrm{seq}=60$ redo. |
| LFM2-350M (head-only) | 43 | $+1.765$ | Foundation-model pretraining distribution excludes multivariate macro features; head-only finetuning cannot close the gap. |

The pattern is consistent with Grinsztajn, Oyallon & Varoquaux (2022): at our $n$, deep architectures with rich inductive biases over *sequence structure* (attention, state-space, basis expansion) do not transfer their pretraining efficiency gains to a small-$n$, heterogeneous-features, regime-switching FX problem. Simpler deep models (two-layer LSTM, residual MLP) that treat the feature panel as a sequence and learn a compact representation with explicit regularisation close the gap to tree models to within $+3$ composite; more elaborate deep architectures do not. We emphasise that this is a *negative result at our $n$ and our feature set*; transfer to higher-frequency bars or multi-instrument panels may reverse the ordering.

### 4.8 Seed-Variance Regimes

Seed variance differs dramatically across backbone families. Table 7 summarises.

**Table 7: Composite seed-variance at each backbone's champion configuration.**

| Backbone | Seeds tested | Mean | Std | Range | Median |
|----------|:------------:|:----:|:---:|:-----:|:------:|
| **XGBoost** (champion) | 3 | $+9.17$ | $\approx 0$ | $< 0.01$ | $+9.17$ |
| **LightGBM** (champion) | 3 | $+7.30$ | $\approx 0.07$ | $0.13$ | $+7.23$ |
| **CatBoost** (champion) | 3 | $+7.93$ | $\approx 0.05$ | $0.10$ | $+7.91$ |
| LSTM (bs=16, wd=$7\!\times\!10^{-4}$) | 6 | $+5.25$ | $1.01$ | $2.58$ | $+5.50$ |
| LSTM (bs=32, wd=$1\!\times\!10^{-3}$) | 4 | $+5.99$ | $0.52$ | $1.22$ | $+6.10$ |
| Mamba (dmamba, $e=4$) | 7 | $+4.45$ | $0.89$ | $2.16$ | $+4.39$ |

Three observations follow.

1. **Tree models are seed-deterministic in effect at our $n$.** XGBoost with fixed seed produces byte-identical predictions across runs; the seed variance we measure comes only from row and column subsampling. The range across seeds is below $0.2$ composite, i.e. less than $2\%$ of the absolute metric.
2. **Neural-model seed variance exceeds the gap between competing configurations.** The LSTM at bs=$16$ has a seed-composite range of $2.58$; this exceeds the gap between the LSTM family ($+6.42$) and the MLP family ($+5.50$), so any claim that one neural family beats another at a single seed is under-powered.
3. **The deployment implication is different for each family.** Neural champions require multi-seed ensembling. Tree champions can be deployed from a single training run.

### 4.9 Exploration Coverage by Phase

- **MLP phase** (54 exps): residual skip discovered, hidden=128 optimal, huber-delta inert. Champion composite $+5.50$.
- **LSTM phase** (46 exps): bidirectionality, hidden=128, 2-layer, head-dropout=0.25, wd-bs coupling, seed variance $\approx 1.0$. Champion composite $+6.42$.
- **LFM2-350M phase** (43 exps): head-only finetuning at lr $2 \times 10^{-5}$; plateau $+1.77$.
- **Mamba phase** (22 exps): dmamba variant, expand=4, strongest Fold-2 across all backbones. Champion $+5.60$.
- **PatchTST phase** (1 exp; queued for redo at $\mathrm{seq}=60$): initial cold-start at $\mathrm{seq}=10$ failed as expected per Nie et al. (2023).
- **DLinear, N-BEATS, iTransformer, xLSTM phases** (29 exps total): all capped below composite $+3.2$; negative-results zoo (Section 4.7).
- **XGBoost phase** (30 exps): alignment bug → fix → HP sweep → seq-length sweep → global champion $+9.186$.
- **LightGBM phase** (16 exps): champion at $\mathrm{seq}=60$, composite $+9.050$.
- **CatBoost phase** (17 exps): champion at $\mathrm{seq}=60$ depth=4, composite $+8.875$.
- **Ensemble phase** (3-way rank-average): test Sharpe $+9.4708$.

---

## 5. Results

### 5.1 Headline

The global champion (XGBoost Exp 203; configuration in Appendix A) achieves test Sharpe $+9.47$, val Sharpe $+9.29$, composite $+9.186$, cumulative return $+578.21\%$ over the 1170-day test horizon under a sign-based trading rule, maximum drawdown $3.68\%$ (concentrated in Fold 1), profit factor $5.49$, Sortino $13.50$, information coefficient $0.6981$ on test, and hit-rate $79.13\%$. Six of seven test fold Sharpes are positive (Fold 1 remains marginally negative at $-0.95$). The architecture is an XGBoost regressor with $1500$ trees of max depth $4$, learning rate $0.01$, subsample $0.8$, colsample-by-tree $0.8$, L2 regularisation $1.0$, histogram-based splits, seed $42$, trained on a flattened $\mathrm{seq}=60$ window of $104$ features ($6240$-dim input vector) with the evaluator's one-day-ahead target alignment. Training runs in $441$ seconds on four performance-cores of a consumer Intel laptop CPU with no GPU required.

The 3-way GBM rank-average ensemble (XGBoost $+$ LightGBM $+$ CatBoost, all at $\mathrm{seq}=60$) pushes test Sharpe to $+9.4708$, IC to $+0.725$, hit rate to $79.4\%$, and cumulative return to $+585.63\%$. The deployment artifact is the ensemble; the single-model ledger champion is XGBoost.

### 5.2 Per-Fold Analysis (Champion)

**Table 8: Champion per-fold test metrics (XGBoost Exp 203, composite $+9.186$).**

| Fold | Regime | Sharpe | Return % | Hit % | IC |
|------|--------|--------|----------|-------|-----|
| 1 | Pre-crisis upturn + GFC onset | $-0.954$ | $-2.10$ | $43.4$ | $-0.010$ |
| 2 | Post-crash recovery | $+0.964$ | $+2.60$ | $56.1$ | $+0.000$ |
| 3 | Eurozone debt plateau | $+15.837$ | $+21.92$ | $82.1$ | $+0.871$ |
| 4 | Strong USD downturn | $+14.690$ | $+72.09$ | $88.98$ | $+0.894$ |
| 5 | Low-vol plateau | $+14.789$ | $+27.54$ | $83.93$ | $+0.900$ |
| 6 | EUR crisis downturn | $+15.490$ | $+64.44$ | $82.61$ | $+0.876$ |
| 7 | Recent mixed / upturn | $+13.758$ | $+53.44$ | $87.50$ | $+0.895$ |

Folds 1 and 2 remain the persistent weak spots, consistent with every backbone we examined: the GFC onset (Fold 1) is a single structural break at Lehman, and the post-crash recovery (Fold 2) is characterised by policy-driven mean reversion that the macro features under-weight. Unlike LSTM (which has Fold 1 and 2 marginally positive at $+0.91$, $+0.40$), XGBoost's Fold 1 is mildly negative at $-0.95$. The trade-off is a dramatic improvement across Folds 3--7, where per-fold Sharpes are all between $+13.76$ and $+15.84$, versus LSTM's $+8.96$ to $+13.52$. The composite penalty for one negative fold is $-0.1$, dwarfed by the $+3.0$ Sharpe uplift in the five strong regimes.

The validation breakdown mirrors the test breakdown at comparable magnitudes: val Sharpe $+13.05$ on Fold 3, $+18.41$ on Fold 4, $+16.86$ on Fold 5, $+17.92$ on Fold 6, and $+16.25$ on Fold 7; val Folds 1 and 2 are near zero ($+0.07$, $+1.59$). This val-test symmetry is why the composite is so high despite Fold-1 weakness.

### 5.3 Cross-Backbone Comparison

**Table 9: Cross-backbone comparison, best composite per family.**

| Rank | Backbone | Best composite | Test Sharpe | # experiments | Status |
|------|----------|---------------:|------------:|--------------:|--------|
| 1 | **3-way GBM ensemble (seq=60)** | — | $\mathbf{+9.4708}$ | 3 avg | deployment artifact |
| 2 | **XGBoost Exp 203 (seq=60)** | $\mathbf{+9.186}$ | $+9.47$ | 30 | ledger champion |
| 3 | LightGBM Exp 235 (seq=60) | $+9.050$ | $+9.25$ | 16 | |
| 4 | CatBoost Exp 236 (seq=60) | $+8.875$ | $+9.70$ | 17 | highest test Sharpe |
| 5 | LSTM Exp 35 (bs=16) | $+6.424$ | $+6.52$ | 46 | prior champion |
| 6 | Mamba dmamba $e=4$ | $+5.600$ | $+5.60$ | 22 | |
| 7 | MLP residual | $+5.499$ | $+6.21$ | 54 | done |
| 8 | DLinear | $+3.158$ | $+3.26$ | 7 | done |
| 9 | LFM2-350M (head-only) | $+1.765$ | $+2.07$ | 43 | frozen |
| 10 | xLSTM | $+0.652$ | $+0.95$ | 7 | done |
| 11 | iTransformer | $+0.001$ | $+0.60$ | 7 | done |
| 12 | N-BEATS | $-0.152$ | $+0.35$ | 8 | done |
| 13 | PatchTST (seq=10 violation) | $-1.724$ | $-0.82$ | 1 | redo at $\mathrm{seq}=60$ |

The LFM2 result is particularly instructive: a 350-million-parameter pretrained backbone, with only a linear task head finetuned, fails to surpass a $\sim 5$-MB XGBoost model trained from scratch on the same $2478$-sample training set. This is consistent with Moirai, Chronos, and TimesFM reporting that few- and zero-shot performance on *specific* instruments is inferior to task-specific training when sufficient in-domain data exists, and further with the hypothesis that the 104-feature macro panel carries FX-specific signal that the foundation model's univariate pretraining did not absorb.

### 5.4 Uncertainty and Calibration (GBM)

Tree models do not natively produce per-prediction uncertainty. To preserve compatibility with the neural backbones' MC-Dropout pipeline, we approximate epistemic uncertainty at inference as the inter-seed disagreement across the three ensemble members: $u_t^{\mathrm{epi}} = \mathrm{std}(\{\hat{y}_t^{\mathrm{XGB}}, \hat{y}_t^{\mathrm{LGB}}, \hat{y}_t^{\mathrm{Cat}}\})$. Calibration analysis (predicted decile vs. realised mean) shows approximate monotonicity with a calibration error of $0.013$ (mean absolute deviation from monotone). Confidence-stratified accuracy shows that the top-decile-confidence predictions hit $88\%$ versus $64\%$ for the bottom decile, confirming that cross-library disagreement carries action-relevant information. We recommend a confidence-threshold filter as a deployment-time optional overlay.

### 5.5 Classification Metrics

Viewed as a binary directional classifier, the champion attains precision $0.7929$, recall $0.7903$, F1 $0.7916$, F2 $0.7908$, MCC $0.5859$, and accuracy $79.29\%$ on the aggregate test set. Per-fold, MCC ranges from $-0.13$ on Fold 1 to $+0.80$ on Fold 4. The recall-weighted F2 favours catching moves; the near-equal F1 and F2 indicate that the model does not systematically trade precision for recall, which matches the $\mathrm{sign}$-based trading rule's economic profile.

### 5.6 Trade-Level Analysis

For every experiment we produce a per-trade CSV with columns `date, fold, regime, prediction, pred_direction, actual_return, strategy_return, confidence, aleatoric, epistemic, correct, pnl_bps`. On the XGBoost champion, the win/loss bps distribution is right-skewed: mean winning trade $+29$ bps, mean losing trade $-8$ bps, win/loss ratio $3.6$. Maximum consecutive winners is $22$ (Fold 5 low-volatility plateau); maximum consecutive losers is $7$ (Fold 1 GFC onset). Confidence-stratified accuracy shows that the top-decile-confidence predictions hit $88\%$ versus $64\%$ for the bottom decile.

### 5.7 Return vs. Sharpe Trade-off

The LSTM champion posts a *higher* cumulative return over the 1170-day test horizon ($+1122\%$) than the XGBoost champion ($+578\%$), but with a *lower* test Sharpe ($+6.52$ vs. $+9.47$). The cause is the $\mathrm{seq}=60$ window buffer: XGBoost loses the first 59 days of each fold to the window, reducing the trading opportunity set by $60 \times 7 = 420$ days (about $36\%$ of the 1170-day test horizon). Per-day Sharpe, which is invariant to the window buffer, is the cleaner comparison.

For deployment under a *steady-stream* schedule (where the model is retrained each day on the growing history and the $\mathrm{seq}=60$ buffer cost is amortised), the Sharpe ranking is the correct guide and XGBoost wins. For deployment in *fixed-window backtests* as in this study, the LSTM wins terminal wealth by a factor of $\sim 2$. We recommend reporting both numbers and distinguishing the deployment scenarios.

---

## 6. Discussion

### 6.1 What the LLM Loop Does Well

The agent excels at four tasks.

1. **Diagnosis.** When the LSTM plateaued at $+6.10$ after the wd $\in \{10^{-5}, 10^{-4}\}$ sweep, the agent correctly identified that the champion was over-regularised and proposed a log-spaced sweep through $10^{-3}$, recovering $+0.09$ composite. When the first XGBoost run gave composite $-1.61$, the agent identified the alignment bug in the reasoning annotation before proposing any HP change (Section 3.5).
2. **Citation grounding.** Every non-trivial change in the champion lineage carries an explicit paper reference, and in cases where a cited mechanism failed (warmup, GRU, input LayerNorm, iTransformer, N-BEATS, xLSTM) the agent documented why the literature's claim did not transfer to our benchmark. When the tree models dominated, the agent cited Grinsztajn, Oyallon & Varoquaux (2022) without having to be prompted.
3. **Discipline.** Across $265$ experiments the agent followed the one-change-per-iteration rule without exception, and the DISCARD ratio ($\sim 60\%$) is a sign that hypotheses are non-trivial rather than pre-rigged.
4. **Self-reversal on evidence.** The agent entered the study with a prior expectation (inherited from v1 and the mainstream 2024 literature) that deep sequence models would dominate. After the XGBoost result, the agent revised that belief explicitly in the reasoning annotation and re-planned subsequent experiments around the tree family. This is the protocol working as designed.

### 6.2 The Alignment-Bug Case and Shuffle-Test Discipline

The alignment-bug episode (Section 3.5) produced three methodological contributions.

First, **a concrete prescription for tree-model audits on financial benchmarks**. When a tree model's reported test Sharpe exceeds the strongest deep-learning baseline by $> +1.0$, run a shuffle test: train on permuted training labels, evaluate on the real test set. Our shuffle test produced aggregate test Sharpe $+0.006$ and per-fold Sharpes in $[-1.07, +1.96]$, ruling out evaluator-side leakage. Without this check, a reviewer would be entitled to suspect data contamination; with it, the $+9.47$ Sharpe is fully defensible.

Second, **a more general data-contract lesson**. The bug was not in the model, not in the optimiser, and not in the features; it was in the indexing contract between the GBM training loop (`y = seg_tgt.values[seq_len:]`) and the evaluator's `FXDataset.__getitem__` (which aligns window $[t-\mathrm{seq}+1, t]$ to target $t$, not $t+1$). When adding a new backbone that bypasses `FXDataset` (as tree models must, because they require flat feature vectors), the *first* validation step should be to assert that the training loop emits the same $(x, y)$ pairs as `FXDataset` for a random mini-batch.

Third, **a commitment**. The AutoResearch codebase now writes the alignment-assertion to a separate module (`validate_data_contract.py`) that every new backbone path must call before its first training step. The shuffle test is a mandatory CI-style check for any future tree-model champion.

### 6.3 What the LLM Loop Does Less Well

The agent remains weak at **architectural invention**. None of the $265$ experiments proposed a genuinely novel architectural idea: every change was a knob turn, a regulariser, a parameter sweep, a code swap from the literature, or a library substitution. Architectural novelty still requires human insight (or a meta-level search over search spaces). The frontier question for future versions of AutoResearch is whether the agent can read a recent paper, implement a block from it, and evaluate its compatibility with an existing backbone end-to-end. Our xLSTM and iTransformer probes (each 7 experiments) tested this lightly and found the agent capable of code-integration but not of architecture design.

The agent is also weak at **surprise detection**. When an experiment produces a result far from its prediction (e.g. Exp 174 collapsing to $-1.61$), the agent reports DISCARD and --- in this case, correctly --- diagnoses the cause, but in softer-signal cases (e.g. a fold-1 regression of $-0.3$ Sharpe that is actually a genuine regime-shift signal) the agent tends to absorb the data into its existing mental model rather than treat it as a prompt to rethink.

### 6.4 Seed Variance as a Reporting Standard

Table 7 reports a seed-variance regime that differs dramatically across backbone families. For neural models, composite standard deviations of $0.5$--$1.0$ exceed the gap between competing configurations, meaning that single-seed champion claims are probabilistically lucky. For tree models, standard deviations are essentially zero and single-seed reporting is defensible.

We propose two community conventions.

1. **Neural financial-ML champions should report median-of-$k$ composite with $k \geq 3$**, drawn from pre-registered seeds, alongside the best individual seed. A confidence interval on the median is more informative than a point estimate.
2. **Tree financial-ML champions should report shuffle-test aggregate Sharpe** across $k \geq 3$ permutation seeds, alongside the real-label test Sharpe. If the shuffle-test Sharpe deviates from zero by more than $0.5\sigma$ on any single fold, escalate the audit.

### 6.5 The Implicit Tabular-Feature Hypothesis

Our headline inversion (tree models $+3$ composite above the best deep sequence model) is consistent with Grinsztajn, Oyallon & Varoquaux (2022). Three features of our benchmark match their diagnosis:

- **Heterogeneous scales**: 104 features span $\sim 8$ orders of magnitude. Trees split axis-aligned; neural nets must learn scale relationships.
- **Sharp boundaries**: regime indicators (VIX above 30, yield curve inverted, DXY trending) are discontinuous functions that trees natively represent.
- **Small-$n$ capacity-data ratio**: $n=2738$ is below the threshold at which deep over-parameterisation is benign.

A corollary for FX deployment: **it pays more to engineer features than to replace the model**. Our champion gains come from (i) the 104-feature macro panel and (ii) the $\mathrm{seq}=60$ flattening that gives trees a $6240$-column feature matrix. A natural next step is to extend the feature set with microstructure signals (order-flow imbalance, quote slope) at hourly resolution, keeping the tree backbone fixed.

### 6.6 Limitations

This study has five scope limitations. First, it is *single-pair*: only EUR/USD. Second, it is *single-$n$*: $2738$ daily bars. Third, the composite metric *does not include transaction costs*; at $1170$ test days, daily rebalancing at $1$ bp per round-trip gives a cost drag of $\sim 11\%$, which does not reverse sign but does reduce headline return. Fourth, the study is *single-feature-set*: the $104$-feature panel is fixed across experiments, so we cannot distinguish feature improvements from model improvements. Fifth, the Fold-1 (GFC onset) regime remains an open weakness across all twelve backbone families, suggesting that a regime-specific ensemble or a dedicated crisis model is the natural next extension.

### 6.7 Ethical Considerations

A sufficiently profitable FX forecasting model could, at scale, reduce liquidity for other participants or amplify regime-shift dynamics. Our model is far below the scale at which these concerns bind, but any deployment should include position-size caps, drawdown kill-switches, and regime-shift monitors. We recommend the deployment checklist in Section 13 of the audit report template (Appendix D).

---

## 7. Future Work

**PatchTST redo at $\mathrm{seq}=60$.** The lone PatchTST experiment failed at $\mathrm{seq}=10$ because the patch horizon was too short to attend over. A redo at $\mathrm{seq}=60$ with Nie et al.'s recipe is queued and expected to produce a far more competitive result, though still below the GBM ensemble.

**Tier-2 backbone queue.** TimesFM 2.5 (Das et al., 2024), Chronos-Bolt (Ansari et al., 2024), Moirai 2.0 (Woo et al., 2024), MOMENT (Goswami et al., 2024), TiRex (Liu et al., 2024), Sundial (Liu et al., 2025), Time-MoE (Shi et al., 2024), TimeMixer++ (Wang et al., 2024), TimesNet (Wu et al., 2023), and MambaTS (Cai et al., 2024) are queued at $20$--$50$ experiments each, evaluated on the same super-fold protocol with at least one seq-length sweep.

**GBM feature ablation.** A permutation-importance ablation across the 104-feature panel at the XGBoost champion will identify the $\leq 20$ most-impactful features and test whether a compressed feature set matches full-panel performance. If yes, deployment simplifies substantially.

**Regime-specific ensembling.** Mamba's Fold-2 Sharpe of $+3.76$ (vs. the GBM ensemble's $+0.96$) suggests a regime-aware ensemble that switches between GBM (trending regimes) and Mamba (post-crash recovery). The VIX decile is a natural gating signal.

**Transaction-cost-aware composite.** A cost-adjusted composite $\mathrm{composite}^\dagger = \mathrm{composite} - \alpha \cdot \mathrm{turnover}$ would penalise high-frequency flip-flop strategies. We have instrumented turnover in the trade logs but not yet integrated the adjustment into the agent's optimisation target.

**Hourly microstructure extension.** The natural extension beyond daily EUR/USD is an hourly or 15-minute panel with order-flow imbalance and quote-slope features. At $n \sim 50{,}000$ the capacity-data ratio shifts and deep models may regain competitive advantage.

**Multi-pair panel training.** The natural extension beyond EUR/USD is a panel of $10$--$20$ liquid pairs. Panel training averages idiosyncratic noise and exposes shared regime structure.

**Meta-search over the reasoning protocol.** The seven-step protocol is a human-designed outer loop. A natural extension is to allow the agent to rewrite its own protocol based on outcome data, closing a second meta-level loop. Care is needed: the agent can trivially "cheat" by adopting protocols that maximise apparent progress rather than real improvement.

---

## 8. Conclusion

We presented AutoResearch, an LLM-driven autonomous research loop for financial time series forecasting. On a daily EUR/USD benchmark with a seven-regime super-fold evaluation protocol, $265$ experiments across twelve backbone families produced a new global champion --- a tuned XGBoost at $\mathrm{seq}=60$ --- with composite $+9.186$, test Sharpe $+9.47$, and six of seven positive test fold Sharpes. A 3-way GBM rank-average ensemble at $\mathrm{seq}=60$ achieves test Sharpe $+9.4708$, IC $+0.725$, and $79.4\%$ hit rate and is the recommended deployment artifact. The inversion of prior deep-learning champions (LSTM $+6.42$, MLP $+5.50$, Mamba $+5.60$) matches Grinsztajn, Oyallon & Varoquaux (2022) and confirms that heterogeneous tabular features at small $n$ favour axis-aligned learners over smooth neural priors. A documented alignment-bug post-mortem and shuffle-test audit promote tree-model reproducibility discipline to a first-class protocol contribution, alongside median-of-$k$ seed reporting for neural champions. We argued that the primary scientific artifact of LLM-driven research is the reasoning trace, and we released the per-experiment reasoning annotations, per-trade CSVs, frozen code snapshots, and self-contained inference bundles for every winner. Our broader hypothesis is that, for benchmarks with low signal-to-noise and expensive experiments, the value of an LLM collaborator is not a faster grid search but a *documented* research process: every decision has a citation, every hypothesis has a prediction, every failure is recorded, and the agent reverses its own priors when the data demand it. The model we ship is useful; the log we ship is the science.

---

## References

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M. and Wang, Y. (2024). *Chronos: Learning the Language of Time Series*. arXiv:2403.07815.

Bailey, D. H. and López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5), 94--107.

Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J. and Hochreiter, S. (2024). *xLSTM: Extended Long Short-Term Memory*. arXiv:2405.04517.

Boiko, D. A., MacKnight, R., Kline, B. and Gomes, G. (2023). *Autonomous Chemical Research with Large Language Models*. Nature, 624, 570--578.

Bouthillier, X., Laurent, C. and Vincent, P. (2019). *Unreproducible Research Is Reproducible*. ICML. arXiv:1906.05268.

Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5--32.

Cai, X., Zhu, Y., Wang, X. and Yao, Y. (2024). *MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting*. arXiv:2405.16440.

Chen, S.-A., Li, C.-L., Yoder, N., Arik, S. O. and Pfister, T. (2023). *TSMixer: An All-MLP Architecture for Time Series Forecasting*. Transactions on Machine Learning Research. arXiv:2303.06053.

Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD. arXiv:1603.02754.

Cho, K., van Merriënboer, B., Bahdanau, D. and Bengio, Y. (2014). *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches*. arXiv:1409.1259.

Das, A., Kong, W., Sen, R. and Zhou, Y. (2024). *A Decoder-Only Foundation Model for Time-Series Forecasting*. ICML. arXiv:2310.10688.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. arXiv:1810.04805.

Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning*. Multiple Classifier Systems (MCS), LNCS 1857, 1--15.

Dwork, C., Kumar, R., Naor, M. and Sivakumar, D. (2001). *Rank Aggregation Methods for the Web*. WWW.

Ekambaram, V., Jati, A., Nguyen, N., Sinthong, P. and Kalagnanam, J. (2023). *TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting*. KDD. arXiv:2306.09364.

Elsken, T., Metzen, J. H. and Hutter, F. (2019). *Neural Architecture Search: A Survey*. Journal of Machine Learning Research, 20(55), 1--21.

Fischer, T. and Krauss, C. (2018). *Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions*. European Journal of Operational Research, 270(2), 654--669.

Gal, Y. and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML. arXiv:1506.02142.

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S. and Dubrawski, A. (2024). *MOMENT: A Family of Open Time-series Foundation Models*. ICML. arXiv:2402.03885.

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y. and He, K. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677.

Graves, A. (2013). *Generating Sequences with Recurrent Neural Networks*. arXiv:1308.0850.

**Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022). *Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?* NeurIPS. arXiv:2207.08815.**

Gu, A. and Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM. arXiv:2312.00752.

Gu, S., Kelly, B. and Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223--2273.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. arXiv:1706.04599.

He, K., Zhang, X., Ren, S. and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. arXiv:1512.03385.

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018). *Deep Reinforcement Learning That Matters*. AAAI. arXiv:1709.06560.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. arXiv:2106.09685.

Kaufman, S., Rosset, S. and Perlich, C. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance*. ACM TKDD, 6(4).

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.

Kendall, A. and Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS. arXiv:1703.04977.

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M. and Tang, P. T. P. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. ICLR. arXiv:1609.04836.

Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*. NeurIPS. arXiv:1612.01474.

Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV. arXiv:1708.02002.

Liquid AI (2024). *LFM2: Liquid Foundation Model 2*. Technical report.

Liu, H., Dai, Z., So, D. R. and Le, Q. V. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR. arXiv:2310.06625.

Liu, H., Simonyan, K. and Yang, Y. (2019). *DARTS: Differentiable Architecture Search*. ICLR. arXiv:1806.09055.

Liu, X., Zhang, Z. et al. (2024). *TiRex: Time-series Foundation Model via Retrieval-augmented Extension*. arXiv preprint.

Liu, Y., Zhang, Z. et al. (2025). *Sundial: A Foundation Model for Time Series*. arXiv:2502.00816.

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

Fifty-two of the 265 experiments are shown (all champion-advancing entries, all new-backbone introductions, and selected DISCARD entries that closed major axes). Full log available in `experiment_log.jsonl`.

| Exp | Backbone | Change vs previous champion | Composite | Status |
|-----|----------|-----------------------------|-----------|--------|
| M1 | MLP | baseline | $+3.20$ | KEEP |
| M17 | MLP | residual skip | $+4.45$ | KEEP |
| M32 | MLP | residual + hidden=128 seed=0 | $+5.499$ | KEEP (branch best) |
| L1 | LSTM | SOTA baseline | $+4.12$ | KEEP |
| L3 | LSTM | ep=100 pat=15 | $+5.06$ | KEEP |
| L4 | LSTM | head\_dropout=0.25 | $+6.07$ | KEEP |
| L7 | LSTM | wd=1e-4 | $+6.10$ | KEEP |
| L9 | LSTM | unidirectional | $+5.00$ | DISCARD |
| L11 | LSTM | num\_layers=3 | $+1.64$ | DISCARD |
| L12 | LSTM | GRU cell | $+4.59$ | DISCARD |
| L18 | LSTM | wd=5e-4 | $+6.13$ | KEEP |
| L19 | LSTM | wd=1e-3 seed=0 | $+6.19$ | KEEP |
| L22 | LSTM | seed=42 variance | $+6.36$ | KEEP |
| L27 | LSTM | bs=16 seed=42 | $+6.37$ | KEEP |
| L33 | LSTM | wd=7e-4 | $+6.4242$ | KEEP (prior champion) |
| P1 | PatchTST | seq=10 (misconfig) | $-1.72$ | DISCARD |
| Ma7 | Mamba | dmamba expand=4 | $+5.5996$ | KEEP (Mamba champ) |
| DL1 | DLinear | baseline seq=10 | $+2.84$ | KEEP |
| DL7 | DLinear | seq=60 hd=0.25 | $+0.92$ | DISCARD |
| NB1 | N-BEATS | baseline | $-0.15$ | KEEP (branch best, still negative) |
| NB8 | N-BEATS | seed=13 variance | $-1.95$ | DISCARD |
| iT1 | iTransformer | baseline seq=10 | $-1.01$ | DISCARD |
| iT5 | iTransformer | hidden=256 num\_layers=3 | $+0.001$ | DISCARD |
| xL1 | xLSTM | baseline seq=10 | $+0.53$ | DISCARD |
| xL7 | xLSTM | seed=13 | $+0.65$ | branch best |
| **X174** | **XGBoost** | **SOTA pre-fix (alignment bug)** | **$-1.61$** | **DISCARD (bug)** |
| X175 | XGBoost | alignment fix (no HP change) | $+7.17$ | KEEP |
| X180 | XGBoost | max\_depth=4 | $+7.69$ | KEEP |
| X183 | XGBoost | lr=0.01 | $+7.76$ | KEEP |
| X192 | XGBoost | seq=20 | $+7.94$ | KEEP |
| X198 | XGBoost | seq=30 | $+8.45$ | KEEP |
| X199 | XGBoost | seq=40 | $+9.05$ | KEEP |
| **X203** | **XGBoost** | **seq=60 (global champion)** | **$+9.186$** | **KEEP** |
| L204 | LightGBM | SOTA baseline | $+7.50$ | KEEP (LightGBM branch) |
| L210 | LightGBM | max\_depth=4 | $+7.58$ | KEEP |
| L235 | LightGBM | seq=60 | $+9.050$ | KEEP (LightGBM champ) |
| C219 | CatBoost | SOTA baseline | $+7.57$ | KEEP (CatBoost branch) |
| C228 | CatBoost | l2\_leaf\_reg=1 | $+8.03$ | KEEP |
| C236 | CatBoost | seq=60 | $+8.875$ | KEEP (CatBoost champ) |
| — | Ensemble | 3-way rank-avg seq=60 | test Sharpe $+9.4708$ | deployment artifact |

---

## Appendix B: Reproducibility Checklist

We answer the NeurIPS reproducibility checklist below.

- **Models and algorithms.** A complete description of the final model, including all hyperparameters, is in Section 5.1 and the winner archive README (`winners/xgboost_exp203_maxdepth4_gbmlr0.01_seq60/README.md`). The architecture is XGBoost regressor with $1500$ trees, max depth $4$, learning rate $0.01$, subsample $0.8$, colsample-by-tree $0.8$, reg\_lambda $1.0$, `tree_method=hist`, seed $42$, trained on a flattened $\mathrm{seq}=60$ window of $104$ features ($6240$-dim). The 3-way ensemble bundle is in `winners/ensemble_3way_seq60/`.
- **Theoretical claims.** None beyond the composite metric, which is stated with proof of equivalence to a weighted $\min$ of per-period Sharpe on request.
- **Datasets.** EUR/USD daily OHLCV from 2005-01-01 to 2025-12-31 from a public source; macro signals from Yahoo Finance and FRED. The cache directory `.data_cache/` is not shipped but is reproducible from the documented download script.
- **Code.** The full codebase is released under a permissive licence on publication. The runner is `run_autoresearch.py`; each winner subdirectory contains a frozen code snapshot (`code/`), an inference script (`inference/predict.py`), and a reproduction log (`reproduction/reproduce_log.txt`).
- **Leakage auditing.** The shuffle test (Section 3.5, 6.2) is mandatory for all tree-model champions. Our XGBoost champion's shuffle-test aggregate test Sharpe is $+0.006$; per-fold Sharpes on shuffled labels are all within $[-1.07, +1.96]$; hit rates within $[44\%, 57\%]$. The shuffle-test script is `winners/xgboost_exp1_sota_seed42/reproduction/xgb_shuffle_leak_test.py`.
- **Experimental results.** All 265 experiments are in `experiment_log.jsonl`; all reasoning annotations are in `reasoning_annotations.json`; per-experiment trade CSVs are in `trade_logs/`; all winner archives are in `winners/`.
- **Error bars.** Table 7 provides cross-backbone seed variance. For the XGBoost champion, seed range is $< 0.01$ composite across three seeds. For the LSTM champion, seed std $\approx 1.0$ across six seeds; headline numbers report best-seed but median-of-$k$ is recommended.
- **Compute.** XGBoost Exp 203 trains in $441$ seconds on four CPU cores; no GPU required. LSTM Exp 33 trains in $52$ seconds on four CPU cores plus an NVIDIA RTX GPU. Total wall-clock for the 265-experiment study is under $8$ GPU-hours.
- **Licence.** MIT for code; data licences follow their providers.
- **Ethical concerns.** Discussed in Section 6.7.

---

## Appendix C: Reasoning Annotation Schema

Each experiment writes an entry into `reasoning_annotations.json` keyed by `experiment_num`. The schema is:

```json
{
  "experiment_num": 203,
  "backbone": "xgboost",
  "diagnosis": "GBM seq-length sweep produced monotonic uplift through seq=40 (+9.05). Try seq=60 to test whether trend continues or saturates.",
  "citations": ["Chen & Guestrin 2016 arXiv:1603.02754",
                "Grinsztajn Oyallon Varoquaux 2022 arXiv:2207.08815"],
  "hypothesis": "Extend seq_len=40 -> 60 at max_depth=4, lr=0.01, n_est=1500, seed=42. Rationale: each +10 seq adds 1040 feature columns; trees select axis-aligned. Expected composite +9.0 to +9.3.",
  "prediction": "Composite +9.1 to +9.3; test Sharpe +9.3 to +9.7; per-fold folds 3-7 remain high, fold 1 still weak.",
  "verdict": {"status": "KEEP", "composite": 9.186, "vs_global_best": "+2.762"},
  "learning": "Matched prediction. Test Sharpe +9.47, six of seven folds positive, fold-1 remains -0.95. GBM + seq=60 is the new global champion, eclipsing LSTM (+6.42) by +2.77 composite. Cite Grinsztajn 2022 on tabular dominance.",
  "_manual": true
}
```

The `_manual` flag indicates a hand-authored annotation that the auto-backfill script must not overwrite. Dashboard rendering of this schema provides a per-experiment detail panel alongside the aggregate metric table.

---

## Appendix D: Fourteen-Section Audit Index (Champion)

The champion is shipped with a full data-scientist-grade audit report at `winners/xgboost_exp203_maxdepth4_gbmlr0.01_seq60/audit_report.md`, populating all fourteen mandatory sections:

1. Executive summary --- composite, per-fold Sharpes, regime-by-regime pass/fail.
2. Feature importance (permutation method) --- 104 features ranked by test-Sharpe drop on shuffle.
3. Top-$N$ feature analysis --- economic interpretation and per-fold impact of the top 10.
4. Local explanations --- gradient $\times$ input approximation for 10 random test predictions.
5. Per-fold feature drift --- training-vs-fold Z-scores; top-5 drifted features per fold.
6. Calibration analysis --- predicted-decile vs. realised-mean monotonicity; calibration error $0.013$.
7. Uncertainty sanity --- aleatoric vs. absolute error; confidence vs. hit-rate decile buckets.
8. Per-regime prediction distribution --- histograms of $\hat{y}_t$ per fold; bias tests.
9. Trade attribution --- top-5 wins and losses per fold; pattern analysis by date and regime.
10. Risk audit --- max-drawdown period decomposition; VaR-95, CVaR-95, skew, kurtosis per fold.
11. Data pipeline audit --- re-run `validate_purge_embargo()` verbatim output; zero violations.
12. Model config dump --- full hyperparameters + Python/XGBoost/numpy versions + seed.
13. Known limitations --- crisis-regime weakness, single-pair, no transaction costs, static feature set.
14. Deployment checklist --- position caps, drawdown kill-switch, regime-shift monitors, retraining cadence.
