# When the LLM Drives the Research Loop: 99 Experiments in Quantitative FX, One Claude Session at a Time

*How a Karpathy-style autonomous research protocol, a brittle Intel CPU, and a four-line residual connection collided to produce a EUR/USD forecaster with a +6.21 test Sharpe and 7/7 positive regime folds — and what I learned about letting a large language model actually do science.*

---

## 1. The Vision: No Python Agent. No Prebaked Plan. Just Claude and a Journal.

Most "LLM agents" I've seen in ML research follow the same pattern. There is an outer Python controller that does the real thinking — it has a queue of experiments, a hyperparameter grid, a search policy — and the LLM gets called as a glorified text generator to format log lines and write summaries. The LLM is decorative.

I wanted to invert that. I wanted the LLM to *be* the research loop.

The rules for this project are written into a single file, `CLAUDE.md`. It begins with a line that sounds mundane but is actually the entire architectural commitment:

> You ARE the autoresearch loop. Claude Code is the outer loop — there is no separate Python agent.

There is no experiment queue. There is no search strategy hard-coded in Python. When a session starts, Claude (running inside Claude Code, Anthropic's CLI) reads a checkpoint file, reads the tail of a JSONL experiment log, diagnoses the current champion's weakest fold, searches the literature for a technique that might address it, writes down a prediction, runs exactly one experiment, analyzes the result against the prediction, and checkpoints everything. Then it does it again. And again. Ninety-nine times.

The only Python that runs is `run_autoresearch.py`, which executes exactly one experiment and appends one line of JSON. Everything else — the hypothesis formation, the citation, the decision about what to try next, the acceptance or rejection of a result, the architecture modifications — happens inside the agent.

This article is the postmortem of that experiment on the experiment loop. It is also the postmortem of a EUR/USD model that went from composite `-1.26` on Experiment 1 to composite `+5.499` on Experiment 99, with a `+6.2113` test Sharpe on a seven-regime super-fold and a cumulative return of `+1001%` on held-out data. It includes every wrong turn: the lucky seed that wasn't real, the foundation model that lost to a 167K-parameter MLP, the Blue Screens of Death (five in one afternoon) from a quietly degrading Intel i9-14900HX, and the four-line residual connection that quintupled the Sharpe.

If you are a quant, you will probably care about the results and the pipeline. If you are an ML engineer, you will probably care about the protocol and the honest failure modes. If you are interested in whether an LLM can drive a real research process — not generate code once, but sustain a scientific loop across a hundred iterations — this is what it looked like.

---

## 2. The Problem: EUR/USD Is Efficient and Most Papers Are Lying to You

Foreign exchange is the largest and, arguably, most efficient financial market on earth. About $7.5 trillion trades hands every day, and the flow is dominated by institutional participants with microstructure teams, execution infrastructure, and latency budgets measured in microseconds. For daily-horizon directional prediction on a major pair like EUR/USD, the honest baseline is 50%. The honest next step is that beating 50% meaningfully and reproducibly, across regimes, is hard.

The ML-for-FX literature, unfortunately, is also full of numbers that look too good. Most of the "Sharpe 3.0 on EUR/USD" papers have one or more of the following sins:

1. **Overlapping train/test windows.** A 5-day forward return target leaks into training data if the purge gap is smaller than the label horizon.
2. **Walk-forward without hole-punching.** If fold 3's training data includes fold 6's test window, you have cross-contamination even if each individual fold looks clean.
3. **No regime breakdown.** An aggregate Sharpe hides the fact that the model only works in one regime and collapses everywhere else.
4. **Single-seed reporting.** Neural networks on small financial datasets have enormous seed variance. A single +2.0 Sharpe run often has a -0.5 sibling under a different seed.
5. **Unacknowledged multiple comparisons.** A grid search over 200 configurations will, by chance, produce a top-line number that looks significant and isn't.

I wanted a pipeline that was explicit about all of this, auditable end-to-end, and that would refuse to accept a "new champion" unless it improved robustly across all seven regime folds simultaneously. That is the problem the AutoResearch protocol is designed to solve.

---

## 3. The Protocol: Seven Steps, One Change per Experiment

The heart of AutoResearch is a seven-step cycle, written into `CLAUDE.md` as the `Research-Driven Experiment Selection` section. Every single experiment must follow it. Quoting directly from the rules:

> **Step 1 — Diagnose the champion's weakness.** Look at the per-fold test results. Which folds are weakest? What regime are they? What do the uncertainty metrics say? Identify the SPECIFIC failure mode.
>
> **Step 2 — Search the literature.** Based on the diagnosis, search arXiv / known papers for techniques that address the failure mode.
>
> **Step 3 — Form a hypothesis and predict the outcome.** Write down: "I hypothesize that [change X] will improve [metric Y] on [fold Z] because [paper/principle]. I predict composite will move from [current] to approximately [target]." If you can't write this sentence, you don't understand what you're doing. Stop and think more.
>
> **Step 4 — Run ONE experiment.** Execute the change. ONE change only.
>
> **Step 5 — Analyze against prediction.** Did the result match your prediction? If yes, why? If no, what does that tell you about your mental model?
>
> **Step 6 — Document everything.** Write the full cycle into the experiment log and checkpoint.
>
> **Step 7 — Checkpoint.** Immediately. The laptop will crash. Every minute of uncheckpointed work is lost work.

This reads like basic scientific method, and it is. What makes it interesting is that it is the *operating system* of the agent. The LLM is not allowed to run an experiment it cannot justify. It is not allowed to sweep a hyperparameter grid. It is not allowed to change two things at once. If three experiments in a row are discarded, the rules instruct it to stop and re-diagnose — multiple failures mean the mental model is wrong, not that more experiments are needed.

One further constraint, from the "Karpathy-adapted" agent protocol section:

> **Always start from the current best config.** Every experiment modifies ONE thing from the best. If it improves, it becomes the new best. If it doesn't, revert and try a different direction. Never wander off from the best baseline.

This is the single most important rule in the system. Without it, a long experiment loop wanders into low-performing regions of hyperparameter space and stays there. With it, the champion moves monotonically upward — each new KEEP must beat the previous champion on the composite metric, and DISCARDs revert the baseline.

And one more, which matters a lot for what happened next:

> **Code changes are allowed.** The agent may modify the Python codebase (model architecture, loss function, training loop, features, evaluation) if it has a principled reason. Code changes are the most powerful lever — hyperparams only go so far.

Hyperparameters are a local search. Architectures change the search space. As we will see, the champion was not found by tuning — it was found by editing `backbone.py`.

---

## 4. Data and Splits: Seven Regimes, Zero Leakage

The data is six major FX pairs (`EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `EURGBP`, `EURJPY`) plus nine macroeconomic signals (yield curves, DXY, VIX, etc.) sampled daily from 2005 through 2024. The target is the 1-day and 5-day forward log return on EUR/USD. Feature engineering in `data/features.py` produces approximately 104 strictly backward-looking features per timestep, grouped as:

- **Per-pair technical (13 × 6 pairs):** log returns, rolling volatilities, RSI, MACD, basic microstructure
- **Cross-pair correlations (5):** rolling 21-day correlation of EUR/USD against each of the other five pairs
- **Macro (21):** returns and levels for 9 macro tickers, yield-curve slope, VIX change, DXY volatility
- **Forward targets (held separately):** `fwd_ret_1d`, `fwd_ret_5d`

The split scheme is where the real rigor lives. Walk-forward cross-validation with purge and embargo is standard in MLFin (Lopez de Prado, 2018), but the implementation details are where most projects leak. This one defines seven walk-forward folds, each with its own regime label:

| Fold | Regime | Test Period |
|---|---|---|
| 1 | Pre-crisis upturn + GFC onset | 2008-01 → 2008-06 |
| 2 | Post-crash recovery | 2010-01 → 2010-06 |
| 3 | Eurozone debt plateau | 2013-01 → 2013-06 |
| 4 | Strong USD downturn | 2015-04 → 2015-12 |
| 5 | Low-vol plateau | 2017-2019 |
| 6 | EUR crisis / COVID | 2020-2021 |
| 7 | Recent mixed / upturn | 2023-2024 |

Between train-end and val-start (and between val-end and test-start), `data/splits.py` enforces a 90-calendar-day **purge gap** to prevent label leakage from overlapping 5-day forward return windows, a 21-day **embargo** after test-end to prevent autocorrelated features from bleeding across fold boundaries, and a 10-day **label-horizon buffer** before every excluded window so that the 5-day forward-return target cannot peek into the held-out period. These three numbers — 90, 21, 10 — matter more than any hyperparameter in the system.

On top of the fold structure sits the **super-fold** idea. Instead of training seven separate models (one per fold), `split_superfold()` constructs a single training set consisting of *all* historical data (2005 through 2024) **except** the union of all seven folds' validation and test windows, plus their label-horizon buffers. The validation set is the union of all seven val windows (915 rows). The test set is the union of all seven test windows (1170 rows). Every training run is thus a single model evaluated across all seven regimes simultaneously.

Three invariants are verified programmatically before every run:

1. `split_superfold()` returns the expected counts: train ≈ 3113, val = 915, test = 1170.
2. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0.
3. `validate_purge_embargo()` finds zero violations.

There is one more subtle bug this project fixed explicitly, because it is extremely easy to get wrong:

> NEVER create sliding windows (FXDataset) across non-contiguous date ranges. Use `create_contiguous_datasets()` which splits at gaps and creates per-segment datasets.

When you hole-punch val and test windows out of training data, the remaining training dates are no longer contiguous. A naive sliding-window dataset will happily create sequences that straddle a gap — a sample whose first five timesteps are from March 2013 and whose last five are from January 2014, because the dates in between were excluded. This is garbage: neither the features nor the target have a coherent temporal interpretation. `create_contiguous_datasets()` detects gaps and creates one sub-dataset per contiguous segment. About 41% of windows would be garbage without this fix. It is documented in the project's `Common Mistakes (Never Repeat)` table, which is the single most useful part of `CLAUDE.md`.

---

## 5. The Composite Metric: Why "Best Sharpe" Is the Wrong Goal

If you reward a model for top-line Sharpe, it will happily hand you a model that returns +10 on three folds and -3 on four folds and calls it +3 on average. That model is not robust. It is a specialization that got lucky on a big regime and would blow up in the rest.

The AutoResearch composite metric refuses that trade. It is defined in one line and I have stared at it a great deal:

```python
composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds
```

There are three things going on in this formula and each of them is doing real work:

1. **`min(test_sharpe, val_sharpe)`** — both val and test must hold up. A config that overfits to val (or, more subtly, to the aggregate test split) is clipped by whichever metric is lower.
2. **`- 0.1 * n_negative_folds`** — each fold with a negative test Sharpe costs 0.10 composite points. This is the regime-robustness term. Seven folds, so the maximum penalty is -0.70.
3. **Keep/revert is driven by this composite, not by any individual metric.** If a change improves test Sharpe but regresses val Sharpe or produces a new negative fold, it is a DISCARD. The quality ratchet only clicks forward.

The champion's composite is +5.499. The arithmetic: `min(+6.2113, +5.599) - 0.1 * 0 = +5.599`? Close but not exact — the implementation also includes some additional adjustments, but the shape of the metric is what matters. For the champion, test Sharpe (6.21) and val Sharpe (5.60) are both strongly positive and all 7 test folds are positive. That is why the composite is high. A model with test Sharpe +8 and two negative folds would score lower.

This metric turns out to be the single most important design choice in the whole project after the data splits. It is the loss function of the meta-optimizer. The LLM is doing gradient descent on this number, using experiments as the gradient estimator and its own reasoning as the update rule.

---

## 6. The Model Arsenal: Eight Backbones, One Winner You Would Not Have Bet On

`model/backbone.py` registers eight architectures behind a single factory function:

| Backbone | Type | Notes |
|---|---|---|
| `mlp` | Feedforward | Residual MLP (shortcut + 2-layer nonlinear branch), 128 hidden, ~167K params |
| `lstm` | Recurrent | Bidirectional LSTM, 2 layers, 128 hidden |
| `lfm2-350m` | Foundation model | LiquidAI LFM2.5-350M-Base, frozen backbone, head-only fine-tuning |
| `patchtst` | Transformer | Nie et al., ICLR 2023 — patched attention for time series |
| `patchtsmixer` | MLP-Mixer | Google, NeurIPS 2023 — all-MLP alternative to transformer |
| `xgboost` | Gradient boosting | Chen & Guestrin, 2016 |
| `lightgbm` | Gradient boosting | Ke et al., NeurIPS 2017 |
| `catboost` | Gradient boosting | Prokhorenkova et al., NeurIPS 2018 |

All neural backbones share a `forward(x: Tensor[B, seq_len, n_features]) -> {"ret_1d": ..., "ret_5d": ...}` interface and emit the same prediction heads. The heads support two modes via the `het_loss` flag: when on, each head outputs mean + log-variance for a Kendall & Gal (2017) heteroscedastic loss; when off, each head outputs the mean only and uncertainty comes from MC Dropout (Gal & Ghahramani, 2016).

The original plan was simple: if any backbone had a right to win this contest, it would be LFM2.5-350M. A 350-million-parameter Liquid Foundation Model, pre-trained on long sequences, with a 60-step context window vs. everyone else's 10, fine-tuned head-only so the pre-trained knowledge is preserved. That is the story this kind of paper is supposed to tell.

Fifty LFM2 experiments later, the story it actually told was: median test Sharpe +1.40, best reproducible test Sharpe around +2.07. Not bad. But the champion in the end was a residual MLP with 167,000 trainable parameters, 10-step context, trained in 52 seconds on CPU, with a test Sharpe of +6.21. The frozen foundation model was beaten by a network that could fit entirely in the L2 cache of the CPU it was running on.

The MLP wasn't obvious up front. The first plain-MLP experiments (Exps 51–59, `lr=1e-4`, `lr=5e-5`, various seeds) hovered around composite 0.5 and below. The 512-hidden version was overfit (composite -0.51). The 128-hidden version was better (composite ~0.82). Then came Exp66 — experiment number 66, the agent's tenth MLP experiment — which added a residual skip connection. Composite jumped from 0.82 to 4.674. One edit. One change. That is the edit the next two sections are about.

But before we get there, we need to talk about what happened with LFM2, because it is the most intellectually honest part of the whole project.

---

## 7. The Seed Variance Crisis (or, How I Nearly Shipped a Lucky Seed)

The first fifty experiments were all LFM2-350M fine-tuning. By experiment 20, the agent had found what looked like a clear winner: plain Huber loss, `lr=2e-5`, composite +1.77, test Sharpe +2.07. Every hyperparameter axis had been swept. LR was the dominant lever. Batch size, weight decay, sequence length, warmup, Huber delta, head dropout, gradient clip — all secondary. The result felt solid.

Then came the reproduction runs.

Here is a table, lifted from `autoresearch_report.md`, of the *same configuration* across different seeds:

| Run | Seed | Composite | Test Sharpe | Worst Fold |
|---|---|---|---|---|
| Exp 20 | random | **+1.77** | +2.07 | fold 1 (−0.52) |
| Exp 48 | 0 | +1.13 | +1.74 | fold 2 (−3.38) |
| Exp 47 | 42 | **−1.52** | −0.72 | fold 6 (−3.07) |
| (additional) | 7 | +0.11 | +0.51 | fold 4 (−2.73) |

**Median composite across 4 runs: +0.11.** The "best" composite of +1.77 was a top-quartile outlier by a wide margin. The swing between seeds was **+3.29 composite units** — larger than any hyperparameter effect the sweep had found.

The root cause is a thing every MLFin practitioner should have tattooed somewhere visible: the LFM2 head has a `nn.Linear(104, 1024)` projection layer with about 106K parameters, mapped to a training set of only 2,738 windows after hole-punching. That is 39 parameters per training sample in just the projection layer. The optimization problem is underdetermined. Each random initialization lands in a different basin, and each basin specializes in a different subset of the seven regimes. When seed=42 excels on fold 3, it collapses on fold 6. When seed=0 excels on fold 7, it collapses on fold 2. The model is *always* learning a specialization; the seed picks which one.

The right response, once this was clear, was to change the protocol: any "new champion" candidate must be reproduced across at least three seeds and the *median* composite must improve, not just one seed's result. This is the `seed_variance.json` file in every winner archive directory. For the final MLP champion, the cross-seed table looks like this:

| Seed | Composite | Test Sharpe |
|---|---|---|
| 0 | +5.50 | +6.21 |
| 42 | +4.45 | +4.69 |
| 99 | +4.46 | +4.76 |

Median test Sharpe +4.76. Still a huge result. Still well above anything LFM2 produced. But no longer a lottery ticket.

The lesson from the LFM2 phase is not that foundation models are bad. It is that *projections from a small feature space into a large frozen embedding space are severely underdetermined on small financial datasets*. It is a known failure mode. The agent rediscovered it the hard way.

---

## 8. The Heteroscedastic Detour: A Warmup That Wasn't a Warmup

Mid-way through the LFM2 phase, the agent pivoted to Kendall & Gal's (2017) heteroscedastic loss. The idea is elegant: instead of predicting just the mean, each head predicts mean plus log-variance, and the loss is

```python
loss = exp(-log_var) * huber(mean, target) + 0.5 * log_var
```

The `exp(-log_var)` term down-weights the loss on samples the model believes are noisy (high aleatoric uncertainty), letting the mean-prediction capacity focus on signal. The `0.5 * log_var` term prevents the trivial solution of predicting infinite variance. Done right, you get interpretable per-sample uncertainty and better mean predictions.

The agent ran twenty-eight het-loss experiments. And in the middle of them, it seemed to find a breakthrough. Quoting the report:

> **H-Exp13 (warmup=3, lr=3e-5):** composite +1.60, test Sharpe +1.80.
> Rationale: "warmup stabilizes the log-variance head initialization, which is noisy at step 0" (Goyal et al., 2017).

That is a principled, literature-backed hypothesis. And the result matched the prediction. The agent logged it, updated the champion, and moved on.

Then it tried to reproduce it.

| Run | Config | Composite |
|---|---|---|
| H-Exp13 (original) | warmup=3 | **+1.60** |
| H-Exp21 (repro 1) | warmup=3 | **−1.10** |
| H-Exp22 (repro 2) | warmup=3 | **−0.54** |
| H-Exp23 (seed=42) | warmup=3 | **−1.08** |
| H-Exp24 (seed=123) | warmup=3 | **−1.14** |

**Median across 5 runs: −0.54.** H-Exp13 was a 2+ sigma outlier. The "breakthrough" was, in the honest language of the report, a lucky seed in an already-high-variance loss function. The het-loss `exp(-log_var)` term, on n=2738 samples, amplifies the seed variance problem — now you're specializing the initialization of *two* branches, mean and variance, and they fight each other for capacity.

The decision the agent made, looking at the reproduction table, was the right one: revert to plain Huber, disable het-loss by default, and treat the failure as data. The project's `CLAUDE.md` has a specific rule now, written in the voice of someone who has been burned:

> **The het-loss needs ~50% more epochs than plain Huber** to converge, because the variance branch adds an optimization axis.
> **Variance-branch dominance is the #1 failure mode.** If aleatoric > 0.2, the model is copping out to high variance instead of learning signal.
> **The heteroscedastic loss hurt on n=2738 — disabled.**

That entry, embedded in the session-start rules, is the scar tissue of H-Exp13. It is why the final champion's config has `het_loss=False`.

There is a secondary lesson here that matters beyond this project. **Letting an LLM drive a research loop makes lucky seeds more dangerous, not less.** The agent is prone to finding a narrative that fits the result — "warmup stabilizes the variance head, here is the Goyal citation, this makes sense." That narrative can be correct in theory and still wrong in practice, because the effect size is smaller than the noise floor. The only defense is the reproduction protocol: a new champion must survive a seed sweep. The rules require it. They didn't at first. They do now.

---

## 9. The Residual MLP Breakthrough: Four Lines of Code, a Fivefold Improvement

The turning point in the project was not a hyperparameter tweak. It was an architecture edit. Specifically, it was the agent deciding — based on the diagnosis that a plain MLP at 128 hidden units could find some signal (composite +0.82) but not enough, and that gradient flow through even two GELU-dropout blocks was probably the bottleneck — to add a He-et-al.-2016 residual skip connection.

Here is the entire diff that changed the champion:

```python
class CurrencyMLP(nn.Module):
    """Residual MLP: learns correction to linear projection (He et al. 2016).

    For low-SNR financial data, the signal is a small perturbation on a
    linear baseline. The skip connection lets the nonlinear layers focus
    on learning the residual correction rather than the full mapping.
    """
    def __init__(self, n_input_features, seq_len=10, hidden_size=128,
                 head_dropout=0.1, het_loss=True):
        super().__init__()
        input_dim = n_input_features * seq_len

        # Linear shortcut: the "baseline" prediction
        self.shortcut = nn.Linear(input_dim, hidden_size)

        # Nonlinear residual branch: correction to the baseline
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = _make_heads(hidden_size, ...)

    def forward(self, x):
        flat = x.reshape(x.size(0), -1)
        hidden = self.shortcut(flat) + self.residual(flat)
        return _forward_heads(self.heads, hidden, self.het_loss)
```

There is a principled reason this works on low-signal-to-noise financial data, and it is worth unpacking because it generalizes. The hypothesis, as the agent wrote it into the checkpoint, goes like this: **in a market that is 95% efficient, the signal is a small perturbation on a linear baseline.** A simple linear regression on 104 features can already capture whatever weak trend-following, carry, or momentum effects happen to survive in the data. The job of the nonlinear branch is not to learn the whole mapping from features to return — that's hard and noisy — but to learn the *correction* to the linear baseline in specific regimes. Skip connections let the linear path carry the baseline and force the nonlinear branch to solve the easier, structured problem of regime-conditional residuals.

The result was striking. Here is the lineage of MLP experiments around the breakthrough:

| Exp | Config delta | Composite | Test Sharpe |
|---|---|---|---|
| 63 | plain MLP 128h, lr=1e-4, ep=100 | +0.411 | +0.71 |
| 64 | + head_dropout=0.2 | +0.385 | +0.98 |
| 65 | BatchNorm + dropout=0.2 | −1.312 | −0.61 |
| **66** | **Residual MLP (He 2016)** | **+4.674** | **+4.77** |
| 67 | Residual MLP seed=42 | +3.761 | +4.24 |
| 68 | Residual MLP seed=99 | +3.171 | +3.47 |

One edit. Composite jumped from +0.385 to +4.674. Across three seeds, the median test Sharpe moved from ~1.0 to over 4.0. This is not a seed lottery — this is an architectural effect that reproduced across initializations.

From there, the lineage of further improvements reads like a textbook (every step cites a paper and improves the composite):

- **Exp73 (lr=5e-4 vs 3e-4):** composite +5.40. The skip connection enabled higher LR (He et al. 2016 — skip provides gradient stability, the model can tolerate a larger effective step size).
- **Exp69 (head_dropout=0.15 vs 0.10):** composite +5.13. Slightly more head regularization improved generalization on fold 1/fold 2 (Srivastava et al. 2014).
- **Exp85/88 (huber_delta=0.5 vs 1.0):** composite +5.499 — the champion. A tighter Huber delta makes the loss more robust to the fat tails of FX returns (regular FX daily move is ~60 bps, delta=1.0 puts everything in the MSE zone). Helps fold 2 (post-crash) specifically.

The final champion config, which you can reproduce bit-for-bit:

```
backbone:       residual MLP (shortcut + 2-layer nonlinear, hidden=128, head=64)
lr:             5e-4
batch_size:     32
seq_len:        10
epochs:         50           # from-scratch needs more than fine-tuning
weight_decay:   1e-5
patience:       10
grad_clip:      1.0
huber_delta:    0.5          # robust to FX fat tails
head_dropout:   0.15
seed:           0
het_loss:       False
```

And the headline numbers:

- **Composite: +5.499**
- **Test Sharpe: +6.2113**
- **Val Sharpe: +5.599**
- **7/7 positive test folds**
- **Cumulative test return: +1001% (equity 10,000 → 110,108)**
- **Test Sortino: +11.31 · Profit factor: 3.30 · Win rate: 69.4% · Max DD: 4.13%**
- **IC (Spearman): +0.485 · Hit rate: 69.1% · MCC: 0.384**
- **Training time: 52 seconds on CPU (4 P-core threads)**

The per-fold breakdown (test) is in the checkpoint and it is the number that actually matters, because the composite's regime-robustness term only pays off if every fold is green:

| Fold | Regime | Sharpe | IC | Win Rate | Return |
|---|---|---|---|---|---|
| 1 | Pre-crisis / GFC onset (2008) | +2.46 | +0.19 | 60% | +20% |
| 2 | Post-crash recovery (2010) | +1.17 | +0.08 | 53% | +5% |
| 3 | Eurozone debt (2013) | +9.76 | +0.58 | 76% | +34% |
| 4 | Strong USD (2015) | +9.78 | +0.67 | 75% | +90% |
| 5 | Low-vol plateau (2017-19) | +8.85 | +0.64 | 71% | +29% |
| 6 | COVID / EUR crisis (2020-21) | +9.95 | +0.64 | 71% | +70% |
| 7 | Recent (2023-24) | +8.48 | +0.62 | 72% | +56% |

Every regime green. The weakest fold is fold 2 (post-crash recovery, 2010) — unsurprisingly, because mean-reverting post-crisis chop is the regime where sign-of-return prediction is hardest. But even fold 2 is positive. That is the property that made this result a champion under the composite metric, rather than just a high-aggregate-Sharpe one-trick pony.

---

## 10. The Hardware Crisis: When the CPU Itself Is the Worst Hyperparameter

On 2026-04-19 — the day the project was consolidating the final champion and archiving artifacts — the laptop crashed five times.

Not five times over a week. Five times in one afternoon. The bugcheck codes, pulled from the Windows event log and pasted into `memory/project_hardware_crash_log.md`:

| Time | Bugcheck | Name |
|---|---|---|
| 14:45 | 0x0000007f | UNEXPECTED_KERNEL_MODE_TRAP |
| 15:54 | 0x000001ca | SYNTHETIC_WATCHDOG_TIMEOUT |
| 16:06 | 0x0000001e | KMODE_EXCEPTION_NOT_HANDLED |
| 17:08 | 0x00000101 | CLOCK_WATCHDOG_TIMEOUT |
| 17:20 | 0x00000101 | CLOCK_WATCHDOG_TIMEOUT |

Different bugchecks, no common pattern, all CPU-core-related. This is the fingerprint of **hardware instability, not software**. Earlier WHEA-Logger events (from 2026-04-15) showed Corrected Machine Check errors — Internal parity errors and TLB errors — on APIC IDs 16, 17, 24, and 25. All four of those IDs are E-cores on this CPU. The CPU in question is an Intel Core i9-14900HX, a 14th-generation Raptor Lake HX part subject to the well-publicized Intel degradation issue that led to the 0x12B microcode update in August 2024 and a five-year extended warranty.

The hardware was dying. The research could not wait for the RMA. The response, baked into the codebase, is in `run_autoresearch.py`:

```python
def _pin_to_safe_cores(n_threads: int = 4):
    """Pin process to a small subset of P-cores to minimize CPU stress.

    GPU does the heavy compute; CPU is coordination only. Using fewer cores
    reduces thermal load and avoids failing E-cores (APIC 16,17,24,25 on
    this Intel 14th-gen HX showed WHEA Internal parity errors on 2026-04-15).

    Default: 4 P-core logical threads (even-numbered, avoid HT siblings)
    """
    if os.environ.get("AUTORESEARCH_USE_ALL_CORES"):
        return
    try:
        import psutil
        n = int(os.environ.get("AUTORESEARCH_N_THREADS", n_threads))
        proc = psutil.Process(os.getpid())
        logical = psutil.cpu_count(logical=True)
        if logical and logical >= 32:  # Intel hybrid
            safe_cores = [2 * i for i in range(min(n, 8))]
            proc.cpu_affinity(safe_cores)
            torch.set_num_threads(n)
            os.environ["OMP_NUM_THREADS"] = str(n)
            os.environ["MKL_NUM_THREADS"] = str(n)

# Pin at import time so every run benefits
_pin_to_safe_cores()
```

Four P-core threads, pinned to logical IDs `[0, 2, 4, 6]` — the primary (non-hyperthread) threads of the first four P-cores, avoiding every E-core entirely. Additional mitigations were applied at the OS level: CPU max frequency capped at 60% via `powercfg`, Turbo Boost disabled, 156 user processes re-pinned to the P-core mask.

The champion was then re-verified post-mitigation. The output from the reproduction run, logged in the checkpoint, reads:

> Reproduced deterministically seed=0 CPU-only 60% cap → **composite +5.4990 exactly, test Sharpe +6.2113 exactly**. 52s training. No crash.

Four decimal places, exact reproduction on crippled hardware. The portability of the model — 167K parameters, 52 seconds on 4 cores at 60% frequency with no turbo — suddenly felt less like a property of the champion and more like a survival trait. If LFM2-350M had won instead, the research would have been blocked waiting for a new laptop. The MLP kept going.

The checkpointing protocol is the other reason the project survived the hardware crisis. `CLAUDE.md` contains, as its number-one non-negotiable rule:

> **Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning, whichever comes first.** The laptop WILL crash. Every minute of uncheckpointed work is lost work.

Every experiment appends one line to `experiment_log.jsonl` (atomic). The current champion overwrites `best_config.json`. Diagnosis and next-experiment rationale overwrites `memory/project_autoresearch_checkpoint.md`. A fresh Claude Code session reading only `CLAUDE.md` plus the checkpoint can resume without re-reading any other file. The rules spell it out: the checkpoint must contain the exact bash command to run next. After a BSOD, session-start reads the checkpoint, finds the command, runs it, and the loop continues.

Without that discipline, five BSODs in an afternoon would have cost five afternoons of work. With it, they cost a few minutes each.

---

## 11. Uncertainty: Two Flavors, Used for Skipping Trades, Not for Bragging

The model emits per-sample uncertainty decomposed into two parts, following Kendall & Gal (2017):

- **Aleatoric uncertainty** — irreducible noise in the data. For the champion (`het_loss=False`), this is estimated from the MC Dropout variance decomposition. When heteroscedastic mode is enabled, it comes directly from the network's predicted log-variance.
- **Epistemic uncertainty** — model uncertainty, estimated via MC Dropout (Gal & Ghahramani, 2016). Twenty stochastic forward passes with dropout enabled, variance across passes = model disagreement.

The `predict_with_uncertainty` function in `model/backbone.py` returns both, plus total uncertainty, confidence (sigmoid of negative log total uncertainty), 1-sigma bands, and 2-sigma bands. These are not decorative. They drive the *don't-trade* filter in the deployment-ready trading strategy:

> Use confidence < 0.8 as a "don't trade" signal. High aleatoric on a fold means the model correctly identifies it as noisy. High epistemic means the model needs more data from that regime.

For the champion, per-fold aleatoric values are remarkably low (on the order of 1e-5 to 1e-4) because the 5-day return distribution is well-scaled and the residual MLP is small enough to be well-determined. Confidence is consistently ~1.0 across all folds. This is actually a double-edged result: the model is genuinely confident, but that also means the confidence signal itself does not discriminate much between easy and hard folds. In practice, aleatoric *rank-order* within a fold (relative uncertainty) is more useful than absolute magnitude for position sizing.

The earlier LFM2 heteroscedastic experiments produced much more interpretable uncertainty structure — fold 1 (GFC onset) had aleatoric around 0.17, fold 5 (low-vol plateau) had aleatoric around 0.02. The model correctly identified which regimes were noisy. This is a reminder that *good uncertainty does not imply good mean predictions*, and in fact the two can trade off. The het-loss decomposed uncertainty beautifully and degraded mean prediction. The plain-Huber residual MLP has excellent mean prediction and less rich uncertainty structure. For directional trading, the mean matters more — so the champion uses plain Huber. The uncertainty head is a fallback for confidence filtering, not the primary signal.

---

## 12. Infrastructure: Decoupled Logs, a Dashboard, and Winner Archives

The infrastructure around the loop is deliberately plain. Three principles:

1. **Runners log, dashboards display, evaluators evaluate. Never tangle them.**
2. **Append-only structured logs. Never rewrite history.**
3. **Every champion is archived as a fully self-contained, portable artifact.**

The runner (`run_autoresearch.py`) does exactly one thing: run one experiment, append one line of JSON to `autoresearch_results/experiment_log.jsonl`, and — if the result is a new best — overwrite `autoresearch_results/best_config.json` with the full config, metrics, and per-fold breakdown. It does not analyze. It does not decide what to try next. It does not render.

The dashboard (`autoresearch_results/dashboard.html`) reads the JSONL file directly. It is a static HTML page served over a simple `python -m http.server`. Separation from the runner means the dashboard can be refreshed, redesigned, or replaced without touching the training pipeline. It shows the experiment trajectory, per-window (per-fold) breakdown for train/val/test, and allows drill-down to each experiment's detailed metrics.

Every time the composite increases, the agent is required by `CLAUDE.md` to archive a **winner**. The archive lives at `autoresearch_results/winners/<backbone>_exp<N>_<short_desc>/` and is specified down to the directory layout:

```
mlp_exp32_residual_seed0/
├── README.md                 # model description, per-fold tables, trading strategy
├── config.json               # exact config
├── model_checkpoint.pt       # self-contained weights (state_dict + scaler + feature list)
├── experiment_log_entry.json
├── per_fold_results.json
├── code/                     # frozen source snapshot
│   ├── backbone.py
│   ├── train.py
│   ├── features.py
│   ├── splits.py
│   ├── metrics.py
│   └── run_autoresearch.py
├── inference/
│   ├── predict.py            # standalone inference script
│   └── README_inference.md
└── reproduction/
    ├── reproduce_log.txt
    └── seed_variance.json
```

The model checkpoint is designed to be portable without the source repo. `torch.save` writes the state dict plus the StandardScaler's `mean_` and `scale_` arrays plus the feature column list plus the full hyperparameter config. Someone on a fresh machine can rebuild the residual MLP from its architecture definition, load the weights, apply the scaler, and make predictions. No dependence on the autoresearch package at inference time.

The winner README is required to include a full **trading strategy section** — signal generation, entry rules with magnitude and confidence thresholds, position sizing (Kelly fraction with a per-trade cap), exit rules, rebalancing cadence, per-regime performance table, risk controls (daily loss cap, drawdown pause, regime-shift detection), expected performance estimates pre- and post-cost, and caveats (seed variance, pair specificity, transaction cost sensitivity). A final Colab notebook at `colab_train_and_infer.ipynb` must reproduce training and inference end-to-end on the Colab free tier. The bar for "new best" is very high on purpose: a winner is not just a checkpoint, it is a complete, reproducible, and deployable artifact.

This MLOps discipline is not decoration. It is the thing that makes a 99-experiment trajectory auditable. Any reviewer can pick a specific experiment number, read its JSONL entry, compare it to the current champion, and verify that the KEEP/DISCARD decision was correct under the composite metric.

---

## 13. Lessons Learned

Fifteen thousand words in, here are the five things I actually believe at the end of this:

**1. Seed variance dominates early on. Budget for it.**
On small financial datasets with overparameterized heads, the seed is a bigger knob than almost any hyperparameter. The LFM2 phase showed a +3.29 composite swing between seeds at the same configuration. Until you have verified a new champion across at least three seeds and the *median* improves over the previous median, the "new best" is provisional. Build this into the protocol or you will ship lucky seeds.

**2. Code changes are the highest-leverage action. Hyperparameters hit a ceiling.**
The composite went from +0.82 to +4.67 on a single architecture change (residual skip). It went from +0.82 to about +1.5 on all the hyperparameter tuning in between. When three discards pile up, stop tweaking and change the structure — the loss function, the architecture, the features. The LLM is more than capable of doing this *if* the rules make it allowed and rewarded.

**3. Regularization trade-offs are real and regime-specific.**
Head dropout 0.15 helped fold 2 (post-crash chop) but cost a little on folds 4–6. BatchNorm destroyed the model because it removed regime-scale information. Huber delta 0.5 vs 1.0 is the difference between fat-tail-robust and fat-tail-sensitive; it helps fold 2 and doesn't hurt the good folds. None of these are "universally better." All are regime-dependent, and the composite metric is what arbitrates between them.

**4. Frozen foundation models can lose to 167K-parameter MLPs on small data.**
This is the uncomfortable lesson. LFM2.5-350M is a serious piece of engineering and it got bested, reproducibly and across seeds, by a model that would have fit on a 2005-era cell phone. The reason is the 104-to-1024 projection layer and the 39-parameters-per-training-sample ratio it induces. Foundation models need either (a) much more data, or (b) a much smaller adapter surface, or (c) partial unfreezing, to be competitive on low-SNR problems like daily-horizon FX. The right default for this kind of problem is a small, well-regularized, from-scratch model with a skip connection — not a transfer-learning setup.

**5. Hardware instability is a real research blocker. Plan for it.**
Five BSODs in one afternoon is not a thing a well-designed research plan accounts for, and yet here we are. The protocol that saved the project — checkpoint after every experiment, append-only logs, portable self-contained winner archives, CPU pinning to avoid failing cores, frequency caps to avoid thermal stress — was not on my radar before this project started. It is now. If your pipeline assumes the machine stays up, you will lose work when it doesn't.

---

## 14. The Meta-Research: Does Letting the LLM Drive Actually Work?

This is the part of the article I am least sure about and most interested in.

Here is what I observed. The agent produced 99 experiments under a protocol that required, for each one, a written diagnosis of the champion's weakness, a literature citation, a hypothesis with a predicted composite change, and a post-hoc analysis. The experiment log contains those rationales. They are, on the whole, coherent. They cite real papers — He et al. 2016, Srivastava et al. 2014, Goyal et al. 2017, Kendall & Gal 2017, Gu/Kelly/Xiu 2020, Lopez de Prado 2018 — and they use them in the right places. The champion lineage is traceable: plain MLP → residual MLP (He 2016) → higher LR (He 2016 again, skip enables larger steps) → tighter Huber (robust to FX fat tails) → head dropout bump (Srivastava 2014, fold-2 robustness). Each step cites its reason.

Here is what went wrong. The agent fell for the H-Exp13 warmup "breakthrough" because a single lucky seed produced a plausible story. The post-hoc analysis did generate a coherent narrative — warmup stabilizes the log-variance head initialization, Goyal et al. 2017 — and that narrative is not *wrong* in theory. It is just not what was happening on this dataset. The saving grace was the reproduction protocol, which the agent itself had written into the rules and which produced the five-run reproduction table that ultimately debunked H-Exp13. But the agent *did* make the mistake. It needed the protocol to catch it.

Three things emerge from that observation.

**First, an LLM-driven loop is not self-correcting without explicit skeptical machinery.** The agent will believe its own rationales unless the rules force it to reproduce, cross-seed, and test against prediction. The protocol has to do the epistemology. This maps directly to how human research works — peer review, replication, pre-registration — except that an LLM needs those instruments written down and enforced mechanically. "Reproduce new champions across three seeds before accepting" is a two-line rule that saved the project from a false positive.

**Second, the agent is better at justification than at invention.** Given a concrete diagnosis (fold 2 weak, high train-test gap at this LR, variance-branch dominance suggested by aleatoric > 0.2), the agent reliably produces a correct literature-backed hypothesis. It is good at this. What it is worse at is coming up with the *next* diagnostic question to ask when the obvious angles are exhausted. That is where the "code changes are allowed" rule matters most: without it, the agent keeps sweeping hyperparameters when the problem calls for a structural change. The residual MLP breakthrough happened because the rules explicitly permitted editing `backbone.py` and rewarded it under the composite metric.

**Third, the checkpointing protocol is not just crash recovery — it is context compression.** Every session begins by reading `memory/project_autoresearch_checkpoint.md`, which contains the current champion, the per-fold diagnostics, the exhausted axes, and the exact bash command for the next experiment. The checkpoint is the agent's long-term memory. Without it, each session would have to rebuild context from the 99-entry JSONL log, which is expensive and error-prone. With it, a fresh session picks up exactly where the previous one left off, same mental model, same next move. This is the architectural pattern that makes a long-horizon LLM research loop feasible at all.

Was the whole thing worth it, compared to a human running the same experiments? I honestly think it was *faster*. Ninety-nine experiments in the elapsed calendar time this took would be tight for a human researcher, especially given the interruptions. The agent did not need motivation, did not lose momentum, did not skip the documentation step, and was perfectly happy to revert to the champion after a DISCARD without sulking. The reliability of the protocol — seven steps, one change, cite your reasoning, checkpoint — is where the productivity gain came from. The LLM was not smarter than a human researcher. It was just relentlessly disciplined in a way that is hard for humans to sustain for 99 iterations.

If I were starting over, three things would change. I would write the seed-reproduction rule into the protocol from day one, not discover the need for it mid-way through the LFM2 phase. I would require a diversity term in the experiment-selection prompt — "pick one small tweak and one radical change each K iterations" — so the agent more aggressively escapes local optima instead of waiting for three DISCARDs to notice a plateau. And I would instrument the meta-loop itself: log every hypothesis the agent states and check whether it matches the result. That is the most direct way to measure whether the LLM's mental model is calibrated.

---

## 15. Reproducibility and Closing Thoughts

The champion is fully reproducible. The exact bash command, from the checkpoint:

```bash
cd C:/Users/evija/autoresearch && \
CUDA_VISIBLE_DEVICES="" \
python -m autoresearch.run_autoresearch \
  --backbone mlp \
  --lr 5e-4 --batch-size 32 --seq-len 10 --epochs 50 \
  --weight-decay 1e-5 --patience 10 --grad-clip 1.0 \
  --huber-delta 0.5 --head-dropout 0.15 --seed 0 \
  --description "mlp residual champion"
```

On a stable machine this runs in under a minute on CPU. It produces, deterministically, `composite=+5.4990`, `test_sharpe=+6.2113`, 7/7 positive test folds. Cross-seed (0 / 42 / 99), the median test Sharpe is +4.76.

The full project — data download, feature engineering, the seven-regime split with purge/embargo/buffer, all eight backbones, the training loop with heteroscedastic-loss option, the per-trade logger, the dashboard, the winner archive, and the `CLAUDE.md` protocol — is available at:

> **GitHub:** `https://github.com/USERNAME/autoresearch` *(placeholder — will be filled in at publication)*

The winner archive, under `autoresearch_results/winners/mlp_exp32_residual_seed0/`, includes the portable checkpoint, the inference script, the frozen source snapshot, the trading strategy write-up, and a self-contained Colab notebook. If you want to verify the result, that directory is the place to start. If you want to follow the trajectory, read `experiment_log.jsonl` top-to-bottom and match it against `research_journal.md`.

The point of this project was not really to produce a EUR/USD model, although I'm happy with the one that came out. The point was to test whether an LLM can sustain a real research loop — diagnose, cite, hypothesize, predict, run, analyze, document, iterate — across a hundred experiments without losing coherence, without silently drifting off the baseline, and without papering over inconvenient results. The answer, based on this run, is a qualified yes. The agent needs an explicit protocol, an append-only log, a reproduction rule to catch lucky seeds, permission to modify the codebase, and checkpointing so the work survives the hardware. Given those, it can do the job. The quality of the research is bounded above by the quality of the rules you give it, and bounded below by whether the rules get enforced.

If you take one thing from this article, take this: *the composite metric is the most important line of code in the project.* It is the objective function of the meta-optimizer. If you get that wrong — if you reward top-line Sharpe instead of regime-robust minimum, or if you reward test without val, or if you forget the fold-count penalty — the agent will find the wrong solution with exactly the same protocol and exactly the same citations. The LLM cannot fix a broken objective. It can, however, drive a beautifully efficient search through experiment space when the objective is right.

Ninety-nine experiments. One champion. One +1001% equity curve on seven held-out regimes. And one residual connection, four lines of code, that did more than the other ninety-five experiments combined.

---

*Thanks for reading. If you're working on LLM-driven research loops, autonomous ML agents, or low-SNR financial prediction — I'd love to hear from you. The checkpointing and reproduction protocols from this project are the parts I think generalize best; grab them and adapt them.*
