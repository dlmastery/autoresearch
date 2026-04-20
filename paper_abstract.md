# AutoResearch — Paper Abstract

**Title:** *AutoResearch: An LLM-Driven Autonomous Research Loop for Financial Time Series Forecasting*

**Full paper:** [`docs/paper.md`](docs/paper.md)

---

## Abstract

We study whether a large language model, operating as an autonomous researcher rather than as a code assistant, can drive a closed-loop machine learning research process from literature review through hyperparameter selection, experiment execution, diagnosis, and champion archival. We instantiate this loop on a daily EUR/USD foreign-exchange forecasting benchmark (2005–2025, $n=2738$ trading days, 104 engineered features) using a seven-regime super-fold evaluation protocol with 90-day purge, 21-day embargo, and 10-day label-horizon buffers.

Over 151 experiments across four backbones (MLP, LSTM, LFM2-350M, PatchTST), the agent identifies a bidirectional two-layer LSTM at hidden size 128 as the global champion, achieving a composite score of **+6.4242**, a test Sharpe of **+6.5242**, a validation Sharpe of **+7.1539**, and positive Sharpe across all **seven of seven test fold windows** spanning the 2008 Global Financial Crisis onset, post-crash recovery, Eurozone debt, strong-USD downturn, low-volatility, EUR crisis, and recent mixed regimes. Cumulative test return across the 1170-day test horizon reaches **+1122%** under a simple sign-based trading rule.

A multi-seed variance study at the champion configuration reveals composite standard deviation $\approx 1.0$ across six seeds (range $\approx 2.58$), which we interpret as evidence that single-seed "champions" in financial machine learning are probabilistically lucky and that **median-of-$k$ reporting should become a community standard**. We release the complete autoresearch protocol, reasoning annotations, and per-experiment trade logs, and argue that the primary scientific artifact of such work is the *reasoning trace* rather than the final model.

---

## Key numbers at a glance

| Metric | Value |
|---|---|
| Composite | **+6.4242** |
| Test Sharpe | **+6.5242** |
| Val Sharpe | +7.1539 |
| Positive test folds | **7 / 7** |
| Cumulative test return | +1122.29% |
| Max drawdown (test) | 7.54% |
| Accuracy | 72.76% |
| MCC | 0.4554 |
| Experiments run | 151 |
| Backbones explored | 4 (MLP, LSTM, LFM2-350M, PatchTST) |

## Contributions

1. **Autonomous LLM research loop** — a seven-step diagnose → cite → hypothesise → predict → run → analyse → checkpoint protocol.
2. **New SOTA on the EUR/USD super-fold benchmark** — LSTM BiLSTM $(128, 2)$, composite +6.4242.
3. **Seed-variance revelation** — std ≈ 1.0 composite at champion config; median-of-$k$ should be the minimum reporting bar.
4. **Ten-backbone Tier-2 roadmap** — TimesFM 2.5, Chronos, Moirai, MOMENT, TiRex, Sundial, Time-MoE, TimeMixer++, TimesNet, MambaTS.
5. **Institutional-memory dashboard design** — `reasoning_annotations.json` schema plus dashboard rendering; the reasoning trace is a primary scientific artifact.
