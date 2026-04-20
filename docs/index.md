---
layout: default
title: AutoResearch — Karpathy-style autonomous FX prediction
---

# AutoResearch

> **An LLM-driven research loop that ran 104 experiments on EUR/USD directional forecasting and converged on a residual MLP with test Sharpe +6.21 across 7 held-out regimes.**

## TL;DR

- **Champion**: 167K-parameter residual MLP
- **Test Sharpe**: +6.2113 (7/7 positive folds, 2005–2024 held-out regimes)
- **Val Sharpe**: +5.599 (6/7 positive folds)
- **Cumulative return**: +1001% on held-out test set
- **Reproducible**: deterministic seed=0, CPU-only, 52 seconds
- **Composite metric**: `min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds` = +5.499

## Read the full story

📖 **[Medium Article — How an LLM Drove 104 Experiments Without Losing the Plot](./medium_article.html)** *(rendered from [medium_article.md](https://github.com/dlmastery/autoresearch/blob/master/autoresearch/autoresearch_results/medium_article.md))*

7,690 words, 15 sections covering:
- The autoresearch 7-step protocol (diagnose → cite → hypothesize → predict → run ONE → analyze → checkpoint)
- Super-fold splits with 90-day purge + 21-day embargo + 10-day label buffer (zero leakage)
- The composite objective function — "the most important line of code in the project"
- The seed variance crisis — why one lucky seed can fool the agent (and how the reproduction protocol caught it)
- The heteroscedastic loss detour (Kendall & Gal 2017) — and why it hurt on small data
- The **residual MLP breakthrough** — 4 lines of code (He et al. 2016 skip connection) that took composite from +0.82 to +5.499
- The Intel i9-14900HX hardware crisis — 5 BSODs in one day, WHEA parity errors, software mitigations
- Meta-lessons on LLM-driven research

## Repo

- **Source**: [github.com/dlmastery/autoresearch](https://github.com/dlmastery/autoresearch)
- **Champion config**: [best_config.json](https://github.com/dlmastery/autoresearch/blob/master/autoresearch/autoresearch_results/best_config.json)
- **Winner archive**: [autoresearch_results/winners/mlp_exp32_residual_seed0/](https://github.com/dlmastery/autoresearch/tree/master/autoresearch/autoresearch_results/winners/mlp_exp32_residual_seed0) — portable checkpoint, inference script, frozen source snapshot, Colab notebook
- **Experiment log** (104 entries): [experiment_log.jsonl](https://github.com/dlmastery/autoresearch/blob/master/autoresearch/autoresearch_results/experiment_log.jsonl)
- **Research journal** (arxiv-cited): [research_journal.md](https://github.com/dlmastery/autoresearch/blob/master/autoresearch/autoresearch_results/research_journal.md)
- **Agent protocol**: [CLAUDE.md](https://github.com/dlmastery/autoresearch/blob/master/CLAUDE.md) — the 14-section audit requirement + 7-step research process

## Reproduce the champion

```bash
git clone https://github.com/dlmastery/autoresearch.git
cd autoresearch
pip install -e .

# Reproduce (52 seconds on CPU, deterministic):
CUDA_VISIBLE_DEVICES="" python -m autoresearch.run_autoresearch \
  --backbone mlp --lr 5e-4 --batch-size 32 --seq-len 10 --epochs 50 \
  --weight-decay 1e-5 --patience 10 --grad-clip 1.0 \
  --huber-delta 0.5 --head-dropout 0.15 --seed 0 \
  --description "reproduce champion"
# → composite +5.499, test Sharpe +6.2113, 7/7 positive folds
```

## Per-fold results (champion, held-out test set)

| Fold | Regime | Period | Test Sharpe | Return | Hit Rate |
|------|--------|--------|-------------|--------|----------|
| 1 | Pre-crisis + GFC onset | 2006–2008 | +2.46 | +19.79% | 60.2% |
| 2 | Post-crash recovery | 2009–2010 | +1.17 | +5.49% | 53.3% |
| 3 | Eurozone debt | 2011–2012 | +9.76 | +34.13% | 74.5% |
| 4 | Strong USD | 2014–2016 | +9.78 | +90.30% | 75.0% |
| 5 | Low-vol plateau | 2017–2019 | +8.85 | +29.29% | 71.0% |
| 6 | COVID/EUR crisis | 2020–2021 | +9.95 | +69.54% | 70.9% |
| 7 | Recent mixed | 2023–2024 | +8.48 | +55.75% | 71.6% |

## License & caveats

Research / educational use. Past backtest performance on EUR/USD ≠ live trading results. FX is notoriously efficient and slippage-sensitive; the residual skip advantage documented here may not survive transaction costs or different pairs. See the Medium article's "Known Limitations" section.

---

*Built with Claude Code as the outer research loop. Hardware: Intel i9-14900HX (degraded), RTX 4090 Laptop (unused for champion — CPU-only trains in 52s).*


---

## Live experiment dashboard

[Live dashboard](./dashboard/) — every experiment, per-fold Sharpe/IC/hit-rate, click a row to see its arXiv-cited reasoning. Auto-synced from `autoresearch/autoresearch_results/` on every commit per CLAUDE.md 'GitHub Pages Dashboard Sync' rule.
