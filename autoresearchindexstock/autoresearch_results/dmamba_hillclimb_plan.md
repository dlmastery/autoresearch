# dMamba 25-Experiment Hill-Climb Plan (from exp 52 winner)

> User directive 2026-04-30: stop panel/LSTM/MLP exploration; pivot to 25-experiment dmamba hill-climb from the global champion config. Start from EXACT exp 52 config. Optimize on target A (fwd_ret_1d composite — the runner's KEEP/DISCARD metric). Each experiment must cite arxiv literature and be a single-config change.

## Champion baseline (exp 52)

| Param | Value | Source |
|---|---|---|
| backbone | mamba | — |
| mamba-variant | dmamba | Cai et al. 2024 NeurIPS MambaTS (arXiv:2405.16440); DMamba (arXiv:2602.09081) |
| expand | 2 | Gu & Dao 2024 COLM Mamba (arXiv:2312.00752) sec 5.2 sweet spot 1-4 |
| d-state | 32 | Gu & Dao 2024 COLM Mamba sec 5.2 canonical 16-128 |
| seq-len | 60 | Nie et al. 2023 ICLR PatchTST (arXiv:2211.14730) min seq for attention/SSM heads |
| lr | 5e-4 | Smith 2017 (arXiv:1708.07120) recommended for SSM scale |
| bs | 32 | Keskar 2017 ICLR (arXiv:1609.04836) flat-minima |
| epochs | 100 | Gu & Dao 2024 sec 5 standard |
| weight-decay | 0.1 | Loshchilov & Hutter 2019 ICLR (arXiv:1711.05101) AdamW |
| patience | 20 | 20% of epochs |
| warmup-epochs | 10 | Goyal et al. 2017 (arXiv:1706.02677) |
| head-dropout | 0.1 | empirical sweet spot |
| huber-delta | 1.0 | Huber 1964 |
| grad-clip | 1.0 | Pascanu et al. 2013 ICML (arXiv:1211.5063) RNN-style |
| seed | 42 | (also test 0, 99, 7, 2024) |

Result: **composite +1.3216**, val_sharpe +1.4831, IC +0.0737, hit 52.51%, equity $3144 (+214%), 7/7 positive test folds. PSR 0.997.

## 25 Hill-Climb Experiments

Format: `Exp# | Axis | Change | Citation | Prediction`

### Architectural axes (Mamba-specific) — exps 260-268

| # | Axis | Change | Citation | Hypothesis (composite range) |
|---|---|---|---|---|
| 260 | d-state | 32 → 64 | Gu & Dao 2024 COLM Mamba sec 5.2 — canonical d_state range 16-128, larger SSM memory may capture longer-range dependencies | [+1.10, +1.45]; mechanism: doubled SSM hidden state captures more regime persistence |
| 261 | d-state | 32 → 16 | Gu & Dao 2024 sec 5.2 — small d_state baseline; ablation toward exp 48 prior | [+1.00, +1.35]; mechanism: revert toward exp 48 +1.19 baseline; verifies d_state=32 was real lift |
| 262 | expand | 2 → 4 | Gu & Dao 2024 sec 5.2 — 2x more inner channels (revert to exp 17 +0.86) | [+0.80, +1.20]; verifies the exp 48 "expand=2 down from 4" lift |
| 263 | expand | 2 → 1 | Gu & Dao 2024 — minimal-parameter mamba variant | [+0.90, +1.30]; mechanism: smaller model may use 9-epoch budget more productively given QQQ small-N |
| 264 | num-layers | (default) → 3 | Gu & Dao 2024 sec 5 — deeper SSM stack; Beck 2024 xLSTM (arXiv:2405.04517) shows depth lift | [+1.00, +1.50]; mechanism: 3-layer stack adds compositional regime learning |
| 265 | num-layers | (default) → 4 | xLSTM/Mamba2 sec 4 — 4-layer often sweet spot for ~10k sample tasks | [+0.90, +1.50]; mechanism: deeper but riskier on small data |
| 266 | mamba-variant | dmamba → mambats | Cai et al. 2024 NeurIPS MambaTS (arXiv:2405.16440) — same paper, no decomposition variant | [+0.50, +1.30]; verifies the decomposition was the lever |
| 267 | mamba-variant | dmamba → s_mamba | Wang et al. 2024 (arXiv:2403.09898) — bidirectional Mamba | [+0.40, +1.20]; mechanism: scan in both directions captures retrospective regime detection |
| 268 | mamba-variant | dmamba → mamba2 (vanilla) | Gu & Dao 2024 baseline; ablation | [+0.50, +1.20]; verifies d-mamba decomposition value |

### Optimization axes — exps 269-275

| # | Axis | Change | Citation | Hypothesis |
|---|---|---|---|---|
| 269 | lr | 5e-4 → 3e-4 | Smith 2017 (arXiv:1708.07120) 0.5x sweep step; Goyal 2017 — implicit reg from data | [+1.20, +1.45]; flatter minimum |
| 270 | lr | 5e-4 → 7e-4 | Smith 2017 1.4x sweep step | [+1.10, +1.40]; faster convergence |
| 271 | warmup-epochs | 10 → 5 | Goyal et al. 2017 (arXiv:1706.02677) sec 2.2 — shorter warmup at small batch | [+1.10, +1.40] |
| 272 | warmup-epochs | 10 → 20 | Goyal et al. 2017 — longer warmup for stable SSM | [+1.10, +1.45] |
| 273 | bs | 32 → 16 | Keskar 2017 (arXiv:1609.04836) — smaller batch → flatter minima | [+1.00, +1.40] |
| 274 | bs | 32 → 64 | Goyal et al. 2017 — larger batch + linear LR scaling | [+1.00, +1.40] |
| 275 | patience | 20 → 30 | Smith 2017 — longer patience for SSM convergence | [+1.20, +1.45] |

### Regularization axes — exps 276-281

| # | Axis | Change | Citation | Hypothesis |
|---|---|---|---|---|
| 276 | weight-decay | 0.1 → 0.05 | Loshchilov-Hutter 2019 (arXiv:1711.05101) — AdamW decoupled wd; 0.5x sweep | [+1.10, +1.40] |
| 277 | weight-decay | 0.1 → 0.2 | Loshchilov-Hutter 2019 — 2x sweep, stronger shrinkage | [+1.00, +1.40] |
| 278 | head-dropout | 0.1 → 0.05 | Srivastava et al. 2014 JMLR Dropout — less reg at SSM scale | [+1.10, +1.40] |
| 279 | head-dropout | 0.1 → 0.2 | Srivastava 2014 — 2x reg | [+1.00, +1.40] |
| 280 | grad-clip | 1.0 → 0.5 | Pascanu 2013 (arXiv:1211.5063) — stricter clip for stable SSM | [+1.10, +1.40] |
| 281 | grad-clip | 1.0 → 2.0 | Pascanu 2013 — looser clip allows larger gradient steps | [+1.05, +1.40] |

### Sequence length axes — exps 282-284

| # | Axis | Change | Citation | Hypothesis |
|---|---|---|---|---|
| 282 | seq-len | 60 → 90 | Beck et al. 2024 NeurIPS xLSTM (arXiv:2405.04517) sec 4 — monotonic accuracy lift with seq | [+1.20, +1.50] |
| 283 | seq-len | 60 → 40 | Nie et al. 2023 PatchTST — shorter seq if signal is short-range | [+1.00, +1.35] |
| 284 | seq-len | 60 → 120 | Hochreiter-Schmidhuber 1997 / Beck 2024 — long-range gates; doubled history | [+1.10, +1.55] |

## Decision rules

- **KEEP** if composite > +1.3216 with all 7 test folds positive
- **NEAR-MISS** if composite in [+1.20, +1.32] (within 0.12 of champion)
- **DISCARD** otherwise
- **Multi-seed lock** required before archiving any new champion (≥3 seeds, median > +1.10 baseline equivalence)

## Run cadence

- One experiment at a time per CLAUDE.md
- Pre-run reasoning into reasoning_annotations.json BEFORE launch
- Verdict + learning written immediately after lands
- Commit + push every 5 experiments with dashboard sync via `python autoresearch/_sync_dashboard_to_docs.py`
- Update memory checkpoint after each KEEP

## Expected timeline

- Each experiment ~25-35 min (similar to exp 52's 1867s = 31 min)
- 25 experiments × 30 min = ~12.5 hours total
- Roll up commits every ~3 hours / 5 experiments
