---
name: Autoresearch Checkpoint
description: 125 exps. NEW GLOBAL CHAMPION LSTM Exp20 composite +6.1312 test Sharpe +6.3363. LSTM 20/50.
type: project
---

## Session Recovery
1. Read this
2. Read `memory/project_hardware_crash_log.md` — CPU 60% cap, Turbo off, 0 crashes since mitigation
3. Read JSONL tail (125 entries)
4. Dashboard: http://localhost:8765/dashboard.html (per-backbone tabs + reasoning panel)

## 🏆 GLOBAL CHAMPION
**LSTM Exp20** — composite **+6.1312** | test Sharpe **+6.3363** | 7/7 positive test | +1048% return
- Config: BiLSTM h=128, 2-layer, lr=1e-3, bs=32, seq=10, ep=100, wd=5e-4, pat=15, hd=0.25, huber=1.0, seed=0
- Archived `winners/lstm_exp20_wd5e4_seed0/`
- Prior champions: LSTM Exp9 (+6.10), LSTM Exp4 (+6.07), MLP Exp32 residual (+5.50)

## Per-Backbone Status
| Backbone | Exps | Best Comp | Best Test Sharpe | Status |
|----------|------|-----------|------------------|--------|
| lfm2-350m | 43 | +1.77 | +2.07 | done (need 7 more per 50-mandate) |
| mlp | 54 | +5.499 | +6.21 | done |
| **lstm** | **20** | **+6.1312** | **+6.3363** | **IN PROGRESS (20/50) — GLOBAL CHAMP** |
| patchtst | 1 | -1.72 | -0.82 | pending (49/50) |
| patchtsmixer | 0 | — | — | pending |
| xgboost | 0 | — | — | pending |
| lightgbm | 0 | — | — | pending |
| catboost | 0 | — | — | pending |

## LSTM Experiment Summary (20 so far)
| # | Change | Composite | Learning |
|---|--------|-----------|----------|
| 108 | SOTA baseline | +4.12 | baseline |
| 109 | huber=0.5 | +3.98 | huber doesn't help LSTM |
| 110 | ep=100 pat=15 | +5.06 | SOTA epochs help |
| 111 | hd=0.25 | +6.07 | GLOBAL CHAMP — head dropout breakthrough |
| 112 | hd=0.30 | +6.02 | 0.25 peaks |
| 113 | wd=1e-4 | +6.10 | GLOBAL CHAMP — 10x L2 |
| 114 | lr=5e-4 | +4.95 | flat minima hurt test |
| 115 | unidirectional | +5.00 | val/test split |
| 116 | seq=20 | +4.25 | longer context hurts |
| 117 | PatchTST Exp1 (different backbone) | — | — |
| 118 | 3-layer stacked | +1.64 | depth hurts small n |
| 119 | GRU cell | +4.59 | LSTM better |
| 120 | LayerNorm input | +4.51 | double-norm destabilizes |
| 121 | seq=5 | +5.70 | fold 2 test +3.70 BEST EVER but fold 1/7 weaker |
| 122 | warmup=3 | +4.37 | warmup hurts |
| 123 | hd=0.20 | +5.53 | 14/14 folds positive BEST but lower peak |
| 124 | grad_clip=0.5 | +5.46 | tighter clip hurts fold 2 |
| **125** | **wd=5e-4** | **+6.13** | **GLOBAL CHAMP — 50x L2** |

## Code Changes This Session
- CurrencyLSTM: `num_layers`, `bidirectional`, `cell` (lstm/gru), `input_layernorm` parameters
- Runner: `--num-layers`, `--rnn-cell`, `--unidirectional`, `--input-layernorm` flags
- **Bug fix**: best_config.json now tracks GLOBAL champion (was per-backbone)
- **New**: Runner auto-writes reasoning_annotations.json per experiment (for dashboard)

## Next LSTM Experiments (need 30+ more to hit 50)
Per CLAUDE.md SOTA mandate, explore 2024-2026 variants:
- **xLSTM** (Beck et al. 2024) — extended LSTM with exponential gating
- **Mamba / SSM** (Gu & Dao 2024) — state-space model replacement for RNN
- **AWD-LSTM** (Merity 2018) — weight-dropped LSTM for regularization
- **DA-RNN attention** (Qin 2017) — dual-stage attention on input+temporal
- Fine-grained search: wd sweep (5e-4 → 1e-3), hd × wd combos, lr×patience grid
- Multi-seed variance study on Exp20 champion (seeds 7, 42, 99, 2024)
