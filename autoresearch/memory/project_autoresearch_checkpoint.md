---
name: Autoresearch Checkpoint
description: MLP 34/50. Champion residual MLP composite +5.50 test Sharpe +6.21 7/7 folds. 90 total experiments. Next try bs=16.
type: project
---

## Session Recovery
1. Read this checkpoint
2. Read JSONL tail (last 3) + best_config.json
3. Start dashboard: `"C:/Users/evija/anaconda3/python.exe" -m http.server 8765 --directory C:/Users/evija/autoresearch/autoresearch/autoresearch_results`
4. Dashboard at http://localhost:8765/dashboard.html
5. Resume from next experiment below

## Completed: LFM2 (50/50) — median test Sharpe +1.40
## Current: MLP (34/50, 90 total experiments)

### CHAMPION: Exp29/32 residual MLP seed=0 (DETERMINISTIC — verified reproduces exactly)
**Config:** residual MLP (shortcut + 2-layer), hidden=128, head=64, lr=5e-4, bs=32, seq=10, ep=50, wd=1e-5, pat=10, hd=0.15, huber=0.5, seed=0

**Per-fold test (7/7 positive):**
| Fold | Regime | Sharpe | Return | WR | IC |
|------|--------|--------|--------|-----|-----|
| 1 | Pre-crisis/GFC | +2.46 | +19.8% | 60% | +0.19 |
| 2 | Post-crash recovery | +1.17 | +5.5% | 53% | +0.08 |
| 3 | Eurozone debt | +9.76 | +34.1% | 75% | +0.58 |
| 4 | Strong USD | +9.78 | +90.3% | 75% | +0.67 |
| 5 | Low-vol plateau | +8.85 | +29.3% | 71% | +0.64 |
| 6 | EUR crisis | +9.95 | +69.5% | 71% | +0.64 |
| 7 | Recent mixed | +8.48 | +55.8% | 72% | +0.62 |

Test Sharpe +6.21 | Val Sharpe +5.60 | Composite +5.50 | Total Return +1001%

### Cross-seed verification (median test Sharpe +4.76):
| Seed | Composite | Test Sharpe |
|------|-----------|-------------|
| 0 | +5.50 | +6.21 |
| 42 | +4.45 | +4.69 |
| 99 | +4.46 | +4.76 |

### Exhausted MLP Axes
- Architecture: plain → **residual skip** (5x improvement, He 2016)
- Hidden: 512, **128** — smaller better (Gu, Kelly & Xiu 2020)
- LR: 3e-4, **5e-4**, 7e-4
- Epochs: 20, **50**, 100
- Head dropout: 0.1, **0.15**, 0.2
- Huber delta: **0.5**, 1.0
- Seq len: **10**, 20
- Weight decay: **1e-5**, 1e-3 (dead on MLP)
- BatchNorm: hurt (removes regime-scale info)
- Seeds verified: 0, 42, 99

### Key Architectural Findings
1. Residual skip connection = 5x improvement over flat MLP
2. Higher LR (5e-4) enabled by skip connection stability
3. Head dropout 0.15 optimal — balances fold 2 vs other folds
4. Huber delta 0.5 better than 1.0 for residual arch
5. MLP hidden 128 + head 64 vs old 512 + 256 = eliminated memorization
6. LFM2 foundation model underperforms simple residual MLP on daily FX

### Next Experiment
**bs=16** — Smaller batch for implicit regularization (Smith & Le 2018).

```bash
cd C:/Users/evija/autoresearch && "C:/Users/evija/anaconda3/python.exe" -m autoresearch.run_autoresearch --backbone mlp --lr 5e-4 --batch-size 16 --seq-len 10 --epochs 50 --weight-decay 1e-5 --patience 10 --grad-clip 1.0 --huber-delta 0.5 --head-dropout 0.15 --seed 0 --description "mlp: Exp35 bs=16 more gradient noise (Smith2018) seed=0"
```

### Remaining MLP experiments (16 to go): bs=16, bs=64, warmup=3, grad_clip=0.5, wd=1e-4, then seed sweeps on any improvements, then move to LSTM backbone (50 experiments).
