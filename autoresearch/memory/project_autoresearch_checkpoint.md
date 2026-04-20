---
name: Autoresearch Checkpoint
description: MLP 37/50 (93 total). Champion Exp32/36 residual MLP composite +5.50 test Sharpe +6.21 7/7 folds. BIOS upgrade verified safe — Exp36 reproduced Exp32 EXACTLY.
type: project
---

## Session Recovery
1. Read this checkpoint
2. Read JSONL tail (last 3) + best_config.json
3. Start dashboard: `"C:/Users/evija/anaconda3/python.exe" -m http.server 8765 --directory C:/Users/evija/autoresearch/autoresearch/autoresearch_results`
4. Dashboard at http://localhost:8765/dashboard.html
5. Resume from next experiment below

## Completed: LFM2 (50/50) — median test Sharpe +1.40
## Current: MLP (37/50, 93 total experiments)

### CHAMPION: Exp32/36 residual MLP seed=0 (DETERMINISTIC — verified post-BIOS)
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

### BIOS Upgrade Verification (Exp36 post-BIOS)
All metrics reproduced EXACTLY:
- Composite: +5.499 (was +5.499) ✓
- Test Sharpe: +6.2113 (was +6.2113) ✓
- All 7 fold Sharpes match to 4 decimals ✓
- Total return $11,011 (was $11,011) ✓

Conclusion: Training is fully deterministic. BIOS upgrade did not affect numerics. Crashes were real hardware issues — now resolved.

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
- **Batch size: 16, **32**, 64** (Exp37 just completed, bs=32 best)
- BatchNorm: hurt (removes regime-scale info)
- Seeds verified: 0, 42, 99

### Recent Experiments (post-BIOS)
| # | Config | Composite | Test Sharpe | Status |
|---|--------|-----------|-------------|--------|
| 36 | VERIFY champion s=0 | +5.499 | +6.2113 | KEEP (matches Exp32) |
| 37 | bs=64 s=0 | +4.09 | +4.19 | DISCARD |

### Next Experiments (13 remaining in MLP, in priority order)

1. **warmup=3** — Goyal et al. (2017). Stabilizes early-training gradients with high lr=5e-4.
   ```bash
   cd C:/Users/evija/autoresearch && "C:/Users/evija/anaconda3/python.exe" -m autoresearch.run_autoresearch --backbone mlp --lr 5e-4 --batch-size 32 --seq-len 10 --epochs 50 --weight-decay 1e-5 --patience 10 --grad-clip 1.0 --huber-delta 0.5 --head-dropout 0.15 --warmup-epochs 3 --seed 0 --description "mlp: Exp38 warmup=3 stabilize high-LR (Goyal2017) seed=0"
   ```

2. **grad_clip=0.5** — tighter clipping for the high-LR regime
3. **lr=4e-5** — between champion 5e-4 and 3e-4
4. **lr=6e-4** — push higher
5. **huber=0.3** — even more robust to tails (fold 2 post-crash)
6. **hd=0.2 recheck at lr=5e-4** (was tested at 3e-4)
7-9. **seed sweeps on any winner** (seeds 7, 123, 2024)
10-13. If exhausted: consider architectural experiments on residual connection (e.g., 3-layer residual, attention over shortcut)

### After MLP (50): Move to LSTM (50 experiments)
Start with ablation config as baseline, then optimize per CLAUDE.md process.
