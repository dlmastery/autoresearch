---
name: Autoresearch Checkpoint
description: 108 exps. NEW GLOBAL CHAMPION: LSTM Exp4 composite +6.0725 test Sharpe +6.2282 (beat MLP +5.499).
type: project
---

## Session Recovery
1. Read this
2. Read `memory/project_hardware_crash_log.md` (CPU 60% cap, Turbo off, P-core only, no crashes since mitigation)
3. Read JSONL tail
4. Dashboard at http://localhost:8765/dashboard.html (now has per-backbone tabs)

## 🏆 GLOBAL CHAMPION (across ALL backbones)
**LSTM Exp4** (108 in JSONL) — composite **+6.0725**, test Sharpe **+6.2282**, 7/7 positive test folds
- Config: lr=1e-3, bs=32, seq=10, ep=100, wd=1e-5, pat=15, huber=1.0, hd=0.25, seed=0, het_loss=False
- Architecture: Bidirectional 2-layer LSTM, hidden=128
- Early-stopped at epoch 30
- +1007% return on held-out test (1000 → 11074)
- Archived at `winners/lstm_exp4_hd025_seed0/` (all 17 artifacts)

Previous champion (now 2nd): MLP residual Exp42, composite +5.499.

## Per-Backbone Status
| Backbone | Exps | Best Comp | Best Test Sharpe | Status |
|----------|------|-----------|------------------|--------|
| lfm2-350m | 43 | +1.77 | +2.07 | Done |
| mlp | 54 | +5.499 | +6.21 | Done |
| lstm | **4** | **+6.07** | **+6.23** | **IN PROGRESS (4/50) — GLOBAL WINNER** |
| patchtst | 0 | - | - | Pending |
| patchtsmixer | 0 | - | - | Pending |
| xgboost | 0 | - | - | Pending |
| lightgbm | 0 | - | - | Pending |
| catboost | 0 | - | - | Pending |

## LSTM Experiment Summary
| # | Change | Composite | Test Sharpe | Key learning |
|---|--------|-----------|-------------|--------------|
| 1 | SOTA baseline lr=1e-3 ep=50 pat=10 | +4.12 | +4.32 | Good baseline |
| 2 | huber=0.5 | +3.98 | +4.18 | DISCARD — LSTM doesn't respond to huber delta like MLP |
| 3 | ep=100 pat=15 (Fischer&Krauss SOTA) | +5.06 | +5.81 | KEEP — more epochs + patience helps |
| **4** | **hd=0.25** | **+6.07** | **+6.23** | **GLOBAL CHAMPION — head dropout 0.15→0.25 fixed fold 2** |

## Next LSTM Experiment (5/50)
**Hypothesis:** hd=0.25 was the key. Try hd=0.30 — maybe even more dropout helps. Or try hidden_size=64 (half) for more regularization by capacity reduction per Gu et al. 2020.

### Option A (explore dropout direction further):
```bash
python -m autoresearch.run_autoresearch --backbone lstm --lr 1e-3 --batch-size 32 --seq-len 10 --epochs 100 --weight-decay 1e-5 --patience 15 --grad-clip 1.0 --huber-delta 1.0 --head-dropout 0.30 --seed 0 --description "lstm: Exp5 hd=0.30 push dropout further"
```

### Option B (capacity reduction):
```bash
python -m autoresearch.run_autoresearch --backbone lstm --lr 1e-3 --batch-size 32 --seq-len 10 --epochs 100 --weight-decay 1e-5 --patience 15 --grad-clip 1.0 --huber-delta 1.0 --head-dropout 0.25 --hidden-size 64 --seed 0 --description "lstm: Exp6 hidden=64 Gu2020 SOTA LSTM capacity"
```

Try A first — smaller perturbation from new champion.

## Hardware Mitigations Active
- CPU max 60%, min 30%, Turbo OFF
- 156 user processes pinned to P-cores (0xFFFF)
- Python runner `_pin_to_safe_cores()` active (4 threads)
- **0 crashes since mitigation (17:30 today) — even with GPU enabled in Exp2**
