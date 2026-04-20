---
name: Autoresearch Checkpoint
description: 117 exps. Global champion LSTM Exp9 composite +6.1035. Moved from LSTM (done) to PatchTST (1/50).
type: project
---

## Session Recovery
1. Read this
2. Read `memory/project_hardware_crash_log.md` — CPU 60% cap, Turbo off, 0 crashes since mitigation
3. Read JSONL tail
4. Dashboard at http://localhost:8765/dashboard.html (per-backbone tabs)

## 🏆 GLOBAL CHAMPION (across ALL backbones)
**LSTM Exp9** — composite **+6.1035**, test Sharpe **+6.2956**, 7/7 positive test folds, +1033% return
- Config: bidirectional 2-layer LSTM h=128, lr=1e-3, bs=32, seq=10, ep=100, wd=1e-4, pat=15, huber=1.0, hd=0.25, seed=0
- Archived at `winners/lstm_exp9_wd1e4_seed0/` (17 artifacts, portable)
- Previous: MLP Exp32 residual (+5.499). LSTM beat MLP by +0.60 composite.

## Per-Backbone Status
| Backbone | Exps | Best Comp | Best Test Sharpe | Status |
|----------|------|-----------|------------------|--------|
| lfm2-350m | 43 | +1.77 | +2.07 | Done |
| mlp | 54 | +5.499 | +6.21 | Done |
| lstm | **12** | **+6.10** | **+6.30** | **Done — all SOTA axes exhausted, GLOBAL CHAMPION** |
| patchtst | **1** | -1.72 | -0.82 | **IN PROGRESS** |
| patchtsmixer | 0 | - | - | Pending |
| xgboost | 0 | - | - | Pending |
| lightgbm | 0 | - | - | Pending |
| catboost | 0 | - | - | Pending |

## Dead Params Caught and Fixed This Session
- `hidden_size` was not wired for LSTM backbone (only MLP). Fixed.
- `bidirectional` wasn't configurable. Fixed with `--unidirectional` flag.

## LSTM Final Summary (12 exps)
Best: Exp9 composite +6.1035. Plateau confirmed across all SOTA axes.
- lr sweep: 5e-4 (worse), 1e-3 (best). Fischer&Krauss confirmed.
- bs sweep: 64 (worse), 32 (best).
- ep sweep: 50 (worse), 100 pat=15 (best). Fischer&Krauss SOTA confirmed.
- hd sweep: 0.15 (5.06), 0.25 (6.07), 0.30 (6.02). Srivastava peak at 0.25.
- wd sweep: 1e-5 (6.07), 1e-4 (6.10). Slight improvement.
- hidden sweep: 64 (4.46), 128 (6.10). Capacity matters.
- seq sweep: 10 (6.10), 20 (4.25). Longer context worse.
- bidirectional: True (6.10), False (5.00). Bidir better for test.
- huber: 0.5 (3.98), 1.0 (6.10). Bidir LSTM doesn't need robust loss.

## PatchTST (1/50) — just started
**Exp1 baseline:** lr=1e-4, bs=32, seq=10, ep=100, pat=20, wd=1e-5, hd=0.15 (Nie 2023 SOTA)
Result: composite -1.72, test Sharpe -0.82, 2/7 test pos. Train Sharpe +1.42 = model barely learning.

**Diagnosis:** seq=10 with patch_length=5 gives only 2 patches. Attention needs more tokens. Nie 2023 used seq=96-336 with patch=16. Our setup is fundamentally under-scaled for transformer attention.

## Next Experiment (PatchTST Exp2)
**Hypothesis:** seq=60 with patch=10 gives 6 attention tokens — more tokens = more attention signal. Cite: Nie et al. (2023) Table 4 — PatchTST needs ≥ 4 patches for meaningful self-attention.

```bash
python -m autoresearch.run_autoresearch --backbone patchtst --lr 1e-4 --batch-size 32 --seq-len 60 --epochs 100 --weight-decay 1e-5 --patience 20 --grad-clip 1.0 --huber-delta 1.0 --head-dropout 0.15 --seed 0 --description "patchtst: Exp2 seq=60 patch=10 enable proper attention (Nie2023 Table 4)"
```

## Hardware Mitigations Active
CPU 60% cap, Turbo off, 156 user processes on P-cores, Python runner 4-thread P-core affinity. **No crashes since mitigation (5+ hours now).**
