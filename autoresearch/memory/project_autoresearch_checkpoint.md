---
name: Autoresearch Checkpoint
description: 99 exps. Champion Exp41 +5.499/+6.2113 (VERIFIED POST-5th-CRASH). HW cap 60%, turbo off, P-cores only.
type: project
---

## 2026-04-19 5th-CRASH MITIGATION APPLIED
New mitigations this session (after 5 BSODs today, i9-14900HX degraded):
1. CPU max freq **capped at 60%** (`powercfg PROCTHROTTLEMAX=60`) — base 2200MHz, max ~1320MHz
2. CPU min freq floor at 30%
3. Intel **Turbo Boost DISABLED** (`powercfg PERFBOOSTMODE=0`)
4. **156 user processes pinned to P-cores** (affinity mask 0xFFFF, logical 0-15)
5. HWiNFO64 installed for real-time monitoring

Previous mitigations (still active):
1. `run_autoresearch.py`: `_pin_to_safe_cores()` pins Python to 4 P-core threads [0,2,4,6]
2. E-cores APIC 16,17,24,25 banned (WHEA parity errors)
3. Runner env override: `AUTORESEARCH_USE_ALL_CORES=1` or `AUTORESEARCH_N_THREADS=N`

## Champion VERIFIED at 60% CPU cap (2026-04-19 17:30)
Reproduced deterministically seed=0 CPU-only 60% cap → **composite +5.4990 exactly, test Sharpe +6.2113 exactly**. 52s training. No crash.


## Session Recovery
1. Read this checkpoint
2. Verify JSONL tail (97 entries) + best_config.json
3. Start dashboard: `"C:/Users/evija/anaconda3/python.exe" -m http.server 8765 --directory C:/Users/evija/autoresearch/autoresearch/autoresearch_results`
4. Dashboard at http://localhost:8765/dashboard.html
5. Resume from next experiment below

## HARDWARE CRASH DIAGNOSIS (2026-04-19)
**NOT a software bug. CPU hardware instability.** Windows Event Log shows:

**Today (2026-04-19) — 4 BSODs:**
- 14:45 — 0x0000007f UNEXPECTED_KERNEL_MODE_TRAP
- 15:54 — 0x000001ca SYNTHETIC_WATCHDOG_TIMEOUT
- 16:06 — 0x0000001e KMODE_EXCEPTION_NOT_HANDLED
- 17:08 — 0x00000101 CLOCK_WATCHDOG_TIMEOUT

Different bugchecks + no common pattern = hallmark of hardware instability.

**2026-04-15 — WHEA CPU Corrected Machine Check errors:**
- Internal parity errors on cores APIC 16, 17, 24, 25
- TLB errors on same cores

**Root cause:** BIOS update reset voltage/C-state settings. Bad cores (16,17,24,25) fail under sustained compute.

### User action items (hardware):
1. BIOS: disable C-states, set Intel SVID to "Fail Safe", check power limits
2. Reseat RAM (SODIMM)
3. Run MemTest86 overnight
4. Update Intel chipset + MEI drivers
5. Roll back BIOS if recent version unstable

### Software mitigation (until fixed):
1. **Force CPU-only** — set `CUDA_VISIBLE_DEVICES=""` before runs
2. MLP trains in 15s CPU — doesn't need GPU
3. Already saving model after every experiment (good)
4. Checkpoint after every experiment (ALWAYS)

## CURRENT CHAMPION: Exp41 (Residual MLP)

**Config:**
- Architecture: Residual MLP (shortcut + 2-layer, hidden=128, head=64)
- lr=5e-4, bs=32, seq=10, ep=50
- wd=1e-5, pat=10, gc=1.0
- huber=0.5, hd=0.15
- seed=0, het_loss=False

**Composite +5.50 | Test Sharpe +6.21 | 7/7 positive test folds | Return +1001%**

### Per-fold test (champion)
| Fold | Period | Regime | Sharpe | IC | WR |
|------|--------|--------|--------|-----|-----|
| 1 | 2006-08 | Pre-crisis/GFC | +2.46 | +0.19 | 60% |
| 2 | 2009-10 | Post-crash | +0.44 | +0.08 | 51% |
| 3 | 2011-12 | EZ debt | +9.76 | +0.58 | 75% |
| 4 | 2014-16 | Strong USD | +9.78 | +0.67 | 75% |
| 5 | 2017-19 | Low-vol | +8.85 | +0.64 | 71% |
| 6 | 2020-21 | COVID | +10.22 | +0.64 | 72% |
| 7 | 2023-24 | Recent | +8.33 | +0.62 | 71% |

### Cross-seed verified (median test Sharpe +4.76)
| Seed | Composite | Test Sharpe |
|------|-----------|-------------|
| 0 | +5.50 | +6.21 |
| 42 | +4.45 | +4.69 |
| 99 | +4.46 | +4.76 |

## Experiment History
- **LFM2**: 50 exps (median test Sharpe +1.40, best +2.07)
- **MLP**: 47 exps (champion +6.21)

### Key Architectural Discoveries
1. Residual skip (He 2016): +0.82 → +4.24 Sharpe
2. Higher LR 5e-4 enabled by skip: +4.24 → +5.23
3. hd=0.15: +5.23 → +6.21
4. huber=0.5: regime-balancing (helps fold 2)
5. MLP 512h→128h: essential (Gu Kelly Xiu 2020)
6. 50 epochs for from-scratch training
7. Heteroscedastic loss HURT on n=2738 — disabled
8. BatchNorm HURT (removes regime-scale info)

### Exhausted MLP Axes
- Architecture: plain → **residual skip**
- Hidden: 512, **128**
- LR: 3e-4, **5e-4**, 7e-4
- Epochs: 20, **50**, 100
- Head dropout: 0.1, **0.15**, 0.2
- Huber delta: **0.5**, 1.0
- Seq len: **10**, 20
- Batch size: 16, **32**, 64
- Warmup: 0, **3** (Exp38 small improvement +4.89)
- Seeds verified: 0, 42, 99

## Next Experiments (continue to 50 MLP total)
Currently at 47 MLP exps. 3 remaining before LSTM.

**STEP 1: Re-verify champion post-crash (CPU-only)**
```bash
cd C:/Users/evija/autoresearch && CUDA_VISIBLE_DEVICES="" "C:/Users/evija/anaconda3/python.exe" -m autoresearch.run_autoresearch --backbone mlp --lr 5e-4 --batch-size 32 --seq-len 10 --epochs 50 --weight-decay 1e-5 --patience 10 --grad-clip 1.0 --huber-delta 0.5 --head-dropout 0.15 --seed 0 --description "mlp: Exp42 POST-CRASH CPU verify champion Exp41 (must match)"
```

**STEP 2-3: Final MLP explorations before LSTM**
- grad_clip=0.5 (tighter clipping for lr=5e-4)
- lr=4e-4 (between 3e-4 and 5e-4)
