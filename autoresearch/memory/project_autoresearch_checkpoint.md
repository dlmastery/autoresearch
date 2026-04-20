---
name: Autoresearch Checkpoint
description: 152 exps. GLOBAL CHAMPION LSTM Exp35 composite +6.4242. LSTM halted 46/50 per user. NOW IN MAMBA PHASE 1/50 (Exp152 +5.27 val fold 2 breakthrough).
type: project
---

## PHASE TRANSITION 2026-04-19
- **LSTM phase halted** at 46/50 per user instruction ("enough lstm - move on to mamba")
- Champion preserved: LSTM Exp35 (bs=16 wd=7e-4 seed=42) composite +6.4242
- **Mamba phase STARTED** — Exp1/50 complete, exceeded first-experiment prediction

## Mamba Phase — Experiments so far (6/50)

| # | Variant | d_state | Composite | Test Sharpe | Notes |
|---|---------|---------|-----------|-------------|-------|
| 1 | vanilla | 16 | +5.2714 | +5.37 | Val fold 2 +1.37 — SSM breakthrough |
| 2 | s_mamba (placeholder) | 16 | +5.2714 | +5.37 | NULL; my no-op bug |
| 3 | s_mamba (real) | 16 | +5.1861 | +5.29 | variate-axis hurt; closed |
| **4** | **dmamba** | **16** | **+5.3641** | **+5.56** | **MAMBA FAMILY CHAMP** — test fold 2 lifted |
| 5 | dmamba | 32 | +4.1995 | +4.50 | over-capacity; closed |
| 6 | dmamba | 8 | +4.5319 | +4.63 | under-capacity; closed |

**d_state axis fully explored: 16 confirmed optimal**

**Next axes:**
- expand ∈ {1, 4} (Exp7, Exp8)
- num_layers ∈ {1, 3} (Exp9, Exp10)
- bs ∈ {16, 64} (Exp11, Exp12) — try Keskar 2017 trick that helped LSTM
- lr ∈ {1e-3, 3e-4} (Exp13, Exp14)
- multi-seed variance on dmamba (Exp15-19)
- ensemble (vanilla, dmamba) (Exp20+)

## Checkpoint Discipline (user-requested 2026-04-19 session)
- **Checkpoint every 10 minutes minimum** during active work
- Every experiment auto-triggers checkpoint update (experiments are 6-7 min each)
- Before/after every code change, update this file with a note on what changed
- Runner auto-writes experiment_log.jsonl + trade_logs + reasoning_annotations on each run

## Next Mamba Experiments (in order)
1. **Exp3 (JSONL 154): re-run s_mamba** now that _forward_s_mamba has real variate-axis scan
2. **Exp4: dmamba** — trend+seasonal decomposition (arXiv:2602.09081) — already implemented
3. **Exp5: vanilla d_state=32** — double state capacity
4. **Exp6: vanilla d_state=8** — half state capacity (regularise)
5. **Exp7: vanilla expand=1** — minimal inner dim
6. **Exp8: vanilla expand=4** — larger inner dim

## Next Mamba Experiments (planned)
- **Exp2**: `--mamba-variant s_mamba` — channel-token inversion (Liu 2024 arXiv:2403.11144)
- **Exp3**: `--mamba-variant dmamba` — trend+seasonal decomposition (arXiv:2602.09081)
- **Exp4**: `--mamba-variant mambats` — LTSF-tuned (Cai et al. 2024 arXiv:2405.16440)
- **Exp5**: d_state=32 (2x state capacity)
- **Exp6**: d_state=8 (½ state capacity — regularise)
- **Exp7**: expand=1 (minimal inner dim)
- **Exp8**: expand=4 (larger inner dim)
- **Exp9-10**: num_layers ∈ {1, 3}
- **Exp11-15**: HP tuning around winner (lr, wd, bs, head_dropout, warmup)
- **Exp16-20**: multi-seed variance on best variant
- **Exp21-40**: cross-variant combinations + hyperparameter refinement
- **Exp41-50**: LSTM+Mamba ensemble experiments (they are clearly complementary on fold 2)

## LEGACY (unchanged below)


## Session Recovery
1. Read this
2. Read `memory/project_hardware_crash_log.md` — CPU 60% cap, Turbo off, 0 crashes since mitigation
3. Read JSONL tail (147 entries)
4. Dashboard: http://localhost:8765/dashboard.html (per-backbone tabs + reasoning panel)

## 🏆 GLOBAL CHAMPION
**LSTM Exp35 (wd=7e-4 bs=16 seed=42)** — composite **+6.4242** | test Sharpe **+6.5242** | val Sharpe **+7.1539** | 7/7 positive test | +1122% return
- Config: BiLSTM h=128, 2-layer, lr=1e-3, bs=16, seq=10, ep=100, wd=7e-4, pat=15, hd=0.25, huber=1.0, seed=42
- Archived `winners/lstm_exp35_wd7e4_bs16_seed42/`
- Prior champions: Exp29 (bs=16, +6.37), Exp24 (seed=42, +6.36), Exp21 (wd=1e-3, +6.19), Exp20 (wd=5e-4, +6.13), Exp9 (+6.10), Exp4 (+6.07), MLP residual (+5.50)

## Per-Backbone Status
| Backbone | Exps | Best Comp | Best Test Sharpe | Status |
|----------|------|-----------|------------------|--------|
| ~~lfm2-350m~~ | 43 | +1.77 | +2.07 | **SKIPPED per user 2026-04-19** — 43 exps frozen, not extended to 50 |
| mlp | 54 | +5.499 | +6.21 | done |
| **lstm** | **44** | **+6.4242** | **+6.5242** | **IN PROGRESS (44/50) — GLOBAL CHAMP** |
| patchtst | 1 | -1.72 | -0.82 | pending (49/50) |
| patchtsmixer | 0 | — | — | pending |
| xgboost | 0 | — | — | pending |
| lightgbm | 0 | — | — | pending |
| catboost | 0 | — | — | pending |

## LSTM Experiment Summary (44 so far)
| LSTM # | Change | Composite | Learning |
|-----|--------|-----------|----------|
| 1 | SOTA baseline | +4.12 | baseline |
| 2 | huber=0.5 | +3.98 | huber doesn't help |
| 3 | ep=100 pat=15 | +5.06 | SOTA epochs help |
| 4 | hd=0.25 | +6.07 | CHAMP — head dropout breakthrough |
| 5 | hd=0.30 | +6.02 | 0.25 peaks |
| 6 | hidden=256 (dead bug) | +6.07 | wiring bug, fixed |
| 7 | wd=1e-4 | +6.10 | CHAMP |
| 8 | lr=5e-4 | +4.95 | flat minima hurt test |
| 9 | unidirectional | +5.00 | val/test split |
| 10 | seq=20 | +4.25 | too long |
| 11 | 3-layer stacked | +1.64 | depth hurts |
| 12 | GRU cell | +4.59 | LSTM better |
| 13 | LayerNorm input | +4.51 | double-norm bad |
| 14 | seq=5 | +5.70 | peak drops |
| 15 | warmup=3 | +4.37 | warmup hurts |
| 16 | hd=0.20 | +5.53 | peak drops |
| 17 | grad_clip=0.5 | +5.46 | tighter hurts |
| 18 | wd=5e-4 | +6.13 | CHAMP |
| 19 | wd=1e-3 seed=0 | +6.19 | CHAMP |
| 20 | wd=2e-3 | +5.96 | peak reached |
| 21 | lr=1.5e-3 | +5.55 | too fast |
| 22 | seed=42 variance | +6.36 | CHAMP — seed matters |
| 23 | seed=99 | +6.24 | near champ |
| 24 | seed=7 | +5.17 | wide variance |
| 25 | grad_clip=2.0 (xLSTM) | +6.33 | near miss |
| 26 | hidden=256 (Gu 2020) | +4.27 | overfits |
| 27 | bs=16 seed=42 (Keskar 2017) | +6.37 | CHAMP (bs axis) |
| 28 | bs=8 seed=42 | +5.84 | too small |
| 29 | bs=16 seed=0 | +4.24 | seed-dependent |
| 30 | bs=16 seed=99 | +5.44 | seed-dependent |
| 31 | bs=24 midpoint | +6.00 | robust but lower peak |
| 32 | het_loss at champ (Kendall-Gal) | +6.12 | fold 2 BIG gain +2.31, val1 hurt |
| 33 | wd=7e-4 seed=42 | **+6.42** | **CHAMP — current** |
| 34 | wd=8e-4 (AdamW inert) | +6.42 | identical (decoupled wd) |
| 35 | hd=0.22 | +5.68 | peak drops |
| 36 | lr=8e-4 | +5.20 | too slow |
| 37 | num_layers=1 | +3.57 | underfit |
| 38 | hidden=96 | +4.05 | underfit |
| 39 | seq=12 | +4.35 | slightly too long |
| 40 | grad_clip=1.5 | +5.97 | peak at 1.0 |
| 41 | huber=1.5 (inert) | +6.42 | identical (Huber unused at our scale) |
| 42 | seed=2024 champ var | +6.01 | variance wide |

*(Note: LSTM# 33 = JSONL Exp134 = Exp35 in this session's experiment naming.)*

## Code Changes This Session
- CurrencyLSTM: `num_layers`, `bidirectional`, `cell` (lstm/gru), `input_layernorm`, `hidden_size` parameters
- Runner: `--num-layers`, `--rnn-cell`, `--unidirectional`, `--input-layernorm`, `--hidden-size`, `--seed`, `--het-loss` flags
- Runner auto-writes `reasoning_annotations.json` per experiment (dashboard feed)
- best_config.json tracks GLOBAL champion (not per-backbone)

## Seed Variance at Champion Config (wd=7e-4 bs=16)
| Seed | Composite | Test Sharpe |
|------|-----------|-------------|
| 42 | +6.42 | +6.52 |
| 2024 | +6.01 | +6.11 |
| 0 | +4.24 (bs=16 wd=1e-3 approx) | +4.54 |
| 99 | +5.44 (bs=16 wd=1e-3 approx) | +5.54 |

Mean ≈ 5.5, std ≈ 1.0. Single-seed champions are lucky; deployment requires seed ensembling.

## Next 6 LSTM Experiments to Reach 50/50
Per CLAUDE.md 50-mandate. All use champion base: BiLSTM h=128 2L bidir, bs=16, seq=10, lr=1e-3, wd=7e-4, hd=0.25, pat=15, ep=100.

- **LSTM #43 / JSONL Exp148**: `--seed 13` variance
- **LSTM #44 / Exp149**: `--seed 77` variance
- **LSTM #45 / Exp150**: `--seed 123` variance
- **LSTM #46 / Exp151**: `--seed 2026` variance
- **LSTM #47 / Exp152**: champion + huber=0.8 (unexplored narrow)
- **LSTM #48 / Exp153**: champion + cosine no-restart (already using cosine; try constant lr via `--warmup-epochs -1` if wired — skip if not)
- **LSTM #49-50**: hand-pick after variance set to decide if ensemble needed

## After LSTM 50 → Move to PatchTST
Reset to seq_len=60 (SOTA recommended per Nie et al. 2023). Our first PatchTST experiment used seq=10 (in our runner default), giving composite −1.72. Redo with seq=60, patch_length=12, stride=6.

## Next Experiment Command
```bash
cd C:/Users/evija/autoresearch && "C:/Users/evija/anaconda3/python.exe" -m autoresearch.run_autoresearch --backbone lstm --epochs 100 --patience 15 --batch-size 16 --seq-len 10 --lr 1e-3 --weight-decay 7e-4 --head-dropout 0.25 --huber-delta 1.0 --seed 13 --description "lstm: Exp45 champion seed=13 variance"
```
