---
name: AutoResearch QQQ Checkpoint
description: 23 single-model exps + 1 mega-ensemble. Champion dMamba exp 17 +0.8625 (lucky seed). Next exp 24: LightGBM exp 10 multi-seed variance check (seed=0).
type: project
---

## 🏆 GLOBAL CHAMPION (exp 17 — but lucky seed)

**dMamba @ FX-Mamba-winner config (variant=dmamba, expand=4, d_state=16, n_layers=2, seq=60, lr=5e-4, bs=32, ep=100, wd=0.1, hd=0.1, warmup=10, seed=42)** — composite **+0.8625**, A_sharpe +0.8625, **7/7 positive test folds**, excess_sharpe **-0.3576** (under BH +1.22). Runtime 2473s.

**⚠️ Champion is a lucky seed.** 4-seed sweep (exps 17, 19, 20, 21):
| Seed | Composite | A_sharpe | excess |
|---:|---:|---:|---:|
| 42 (champ) | +0.8625 | +0.8625 | -0.3576 |
| 0  | +0.0169 | +0.5952 | -0.6249 |
| 99 | -0.8620 | -0.4620 | -1.6821 |
| 7  | -0.5230 | -0.1230 | -1.3431 |
| **median** | **-0.25** | **+0.24** | **-1.0** |

Archived: `winners/mamba_exp17_dmamba_e4_seed42/`.

## 🥈 MEGA-5 ENSEMBLE (rank-avg)

**LightGBM + CatBoost + XGBoost + LSTM + MLP, rank-avg of FX-winner-config predictions** — Sharpe **+0.876**, return +107.4%, win-rate 51.0%, excess **-0.343** (under BH +1.219). Best ensemble variant. Archived: `winners/ensemble_mega5_rank/`.

## Phase summary (FX-winner-config transfer phase complete)

| Phase | Exps | Champion (seed=42) | 4-seed median | Status |
|---|---:|---|---:|---|
| MLP @ FX-Exp32 HPs | exps 3,4,6,7,13,16 | exp 6 +0.5799 (seed=0) | -1.16 | DONE |
| LSTM @ FX-Exp35 HPs | exps 5,8,11,23 | exp 5 +0.83 A_sh (seed=42) | +0.11 | DONE |
| XGBoost @ FX-Exp203 HPs | exps 1,2,22 | exp 22 -1.78 (n_est=1500) | n/a | DONE |
| LightGBM @ FX-Exp235 HPs | exps 9,10,12 | **exp 10 +0.48 (seed=42)** | **UNKNOWN — needs multi-seed** | **VERIFICATION REQUIRED** |
| CatBoost @ FX-Exp236 HPs | exps 14,18 | exp 18 -0.92 (n_est=2000) | n/a | DONE |
| Mamba @ FX-Mamba HPs | exps 15,17,19,20,21 | exp 17 +0.86 (lucky) | -0.25 | DONE |
| MEGA-5 ensemble | rank-avg | +0.876 | n/a | DONE |

**Headline gap**: best raw Sharpe +0.876 vs FX target +9.7 → 8.8 Sharpe units short. All ensembles trail BH +1.219 by 0.3-0.9 excess Sharpe.

**Diagnostic conclusion (from commit 54868a1):** "Path to FX parity is QQQ-specific HP discovery, not config transfer."

## STRATEGY (updated 2026-04-27 by user direction)

**Cheap-first HP exhaustion before heavy backbones.** Per user instruction: finish 25-exp HP search on MLP (~30s/run) and LSTM (~90s/run) BEFORE re-running heavier backbones (LightGBM/CatBoost/XGBoost/Mamba). Cheap iteration loops let us cover the 25-exp mandate in ~12 min vs ~42 min per single heavy-backbone run.

**Backbone tiers by per-run cost on QQQ:**
| Tier | Backbones | Per-run | Status |
|---|---|---|---|
| Cheap | MLP, LSTM | 28-92s | **IN PROGRESS — start here** |
| Medium | XGBoost (lite), LightGBM (lite n_est=300) | 100-600s | queued |
| Heavy | LightGBM full, XGBoost full, CatBoost, Mamba/dMamba | 1000-18000s | queued (after cheap done) |

**Multi-seed lessons (already learned, applies to ALL backbones):**
- dMamba seed=42 +0.86 was lucky; 4-seed median -0.25
- MLP seed=0 +0.58 was lucky; 4-seed median -1.35
- LightGBM seed=42 +0.48 was high; 2-seed mean +0.27 (3rd seed pending — DEFERRED to medium-tier work)

## DONE THIS SESSION (2026-04-27)

- Fixed dashboard Status column (KEEP/DISCARD) — runner patch + 23-row JSONL backfill, committed `0debb50`, pushed to GH Pages.
- Exp 24 LightGBM seed=0 returned **+0.0663 composite** (seed=0 vs seed=42's +0.4825 → -0.42 delta; 2-seed LightGBM mean now +0.27). DISCARD. Verdict + learning written to reasoning_annotations.

## NEXT EXPERIMENT: #25 MLP HP hill-climb — lr 3e-4 → 1e-4

**Pre-flight rationale:** MLP @ FX-Exp32 HPs (exp 6 champion) is a lucky-seed result like dMamba (4-seed median -1.35). Most-likely-to-help single-knob change for stability: lr 3e-4 → 1e-4 (Keskar 2017 ICLR flat-minima theory; Smith 2017 cyclical LR for small-data). Lower LR statistically lands SGD in flatter minima → reduced seed-to-seed variance. Trade peak (seed=0 may drop from +0.58 to +0.45) for stability (4-seed median may rise from -1.35 toward +0.0).

**Bash command (cwd = C:/Users/evija/autoresearch, run in background):**
```bash
"C:/Users/evija/anaconda3/python.exe" -u -m autoresearchindexstock.run_autoresearch \
  --backbone mlp --seq-len 10 --lr 1e-4 --bs 32 --epochs 50 --patience 10 \
  --weight-decay 1e-5 --head-dropout 0.25 --huber-delta 1.0 --grad-clip 1.0 \
  --warmup-epochs 0 --seed 0 \
  --description "MLP @ FX-Exp32 lr=1e-4 (down from 3e-4) — flat-minima hill-climb; Keskar 2017 ICLR, Smith 2017 cyclical LR; targets 4-seed median improvement vs exp 6 seed=0 lucky"
```

**Expected runtime:** ~30-50s (slightly longer than exp 6's 28s).

**Decision rule after run:**
- composite ∈ [+0.2, +0.7]: hypothesis viable → schedule seeds 7/42/99 to confirm flatter-minima via reduced 4-seed std-dev
- composite < 0.0: lr=1e-4 too small (under-train in 50 epochs) → try lr=2e-4 next
- composite > +0.7: surprise upside — keep + multi-seed immediately

## OLD next-exp record (DEFERRED — to revisit after cheap tier complete)

#24 LightGBM exp 10 multi-seed (seed=0) — completed, +0.0663 DISCARD.
LightGBM seeds 7/99 pending, scheduled for medium-tier phase.



**Pre-flight rationale:** Same diagnostic discipline that exposed dMamba as lucky must apply to LightGBM exp 10 (the strongest non-cherry-picked single-model: +0.48 composite, A_sharpe +1.07, 6/7 folds). Before investing the remaining 25-experiment QQQ-native HP-tuning budget on LightGBM as the primary backbone, we must verify whether +0.48 is a stable expectation or another +1σ lucky-seed artefact.

**Bash command (cwd = C:/Users/evija/autoresearch, run in background):**
```bash
"C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock.run_autoresearch \
  --backbone lightgbm --seq-len 60 --max-depth 4 --gbm-lr 0.01 --n-estimators 1000 \
  --lr 3e-4 --bs 32 --epochs 50 --patience 10 --weight-decay 1e-5 \
  --head-dropout 0.1 --huber-delta 1.0 --grad-clip 1.0 --warmup-epochs 0 \
  --seed 0 \
  --description "LightGBM @ FX-Exp235 HPs seed=0 — multi-seed variance check (vs exp 10 seed=42 +0.48); same discipline that exposed dMamba lucky-seed; Lakshminarayanan 2017 NeurIPS"
```

**Expected runtime:** ~1640s = 27 min (matching exp 10).

**Decision rule after run:**
- If composite ∈ [+0.08, +0.88]: LightGBM is real → schedule seeds 7, 99 for full 4-seed median, then begin QQQ-native HP hill-climb on LightGBM
- If composite < 0.0: LightGBM was lucky too → pivot to a different baseline (MEGA-5 itself, or different feature engineering)

## Subsequent queue (if exp 24 confirms LightGBM)

- **#25** LightGBM seed=7 (3rd seed)
- **#26** LightGBM seed=99 (4th seed → median estimate locked)
- **#27** LightGBM hill-climb #1: max_depth 4 → 6 (Chen-Guestrin 2016 default; tabular ceiling) at the seed-median config
- **#28** LightGBM hill-climb #2: gbm_lr 0.01 → 0.005 + n_est 1000 → 2000 (more conservative boosting)
- ... continue 25-exp HP search per CLAUDE.md mandate

## Process debt to address (NOT blocking exp 24)

- `reasoning_annotations.json` only has entries 1-6, 24. Entries 7-23 are MISSING (CLAUDE.md violation). Backfill verdict/learning for 7-23 from JSONL between experiments — work to do during the 27-min exp 24 run.

## Hardware (mandatory)

P-cores 0,2,4,6 only via `_pin_to_safe_cores()` (parent CLAUDE.md mandate, BSOD prevention; 5 BSODs on 2026-04-19 from E-core WHEA errors). Runner pins automatically at import.

## Files in current state

- `data/download.py` — 56 tickers including ^VXN/^MOVE/SOXX/SMH/^IXIC/ARKK/IBB/AGG/BTC-USD
- `data/features.py` — 205 features, equity-native
- `data/splits.py` — 7 regime-aware folds (GFC peak / 2011 EU debt / Taper / China-oil / Vol-mageddon / COVID / AI rally)
- `evaluation/metrics.py` — composite + excess-Sharpe + multi-target eval (A/B/D)
- `run_autoresearch.py` — runner with A/B/D logging, auto-pin to P-cores
- `_qqq_mega_ensemble.py` — MEGA-5 rank-avg ensemble script
- `_sync_dashboard_to_docs.py` — copies autoresearch_results/ → docs/index_stock_dashboard/
- `autoresearch_results/experiment_log.jsonl` — 23 entries
- `autoresearch_results/best_config.json` — exp 17 dMamba champion
- `autoresearch_results/reasoning_annotations.json` — entries 1-6, 24 (gap: 7-23)
- `autoresearch_results/winners/` — `mamba_exp17_dmamba_e4_seed42/` and `ensemble_mega5_rank/`
- `docs/index_stock_dashboard/` — Pages mirror in sync (md5-verified 2026-04-27)

## Live dashboard
- Local: `python -m http.server 8888 --directory C:/Users/evija/autoresearch/autoresearchindexstock/autoresearch_results`
- Pages: <https://dlmastery.github.io/autoresearch/index_stock_dashboard/>

## Last session start (this resume)

2026-04-27 — Crash-recovery resume after 54868a1. Verified clean state (HEAD == origin/master, all docs/ md5-matched). User confirmed multi-seed-LightGBM diagnostic over dMamba hill-climb. Wrote pre-launch annotation 24, this checkpoint, ready to launch exp 24.
