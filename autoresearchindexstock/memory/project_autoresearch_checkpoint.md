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

## NEXT EXPERIMENT: #24 LightGBM exp 10 multi-seed (seed=0)

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
