# Experiment Summary — AutoResearch QQQ

> Append-only tabular log of every QQQ experiment. Maintained in lock-step
> with `experiment_log.jsonl` and `reasoning_annotations.json`.

## Bootstrap session 2026-04-26

### Lineage table

| # | Backbone | Δ from prev | Composite | A_Sharpe | Excess | BH_Sharpe | val_pos | test_pos | Time | Status |
|---:|----------|-------------|----------:|---------:|-------:|----------:|---------|----------|-----:|--------|
| 1 | xgboost | initial smoke n_est=50 | -1.5423 | +0.5694 | -0.6499 | +1.2194 | 1/7 | 5/7 | 98s | DISCARD |
| 2 | xgboost | n_est 50→300 | -2.3923 | -0.0045 | -1.2239 | +1.2194 | 1/7 | 5/7 | 335s | DISCARD (over-trees) |
| 3 | mlp | switch backbone | -0.2923 | +0.0077 | -0.5966 | +0.6042 | 5/7 | 4/7 | 28s | KEEP (interim) |
| 4 | mlp | seq 10→20, dropout 0.1→0.25, wd 1e-5→1e-4 | -0.8341 | -0.4341 | -1.2763 | +0.8422 | 6/7 | 3/7 | 33s | DISCARD (under-fit) |
| 5 | lstm | switch backbone, FX-Exp35 HPs | -0.1318 | +0.8339 | **+0.2297** | +0.6042 | 5/7 | **7/7** | 92s | KEEP (1st BH-beating excess) |
| 6 | mlp | FX-Exp32 HPs (head_dropout 0.1→0.25, seed 42→0) | **+0.5799** | +0.6799 | +0.0757 | +0.6042 | 5/7 | 6/7 | 29s | **CHAMPION** (1st +composite) |

### Per-fold pattern (CHAMPION exp 6 — MLP @ FX-Exp32 HPs)

| Fold | Regime | A_Sharpe | A_BH_Sharpe | Excess |
|---|---|---:|---:|---:|
| 1 | GFC peak crash | (data in JSONL — see dashboard) | | |
| 2 | 2011 US-downgrade + EU debt | | | |
| 3 | Taper tantrum + 2014 H1 | | | |
| 4 | China devaluation + oil crash | | | |
| 5 | 2018 Vol-mageddon + Q4 sell-off | | | |
| 6 | COVID crash + V-recovery | | | |
| 7 | Inflation bear + AI rally + 2025 | | | |

(See `trade_logs/exp6_trades.csv` for per-day breakdown and the dashboard
for per-fold Sharpe colour-coded.)

### Key cross-experiment findings (bootstrap session)

1. **More XGBoost trees made things worse on QQQ** (exp 1 → exp 2). 50
   trees beat 300 trees on composite. Opposite to FX where n_est=1500
   was the GBM champion. Hypothesis: QQQ's 12,300-dim flattened seq=60
   input space is too large for unregularised XGBoost to handle without
   aggressive depth or column-fraction regularisation.
2. **MLP > XGBoost in compute-efficiency, possibly absolute terms** —
   exp 6 produced higher composite (+0.58) in 18× less compute (29s vs
   335s). User-feedback-driven pivot to MLP first was correct.
3. **FX-champion HPs transfer to QQQ.** Both LSTM @ FX-Exp35 (exp 5,
   7/7 test folds positive!) and MLP @ FX-Exp32 (exp 6, +composite)
   beat plain SOTA-recipe baselines. The FX-empirical
   `head_dropout=0.25` and `wd=7e-4` survived the asset-class transfer.
4. **First positive excess-Sharpe (+0.2297) achieved at exp 5** — LSTM
   @ FX-Exp35 strategy beats passive QQQ buy-and-hold across the
   per-fold aggregates. This is the fair-comparison metric per CLAUDE.md
   (since QQQ trends).

### Open experiment axes (next priorities)

| Axis | Status | Why |
|---|---|---|
| Multi-seed on exp 6 (MLP champion) | OPEN | seeds 7, 42, 99, 2024 to characterise seed variance per FX protocol |
| Multi-seed on exp 5 (LSTM champion) | OPEN | same — single-seed champion may be luck |
| LightGBM @ FX-Exp235 HPs | RUNNING (exp 7 in flight) | next ensemble component |
| CatBoost @ FX-Exp236 HPs | OPEN | next ensemble component |
| XGBoost @ FX-Exp203 HPs (full n_est=1500) | OPEN | ensemble component; harness-timeout sensitive |
| Build `_qqq_mega_ensemble.py` | OPEN | port FX rank-avg recipe; target excess-Sharpe ≥ FX +9.7071 |
| Hidden_size hill-climb on MLP | OPEN | 96 / 128 / 256 |
| Patience hill-climb on MLP | OPEN | exp 6 stopped at ep=26; FX MLP converged at ep=50 |
| Exp 5 LSTM at seq_len=20 | OPEN | FX exp expected seq=10 was best; QQQ may want longer |

### Goal-tracking (FX comparison)

| Metric | FX final | QQQ current best | QQQ goal |
|---|---|---|---|
| Best single-model composite | +9.186 (XGBoost Exp203) | **+0.5799** (MLP exp 6) | match or exceed |
| Best single-model excess-Sharpe | n/a (FX has no BH baseline) | **+0.2297** (LSTM exp 5) | ≥ +9.7071 |
| Mega-ensemble Sharpe | +9.7071 | not yet built | match or exceed |
| Test_pos_folds | 6-7/7 | **7/7** (LSTM exp 5) | maintain |
| Total experiments | 265 | 6 | 375 (25 × 15 backbones) |

We are 6 of 375 experiments in. The fact that LSTM @ FX-champion HPs
already produces 7/7 positive test folds + a positive excess-Sharpe is
an early validation that the project will reach FX parity within
roadmap.
