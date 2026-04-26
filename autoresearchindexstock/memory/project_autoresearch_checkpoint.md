---
name: AutoResearch QQQ Checkpoint
description: 0 exps. Bootstrap session 2026-04-26. Pipeline scaffold complete; first wave queued.
type: project
---

## Status

**BOOTSTRAP — 0 experiments run.** Infrastructure built this session,
first experiment queued for verification.

## Project shape

- Asset: **QQQ** (Nasdaq-100 ETF)
- Window: 2004-01-01 → **2025-12-31** (no 2026 data; hard cap in
  `data/download.py`)
- Optimisation target: **target A — `fwd_ret_1d`** (1-day forward log
  return). Composite = `min(test_A_sharpe, val_A_sharpe) - 0.1 *
  n_negative_folds`.
- Track + plot: A (1d), B (5d), C (sign concordance), D (vol-adjusted 1d).
- Splits: 7-fold walk-forward, last test 2025-05 → 2025-12.
- Backbones: 15-roster mirror of FX, **25 hill-climb experiments per
  backbone**.
- Goal: meet or beat FX mega-ensemble Sharpe **+9.7071** on
  excess-Sharpe (strategy − buy-and-hold).

## Backbone queue (priority order)

1. xgboost      — pending
2. lightgbm     — pending
3. catboost     — pending
4. lstm         — pending
5. mlp          — pending
6. mamba        — pending
7. xlstm        — pending
8. itransformer — pending
9. patchtst     — pending
10. patchtsmixer — pending
11. timesnet    — pending
12. dlinear     — pending
13. nbeats      — pending
14. nhits       — pending
15. tft         — pending

## Next experiment (verify pipeline)

```bash
cd C:/Users/evija/autoresearch
"C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock.run_autoresearch \
  --backbone xgboost --max-depth 4 --gbm-lr 0.03 --n-estimators 1500 \
  --seq-len 60 --seed 42 \
  --description "xgboost: SOTA baseline (Chen & Guestrin 2016 KDD), bootstrap"
```

**Rationale**: XGBoost was the FX single-model champion (+9.186 composite).
Reproducing that recipe on QQQ first gives an apples-to-apples baseline
and a sanity check that the entire pipeline (download → features → splits
→ train → eval → log) is wired correctly. If composite ≥ 0 with sane
fold-level Sharpe, pipeline works; we then iterate the 25-exp hill-climb.

## Files / scaffolding (complete)

- `autoresearchindexstock/CLAUDE.md` (extends parent)
- `autoresearchindexstock/data/download.py` — QQQ + ~30 cross-asset signals
- `autoresearchindexstock/data/features.py` — ~120 equity-native features
- `autoresearchindexstock/data/splits.py` — 7 regime-labelled folds
- `autoresearchindexstock/evaluation/metrics.py` — composite + excess
- `autoresearchindexstock/run_autoresearch.py` — runner (multi-target eval)
- `autoresearchindexstock/_sync_dashboard_to_docs.py` — Pages mirror
- `autoresearchindexstock/autoresearch_results/dashboard.html` — to be
  adapted with A/B/C/D plotting
- `autoresearchindexstock/memory/project_autoresearch_checkpoint.md`
  (this file)

## Files / scaffolding (TODO before first experiment)

- [ ] Adapt `dashboard.html` title + add A/B/C/D plot selector
- [ ] Smoke-test `python -m autoresearchindexstock.run_autoresearch ...`
- [ ] First sync to `docs/index_stock_dashboard/`
- [ ] Commit + push

## Hardware

Same as parent — P-cores 0,2,4,6 only. `_pin_to_safe_cores()` imported
from FX runner.
