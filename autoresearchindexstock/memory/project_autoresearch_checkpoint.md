---
name: AutoResearch QQQ Checkpoint
description: 6 experiments. Champion MLP @ FX-Exp32 HPs composite +0.5799. LSTM @ FX-Exp35 first BH-beating excess +0.2297.
type: project
---

## 🏆 GLOBAL CHAMPION (exp 6)

**MLP @ FX-Exp32 HPs (residual MLP, head_dropout=0.25, seed=0)** —
composite **+0.5799**, A_sharpe +0.6799, excess +0.0757, BH +0.6042,
test_pos_folds 6/7, val_pos_folds 5/7, **runtime 28.7s**.

CLI:
```bash
"C:/Users/evija/anaconda3/python.exe" -u -m autoresearchindexstock.run_autoresearch \
  --backbone mlp --seq-len 10 --lr 3e-4 --bs 32 --epochs 50 --patience 10 \
  --weight-decay 1e-5 --head-dropout 0.25 --seed 0 \
  --description "MLP @ FX champion HPs (Exp32) — Gu-Kelly-Xiu 2020 RFS"
```

## 📈 Most strategy-vs-passive performance (exp 5)

**LSTM @ FX-Exp35 HPs** — composite -0.1318, A_sharpe **+0.8339**,
**excess_sharpe +0.2297** (first BH-beating excess of the session),
test_pos_folds **7/7**, val_pos_folds 5/7, runtime 91.8s.

## Phase summary

| Phase | Exps | Champion | Status |
|---|---:|---|---|
| MLP | 3 (exps 3, 4, 6) | exp 6 (FX-Exp32 HPs) +0.5799 | OPEN — multi-seed + hidden hill-climb queued |
| LSTM | 1 (exp 5) | exp 5 (FX-Exp35 HPs) excess +0.23 | OPEN — multi-seed needed |
| XGBoost | 2 (exps 1, 2) | exp 1 (smoke) -1.5423 | DEFERRED — harness-timeout, foreground re-run pending |
| LightGBM | 0 (exp 7 attempted in background, reaped) | — | OPEN — needs foreground run |
| CatBoost | 0 | — | OPEN |
| Mamba/xLSTM/iTransformer/PatchTST/TSMixer/TimesNet/DLinear/N-BEATS/N-HiTS/TFT | 0 | — | QUEUED |
| Tier-1.5 — **needs PyTorch implementation before runnable**: |||
| - StockMixer (Ye 2024 AAAI 2401.05917) | 0 | — | NEEDS IMPL — MLP-mixer industry × style × temporal |
| - MASTER (Li 2024 AAAI 2312.15235) | 0 | — | NEEDS IMPL — market-guided transformer |
| - CARD (Wang 2024 ICLR 2305.12095) | 0 | — | NEEDS IMPL — channel-aligned blend transformer |
| - Crossformer (Zhang-Yan 2023 ICLR) | 0 | — | NEEDS IMPL — cross-dim attention |
| - PatchMixer (Cong 2024 KDD 2310.00655) | 0 | — | NEEDS IMPL — patches + MLP-mixing |
| - Reversible Mixer (Sun 2024 NeurIPS) | 0 | — | NEEDS IMPL — reversible long-seq |
| - Adv-ALSTM (Feng 2019 IJCAI) | 0 | — | NEEDS IMPL — adversarial robust LSTM |
| - StockNet (Xu-Cohen 2018 ACL) | 0 | — | NEEDS IMPL — equity-prediction baseline |
| Mega-ensemble (phase b) | n/a | — | BLOCKED on completing the 4 ensemble components |

## Next experiments (priority queue)

1. **Multi-seed exp 6** — MLP @ FX-Exp32 HPs across seeds [7, 42, 99, 2024] to characterise seed variance (4 experiments).
2. **Multi-seed exp 5** — LSTM @ FX-Exp35 HPs across seeds [0, 7, 99, 2024] (4 experiments).
3. **LightGBM @ FX-Exp235 HPs** (depth=4, gbm_lr=0.01, n_est=2000, seq=60) — must run foreground (or split smaller batches).
4. **CatBoost @ FX-Exp236 HPs** (depth=4, gbm_lr=0.01, n_est=2000, seq=60).
5. **XGBoost @ FX-Exp203 HPs** (depth=4, gbm_lr=0.03, n_est=1500, seq=60) — needs foreground.
6. **Build `_qqq_mega_ensemble.py`** — port FX rank-avg recipe.
7. Continue 25-experiment hill-climb per backbone.

## Lessons learned this session

1. **Use `--lr` (not `--learning-rate`).** Common mistake; wasted runs.
2. **More XGBoost trees made things WORSE on QQQ.** 50 trees beat 300 trees on composite. Opposite of FX. Diagnosis: 12,300-dim flattened seq=60 input space too large for unregularised XGBoost.
3. **MLP > XGBoost in compute-efficiency on QQQ.** 18× faster + higher composite.
4. **FX-champion HPs transfer to QQQ.** Both LSTM @ FX-Exp35 (7/7 test folds) and MLP @ FX-Exp32 (+composite) beat plain SOTA-recipe baselines. `head_dropout=0.25` and `wd=7e-4` empirical FX optima survive the asset transfer.
5. **First positive excess-Sharpe (+0.2297) achieved at exp 5.** LSTM @ FX-Exp35 strategy beats passive QQQ.
6. **Bash background tasks have a 2-10 min harness timeout.** Long-running experiments must run foreground.
7. **Target D vol-adjusted returns can be < -1.** Strat realisation must use UNSCALED 1d returns + safety clip in `evaluate_target_variant`.
8. **BTC-USD outer-join inflates rows by ~30% via weekend dates.** Reindex to NYSE business days post-concat.
9. **Late-starting tickers must be auto-dropped** or `dropna()` eats 2007-2018 history.

## Files in current state

- `autoresearchindexstock/CLAUDE.md` (753 lines, self-contained, audit complete)
- `data/download.py` — 56 tickers including ^VXN/^MOVE/SOXX/SMH/^IXIC/ARKK/IBB/AGG/BTC-USD
- `data/features.py` — 205 features, equity-native
- `data/splits.py` — 7 regime-aware folds (GFC peak / 2011 EU debt / Taper / China-oil / Vol-mageddon / COVID / AI rally)
- `evaluation/metrics.py` — composite + excess-Sharpe + multi-target eval
- `run_autoresearch.py` — runner with A/B/D logging
- `autoresearch_results/experiment_log.jsonl` — 6 entries
- `autoresearch_results/best_config.json` — exp 6 champion
- `autoresearch_results/reasoning_annotations.json` — full 6-experiment annotations
- `autoresearch_results/research_journal.md` — narrative
- `autoresearch_results/experiment_summary.md` — tabular log
- `autoresearch_results/dashboard.html` — A/B/D selector wired
- `autoresearch_results/trade_logs/` — 6 per-experiment CSVs + summaries + manifest

## Hardware

P-cores 0,2,4,6 only (parent CLAUDE.md mandate, BSOD prevention).
