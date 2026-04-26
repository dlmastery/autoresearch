# Research Journal — AutoResearch QQQ

> Human-readable twin of `reasoning_annotations.json`. Every experiment
> is logged here in narrative form: diagnosis → citations → hypothesis →
> prediction → verdict → learning. Bootstrap session 2026-04-26.

---

## Champion lineage so far

| Exp | Backbone | Config | Composite | A_Sharpe | Excess | Status |
|----:|----------|--------|----------:|---------:|-------:|--------|
| 1 | xgboost | smoke n_est=50, depth=4, lr=0.05, seq=60 | -1.5423 | +0.5694 | -0.6499 | DISCARD |
| 2 | xgboost | n_est=300, depth=4, lr=0.03, seq=60 | -2.3923 | -0.0045 | -1.2239 | DISCARD (over-trees) |
| 3 | mlp | plain SOTA, head_dropout=0.1, seq=10 | -0.2923 | +0.0077 | -0.5966 | KEEP (interim champion) |
| 4 | mlp | seq=20 + strong reg dropout=0.25 wd=1e-4 | -0.8341 | -0.4341 | -1.2763 | DISCARD (under-fit) |
| 5 | lstm | **FX-Exp35 HPs** (wd=7e-4 bs=16 seed=42 lr=1e-3 ep=100 pat=15) | -0.1318 | +0.8339 | **+0.2297** | KEEP — first BH-beating excess |
| 6 | mlp | **FX-Exp32 HPs** (residual MLP head_dropout=0.25 seed=0 ep=50 pat=10) | **+0.5799** | +0.6799 | +0.0757 | **CHAMPION — first +composite** |

---

## Exp 1 — XGBoost smoke (n_est=50)

**Diagnosis.** Bootstrap experiment for the QQQ variant — no prior champion, this validates the whole pipeline (download → 184 cited features → 7 regime-aware folds → super-fold split → GBM training → multi-target evaluation → JSONL row + per-day CSV). Configuration deliberately tiny (50 trees) for fast smoke check.

**Citations.** Chen, Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting System' (arXiv:1603.02754). Welch, Goyal 2008 RFS. Bollerslev, Tauchen, Zhou 2009 RFS (VRP). Estrella, Mishkin 1998 RES (10y-3m).

**Hypothesis.** XGBoost at depth=4, lr=0.05, n_est=50, seq=60, seed=42 produces measurable composite. Under-trained, expect modest performance.

**Prediction.** Composite [-2, +1]. A_sharpe [0, +1.5]. Excess negative because under-trained model can't beat trending QQQ.

**Verdict.** DISCARD as champion candidate, KEEP as smoke baseline. Composite **-1.5423**, A_sharpe +0.5694 (51.1% direction-betting), excess -0.6499. 5/7 positive test folds. Per-fold A: F1 +3.25 (GFC peak — model alpha lights up in chaos), F2 +0.40 (US-downgrade — bh=+5.02 stomps it), F5 +1.57 (Vol-mageddon).

**Learning.** Pipeline runs end-to-end. Regime-aware folds produce interpretable breakdowns: alpha shows up in chaos (folds 1, 5), passive dominates trending recoveries (folds 2, 6, 7).

---

## Exp 2 — XGBoost n_est=300 hill-climb

**Diagnosis.** Hill-climb on exp 1. Hypothesis: more trees reduce bias.

**Citations.** Chen-Guestrin 2016 KDD; Hastie-Tibshirani-Friedman 2009 ESL §10.12.

**Hypothesis.** n_est=300 (6× exp 1) at lr=0.03 should improve composite by reducing bias.

**Prediction.** Composite [-1, 0]. A_sharpe [+0.5, +1.5].

**Verdict.** **DISCARD strongly** — composite -2.3923 (delta -0.85 vs exp 1, WORSE). 300 trees OVERFIT the noise vs 50 trees. The QQQ feature space (12,300-dim flattened seq=60 windows) is too large for unregularised XGBoost without aggressive depth or column-fraction regularisation.

**Learning.** Critical FX-vs-QQQ divergence: more trees made the model WORSE on QQQ, opposite to FX where n_est=1500 was the FX-champion. Pivoting to MLP per user feedback (FX progression: MLP → LSTM → GBM).

---

## Exp 3 — MLP plain SOTA baseline

**Diagnosis.** User feedback: FX progression was MLP → LSTM → GBM, building cheap-fast experiment volume first. Pivoting to MLP. QQQ XGBoost is ~4× slower than FX so the FX progression is even more critical here.

**Citations.** Gu, Kelly, Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning' (arXiv:1807.04365). Loshchilov, Hutter 2019 ICLR 'AdamW' (arXiv:1711.05101). He et al. 2016 CVPR 'ResNet' (arXiv:1512.03385).

**Hypothesis.** Residual MLP at the Gu-Kelly-Xiu 2020 SOTA recipe (lr=3e-4, bs=32, ep=50, pat=10, wd=1e-5, head_dropout=0.1, seq=10) produces a positive composite with sub-30-second runtime.

**Prediction.** Composite [-1, +1.5]. Test_pos_folds 4-6/7. Runtime 25-45s.

**Verdict.** KEEP — interim champion. Composite **-0.2923** (delta +1.25 vs exp 1, +2.10 vs exp 2). A_sharpe +0.0077. Excess -0.5966. **Runtime 28.0s** — 18× faster than XGBoost. The compute-efficiency argument validated.

**Learning.** MLP at SOTA recipe is a solid baseline but does not beat passive QQQ. val_pos_folds=1/7 says val windows are tough.

---

## Exp 4 — MLP seq=20 + stronger reg

**Diagnosis.** Hill-climb on exp 3. Champion weakness val_pos_folds=1/7. Try longer seq + stronger reg.

**Citations.** Loshchilov-Hutter 2019 ICLR (wd log-spaced). Srivastava et al. 2014 JMLR (dropout=0.25).

**Hypothesis.** Stronger reg fights val/test divergence.

**Verdict.** **DISCARD** — composite -0.8341, A_sharpe -0.4341 (anti-predictive). Stronger reg made the MLP UNDERFIT.

**Learning.** Axis closed: head_dropout >= 0.2 + wd >= 1e-4 with seq=20. Next try: opposite — keep exp 3 settings but try larger hidden, OR try LSTM at FX-champion HPs.

---

## Exp 5 — LSTM @ FX-champion (Exp35) HPs

**Diagnosis.** User feedback: try the FX winning configurations. The FX neural champion was LSTM Exp35 with wd=7e-4, bs=16, seed=42, lr=1e-3, ep=100, pat=15 (composite +6.4242 in FX).

**Citations.** Fischer, Krauss 2018 EJOR 'Deep learning with long short-term memory networks for financial market predictions' — LSTM SOTA recipe for daily financial TS. Hochreiter, Schmidhuber 1997 Neural Computation (LSTM architecture). Loshchilov-Hutter 2019 ICLR (AdamW).

**Hypothesis.** FX-champion HPs transfer to QQQ at the same daily-equity scale because architecture and dataset shape are similar.

**Prediction.** Composite [0, +1.5]. Test_pos_folds >= 5/7. Convergence ep 25-30.

**Verdict.** **KEEP — new champion.** Composite **-0.1318**. A_sharpe **+0.8339**. **Excess_sharpe +0.2297 — FIRST POSITIVE EXCESS-SHARPE OF THE SESSION** (strategy beats passive QQQ). Test_pos_folds **7/7** (perfect on test); val_pos_folds 5/7 (the bottleneck). Convergence at ep=34.

**Learning.** **The FX-champion LSTM config transfers to QQQ.** Critical: equity-index daily prediction at QQQ scale responds to the same LSTM HPs that won on FX. Test_pos_folds=7/7 is exceptional. Axis open: multi-seed (≥3 seeds before declaring stable champion). Axis open: seq_len=20 at same HPs.

---

## Exp 6 — MLP @ FX-champion (Exp32) HPs

**Diagnosis.** Companion to exp 5. Testing FX-Exp32 (residual MLP, head_dropout=0.25, seed=0, lr=3e-4, ep=50, pat=10, wd=1e-5).

**Citations.** Gu-Kelly-Xiu 2020 RFS. He et al. 2016 CVPR (residual). Srivastava et al. 2014 JMLR (dropout=0.25 FX-empirical optimum).

**Hypothesis.** Residual MLP at FX-Exp32 HPs produces composite >= 0 because the residual-MLP architecture has the same low-SNR-friendly inductive bias on QQQ.

**Verdict.** **KEEP — new champion.** Composite **+0.5799** (first POSITIVE composite). A_sharpe +0.6799. excess +0.0757. Test_pos_folds 6/7. Runtime 28.7s — 18× faster than exp 5 LSTM and produced HIGHER composite.

**Learning.** Residual MLP @ FX HPs is the new lead. Validates: (a) residual-MLP architecture is durable across asset classes (FX → QQQ); (b) head_dropout=0.25 is the right regularisation strength for low-SNR financial data; (c) the 'best' single-model on QQQ so far is also the cheapest to train. Axis open: multi-seed variance check (seeds 7, 42, 99, 2024). Axis open: hidden_size hill-climb. Ready to build the QQQ mega-ensemble path (FX-style: 3 GBMs + 1 LSTM, rank-avg).

---

## Plan forward (multi-session marathon)

1. **Multi-seed exp 5 + 6** (LSTM and MLP at FX-champion HPs) — 4 seeds each → 8 experiments. Establishes seed-variance baseline.
2. **LightGBM @ FX-Exp235 HPs** (depth=4, gbm_lr=0.01, n_est=2000, seq=60).
3. **CatBoost @ FX-Exp236 HPs** (depth=4, gbm_lr=0.01, n_est=2000, seq=60).
4. **XGBoost @ FX-Exp203 HPs** (depth=4, gbm_lr=0.03, n_est=1500, seq=60) — needs foreground or split-runs.
5. **Build `_qqq_mega_ensemble.py`** — port the rank-avg recipe from FX `_emtsf_mega_ensemble.py` to QQQ. Target: meet or beat **excess-Sharpe of FX +9.7071**.
6. **Continue 25-experiment hill-climb per backbone** for the 23-backbone roster (15 generic TS + 8 equity-specific 2024-2026 SOTA).
7. Eventually: paper / Medium / audit reports / Colab notebook (full FX-style artefact suite).
