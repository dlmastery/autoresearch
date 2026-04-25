# Crash-Recovery Checkpoint - FDB fraudecom

_Last update: 2026-04-25T12:22:15_

## Current champion
- **Exp:** 18 (xgboost)
- **Test AUC:** 0.6938
- **Val AUC:** 0.8040
- **Composite:** 0.6938
- **Description:** STRICT Exp 18 — XGBoost walk-forward 4-fold CV (Bergmeir 2018)

## Last experiment
- **Exp:** 22 (contrastive_simclr_tabular)
- **Result:** Test AUC 0.5390 | Val AUC 0.5324
- **Status:** KEEP
- **Verdict:** DISCARD - composite=0.5324, test_auc=0.5390, val_auc=0.5324. Within predicted range. Contrastive pre-training came VERY CLOSE to XGBoost (0.5390 vs 0.5414, delta -0.0024) and was the best-performing o
- **Learning:** Axis open: contrastive learning is the most promising NOVEL approach found - it nearly matches XGBoost on a fundamentally different inductive bias (instance-level invariance). Mental model update: tab

## Experiment history (Exps in live log)

| Exp | Backbone | Test AUC | Val AUC | Status | One-line |
|-----|----------|----------|---------|--------|----------|
| 2 | xgboost | 0.5098 | 0.5291 | KEEP | XGBoost — chronological 80/20 holdout (FDB-compa |
| 3 | xgboost | 0.5116 | 0.5134 | KEEP | XGBoost + cyclical temporal features (hour, dow, |
| 4 | xgboost | 0.4960 | 0.5054 | DISCARD | Exp 4 — XGBoost chronological holdout WITHOUT th |
| 5 | xgboost | 0.5297 | 0.9988 | KEEP | Exp 5 — XGBoost + velocity/frequency features (A |
| 6 | xgboost | 0.5414 | 0.5403 | KEEP | Exp 6 — XGBoost + velocity features (LEAKAGE FIX |
| 7 | lightgbm | 0.5305 | 0.5413 | KEEP | Exp 7 — LightGBM (Ke 2017 NeurIPS) chronological |
| 8 | catboost | 0.5245 | 0.5400 | KEEP | Exp 8 — CatBoost (Prokhorenkova 2018 NeurIPS) ch |
| 9 | mlp | 0.4883 | 0.4775 | DISCARD | Exp 9 — MLP (Gu/Kelly/Xiu 2020) chronological +  |
| 10 | xgboost | 0.5432 | 0.5334 | KEEP | STRICT Exp 10 — XGBoost scale_pos_weight=50 (ext |
| 11 | xgboost | 0.5337 | 0.5416 | KEEP | STRICT Exp 11 — XGBoost shallow trees (max_depth |
| 12 | lightgbm | 0.5305 | 0.5413 | KEEP | STRICT Exp 12 — LightGBM (Ke 2017) leaf-wise vs  |
| 13 | catboost | 0.5245 | 0.5400 | KEEP | STRICT Exp 13 — CatBoost (Prokhorenkova 2018) or |
| 14 | ensemble | 0.5286 | 0.5286 | KEEP | STRICT Exp 14 — Ensemble (3-GBM mean) of Exps 6, |
| 15 | xgboost | 0.5242 | 0.5232 | KEEP | STRICT Exp 15 — XGBoost + interaction features ( |
| 16 | xgboost | 0.5239 | 0.5232 | KEEP | STRICT Exp 16 — XGBoost on CURATED 9-feature int |
| 17 | xgboost | 0.5294 | 0.5354 | KEEP | STRICT Exp 17 — XGBoost + 50/50 undersampled tra |
| 18 | xgboost | 0.6938 | 0.8040 | KEEP | STRICT Exp 18 — XGBoost walk-forward 4-fold CV ( |
| 19 | xgboost | 0.5283 | 0.5389 | KEEP | STRICT Exp 19 — XGBoost with min_train_idx=60000 |
| 20 | energy_based_model | 0.5214 | 0.4751 | DISCARD | STRICT Exp 20 - Energy-Based Classifier (Liu 202 |
| 21 | autoencoder_anomaly | 0.4985 | 0.5324 | DISCARD | STRICT Exp 21 - Autoencoder anomaly (Sakurada Ya |
| 22 | contrastive_simclr_tabular | 0.5390 | 0.5324 | KEEP | STRICT Exp 22 - Contrastive Learning (SimCLR-tab |

## Quarantined experiments
- `_quarantined_exp1/`: Exp 1 (stratified CV, methodologically invalid for time-ordered data).
- `_quarantined_blind_sweep/`: Exps 10-44 (old numbering) - blind grid sweep violating Research-Driven Experiment Selection rule.
- `_quarantined_reward_hack/`: Exps 19-23 (old numbering) - REWARD HACKING (changed test set size).

## Next experiment
After Exp 22 (contrastive at test_auc=0.5390, very close to XGBoost), the next principled
experiment is an ENSEMBLE of XGBoost (Exp 6) + Contrastive (Exp 22) predictions to test if
their decorrelated errors yield additive AUC gain. Expected lift: 0.005 to 0.015.

```bash
python generalized_ml_autoresearch/examples/fraud_ecommerce/run_ensemble_xgb_contrastive.py
```

## Session-start instructions
1. Read this checkpoint (you are here).
2. Read `examples/fraud_ecommerce/CLAUDE.md` for project rules.
3. Read tail of `autoresearch_results/experiment_log.jsonl` (last 3 entries).
4. Resume the 7-step research-driven loop from `Next experiment` above.
5. Start dashboard: `python -m http.server 8765 --directory examples/fraud_ecommerce/autoresearch_results`
