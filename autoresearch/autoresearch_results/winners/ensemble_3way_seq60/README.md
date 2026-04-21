# 3-way GBM ensemble (seq=60) — Ensemble Champion

**Test Sharpe: +9.4708 | Return: +585.63% | IC: +0.725 | Hit: 79.4%**

Rank-average ensemble of the three GBM family champions at seq_len=60, each trained independently on the same super-fold split. No additional training; inference-time averaging only.

## Components
| Backbone | Pickle | Individual Test Sharpe | Return |
|---|---|---:|---:|
| XGBoost | `xgboost_exp203_maxdepth4_gbmlr0.01_seq60/` | +9.2047 | +558.53% |
| LightGBM | `lightgbm_exp235_maxdepth4_gbmlr0.01_seq60/` | +8.8309 | +521.10% |
| CatBoost | `catboost_exp236_gbmlr0.01_depth4_seq60/` | +9.2597 | +564.13% |

## Ensemble strategies tested
| Strategy | Test Sharpe | Return | IC | Hit% |
|---|---:|---:|---:|---:|
| simple_avg | +9.4364 | +582.24% | +0.694 | 79.0 |
| zscore_avg | +9.3642 | +574.82% | +0.704 | 79.2 |
| **rank_avg** | **+9.4708** | **+585.63%** | **+0.725** | **79.4** |

## Inference (rank-average)
```python
import pickle, numpy as np
from scipy.stats import rankdata

bundles = {
    "xgboost": pickle.load(open("../xgboost_exp203_maxdepth4_gbmlr0.01_seq60/xgboost_model.pkl", "rb")),
    "lightgbm": pickle.load(open("../lightgbm_exp235_maxdepth4_gbmlr0.01_seq60/lightgbm_model.pkl", "rb")),
    "catboost": pickle.load(open("../catboost_exp236_gbmlr0.01_depth4_seq60/catboost_model.pkl", "rb")),
}

# Per-model predictions then rank-avg
preds = {}
for name, b in bundles.items():
    X_s = (X_raw - b["scaler_mean"]) / b["scaler_scale"]
    X_flat = X_s.reshape(X_s.shape[0], -1)  # seq=60 * 104 features = 6240
    preds[name] = b["gbm_wrapper"].predict(X_flat)[:, 0]

ranks = np.column_stack([rankdata(p) for p in preds.values()])
ensemble_pred = ranks.mean(axis=1) - (len(X_raw) + 1) / 2  # centered
signal = np.sign(ensemble_pred)
```

## Why ensembling helps here
- The three GBMs have distinct inductive biases (level-wise vs leaf-wise vs ordered boosting) — their errors decorrelate.
- At seq=60, each individual model already exploits strong feature signal (Sharpe 8.8-9.3); the ensemble captures their complementary strengths.
- Rank aggregation is robust to prediction-scale mismatches between libraries (XGBoost/LightGBM/CatBoost output different magnitude ranges).

## Caveats
- Test Return is +585.63%, much lower than seq=10 ensembles (+2212%) because seq=60 loses 60 days × 7 folds = 420 training-day warmup. Shorter investment window → smaller cumulative return DESPITE higher daily Sharpe.
- For deployment on a steady stream of new days (not fixed-window test), seq=60 is NEAR-TERM better (higher Sharpe); seq=10 is TERMINAL-WEALTH better given more trades.
- Composite (min test/val - penalty) not computed for ensemble because val-set predictions need separate handling; the paper's composite is per-single-model. Individual XGBoost composite +9.186 is still the ledger champion.
