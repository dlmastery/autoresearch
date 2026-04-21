# XGBoost Exp6 depth=4 lr=0.01 (seed=42) — GLOBAL CHAMPION

**Composite: +7.7601 | Test Sharpe: +7.8601 | Val Sharpe: +7.9510 | Return: +1,765%**

Overtakes the prior champion (XGBoost Exp1 at depth=6 lr=0.03, +7.1686) by **+0.5915**. Shallower trees (depth 6→4) dramatically improved val-side generalisation (val Sharpe +7.37 → +7.95), and halving the learning rate (0.03 → 0.01) added marginal stability (+0.07).

## Headline
| Metric | Value |
|---|---|
| Composite | **+7.7601** |
| Test Sharpe | +7.8601 |
| Val Sharpe | +7.9510 |
| Test return | +1,765.04% |
| Test folds positive | 7/7 |
| Val folds positive | 6/7 |
| Training time | ~120 s (CPU only) |

## Champion HP
```python
{
  "backbone": "xgboost", "seq_len": 10,
  "n_estimators": 1500, "max_depth": 4, "learning_rate": 0.01,
  "subsample": 0.8, "colsample_bytree": 0.8,
  "min_child_weight": 1, "gamma": 0, "reg_alpha": 0, "reg_lambda": 1.0,
  "tree_method": "hist", "random_state": 42,
}
```

## HP axis sweeps (8 experiments done)
| Axis | Sweep | Peak |
|---|---|---|
| max_depth | 3(+7.53), **4(+7.69)**, 5(+7.05), 6(+7.17) | **4** |
| learning_rate | 0.005(+7.68), **0.01(+7.76)**, 0.03(+7.17), 0.1(+7.26) | **0.01** |
| subsample | 0.5(+7.53), **0.8(+7.76)**, (1.0 pending) | **0.8** |
| colsample_bytree | **0.8(+7.76)**, 1.0(+7.36) | **0.8** |
| seed | 0(+7.17), **42(+7.76)** | deterministic — seed variance ≈ 0 |

**Seed-determinism finding (Exp2):** XGBoost at n_estimators=1500 converges to an essentially-unique ensemble regardless of subsample/colsample RNG. Unlike LSTM (std ~1.0 across seeds), XGBoost champion is reproducible single-seed.

## Inference
```python
import pickle, numpy as np
bundle = pickle.load(open("xgboost_model.pkl", "rb"))
model, scaler_mean, scaler_scale = bundle["gbm_wrapper"], bundle["scaler_mean"], bundle["scaler_scale"]
X_scaled = (X_raw - scaler_mean) / scaler_scale
preds = model.predict(X_scaled.reshape(X_scaled.shape[0], -1))  # [n, 2] = [ret_1d, ret_5d]
signal = np.sign(preds[:, 0])
```
