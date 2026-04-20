# Example: airline-passengers time-series forecasting

Demonstrates the framework on a time-series forecasting task using the LightGBM backbone.

## What this demonstrates

- Task-type `time_series_forecasting` (different from examples 1 and 2)
- Backbone `lightgbm` (a different Tier-3 GBM from example 2's xgboost — proves each is a separate backbone)
- Split protocol `time_series_split` (expanding window with a purge gap)
- Task-specific metric bundle: RMSE + MAE + IC + hit_rate + sharpe
- Feature engineering in `build_features.py` (lag features + cyclical month encoding)

## Dataset

Synthesized 144-month series mimicking the Box-Jenkins airline-passengers benchmark
(trend + seasonality + noise). This avoids any network dependency; the feature
table is regenerated deterministically via `build_features.py`.

## Run

```bash
python run_example.py
```

Expected: RMSE around 8-14, composite around -14 to -8, KEEP status.

## Dashboard

```bash
cd autoresearch_results
python -m http.server 8765
# then visit http://localhost:8765/dashboard.html
```

## Generality audit

Three examples, three different task types, three different backbones, one core:
- Example 1: regression + MLP + kfold
- Example 2: binary_classification + xgboost + stratified_kfold
- Example 3: time_series_forecasting + lightgbm + time_series_split

Zero changes to `generalized_ml_autoresearch/core/` between any of these. The entire
configuration happens via `config.yaml` and `seed_reasoning.json` — exactly what
the Skill's 12-step wizard would generate.
