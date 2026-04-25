# Winner Archive - Exp 6 XGBoost (FDB fraudecom champion)

## Summary

XGBoost binary classifier with 18 velocity features on chronological 80/20 holdout.
Test AUC = 0.5414, Val AUC = 0.5403, Composite = 0.5403 (KEEP under floor=0.50).

## vs FDB published baselines

| System | AUC | Δ vs us |
|--------|-----|---------|
| AFD-TFI (proprietary) | 0.636 | -0.095 |
| **Our XGBoost (Exp 6)** | **0.5414** | — |
| AutoGluon | 0.522 | +0.019 |
| H2O | 0.518 | +0.023 |
| Auto-sklearn | 0.515 | +0.026 |

## Files

- `config.json` — exact hyperparameter config
- `model_checkpoint.pt` — pickled XGBoost model + scaler params + feature columns
- `experiment_log_entry.json` — full JSONL entry
- `code/` — frozen source snapshot (runner.py, gbm.py, splits.py, metrics.py, prepare_data.py, add_velocity_features.py)
- `inference/predict.py` — standalone inference script
- `audit_report.md` — 14-section explainability + risk audit
- `colab_train_and_infer.ipynb` — self-contained Colab notebook
- `reproduction/` — re-run logs

## Reproducing

```bash
cd ../../  # back to fraud_ecommerce/
python prepare_data.py
python add_velocity_features.py
python run_example.py
```

Expected exit: composite 0.5403 +/- 0.005 (seed variance).
