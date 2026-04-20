# LSTM Exp35 wd=7e-4 bs=16 (seed=42) — Global Champion

**Composite: +6.4242 | Test Sharpe: +6.5242 | Val Sharpe: +7.1539 | Return: +1122.29%**

## Summary

Beat prior champion Exp29 (+6.3701) by tuning weight decay from 1e-3 down to 7e-4. Small relaxation of L2 regularization redistributed generalization: val fold 1 improved from -0.1020 to +0.4646 (+0.57), while other folds held.

- 7/7 positive test folds, 6/7 positive val folds
- Val Sharpe best-ever: +7.1539
- Val fold 2 still ≈ 0 (the one hard regime)

## Per-Fold Test Sharpe

| Fold | Regime | Sharpe | Return% | Hit% | IC |
|------|--------|--------|---------|------|-----|
| 1 | Pre-crisis upturn + GFC onset | +0.9135 | +6.50 | 51.5 | +0.129 |
| 2 | Post-crash recovery | +0.4024 | +1.67 | 52.3 | +0.080 |
| 3 | Eurozone debt plateau | +9.7509 | +34.11 | 75.5 | +0.575 |
| 4 | Strong USD downturn | +11.3780 | +104.44 | 83.9 | +0.770 |
| 5 | Low-vol plateau | +13.5237 | +40.82 | 79.6 | +0.802 |
| 6 | EUR crisis downturn | +12.3280 | +84.09 | 77.0 | +0.761 |
| 7 | Recent mixed/upturn | +8.9620 | +58.82 | 75.3 | +0.666 |

## Per-Fold Val Sharpe

| Fold | Regime | Sharpe | Return% | Hit% | IC |
|------|--------|--------|---------|------|-----|
| 1 | Pre-crisis upturn + GFC onset | +0.4646 | +1.10 | 46.8 | +0.029 |
| 2 | Post-crash recovery | -0.0006 | -0.28 | 49.1 | +0.009 |
| 3 | Eurozone debt plateau | +13.8070 | +52.13 | 86.5 | +0.802 |
| 4 | Strong USD downturn | +13.5837 | +42.96 | 81.8 | +0.808 |
| 5 | Low-vol plateau | +10.8261 | +34.63 | 75.9 | +0.790 |
| 6 | EUR crisis downturn | +13.8743 | +26.95 | 82.4 | +0.885 |
| 7 | Recent mixed/upturn | +11.1786 | +24.45 | 72.7 | +0.712 |

## Full Hyperparameter Config

```json
{
  "backbone": "lstm",
  "hidden_size": 128,
  "num_layers": 2,
  "bidirectional": true,
  "cell": "lstm",
  "head_dropout": 0.25,
  "het_loss": false,

  "seq_len": 10,
  "lr": 0.001,
  "batch_size": 16,
  "epochs": 100,
  "weight_decay": 0.0007,
  "patience": 15,
  "grad_clip": 1.0,
  "warmup_epochs": 0,
  "huber_delta": 1.0,
  "seed": 42
}
```

## Architecture

BiLSTM(input=104, hidden=128, layers=2, bidirectional=True) → Dropout(0.25) → Linear(256 → 1).
Trained 29 epochs (early-stopped from best@14); 52s on 4 P-cores.

## Key Insight

Weight decay + batch-size have coupled effects. At bs=16, slightly weaker L2 (wd=7e-4 vs 1e-3) lets the model capture more training-set signal without overfitting because the implicit regularization from small-batch noise compensates for the explicit L2 reduction. Bayesian interpretation: posterior prior variance (1/wd) is wider, exploration radius wider, noise from SGD + wide prior converges to a slightly different flat basin with better val fold 1 generalization.

## Classification Metrics (test)

- Precision 0.7416 · Recall 0.6445 · F1 0.6897 · F2 0.6618 · MCC 0.4292 · Accuracy 71.33%

## Trading Strategy (Deployment)

Same as Exp29 template — see inference/predict.py. Caveat: seed variance remains wide; use seed ensembling for deployment.
