# Inference Guide — Residual MLP Champion (Exp32)

## Quick Start

```bash
# From the winner directory
cd autoresearch_results/winners/mlp_exp32_residual_seed0/inference

# Predict last 5 days
python predict.py --days 5

# Predict last 20 days with 50 MC samples
python predict.py --days 20 --mc-samples 50

# Use a specific checkpoint
python predict.py --checkpoint /path/to/model_checkpoint.pt
```

## Requirements

```
torch>=2.5.0
yfinance
pandas
numpy
```

## Output Format

Each prediction includes:

| Field | Description |
|-------|-------------|
| `prediction` | Raw predicted return (mean of MC samples) |
| `direction` | LONG (+1) or SHORT (-1) |
| `confidence` | 0-1 score (sigmoid of negative log uncertainty) |
| `aleatoric` | Data noise uncertainty |
| `epistemic` | Model uncertainty (from MC Dropout) |
| `1σ band` | 68% prediction interval [lower, upper] |
| `2σ band` | 95% prediction interval [lower, upper] |

## Interpreting Uncertainty

- **High confidence (>0.95):** Strong signal, model is certain. Use full position size.
- **Medium confidence (0.8-0.95):** Decent signal, some noise. Reduce position.
- **Low confidence (<0.8):** Noisy regime. Consider skipping this trade.
- **High epistemic:** Model hasn't seen enough data like this. Be cautious.
- **High aleatoric:** Inherently noisy market conditions. Expected in crisis periods.

## Using for Options / Trading Bands

The prediction bands can be used for:
1. **Strike selection:** 1σ band → ATM options, 2σ band → OTM strikes
2. **Stop loss:** Place stops at 2σ below predicted direction
3. **Position sizing:** Kelly criterion with confidence as edge estimate
4. **Regime detection:** High aleatoric = volatile regime → widen stops

## Python API

```python
from predict import ResidualMLP, predict_with_uncertainty
import torch
import numpy as np

# Load model
model = ResidualMLP()
state = torch.load("../model_checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)

# Prepare input: (batch, seq_len=10, n_features=104)
x = torch.randn(1, 10, 104)  # replace with real features

# Get predictions with uncertainty
results = predict_with_uncertainty(model, x, n_mc=20)
print(f"Prediction: {results['prediction'][0]:.6f}")
print(f"Direction:  {'LONG' if results['direction'][0] > 0 else 'SHORT'}")
print(f"Confidence: {results['confidence'][0]:.4f}")
```
