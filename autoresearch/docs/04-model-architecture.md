# 04 - Model Architecture

**SWEBoK Knowledge Area:** KA2 — Software Design (Detailed Design)
**Google SWE Reference:** Ch. 15 — "Deprecation" (managing model variants)

---

## 1. Architecture Overview

All models share a common interface:
- **Input:** Tensor `[batch, seq_len, n_features=104]` (neural) or flattened `[batch, seq_len*104]` (GBM). `seq_len` is per-backbone: 60 for LFM2.5, 10 for all others.
- **Output:** Dict `{"ret_1d": [batch, 6], "ret_5d": [batch, 6]}` (neural) or `[batch, 2]` (GBM)
- **Prediction targets:** Forward returns for 6 currency pairs at 1-day and 5-day horizons

## 2. Backbone Registry

The system implements 8 backbones spanning four paradigms:

### 2.1 Classical Feedforward

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **mlp** | `CurrencyMLP` | Flatten → 512 → GELU → 512 → GELU → heads | ~3.3M |

- Flattens `(B, 60, 104)` → `(B, 6240)` before FC layers
- Serves as simplest baseline; no temporal modeling
- Dropout 0.1 between layers

### 2.2 Recurrent

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **lstm** | `CurrencyLSTM` | BiLSTM(256, 2 layers) → heads | ~1.5M |

- Bidirectional: concatenates forward/backward hidden states → 512-dim
- Uses last hidden state (not full sequence) for prediction
- Dropout 0.1 between LSTM layers

### 2.3 Foundation Models (Frozen Backbone)

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **lfm2-350m** | `CurrencyLFM` | Linear(104→1024) → LFM2.5(frozen) → heads | 354.5M (mostly frozen) |
| **lfm2-1.2b** | `CurrencyLFM` | Linear(104→2048) → LFM2.5(frozen) → heads | 1.2B (mostly frozen) |

- **LFM2.5** = Liquid Foundation Model by Liquid AI (released March 2026)
- Architecture: 10 double-gated LIV convolution blocks + 6 grouped query attention blocks
- Uses `inputs_embeds` pathway (bypasses text tokenizer for numeric data)
- Projection layer maps financial features to model's hidden dimension
- `freeze_backbone=True` by default (only projection + heads are trained)
- Last hidden state → prediction heads

### 2.4 Time-Series Transformers

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **patchtst** | `CurrencyPatchTST` | PatchTST(d=256, heads=4, layers=3) → heads | ~3M |
| **patchtsmixer** | `CurrencyPatchTSMixer` | PatchTSMixer(d=256, layers=4) → heads | ~2M |
| **informer** | `CurrencyInformer` | Informer(d=256, heads=4, enc=2, dec=1) → heads | ~2.5M |

- **PatchTST** (Nie et al., ICLR 2023): Patches sequence into non-overlapping segments of length 12, applies channel-independent attention
- **PatchTSMixer** (Google, NeurIPS 2023): MLP-Mixer variant for time series; temporal mixing + channel mixing without attention
- **Informer** (Zhou et al., AAAI 2021): ProbSparse attention mechanism reduces O(n^2) to O(n log n); encoder-decoder architecture with time features

### 2.5 State-Space Models

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **mamba2** | `CurrencyMamba2` | Mamba2(d=256, layers=4, head_dim=64) → heads | ~2.5M |

- **Mamba2** (Gu & Dao, ICML 2024): Selective state-space model with hardware-aware design
- Constraint: `hidden_size * expand == num_heads * head_dim` (expand=2)
- No attention mechanism; processes sequences in linear time O(n)
- Projection: features → hidden_size before Mamba blocks

### 2.6 Gradient Boosting (Tree-Based)

| Backbone | Class | Architecture | Estimators |
|----------|-------|-------------|-----------|
| **xgboost** | `GBMWrapper` | XGBoost(n=200, depth=6, lr=0.05) | 200 trees |
| **lightgbm** | `GBMWrapper` | LightGBM(n=200, depth=6, lr=0.05) | 200 trees |
| **catboost** | `GBMWrapper` | CatBoost(n=200, depth=6, lr=0.05) | 200 iters |

- Input: flattened sliding windows `[n_samples, seq_len * n_features]`
- One estimator per target (ret_1d, ret_5d) via `MultiOutputRegressor` pattern
- GPU-accelerated where available (tree_method="hist", device="cuda")
- No sequence modeling; treats windowed features as flat feature vectors

## 3. Shared Prediction Heads

All neural models use identical prediction heads (created by `_make_heads()`):

```
Per horizon (ret_1d, ret_5d):
  LayerNorm(hidden_size)
  → Linear(hidden_size, 256)
  → GELU
  → Dropout(0.1)
  → Linear(256, N_PAIRS=6)
```

This ensures:
- **Fair comparison:** Same head capacity across all backbones
- **Isolated backbone contribution:** Performance differences come from representation quality
- **Multi-target output:** 6 currency pairs per horizon

## 4. Factory Function

```python
def create_model(
    backbone: str,
    n_input_features: int,
    seq_len: int = 60,
    freeze_backbone: bool = True,
) -> nn.Module | GBMWrapper:
```

- Routes to correct class based on `backbone` string
- Raises `ValueError` for unknown backbones
- Handles GBM vs neural branching transparently

## 5. Model Comparison Matrix

| Property | MLP | LSTM | LFM2.5 | PatchTST | Mamba2 | XGBoost |
|----------|-----|------|--------|----------|--------|---------|
| Temporal modeling | No | Yes (BiRNN) | Yes (LIV+GQA) | Yes (attention) | Yes (SSM) | No |
| Pretrained | No | No | Yes (28T tokens) | No | No | No |
| Attention mechanism | No | No | GQA (partial) | Full | No | No |
| Complexity (seq) | O(1) | O(n) | O(n) | O(n*p) | O(n) | O(1) |
| GPU memory | Low | Medium | High | Medium | Medium | Low |
| Training speed | Fast | Medium | Slow | Medium | Medium | Fast |
| Feature interaction | Dense | Sequential | Sequential | Patched | Sequential | Tree splits |

## 6. Constants

```python
N_PAIRS = 6                    # Currency pairs predicted
HORIZONS = ["ret_1d", "ret_5d"] # Prediction horizons
DEFAULT_BACKBONE = "lfm2-350m"  # Default model
GBM_BACKBONES = {"xgboost", "lightgbm", "catboost"}
```
