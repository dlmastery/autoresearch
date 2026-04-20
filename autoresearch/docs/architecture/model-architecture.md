# 04 - Model Architecture

**SWEBoK Knowledge Area:** KA2 -- Software Design (Detailed Design)
**Google SWE Reference:** Ch. 15 -- "Deprecation" (managing model variants)

---

## Key Highlights

- **8 backbone architectures** spanning 4 paradigms: feedforward, recurrent, transformer/mixer, and gradient boosting
- **Champion: Residual MLP** with 300,812 parameters -- shortcut + residual branch with GELU activation, 128 hidden units, 64 head hidden units
- **All neural models share identical prediction heads** via `_make_heads()` for fair comparison
- **Registry + factory pattern** enables adding new backbones without modifying calling code
- **Key finding:** Simple residual MLP (301K params) outperforms 354M-param LFM2.5 foundation model -- simplicity wins for low-SNR financial data

---

## 1. Architecture Overview

All models share a common interface:
- **Input:** Tensor `[batch, seq_len, n_features=104]` (neural) or flattened `[batch, seq_len*104]` (GBM). `seq_len` is per-backbone: 60 for LFM2.5, 10 for all others.
- **Output:** Dict `{"ret_1d": [batch, 6], "ret_5d": [batch, 6]}` (neural) or `[batch, 2]` (GBM)
- **Prediction targets:** Forward returns for 6 currency pairs at 1-day and 5-day horizons

### 1.1 Architecture Decision Tree

```
Is the backbone a gradient boosting model?
  |
  ├── YES -> GBMWrapper (XGBoost / LightGBM / CatBoost)
  |          Input: flattened [batch, seq_len * 104]
  |          Output: [batch, 2] (ret_1d, ret_5d)
  |          Training: sklearn-compatible .fit() / .predict()
  |
  └── NO -> nn.Module subclass
             Input: [batch, seq_len, 104]
             Output: {"ret_1d": [batch, 6], "ret_5d": [batch, 6]}
             Training: PyTorch gradient-based (AdamW, Huber loss)
             |
             ├── CurrencyMLP     (flatten -> residual branch -> heads)
             ├── CurrencyLSTM    (BiLSTM -> last hidden -> heads)
             ├── CurrencyLFM     (projection -> frozen LFM2.5 -> heads)
             ├── CurrencyPatchTST (patching -> attention -> heads)
             └── CurrencyPatchTSMixer (patching -> MLP-mixer -> heads)
```

## 2. Backbone Registry

The system implements 8 backbones spanning four paradigms:

### 2.1 Residual MLP (Champion)

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **mlp** | `CurrencyMLP` | Flatten -> Shortcut + Residual -> heads | **300,812** |

The champion model. A residual MLP inspired by He et al. (2016) that learns a correction to a linear baseline. This architecture is particularly well-suited to low-SNR financial data where the signal is a small perturbation on near-random walk dynamics.

```
                        Input: [batch, 10, 104]
                              |
                         Flatten to [batch, 1040]
                              |
                    ┌─────────┴─────────┐
                    |                   |
              ┌─────▼─────┐      ┌──────▼──────┐
              |  SHORTCUT  |      |  RESIDUAL   |
              |            |      |             |
              | Linear     |      | Linear      |
              | 1040->128  |      | 1040->128   |
              |            |      | GELU        |
              |            |      | Dropout(0.1)|
              |            |      | Linear      |
              |            |      | 128->128    |
              |            |      | GELU        |
              |            |      | Dropout(0.1)|
              └─────┬─────┘      └──────┬──────┘
                    |                   |
                    └─────────┬─────────┘
                          ADD (+)
                              |
                         hidden [batch, 128]
                              |
                    ┌─────────┴─────────┐
                    |                   |
              ┌─────▼─────┐      ┌─────▼─────┐
              | HEAD: 1d   |      | HEAD: 5d   |
              | LN(128)    |      | LN(128)    |
              | Linear     |      | Linear     |
              | 128->64    |      | 128->64    |
              | GELU       |      | GELU       |
              | Dropout    |      | Dropout    |
              | (0.15)     |      | (0.15)     |
              | Linear     |      | Linear     |
              | 64->6      |      | 64->6      |
              └─────┬─────┘      └─────┬─────┘
                    |                   |
              ret_1d [B,6]        ret_5d [B,6]
```

**Why residual architecture for FX?**
- The shortcut path provides a linear baseline that captures simple linear relationships between features and returns
- The residual branch learns nonlinear corrections to this baseline
- In low-SNR regimes (like FX), the signal IS a small correction to a linear model -- the residual architecture makes this explicit
- Gradient flow through the shortcut prevents vanishing gradients during training
- With only 128 hidden units and 301K parameters, the model has strong inductive bias against overfitting

**Champion configuration (Experiment 88, verified in Experiment 90):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_size` | 128 | Sweet spot: 64 too small (underfits), 256 too large (overfits on 2478 training samples) |
| `head_hidden` | 64 | min(hidden_size, 64) -- smaller heads prevent overfitting in prediction layer |
| `head_dropout` | 0.15 | Slightly above default 0.1; regularizes prediction heads |
| `seq_len` | 10 | 10 business days (~2 weeks); longer windows add noise for MLP (no temporal modeling) |
| `het_loss` | False | Heteroscedastic loss adds optimization complexity without benefit for MLP |
| `lr` | 5e-4 | Higher than default 3e-4; compensates for small model capacity |
| `epochs` | 50 | Longer training with patience=10 early stopping |
| `huber_delta` | 0.5 | Robust to fat-tailed returns; delta=0.5 balances MSE and MAE behavior |
| `seed` | 0 | Reproducible results |

**Parameter breakdown (300,812 total):**

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Shortcut (Linear 1040->128) | 133,248 | 44.3% |
| Residual branch (2x Linear) | 149,760 | 49.8% |
| Prediction heads (2x head) | 17,804 | 5.9% |
| **Total** | **300,812** | **100%** |

### 2.2 Recurrent

| Backbone | Class | Architecture | Parameters |
|----------|-------|-------------|-----------|
| **lstm** | `CurrencyLSTM` | BiLSTM(128, 2 layers) -> heads | **770,572** |

- Bidirectional: concatenates forward/backward hidden states -> 256-dim
- Uses last hidden state (not full sequence) for prediction
- Dropout 0.1 between LSTM layers
- 2.6x more parameters than champion MLP but underperforms on test Sharpe

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
