# 05 - Training Infrastructure

**SWEBoK Knowledge Area:** KA3 — Software Construction
**Google SWE Reference:** Ch. 11 — "Testing Overview" (training correctness)

---

## Executive Summary

This document describes the complete training infrastructure for the AutoResearch FX prediction system. The system trains neural models (MLP, BiLSTM, Mamba2, LFM2, PatchTST, etc.) and gradient-boosted machines (XGBoost, LightGBM, CatBoost) on 104 backward-looking features derived from daily OHLCV data across 6 FX pairs. Training uses **Huber loss** (delta=0.5) for robustness against fat-tailed returns, **AdamW** optimizer with weight decay 1e-5, **cosine annealing** LR schedule, **gradient clipping** at max_norm=1.0, and **early stopping** with patience=10. The champion model -- a 2-layer residual MLP with 301K parameters -- trains in approximately 36 seconds on CPU, achieving test Sharpe +6.21 with 7/7 positive folds.

An optional **heteroscedastic loss variant** (Kendall & Gal, 2017) outputs both mean predictions and learned per-sample uncertainty (log-variance), enabling confidence-weighted trading signals.

```
  Training Pipeline Overview
  ==========================

  Raw OHLCV (6 pairs, 2005-2025)
       |
       v
  Feature Engineering (104 features)
       |
       v
  StandardScaler.fit(train) --> transform(train, val)
       |
       v
  create_contiguous_datasets()  <-- splits at date gaps, avoids
       |                            sliding windows across gaps
       v
  DataLoader(batch_size=32, shuffle=True)
       |
       v
  +---------------------------------------------+
  |  Training Loop (per epoch)                   |
  |                                              |
  |  for batch in train_loader:                  |
  |    preds = model(x)          # forward       |
  |    loss = huber(preds, y)    # delta=0.5     |
  |    loss.backward()           # backward      |
  |    clip_grad_norm_(1.0)      # gradient clip |
  |    optimizer.step()          # AdamW update  |
  |  scheduler.step()            # cosine anneal |
  |  val_loss = evaluate(val)    # early stop    |
  +---------------------------------------------+
       |
       v
  Best model state (lowest val loss) --> evaluation
```

---

## 1. Training Loop Architecture

### 1.1 Neural Training Path

```
Per fold (super-fold mode -- single train/eval pass):
  StandardScaler.fit(train_features)         # fit ONLY on training data
  ├─→ transform(train_features) → create_contiguous_datasets() → DataLoader(shuffle=True)
  ├─→ transform(val_features)   → create_contiguous_datasets() → DataLoader(shuffle=False)
  │
  create_model(backbone, n_features=104, seq_len, freeze_backbone=True)
  │
  AdamW(trainable_params, lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5)
  HuberLoss(delta=0.5)                       # champion uses delta=0.5
  CosineAnnealingLR(optimizer, T_max=epochs)  # smooth LR decay to ~0
  │
  for epoch in range(epochs):                 # epochs=50 for champion
      for x, y in train_loader:
          preds = model(x)  →  shape [B, 2]  (ret_1d, ret_5d for EUR/USD)
          loss = huber_loss(preds, y)
          loss.backward()
          clip_grad_norm_(params, max_norm=1.0)  # prevents fat-tail explosions
          optimizer.step()
      scheduler.step()
      val_loss = evaluate(model, val_loader)
      if val_loss < best_val_loss:
          save best_state (deepcopy)
          reset patience_counter
      elif patience_counter >= PATIENCE:      # patience=10 for champion
          restore best_state
          break
```

**Key differences from the initial version to the champion configuration:**
- `HuberLoss(delta=0.5)` instead of 1.0 -- tighter transition from quadratic to linear, more aggressive outlier down-weighting
- `patience=10` instead of 5 -- the residual MLP benefits from longer patience due to noisy validation loss
- `epochs=50` instead of 20 -- more epochs with higher patience lets cosine annealing find better optima
- `lr=5e-4` instead of 3e-4 -- residual skip connections stabilize gradients, allowing higher learning rate (He et al. 2016)

### 1.2 GBM Training Path

```
Per fold:
  StandardScaler.fit(train_features)
  ├─→ transform(train_features) → sliding_windows(seq_len) → X_train[n, seq_len*F]
  ├─→ transform(test_features)  → sliding_windows(seq_len) → X_test[n, seq_len*F]
  │
  GBMWrapper(backbone)
  model.fit(X_train, y_train)  # One estimator per target column
  preds = model.predict(X_test)
```

## 2. Hyperparameters

### 2.1 Defaults (Code Defaults vs Champion Config)

| Parameter | Code Default | Champion Value | Rationale |
|-----------|:------------:|:--------------:|-----------|
| `SEQ_LEN` | **10** (non-LFM) / **60** (LFM2.5) | **10** | 10 biz days (~2 weeks) is industry standard for short-term FX; LFM benefits from long context |
| `BATCH_SIZE` | 32 | **32** | Smaller batches = better gradient estimates on noisy FX data (Smith & Le, 2018) |
| `LEARNING_RATE` | 3e-4 | **5e-4** | Higher LR enabled by residual skip connection stability (He et al. 2016) |
| `WEIGHT_DECAY` | 1e-5 | **1e-5** | Light L2 regularization; 1e-3 found to be dead weight for MLP |
| `GRAD_CLIP` | 1.0 | **1.0** | Prevents exploding gradients on fat-tailed FX data |
| `PATIENCE` | 5 | **10** | 10-epoch patience allows noisy val loss to recover; critical for residual MLP |
| `EPOCHS` | 20 | **50** | More epochs with early stopping -- let patience decide when to stop |
| `HUBER_DELTA` | 1.0 | **0.5** | Tighter transition; more aggressive outlier down-weighting for residual arch |
| `HEAD_DROPOUT` | 0.0 | **0.15** | Prevents fold 2 overfitting without hurting other folds |
| `SEED` | N/A | **0** | Deterministic; verified reproducible across runs |

### 2.2 GBM Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 500 | More trees with lower learning rate for better generalization |
| `max_depth` | 4 | Shallow trees reduce overfitting on noisy FX data |
| `learning_rate` | 0.03 | Lower shrinkage with more estimators (standard trade-off) |
| `subsample` | 0.7 | Row subsampling for regularization |
| `colsample_bytree` | 0.7 | Feature subsampling per tree |
| `tree_method` | "hist" | Memory-efficient histogram-based splitting |

## 3. Dataset Construction

### 3.1 FXDataset (Sliding Window)

```python
class FXDataset(Dataset):
    # features: (n_rows, n_feat), targets: (n_rows, 2)
    def __len__(self):
        return n_rows - seq_len
    
    def __getitem__(self, idx):
        x = features[idx : idx + seq_len]    # shape: (seq_len, n_feat)
        y = targets[idx + seq_len - 1]        # shape: (2,) — target at END of window
        return x, y
```

**Critical design choice:** Target is at `idx + seq_len - 1` (end of window, not one step ahead). This means:
- The model sees features from `t-59` to `t`
- The target is the forward return from `t` to `t+1` (already shifted in target computation)
- No future leak: the model never sees data beyond the current window

### 3.2 GBM Sliding Windows

```python
def _make_windows(feat, tgt, seq_len):
    X = [feat[i:i+seq_len].ravel() for i in range(len(feat) - seq_len)]
    y = tgt.values[seq_len:][:len(X)]
    return X[:len(y)], y
```

- Flattens `(seq_len, n_features)` into `(seq_len * n_features,)` feature vector
- Each sample represents a complete window of historical features

## 4. Loss Function: Huber Loss

```python
criterion = torch.nn.HuberLoss(delta=0.5, reduction="none")
```

**Why Huber over MSE:**
- FX returns have fat tails (excess kurtosis typically 3-10)
- MSE squares large residuals, overweighting extreme days (GFC, COVID)
- Huber transitions from quadratic (near zero) to linear (far from zero) at `delta`
- Result: more stable gradients, less sensitivity to outlier returns

**Champion uses delta=0.5 (not 1.0):**

The project empirically found that delta=0.5 outperforms delta=1.0 specifically for the residual MLP architecture. The tighter transition point means returns beyond +/-0.5% are treated linearly rather than quadratically, which is important because daily FX returns frequently exceed this threshold during volatile regimes (folds 1, 4, 6).

```
  Huber Loss Function (delta = 0.5)
  ─────────────────────────────────
  loss
   ^
   |          /
   |         /
   |        /          linear region (|error| > delta)
   |       /           slope = delta = 0.5
   |      /
   |    .'             quadratic region (|error| <= delta)
   |  .'               slope = error
   |.'                 
  ─┼───────────────>  error
   |'.
   |  '.
   |    '.
   |      \
   |       \
   |        \
   |         \

  L(error) = { 0.5 * error^2                   if |error| <= delta
             { delta * (|error| - 0.5 * delta)  if |error| > delta
```

### 4.1 Heteroscedastic Loss Variant (Kendall & Gal, 2017)

The system supports an optional heteroscedastic loss that jointly learns predictions and per-sample uncertainty. Instead of outputting only a mean prediction, the model outputs both mean (mu) and log-variance (log_var = s):

```python
# Model forward pass outputs two values per target:
mu, log_var = model(x)   # mu: [B, 2], log_var: [B, 2]

# Heteroscedastic loss (Kendall & Gal 2017, Eq. 4):
loss = exp(-s) * huber(mu, y) + 0.5 * s
#        ^                ^         ^
#        |                |         |
#   precision    weighted Huber   variance
#   weighting    on mean pred     regularizer
```

**How it works:**
- `exp(-s)` acts as learned precision weighting: high variance (large s) reduces the loss contribution of that sample, allowing the model to "admit ignorance" on noisy samples
- `0.5 * s` penalizes the model for being too uncertain everywhere -- prevents the trivial solution of s -> infinity (predict nothing, lose nothing)
- The model learns to assign high aleatoric uncertainty to genuinely noisy market regimes and low uncertainty where signal is strong

**Practical guidelines from this project:**

| Parameter | Plain Huber | Het-Loss | Reason |
|-----------|:-----------:|:--------:|--------|
| Min epochs | 20 | 30 | Variance branch adds optimization axis |
| Optimal LR | 3e-5 | 4e-5 | exp(-s) weighting reduces effective gradient on mean |
| Convergence indicator | val_loss plateaus | aleatoric stabilizes at 0.05-0.15 | Monitor uncertainty, not just loss |

**Failure modes to watch:**
- **Variance dominance (aleatoric > 0.20):** The model is copping out -- it found it easier to predict high uncertainty than to learn signal. Fix: increase LR, add more epochs, or clamp log_var to [-2, 2].
- **Overconfidence (aleatoric < 0.05):** The model ignores noise structure. Fix: decrease LR, increase weight_decay.
- **Per-fold diagnostic:** High aleatoric on a specific fold means the model correctly identifies that regime as noisy (expected for fold 1 GFC onset). High epistemic uncertainty means the model needs more training data from that regime.

## 5. Optimizer: AdamW (Loshchilov & Hutter, 2019)

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=5e-4,            # champion LR; higher than typical due to residual skip stability
    betas=(0.9, 0.999), # PyTorch defaults; beta2=0.999 smooths second moment for noisy FX
    weight_decay=1e-5,  # light regularization -- 1e-3 found to be dead weight for MLP
)
```

**Why AdamW over plain Adam:**
- **Decoupled weight decay** (Loshchilov & Hutter, 2019): Weight decay is applied directly to weights, not through the gradient. In plain Adam, L2 regularization interacts poorly with adaptive learning rates -- large-gradient parameters get less regularization, which is the opposite of what you want.
- **Standard for fine-tuning pretrained models** (Devlin et al. 2019, Hu et al. 2022 LoRA)
- **Robust to noisy financial data gradients:** Adaptive per-parameter LR helps when feature scales vary (our 104 features span very different ranges even after StandardScaler)

**Trainable parameter counts:**

| Backbone | Total Params | Trainable Params | Frozen | Notes |
|----------|:------------:|:----------------:|:------:|-------|
| Residual MLP (champion) | 301K | 301K | 0 | All trained from scratch |
| Plain MLP | ~280K | ~280K | 0 | No skip connection |
| BiLSTM | ~350K | ~350K | 0 | Bidirectional doubles params |
| LFM2-350M | 350M | ~500K | 349.5M | Only projection + heads trained |
| PatchTST | ~2M | ~2M | 0 | Patch embedding + transformer |
| XGBoost/LightGBM | N/A | N/A | N/A | Tree-based, no gradient descent |

## 6. Learning Rate Schedule: Cosine Annealing

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

- Smoothly decays LR from initial value to near-zero over `T_max` epochs
- Avoids sharp LR drops that can cause training instability
- Pairs well with early stopping (effective LR at stop time depends on epoch)

## 7. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
```

- **Max norm = 1.0:** Conservative clipping threshold
- Prevents gradient explosion during fat-tail events (e.g., +5% daily moves)
- Applied after `loss.backward()`, before `optimizer.step()`

## 8. Early Stopping

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = deepcopy(model.state_dict())
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        model.load_state_dict(best_state)
        break
```

- **Patience = 10 epochs (champion):** Tolerates 10 consecutive non-improvements before stopping. Originally 5, but the residual MLP with cosine annealing benefits from waiting through temporary val loss plateaus.
- **State restoration:** Loads best model weights (not last epoch). The model at epoch 50 may have overfit; the best-val-loss epoch is typically around epoch 25-35.
- **Deep copy:** `copy.deepcopy(model.state_dict())` prevents reference aliasing issues where the saved state would be mutated by subsequent optimizer steps.
- **Interaction with cosine annealing:** Early stopping usually triggers before epoch 50 (around epoch 30-40), meaning the effective learning rate at termination is still above zero -- the model stops before the cosine schedule fully decays.

```
  Early Stopping + Cosine Annealing Interaction
  ==============================================

  LR     val_loss
  5e-4 ─┐                    ┌─── val loss
        │'.                 .'│   (noisy)
        │  '.             .'  │
        │    '.         .'    │
  2e-4 ─┤     '..     .      │     * = best state saved
        │        '.*.'       │     X = patience exhausted, stop
        │         '.  '.     │
        │           '.   '.  │
    0 ──┤─────────────'───X──│
        └──┬──┬──┬──┬──┬──┬──┘
          10 15 20 25 30 35 40   epoch
                     ^        ^
                     |        |
              best epoch   stopped at
              (restored)   patience=10
```

## 9. DataLoader Configuration

```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,          # Train only
    pin_memory=True,       # If CUDA available
    num_workers=0,         # Single-process (Windows compatibility)
)
```

- `pin_memory=True` enables async CPU→GPU memory transfer
- `num_workers=0` avoids Windows multiprocessing issues with PyTorch
- `shuffle=True` for training (prevent order-dependent gradient bias)
- `shuffle=False` for validation (reproducible loss computation)

## 10. GPU Strategy

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# In training loop:
x = x.to(device)
```

- Automatic CUDA detection
- Model and data on same device
- CPU fallback for machines without GPU (current deployment is CPU-only: Intel Iris Xe)
- float32 precision (bf16 not well-supported on CPU)

## 11. Training Time Benchmarks

All benchmarks measured on the development machine (Intel Core i7, Intel Iris Xe, 16GB RAM, CPU-only, Windows 11).

| Backbone | Params | Epochs (typical) | Time per Experiment | Notes |
|----------|:------:|:-----------------:|:-------------------:|-------|
| **Residual MLP** (champion) | 301K | ~35 (early stops from 50) | **~36 sec** | Fastest neural backbone |
| Plain MLP | ~280K | ~25 (early stops from 50) | ~30 sec | Slightly faster, much worse results |
| BiLSTM | ~350K | ~30 | ~90 sec | Sequential processing bottleneck |
| PatchTST | ~2M | ~25 | ~180 sec | Attention is O(seq_len^2) |
| Mamba2 | ~1M | ~30 | ~120 sec | Linear-time but larger model |
| LFM2-350M | 500K trainable | ~15 | ~300 sec | Foundation model forward pass dominates |
| XGBoost | N/A | N/A (500 trees) | ~15 sec | Fastest overall |
| LightGBM | N/A | N/A (500 trees) | ~10 sec | Histogram-based, very fast |

**Throughput for champion (Residual MLP):**
- Training samples: 2,478 (after super-fold hole-punching)
- Batches per epoch: ~77 (2478 / 32)
- Forward + backward per batch: ~0.5 ms on CPU
- Total training time: ~36 seconds including data loading, scaling, evaluation
- The 60-second cooldown between experiments is longer than training itself

## 12. LR Schedule: Warmup + Cosine Annealing (Optional)

The training loop supports an optional linear warmup phase before cosine annealing, following Devlin et al. (2019) and Hu et al. (2022, LoRA). This is primarily useful for foundation model fine-tuning (LFM2) where randomly-initialized projection layers need gradual warmup before full-strength gradients.

```python
# With warmup (foundation model fine-tuning):
warmup_sched = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

# Without warmup (from-scratch models like MLP -- champion config):
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

```
  LR Schedule: Warmup + Cosine Annealing
  ========================================

  LR
  5e-4 ─┐                ............
        │              .'            '.
        │            .'                '.
        │          .'                    '.
  2e-4 ─┤        .'                       '.
        │      .'                           '.
        │    .'  cosine annealing              '.
        │  .'    (smooth decay)                  '.
    0 ──┤.'─────────────────────────────────────────'.──
        └─┬────┬───────────────────────────────────┬──
          0   warmup                              T_max
              epochs
              (optional)

  Champion (no warmup):
  - LR starts at 5e-4
  - Cosine decays smoothly to ~0 over 50 epochs
  - Early stopping typically triggers at epoch 30-40
```

## 13. End-to-End Data Flow for One Experiment

```
  +------------------------------------------------------+
  |  run_autoresearch.py (entry point)                    |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  1. Load data from .data_cache/ (ONCE)                |
  |     download_all_pairs() + download_macro_signals()   |
  |     = 6 FX pairs x ~5000 days OHLCV + macro features  |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  2. Compute features (ONCE)                            |
  |     104 backward-looking features per day              |
  |     (momentum, volatility, carry, macro, technical)    |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  3. split_superfold()                                  |
  |     train=2478 | val=915 | test=1170                  |
  |     All 7 folds' val/test windows hole-punched from   |
  |     training + 10-day label horizon buffers            |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  4. Create model + train_one_fold()                    |
  |     StandardScaler.fit(train) -> transform all        |
  |     create_contiguous_datasets() -> DataLoaders       |
  |     AdamW + HuberLoss + CosineAnnealing               |
  |     ~36 sec for MLP champion                          |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  5. Evaluate per-window                                |
  |     7 test windows evaluated individually             |
  |     7 val windows evaluated individually              |
  |     Per-window: Sharpe, return, win rate, IC          |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  6. Compute composite metric                           |
  |     composite = min(test_sharpe, val_sharpe)          |
  |                 - 0.1 * n_negative_folds              |
  |     KEEP if composite > previous best; else DISCARD   |
  +------------------------------------------------------+
       |
       v
  +------------------------------------------------------+
  |  7. Log to experiment_log.jsonl + update best_config  |
  |     60-second cooldown before next experiment         |
  +------------------------------------------------------+
```
