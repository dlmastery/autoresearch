# 05 - Training Infrastructure

**SWEBoK Knowledge Area:** KA3 — Software Construction
**Google SWE Reference:** Ch. 11 — "Testing Overview" (training correctness)

---

## 1. Training Loop Architecture

### 1.1 Neural Training Path

```
Per fold:
  StandardScaler.fit(train_features)
  ├─→ transform(train_features) → FXDataset → DataLoader(shuffle=True)
  ├─→ transform(val_features)   → FXDataset → DataLoader(shuffle=False)
  │
  create_model(backbone, n_features, seq_len, freeze_backbone=True)
  │
  AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
  HuberLoss(delta=1.0)
  CosineAnnealingLR(optimizer, T_max=epochs)
  │
  for epoch in range(epochs):
      for x, y in train_loader:
          preds = model(x)  →  {"ret_1d": [B,6], "ret_5d": [B,6]}
          stacked = stack([preds["ret_1d"][:,0], preds["ret_5d"][:,0]], dim=1)
          loss = huber_loss(stacked, y)
          loss.backward()
          clip_grad_norm_(params, max_norm=1.0)
          optimizer.step()
      scheduler.step()
      val_loss = evaluate(model, val_loader)
      if val_loss < best_val_loss:
          save best_state
          reset patience_counter
      elif patience_counter >= PATIENCE:
          restore best_state
          break
```

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

### 2.1 Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SEQ_LEN` | **10** (non-LFM) / **60** (LFM2.5) | 10 biz days (~2 weeks) is industry standard for short-term FX; LFM benefits from long context |
| `BATCH_SIZE` | 32 | Smaller batches = better gradient estimates on noisy FX data |
| `LEARNING_RATE` | 3e-4 | AdamW sweet spot for fine-tuning (Loshchilov & Hutter, 2019) |
| `WEIGHT_DECAY` | 1e-5 | Light L2 regularization |
| `GRAD_CLIP` | 1.0 | Prevents exploding gradients on fat-tailed FX data |
| `PATIENCE` | 5 | 5-epoch patience avoids premature stopping on noisy val loss |
| `EPOCHS` | 20 | More epochs with early stopping -- let patience decide when to stop |
| `HUBER_DELTA` | 1.0 | Standard delta; robust to outlier returns |

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
criterion = torch.nn.HuberLoss(delta=1.0)
```

**Why Huber over MSE:**
- FX returns have fat tails (excess kurtosis typically 3-10)
- MSE squares large residuals, overweighting extreme days (GFC, COVID)
- Huber transitions from quadratic (near zero) to linear (far from zero) at `delta`
- Result: more stable gradients, less sensitivity to outlier returns

## 5. Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
```

**Why AdamW:**
- Decoupled weight decay (vs Adam's L2 in gradient)
- Standard for fine-tuning pretrained models
- Robust to noisy financial data gradients

**Trainable parameters:**
- Foundation models (LFM2.5): Only projection layer + prediction heads (~500K params)
- From-scratch models (MLP, LSTM, etc.): All parameters

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

- **Patience = 3 epochs:** Tolerates 3 consecutive non-improvements before stopping
- **State restoration:** Loads best model weights (not last epoch)
- **Deep copy:** Prevents reference aliasing issues with state dicts

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
- CPU fallback for machines without GPU
- float32 precision (bf16 not well-supported on CPU)
