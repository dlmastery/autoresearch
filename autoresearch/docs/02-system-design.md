# 02 - System Design & Architecture

**SWEBoK Knowledge Area:** KA2 — Software Design
**Google SWE Reference:** Ch. 3 — "Knowledge Sharing"; Ch. 12 — "Unit Testing"

---

## 1. High-Level Architecture

```
                        ┌─────────────────────────────┐
                        │      Entry Points            │
                        │  baseline.py  run_ablation.py│
                        │  run_optimizer.py             │
                        └──────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
     │  Data Pipeline   │  │   Model Layer   │  │   Evaluation    │
     │                  │  │                 │  │                 │
     │  download.py     │  │  backbone.py    │  │  metrics.py     │
     │  features.py     │  │  train.py       │  │  leakage_check  │
     │  splits.py       │  │                 │  │                 │
     └────────┬────────┘  └───────┬────────┘  └────────┬────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │    Optimizer (optional)       │
                        │  agent_loop.py + prompts.py   │
                        │  (Claude API driven)          │
                        └─────────────────────────────┘
```

## 2. Module Decomposition

### 2.1 Package Structure

```
autoresearch/
├── data/                     # Data acquisition and preprocessing
│   ├── __init__.py
│   ├── download.py           # Yahoo Finance API + Parquet caching
│   ├── features.py           # 104 backward-looking features
│   └── splits.py             # 7-fold regime-aware walk-forward splits
├── model/                    # Model definitions and training
│   ├── __init__.py
│   ├── backbone.py           # 11 backbone architectures + registry
│   └── train.py              # Training loop (Huber, AdamW, cosine LR)
├── evaluation/               # Metrics and validation
│   ├── __init__.py
│   ├── metrics.py            # 40+ trading + ML-finance metrics
│   └── leakage_check.py      # Purge/embargo/scaler isolation checks
├── optimizer/                # Autonomous improvement loop
│   ├── __init__.py
│   ├── agent_loop.py         # Claude API experiment loop
│   └── prompts.py            # Brainstorm + code generation templates
├── tests/                    # pytest test suite (9 files)
├── baseline.py               # Walk-forward evaluation runner
├── run_ablation.py           # Multi-backbone comparison + reporting
├── run_optimizer.py          # CLI for optimizer
└── run_overnight.py          # Full pipeline (baseline + optimizer)
```

### 2.2 Dependency Graph

```
baseline.py
  ├── data.download      (download_all_pairs, download_macro_signals)
  ├── data.features       (compute_all_features, compute_targets)
  ├── data.splits         (FOLDS, split_data, validate_purge_embargo)
  ├── evaluation.metrics  (sharpe_ratio, trading_report, IC, PSR, DSR, ...)
  ├── model.backbone      (create_model, BACKBONE_REGISTRY, is_gbm, GBMWrapper)
  └── model.train         (create_dataset, train_one_fold, SEQ_LEN)

run_ablation.py
  ├── baseline            (run_baseline)
  └── model.backbone      (BACKBONE_REGISTRY, DEFAULT_BACKBONE)

run_optimizer.py
  └── optimizer.agent_loop (run_optimizer)

optimizer.agent_loop
  ├── baseline            (run_baseline)
  └── anthropic           (Claude API client)
```

## 3. Design Patterns

### 3.1 Registry Pattern (Backbone Selection)

The `BACKBONE_REGISTRY` dictionary maps string identifiers to descriptions. The `create_model()` factory function routes to the correct class based on the backbone string. This enables:
- Adding new backbones without modifying calling code
- CLI selection via `--backbone` argument
- Ablation study iteration over `BACKBONE_REGISTRY.keys()`

```python
BACKBONE_REGISTRY = {
    "mlp": "Simple MLP (no pretrained weights)",
    "lstm": "Bidirectional LSTM baseline",
    "lfm2-350m": "LiquidAI/LFM2.5-350M-Base (frozen)",
    # ... 8 more
}

def create_model(backbone, n_input_features, seq_len, freeze_backbone) -> nn.Module | GBMWrapper
```

### 3.2 Strategy Pattern (Neural vs GBM Training Paths)

Two distinct training/inference paths branch at runtime:
- **Neural models:** `train_one_fold()` → DataLoader → gradient-based training
- **GBM models:** `GBMWrapper.fit()` → sliding window features → tree-based training

The branching happens in `baseline.py` via `is_gbm(backbone)`.

### 3.3 Template Method (Walk-Forward Evaluation)

`run_baseline()` defines the skeleton of walk-forward evaluation:
1. Download data
2. Compute features/targets
3. For each fold: split → scale → train → predict → evaluate
4. Aggregate across folds

Subclasses (neural vs GBM) override the train/predict step, but the evaluation framework is shared.

### 3.4 Checkpoint-Resume Pattern

Both `baseline.py` and `optimizer/agent_loop.py` implement crash recovery:
- **Baseline:** Per-fold checkpointing to `baseline_checkpoint.json` with backbone validation
- **Optimizer:** Per-experiment state to `optimizer_state.json` with file backups

## 4. Data Flow

```
Yahoo Finance API
    │
    ▼
download.py ─── Parquet cache (data/*.parquet)
    │
    ├─→ 6 FX pairs: Dict[str, DataFrame(OHLCV)]
    └─→ 9 macro tickers: Dict[str, DataFrame(OHLCV)]
            │
            ▼
features.py
    │
    ├─→ Per-pair technical (13 per pair × ~8 deduplicated = ~78 features)
    ├─→ Cross-pair correlations (5 features)
    └─→ Macro signals (21 features)
            │
            ▼
    Feature matrix: DataFrame(~4914 rows × 104 columns)
    Target matrix:  DataFrame(~4914 rows × 2 columns: fwd_ret_1d, fwd_ret_5d)
            │
            ▼
splits.py ─── 7 fold definitions (date ranges + regime labels)
    │
    ▼ (per fold)
    train_feat, val_feat, test_feat = split_data(features, fold)
    train_tgt,  val_tgt,  test_tgt  = split_data(targets, fold)
            │
            ▼
    StandardScaler.fit(train_feat) ─── ONLY on training data
            │
            ▼
train.py ─── FXDataset (sliding windows, seq_len=60)
    │
    ▼
backbone.py ─── create_model(backbone) → forward(x) → {"ret_1d", "ret_5d"}
            │
            ▼
    predictions + actuals
            │
            ▼
metrics.py ─── strategy_returns = sign(pred) × actual
    │
    ├─→ trading_report() → 40+ metrics
    ├─→ information_coefficient() → IC, hit_rate
    ├─→ probabilistic_sharpe_ratio() → PSR
    └─→ deflated_sharpe_ratio() → DSR
            │
            ▼
    Per-fold results → aggregate → average_sharpe, weighted_sharpe
            │
            ▼
    baseline_results.json / ablation_results/*.json / reports/*.md
```

## 5. Interface Contracts

### 5.1 Neural Model Interface

All `nn.Module` backbones must implement:
```python
def forward(self, x: Tensor[batch, seq_len, n_features]) -> Dict[str, Tensor]:
    return {
        "ret_1d": Tensor[batch, 6],   # 6 currency pairs, 1-day prediction
        "ret_5d": Tensor[batch, 6],   # 6 currency pairs, 5-day prediction
    }
```

### 5.2 GBM Wrapper Interface

```python
class GBMWrapper:
    def fit(self, X: ndarray[n, seq_len*n_features], y: ndarray[n, 2]) -> None
    def predict(self, X: ndarray[n, seq_len*n_features]) -> ndarray[n, 2]
```

### 5.3 Feature Pipeline Interface

```python
def compute_all_features(
    all_pairs: Dict[str, DataFrame],
    macro_data: Dict[str, DataFrame],
    primary: str = "EURUSD=X"
) -> DataFrame  # rows=dates, cols=feature_names

def compute_targets(df: DataFrame) -> DataFrame  # cols=[fwd_ret_1d, fwd_ret_5d]
```

### 5.4 Split Interface

```python
def split_data(df: DataFrame, fold: dict) -> Tuple[DataFrame, DataFrame, DataFrame]
    # Returns (train, val, test) subsets filtered by fold date ranges
```

## 6. Error Handling Strategy

| Layer | Strategy | Rationale |
|-------|----------|-----------|
| Data download | Log warning, skip failing ticker | Macro signals are supplementary |
| Feature computation | Drop NaN rows after warmup | Rolling windows create NaNs at edges |
| Fold validation | Raise RuntimeError on violation | Leakage prevention is a hard constraint |
| Training | Early stopping + best state restore | Prevent overfitting; recover from divergence |
| Gradient | Clip at norm 1.0 | Prevent exploding gradients on fat-tailed data |
| Optimizer | Backup + revert on failure | Protect against generated code that breaks |
| Checkpoint | Validate backbone match + array lengths | Prevent stale checkpoint corruption |

## 7. Concurrency & Threading

The system is single-threaded by design. Parallelism considerations:
- **DataLoader:** `num_workers=0` (single-process data loading, avoids Windows multiprocessing issues)
- **CUDA:** `pin_memory=True` when GPU available (async CPU→GPU transfer)
- **GBM:** Tree-based models use internal parallelism via native libraries
- **Data download:** Sequential per-ticker (yfinance rate limiting)
