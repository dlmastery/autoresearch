# 02 - System Design & Architecture

**SWEBoK Knowledge Area:** KA2 -- Software Design
**Google SWE Reference:** Ch. 3 -- "Knowledge Sharing"; Ch. 12 -- "Unit Testing"

---

## Key Highlights

- **Three-layer architecture:** Data Pipeline, Model Layer, and Evaluation are cleanly separated with explicit interface contracts
- **Registry pattern** for backbone selection enables adding new architectures without modifying calling code
- **Super-fold evaluation** trains once and evaluates across all 7 regime windows, reducing experiment time from ~4 minutes to ~36 seconds
- **Crash-recovery checkpointing** at every experiment boundary enables seamless resume after laptop/process crashes
- **Single-threaded by design** -- avoids Windows multiprocessing pitfalls while keeping code simple and debuggable

---

## 1. High-Level Architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                        CLAUDE CODE AGENT                            │
 │  Reads results -> Diagnoses per-fold failures -> Proposes changes   │
 │  -> Runs ONE experiment -> Analyzes -> Keeps or reverts             │
 └────────────────────────────────┬─────────────────────────────────────┘
                                  │ invokes via CLI
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                      run_autoresearch.py                            │
 │              (single-experiment executor, logs JSONL)                │
 └────────────────────────────────┬─────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
 ┌────────▼─────────┐   ┌────────▼─────────┐   ┌────────▼─────────┐
 │  Data Pipeline    │   │  Model Layer      │   │  Evaluation      │
 │                   │   │                   │   │                  │
 │  download.py      │   │  backbone.py      │   │  metrics.py      │
 │  features.py      │   │  train.py         │   │                  │
 │  splits.py        │   │                   │   │                  │
 └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                      autoresearch_results/                          │
 │  experiment_log.jsonl  |  best_config.json  |  dashboard.html      │
 └──────────────────────────────────────────────────────────────────────┘
```

### 1.1 Component Interaction Diagram

```
                    ┌──────────────┐
                    │  Yahoo       │
                    │  Finance API │
                    └──────┬───────┘
                           │ (cached to .data_cache/*.parquet)
                           ▼
┌───────────────────────────────────────────────────┐
│              download.py                          │
│  download_all_pairs() -> Dict[str, DataFrame]     │
│  download_macro_signals() -> Dict[str, DataFrame] │
└─────────────────────┬─────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────┐
│              features.py                          │
│  compute_all_features() -> DataFrame[4914 x 104]  │
│  compute_targets() -> DataFrame[4914 x 2]         │
└─────────────────────┬─────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────┐
│              splits.py                            │
│  split_superfold() -> (train, val, test)          │
│  Hole-punching: removes ALL val/test windows +    │
│  10-day label buffers from training data           │
└─────────────────────┬─────────────────────────────┘
                      │
            ┌─────────┼──────────┐
            ▼         ▼          ▼
        ┌───────┐ ┌───────┐ ┌───────┐
        │ train │ │  val  │ │ test  │
        │ 2478  │ │  915  │ │ 1170  │
        │ rows  │ │ rows  │ │ rows  │
        └───┬───┘ └───┬───┘ └───┬───┘
            │         │         │
            ▼         │         │
┌──────────────────┐  │         │
│  StandardScaler  │  │         │
│  .fit(train)     │──┤         │
│  .transform(*)   │  │         │
└────────┬─────────┘  │         │
         │            │         │
         ▼            ▼         ▼
┌───────────────────────────────────────────────────┐
│              train.py                             │
│  create_contiguous_datasets() -> DataLoaders      │
│  train_one_fold() -> trained model + val_loss     │
└─────────────────────┬─────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────┐
│              backbone.py                          │
│  create_model("mlp") -> CurrencyMLP(300K params)  │
│  forward(x) -> {ret_1d, ret_5d}                   │
└─────────────────────┬─────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────┐
│              metrics.py                           │
│  _evaluate_per_window() -> per-fold Sharpe, IC,   │
│  Sortino, max DD, win rate, profit factor, PSR    │
│  + uncertainty: aleatoric, epistemic, confidence   │
└─────────────────────┬─────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────┐
│              Results (JSONL + JSON)                │
│  experiment_log.jsonl (append-only)               │
│  best_config.json (overwrite on KEEP)             │
└───────────────────────────────────────────────────┘
```

### 1.2 Deployment Topology

The system runs entirely on a single machine (Windows 11, Intel Iris Xe GPU, float32 mode). There is no distributed training, no cloud deployment, and no containerization. This is intentional for a research prototype.

```
┌───────────────────────────────────────────────────────────────┐
│  Windows 11 Laptop (Intel Iris Xe, 16GB RAM)                  │
│                                                               │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │  Claude Code CLI     │   │  Python 3.11.9 (Anaconda)    │  │
│  │  (outer agent loop)  │──>│  autoresearch package        │  │
│  │  1M context window   │   │  PyTorch 2.5+ (CPU/float32)  │  │
│  └─────────────────────┘   └──────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │  .data_cache/        │   │  autoresearch_results/       │  │
│  │  Parquet files       │   │  JSONL logs + dashboard      │  │
│  │  (15 instruments)    │   │  (http.server on port 8765)  │  │
│  └─────────────────────┘   └──────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────┐                                      │
│  │  memory/             │                                      │
│  │  Crash-recovery      │                                      │
│  │  checkpoint (.md)    │                                      │
│  └─────────────────────┘                                      │
└───────────────────────────────────────────────────────────────┘
```

## 2. Module Decomposition

### 2.1 Package Structure

```
autoresearch/                       # Package root
├── data/                           # Layer 1: Data acquisition & preprocessing
│   ├── __init__.py
│   ├── download.py                 # Yahoo Finance API + Parquet caching
│   ├── features.py                 # 104 backward-looking features from 15 instruments
│   └── splits.py                   # 7-fold regime-aware splits, super-fold, purge/embargo
├── model/                          # Layer 2: Model definitions & training
│   ├── __init__.py
│   ├── backbone.py                 # 8 backbone architectures + registry + factory
│   └── train.py                    # Training loop (Huber, AdamW, cosine LR, early stopping)
├── evaluation/                     # Layer 3: Metrics & validation
│   ├── __init__.py
│   └── metrics.py                  # Sharpe, Sortino, PSR, DSR, IC, trading_report, etc.
├── run_autoresearch.py             # Single-experiment executor (CLI, logs JSONL)
├── baseline.py                     # Walk-forward evaluation runner (all folds)
├── run_ablation.py                 # Multi-backbone comparison + reporting
├── autoresearch_results/           # Output directory
│   ├── experiment_log.jsonl        # Append-only structured experiment log
│   ├── best_config.json            # Current champion configuration + metrics
│   └── dashboard.html              # Live HTML dashboard (reads JSONL, decoupled)
├── docs/                           # Documentation (this directory)
└── memory/                         # Crash-recovery checkpoints
    └── project_autoresearch_checkpoint.md
```

### 2.2 Dependency Graph

```
run_autoresearch.py                       # Primary entry point
  ├── data.download      (download_all_pairs, download_macro_signals)
  ├── data.features      (compute_all_features, compute_targets)
  ├── data.splits        (FOLDS, split_superfold, validate_purge_embargo, get_fold_dates)
  ├── evaluation.metrics (sharpe_ratio, trading_report, IC, PSR, ...)
  ├── model.backbone     (create_model, get_seq_len, BACKBONE_REGISTRY, is_gbm,
  │                        GBMWrapper, predict_with_uncertainty)
  └── model.train        (create_dataset, create_contiguous_datasets,
                           find_contiguous_segments, train_one_fold)

baseline.py                                # Legacy walk-forward runner
  ├── data.download      (download_all_pairs, download_macro_signals)
  ├── data.features      (compute_all_features, compute_targets)
  ├── data.splits        (FOLDS, split_data, validate_purge_embargo)
  ├── evaluation.metrics (sharpe_ratio, trading_report, IC, PSR, DSR, ...)
  ├── model.backbone     (create_model, BACKBONE_REGISTRY, is_gbm, GBMWrapper)
  └── model.train        (create_dataset, train_one_fold, SEQ_LEN)

run_ablation.py                            # Multi-backbone comparison
  ├── baseline           (run_baseline)
  └── model.backbone     (BACKBONE_REGISTRY, DEFAULT_BACKBONE)
```

### 2.3 Data Flow Diagram

The end-to-end data flow from raw market data to experiment results:

```
                         ┌─────────────────┐
                         │  Yahoo Finance   │
                         │  (15 tickers)    │
                         └────────┬────────┘
                                  │
                    download_all_pairs()
                    download_macro_signals()
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   .data_cache/*.parquet  │  <-- Parquet cache (never re-download)
                    │   6 FX + 9 macro files   │
                    └────────────┬────────────┘
                                 │
                   compute_all_features()
                   compute_targets()
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │  Feature Matrix   Target Matrix  │
              │  [4914 x 104]     [4914 x 2]     │
              │  (backward-       (fwd_ret_1d,   │
              │   looking only)    fwd_ret_5d)    │
              └──────────────┬───────────────────┘
                             │
                   split_superfold()
                   (hole-punching: removes ALL val/test
                    windows + 10-day buffers from train)
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌─────────┐   ┌─────────┐   ┌─────────────┐
         │  TRAIN   │   │   VAL   │   │    TEST     │
         │  2478    │   │   915   │   │    1170     │
         │  rows    │   │  rows   │   │   rows      │
         │          │   │ 7 folds │   │  7 folds    │
         └────┬────┘   └────┬────┘   └──────┬──────┘
              │              │               │
     StandardScaler.fit()    │               │
     .transform(all three)   │               │
              │              │               │
              ▼              │               │
     create_contiguous_      │               │
     datasets()              │               │
     (no windows across      │               │
      date gaps)             │               │
              │              │               │
              ▼              │               │
     train_one_fold()        │               │
     - Huber loss (d=0.5)    │               │
     - AdamW (lr=5e-4)       │               │
     - Cosine annealing      │               │
     - Early stopping (p=10) │               │
     - 50 epochs max         │               │
              │              │               │
              ▼              ▼               ▼
     ┌───────────────────────────────────────────┐
     │  _evaluate_per_window()                   │
     │  Per-fold: Sharpe, Sortino, IC, PSR,      │
     │  max DD, win rate, profit factor,          │
     │  aleatoric/epistemic uncertainty           │
     └────────────────────┬──────────────────────┘
                          │
                          ▼
     ┌───────────────────────────────────────────┐
     │  Composite = min(test_sharpe, val_sharpe) │
     │              - 0.1 * n_negative_folds     │
     │                                           │
     │  if composite > champion.composite:       │
     │      KEEP  -> update best_config.json     │
     │  else:                                    │
     │      DISCARD -> revert                    │
     └────────────────────┬──────────────────────┘
                          │
                          ▼
     ┌───────────────────────────────────────────┐
     │  experiment_log.jsonl (append)            │
     │  best_config.json (overwrite on KEEP)     │
     │  memory/checkpoint.md (always update)     │
     └───────────────────────────────────────────┘
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
    "patchtst": "PatchTST -- Patch Time Series Transformer (Nie et al., ICLR 2023)",
    "patchtsmixer": "PatchTSMixer -- MLP-Mixer for time series (Google, NeurIPS 2023)",
    "xgboost": "XGBoost gradient boosting (Chen & Guestrin, 2016)",
    "lightgbm": "LightGBM gradient boosting (Ke et al., NeurIPS 2017)",
    "catboost": "CatBoost gradient boosting (Prokhorenkova et al., NeurIPS 2018)",
}

def create_model(backbone, n_input_features, seq_len, freeze_backbone) -> nn.Module | GBMWrapper
```

To add a new backbone, a developer needs to:
1. Create a new `nn.Module` class implementing the forward interface (see [Model Architecture](model-architecture.md))
2. Add a key to `BACKBONE_REGISTRY`
3. Add a routing case in `create_model()`
4. Optionally add a custom `seq_len` to `BACKBONE_SEQ_LEN`

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

## 4. Data Flow (Detailed)

For the detailed data flow diagram, see Section 2.3 above.

### 4.1 Feature Breakdown

```
104 features total:
  ├── Per-pair technical (13 per pair x 6 pairs = 78 features)
  │     momentum (3): log_ret_1d, log_ret_5d, log_ret_21d
  │     volatility (3): rvol_5d, rvol_21d, rvol_63d
  │     mean-reversion (1): rsi_14
  │     trend (3): macd_line, macd_signal, macd_hist
  │     microstructure (3): ohlc_range, overnight_gap, norm_true_range
  │
  ├── Cross-pair correlations (5 features)
  │     21-day rolling correlation of EUR/USD vs each of 5 other pairs
  │
  └── Macro signals (21 features)
        per-ticker (2 each x 9 tickers = 18): level + 1d return
        derived (3): yield_curve_slope, VIX_5d_chg, DXY_rvol_21d
```

### 4.2 Trading Strategy

The evaluation uses a simple sign-based trading strategy:

```
position(t) = sign(predicted_return(t))
strategy_return(t) = position(t) * actual_return(t+1)
annualized_sharpe = mean(strategy_return) / std(strategy_return) * sqrt(252)
```

This is intentionally simple -- the goal is to measure prediction quality, not portfolio optimization.

### 4.3 Composite Scoring

The keep/discard decision uses a composite metric that penalizes models with inconsistent performance:

```
composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_test_folds
```

This ensures:
- The model must perform well on BOTH validation and test sets (takes the minimum)
- Each negative test fold imposes a 0.1 penalty, discouraging models that excel on some regimes but fail on others
- The champion (composite 5.499) has min(6.21, 5.60) - 0.1 * 0 = 5.60

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
- **CUDA:** `pin_memory=True` when GPU available (async CPU->GPU transfer)
- **GBM:** Tree-based models use internal parallelism via native libraries
- **Data download:** Sequential per-ticker (yfinance rate limiting)

## 8. Experiment Lifecycle

Each experiment follows this lifecycle, managed by the Claude Code agent:

```
  ┌─────────────────────┐
  │ 1. DIAGNOSE          │  Read per-fold results, identify weak folds
  │    Which folds fail? │  Check aleatoric/epistemic uncertainty
  │    Why?              │  Identify regime-specific failure modes
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ 2. CITE              │  Find relevant literature or prior results
  │    What does theory  │  "He et al. 2016 suggests residual connections
  │    suggest?          │   help with low-SNR signal learning"
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ 3. HYPOTHESIZE       │  Formulate a testable hypothesis
  │    ONE change from   │  "Adding a linear shortcut will improve
  │    current champion  │   fold 1 and 2 where signal is weakest"
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ 4. PREDICT           │  State expected outcome before running
  │    Expected delta?   │  "Expect Sharpe +0.3 on weak folds,
  │                      │   flat on strong folds"
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ 5. RUN               │  Execute ONE experiment via CLI
  │    ~36s for MLP      │  python -m autoresearch.run_autoresearch ...
  │    ~4min for LFM2    │  Wait for completion, capture stdout
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ 6. ANALYZE           │  Compare per-fold results vs champion
  │    KEEP or DISCARD?  │  composite > champion.composite? -> KEEP
  │    Update champion   │  Otherwise -> DISCARD, try different axis
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ 7. CHECKPOINT        │  Save state to memory/ + experiment_log.jsonl
  │    Crash-safe state  │  Must be self-contained for cold restart
  │                      │  Wait 60s cooldown, then loop to step 1
  └─────────────────────┘
```

## 9. Key Design Tradeoffs

| Decision | Alternative Considered | Rationale for Current Choice |
|----------|----------------------|------------------------------|
| Super-fold (single train) | 7 separate folds | 7x faster experiments; same statistical validity |
| Huber loss (d=0.5) | MSE, MAE, het-loss | Robust to fat tails; het-loss added complexity without benefit (tested in experiments 20-40) |
| CPU-only training | GPU (CUDA) | Intel Iris Xe; MLP trains in 36s on CPU anyway |
| Sign-based strategy | Kelly criterion, vol-targeting | Isolates prediction quality from portfolio optimization |
| JSONL experiment log | SQLite, PostgreSQL | Human-readable, append-only, trivially parseable |
| Claude Code as outer loop | Python agent loop | More flexible, can modify code, natural language reasoning |

---

*See also:* [Project Overview](project-overview.md) | [Data Engineering](../data/data-engineering.md) | [Model Architecture](model-architecture.md)
