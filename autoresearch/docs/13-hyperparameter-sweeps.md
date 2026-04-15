# 13 - Hyperparameter Sweeps and Tuning Strategy

**SWEBoK Knowledge Area:** KA3 -- Software Construction (Optimization)
**Google SWE Reference:** Ch. 12 -- "Unit Testing" (reproducible experiments); Ch. 24 -- "Continuous Delivery" (automated pipelines)

---

## 1. Executive Summary

This document specifies the hyperparameter tuning strategy for the FX prediction
system. The system evaluates 11 backbones (MLP, BiLSTM, LFM2.5-350M, PatchTST,
PatchTSMixer, Mamba2, Informer, XGBoost, LightGBM, CatBoost) across 7
regime-aware walk-forward folds with 90-day purge gaps, optimizing for Sharpe
ratio.

**Key decisions:**
- **Framework:** Optuna with a PostgreSQL backend for persistence and pruning
- **Search strategy:** TPE (Tree-structured Parzen Estimator) with Hyperband pruning
- **Leakage prevention:** Nested walk-forward with inner validation folds drawn
  strictly from the outer training set
- **Priority:** Learning rate > sequence length > regularization > architecture >
  loss function > feature selection > ensemble weights

---

## 2. Framework Selection

### 2.1 Comparison

| Criterion | Optuna | Ray Tune | W&B Sweeps |
|-----------|--------|----------|------------|
| **TPE/Bayesian search** | Native, highly optimized | Via Optuna/BayesOpt | Via Bayes method |
| **Pruning (Hyperband)** | Built-in `MedianPruner`, `HyperbandPruner` | Native ASHA/Hyperband | Not available |
| **Walk-forward support** | Full control via custom objective | Requires manual setup | Limited callback API |
| **Multi-objective** | Native `NSGAIISampler` | Via Optuna integration | Not supported |
| **State persistence** | RDB (SQLite/PostgreSQL) or journal | Redis/DB backend | Cloud (paid tier) |
| **Parallelism** | `n_jobs` or distributed workers | Native distributed | Cloud workers (paid) |
| **Learning curve** | Low -- single `study.optimize()` call | Medium -- needs Ray cluster setup | Low but limited |
| **Cost** | Free, open-source | Free, open-source | Free tier limited, paid for teams |
| **GPU scheduling** | Manual or via plugins | Native GPU placement | Not applicable |
| **Integration effort** | ~50 lines to wrap existing `train_one_fold` | ~200 lines for Ray Trainable | ~100 lines for sweep agent |

### 2.2 Recommendation: Optuna

Optuna is the clear choice for this project because:

1. **TPE is state-of-the-art for ML hyperparameter search** and Optuna's
   implementation is the reference.
2. **Pruning via Hyperband** is critical: with 7 folds each taking minutes, early
   stopping of bad configurations saves hours.
3. **Multi-objective optimization** (Sharpe + max_drawdown) is native.
4. **Walk-forward compatibility:** Optuna's objective function is a plain Python
   callable, so the nested CV logic is straightforward.
5. **Persistence:** SQLite backend survives crashes; PostgreSQL enables
   distributed workers across machines.
6. **No external infrastructure** needed (no Ray cluster, no W&B account).

Required addition to `requirements.txt`:

```
optuna>=3.6
optuna-dashboard  # optional: web UI for monitoring studies
```

---

## 3. Search Strategies

### 3.1 TPE (Tree-structured Parzen Estimator)

The default and recommended sampler for this project.

**How it works:** Rather than modeling p(y|x) (performance given hyperparameters),
TPE models p(x|y) -- the probability of hyperparameters given good vs. bad
performance. It splits trials into "good" (top gamma percentile) and "bad"
groups, fits kernel density estimators to each, and samples where the ratio
l(x)/g(x) is high.

**Why it fits financial ML:**
- Handles conditional search spaces (e.g., `num_layers` only exists for
  transformers, not GBMs)
- Works well with mixed discrete/continuous parameters
- More sample-efficient than random search with 50-200 trials
- Does not assume smooth objective landscape (Sharpe is noisy)

**Configuration:**

```python
import optuna

sampler = optuna.samplers.TPESampler(
    n_startup_trials=20,       # random exploration before Bayesian kicks in
    n_ei_candidates=24,        # candidates per acquisition step
    multivariate=True,         # model parameter correlations
    seed=42,                   # reproducibility
)
```

### 3.2 Hyperband Pruning

Hyperband addresses the explore-exploit tradeoff by allocating small budgets
(fewer epochs or fewer folds) to many configurations, then progressively
allocating more budget to promising ones.

**For this project:** Use fold-level pruning. Report intermediate Sharpe after
each of the 7 folds. If a configuration is clearly underperforming after 3
folds, prune it.

```python
pruner = optuna.pruners.HyperbandPruner(
    min_resource=3,            # minimum 3 folds before pruning
    max_resource=7,            # all 7 folds
    reduction_factor=3,        # bracket width
)
```

### 3.3 BOHB (Bayesian Optimization and Hyperband)

BOHB combines TPE-style Bayesian optimization with Hyperband scheduling. In
practice, Optuna's TPE + HyperbandPruner achieves equivalent results without
needing the separate BOHB library. Use this approach rather than importing
`hpbandster`.

### 3.4 Random Search Baseline

Always run 20 random trials first (controlled by `n_startup_trials`) to
establish a baseline distribution. This prevents TPE from getting stuck in a
local basin early.

### 3.5 Multi-Objective: NSGA-II

For simultaneous optimization of Sharpe, max drawdown, and IC:

```python
sampler = optuna.samplers.NSGAIISampler(seed=42)

study = optuna.create_study(
    directions=["maximize", "minimize", "maximize"],  # Sharpe, MaxDD, IC
    sampler=sampler,
)
```

---

## 4. Per-Backbone Sweep Ranges

### 4.1 Shared Neural Hyperparameters

These apply to all neural backbones (MLP, LSTM, PatchTST, PatchTSMixer, Mamba2,
Informer, LFM2.5):

| Parameter | Current | Sweep Range | Scale | Rationale |
|-----------|---------|-------------|-------|-----------|
| `learning_rate` | 1e-4 | [1e-5, 1e-2] | log-uniform | Most impactful HP; log scale covers 3 orders of magnitude |
| `weight_decay` | 1e-5 | [1e-6, 1e-2] | log-uniform | Regularization strength; too high kills signal in noisy FX |
| `batch_size` | 64 | {32, 64, 128, 256} | categorical | Affects gradient noise; smaller batches = implicit regularization |
| `seq_len` | 60 | {20, 40, 60, 90, 120} | categorical | Lookback window; see Section 8 |
| `epochs` | 5 | [5, 30] | int | With early stopping, upper bound matters less |
| `patience` | 3 | {3, 5, 7, 10} | categorical | Tied to epochs; more patience = more training time |
| `grad_clip` | 1.0 | [0.1, 5.0] | uniform | Prevents explosion; rarely needs tuning |
| `huber_delta` | 1.0 | [0.5, 2.0] | uniform | Controls robustness/efficiency tradeoff |
| `dropout` | 0.1 | [0.0, 0.5] | uniform | Head dropout; architecture-specific body dropout below |
| `scheduler` | cosine | {cosine, linear_warmup_cosine, one_cycle, reduce_on_plateau} | categorical | Schedule shape affects convergence |

```python
def suggest_shared_neural(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 90, 120]),
        "epochs": trial.suggest_int("epochs", 5, 30),
        "patience": trial.suggest_int("patience", 3, 10),
        "grad_clip": trial.suggest_float("grad_clip", 0.1, 5.0),
        "huber_delta": trial.suggest_float("huber_delta", 0.5, 2.0),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.5),
    }
```

### 4.2 MLP-Specific

| Parameter | Current | Sweep Range | Scale |
|-----------|---------|-------------|-------|
| `hidden_size` | 512 | {128, 256, 512, 1024} | categorical |
| `n_layers` | 2 | {2, 3, 4} | categorical |
| `body_dropout` | 0.1 | [0.0, 0.5] | uniform |
| `activation` | GELU | {GELU, SiLU, ReLU} | categorical |

```python
def suggest_mlp(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    params["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024])
    params["n_layers"] = trial.suggest_int("n_layers", 2, 4)
    params["body_dropout"] = trial.suggest_float("body_dropout", 0.0, 0.5)
    return params
```

### 4.3 BiLSTM-Specific

| Parameter | Current | Sweep Range | Scale |
|-----------|---------|-------------|-------|
| `hidden_size` | 256 | {128, 256, 512} | categorical |
| `num_layers` | 2 | {1, 2, 3} | categorical |
| `recurrent_dropout` | 0.1 | [0.0, 0.3] | uniform |
| `bidirectional` | True | {True, False} | categorical |

```python
def suggest_lstm(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    params["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", [128, 256, 512])
    params["lstm_layers"] = trial.suggest_int("lstm_layers", 1, 3)
    params["recurrent_dropout"] = trial.suggest_float("recurrent_dropout", 0.0, 0.3)
    return params
```

### 4.4 PatchTST-Specific

| Parameter | Current | Sweep Range | Scale | Notes |
|-----------|---------|-------------|-------|-------|
| `patch_length` | 12 | {5, 10, 12, 15, 20} | categorical | Must divide evenly or be handled by stride |
| `stride` | 12 (= patch_length) | {patch_length // 2, patch_length} | conditional | 50% overlap or no overlap |
| `d_model` | 256 | {128, 256, 512} | categorical | Transformer width |
| `num_attention_heads` | 4 | {2, 4, 8} | categorical | Must divide d_model |
| `num_hidden_layers` | 3 | {2, 3, 4, 6} | categorical | Depth |
| `head_dropout` | 0.1 | [0.0, 0.3] | uniform | -- |
| `positional_encoding` | learned | {learned, sinusoidal} | categorical | -- |

```python
def suggest_patchtst(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    params["d_model"] = d_model
    params["patch_length"] = trial.suggest_categorical("patch_length", [5, 10, 12, 15, 20])
    params["num_heads"] = trial.suggest_categorical("num_heads", [h for h in [2, 4, 8] if d_model % h == 0])
    params["num_layers"] = trial.suggest_int("num_layers", 2, 6)
    return params
```

### 4.5 PatchTSMixer-Specific

| Parameter | Current | Sweep Range | Scale |
|-----------|---------|-------------|-------|
| `patch_length` | 12 | {5, 10, 12, 15, 20} | categorical |
| `d_model` | 256 | {128, 256, 512} | categorical |
| `num_layers` | 4 | {2, 4, 6, 8} | categorical |
| `expansion_factor` | 4 (default) | {2, 4, 8} | categorical |
| `dropout` | 0.1 | [0.0, 0.3] | uniform |

```python
def suggest_patchtsmixer(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    params["d_model"] = trial.suggest_categorical("d_model", [128, 256, 512])
    params["patch_length"] = trial.suggest_categorical("patch_length", [5, 10, 12, 15, 20])
    params["num_layers"] = trial.suggest_int("num_layers", 2, 8)
    return params
```

### 4.6 Mamba2-Specific

| Parameter | Current | Sweep Range | Scale | Notes |
|-----------|---------|-------------|-------|-------|
| `hidden_size` | 256 | {128, 256, 384, 512} | categorical | Must be divisible by head_dim |
| `num_hidden_layers` | 4 | {2, 4, 6, 8} | categorical | SSM depth |
| `expand` | 2 | {2, 4} | categorical | Inner dimension multiplier |
| `head_dim` | 64 | {32, 64, 128} | categorical | Per-head dimension |
| `n_groups` | 1 | {1, 2, 4} | categorical | Grouped state space |

**Constraint:** `hidden_size * expand == num_heads * head_dim`, so `hidden_size`
must be chosen such that it is a multiple of `head_dim`. The factory already
rounds up.

```python
def suggest_mamba2(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    head_dim = trial.suggest_categorical("head_dim", [32, 64, 128])
    hidden_raw = trial.suggest_categorical("hidden_size", [128, 256, 384, 512])
    # Round up to multiple of head_dim
    hidden_size = ((hidden_raw + head_dim - 1) // head_dim) * head_dim
    params["hidden_size"] = hidden_size
    params["head_dim"] = head_dim
    params["num_layers"] = trial.suggest_int("mamba_layers", 2, 8)
    params["expand"] = trial.suggest_categorical("expand", [2, 4])
    return params
```

### 4.7 Informer-Specific

| Parameter | Current | Sweep Range | Scale |
|-----------|---------|-------------|-------|
| `d_model` | 256 | {128, 256, 512} | categorical |
| `encoder_layers` | 2 | {2, 3, 4} | categorical |
| `decoder_layers` | 1 | {1, 2} | categorical |
| `num_attention_heads` | 4 | {2, 4, 8} | categorical |
| `prob_sparse_factor` | 5 (default) | {3, 5, 7} | categorical |
| `distil` | True (default) | {True, False} | categorical |

```python
def suggest_informer(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    params["d_model"] = d_model
    params["encoder_layers"] = trial.suggest_int("enc_layers", 2, 4)
    params["decoder_layers"] = trial.suggest_int("dec_layers", 1, 2)
    params["num_heads"] = trial.suggest_categorical("num_heads", [h for h in [2, 4, 8] if d_model % h == 0])
    return params
```

### 4.8 LFM2.5 Foundation Model

See Section 6 for the full fine-tuning strategy. Key sweep parameters for the
frozen-backbone regime:

| Parameter | Current | Sweep Range | Scale | Notes |
|-----------|---------|-------------|-------|-------|
| `projection_lr` | 1e-4 | [1e-5, 1e-3] | log-uniform | Projection layer LR |
| `head_lr` | 1e-4 | [1e-5, 1e-3] | log-uniform | Prediction head LR |
| `freeze_backbone` | True | {True, False} | categorical | See progressive unfreezing |
| `backbone_lr_factor` | 0.0 (frozen) | [0.0, 0.1] | uniform | Fraction of head LR |
| `warmup_epochs` | 0 | {0, 1, 2, 3} | categorical | Linear warmup before cosine |
| `projection_hidden` | 1024 | {512, 1024, 2048} | categorical | Projection dim (must match backbone) |

```python
def suggest_lfm(trial: optuna.Trial) -> dict:
    params = suggest_shared_neural(trial)
    params["projection_lr"] = trial.suggest_float("projection_lr", 1e-5, 1e-3, log=True)
    params["head_lr"] = trial.suggest_float("head_lr", 1e-5, 1e-3, log=True)
    freeze = trial.suggest_categorical("freeze_backbone", [True, False])
    params["freeze_backbone"] = freeze
    if not freeze:
        params["backbone_lr_factor"] = trial.suggest_float("backbone_lr_factor", 0.001, 0.1, log=True)
        params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 1, 3)
    return params
```

### 4.9 XGBoost

| Parameter | Current | Sweep Range | Scale | Notes |
|-----------|---------|-------------|-------|-------|
| `n_estimators` | 200 | [100, 2000] | int | More trees with lower LR |
| `max_depth` | 6 | [3, 10] | int | Deeper = more capacity, more overfit |
| `learning_rate` | 0.05 | [0.005, 0.3] | log-uniform | "eta" -- shrinkage |
| `subsample` | 0.8 | [0.5, 1.0] | uniform | Row sampling per tree |
| `colsample_bytree` | 0.8 | [0.3, 1.0] | uniform | Column sampling per tree |
| `min_child_weight` | 1 | [1, 10] | int | Minimum sum of hessian in leaf |
| `gamma` | 0 | [0, 5.0] | uniform | Minimum loss reduction for split |
| `reg_alpha` | 0 | [1e-8, 10.0] | log-uniform | L1 regularization |
| `reg_lambda` | 1 | [1e-8, 10.0] | log-uniform | L2 regularization |

```python
def suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 90, 120]),
    }
```

### 4.10 LightGBM

| Parameter | Current | Sweep Range | Scale |
|-----------|---------|-------------|-------|
| `n_estimators` | 200 | [100, 2000] | int |
| `max_depth` | 6 | [3, 12] | int |
| `learning_rate` | 0.05 | [0.005, 0.3] | log-uniform |
| `num_leaves` | 31 (default) | [15, 127] | int |
| `subsample` | 0.8 | [0.5, 1.0] | uniform |
| `colsample_bytree` | 0.8 | [0.3, 1.0] | uniform |
| `min_child_samples` | 20 (default) | [5, 100] | int |
| `reg_alpha` | 0 | [1e-8, 10.0] | log-uniform |
| `reg_lambda` | 0 | [1e-8, 10.0] | log-uniform |
| `min_gain_to_split` | 0 | [0, 1.0] | uniform |

```python
def suggest_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 90, 120]),
    }
```

### 4.11 CatBoost

| Parameter | Current | Sweep Range | Scale |
|-----------|---------|-------------|-------|
| `iterations` | 200 | [100, 2000] | int |
| `depth` | 6 | [4, 10] | int |
| `learning_rate` | 0.05 | [0.005, 0.3] | log-uniform |
| `l2_leaf_reg` | 3 (default) | [1, 30] | uniform |
| `random_strength` | 1 (default) | [0.1, 10.0] | log-uniform |
| `bagging_temperature` | 1 (default) | [0, 5.0] | uniform |
| `border_count` | 254 (default) | [32, 255] | int |

```python
def suggest_catboost(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5.0),
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 90, 120]),
    }
```

---

## 5. Walk-Forward Aware Tuning (Nested Cross-Validation)

### 5.1 The Data Leakage Problem

The most dangerous pitfall in financial ML hyperparameter tuning is data
leakage. In a standard ML pipeline, one might:

1. Tune hyperparameters using the validation set
2. Evaluate the best configuration on the test set

But in walk-forward evaluation, **the validation set is temporally adjacent to
the test set**. If hyperparameters are tuned to maximize performance on val,
information about the near-future market regime leaks into the model selection
process.

**The correct approach: nested walk-forward.**

### 5.2 Nested Walk-Forward Architecture

```
OUTER LOOP (unchanged -- the 7 existing folds for final evaluation)
==================================================================
For each outer fold:
  train_outer = [2005-01 ... train_end]
  val_outer   = [val_start ... val_end]      <-- 90-day purge gap
  test_outer  = [test_start ... test_end]    <-- 90-day purge gap

  INNER LOOP (hyperparameter tuning -- uses ONLY train_outer + val_outer)
  ======================================================================
  Split train_outer into 3 inner time-series folds:
    inner_fold_1: train_inner=[..., t1], val_inner=[t1+purge, t2]
    inner_fold_2: train_inner=[..., t2], val_inner=[t2+purge, t3]
    inner_fold_3: train_inner=[..., t3], val_inner=[t3+purge, t4]
  
  Optuna objective = mean Sharpe across inner folds
  
  Best hyperparameters from inner loop are used to train on
  full train_outer, validate on val_outer, and evaluate on test_outer.
```

### 5.3 Implementation

```python
import optuna
import numpy as np
from data.splits import FOLDS, get_fold_dates, PURGE_DAYS

def create_inner_folds(outer_fold: dict, n_inner: int = 3) -> list[dict]:
    """Split an outer fold's training period into inner time-series folds.
    
    Each inner fold uses an expanding window within the outer training period.
    Inner validation periods are separated by PURGE_DAYS from inner training.
    
    CRITICAL: Inner folds NEVER touch the outer val or test periods.
    """
    dates = get_fold_dates(outer_fold)
    train_start = dates["train_start"]
    # Inner folds must end BEFORE val_outer starts (minus purge)
    inner_end = dates["val_start"] - pd.Timedelta(days=PURGE_DAYS)
    
    total_days = (inner_end - train_start).days
    segment = total_days // (n_inner + 1)  # +1 because we need val after last train
    
    inner_folds = []
    for i in range(n_inner):
        inner_train_end = train_start + pd.Timedelta(days=segment * (i + 1))
        inner_val_start = inner_train_end + pd.Timedelta(days=PURGE_DAYS)
        inner_val_end = inner_val_start + pd.Timedelta(days=segment)
        
        # Ensure inner val doesn't exceed the allowed boundary
        if inner_val_end > inner_end:
            inner_val_end = inner_end
        
        if inner_val_start >= inner_val_end:
            continue  # Skip degenerate folds
        
        inner_folds.append({
            "name": f"inner_{i+1}",
            "train": {"start": str(train_start.date()), "end": str(inner_train_end.date())},
            "val": {"start": str(inner_val_start.date()), "end": str(inner_val_end.date())},
            "test": {"start": str(inner_val_start.date()), "end": str(inner_val_end.date())},
        })
    
    return inner_folds


def optuna_objective(
    trial: optuna.Trial,
    backbone: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    outer_fold: dict,
    suggest_fn: callable,
) -> float:
    """Optuna objective: mean Sharpe across inner walk-forward folds.
    
    This function is called by Optuna for each trial. It:
    1. Suggests hyperparameters via the backbone-specific suggest_fn
    2. Creates inner folds from the outer training period ONLY
    3. Trains and evaluates on each inner fold
    4. Reports intermediate results for pruning
    5. Returns mean inner Sharpe
    """
    params = suggest_fn(trial)
    inner_folds = create_inner_folds(outer_fold, n_inner=3)
    
    inner_sharpes = []
    for i, inner_fold in enumerate(inner_folds):
        # Train and evaluate on this inner fold using params
        sharpe = _train_and_evaluate_fold(backbone, features, targets, inner_fold, params)
        inner_sharpes.append(sharpe)
        
        # Report intermediate value for pruning
        trial.report(np.mean(inner_sharpes), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(inner_sharpes))


def run_hyperparameter_sweep(
    backbone: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    outer_fold: dict,
    n_trials: int = 100,
    timeout: int = 3600,  # seconds
) -> dict:
    """Run Optuna sweep for one backbone on one outer fold.
    
    Returns the best hyperparameter dict.
    """
    suggest_fn = BACKBONE_SUGGEST_FNS[backbone]  # dispatcher dict
    
    study = optuna.create_study(
        study_name=f"{backbone}_{outer_fold['name']}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,
            multivariate=True,
            seed=42,
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=2, max_resource=3, reduction_factor=3,
        ),
        storage="sqlite:///optuna_sweeps.db",  # persistence
        load_if_exists=True,
    )
    
    study.optimize(
        lambda trial: optuna_objective(
            trial, backbone, features, targets, outer_fold, suggest_fn,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,  # sequential for GPU models; increase for CPU-only
    )
    
    return study.best_params
```

### 5.4 Leakage Prevention Checklist

| Check | Description | How to verify |
|-------|-------------|---------------|
| **Temporal ordering** | Inner folds strictly chronological | Assert inner_fold[i].val_end < inner_fold[i+1].train_start |
| **Purge gap** | 90+ calendar days between all train/val boundaries | `validate_purge_embargo()` on inner folds |
| **Scaler isolation** | StandardScaler fit on inner train only | Verify scaler.fit() called per inner fold |
| **No test snooping** | Outer test set never seen during inner tuning | Inner folds reference only outer train period |
| **Feature computation** | Features computed before splitting | No forward-looking features (already guaranteed by design) |
| **Random seed** | Fixed seed for reproducibility | Set in sampler + model initialization |

### 5.5 Deflated Sharpe Ratio for Multiple Trials

When running N Optuna trials, the reported "best Sharpe" is inflated by
selection bias. Use the Deflated Sharpe Ratio (DSR) to correct for this:

```python
from evaluation.metrics import deflated_sharpe_ratio

# After sweep: the best trial's Sharpe must survive DSR correction
best_trial_returns = get_returns_for_trial(study.best_trial)
dsr = deflated_sharpe_ratio(best_trial_returns, n_trials=len(study.trials))

if dsr < 0.95:
    print(f"WARNING: Best Sharpe does not survive multiple-testing correction "
          f"(DSR={dsr:.3f}, n_trials={len(study.trials)})")
```

---

## 6. Foundation Model Fine-Tuning (LFM2.5)

### 6.1 Progressive Unfreezing Strategy

The LFM2.5-350M has ~354M parameters. Unfreezing all at once with the same
learning rate will catastrophically overwrite the pretrained representations.

**Three-phase progressive unfreezing:**

```
Phase 1: Head-only (current default)
  - Freeze ALL backbone parameters
  - Train projection layer + prediction heads
  - 5-10 epochs with lr=1e-4
  - This establishes a good head baseline

Phase 2: Last-N layers
  - Unfreeze last 2-4 transformer/LIV blocks
  - Discriminative LR: backbone_lr = head_lr * 0.01
  - 5-10 more epochs
  - Monitor validation loss for catastrophic forgetting

Phase 3: Full fine-tune (optional, high risk)
  - Unfreeze all layers
  - Layer-wise LR decay: lr_layer_i = lr_base * (decay_rate ^ (n_layers - i))
  - Very short training (2-3 epochs)
  - Only if Phase 2 shows clear improvement
```

### 6.2 Implementation: Discriminative Learning Rates

```python
def create_discriminative_optimizer(
    model: CurrencyLFM,
    head_lr: float = 1e-4,
    backbone_lr_factor: float = 0.01,
    weight_decay: float = 1e-5,
    layer_decay: float = 0.85,
) -> torch.optim.AdamW:
    """Create AdamW with per-layer learning rates for LFM2.5.
    
    Parameters
    ----------
    head_lr : float
        Learning rate for projection + prediction heads.
    backbone_lr_factor : float
        Multiplier applied to head_lr for backbone layers.
    layer_decay : float
        Multiplicative decay per layer from top to bottom.
        Layer N (topmost): lr = head_lr * backbone_lr_factor
        Layer N-1: lr = head_lr * backbone_lr_factor * layer_decay
        Layer 0 (bottom): lr = head_lr * backbone_lr_factor * layer_decay^N
    """
    param_groups = []
    
    # 1. Prediction heads (highest LR)
    param_groups.append({
        "params": list(model.heads.parameters()),
        "lr": head_lr,
        "weight_decay": weight_decay,
        "name": "heads",
    })
    
    # 2. Projection layer (high LR -- it maps financial features to LFM space)
    param_groups.append({
        "params": list(model.projection.parameters()),
        "lr": head_lr,
        "weight_decay": weight_decay,
        "name": "projection",
    })
    
    # 3. Backbone layers (low LR with layer-wise decay)
    backbone_base_lr = head_lr * backbone_lr_factor
    backbone_layers = list(model.backbone.named_parameters())
    n_layers = len(set(name.split(".")[1] for name, _ in backbone_layers if "." in name))
    
    for name, param in backbone_layers:
        if not param.requires_grad:
            continue
        # Estimate layer depth from parameter name
        parts = name.split(".")
        depth = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        layer_lr = backbone_base_lr * (layer_decay ** (n_layers - depth))
        param_groups.append({
            "params": [param],
            "lr": layer_lr,
            "weight_decay": weight_decay,
            "name": f"backbone.{name}",
        })
    
    return torch.optim.AdamW(param_groups)
```

### 6.3 Learning Rate Warmup

For fine-tuning pretrained models, a linear warmup prevents large gradient
updates from damaging representations in early training when the head is still
randomly initialized:

```python
def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Linear warmup followed by cosine annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps,
    )
    
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )
```

### 6.4 LFM2.5 Sweep Ranges for Fine-Tuning

| Phase | Parameter | Range | Notes |
|-------|-----------|-------|-------|
| 1 (frozen) | `head_lr` | [1e-5, 1e-3] | Standard neural HP sweep |
| 1 (frozen) | `projection_dim` | {512, 1024} | Must match backbone hidden_size |
| 2 (partial) | `unfreeze_n_layers` | {1, 2, 4, 6} | Last N blocks unfrozen |
| 2 (partial) | `backbone_lr_factor` | [0.001, 0.05] | Relative to head_lr |
| 2 (partial) | `layer_decay` | [0.75, 0.95] | Per-layer LR decay |
| 2 (partial) | `warmup_epochs` | {1, 2, 3} | Linear warmup duration |
| 3 (full) | `lr` | [1e-6, 1e-4] | Very conservative |
| 3 (full) | `layer_decay` | [0.65, 0.85] | Stronger decay for deep layers |

---

## 7. Feature Selection as Hyperparameter

### 7.1 Motivation

The current feature set has ~104 features. In financial ML, many features are
noise, and including them degrades model performance through the "curse of
dimensionality" and increased overfitting risk. Feature selection should be
treated as a hyperparameter.

### 7.2 Approaches

#### 7.2.1 LASSO / Elastic-Net Screening

Run a LASSO regression on the inner training set. Features with non-zero
coefficients survive:

```python
from sklearn.linear_model import LassoCV, ElasticNetCV

def lasso_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    alpha: float = None,  # None = auto via cross-validation
) -> list[str]:
    """Select features using LASSO (L1) regularization.
    
    Features with non-zero coefficients are retained.
    """
    if alpha is None:
        model = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
    else:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=alpha, max_iter=10000)
    
    model.fit(X_train, y_train)
    mask = np.abs(model.coef_) > 1e-8
    selected = [name for name, keep in zip(feature_names, mask) if keep]
    
    return selected if len(selected) >= 5 else feature_names  # fallback
```

#### 7.2.2 Mutual Information

Non-linear feature importance measure:

```python
from sklearn.feature_selection import mutual_info_regression

def mi_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
) -> list[str]:
    """Select top-k features by mutual information with target."""
    mi_scores = mutual_info_regression(X_train, y_train, n_neighbors=10, random_state=42)
    top_indices = np.argsort(mi_scores)[::-1][:top_k]
    return [feature_names[i] for i in top_indices]
```

#### 7.2.3 SHAP-Based Selection

Uses model-specific feature importance from a trained GBM:

```python
def shap_feature_selection(
    model,  # fitted XGBRegressor or LGBMRegressor
    X_train: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
) -> list[str]:
    """Select features by mean absolute SHAP value.
    
    Requires: pip install shap
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:500])  # subsample for speed
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    return [feature_names[i] for i in top_indices]
```

### 7.3 Integration with Optuna

Feature selection method and its parameters become trial hyperparameters:

```python
def suggest_feature_selection(trial: optuna.Trial) -> dict:
    method = trial.suggest_categorical("feat_select_method", ["none", "lasso", "mi", "elastic_net"])
    params = {"method": method}
    
    if method == "lasso":
        params["alpha"] = trial.suggest_float("lasso_alpha", 1e-5, 1.0, log=True)
    elif method == "elastic_net":
        params["alpha"] = trial.suggest_float("enet_alpha", 1e-5, 1.0, log=True)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
    elif method == "mi":
        params["top_k"] = trial.suggest_int("mi_top_k", 20, 80)
    
    return params
```

---

## 8. Sequence Length Optimization

### 8.1 Why Sequence Length Matters for FX

Sequence length (lookback window) determines how much historical context the
model sees. In FX markets:

- **Too short (< 20 days):** Loses momentum and mean-reversion signals; misses
  regime context.
- **Too long (> 120 days):** Includes stale data that dilutes recent signals;
  increases memory/compute; may not fit within training sample sizes for early
  folds.
- **The "memory" of FX markets:** Academic literature (Cont, 2001) shows FX
  autocorrelation decays rapidly (~5 days for returns, ~60 days for volatility).

### 8.2 Empirical Approach

Treat `seq_len` as a categorical hyperparameter and sweep over:

```python
seq_len_candidates = [20, 40, 60, 90, 120]
```

**Interaction with patch-based models:** For PatchTST and PatchTSMixer, the
sequence length must be compatible with the patch length (seq_len should ideally
be a multiple of patch_length). Add a constraint:

```python
# In the suggest function:
seq_len = trial.suggest_categorical("seq_len", [20, 40, 60, 90, 120])
patch_length = trial.suggest_categorical("patch_length", [p for p in [5, 10, 12, 15, 20] if seq_len % p == 0])
```

### 8.3 Horizon-Dependent Lookback

For the 5-day return target, a longer lookback may be optimal than for the 1-day
target. Consider training separate models per horizon with different seq_len
values, or using multi-task loss weighting:

```python
# Multi-task loss weighting as a hyperparameter
w_1d = trial.suggest_float("loss_weight_1d", 0.3, 0.7)
w_5d = 1.0 - w_1d
loss = w_1d * criterion(pred_1d, y_1d) + w_5d * criterion(pred_5d, y_5d)
```

### 8.4 Minimum Sample Size Constraint

With expanding window walk-forward, early folds have fewer training samples. A
seq_len of 120 with fold_1 (training on 2005-2006, ~500 trading days) leaves
only ~380 usable samples. Set a minimum:

```python
min_samples_per_fold = 200  # hard floor
if (train_size - seq_len) < min_samples_per_fold:
    raise optuna.TrialPruned()  # skip this seq_len for this fold
```

---

## 9. Loss Function Alternatives

### 9.1 Current: Huber Loss

```python
criterion = torch.nn.HuberLoss(delta=1.0)
```

Huber loss is a reasonable default for financial returns (heavy-tailed,
occasionally extreme outliers). However, it treats positive and negative errors
symmetrically, which may not be optimal for a trading system.

### 9.2 Alternatives to Sweep

| Loss | Formula | Use Case | Sweep Parameter |
|------|---------|----------|-----------------|
| **Huber** | L2 for small errors, L1 for large | General robustness | `delta`: [0.1, 5.0] |
| **Quantile** | Asymmetric L1 at quantile q | Predict specific quantile | `quantile`: [0.1, 0.9] |
| **Pinball** | Same as quantile, FX convention | Risk budgeting | `tau`: [0.1, 0.9] |
| **Asymmetric Huber** | Huber with different deltas for +/- errors | Penalize missed moves more | `delta_pos`, `delta_neg` |
| **MSE** | Standard L2 | If outliers are clipped | None |
| **Sharpe-aware** | -Sharpe(predictions) | Directly optimize target metric | None (non-differentiable proxy) |
| **IC loss** | -corr(pred, actual) | Maximize rank correlation | None |

### 9.3 Implementation: Custom Loss Functions

```python
class AsymmetricHuberLoss(torch.nn.Module):
    """Huber loss with different thresholds for positive and negative errors.
    
    In FX trading, missing a large move (underestimating magnitude) may be
    worse than overestimating a small move. This loss allows penalizing
    one direction more than the other.
    """
    def __init__(self, delta_pos: float = 1.0, delta_neg: float = 0.5):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        delta = torch.where(diff > 0, self.delta_pos, self.delta_neg)
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        return torch.mean(0.5 * quadratic ** 2 + delta * linear)


class QuantileLoss(torch.nn.Module):
    """Quantile (pinball) loss for conditional quantile regression."""
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        return torch.mean(torch.max(self.quantile * diff, (self.quantile - 1) * diff))


class ICLoss(torch.nn.Module):
    """Negative Spearman rank correlation as a differentiable loss.
    
    Uses soft ranking (differentiable approximation to argsort).
    Maximizing IC directly aligns model training with the evaluation metric.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable ranking using softmax over pairwise comparisons."""
        diff = x.unsqueeze(-1) - x.unsqueeze(-2)
        return torch.sigmoid(diff / self.temperature).sum(dim=-1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rank = self._soft_rank(pred)
        target_rank = self._soft_rank(target)
        # Pearson correlation of ranks (approximates Spearman)
        pred_centered = pred_rank - pred_rank.mean()
        target_centered = target_rank - target_rank.mean()
        corr = (pred_centered * target_centered).sum() / (
            torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-8
        )
        return -corr  # negative because we minimize
```

### 9.4 Loss Function Sweep

```python
def suggest_loss(trial: optuna.Trial) -> dict:
    loss_type = trial.suggest_categorical("loss", ["huber", "mse", "asymmetric_huber", "quantile", "ic"])
    params = {"type": loss_type}
    
    if loss_type == "huber":
        params["delta"] = trial.suggest_float("huber_delta", 0.1, 5.0)
    elif loss_type == "asymmetric_huber":
        params["delta_pos"] = trial.suggest_float("delta_pos", 0.3, 3.0)
        params["delta_neg"] = trial.suggest_float("delta_neg", 0.3, 3.0)
    elif loss_type == "quantile":
        params["quantile"] = trial.suggest_float("quantile", 0.3, 0.7)
    elif loss_type == "ic":
        params["temperature"] = trial.suggest_float("ic_temperature", 0.01, 1.0, log=True)
    
    return params
```

---

## 10. Regularization Sweeps

### 10.1 Dropout

Dropout rates should be swept independently for different model components:

| Component | Current | Sweep Range | Notes |
|-----------|---------|-------------|-------|
| Body/backbone dropout | 0.1 | [0.0, 0.5] | Between backbone layers |
| Head dropout | 0.1 | [0.0, 0.3] | In prediction heads |
| Attention dropout | 0.0 (default) | [0.0, 0.2] | PatchTST, Informer only |
| Recurrent dropout | 0.1 | [0.0, 0.3] | LSTM inter-layer |

### 10.2 Weight Decay

Weight decay (L2 regularization) interacts with learning rate. In AdamW, weight
decay is decoupled from the gradient update, making it more interpretable:

```python
# Sweep
weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
```

**Rule of thumb for financial data:** Start with weight_decay in [1e-5, 1e-3].
Financial features are standardized (mean=0, std=1), so weight decay penalizes
large weights that amplify noisy features.

### 10.3 Label Smoothing for Regression

While label smoothing is typically used for classification, an analogous
technique for regression adds Gaussian noise to targets during training:

```python
class NoisyTargetRegularization:
    """Add Gaussian noise to regression targets during training.
    
    Acts as a form of regularization by preventing the model from
    fitting exact target values (which may be noisy in financial data).
    """
    def __init__(self, noise_std: float = 0.001):
        self.noise_std = noise_std
    
    def __call__(self, targets: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(targets) * self.noise_std
            return targets + noise
        return targets

# Sweep parameter
noise_std = trial.suggest_float("target_noise", 0.0, 0.01)
```

### 10.4 Mixup for Time Series

Mixup creates virtual training examples by interpolating between pairs of
samples. For time series, this must be done carefully to preserve temporal
structure:

```python
class TimeSeriesMixup:
    """Mixup augmentation adapted for time-series sliding windows.
    
    Interpolates between pairs of (sequence, target) within the same batch.
    Only applies during training.
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        if self.alpha <= 0:
            return x, y
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y

# Sweep parameter
mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.4)
```

### 10.5 Gradient Clipping

```python
grad_clip = trial.suggest_float("grad_clip", 0.1, 5.0)
```

Lower values provide stronger regularization against gradient spikes, which are
common in financial time series (regime changes, flash crashes).

### 10.6 Stochastic Weight Averaging (SWA)

SWA averages model weights over the tail of training, producing a flatter
minimum that generalizes better:

```python
use_swa = trial.suggest_categorical("use_swa", [True, False])
if use_swa:
    swa_start_epoch = trial.suggest_int("swa_start", 3, max(epochs - 2, 3))
    swa_lr = trial.suggest_float("swa_lr", 1e-5, lr)
```

---

## 11. Ensemble Hyperparameters

### 11.1 Stacking Weights

The ablation study produces predictions from multiple backbones. The ensemble
weights for blending these predictions are themselves hyperparameters:

```python
def suggest_ensemble_weights(trial: optuna.Trial, backbones: list[str]) -> dict:
    """Suggest blend weights for model ensemble using softmax parameterization.
    
    We parameterize N-1 logits and compute weights via softmax to ensure
    they sum to 1 and are all non-negative.
    """
    raw_weights = {}
    for backbone in backbones:
        raw_weights[backbone] = trial.suggest_float(f"w_{backbone}", -3.0, 3.0)
    
    # Softmax normalization
    import torch.nn.functional as F
    logits = torch.tensor(list(raw_weights.values()))
    weights = F.softmax(logits, dim=0).numpy()
    
    return {b: float(w) for b, w in zip(backbones, weights)}
```

### 11.2 Meta-Learner for Stacking

Instead of fixed weights, train a meta-learner on out-of-fold predictions:

```python
def train_stacking_meta_learner(
    oof_predictions: dict[str, np.ndarray],  # backbone -> out-of-fold preds
    oof_actuals: np.ndarray,
    method: str = "ridge",  # or "lasso", "elastic_net", "xgboost"
) -> object:
    """Train a meta-learner on out-of-fold predictions.
    
    CRITICAL: Use only out-of-fold predictions to avoid leakage.
    Each backbone's predictions must come from models that never saw
    the data being predicted.
    """
    X_meta = np.column_stack(list(oof_predictions.values()))
    
    if method == "ridge":
        from sklearn.linear_model import RidgeCV
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    elif method == "lasso":
        from sklearn.linear_model import LassoCV
        meta = LassoCV(cv=3)
    elif method == "xgboost":
        from xgboost import XGBRegressor
        meta = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    
    meta.fit(X_meta, oof_actuals)
    return meta
```

### 11.3 Ensemble Sweep Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| `ensemble_method` | {mean, weighted, stacking_ridge, stacking_xgb} | Complexity level |
| `n_top_models` | {3, 5, 7, all} | How many backbones to include |
| `diversity_penalty` | [0.0, 0.5] | Penalize correlated model pairs |
| `regime_conditional` | {True, False} | Different weights per market regime |

---

## 12. Bayesian Optimization for Sharpe Ratio

### 12.1 The Non-Differentiability Challenge

Sharpe ratio is non-differentiable and noisy, making it unsuitable for
gradient-based optimization. This is why we use TPE (which models p(x|y) and
does not require gradients of the objective).

### 12.2 Noise Reduction Strategies

Sharpe ratio estimates from finite samples are noisy. Strategies to reduce noise
in the Optuna objective:

1. **Use all inner folds:** Average Sharpe across 3 inner folds instead of
   using a single fold. This reduces variance of the objective by ~sqrt(3).

2. **Minimum sample size:** Ensure each inner fold produces at least 60 trading
   days of predictions (about 3 months). Sharpe estimates from shorter periods
   are unreliable.

3. **Bootstrapped Sharpe:** Instead of point-estimate Sharpe, use the mean of
   100 bootstrap resamples:

   ```python
   def bootstrapped_sharpe(returns: np.ndarray, n_bootstrap: int = 100) -> float:
       sharpes = []
       n = len(returns)
       for _ in range(n_bootstrap):
           idx = np.random.choice(n, size=n, replace=True)
           sharpes.append(sharpe_ratio(returns[idx]))
       return float(np.mean(sharpes))
   ```

4. **Penalized Sharpe:** Subtract a penalty proportional to the standard error
   of the Sharpe estimate, biasing selection toward configurations with more
   stable performance:

   ```python
   def penalized_sharpe(returns: np.ndarray, penalty: float = 0.5) -> float:
       sr = sharpe_ratio(returns)
       # Standard error of Sharpe (Lo, 2002)
       n = len(returns)
       se = np.sqrt((1 + 0.5 * sr ** 2) / n)
       return sr - penalty * se
   ```

### 12.3 Surrogate Objective

If direct Sharpe optimization is too noisy, use a smooth surrogate that
correlates with Sharpe:

```python
def surrogate_objective(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Composite surrogate for Sharpe ratio.
    
    Combines IC (smooth, high correlation with Sharpe) with a
    directional accuracy term.
    """
    # Rank IC
    from scipy.stats import spearmanr
    ic, _ = spearmanr(predictions, actuals)
    
    # Directional accuracy
    hit_rate = np.mean(np.sign(predictions) == np.sign(actuals))
    
    # Weighted combination (IC is more predictive of out-of-sample Sharpe)
    return 0.7 * ic + 0.3 * (hit_rate - 0.5) * 2  # normalize hit_rate to [-1, 1]
```

---

## 13. Multi-Objective Optimization

### 13.1 Objectives

For a production trading system, optimizing Sharpe alone is insufficient. A
model with Sharpe=2.0 but max_drawdown=50% is unusable. Optimize simultaneously:

| Objective | Direction | Weight | Rationale |
|-----------|-----------|--------|-----------|
| Sharpe ratio | maximize | Primary | Risk-adjusted return |
| Max drawdown | minimize | Secondary | Tail risk control |
| Information coefficient | maximize | Tertiary | Predictive quality |
| PSR (vs 0) | maximize | Validation | Statistical significance |

### 13.2 Implementation with Optuna NSGA-II

```python
def multi_objective(trial: optuna.Trial, ...) -> tuple[float, float, float]:
    """Multi-objective: (Sharpe, -MaxDrawdown, IC).
    
    Returns a tuple of values to simultaneously optimize.
    Optuna's NSGA-II finds the Pareto front.
    """
    params = suggest_fn(trial)
    
    all_sharpes, all_drawdowns, all_ics = [], [], []
    
    for inner_fold in inner_folds:
        returns, predictions, actuals = train_and_evaluate(params, inner_fold)
        all_sharpes.append(sharpe_ratio(returns))
        all_drawdowns.append(max_drawdown(np.cumprod(1 + returns)))
        ic, _ = spearmanr(predictions, actuals)
        all_ics.append(ic)
    
    return (
        float(np.mean(all_sharpes)),        # maximize
        float(np.mean(all_drawdowns)),      # minimize
        float(np.mean(all_ics)),            # maximize
    )


study = optuna.create_study(
    directions=["maximize", "minimize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler(
        population_size=50,
        seed=42,
    ),
)
study.optimize(multi_objective, n_trials=200)

# Extract Pareto-optimal trials
pareto_trials = study.best_trials
for t in pareto_trials:
    print(f"Sharpe={t.values[0]:.3f}, MaxDD={t.values[1]:.3f}, IC={t.values[2]:.3f}")
```

### 13.3 Selecting from the Pareto Front

After obtaining the Pareto front, select the final configuration using a
scalarization function:

```python
def select_from_pareto(
    pareto_trials: list,
    sharpe_weight: float = 0.5,
    drawdown_weight: float = 0.3,
    ic_weight: float = 0.2,
    max_acceptable_drawdown: float = 0.15,  # 15% hard constraint
) -> optuna.Trial:
    """Select a single trial from the Pareto front.
    
    Applies a hard constraint on drawdown, then ranks by weighted score.
    """
    candidates = [
        t for t in pareto_trials
        if t.values[1] <= max_acceptable_drawdown  # drawdown constraint
    ]
    
    if not candidates:
        # Relax constraint: pick lowest drawdown trial
        return min(pareto_trials, key=lambda t: t.values[1])
    
    # Normalize objectives to [0, 1] range
    sharpes = [t.values[0] for t in candidates]
    dds = [t.values[1] for t in candidates]
    ics = [t.values[2] for t in candidates]
    
    def _normalize(vals, higher_better=True):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        normed = [(v - mn) / (mx - mn) for v in vals]
        return normed if higher_better else [1 - n for n in normed]
    
    norm_sharpe = _normalize(sharpes, higher_better=True)
    norm_dd = _normalize(dds, higher_better=False)
    norm_ic = _normalize(ics, higher_better=True)
    
    scores = [
        sharpe_weight * s + drawdown_weight * d + ic_weight * i
        for s, d, i in zip(norm_sharpe, norm_dd, norm_ic)
    ]
    
    best_idx = np.argmax(scores)
    return candidates[best_idx]
```

---

## 14. Implementation Plan

### 14.1 File Structure

```
autoresearch/
  optimizer/
    sweeps.py              # Main sweep orchestration (NEW)
    sweep_objectives.py    # Per-backbone objective functions (NEW)
    sweep_suggest.py       # Parameter suggestion functions (NEW)
    sweep_losses.py        # Custom loss functions (NEW)
    sweep_features.py      # Feature selection methods (NEW)
    agent_loop.py          # Existing autonomous optimizer
  model/
    backbone.py            # MODIFY: accept hyperparameter dicts
    train.py               # MODIFY: accept loss, scheduler, regularization params
  evaluation/
    metrics.py             # No changes needed
  data/
    splits.py              # MODIFY: add create_inner_folds()
    features.py            # No changes needed
  optuna_sweeps.db         # Optuna study persistence (auto-created)
```

### 14.2 Core Orchestrator: `optimizer/sweeps.py`

```python
"""Hyperparameter sweep orchestration using Optuna.

Usage:
    python -m optimizer.sweeps --backbone patchtst --n-trials 100
    python -m optimizer.sweeps --backbone xgboost --n-trials 200
    python -m optimizer.sweeps --backbone lfm2-350m --n-trials 50 --multi-objective
    python -m optimizer.sweeps --all --n-trials 50  # sweep all backbones
"""

import argparse
import json
from pathlib import Path

import optuna
import numpy as np
import pandas as pd

from data.download import download_all_pairs, download_macro_signals
from data.features import compute_all_features, compute_targets
from data.splits import FOLDS
from evaluation.metrics import sharpe_ratio, deflated_sharpe_ratio

from optimizer.sweep_suggest import BACKBONE_SUGGEST_FNS
from optimizer.sweep_objectives import single_objective, multi_objective

RESULTS_DIR = Path(__file__).resolve().parent.parent / "sweep_results"
DB_PATH = Path(__file__).resolve().parent.parent / "optuna_sweeps.db"


def run_sweep(
    backbone: str,
    n_trials: int = 100,
    timeout: int = 7200,
    multi_obj: bool = False,
) -> dict:
    """Run hyperparameter sweep for one backbone across all outer folds.
    
    For each outer fold:
    1. Create inner folds from the outer training period
    2. Run Optuna with inner-fold Sharpe as objective
    3. Use best HPs to train on full outer train, evaluate on outer test
    4. Aggregate results and compute DSR
    """
    # Load data
    all_pairs = download_all_pairs()
    macro_data = download_macro_signals()
    features = compute_all_features(all_pairs, macro_data)
    targets = compute_targets(all_pairs["EURUSD=X"])
    
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    
    suggest_fn = BACKBONE_SUGGEST_FNS[backbone]
    fold_results = []
    
    for outer_fold in FOLDS:
        print(f"\n{'='*50}")
        print(f"Outer fold: {outer_fold['name']} ({outer_fold['regime']})")
        print(f"{'='*50}")
        
        study_name = f"sweep_{backbone}_{outer_fold['name']}"
        
        if multi_obj:
            study = optuna.create_study(
                study_name=study_name,
                directions=["maximize", "minimize", "maximize"],
                sampler=optuna.samplers.NSGAIISampler(seed=42),
                storage=f"sqlite:///{DB_PATH}",
                load_if_exists=True,
            )
            study.optimize(
                lambda trial: multi_objective(
                    trial, backbone, features, targets, outer_fold, suggest_fn,
                ),
                n_trials=n_trials,
                timeout=timeout,
            )
            best_params = select_from_pareto(study.best_trials).params
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=20, multivariate=True, seed=42,
                ),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=2, max_resource=3, reduction_factor=3,
                ),
                storage=f"sqlite:///{DB_PATH}",
                load_if_exists=True,
            )
            study.optimize(
                lambda trial: single_objective(
                    trial, backbone, features, targets, outer_fold, suggest_fn,
                ),
                n_trials=n_trials,
                timeout=timeout,
            )
            best_params = study.best_params
        
        # Evaluate best HPs on outer test set
        outer_result = evaluate_outer_fold(backbone, features, targets, outer_fold, best_params)
        outer_result["best_params"] = best_params
        outer_result["n_trials"] = len(study.trials)
        fold_results.append(outer_result)
        
        print(f"  Best inner Sharpe: {study.best_value:.4f}")
        print(f"  Outer test Sharpe: {outer_result['sharpe']:.4f}")
        print(f"  Trials: {len(study.trials)}")
    
    # Aggregate
    all_returns = np.concatenate([r["returns"] for r in fold_results if r.get("returns") is not None])
    total_trials = sum(r["n_trials"] for r in fold_results)
    
    summary = {
        "backbone": backbone,
        "avg_sharpe": float(np.mean([r["sharpe"] for r in fold_results])),
        "dsr": deflated_sharpe_ratio(all_returns, n_trials=total_trials),
        "fold_results": fold_results,
    }
    
    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{backbone}_sweep.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--multi-objective", action="store_true")
    args = parser.parse_args()
    
    run_sweep(args.backbone, args.n_trials, args.timeout, args.multi_objective)
```

### 14.3 Modifications to Existing Code

#### 14.3.1 `model/backbone.py` Changes

The `create_model()` factory needs to accept a hyperparameter dict:

```python
def create_model(
    backbone: str,
    n_input_features: int,
    seq_len: int = 60,
    freeze_backbone: bool = True,
    hparams: dict | None = None,  # NEW: override defaults with sweep results
) -> nn.Module | GBMWrapper:
    hparams = hparams or {}
    
    if backbone == "mlp":
        return CurrencyMLP(
            n_input_features, seq_len=seq_len,
            hidden_size=hparams.get("hidden_size", 512),
        )
    # ... similar for all backbones
```

#### 14.3.2 `model/train.py` Changes

The training loop needs to accept loss function, scheduler, and regularization
parameters:

```python
def train_one_fold(
    model,
    train_features, train_targets,
    val_features, val_targets,
    scaler=None,
    seq_len=SEQ_LEN,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    loss_config: dict | None = None,      # NEW
    scheduler_config: dict | None = None,  # NEW
    regularization: dict | None = None,    # NEW
) -> dict:
    # ... existing setup ...
    
    # Loss function selection
    loss_config = loss_config or {"type": "huber", "delta": 1.0}
    criterion = _create_loss(loss_config)
    
    # Scheduler selection
    scheduler = _create_scheduler(optimizer, scheduler_config, epochs)
    
    # Regularization (mixup, target noise)
    reg = regularization or {}
    mixup = TimeSeriesMixup(alpha=reg.get("mixup_alpha", 0.0))
    target_noise = reg.get("target_noise", 0.0)
```

#### 14.3.3 `data/splits.py` Changes

Add `create_inner_folds()` function (see Section 5.3 above).

---

## 15. Pitfalls and Anti-Patterns

### 15.1 Data Leakage (Most Critical)

| Pitfall | Description | Prevention |
|---------|-------------|------------|
| **Tuning on test set** | Using outer test performance to select HPs | Strict nested CV; inner folds never touch outer val/test |
| **Scaler leakage** | Fitting scaler on train+val data | Fit scaler on inner-train only; transform inner-val |
| **Feature leakage** | Computing features using future data | Already prevented by backward-looking feature design |
| **Temporal leakage** | Training on data that follows val/test data | Chronological splits with purge gaps enforced |
| **Cross-fold leakage** | Using fold i's test data in fold j's training | Purge (90d) + embargo (21d) already enforce this |
| **Selection bias** | Reporting best-of-N-trials Sharpe without correction | Use DSR with n_trials = total Optuna trials |

### 15.2 Overfitting to Noise

| Pitfall | Description | Prevention |
|---------|-------------|------------|
| **Too many trials** | With 1000+ trials, even random configs "find" good Sharpe by chance | Limit to 100-200 trials per backbone; check DSR |
| **Too many HPs** | Sweeping 20+ parameters simultaneously | Sweep in stages (Section 16); fix unimportant HPs |
| **Small val sets** | Inner validation sets with < 60 days | Enforce minimum sample sizes |
| **Sharpe from few trades** | High Sharpe from 10 trades means nothing | Require >= 100 trades per evaluation |

### 15.3 Computational Waste

| Pitfall | Description | Prevention |
|---------|-------------|------------|
| **No pruning** | Running all 7 folds for every trial | Hyperband pruning after 3 folds |
| **Redundant retraining** | Retraining from scratch for each trial | Cache data loading; only retrain model |
| **Sequential execution** | Running GBM sweeps on GPU one at a time | Parallelize CPU-bound GBM trials with `n_jobs` |
| **No persistence** | Losing progress on crash | SQLite storage; Optuna `load_if_exists=True` |

### 15.4 Foundation Model Fine-Tuning

| Pitfall | Description | Prevention |
|---------|-------------|------------|
| **Catastrophic forgetting** | Unfreezing all layers with high LR | Progressive unfreezing; discriminative LR |
| **Head instability** | Random head + pretrained backbone mismatch | Warmup phase: train head first, then unfreeze |
| **Memory overflow** | 354M params + gradients + optimizer states | Gradient checkpointing; mixed precision |
| **Slow convergence** | Backbone LR too low | Sweep backbone_lr_factor in [0.001, 0.1] |

### 15.5 GBM-Specific

| Pitfall | Description | Prevention |
|---------|-------------|------------|
| **Feature flattening** | seq_len * n_features = 7200 columns | Consider feature aggregation (mean, std over window) |
| **n_estimators vs. early stopping** | Fixed 2000 trees wastes time | Use `early_stopping_rounds=50` instead of fixed count |
| **GPU memory** | Large XGBoost histograms on GPU | `tree_method="hist"` with `max_bin=256` |

---

## 16. Priority Ranking of Sweeps

### 16.1 Phase 1: Quick Wins (1-2 days)

These sweeps have the highest expected impact per compute hour:

| Priority | Sweep | Backbone(s) | Trials | Est. Time | Expected Impact |
|----------|-------|-------------|--------|-----------|-----------------|
| **P0** | Learning rate + weight decay | All neural | 50 each | 4-8h per backbone | HIGH -- LR is almost always the most impactful HP |
| **P1** | GBM full sweep | XGBoost, LightGBM, CatBoost | 200 each | 1-2h total | HIGH -- GBMs are fast to train; wide search space |
| **P2** | Sequence length | All | 5 values x 7 folds | 3-6h | MEDIUM -- directly affects signal quality |

### 16.2 Phase 2: Architecture Tuning (2-3 days)

| Priority | Sweep | Backbone(s) | Trials | Est. Time | Expected Impact |
|----------|-------|-------------|--------|-----------|-----------------|
| **P3** | Hidden size + depth | PatchTST, Mamba2, PatchTSMixer | 50 each | 6-12h | MEDIUM |
| **P4** | Patch length + stride | PatchTST, PatchTSMixer | 30 each | 4-6h | MEDIUM |
| **P5** | Loss function | Best 3 neural backbones | 30 each | 4-6h | MEDIUM -- Huber may not be optimal |

### 16.3 Phase 3: Regularization and Fine-Tuning (3-5 days)

| Priority | Sweep | Backbone(s) | Trials | Est. Time | Expected Impact |
|----------|-------|-------------|--------|-----------|-----------------|
| **P6** | Dropout + regularization | All neural | 30 each | 4-8h | LOW-MEDIUM |
| **P7** | LFM2.5 progressive unfreezing | LFM2.5-350M | 30 | 12-24h | MEDIUM-HIGH (but expensive) |
| **P8** | Feature selection | All | 30 each | 6-12h | LOW-MEDIUM (features already curated) |

### 16.4 Phase 4: Ensemble and Polish (1-2 days)

| Priority | Sweep | Backbone(s) | Trials | Est. Time | Expected Impact |
|----------|-------|-------------|--------|-----------|-----------------|
| **P9** | Ensemble weights | Top 5 backbones | 50 | 2-4h (fast, uses cached predictions) | MEDIUM |
| **P10** | Multi-objective Pareto | Top 3 backbones | 100 each | 6-12h | LOW (refinement) |
| **P11** | Scheduler sweep | Top 3 neural | 20 each | 2-4h | LOW |

### 16.5 Decision Criteria for Stopping

Stop a sweep early if:

1. **DSR < 0.80** after 50 trials -- the search space is too large or the
   signal is too weak for this backbone.
2. **Improvement plateau** -- best trial hasn't improved in 30 consecutive trials
   (Optuna can detect this automatically).
3. **Inner-outer gap > 0.5 Sharpe** -- severe overfitting to inner folds; reduce
   search space or add regularization.
4. **Compute budget exhausted** -- move to the next backbone.

---

## 17. Monitoring and Visualization

### 17.1 Optuna Dashboard

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_sweeps.db
# Opens web UI at http://localhost:8080
```

The dashboard provides:
- Trial history and parameter importance
- Optimization history (best value over time)
- Parallel coordinate plots
- Hyperparameter slice plots
- Pareto front visualization (multi-objective)

### 17.2 Parameter Importance

After a sweep, extract the most important parameters:

```python
importance = optuna.importance.get_param_importances(study)
for param, score in importance.items():
    print(f"  {param}: {score:.3f}")
```

This reveals which hyperparameters actually matter, allowing future sweeps to
fix unimportant parameters and allocate budget to important ones.

### 17.3 Logging Integration

```python
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.INFO)

# Add callback for custom logging
def log_trial(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logging.info(
            f"Trial {trial.number}: Sharpe={trial.value:.4f}, "
            f"params={trial.params}"
        )

study.optimize(objective, n_trials=100, callbacks=[log_trial])
```

---

## 18. Reproducibility Protocol

### 18.1 Seed Management

```python
GLOBAL_SEED = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 18.2 Experiment Tracking

Each sweep run should record:

```json
{
    "backbone": "patchtst",
    "outer_fold": "fold_3",
    "n_trials": 100,
    "best_params": {"lr": 3e-4, "d_model": 256, ...},
    "best_inner_sharpe": 0.45,
    "outer_test_sharpe": 0.32,
    "dsr": 0.87,
    "timestamp": "2026-04-06T14:30:00",
    "git_hash": "abc1234",
    "compute_time_sec": 3600,
    "gpu": "RTX 4090",
    "seed": 42
}
```

---

## 19. Summary of Recommended Changes

| File | Change | Priority |
|------|--------|----------|
| `requirements.txt` | Add `optuna>=3.6`, `optuna-dashboard` | P0 |
| `data/splits.py` | Add `create_inner_folds()` | P0 |
| `model/backbone.py` | Accept `hparams` dict in `create_model()` | P0 |
| `model/train.py` | Accept loss_config, scheduler_config, regularization | P0 |
| `optimizer/sweeps.py` | New: main sweep orchestration | P0 |
| `optimizer/sweep_suggest.py` | New: per-backbone suggest functions | P0 |
| `optimizer/sweep_objectives.py` | New: objective functions with inner CV | P0 |
| `optimizer/sweep_losses.py` | New: custom loss functions | P5 |
| `optimizer/sweep_features.py` | New: feature selection methods | P8 |

---

## 20. References

1. **Optuna:** Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD.
2. **TPE:** Bergstra et al. (2011). "Algorithms for Hyper-Parameter Optimization." NeurIPS.
3. **Hyperband:** Li et al. (2018). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." JMLR.
4. **BOHB:** Falkner et al. (2018). "BOHB: Robust and Efficient Hyperparameter Optimization at Scale." ICML.
5. **Deflated Sharpe:** Bailey & Lopez de Prado (2014). "The Deflated Sharpe Ratio." Journal of Portfolio Management.
6. **Walk-forward CV:** Lopez de Prado (2018). "Advances in Financial Machine Learning." Wiley. Chapters 7, 12.
7. **Discriminative LR:** Howard & Ruder (2018). "Universal Language Model Fine-tuning for Text Classification." ACL.
8. **PatchTST:** Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR.
9. **Mamba:** Gu & Dao (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." ICML.
10. **NSGA-II:** Deb et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE Trans. EC.
