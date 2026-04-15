# 11 - Operations & Deployment

**SWEBoK Knowledge Area:** KA9 — Software Maintenance
**Google SWE Reference:** Ch. 24 — "Continuous Delivery"; Ch. 25 — "Compute as a Service"

---

## 1. Entry Points & CLI

### 1.1 Single Backbone Baseline

```bash
python baseline.py --backbone lfm2-350m --epochs 20
```

| Argument | Default | Options |
|----------|---------|---------|
| `--backbone` | lfm2-350m | mlp, lstm, lfm2-350m, patchtst, patchtsmixer, xgboost, lightgbm, catboost |
| `--epochs` | 20 | Any positive integer |
| `--seq-len` | Per backbone (60 for LFM, 10 for others) | Override with any positive integer |

**Output:**
- Console: Per-fold metrics + summary (20+ metrics)
- File: `baseline_results.json`
- File: `baseline_checkpoint.json` (deleted after success)

### 1.2 Full Ablation Study

```bash
# All 11 backbones
python run_ablation.py --epochs 5

# Subset of backbones
python run_ablation.py --backbones mlp lstm xgboost --epochs 3
```

**Output:**
- Console: Per-backbone progress
- Files: `ablation_results/{backbone}.json` (per-backbone)
- File: `ablation_results/ablation_combined.json`
- File: `reports/ablation_YYYYMMDD_HHMM.md`

### 1.3 Autonomous Optimizer

```bash
# Requires ANTHROPIC_API_KEY
python run_optimizer.py --max-experiments 12
python run_optimizer.py --baseline-only  # Just run baseline, no optimization
```

**Output:**
- Console: Experiment-by-experiment results
- File: `optimizer_state.json`
- Backup: `.optimizer_backups/`

### 1.4 Overnight Pipeline

```bash
python run_overnight.py
```

Runs baseline (2 epochs) + 10 optimizer experiments sequentially.

## 2. Crash Recovery

### 2.1 Baseline Checkpoint Recovery

**Scenario:** Machine crashes during fold 4 of 7.

**Recovery:**
1. Checkpoint file `baseline_checkpoint.json` contains folds 1-3 results
2. Re-run `python baseline.py` — automatically resumes from fold 4
3. Checkpoint includes backbone validation (won't load stale data from different backbone)
4. Checkpoint validates array length consistency (fold_returns == fold_train_sizes)

**Manual reset:** Delete `baseline_checkpoint.json` to force clean re-run.

### 2.2 Ablation Recovery

**Scenario:** Machine crashes during backbone 6 of 11.

**Recovery:**
1. Per-backbone results saved to `ablation_results/{backbone}.json` as each completes
2. However, `run_ablation.py` does NOT skip completed backbones on restart (re-runs all)
3. The per-backbone files serve as result preservation, not resume points

**Improvement needed:** Add resume logic to skip backbones with existing result files.

### 2.3 Optimizer Recovery

**Scenario:** Machine crashes during experiment 5.

**Recovery:**
1. `optimizer_state.json` contains experiments 1-4 results
2. `.optimizer_backups/` contains pre-experiment-5 file states
3. Re-run `python run_optimizer.py` — reads state, continues from experiment 5
4. If experiment 5's code change was partially applied, backups restore clean state

## 3. Output Artifacts

### 3.1 JSON Results

| File | Contents | When Created |
|------|----------|-------------|
| `baseline_results.json` | avg_sharpe, weighted_sharpe, PSR, DSR, IC, overall report, per-fold details | After successful baseline run |
| `baseline_checkpoint.json` | Per-fold progress (transient) | During baseline run (deleted after) |
| `ablation_results/{backbone}.json` | Per-backbone results or error | After each backbone evaluation |
| `ablation_results/ablation_combined.json` | All backbone results combined | After full ablation |
| `optimizer_state.json` | Iteration, sharpe history, experiment log | After each optimizer experiment |

### 3.2 Markdown Reports

| File | Contents | When Created |
|------|----------|-------------|
| `reports/ablation_YYYYMMDD_HHMM.md` | Summary table, per-backbone detail, conclusions, methodology | After full ablation |

### 3.3 Backup Files

| Directory | Contents | When Created |
|-----------|----------|-------------|
| `.optimizer_backups/` | Pre-experiment copies of backbone.py, features.py, train.py | Before each optimizer experiment |

## 4. Monitoring & Observability

### 4.1 Console Output

The system produces structured console output with clear section markers:

```
==========================================================
Fold: fold_1  |  Regime: Pre-crisis upturn + GFC onset
==========================================================
  Split sizes: train=480, val=126, test=53
  Test samples: 53  |  Sharpe: 0.7642  |  Sortino: 1.3396  | ...
  Checkpoint saved (1 folds done)
```

### 4.2 Logging

```python
import logging
logger = logging.getLogger(__name__)
```

Used in: data/download.py, optimizer/agent_loop.py
Level: INFO for progress, WARNING for non-critical failures, ERROR for failures

### 4.3 Progress Indicators

- `[i/N]` counters for ablation backbone progress
- `{'#'*70}` section headers for fold/backbone boundaries
- `{'='*60}` fold headers
- Checkpoint confirmation messages

## 5. Performance Characteristics

### 5.1 Timing Estimates (CPU-only)

| Operation | Time | Notes |
|-----------|------|-------|
| Data download (first run) | 30-60s | Yahoo Finance API, 15 tickers |
| Data download (cached) | <1s | Parquet reads |
| Feature computation | 2-5s | 104 features × ~5000 rows |
| MLP training (1 fold, 5 epochs) | 30-60s | Fast convergence |
| LSTM training (1 fold, 5 epochs) | 60-120s | Sequential computation |
| LFM2.5-350M (1 fold, 5 epochs) | 5-15min | 354M params, CPU inference |
| PatchTST/Mamba2 (1 fold, 5 epochs) | 2-5min | Moderate model size |
| XGBoost/LightGBM (1 fold) | 5-15s | Tree-based, fast |
| Full baseline (7 folds) | 10-90min | Depends on backbone |
| Full ablation (11 backbones) | 2-8hrs | Sequential, CPU-bound |

### 5.2 Memory Estimates

| Model | Peak RAM | Notes |
|-------|----------|-------|
| MLP | ~2GB | Small model + data |
| LSTM | ~2GB | Moderate hidden states |
| LFM2.5-350M | ~4-6GB | 354M params in float32 |
| LFM2.5-1.2B | ~8-12GB | May OOM on 8GB machines |
| PatchTST/Mamba2 | ~3GB | Moderate |
| GBM models | ~1-2GB | Tabular data, efficient |

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `shapes (7,) and (3,) not aligned` | Stale checkpoint from different backbone | Delete `baseline_checkpoint.json` |
| `CUDA out of memory` | Model too large for GPU | Reduce batch_size or use CPU |
| `ModuleNotFoundError: xgboost` | GBM libraries not installed | `pip install xgboost lightgbm catboost` |
| `Lfm2Model not found` | transformers version too old | `pip install transformers>=4.55` |
| `yfinance download returns empty` | Yahoo Finance API issues | Retry; check internet connection |
| `Checkpoint corrupt` | Array length mismatch in checkpoint | Delete checkpoint, re-run |
| Optimizer `SyntaxError` | Claude generated invalid Python | Automatically reverted; logged |
| Very slow training | No GPU, large model | Use `--backbone mlp` for fast iteration |
