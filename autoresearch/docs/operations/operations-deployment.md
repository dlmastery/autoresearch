# 11 - Operations & Deployment

**SWEBoK Knowledge Area:** KA9 — Software Maintenance
**Google SWE Reference:** Ch. 24 — "Continuous Delivery"; Ch. 25 — "Compute as a Service"

---

## Executive Summary

This document covers the operational infrastructure for running the AutoResearch system: CLI entry points, crash recovery, output artifacts, monitoring, performance characteristics, and troubleshooting. As of experiment 90, the system has executed 91 experiments across two backbones (LFM2: 50, MLP: 40+) with a champion Residual MLP achieving test Sharpe +6.21 and +1001% total return across 7 regime folds. The operational design prioritizes crash resilience (the development laptop crashes frequently), append-only logging (experiment history is never lost), and decoupled monitoring (the dashboard reads logs independently of the runner).

**Key operational components:**
- **Experiment runner** (`run_autoresearch.py`): executes one experiment, logs to JSONL, exits
- **Experiment log** (`experiment_log.jsonl`): append-only structured log of all experiments
- **Dashboard** (`dashboard.html`): live HTML dashboard served via Python http.server on port 8765
- **Crash-recovery checkpoint** (`memory/project_autoresearch_checkpoint.md`): self-contained state file enabling session resumption
- **Winner archive** (`best_config.json` + `best_model.pt`): current champion's configuration and model weights

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

---

## 7. MLOps Pipeline

### 7.1 Experiment Logging (JSONL)

All experiments are logged to `autoresearch_results/experiment_log.jsonl` in append-only JSON Lines format. Each line is a self-contained JSON object representing one experiment, enabling streaming reads without loading the entire file.

**Schema (per line):**

| Field | Type | Description |
|-------|------|-------------|
| `backbone` | string | Model backbone (e.g., "mlp", "lfm2-350m") |
| `description` | string | Human-readable experiment description with rationale |
| `config` | object | Full hyperparameter config (seq_len, lr, batch_size, epochs, etc.) |
| `composite` | float | Composite metric: `min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds` |
| `sharpe` | float | Test set aggregate Sharpe ratio |
| `val_sharpe` | float | Validation set aggregate Sharpe ratio |
| `psr` | float | Probabilistic Sharpe Ratio |
| `equity` | float | Final equity (starting from 1000) |
| `sortino` | float | Sortino ratio |
| `return_pct` | float | Total return percentage |
| `max_dd` | float | Maximum drawdown percentage |
| `win_rate` | float | Percentage of winning trades |
| `profit_factor` | float | Gross profit / gross loss |
| `ic` | float | Information coefficient (Spearman correlation) |
| `per_window` | array | Per-fold test breakdown (7 objects, one per regime) |
| `val_per_window` | array | Per-fold validation breakdown |
| `train_per_window` | array | Training set metrics |
| `aleatoric_mean` | float | Mean aleatoric uncertainty (heteroscedastic models) |
| `epistemic_mean` | float | Mean epistemic uncertainty |
| `confidence_mean` | float | Mean prediction confidence |
| `experiment_num` | int | Sequential experiment number |
| `status` | string | "KEEP" or "DISCARD" |
| `elapsed_sec` | float | Wall-clock time for the experiment |
| `timestamp` | string | ISO 8601 timestamp |

**Example JSONL entry (abbreviated):**

```json
{
  "backbone": "mlp",
  "description": "mlp: Exp32 VERIFY champion seed=0",
  "config": {"seq_len": 10, "lr": 0.0005, "batch_size": 32, "epochs": 50},
  "composite": 5.499,
  "sharpe": 6.2113,
  "val_sharpe": 5.599,
  "per_window": [
    {"fold": "fold_1", "regime": "Pre-crisis/GFC", "sharpe": 2.4608},
    {"fold": "fold_2", "regime": "Post-crash recovery", "sharpe": 1.1722},
    ...
  ],
  "experiment_num": 88,
  "status": "KEEP",
  "elapsed_sec": 36.4,
  "timestamp": "2026-04-14T17:30:21"
}
```

**Log integrity rules:**
- **Append-only:** entries are never modified or deleted
- **Self-contained:** each line includes the full config, not just deltas from the previous experiment
- **Timestamped:** enables correlation with system events and crash recovery
- **Archived:** `sweep_history/` keeps timestamped backup copies of the JSONL

### 7.2 Winner Archiving Protocol

When an experiment's composite score exceeds the current champion, the following archiving sequence executes:

```
1. Update best_config.json with the new champion's full config
2. Save model weights to best_model.pt (torch.save)
3. Log the experiment with status: "KEEP" in experiment_log.jsonl
4. Update experiment_summary.md with the new champion row
5. Update crash-recovery checkpoint with new champion details
```

**best_config.json structure:**

```json
{
  "backbone": "mlp",
  "seq_len": 10,
  "lr": 0.0005,
  "batch_size": 32,
  "epochs": 50,
  "weight_decay": 1e-5,
  "patience": 10,
  "grad_clip": 1.0,
  "huber_delta": 0.5,
  "head_dropout": 0.15,
  "seed": 0,
  "het_loss": false,
  "composite": 5.499,
  "test_sharpe": 6.2113,
  "val_sharpe": 5.599,
  "experiment_num": 88,
  "timestamp": "2026-04-14T17:30:21"
}
```

**Model checkpoint management:**
- `best_model.pt` is overwritten only when a new champion is crowned
- Contains `model.state_dict()` for the trained model (typically 50-200 KB for MLP, 1.4 GB for LFM2)
- Can be loaded for inference: `model.load_state_dict(torch.load("best_model.pt"))`
- Per-experiment model weights are NOT saved (disk space constraint) -- only the champion is preserved

### 7.3 Dashboard Deployment

The live dashboard is served as a static HTML page via Python's built-in HTTP server:

```bash
# Start the dashboard (once per session, background)
"C:/Users/evija/anaconda3/python.exe" -m http.server 8765 \
  --directory C:/Users/evija/autoresearch/autoresearch/autoresearch_results
```

**Dashboard URL:** http://localhost:8765/dashboard.html

**Dashboard features:**
- **Tabs:** Train / Val / Test views with per-fold-window metric breakdown
- **Experiment table:** All experiments with composite score, Sharpe, status (KEEP/DISCARD)
- **Per-window breakdown:** 7 regime folds with Sharpe, return, win rate, IC, uncertainty metrics
- **Auto-refresh:** Reads experiment_log.jsonl on load to display latest results
- **Decoupled:** The dashboard reads logs -- it has no dependency on the runner process

**Architecture principle:** The dashboard is a pure reader. It never writes to the experiment log or modifies any state. This decoupling means the dashboard can be open in a browser while experiments run without any interference.

### 7.4 Crash-Recovery Checkpointing as an Operational Concern

The development laptop experiences frequent crashes (power issues, thermal throttling, Windows updates). Crash recovery is therefore an operational priority, not a nice-to-have.

**Checkpoint file:** `memory/project_autoresearch_checkpoint.md`

**Checkpoint frequency:**
1. After every experiment completes (before analysis)
2. Every 5 minutes during reasoning/analysis
3. Before and after any code change
4. Before starting the next experiment

**Checkpoint contents (must be self-contained):**

| Section | Contents | Why Needed |
|---------|----------|-----------|
| Session recovery steps | Numbered instructions for a fresh session | New Claude session can resume without any other context |
| Champion config | Full hyperparameter config + composite score | Know what baseline to compare against |
| Per-fold test Sharpe table | 7-fold breakdown with regime labels | Diagnose which folds are weak |
| Last experiment result | Config delta, composite, per-fold deltas, KEEP/DISCARD | Understand what was just tried |
| Next experiment command | Copy-pasteable bash command | Resume immediately after crash |
| Rationale for next experiment | Diagnosis + literature citation + hypothesis | Continue the reasoning thread |
| Exhausted axes | Parameters already fully explored | Avoid re-trying dead ends |
| Full experiment history summary | Every experiment number, config, result | Track cumulative progress |

**Recovery procedure:**
1. New Claude session starts
2. Reads `CLAUDE.md` (project rules)
3. Reads `memory/project_autoresearch_checkpoint.md` (state)
4. Verifies state against `experiment_log.jsonl` tail and `best_config.json`
5. Starts dashboard in background
6. Runs the next experiment command from the checkpoint
7. Resumes the 7-step experiment loop

**Designed for zero-context-loss recovery:** A fresh session reading only the checkpoint and CLAUDE.md can pick up exactly where the previous session left off -- same experiment number, same champion, same next-experiment rationale.

### 7.5 Colab Notebook Generation for Winners

When a champion configuration is finalized for a backbone, a Colab notebook can be generated for independent verification and demonstration:

**Notebook structure:**
1. **Cell 1: Setup** -- pip install dependencies, clone repo, set seed
2. **Cell 2: Data** -- download EUR/USD data with caching, compute 104 features
3. **Cell 3: Split** -- create super-fold splits, verify zero overlap
4. **Cell 4: Train** -- train the champion config (copy exact hyperparameters)
5. **Cell 5: Evaluate** -- per-fold test evaluation, generate trading report
6. **Cell 6: Visualize** -- equity curves per regime, uncertainty plots

**Generation command:**
```bash
python -m autoresearch.generate_colab --config best_config.json --output champion_notebook.ipynb
```

This enables stakeholders to reproduce the champion's results in a browser without any local setup, and provides a portable artifact for sharing results.
