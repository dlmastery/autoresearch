# 08 - Configuration Management

**SWEBoK Knowledge Area:** KA6 — Software Configuration Management
**Google SWE Reference:** Ch. 16 — "Version Control and Branch Management"

---

## 1. Dependency Management

### 1.1 requirements.txt

```
torch>=2.5.0
transformers>=4.55
safetensors
accelerate
yfinance
pandas
numpy
scikit-learn
anthropic
pytest
```

### 1.2 Dependency Categories

| Category | Packages | Required For |
|----------|----------|-------------|
| Core ML | torch, transformers, safetensors, accelerate | Neural model training |
| Data | yfinance, pandas, numpy | Data acquisition + processing |
| Preprocessing | scikit-learn | StandardScaler |
| Gradient Boosting | xgboost, lightgbm, catboost | GBM backbones (optional) |
| AI Agent | anthropic | Optimizer loop (optional) |
| Testing | pytest | Test suite |

### 1.3 Optional Dependencies

GBM libraries (xgboost, lightgbm, catboost) are imported at runtime in `model/backbone.py`. If not installed, those backbones will fail but the rest of the system works.

## 2. Environment Configuration

### 2.1 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | For optimizer only | Claude API access |

### 2.2 .env File

```
ANTHROPIC_API_KEY=sk-ant-...
```

Loaded by `optimizer/agent_loop.py` via `dotenv` or environment.

### 2.3 .gitignore

```
.env
__pycache__
*.pyc
.pytest_cache
data/*.parquet
*.egg-info
```

## 3. Setup Scripts

### 3.1 setup.sh (Unix/Mac)

```bash
#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install xgboost lightgbm catboost  # Optional GBM support
```

### 3.2 setup.bat (Windows)

```batch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install xgboost lightgbm catboost
```

## 4. Data Versioning

### 4.1 Data Files (Not Version Controlled)

```
data/
  ├── EURUSD_X_2005-01-01_2026-04-01.parquet    # ~500KB
  ├── GBPUSD_X_2005-01-01_2026-04-01.parquet
  ├── ...                                         # 15 total parquet files
```

- Deterministic filenames: same date range = same cache key
- Regenerated from Yahoo Finance if deleted
- Not committed to git (in .gitignore)

### 4.2 Result Files (Should Be Version Controlled)

```
baseline_results.json           # Per-run results
ablation_results/*.json         # Per-backbone results
reports/*.md                    # Markdown reports
optimizer_state.json            # Optimizer experiment history
```

## 5. Configuration Constants

### 5.1 Centralized Constants

| Location | Constants | Values |
|----------|-----------|--------|
| data/download.py | `PAIRS`, `MACRO_TICKERS`, `DEFAULT_START/END` | 6 pairs, 9 tickers, 2005-2026 |
| data/features.py | `WARMUP_PERIOD` | 63 days |
| data/splits.py | `FOLDS`, `PURGE_DAYS`, `EMBARGO_DAYS`, `LABEL_HORIZON_BUFFER` | 7 folds, 90 days, 21 days, 10 days |
| model/backbone.py | `BACKBONE_REGISTRY`, `DEFAULT_BACKBONE`, `BACKBONE_SEQ_LEN`, `get_seq_len()` | 8 backbones, lfm2-350m, {lfm2-350m:60, others:10} |
| model/train.py | `SEQ_LEN`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `GRAD_CLIP`, `PATIENCE` | 60, 32, 3e-4, 20, 1.0, 5 |
| evaluation/metrics.py | `TRADING_DAYS_PER_YEAR` | 252 |
| evaluation/leakage_check.py | `MIN_GAP_DAYS` | 80 |

### 5.2 CLI Override Points

| Script | Arguments | Default |
|--------|-----------|---------|
| baseline.py | `--backbone`, `--epochs`, `--seq-len` | lfm2-350m, 20, per-backbone (60 LFM, 10 others) |
| run_ablation.py | `--backbones`, `--epochs` | all, 20 (seq_len auto per backbone) |
| run_optimizer.py | `--max-experiments`, `--baseline-only`, `--model` | 12, false, claude-sonnet-4 |

## 6. Version Control Strategy

### 6.1 Repository Structure

- **Not currently a git repo** — should be initialized
- **Branch strategy:** Recommended trunk-based development (main + short-lived feature branches)
- **Commit convention:** `feat:`, `fix:`, `refactor:`, `test:`, `docs:` prefixes

### 6.2 Recommended .gitignore Additions

```
# Data caches
data/*.parquet

# Checkpoints (transient)
baseline_checkpoint.json

# Optimizer backups (transient)
.optimizer_backups/

# Environment
.env
venv/

# Python
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/

# Model weights (large)
*.pt
*.bin
*.safetensors
```

## 7. Hardware Configuration

| Property | Value | Impact |
|----------|-------|--------|
| CPU | Intel (Iris Xe integrated) | float32 training, no CUDA |
| RAM | System dependent | Limits model size (1.2B backbone may OOM) |
| Storage | Local SSD | Fast parquet I/O |
| GPU | None (or CUDA if available) | Auto-detected by PyTorch |
