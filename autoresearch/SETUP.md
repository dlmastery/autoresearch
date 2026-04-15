# AutoResearch - Setup & Recovery Guide

## Quick Start

```bash
# 1. Unzip the project
unzip autoresearch_project.zip
cd autoresearch

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full ablation study (all 8 backbones)
python -m autoresearch ablation

# 5. Or run a single backbone
python -m autoresearch baseline --backbone lfm2-350m
```

## Requirements

- **Python:** 3.11+
- **GPU:** NVIDIA GPU with CUDA support (16GB+ VRAM recommended for LFM2.5)
- **OS:** Linux, macOS, or Windows

## Package Dependencies

All dependencies are in `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `torch` | Neural network training (CUDA support required for GPU) |
| `transformers` | HuggingFace models (LFM2.5, PatchTST, PatchTSMixer) |
| `yfinance` | FX and macro data download from Yahoo Finance |
| `scikit-learn` | StandardScaler, preprocessing |
| `scipy` | Statistical tests (PSR, IC) |
| `numpy`, `pandas` | Data manipulation |
| `xgboost` | XGBoost gradient boosting |
| `lightgbm` | LightGBM gradient boosting |
| `catboost` | CatBoost gradient boosting |
| `optuna` | Hyperparameter optimization sweeps |

### Installing PyTorch with CUDA

If `pip install -r requirements.txt` installs CPU-only PyTorch:

```bash
# For CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Installing LightGBM with GPU

```bash
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

If GPU build fails, LightGBM will fall back to CPU automatically.

## Project Structure

```
autoresearch/
    __init__.py
    __main__.py              # CLI entry point (python -m autoresearch)
    baseline.py              # Walk-forward evaluation runner
    run_ablation.py          # Multi-backbone comparison
    run_sweep.py             # Optuna hyperparameter sweep
    run_optimizer.py         # Claude API autonomous optimizer
    data/
        download.py          # FX + macro data from Yahoo Finance
        features.py          # 104 backward-looking features
        splits.py            # 7 walk-forward folds with purge/embargo
    model/
        backbone.py          # 8 model architectures
        train.py             # Training loop (Huber loss, early stopping)
    evaluation/
        metrics.py           # Sharpe, PSR, DSR, IC, trading report
        leakage_check.py     # Data leakage validation
    tests/                   # pytest test suite
    docs/                    # SWEBoK-aligned documentation (14 files)
    ablation_results/        # Per-backbone JSON results
    reports/                 # Generated Markdown reports
```

## Key Configuration

### Per-Backbone Sequence Lengths

| Backbone | seq_len | Rationale |
|----------|---------|-----------|
| LFM2.5-350M | 60 (3 months) | Foundation model benefits from long context |
| All others | 10 (2 weeks) | Industry standard for 1d/5d FX prediction |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Learning rate | 3e-4 (AdamW) |
| Epochs | 20 (with early stopping) |
| Patience | 5 epochs |
| Gradient clip | 1.0 |
| Loss | Huber (delta=1.0) |

### Data Integrity

- **90-day purge gap** between train/val and val/test
- **21-day embargo** between fold boundaries
- **Cross-fold hole-punching:** All val/test windows excluded from all folds' training data
- **10-day label-horizon buffer:** Prevents fwd_ret_5d targets from peeking into excluded windows

## Running Commands

```bash
# Full ablation (all 8 backbones, per-backbone seq_len)
python -m autoresearch ablation

# Single backbone with custom epochs
python -m autoresearch baseline --backbone lstm --epochs 10

# Hyperparameter sweep
python -m autoresearch sweep --backbone patchtst --n-trials 50

# Run tests
pytest tests/ -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or use `--epochs 5` |
| `ModuleNotFoundError: transformers` | `pip install transformers` |
| `LFM2 model not found` | First run downloads ~700MB model from HuggingFace |
| `yfinance download fails` | Check internet connection; data is cached after first download |
| `lightgbm GPU error` | Falls back to CPU automatically; or install GPU build |
| `catboost shapes error` | Known issue being fixed; may need sklearn version alignment |
