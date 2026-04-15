# 10 - Quality Assurance & Best Practices

**SWEBoK Knowledge Area:** KA11 — Software Quality; KA10 — Software Quality Management
**Google SWE Reference:** Ch. 9 — "Code Review"; Ch. 20 — "Static Analysis"

---

## Part A: Best Practices Already Followed

### A1. Data Integrity & Leakage Prevention

| Practice | Implementation | Reference |
|----------|---------------|-----------|
| **Strictly backward-looking features** | All rolling windows use historical data only; `shift(-N)` only in targets | data/features.py |
| **Purge gaps (90 days)** | 3-month temporal buffer between train/val and val/test boundaries | data/splits.py |
| **Embargo gaps (21 days)** | Minimum gap between consecutive fold test and next fold val boundaries | data/splits.py |
| **Disjoint test sets** | Pairwise overlap check validates no test period reuse | data/splits.py |
| **Scaler isolation** | StandardScaler fit on training data only; val/test transformed, never fit | baseline.py, model/train.py |
| **Runtime validation** | `validate_purge_embargo()` runs before every baseline evaluation | baseline.py:141 |
| **Automated leakage tests** | test_leakage.py verifies scaler isolation and split gaps | tests/test_leakage.py |
| **Feature backward-only tests** | test_features.py verifies no future data in features | tests/test_features.py |

**Assessment:** Excellent. The leakage prevention framework exceeds industry standard practices (typically 30-60 day purge). The combination of runtime validation + automated tests + conservative 90-day gaps provides defense-in-depth.

### A2. Evaluation Rigor

| Practice | Implementation | Reference |
|----------|---------------|-----------|
| **Lopez de Prado metrics** | PSR, DSR, IC implemented with correct formulas | evaluation/metrics.py |
| **PSR standard error** | Accounts for skewness and kurtosis (Lo 2002 corrections) | evaluation/metrics.py:93 |
| **DSR multiple testing** | Gumbel extreme value adjustment for number of trials | evaluation/metrics.py:113 |
| **Risk-adjusted primary metric** | Sharpe ratio (not raw returns or accuracy) | baseline.py |
| **Multi-fold averaging** | Average across 7 folds prevents single-regime overfitting | baseline.py:296 |
| **Weighted Sharpe** | Training-size weighting reflects estimation confidence | baseline.py:297 |
| **Regime-aware splits** | 7 distinct market regimes: crisis, recovery, plateau, trend, low-vol, downturn, mixed | data/splits.py |
| **40+ trading metrics** | Comprehensive report including tail stats, trade analytics | evaluation/metrics.py |

**Assessment:** Excellent. Following Lopez de Prado's framework is gold-standard for quantitative finance evaluation. The regime-aware splits are a significant improvement over generic train/test.

### A3. Training Robustness

| Practice | Implementation | Reference |
|----------|---------------|-----------|
| **Huber loss** | Robust to fat-tailed FX returns (δ=1.0) | model/train.py |
| **Gradient clipping** | Max norm 1.0 prevents exploding gradients | model/train.py |
| **Early stopping** | Patience 3 + best state restoration | model/train.py |
| **Cosine annealing LR** | Smooth decay prevents sharp instabilities | model/train.py |
| **AdamW** | Decoupled weight decay for proper regularization | model/train.py |

**Assessment:** Good. Standard production-grade training practices correctly applied.

### A4. Software Engineering

| Practice | Implementation | Reference |
|----------|---------------|-----------|
| **Registry pattern** | Backbone selection via dictionary + factory function | model/backbone.py |
| **Shared prediction heads** | Fair backbone comparison (identical head architecture) | model/backbone.py |
| **Unified interface** | All neural models share `forward(x) -> dict` contract | model/backbone.py |
| **Crash recovery** | Per-fold checkpointing with backbone validation | baseline.py |
| **File backups** | Pre-experiment backup/restore in optimizer | optimizer/agent_loop.py |
| **Modular architecture** | Clean separation: data, model, evaluation, optimizer | Package structure |
| **Error isolation** | GBM failures don't crash neural models; per-backbone error handling | run_ablation.py |
| **Syntax validation** | py_compile before applying generated code | optimizer/agent_loop.py |
| **CLI arguments** | argparse for all entry points with sensible defaults | All runner scripts |

**Assessment:** Good. Clean modular design with appropriate separation of concerns.

### A5. Testing

| Practice | Implementation | Reference |
|----------|---------------|-----------|
| **Module-level tests** | Tests for data, features, splits, model, metrics, leakage | tests/ (9 files) |
| **Property-based testing** | Feature backward-lookingness as testable property | test_features.py |
| **Edge case handling** | Empty arrays, zero std, single-element inputs in metrics | test_metrics.py |
| **Integration test** | E2E mini pipeline test | test_e2e.py |

**Assessment:** Adequate. Coverage exists for critical paths but could be expanded.

---

## Part B: Improvements Recommended

### B1. High Priority (Correctness & Risk)

#### B1.1 Reproducibility: Random Seed Management
**Current:** No explicit random seed setting.
**Risk:** Different runs may produce different results (model initialization, dropout, data shuffling).
**Recommendation:**
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```
Call at the start of each baseline/ablation run.

#### B1.2 Transaction Cost Modeling
**Current:** Strategy returns assume zero transaction costs.
**Risk:** Positive Sharpe may disappear after costs. FX spreads are ~1-2 pips for majors.
**Recommendation:**
```python
# After computing strategy_returns
cost_per_trade = 0.0001  # 1 pip = 0.0001 for EUR/USD
position_changes = np.abs(np.diff(np.sign(predictions), prepend=0))
net_returns = strategy_returns - cost_per_trade * (position_changes > 0)
```

#### B1.3 Out-of-Memory Protection for Large Models
**Current:** No memory checks before loading 1.2B parameter model.
**Risk:** OOM crash with no graceful fallback.
**Recommendation:** Check available memory before model creation; warn and skip if insufficient.

#### B1.4 Git Version Control
**Current:** Not a git repository.
**Risk:** No change tracking, no rollback capability outside optimizer backups.
**Recommendation:** Initialize git, commit current state, use branches for experiments.

### B2. Medium Priority (Quality & Robustness)

#### B2.1 Test Coverage Expansion
**Current gaps:**
- No tests for `optimizer/agent_loop.py` (mocking Claude API)
- No tests for `run_ablation.py` report generation
- No tests for `run_overnight.py`
- No negative tests (expected failures)
- No parameterized tests across backbones

**Recommendation:** Add pytest fixtures for mock API, parameterized backbone tests, and edge case coverage.

#### B2.2 Logging Framework
**Current:** Mix of `print()` and `logging.getLogger()`.
**Recommendation:** Standardize on Python logging module with configurable levels:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler()]
)
```

#### B2.3 Configuration File
**Current:** Hyperparameters spread across multiple files as module-level constants.
**Recommendation:** Centralized config (YAML or dataclass):
```python
@dataclass
class Config:
    seq_len: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 5
    grad_clip: float = 1.0
    patience: int = 3
    purge_days: int = 90
    embargo_days: int = 21
```

#### B2.4 Data Validation Layer
**Current:** NaN drop and warmup removal, but no explicit data quality checks.
**Recommendation:** Add validation for:
- Stale data detection (>5 consecutive identical closes)
- Outlier detection (returns > ±10% daily)
- Missing date detection (unexpected gaps)
- Volume validation (zero-volume days)

#### B2.5 Model Checkpointing Beyond Folds
**Current:** Only per-fold checkpointing during walk-forward.
**Recommendation:** Save trained model weights per fold for later analysis:
```python
torch.save(model.state_dict(), f"checkpoints/{backbone}_fold{i}.pt")
```

### B3. Lower Priority (Enhancements)

#### B3.1 Parallel Backbone Evaluation
**Current:** Sequential backbone evaluation in ablation.
**Recommendation:** Use `multiprocessing` or `concurrent.futures` to evaluate independent backbones in parallel (especially GBM models which are fast).

#### B3.2 Experiment Tracking Integration
**Current:** Custom JSON state files.
**Recommendation:** Integrate with MLflow or Weights & Biases:
- Automatic metric logging
- Hyperparameter tracking
- Artifact storage (models, reports)
- Comparison dashboards

#### B3.3 Type Hints & Documentation
**Current:** Type hints present in most places but inconsistent.
**Recommendation:** Add comprehensive type hints + docstrings to all public functions. Use `mypy --strict` for validation.

#### B3.4 Pre-commit Hooks
**Current:** No pre-commit configuration.
**Recommendation:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks: [{ id: ruff }, { id: ruff-format }]
  - repo: local
    hooks:
      - id: pytest
        entry: pytest tests/ -x --tb=short
```

#### B3.5 Dockerfile
**Current:** No containerization.
**Recommendation:** Dockerfile for reproducible environment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_ablation.py"]
```

#### B3.6 Performance Profiling
**Current:** Only wall-clock timing per backbone.
**Recommendation:** Add memory profiling (peak GPU/RAM) and per-phase timing (data, train, eval).

---

## Part C: Technical Debt Register

| ID | Item | Severity | Effort | Description |
|----|------|----------|--------|-------------|
| TD-1 | No git repo | Medium | Low | Initialize git, commit current state |
| TD-2 | Mixed print/logging | Low | Low | Standardize on logging module |
| TD-3 | No random seeds | Medium | Low | Add seed management for reproducibility |
| TD-4 | Constants scattered | Low | Medium | Centralize in config dataclass |
| TD-5 | GBM optional imports | Low | Low | Add graceful import error messages |
| TD-6 | test_baseline.py placeholder | Low | Low | Expand or merge into test_e2e.py |
| TD-7 | No CI/CD | Medium | Medium | Add GitHub Actions workflow |
| TD-8 | No type checking | Low | Medium | Add mypy configuration |
| TD-9 | Checkpoint path coupling | Low | Low | Make checkpoint path configurable |
| TD-10 | No model weight saving | Medium | Low | Save best model per fold |

---

## Part D: Compliance Matrix (SWEBoK Knowledge Areas)

| KA | Area | Coverage | Notes |
|----|------|----------|-------|
| KA1 | Requirements | Good | Documented in design docs + retrospective |
| KA2 | Design | Good | Modular architecture, clean interfaces |
| KA3 | Construction | Good | Production training loop, crash recovery |
| KA4 | Testing | Adequate | Tests exist but coverage could expand |
| KA5 | Maintenance | Partial | No CI/CD, no monitoring |
| KA6 | Config Mgmt | Partial | No git, scattered constants |
| KA7 | Eng Mgmt | Partial | Optimizer tracks experiments; no formal project tracking |
| KA8 | Process | Good | Design → implement → test → evaluate cycle |
| KA9 | Models/Methods | Good | Walk-forward, regime-aware, Lopez de Prado |
| KA10 | Quality | Good | Leakage prevention, statistical rigor |
| KA11 | Economics | N/A | Research prototype |
