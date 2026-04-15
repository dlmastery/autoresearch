# 07 - Testing Strategy

**SWEBoK Knowledge Area:** KA5 — Software Testing
**Google SWE Reference:** Ch. 11 — "Testing Overview"; Ch. 12 — "Unit Testing"; Ch. 14 — "Larger Testing"

---

## 1. Test Philosophy

Following Google's testing pyramid and SWEBoK verification/validation:

```
            /  E2E Tests  \          <- test_e2e.py (1 file)
           / Integration   \         <- test_baseline.py (1 file)
          /   Module Tests   \       <- test_features, test_splits, test_train, ... (7 files)
         /                    \
        /    Static Analysis    \    <- py_compile in optimizer
       ──────────────────────────
```

## 2. Test Suite Overview

| File | Lines | Category | Tests | Purpose |
|------|-------|----------|-------|---------|
| test_download.py | 93 | Module | 4 | Data acquisition + caching |
| test_features.py | 264 | Module | 6+ | Feature backward-lookingness, shape, NaN handling |
| test_splits.py | 160 | Module | 7+ | Fold validation, purge, embargo, disjointness, cross-fold contamination, label-horizon buffer |
| test_train.py | 143 | Module | 4+ | Dataset creation, target alignment, training loop |
| test_model.py | 62 | Module | 3+ | Backbone instantiation, forward pass shape |
| test_metrics.py | 72 | Module | 4+ | Sharpe, IC, aggregation, edge cases |
| test_leakage.py | 62 | Module | 3+ | Split gaps, scaler isolation |
| test_baseline.py | 5 | Integration | 1 | Minimal baseline placeholder |
| test_e2e.py | 64 | E2E | 1 | Full pipeline mini-run |

**Total:** ~885 lines of test code across 9 files

## 3. Critical Test Categories

### 3.1 No-Leakage Tests (Highest Priority)

These tests verify the most critical property of the system — no future information leaks into predictions.

**test_features.py — `test_all_features_backward_only`**
- Verifies every feature column uses only backward-looking computations
- Checks that no feature contains NaN in the post-warmup region
- Validates feature count matches expectations

**test_splits.py — Purge/Embargo/Contamination Tests**
- `test_purge_gaps`: Verifies 90-day minimum gap between train→val and val→test
- `test_embargo_gaps`: Verifies 21-day minimum gap between consecutive fold boundaries
- `test_disjoint_tests`: No overlap between any two test periods
- `test_no_cross_fold_contamination`: Verifies no fold's training data contains any date from any fold's val/test window (hole-punching prevents expanding-window leakage)
- `test_label_horizon_buffer_enforced`: Verifies no training sample exists within 10 calendar days before any held-out window (prevents `fwd_ret_5d` targets from peeking into excluded periods)
- Uses `validate_purge_embargo()` — same function called at runtime

**test_leakage.py — Scaler Isolation**
- `test_scaler_isolation`: Verifies scaler fit on train only, applied to val/test
- `test_split_gaps`: Verifies minimum gap days between splits
- Checks that val/test statistics differ from train (not accidentally refit)

### 3.2 Data Pipeline Tests

**test_download.py**
- `test_pairs_defined`: 6 FX pairs registered
- `test_macro_tickers_defined`: 9+ macro tickers registered
- `test_download_single_pair`: Downloads EURUSD=X, checks columns and monotonic index
- `test_download_all_pairs`: Downloads all 6 pairs, checks dict size

**test_features.py**
- Feature shape verification (expected column count)
- NaN handling after warmup period
- Cross-pair correlation computation
- Macro feature derivation

### 3.3 Model Tests

**test_model.py**
- `test_backbone_instantiation`: Creates each backbone, checks it's nn.Module or GBMWrapper
- `test_forward_pass_shape`: Verifies output shape `{"ret_1d": [B, 6], "ret_5d": [B, 6]}`
- `test_gbm_fit_predict`: Verifies GBMWrapper fit/predict interface

**test_train.py**
- `test_dataset_creation`: FXDataset produces correct shapes
- `test_target_alignment`: Target at end of window (not one step ahead)
- `test_train_one_fold`: Full training loop completes without error
- `test_scaler_within_training`: Scaler applied correctly during training

### 3.4 Metrics Tests

**test_metrics.py**
- `test_sharpe_ratio`: Known input → known output
- `test_sharpe_zero_std`: Edge case: constant returns → 0.0
- `test_sharpe_empty`: Edge case: empty array → 0.0
- `test_information_coefficient`: IC computation correctness
- `test_trading_report_keys`: All 25+ expected keys present
- `test_average_sharpe_across_folds`: Aggregation correctness

### 3.5 End-to-End Tests

**test_e2e.py**
- Mini pipeline run: download (small date range) → features → split → train (1 epoch) → evaluate
- Verifies entire system integrates correctly
- Uses small data to keep runtime manageable

## 4. Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_features.py -v

# Specific test
pytest tests/test_splits.py::test_purge_gaps -v

# With coverage (if pytest-cov installed)
pytest tests/ --cov=. --cov-report=html
```

## 5. Test Data Strategy

| Test Category | Data Source | Rationale |
|---------------|-----------|-----------|
| Download tests | Live Yahoo Finance (small range) | Verifies real API integration |
| Feature tests | Synthetic DataFrame | Reproducible, fast, controlled |
| Split tests | FOLDS constant (real dates) | Tests actual production fold definitions |
| Model tests | Random tensors | Shape/interface verification only |
| Metrics tests | Known arrays | Verifiable expected outputs |
| E2E tests | Live Yahoo Finance (small range) | Real integration |

## 6. Test Execution in CI/CD

Currently manual (`pytest`). Recommended CI integration:

```yaml
# Proposed GitHub Actions workflow
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
```

## 7. Quality Gates

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| All tests pass | 0 failures | Pre-commit / CI |
| Purge/embargo validation | 0 violations | Runtime check in baseline.py |
| Syntax validation | Valid Python | Optimizer checks before applying code |
| Feature backward-only | All features verified | test_features.py |
| Scaler isolation | Fit on train only | test_leakage.py |
