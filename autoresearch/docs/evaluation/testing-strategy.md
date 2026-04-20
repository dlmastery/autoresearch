# 07 - Testing Strategy

**SWEBoK Knowledge Area:** KA5 — Software Testing
**Google SWE Reference:** Ch. 11 — "Testing Overview"; Ch. 12 — "Unit Testing"; Ch. 14 — "Larger Testing"

---

## Executive Summary

Testing in a financial ML system is fundamentally different from testing a web application. The most critical bugs are not crashes or UI glitches -- they are **silent data leakage** and **invalid evaluation**, which produce inflated performance metrics that evaporate in production. This project employs a defense-in-depth testing strategy where the highest-priority tests verify data integrity invariants (zero overlap, purge/embargo enforcement, backward-only features), followed by model reproducibility tests (fixed-seed determinism), contiguous windowing tests (no sliding windows across date gaps), and feature computation tests (correct rolling statistics). Every experiment session begins with a mandatory validation checklist that programmatically verifies all invariants before any training runs.

**Key statistics:** 9 test files, ~885 lines of test code, covering data integrity, feature correctness, model interface, evaluation metrics, and end-to-end pipeline integration. The leakage prevention tests alone account for 3 dedicated test files (test_splits.py, test_leakage.py, test_features.py).

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

---

## 8. Concrete Test Examples for This Project

This section provides project-specific test examples that go beyond the general descriptions above, showing exactly what is tested and why it matters for EUR/USD FX prediction with the super-fold evaluation framework.

### 8.1 Data Integrity Tests -- Zero Overlap Verification

The most critical invariant in walk-forward financial ML is that no training sample exists in any validation or test set. The super-fold design (fold 7 training data with all val/test windows hole-punched) makes this especially important because the training set is constructed by subtracting multiple non-contiguous date ranges.

```python
# test_splits.py — zero overlap verification
def test_superfold_zero_overlap():
    """Verify train/val/test sets are strictly disjoint in the super-fold."""
    train_df, val_df, test_df = split_superfold(df)
    
    train_dates = set(train_df.index)
    val_dates = set(val_df.index)
    test_dates = set(test_df.index)
    
    # Zero overlap between all pairs
    assert len(train_dates & val_dates) == 0, \
        f"Train-val overlap: {len(train_dates & val_dates)} dates"
    assert len(train_dates & test_dates) == 0, \
        f"Train-test overlap: {len(train_dates & test_dates)} dates"
    assert len(val_dates & test_dates) == 0, \
        f"Val-test overlap: {len(val_dates & test_dates)} dates"
    
    # Verify expected row counts
    assert len(train_df) == 3113, f"Train rows: {len(train_df)}, expected 3113"
    assert len(val_df) == 915, f"Val rows: {len(val_df)}, expected 915"
    assert len(test_df) == 1170, f"Test rows: {len(test_df)}, expected 1170"

def test_label_horizon_buffer():
    """Verify no training sample exists within 10 days before any held-out window.
    
    WHY: The target fwd_ret_5d looks 5 trading days forward. A training sample
    at date T has target value at T+5. If T+5 falls inside a held-out window,
    we have information leakage. The 10-calendar-day buffer (> 5 trading days)
    prevents this.
    """
    train_df, val_df, test_df = split_superfold(df)
    held_out_starts = [window.start for window in all_held_out_windows()]
    
    for start_date in held_out_starts:
        buffer_start = start_date - pd.Timedelta(days=10)
        buffer_dates = train_df.loc[buffer_start:start_date]
        assert len(buffer_dates) == 0, \
            f"Training data found in buffer zone before {start_date}"

def test_no_cross_fold_contamination():
    """Verify no fold's training data contains any date from any fold's val/test.
    
    This tests the hole-punching mechanism: when fold 7 uses all historical
    data for training, it must exclude ALL 7 folds' val and test windows,
    not just fold 7's own val/test.
    """
    train_df, _, _ = split_superfold(df)
    all_excluded = _all_held_out_ranges()  # Returns all val/test date ranges
    
    for start, end in all_excluded:
        overlap = train_df.loc[start:end]
        assert len(overlap) == 0, \
            f"Cross-fold contamination: {len(overlap)} rows in [{start}, {end}]"
```

### 8.2 Model Reproducibility Tests -- Fixed Seed Determinism

Seed variance is a critical finding from this project: the same architecture with seeds 0, 42, and 99 produces test Sharpe ratios of +6.21, +4.69, and +4.76 respectively -- a 32% spread. Reproducibility tests verify that a given seed always produces the identical result.

```python
# test_reproducibility.py — fixed-seed determinism
def test_fixed_seed_produces_identical_results():
    """Verify that seed=0 produces bit-identical results across two runs.
    
    This is essential for experiment validity: if we change one hyperparameter
    and compare results, we must be confident the difference is due to the
    change, not random initialization.
    """
    config = {
        "seq_len": 10, "lr": 5e-4, "batch_size": 32, "epochs": 5,
        "weight_decay": 1e-5, "patience": 3, "seed": 0
    }
    
    # Run 1
    set_seed(0)
    model1 = create_model("mlp", n_features=104)
    result1 = train_and_evaluate(model1, train_df, val_df, config)
    
    # Run 2 (fresh model, same seed)
    set_seed(0)
    model2 = create_model("mlp", n_features=104)
    result2 = train_and_evaluate(model2, train_df, val_df, config)
    
    assert result1["sharpe"] == result2["sharpe"], \
        f"Non-deterministic: {result1['sharpe']} vs {result2['sharpe']}"
    assert result1["val_loss"] == result2["val_loss"], \
        f"Val loss mismatch: {result1['val_loss']} vs {result2['val_loss']}"

def test_different_seeds_produce_different_results():
    """Verify that different seeds actually produce meaningfully different
    initializations, not identical outputs (which would indicate the seed
    is not being used).
    """
    results = {}
    for seed in [0, 42, 99]:
        set_seed(seed)
        model = create_model("mlp", n_features=104)
        results[seed] = train_and_evaluate(model, train_df, val_df, config)
    
    sharpes = [r["sharpe"] for r in results.values()]
    assert max(sharpes) - min(sharpes) > 0.01, \
        "All seeds produced identical results — seed is likely not wired through"
```

### 8.3 Contiguous Windowing Tests

Sliding windows must never cross date gaps. When the super-fold test set concatenates 7 non-contiguous fold windows, naive windowing creates garbage inputs where the first half of a window is from 2012 and the second half from 2015. This bug affected ~41% of windows with seq_len=60 before it was caught.

```python
# test_train.py — contiguous windowing
def test_contiguous_datasets_no_gap_crossing():
    """Verify create_contiguous_datasets() splits data at date gaps
    and never creates a window that spans a gap.
    
    Background: The super-fold training set has holes where val/test windows
    were removed. Concatenating this into a single DataFrame and creating
    sliding windows would produce ~41% garbage windows (seq_len=60).
    create_contiguous_datasets() detects gaps and creates separate
    FXDataset instances for each contiguous segment.
    """
    datasets = create_contiguous_datasets(train_df, seq_len=10)
    
    # Should produce ~7 segments (one gap per hole-punch)
    assert len(datasets) >= 5, \
        f"Expected multiple segments, got {len(datasets)}"
    
    for i, ds in enumerate(datasets):
        # Verify each window's dates are contiguous
        for j in range(len(ds)):
            window_dates = ds.get_dates(j)
            # Max gap between consecutive dates should be <= 4 days
            # (weekends + holidays, but never weeks/months)
            date_diffs = pd.Series(window_dates).diff().dropna()
            max_gap = date_diffs.max().days
            assert max_gap <= 5, \
                f"Segment {i}, window {j}: gap of {max_gap} days detected"

def test_per_window_evaluation_no_concatenation():
    """Verify that test evaluation processes each fold's test window
    separately, then aggregates predictions — never concatenating
    non-contiguous windows into a single dataset.
    """
    # Each fold's test window is evaluated independently
    for fold_name, window in test_windows.items():
        assert window.is_contiguous(), \
            f"Test window {fold_name} is not contiguous"
        assert len(window) >= config["seq_len"] + 1, \
            f"Test window {fold_name} too short: {len(window)} rows"
```

### 8.4 Feature Computation Tests

All 104 features must be strictly backward-looking -- no feature may use future data in its computation. This is verified by checking that modifying a single future data point does not change any feature value at or before the current time.

```python
# test_features.py — backward-only verification
def test_all_features_backward_only():
    """Verify every feature uses only past data by perturbing future values.
    
    Method: For a given row T, modify all data at T+1..T+N to random values.
    Recompute features. If any feature at row T changes, it is using future data.
    """
    df_original = make_test_dataframe(n_rows=500)
    features_original = compute_features(df_original)
    
    T = 300  # Test point (after warmup period)
    df_modified = df_original.copy()
    df_modified.iloc[T+1:] = np.random.randn(len(df_modified) - T - 1, df_modified.shape[1])
    features_modified = compute_features(df_modified)
    
    for col in features_original.columns:
        original_val = features_original.iloc[T][col]
        modified_val = features_modified.iloc[T][col]
        assert original_val == modified_val, \
            f"Feature '{col}' at row {T} changed when future data was modified: " \
            f"{original_val} -> {modified_val}"

def test_feature_count():
    """Verify the exact number of features matches expectations."""
    df = make_test_dataframe(n_rows=500)
    features = compute_features(df)
    assert features.shape[1] == 104, \
        f"Expected 104 features, got {features.shape[1]}"

def test_no_nans_after_warmup():
    """After the warmup period (max rolling window length), no features
    should contain NaN values. NaNs in features propagate to model outputs
    and silently corrupt predictions.
    """
    df = make_test_dataframe(n_rows=500)
    features = compute_features(df)
    warmup = 60  # Longest rolling window
    post_warmup = features.iloc[warmup:]
    nan_counts = post_warmup.isna().sum()
    problematic = nan_counts[nan_counts > 0]
    assert len(problematic) == 0, \
        f"NaN features after warmup: {dict(problematic)}"
```

---

## 9. Pre-Experiment Validation Checklist

This checklist runs programmatically before every experiment session. It verifies all critical invariants and prevents wasted compute on invalid configurations. **No experiment may proceed until all 6 checks pass.**

### Checklist Implementation

```python
def run_pre_experiment_validation(df, config):
    """Mandatory validation before every experiment. Returns True only if
    all checks pass. Called at the start of run_autoresearch.py.
    """
    checks = {}
    
    # 1. Purge/embargo validation — 0 violations
    violations = validate_purge_embargo()
    checks["purge_embargo"] = len(violations) == 0
    
    # 2. Super-fold row counts
    train, val, test = split_superfold(df)
    checks["superfold_counts"] = (
        len(train) == 3113 and len(val) == 915 and len(test) == 1170
    )
    
    # 3. Zero overlap
    train_dates = set(train.index)
    val_dates = set(val.index)
    test_dates = set(test.index)
    checks["zero_overlap"] = (
        len(train_dates & val_dates) == 0 and
        len(train_dates & test_dates) == 0 and
        len(val_dates & test_dates) == 0
    )
    
    # 4. Contiguous datasets produce expected segments
    datasets = create_contiguous_datasets(train, seq_len=config["seq_len"])
    checks["contiguous_segments"] = len(datasets) >= 5
    
    # 5. Each test window has enough rows
    for fold_name, window in test_windows.items():
        if len(window) < config["seq_len"] + 1:
            checks["test_window_sizes"] = False
            break
    else:
        checks["test_window_sizes"] = True
    
    # 6. Data loaded from cache (not re-downloaded)
    checks["data_cached"] = os.path.exists(".data_cache/")
    
    # Report
    all_pass = all(checks.values())
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    return all_pass
```

### Checklist Summary Table

| # | Check | Expected Result | What It Prevents |
|---|-------|----------------|-----------------|
| 1 | `validate_purge_embargo()` | 0 violations | Training data leaking into val/test through insufficient temporal gaps |
| 2 | `split_superfold()` row counts | train=3113, val=915, test=1170 | Incorrect hole-punching that drops or includes wrong dates |
| 3 | Train-val-test overlap | All 3 pairwise intersections = 0 | Any form of data contamination between splits |
| 4 | `create_contiguous_datasets()` segment count | >= 5 segments (typically 7) | Sliding windows crossing date gaps in training data |
| 5 | Test window sizes | All windows >= seq_len + 1 | Windows too short for the model's lookback period |
| 6 | Data from `.data_cache/` | Cache directory exists | Unnecessary re-downloads causing flaky runs and wasted time |

### When the Checklist Fails

If any check fails, the experiment **must not proceed**. Common failure causes and fixes:

| Failed Check | Likely Cause | Fix |
|-------------|-------------|-----|
| purge_embargo | Modified fold definitions | Revert splits.py to last known good state |
| superfold_counts | Changed fold dates or buffer sizes | Verify FOLDS constant and LABEL_HORIZON_BUFFER |
| zero_overlap | Bug in hole-punching logic | Debug split_data() and _all_held_out_ranges() |
| contiguous_segments | Incorrect gap detection threshold | Check create_contiguous_datasets() gap detection |
| test_window_sizes | seq_len too large for smallest test window | Reduce seq_len or verify fold definitions |
| data_cached | First run or cache cleared | Run download_all_pairs() with default cache_dir |
