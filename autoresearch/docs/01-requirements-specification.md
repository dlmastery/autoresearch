# 01 - Requirements Specification

**SWEBoK Knowledge Area:** KA1 — Software Requirements
**Google SWE Reference:** Ch. 5 — "How to Lead a Team" (requirements clarity)

---

## 1. Functional Requirements

### FR-1: Data Acquisition
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-1.1 | Download daily OHLCV data for 6 FX pairs from Yahoo Finance | Must | Done |
| FR-1.2 | Download 9 macroeconomic signal tickers | Must | Done |
| FR-1.3 | Cache downloaded data as Parquet files to avoid redundant API calls | Must | Done |
| FR-1.4 | Support configurable date ranges (default: 2005-01-01 to 2026-04-01) | Should | Done |
| FR-1.5 | Gracefully handle individual ticker download failures | Should | Done |

### FR-2: Feature Engineering
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-2.1 | Compute per-pair technical features (returns, volatility, RSI, MACD, microstructure) | Must | Done |
| FR-2.2 | Compute cross-pair rolling correlations (21-day window) | Must | Done |
| FR-2.3 | Compute macro signal features (levels, returns, derived) | Must | Done |
| FR-2.4 | All features must be strictly backward-looking (no future data) | Must | Done |
| FR-2.5 | Drop warmup period rows (first 63 days for longest lookback) | Must | Done |
| FR-2.6 | Produce ~100+ features from 6 pairs + 9 macro tickers | Should | Done (104) |

### FR-3: Target Computation
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-3.1 | Compute 1-day forward returns: `close.pct_change(1).shift(-1)` | Must | Done |
| FR-3.2 | Compute 5-day forward returns: `close.pct_change(5).shift(-5)` | Must | Done |
| FR-3.3 | Align features and targets on common DatetimeIndex | Must | Done |

### FR-4: Walk-Forward Evaluation
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-4.1 | Implement 7-fold walk-forward cross-validation | Must | Done |
| FR-4.2 | Each fold must test a distinct market regime | Must | Done |
| FR-4.3 | Expanding training window (all data before purge gap) | Must | Done |
| FR-4.4 | 90-day purge between train/val and val/test boundaries | Must | Done |
| FR-4.5 | 21-day embargo between consecutive fold boundaries | Must | Done |
| FR-4.6 | Validate purge/embargo programmatically before every run | Must | Done |
| FR-4.7 | Ensure all test sets are disjoint (no overlap) | Must | Done |

### FR-5: Model Training
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-5.1 | Support 11 model backbones via registry pattern | Must | Done |
| FR-5.2 | Unified forward interface for all neural models | Must | Done |
| FR-5.3 | Separate GBM training path (sliding window features) | Must | Done |
| FR-5.4 | Early stopping with patience and best-state restoration | Must | Done |
| FR-5.5 | Gradient clipping to prevent exploding gradients | Must | Done |
| FR-5.6 | StandardScaler fit on training data only | Must | Done |

### FR-6: Evaluation Metrics
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-6.1 | Annualized Sharpe ratio (primary metric) | Must | Done |
| FR-6.2 | Probabilistic Sharpe Ratio (Lopez de Prado) | Must | Done |
| FR-6.3 | Deflated Sharpe Ratio (multiple testing correction) | Must | Done |
| FR-6.4 | Information Coefficient (Spearman + Pearson) | Must | Done |
| FR-6.5 | Full trading report (40+ metrics) | Must | Done |
| FR-6.6 | Average and weighted-average Sharpe across folds | Must | Done |

### FR-7: Autonomous Optimizer
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-7.1 | Claude API brainstorms 3 experiment ideas per iteration | Must | Done |
| FR-7.2 | Generate and apply code modifications | Must | Done |
| FR-7.3 | Syntax validation before execution | Must | Done |
| FR-7.4 | Keep improvements, revert regressions | Must | Done |
| FR-7.5 | Persist state across crashes (JSON checkpointing) | Must | Done |
| FR-7.6 | Pre-experiment file backups for crash recovery | Must | Done |

### FR-8: Reporting
| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-8.1 | Generate Markdown ablation reports with summary tables | Must | Done |
| FR-8.2 | Per-backbone detail sections with per-fold results | Must | Done |
| FR-8.3 | Conclusions section with statistical significance analysis | Must | Done |
| FR-8.4 | Per-backbone JSON results saved individually | Should | Done |

---

## 2. Non-Functional Requirements

### NFR-1: Data Integrity
| ID | Requirement | Threshold |
|----|-------------|-----------|
| NFR-1.1 | Zero data leakage across all folds | 0 violations |
| NFR-1.2 | Feature backward-lookingness verified by tests | All pass |
| NFR-1.3 | Scaler isolation (fit on train only) | Verified |

### NFR-2: Performance
| ID | Requirement | Threshold |
|----|-------------|-----------|
| NFR-2.1 | Single backbone baseline completes within 30 minutes (CPU) | Achieved |
| NFR-2.2 | Full 11-backbone ablation completes within 8 hours (CPU) | Pending |
| NFR-2.3 | Data download cached after first run | Achieved |

### NFR-3: Reliability
| ID | Requirement | Threshold |
|----|-------------|-----------|
| NFR-3.1 | Per-fold checkpointing for crash recovery | Implemented |
| NFR-3.2 | Optimizer state persists across crashes | Implemented |
| NFR-3.3 | File backups before code modifications | Implemented |
| NFR-3.4 | Checkpoint backbone validation (prevent stale data) | Implemented |

### NFR-4: Maintainability
| ID | Requirement | Threshold |
|----|-------------|-----------|
| NFR-4.1 | Modular design (data, model, evaluation, optimizer) | Achieved |
| NFR-4.2 | Registry pattern for backbone extensibility | Achieved |
| NFR-4.3 | Test suite with >80% module coverage | Partial |

### NFR-5: Reproducibility
| ID | Requirement | Threshold |
|----|-------------|-----------|
| NFR-5.1 | Deterministic data caching (same inputs = same cache) | Achieved |
| NFR-5.2 | Walk-forward splits produce identical folds across runs | Achieved |
| NFR-5.3 | Results saved as JSON for audit trail | Achieved |

---

## 3. Constraints

| Constraint | Description | Impact |
|------------|-------------|--------|
| CPU-only hardware | Intel Iris Xe, no discrete GPU | float32 mode, smaller batch sizes, longer training |
| Yahoo Finance data | Daily frequency only, no intraday | 1d/5d horizons (not 1h/4h) |
| No real-time feed | Historical data only | Backtesting only, no live trading |
| API key required | Anthropic key needed for optimizer | Optimizer module optional if key unavailable |

---

## 4. Requirements Traceability

Every functional requirement traces to either:
- **User feedback** (documented in `research-retrospective.md`)
- **Domain best practice** (Lopez de Prado 2018, walk-forward literature)
- **System constraint** (hardware, data source limitations)

All requirements are verified by at least one of:
- Automated tests in `tests/`
- Runtime validation (e.g., `validate_purge_embargo()`)
- Manual review of output artifacts
