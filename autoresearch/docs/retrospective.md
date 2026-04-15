# AutoResearch Development Retrospective

**Date:** 2026-04-11
**Sessions:** 4 (042290fe, 7294ef78, 50474cfc, current)

---

## What Happened (Timeline)

### Session 1 (042290fe) — Ablation setup + data integrity fix
- Removed informer, mamba2, lfm2-1.2b backbones (11 → 8)
- Discovered cross-fold contamination: expanding training windows included earlier folds' val/test data
- Added hole-punching in `split_data()` + `_all_held_out_ranges()`
- Added `LABEL_HORIZON_BUFFER = 10` calendar days to prevent `fwd_ret_5d` label leakage
- Changed per-backbone seq_len: LFM2.5 keeps 60, others get 10
- Updated hyperparameters to industry defaults (lr=3e-4, bs=32, epochs=20, patience=5)
- Deleted all stale results, created zip for transport

### Session 2 (7294ef78) — First clean ablation run
- Ran ablation on all 8 backbones with clean data
- 5 neural + xgboost completed; lightgbm and catboost had errors
- Fixed `from model.backbone` → `from .model.backbone` (relative import bug in GBM path)
- LFM2 Sharpe dropped from 1.19 (contaminated) to 0.62 (clean) — expected but concerning

### Session 3 (50474cfc) — Sweep attempts
- Started a grid sweep (wrong approach — Karpathy-style is diagnostic, not grid)
- Researched actual Karpathy autoresearch method
- Created `split_superfold()`: fold 7 train + ALL folds' val/test combined
- Verified super-fold data integrity: 0 contamination, 3113 train / 915 val / 1170 test rows
- Started rewriting autoresearch loop but accumulated multiple issues

### Session 4 (current) — Retrospective
- User identified accumulated problems: data re-downloaded every run, evaluation bugs, dead config params, no dashboard, no separation of concerns

---

## Mistakes Made

### 1. No data caching (all sessions)
**What:** `download_all_pairs()` and `download_macro_signals()` default `cache_dir=None`, downloading ~30 tickers from Yahoo Finance on every single run.
**Impact:** Minutes wasted per run, unnecessary network dependency, flaky runs when Yahoo is slow.
**Fix:** Changed defaults to `DEFAULT_CACHE_DIR = .data_cache/`. Applied in session 4.
**Root cause:** Didn't read the download code carefully on day 1. The caching infrastructure existed but was never enabled.

### 2. Cross-fold contamination in original fold design (session 1)
**What:** Expanding training windows in later folds included earlier folds' val/test data.
**Impact:** All ablation results from session 1 were invalid (artificially inflated Sharpe).
**Fix:** Hole-punching in `split_data()` + label-horizon buffer. Applied in session 1.
**Root cause:** Standard expanding-window walk-forward was used without considering that val/test windows become training data in later folds.

### 3. Grid sweep instead of diagnostic approach (session 3)
**What:** Wrote a 4-round grid sweep (8 lr x 5 bs x 6 seq x 7 reg = 26 experiments x 7 folds each). Brute force, no thinking.
**Impact:** Hours of GPU time wasted on uninformed exploration.
**Fix:** Switched to Karpathy-style: single experiment, evaluate, diagnose, adjust, repeat.
**Root cause:** Defaulted to "throw compute at it" instead of understanding the problem first.

### 4. Running all 7 folds per sweep experiment (session 3)
**What:** Each sweep experiment ran the full 7-fold walk-forward evaluation.
**Impact:** 7x slower than necessary. Fold 7 is the only fold that matters for production.
**Fix:** Super-fold approach — train once on fold 7's data, validate/test on combined windows.
**Root cause:** Didn't question the assumption that "more folds = more rigorous."

### 5. Sliding window across non-contiguous data (session 3-4)
**What:** FXDataset creates sliding windows from concatenated non-contiguous test windows. Windows crossing fold boundaries produce garbage inputs.
**Impact:** For super-fold test set, ~41% of windows would span time gaps (with seq_len=60).
**Fix:** Must evaluate each fold's test window separately, then aggregate predictions.
**Root cause:** Treated concatenated DataFrames as contiguous without considering the sliding window semantics.
**Note:** For training data with holes, the purge gaps + label-horizon buffer already protect the boundaries. The "garbage" training windows are just noise, not systematic bias.

### 6. Dead config parameters (session 3)
**What:** Added `dropout` and `huber_delta` to the sweep config but never plumbed them through to `_make_heads()` or `HuberLoss()`. Experiments sweeping these would show no effect.
**Impact:** Wasted experiments if those params were tested.
**Fix:** Either wire them through or remove from sweep config.
**Root cause:** Wrote config first, verified plumbing never.

### 7. No separation of concerns (session 3)
**What:** The autoresearch runner, the experiment evaluator, and the dashboard were all tangled in one script.
**Impact:** Hard to debug, hard to reuse, hard to monitor.
**Fix:** Decouple: runners log to structured JSON/TSV, dashboard reads logs independently.
**Root cause:** Rushing to show results instead of designing the system.

### 8. Making assumptions without data (sessions 2-4)
**What:** Stated training takes "~50s per fold" and "~6 min per experiment" without measuring. Made claims about what hyperparameters "should" work without evidence.
**Impact:** Bad estimates, wrong experiment priorities, user frustration.
**Fix:** Measure everything. State what's measured vs. assumed. Never claim without data.
**Root cause:** Substituting confidence for measurement.

---

## What Went Right

1. **Caught cross-fold contamination** — the hole-punching fix is correct and verified (0 contamination in test)
2. **Label-horizon buffer** — properly prevents fwd_ret_5d from peeking into excluded windows
3. **Super-fold design** — train=3113 rows (holed fold 7), val=915 rows (all val windows), test=1170 rows (all test windows), zero overlap
4. **Per-window breakdown** — reporting Sharpe per fold's test window for explainability
5. **Checkpoint/resume** — autoresearch script saves best config after each experiment

---

### Session 5 (14671f7b) — Autoresearch redesign + agent-driven loop

**Mistakes caught by user:**
9. **Pre-baked experiment lists instead of agent reasoning** — wrote Python code that generated 102 experiments per backbone as a static grid. This is the opposite of Karpathy's autoresearch where the agent decides each experiment based on prior results. The intelligence must be in Claude, not in a for-loop.
10. **Only tweaking learning rate** — anchored on lr sweep for 6 experiments while ignoring batch size, seq_len, weight_decay, and their interactions. Expert reasoning requires exploring all dimensions informed by theory.
11. **Not persisting experiments across crashes** — deleted old JSONL logs between runs, losing results. Fixed: JSONL is append-only, sweep_history/ keeps timestamped archives.
12. **Dashboard missing features** — no train/val/test tabs, no super-fold aggregate row, no per-window trading metrics, no multi-backbone support. Fixed iteratively.

**Architecture redesign:**
- `run_autoresearch.py` is now a single-experiment executor (takes backbone + config + description)
- Claude IS the autoresearch loop: reads JSONL → analyzes deeply → decides next experiment → calls runner
- No pre-baked experiment lists. Every experiment decision is reasoned from prior results + literature.

---

## Architecture Decisions (Current)

### Separation of Concerns
```
Claude (agent)        — reads results, reasons, decides next experiment
run_autoresearch.py   — executes ONE experiment, logs JSONL (no intelligence)
dashboard.html        — reads logs, renders live status (decoupled)
baseline.py           — single-model evaluation (unchanged)
run_ablation.py       — multi-backbone comparison (unchanged)
```

### Experiment Design Principles
1. **Super-fold evaluation** — train on fold 7's holed data, val/test across ALL 7 fold windows combined
2. **Model must do well on ALL folds** — composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds
3. **Per-window inference** — never create sliding windows across non-contiguous data
4. **Epoch-bound** — minimum 20 epochs with early stopping, NOT time-bound
5. **Cache everything** — data to `.data_cache/`, JSONL append-only, sweep_history/ for archives
6. **Measure, don't assume** — log wall time, per-window metrics, val loss curves

### Autoresearch Principles (Karpathy-adapted for MLFin)
1. Claude reads results after EVERY experiment and decides the next one
2. Each decision must be justified: which folds improved? What does val/test gap mean? What does literature say?
3. Never grid search — form a hypothesis, test it, analyze, iterate
4. Every hyperparameter choice cites: published paper, model developer guidelines, or measured empirical result
5. When stuck, research deeper: read the LFM2 technical report, adapter tuning papers, FX microstructure literature
6. Run 100+ experiments per backbone — explore lr, batch_size, seq_len, weight_decay, patience, grad_clip, and their interactions
