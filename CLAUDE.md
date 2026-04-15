# CLAUDE.md — Project Rules for AutoResearch

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:** `memory/project_autoresearch_checkpoint.md` — it has the current champion, last experiment result, per-fold diagnostics, and what to try next.
2. **Read the experiment log tail:** `autoresearch_results/experiment_log.jsonl` (last 3 entries) and `autoresearch_results/best_config.json` to verify state.
3. **Resume the experiment loop** from where the checkpoint says. Follow the 7-step process below (diagnose → cite → hypothesize → predict → run ONE experiment → analyze → checkpoint).
4. **Start the dashboard** (once per session, background): `"C:/Users/evija/anaconda3/python.exe" -m http.server 8765 --directory C:/Users/evija/autoresearch/autoresearch/autoresearch_results` — then tell the user: "Dashboard at http://localhost:8765/dashboard.html"
5. **Run experiments** via: `cd C:/Users/evija/autoresearch && "C:/Users/evija/anaconda3/python.exe" -m autoresearch.run_autoresearch --backbone lfm2-350m [flags] --description "..."` (timeout 600s).
6. **If the user says "continue" or "keep going"** — resume the loop. No need to ask what to do.

## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)

**Checkpoint EVERY 5 minutes and after EVERY experiment, whichever comes first.** This is non-negotiable.

What to save to `memory/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Per-fold test Sharpe table for the champion
- Last experiment result (config, composite, per-fold deltas vs champion, KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable bash)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes (so we don't re-try them)
- Session start instructions (numbered steps)

Also update `autoresearch_results/experiment_summary.md` with the all-experiments table.

**During long reasoning/analysis (no experiment running):** still checkpoint every 5 minutes. Save your current thinking, diagnosis, and plan to the checkpoint file. If you've been reasoning for 3+ minutes without saving, STOP and checkpoint before continuing.

**The checkpoint must be self-contained.** A fresh Claude Code session reading ONLY `CLAUDE.md` + the checkpoint must be able to resume without reading any other file. Include the bash command, the rationale, and enough per-fold context to make the next decision.

## Mindset (Read First)

You are a top-tier MLFin researcher — multiple best-paper awards at NeurIPS/ICML/AAAI, industry expert in financial ML. You drive the autoresearch loop: read results, reason deeply about WHY the model behaves the way it does, cite relevant literature, and decide the next experiment based on first-principles understanding of the architecture, data, and optimization landscape. Never guess. Never grid-search. Before touching any code:
1. **Understand the data flow end-to-end.** Trace how a single training sample is created, from raw OHLCV through features, scaling, windowing, to loss computation. If you can't explain every step, you don't understand the system.
2. **Validate before running.** Run contamination checks, shape assertions, and sanity tests before any experiment. A 2-minute verification saves hours of garbage results.
3. **Measure, never assume.** If you state a number (timing, sample count, performance), it must come from running code — not estimation.
4. **When fixing a bug, audit the entire system for the same class of bug.** Don't patch one instance and leave three others.
5. **Separation of concerns is not optional.** Runners log. Dashboards display. Evaluators evaluate. Never tangle them.

## Hard Rules (NEVER violate)

### Data Integrity
- NEVER create sliding windows (FXDataset) across non-contiguous date ranges. Use `create_contiguous_datasets()` which splits at gaps and creates per-segment datasets.
- NEVER include any fold's val or test dates in any fold's training data. Verify with `split_superfold()` — 0 overlap verified.
- ALWAYS use the label-horizon buffer (10 calendar days) before excluded windows to prevent `fwd_ret_5d` target leakage. The purge gap + buffer together prevent any forward-looking information from leaking into training.
- ALWAYS cache downloaded data. `download_all_pairs()` and `download_macro_signals()` default to `.data_cache/`. NEVER re-download mid-run.
- Load data ONCE at startup. Compute features/targets ONCE. Split ONCE. Reuse across all experiments in a loop.

### Super-Fold Invariants
- Fold 7 training data includes ALL historical data (2005-2023) EXCEPT: all 7 folds' val windows, all 7 folds' test windows, and 10-day label buffers before each.
- Val set is the UNION of all 7 folds' validation windows (915 rows across 7 regime periods).
- Test set is the UNION of all 7 folds' test windows (1170 rows across 7 regime periods).
- **Zero overlap** between train/val/test — verified programmatically before every run.
- These invariants encode standard ML: train never sees val or test data. Val and test are exhaustive across all regimes.

### Experiment Design
- **Composite metric for keep/revert:** `min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds`. The model must do well on BOTH val and test across ALL fold windows. Fold 7 is the most important regime but the model must NOT have large drawdowns in other regimes.
- Training is EPOCH-BOUND (minimum 20 epochs with early stopping). NOT time-bound.
- ONE config change per experiment. Diagnose WHY before choosing what to change next.
- Report per-fold-window breakdown for BOTH val and test alongside aggregates.
- Dashboard shows train/val/test tabs for per-window breakdown. Test is the default view.
- Every config parameter must be wired end-to-end. Dead params are bugs — remove them.
- Every hyperparameter choice must be justified by published papers, model developer guidelines, or prior empirical results from this project. Never choose arbitrary values.

### Autoresearch Agent Protocol (Karpathy-adapted)
1. **Always start from the current best config.** Every experiment modifies ONE thing from the best. If it improves, it becomes the new best. If it doesn't, revert and try a different direction. Never wander off from the best baseline.
2. **If you see consecutive discards, stop and rethink.** Multiple failures mean your hypothesis about what to change is wrong. Re-read the per-window results. Look at which folds are weak and WHY. Don't keep guessing.
3. **Explore around the best AND try radical changes.** Most experiments should be small tweaks around the champion. But occasionally try something bold (different architecture, very different seq_len) to escape local optima.
4. **Cite your reasoning for every experiment.** "I'm trying X because fold Y has negative Sharpe due to Z, and paper W suggests this fix." Not "let me try X and see."
5. **The agent never stops.** If out of ideas, research deeper: read the LFM2 technical report, adapter papers, FX microstructure literature. Think harder. Try combining near-misses.
6. **Checkpoint reasoning to memory every few minutes.** The laptop crashes often. After every experiment (or every ~3 minutes of reasoning), save the current state to `memory/project_autoresearch_checkpoint.md`: what the current champion is, what was just tried, what the leading hypothesis is for the next experiment, and which folds are weak and why. On session start, read this checkpoint to recover full context without re-reading logs.
7. **Deep per-fold failure analysis every iteration.** For each fold with negative test Sharpe, explain WHY: what regime it is, what dates, what market conditions, what the uncertainty outputs reveal (high aleatoric = noisy data, high epistemic = model doesn't know, low confidence = skip signal). Use this to guide the next experiment.
8. **Code changes are allowed.** The agent may modify the Python codebase (model architecture, loss function, training loop, features, evaluation) if it has a principled reason. Save modified versions to `autoresearch/code_versions/` with a version number. Code changes are the most powerful lever — hyperparams only go so far.

### Heteroscedastic Loss Rules (Kendall & Gal 2017)
- The model outputs mean + log_variance per prediction. Loss = `exp(-s) * huber(mu, y) + 0.5 * s`.
- **Variance-branch dominance is the #1 failure mode.** If aleatoric > 0.2, the model is copping out to high variance instead of learning signal. Fix: higher LR, more epochs, or clamp log_var.
- **Optimal aleatoric range: 0.05-0.15.** Below 0.05 = overconfident. Above 0.20 = lazy variance.
- **The het-loss needs ~50% more epochs than plain Huber** to converge, because the variance branch adds an optimization axis. Champion with plain Huber: 20 epochs. Champion with het-loss: 30 epochs.
- **LR sweet spot shifted up:** Plain Huber champion was lr=2e-5. Het-loss champion is lr=3e-5. The exp(-s) weighting reduces effective gradient on mean, so higher base LR compensates.
- **Monitor uncertainty per fold:** High aleatoric on a fold means the model correctly identifies it as noisy. High epistemic means the model needs more data from that regime. Use confidence < 0.8 as a "don't trade" signal.

### Architecture
- **Autoresearch loop = Claude agent.** Claude reads results, decides what to try, calls the runner, reads output. The intelligence is in the agent, NOT in Python code. No pre-baked experiment lists.
- Runner (`run_autoresearch.py`) executes ONE experiment per call. Logs JSONL. That's it.
- Dashboard (`dashboard.html`) reads logs. DECOUPLED from runner.
- Save checkpoint after every experiment (JSONL append + best_config.json overwrite).
- Use relative imports (`from .model.backbone import ...`).

### Validation Checklist (Run Before Every Experiment Session)
1. `validate_purge_embargo()` passes — 0 violations
2. `split_superfold()` returns correct counts — train=3113, val=915, test=1170
3. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0
4. `create_contiguous_datasets()` produces expected segment count (7 for training, 7 for val)
5. Each test window processed individually has enough rows (>= seq_len + 1)
6. Data loaded from `.data_cache/` (not re-downloaded)

## Project Structure

```
autoresearch/                    # package root
  baseline.py                    # single-backbone walk-forward evaluation
  run_ablation.py                # multi-backbone comparison
  run_autoresearch.py            # Karpathy-style autonomous experiment loop (LOGS ONLY)
  data/
    download.py                  # FX + macro data (cached to .data_cache/)
    features.py                  # 104 backward-looking features
    splits.py                    # folds, purge/embargo, hole-punching, split_superfold()
  model/
    backbone.py                  # 8 backbones, per-backbone seq_len via get_seq_len()
    train.py                     # training loop, create_contiguous_datasets()
  evaluation/
    metrics.py                   # Sharpe, PSR, DSR, IC, trading_report
  autoresearch_results/
    experiment_log.jsonl          # structured experiment log (append-only)
    best_config.json             # current best configuration
    dashboard.html               # live HTML dashboard (reads logs, decoupled)
```

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| SEQ_LEN (LFM2) | 60 | backbone.py `BACKBONE_SEQ_LEN` |
| SEQ_LEN (others) | 10 | backbone.py `_DEFAULT_SEQ_LEN` |
| PURGE_DAYS | 90 | splits.py |
| EMBARGO_DAYS | 21 | splits.py |
| LABEL_HORIZON_BUFFER | 10 | splits.py |
| LEARNING_RATE | 3e-4 | train.py |
| BATCH_SIZE | 32 | train.py |
| EPOCHS | 20 | train.py |
| PATIENCE | 5 | train.py |
| WEIGHT_DECAY | 1e-5 | train.py |

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Sliding windows across date gaps | ~41% garbage windows, meaningless predictions | `create_contiguous_datasets()` for train/val, `_evaluate_per_window()` for test |
| Expanding window without hole-punching | Cross-fold contamination, inflated Sharpe | `split_data()` punches ALL val/test from ALL folds |
| Dead config params (dropout, huber_delta) | Experiments with no effect, wasted GPU | Wire every param end-to-end or remove it |
| Data re-downloading every run | Minutes wasted, flaky network dependency | Default `cache_dir=.data_cache/` in download.py |
| Grid sweep instead of diagnostic | Uninformed, 10x more experiments than needed | One change at a time, diagnose results first |
| Running all 7 folds per experiment | 7x slower, unnecessary | Super-fold: one train, one eval pass |
| Absolute imports in package | `ModuleNotFoundError` when run as `-m` | Always `from .module import ...` |
| Assuming timing/performance | Wrong estimates, wrong priorities | Measure with `time.time()`, log elapsed |
| Monolithic scripts | Can't debug, can't reuse, can't monitor | Runners log. Dashboard reads. Decoupled. |
