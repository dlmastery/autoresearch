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

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning, whichever comes first.** This is the #1 non-negotiable rule. The laptop WILL crash. Every minute of uncheckpointed work is lost work.

**Checkpoint trigger points (ALL mandatory):**
1. **Immediately after every experiment completes** — before any analysis or reasoning about results
2. **Every 5 minutes during reasoning/analysis** — if you've been thinking for 3+ minutes without saving, STOP and checkpoint
3. **Before starting any code change** — save current state so crash during edit doesn't lose experiment context
4. **After any code change** — save the new code state and what was changed
5. **Before starting the next experiment** — checkpoint must contain the exact bash command ready to paste

What to save to `memory/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Per-fold test Sharpe table for the champion
- Last experiment result (config, composite, per-fold deltas vs champion, KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable bash)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes (so we don't re-try them)
- Session start instructions (numbered steps)
- **Full experiment history summary** — every experiment number, config delta, result, KEEP/DISCARD

Also update `autoresearch_results/experiment_summary.md` with the all-experiments table.

**During long reasoning/analysis (no experiment running):** still checkpoint every 5 minutes. Save your current thinking, diagnosis, and plan to the checkpoint file. If you've been reasoning for 3+ minutes without saving, STOP and checkpoint before continuing.

**The checkpoint must be self-contained.** A fresh Claude Code session reading ONLY `CLAUDE.md` + the checkpoint must be able to resume without reading any other file. Include the bash command, the rationale, and enough per-fold context to make the next decision. A new session should be able to pick up EXACTLY where the previous one left off — same experiment number, same champion, same next-experiment rationale.

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
- **60-second cooldown after each experiment** to let the GPU/CPU cool. Use `sleep 60` between runs.
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

### Research-Driven Experiment Selection (STRICT — no blind sweeps)
The experiment loop is NOT a grid search. It is a research process. Every single experiment must follow this exact sequence:

**Step 1 — Diagnose the champion's weakness.** Look at the per-fold test results. Which folds are weakest? What regime are they? What do the uncertainty metrics say? What does the win/loss spread look like for those folds? Identify the SPECIFIC failure mode (e.g., "fold 2 post-crash recovery has low IC=0.08, high epistemic uncertainty — model hasn't seen enough crisis-recovery data").

**Step 2 — Search the literature.** Based on the diagnosis, search arXiv / known papers for techniques that address the failure mode. Examples:
- Weak on volatile regimes → regime-aware training, volatility scaling (Kiraly et al. 2020)
- High epistemic in specific folds → data augmentation, ensemble methods (Lakshminarayanan et al. 2017)
- Overfitting to majority regime → focal loss (Lin et al. 2017), re-weighting
- Architecture ceiling hit → residual connections (He et al. 2016), attention mechanisms (Vaswani et al. 2017)
- LR too high/low → cyclical LR (Smith 2017), warmup (Goyal et al. 2017)

**Step 3 — Form a hypothesis and predict the outcome.** Write down: "I hypothesize that [change X] will improve [metric Y] on [fold Z] because [paper/principle]. I predict composite will move from [current] to approximately [target]." If you can't write this sentence, you don't understand what you're doing. Stop and think more.

**Step 4 — Run ONE experiment.** Execute the change. ONE change only.

**Step 5 — Analyze against prediction.** Did the result match your prediction? If yes, why? If no, what does that tell you about your mental model? Update your understanding.

**Step 6 — Document everything.** Write the full cycle (diagnosis → literature → hypothesis → prediction → result → learning) into the experiment log and checkpoint. This creates a research trail that prevents repeating failed ideas.

**The goal is monotonic improvement.** Every experiment should have a principled reason to believe it will improve composite score. Random guessing wastes GPU and time. If you're out of ideas for hyperparameters, the answer is almost always a CODE CHANGE — modify the architecture, loss function, or feature engineering.

### Monotonic Quality Progression (NEVER regress)
The experiment loop must work towards monotonic increase in quality. This means:
- **Never run an experiment you can't justify.** Every experiment must have a written rationale citing literature or prior empirical evidence from this project.
- **Track the champion lineage.** Document the chain: Exp1 (baseline) → Exp5 (residual skip, +3x) → Exp10 (LR bump, +1.2x) → etc. Each link must explain WHY the improvement happened.
- **When you hit a plateau, go deeper.** If 3+ consecutive experiments are DISCARD, you're in a local optimum. The answer is NOT more hyperparameter tweaks — it's a structural change: different architecture, different loss, different features, different training procedure.
- **Protect gains.** When trying bold changes, if the result is far worse (composite drops >2.0), investigate WHY before trying the next thing. Understanding failures is as valuable as finding improvements.
- **Quality ratchet:** once a metric improves, treat the new level as the floor. If a change improves test Sharpe but regresses val Sharpe below the previous champion, it's a DISCARD — both must improve or at least hold.

### MLOps Documentation Standards (MANDATORY)
You are a strong MLOps engineer. Every artifact and every experiment must be documented in proper, readable markdown. No exceptions.

**`autoresearch_results/experiment_summary.md`** — the master experiment log. Updated after EVERY experiment. Format:

```markdown
## Experiment Log — [Backbone] Phase

### Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected composite change]
- **Result:** Composite [X] | Test Sharpe [Y] | Val Sharpe [Z] | [N]/7 positive folds
- **Per-fold test Sharpe:** F1=[X] F2=[X] F3=[X] F4=[X] F5=[X] F6=[X] F7=[X]
- **Classification:** Precision=[X] Recall=[X] F1=[X] F2=[X] MCC=[X]
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned, why result matched/differed from prediction]
- **Win/Loss:** [summary — see per-trade spreadsheet in trade_logs/]
```

**`autoresearch_results/trade_logs/`** — per-experiment trade-level detail (see Trade-Level Win/Loss Logging below).

**Key documentation principles:**
1. **Readable by a human who wasn't there.** Someone reading the experiment summary 6 months from now must understand WHY each experiment was run and WHAT was learned.
2. **No orphan artifacts.** Every file must be referenced from either the checkpoint, experiment summary, or winner README.
3. **Consistent formatting.** Same table format, same metric names, same precision (4 decimal places for ratios, 2 for percentages).
4. **Append-only experiment log.** Never delete or rewrite experiment entries. If an experiment was wrong (e.g., bug found), add a note — don't erase history.

### Heteroscedastic Loss Rules (Kendall & Gal 2017)
- The model outputs mean + log_variance per prediction. Loss = `exp(-s) * huber(mu, y) + 0.5 * s`.
- **Variance-branch dominance is the #1 failure mode.** If aleatoric > 0.2, the model is copping out to high variance instead of learning signal. Fix: higher LR, more epochs, or clamp log_var.
- **Optimal aleatoric range: 0.05-0.15.** Below 0.05 = overconfident. Above 0.20 = lazy variance.
- **The het-loss needs ~50% more epochs than plain Huber** to converge, because the variance branch adds an optimization axis. Champion with plain Huber: 20 epochs. Champion with het-loss: 30 epochs.
- **LR sweet spot shifted up:** Plain Huber champion was lr=2e-5. Het-loss champion is lr=3e-5. The exp(-s) weighting reduces effective gradient on mean, so higher base LR compensates.
- **Monitor uncertainty per fold:** High aleatoric on a fold means the model correctly identifies it as noisy. High epistemic means the model needs more data from that regime. Use confidence < 0.8 as a "don't trade" signal.

### Winner Archiving Protocol (MANDATORY for every NEW BEST)
Every time a new champion is found (status=KEEP and composite > previous best), archive ALL artifacts to a self-contained subdirectory. The archive must be fully portable — someone can copy the directory to another machine and reproduce + run inference without any external dependencies beyond the conda environment.

**Directory structure:** `autoresearch_results/winners/<backbone>_exp<N>_<short_description>/`

```
winners/
  mlp_exp32_residual_seed0/
    README.md                    # Full description (see template below)
    config.json                  # Exact config that produced this winner
    model_checkpoint.pt          # Saved model weights (copy of best_model.pt)
    experiment_log_entry.json    # The JSONL entry for this experiment
    per_fold_results.json        # Full per-fold val + test breakdown
    code/                        # Frozen snapshot of ALL source code at time of win
      backbone.py
      train.py
      features.py
      splits.py
      metrics.py
      run_autoresearch.py
    inference/
      predict.py                 # Standalone inference script with sample usage
      README_inference.md        # How to load model and run predictions
    reproduction/
      reproduce_log.txt          # Output from reproduction run
      seed_variance.json         # Cross-seed results if available
```

**README.md template for each winner:**
- Model name + experiment number
- Champion composite score, test Sharpe, val Sharpe
- Per-fold test Sharpe table (all 7 folds)
- Per-fold val Sharpe table
- Full hyperparameter config
- Architecture description (layers, activation, skip connections, etc.)
- Key insight: WHY this config won (what change from previous champion)
- Training details: epochs run, early stopping epoch, training time
- Uncertainty metrics: aleatoric, epistemic, confidence per fold
- Traditional ML metrics: precision, recall, F1, F2 (direction classification)
- Reproduction status: seeds tested, variance observed
- Sample inference code snippet

**After archiving:** Rerun the winner to verify reproduction. The reproduction log goes into `reproduction/reproduce_log.txt`. If the reproduction fails (composite differs by >0.5), flag it and investigate before proceeding.

**Model checkpoint (`model_checkpoint.pt`) MUST be portable and self-contained:**
Include in the torch.save dict:
- `model_state_dict` — all trainable weights
- `config` — hyperparameters dict (matches the `--seed` run command)
- `scaler_mean`, `scaler_scale` — StandardScaler parameters (np.ndarray[n_features])
- `feature_columns` — list of feature names in order (for schema validation at inference)
- `target_columns` — list of target names (e.g. `['ret_1d', 'ret_5d']`)
- `n_features` — int, feature count
- `composite`, `description`, `backbone`, `experiment_num` — provenance

The checkpoint must be loadable and reusable WITHOUT the source repo. Someone can rebuild the model, apply the scaler, and make predictions from the checkpoint alone + the architecture definition.

**The `predict.py` inference script must:**
1. Load the model checkpoint
2. Accept raw feature input (or date range to download)
3. Output: prediction (mean), confidence, aleatoric uncertainty, epistemic uncertainty
4. Include a `__main__` block with a working example
5. Print results in a clear table format

**Trading Strategy section (MANDATORY in every winner README.md):**
Must include the following for any user to deploy the model:
1. **Signal Generation** — inputs, outputs, MC Dropout usage
2. **Entry rules** — pseudocode with thresholds (magnitude + confidence)
3. **Position sizing** — Kelly fraction, per-trade cap
4. **Exit rules** — horizon matching, stop-loss policy
5. **Rebalancing cadence** — daily/intraday/weekly
6. **Per-regime performance table** — accuracy/MCC/Sharpe per fold
7. **Risk controls** — daily loss cap, drawdown pause, regime shift detection
8. **Expected performance** — Sharpe, return, drawdown estimates (pre/post cost)
9. **Caveats and warnings** — seed variance, pair specificity, feature dependencies, transaction costs
10. **Reference to inference code** — link to `inference/predict.py`

### Google Colab Notebook (MANDATORY for every winner)
For every archived winner, generate a self-contained Google Colab notebook at `autoresearch_results/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb` that anyone can open in Colab and run end-to-end.

**The Colab notebook must contain:**
1. **Setup cell:** `!pip install` all dependencies, clone repo or upload weights
2. **Data download cell:** download FX + macro data using `download.py` logic (or bundled CSV)
3. **Feature engineering cell:** compute all 104 features with clear explanations
4. **Training cell:** full training loop reproducing the winner config exactly — including super-fold split, contiguous datasets, loss function, optimizer, early stopping. Print per-epoch loss + validation metrics.
5. **Evaluation cell:** evaluate on all 7 test fold windows, print per-fold Sharpe/IC/win-rate table, compute composite score
6. **Inference cell:** load trained model, accept a date range, produce predictions with confidence/aleatoric/epistemic bands. Show a sample prediction table.
7. **Visualization cell:** plot equity curves per fold, prediction vs actual scatter, uncertainty calibration, confusion matrix
8. **Export cell:** save model weights + config for deployment

**Notebook principles:**
- Every cell must have a markdown header explaining what it does and WHY
- Include the champion config as a clearly visible dict at the top
- Use `torch.manual_seed()` for reproducibility
- Print all key metrics at the end in a summary table
- Target runtime: <5 minutes on Colab free tier (T4 GPU or CPU)
- The notebook must be SELF-CONTAINED — no imports from the autoresearch package (inline all necessary code)

### Traditional ML Metrics (MANDATORY for every experiment)
In addition to financial metrics (Sharpe, Sortino, IC, etc.), compute and log direction-classification metrics for every experiment. The trading strategy uses `sign(prediction)` as the directional bet, so treat direction prediction as binary classification:
- **Positive class:** model predicts UP (pred > 0) and actual move is UP (actual > 0)
- **Negative class:** model predicts DOWN (pred < 0) and actual move is DOWN (actual < 0)

Metrics to compute per fold AND aggregate:
- **Precision:** TP / (TP + FP) — of all UP predictions, how many were correct
- **Recall:** TP / (TP + FN) — of all actual UP moves, how many did we catch
- **F1 Score:** harmonic mean of precision and recall
- **F2 Score:** weighted harmonic mean favoring recall (beta=2), useful for FX where missing a move costs more than a false signal
- **Accuracy:** (TP + TN) / total — same as hit rate / win rate but explicit
- **Matthews Correlation Coefficient (MCC):** balanced measure even with class imbalance
- **Confusion matrix counts:** TP, FP, TN, FN per fold

These must appear in:
1. `trading_report()` output in `metrics.py`
2. Per-window results in JSONL log entries
3. Dashboard per-window tables
4. Winner archive `per_fold_results.json`
5. Experiment summary markdown

### Trade-Level Win/Loss Logging (MANDATORY for every experiment)
For EVERY experiment, produce a per-trade win/loss spreadsheet on test data. This is critical for understanding WHERE the model makes and loses money — not just aggregate metrics.

**Output file:** `autoresearch_results/trade_logs/exp<N>_trades.csv`

**Columns (one row per trade/day in test data):**
| Column | Description |
|--------|-------------|
| date | Trade date |
| fold | Which test fold window (1-7) |
| regime | Regime label (e.g., "Post-crash recovery") |
| prediction | Raw model prediction (mean) |
| pred_direction | +1 (long) or -1 (short) |
| actual_return | Actual daily return |
| actual_direction | +1 (up) or -1 (down) |
| strategy_return | sign(pred) * actual_return |
| cumulative_return | Running cumulative return within fold |
| confidence | Model confidence (1 - epistemic) |
| aleatoric | Aleatoric uncertainty |
| epistemic | Epistemic uncertainty |
| correct | 1 if pred_direction == actual_direction, else 0 |
| pnl_bps | P&L in basis points |

**Per-fold summary at bottom of CSV (or separate `exp<N>_trade_summary.json`):**
- Total trades, wins, losses per fold
- Average win size (bps), average loss size (bps)
- Largest single win, largest single loss
- Win/loss ratio (avg_win / abs(avg_loss))
- Streak analysis: max consecutive wins, max consecutive losses
- Confidence-stratified accuracy: accuracy when confidence > 0.9 vs < 0.9

**This data enables:**
- Identifying specific dates/regimes where the model fails
- Confidence calibration analysis (does high confidence = high accuracy?)
- Position sizing research (Kelly criterion, volatility scaling)
- Filtering rules (skip trades below confidence threshold)

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
    metrics.py                   # Sharpe, PSR, DSR, IC, trading_report + precision/recall/F1/F2/MCC
  autoresearch_results/
    experiment_log.jsonl          # structured experiment log (append-only)
    best_config.json             # current best configuration
    dashboard.html               # live HTML dashboard (reads logs, decoupled)
    experiment_summary.md        # master human-readable experiment log (updated every experiment)
    trade_logs/                  # per-trade win/loss CSVs for every experiment
      exp<N>_trades.csv          # one row per trade on test data
      exp<N>_trade_summary.json  # per-fold trade statistics
    winners/                     # archived champions (one subdir per winner, fully self-contained)
      <backbone>_exp<N>_<desc>/  # e.g. mlp_exp32_residual_seed0/
        README.md                # full description, metrics, reproduction status
        config.json              # exact config
        model_checkpoint.pt      # saved weights
        code/                    # frozen source snapshot
        inference/               # predict.py + inference README
        reproduction/            # reproduction logs + seed variance
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
