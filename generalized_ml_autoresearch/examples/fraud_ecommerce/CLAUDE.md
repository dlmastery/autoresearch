# CLAUDE.md — Project Rules for FDB fraudecom autoresearch

> Derived from `generalized_ml_autoresearch/templates/CLAUDE_template.md` and the
> source FX `autoresearch/CLAUDE.md`. Encodes the rules that govern this specific
> autoresearch project on the Amazon Science FDB `fraudecom` benchmark.

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no separate Python agent. When a session starts:

1. **Read the experiment log tail:** `autoresearch_results/experiment_log.jsonl` (last 3 entries) and `autoresearch_results/best_config.json` to verify state.
2. **Read this CLAUDE.md** to refresh on project-specific rules.
3. **Read the reasoning_annotations.json tail** to see what was last hypothesized and learned.
4. **Resume the experiment loop** from where the previous session left off. Follow the 7-step process below (diagnose → cite → hypothesize → predict → run ONE experiment → analyze → checkpoint).
5. **Start the dashboard** (background): `python -m http.server 8765 --directory autoresearch_results` — then tell the user: "Dashboard at http://localhost:8765/dashboard.html"
6. **Run experiments** via: `python -m generalized_ml_autoresearch.core.runner --config <config>.yaml --description "..."`.

## Project context

- **Dataset:** Amazon Science Fraud Dataset Benchmark, `fraudecom` task. The most unsaturated of the 9 FDB datasets per Grover et al. 2023 (arXiv:2208.14417).
- **Size:** 151,112 transactions over 2015-01-01 → 2015-12-16, 9.36% fraud rate.
- **Features:** 18 columns after our preprocessing — 9 base (purchase_value, device_id, source, browser, age, ip_address, country, time_since_signup, signup_hour) + 3 cyclical (purchase_hour, purchase_dayofweek) + 7 entity-velocity (device_id_freq, ip_address_freq, country_freq, source_freq, browser_freq, device_fraud_rate_train, country_fraud_rate_train).
- **FDB published baselines (chronological 80/20):** AFD-TFI 0.636 (ceiling), AutoGluon 0.522, H2O 0.518, Auto-sklearn 0.515.
- **Primary metric:** AUC-ROC on the test split.
- **Composite metric:** `min(test_auc, val_auc) − 0.05 × n_folds_below_threshold`, threshold=0.55. This is the gate for KEEP/DISCARD.

## Hard Rules (NEVER violate)

### Data Integrity

1. **NEVER use random or stratified k-fold CV on this dataset.** It has a `purchase_time` column and severe concept drift — stratified CV gives an inflated AUC of 0.7738 while the honest chronological holdout gives 0.5098 on the same XGBoost config. ONLY use `HoldoutSplit(order=time, test_fraction=0.2, val_fraction=0.1)` to mirror the FDB protocol.
2. **NEVER compute entity-aggregation features (frequency counts, target encodings) using val or test rows.** The train-period boundary is `n_train = n - n_val - n_test = 70%` of the chronologically-sorted dataset. Computing on the first 80% (= train + val) leaks val rows into their own predictor values; the diagnostic signature is `val_AUC ≫ test_AUC` (e.g. Exp 5 had val 0.9988 vs test 0.5297).
3. **NEVER re-download data mid-run.** The processed CSVs (`features.csv`, `features_velocity.csv`) are cached at `data/`. The raw_train.csv and raw_test.csv come from the public mirror at `pmarkoo/Identifying-Fraudulent-Activities` (since FDB requires Kaggle credentials).
4. **Load data ONCE at startup.** Compute features ONCE per feature-set version. Split ONCE per experiment. Reuse across all backbones in a session.

### Evaluation Protocol Invariants

The chosen evaluation protocol is **chronological holdout (HoldoutSplit, order=time, 70/10/20)**.

- Train: rows 0 to ~106k (first 70% by purchase_time)
- Val: rows ~106k to ~121k (next 10%, used for early stopping only — never for feature computation)
- Test: rows ~121k to 151k (last 20%, never seen during training or feature engineering)
- Programmatic verification: `validate_no_overlap()` runs before every experiment.

### Experiment Design

- **Composite for KEEP/DISCARD:** `min(test_auc, val_auc) − 0.05 × n_folds_below_0.55`. Below 0.55 is automatic DISCARD.
- ONE config change per experiment. Diagnose WHY before choosing what to change next.
- Report per-fold breakdown alongside aggregates. (Holdout has only one fold, but the dashboard renders the per-fold table for consistency.)
- Every hyperparameter choice must be justified by a published paper or prior project result.

## Per-Backbone N-Experiment Mandate

**Every backbone gets a full exploration cycle before being declared exhausted.** The active backbone roster for this project is:

### Tier 3 — Gradient Boosted Machines (50 experiments TOTAL across the three, split 20/15/15)

| Backbone | Exps | Status |
|----------|------|--------|
| xgboost | 20 | 1 done (Exp 6 champion: test AUC 0.5414); 19 remain |
| lightgbm | 15 | 1 done (Exp 7: test AUC 0.5305); 14 remain |
| catboost | 15 | 1 done (Exp 8: test AUC 0.5245); 14 remain |

### Tier 1 — Neural backbones

| Backbone | Status |
|----------|--------|
| mlp | 1 done (Exp 9: test AUC 0.4883 — well below GBM family) |
| ft_transformer | not yet attempted |
| lstm | not applicable (tabular task, no sequence) |

### Tier 2 — 2024-2026 SOTA tabular foundation models

To be considered after the GBM family is exhausted: TabPFN, TabFoundry, FT-Transformer with attention over high-cardinality embeddings.

**Cite the paper for every experiment.** No "let me try X" — every config change is justified by a published result or a prior empirical finding from this project.

## Per-Backbone SOTA Training Recipes (re-derive per backbone)

| Backbone | Iterations / Epochs | Patience | LR | Special config | Paper |
|----------|---------------------|----------|----|----|------|
| xgboost | 600 trees | 40 | 0.05 | max_depth=6, min_child_weight=5, subsample=0.85 | Chen & Guestrin 2016 KDD (arXiv:1603.02754) |
| lightgbm | 800 trees | 50 | 0.04 | num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85 | Ke et al. 2017 NeurIPS |
| catboost | 1000 iters | 50 | 0.04 | depth=6, l2_leaf_reg=3, bootstrap=Bernoulli | Prokhorenkova et al. 2018 NeurIPS (arXiv:1706.09516) |
| mlp | 30 epochs | 8 | 1e-3 | hidden=[128,64], dropout=0.3, AdamW wd=1e-4 | Gu, Kelly & Xiu 2020 RFS (arXiv:1802.09003) |

## Citation Rigor (MANDATORY format for `citations` field)

Every citation string MUST contain, for every paper referenced:

1. All authors' surnames (not just first-author et al. unless > 6 authors)
2. Year of publication
3. Venue — journal name, conference abbreviation (NeurIPS, ICML, ICLR, KDD, etc.), or `arXiv` if preprint-only
4. Full paper title in single quotes
5. arXiv ID in the form `(arXiv:XXXX.YYYYY)` if available
6. One-sentence relevance note

Multiple papers separated by semicolons + newline. The runner's `validate_pre_run_entry` will refuse to launch otherwise.

## Reasoning Blob Completeness

Each of the 7 fields in `reasoning_annotations.json` has a minimum content spec:

| Field | Min words | Required |
|-------|-----------|----------|
| diagnosis | 60 | reference to ≥1 prior experiment OR per-fold metric |
| citations | 40 (single) / 80 (multi) | full ref + arXiv + relevance note |
| hypothesis | 50 | "mechanism" / "because" / "per [paper]" |
| prediction | 25 | numeric range (e.g. "0.55-0.65") |
| verdict | 30 | KEEP / DISCARD / NEAR-MISS + composite |
| learning | 40 | "axis closed" / "axis open" / "next try" |

The runner enforces these with `validate_pre_run_entry` before every launch.

## Dashboard Files Update Mandate

Every experiment updates these files:

| File | Written by | When |
|------|-----------|------|
| `autoresearch_results/experiment_log.jsonl` | runner | every run, appended |
| `autoresearch_results/best_config.json` | runner | only when new champion |
| `autoresearch_results/reasoning_annotations.json` | Claude (pre-run) + runner (post-run fallback) + Claude (post-run rewrite) | every run, two-phase |
| `autoresearch_results/trade_logs/exp<N>_predictions.csv` | runner | every run |
| `autoresearch_results/dashboard.html` | one-time copy | static — reads JSONL live |

## Dashboard Hosting

The dashboard runs in two modes:

- **Local:** `python -m http.server 8765 --directory autoresearch_results` — open `http://localhost:8765/dashboard.html`.
- **GitHub Pages:** the repo's `docs/fraud_ecommerce_dashboard/` mirror is auto-published when pushed to a GitHub repo with Pages enabled (Settings → Pages → main / docs). To sync: copy `autoresearch_results/{dashboard.html, experiment_log.jsonl, reasoning_annotations.json, best_config.json}` to `docs/fraud_ecommerce_dashboard/` before each `git push`.

## Project Status (as of 2026-04-24)

- 8 experiments complete (Exp 2-9 — Exp 1 quarantined as methodologically invalid stratified-CV).
- Champion: Exp 6 (XGBoost + velocity features), test AUC 0.5414, composite 0.4903 (DISCARD by floor — no formal champion yet).
- Best vs FDB baselines: beats AutoGluon (0.522), H2O (0.518), Auto-sklearn (0.515); 0.10 below AFD-TFI (0.636).
- Next axis: rolling time-windowed velocity counts (transactions per device/IP in last 1d/7d/30d) — the documented AFD-TFI feature class we have not yet implemented.

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Stratified CV on time-ordered data | inflated AUC by ~0.27 (Exp 1: 0.7738 fake vs Exp 2: 0.5098 honest) | Always use chronological holdout for any dataset with a timestamp |
| Velocity features computed on first 80% | val_AUC ≫ test_AUC (Exp 5: 0.9988 vs 0.5297) | Compute on first 70% (n - n_val - n_test) |
| Treating XGBoost result as the only backbone exploration | violates the multi-backbone mandate | Per CLAUDE.md, every backbone tier gets full exploration |
| Single-experiment "SOTA" claims | misleading without out-of-time validation | Require chronological-holdout result before declaring any champion |
| Blind grid sweep instead of 7-step research process | wastes compute, no monotonic progress | Every experiment MUST follow Diagnose → Cite → Hypothesize → Predict → Run ONE → Analyze → Document |
| Trusting the framework wires every config field | Exp 43 dropped scale_pos_weight silently, gave bug-identical-to-baseline result | Verify wiring by either reading backbone source OR running an extreme-value A/B (e.g. wd=0 vs wd=10) |
| Setting too-strict composite floor early | hides progress (all-DISCARD leaderboards) | Set the floor based on the realistic ceiling, not the wishful one (we used 0.55 initially when 0.50 was the right floor — must beat random) |

## STRICT 7-Step Protocol (per CLAUDE.md Research-Driven Experiment Selection)

**Every experiment after the first MUST follow this exact sequence. No grid sweeps. No "let me try X". No batch HP exploration.**

1. **Diagnose the current champion's failure mode.** For chronological-holdout, this means per-prediction analysis: confidence calibration, feature distribution differences between TP/FN/FP/TN, threshold sensitivity, per-segment recall.
2. **Search the literature.** Identify a paper that directly addresses the diagnosed failure mode. Full citation including arXiv ID.
3. **Form a hypothesis with a numeric prediction.** "I hypothesize that change X will move metric Y from current_value to predicted_range, because mechanism Z." Cannot be vague.
4. **Run ONE experiment.** Single config change. Reasoning entry with all 5 fields validated before launch.
5. **Analyze against prediction.** Did the result match? If not, what does that update in the mental model? Surface findings honestly even when they refute the hypothesis.
6. **Document everything.** Post-run verdict + learning, both per-fold-specific. Update the dashboard.
7. **Decide next experiment based on the analysis.** Not from a pre-planned grid.

If 3 consecutive experiments are DISCARD, STOP and rethink — the diagnosis is wrong, not the hyperparameter values.

## Lessons Learned in This Project (Exps 1-44)

1. **Exp 1 was discarded as methodologically invalid.** Stratified 3-fold CV on a time-ordered fraud dataset produced AUC=0.7738. The same XGBoost config under chronological holdout produced AUC=0.5098. The 0.27 gap is purely the test protocol — concept drift in `time_since_signup` reverses the train/test direction.
2. **Exp 5 surfaced a leakage bug via the val/test gap.** val_AUC 0.9988 vs test_AUC 0.5297 was the alarm. Velocity features were computed on first 80% (= train+val) but the runner uses first 70% as train. Fixed by aligning n_train with the runner's slicing.
3. **Exp 6 is the global champion at test_auc=0.5414.** Beats FDB AutoGluon (0.522), H2O (0.518), Auto-sklearn (0.515). 0.10 below FDB AFD-TFI (0.636).
4. **Multi-backbone exploration confirmed plateau.** Across 41 experiments on 4 backbones (xgboost 20 / lightgbm 10 / catboost 10 / mlp 1), all GBM variants plateau in 0.52-0.54 range; MLP collapsed to 0.488. Multi-seed std on champion = 0.006 (very low).
5. **Exp 43 surfaced a framework bug.** The xgboost backbone silently dropped `scale_pos_weight`; experiment produced bit-identical results to Exp 6. Diagnosis caught it; gbm.py patched. Exp 44 with patch applied gave only +0.0022 test AUC — the Pozzolo 2015 rare-events-correction hypothesis is directionally correct but quantitatively too small to move the metric on this dataset.
6. **Composite floor is set to 0.50 (must beat random)**, NOT 0.55. The dataset's true achievable ceiling for our public feature set is ~0.55, with FDB AFD-TFI's 0.636 unreachable without their proprietary entity-velocity features. Setting the floor at 0.55 made every honest experiment a DISCARD, which is misleading. Floor=0.50 = "the model has learned non-trivial signal".
