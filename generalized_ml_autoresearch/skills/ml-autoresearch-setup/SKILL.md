---
name: ml-autoresearch-setup
description: >
  Interactive 12-step setup wizard that generalizes the FX AutoResearch CLAUDE.md
  methodology for ANY supervised ML project. Use when the user says "/ml-autoresearch-setup",
  "set up autoresearch for my ML project", "generate a CLAUDE.md for my <task-type> project",
  or wants to bootstrap a new autoresearch loop against a custom dataset. The skill produces
  a filled-in CLAUDE.md, folder skeleton, runner config, and starter research journal.
version: 0.1.0
arguments:
  - name: project_root
    required: false
    description: Absolute path to the new project's root directory (defaults to user's cwd/autoresearch-project).
triggers:
  - "/ml-autoresearch-setup"
  - "set up autoresearch"
  - "bootstrap autoresearch"
  - "generate CLAUDE.md for ML project"
---

# ml-autoresearch-setup — 12-step wizard

You are the setup-wizard skill. You walk the user through the 12 steps below using
`AskUserQuestion` where interactive, and `Read`/`Write` for file operations. The
goal is to produce a complete, runnable autoresearch project bootstrap that
preserves EVERY rule from the source CLAUDE.md.

## The 12 steps

### Step 1 — Problem framing

Ask the user:

- What is the ML task? (regression / binary-classification / multiclass-classification /
  multilabel / time-series-forecasting / ranking / survival)
- Is the order of samples meaningful? (i.e. is temporal ordering preserved and is
  look-ahead a leakage risk?) → if yes, time-series-style splits; if no, random splits.
- Is there a natural grouping variable (e.g. patient ID, store ID, FX pair) that
  must not cross splits? → if yes, `group_kfold` is mandatory.
- What domain phrase describes this project? (e.g. "churn prediction in SaaS",
  "lidar object classification", "inventory demand forecasting") — fills `{{domain_description}}`.

Persist answers to `<project_root>/autoresearch_setup_answers.json`.

### Step 2 — Dataset pointer

- Path to the dataset file or `sklearn.datasets` loader name
- Format: csv / parquet / numpy / huggingface-dataset / custom callable
- Target column name(s)
- Feature column names (or "all others")
- Grouping / time column name (if applicable)
- Natural primary key (row identifier)

### Step 3 — Data integrity constraints

- Any leakage risks specific to the dataset (derived features using future info,
  target-leakage columns, …)? List them so they become Hard Rules.
- For time-series: label horizon (if predicting `y_{t+h}`, set `label_horizon_buffer_units = h + 2`).
- For grouped: purge distance (groups that must be separated by N time units).
- Class imbalance handling: if binary-classification and positive-rate < 10%,
  include focal-loss / class-weight guidance in CLAUDE.md.

### Step 4 — Evaluation protocol

Pick ONE (or compose):
- `holdout` — single train/val/test split
- `kfold` — standard k-fold CV (i.i.d. only)
- `stratified_kfold` — classification with class balance
- `group_kfold` — grouping required
- `time_series_split` — expanding window
- `walk_forward` — fixed train/val/test windows with purge + embargo
- `super_fold` — FX-style union of multiple regime windows (pre-define them)

Ask for the parameters (n_splits, val_fraction, purge, embargo, label_horizon_buffer).

### Step 5 — Primary metric

Common choices pre-populated by task type:

- regression → RMSE (default), MAE, MAPE, R², IC
- binary classification → F1 (default), AUC-ROC, MCC, F2, Recall@k
- multiclass → macro-F1, accuracy, MCC
- time-series forecasting → RMSE, SMAPE, hit rate, Sharpe (finance)
- ranking → nDCG@k, MAP@k, MRR
- survival → concordance index
- custom → Python dotted path to a function that accepts `(y_true, y_pred, **kwargs) -> float`

Note whether higher-is-better or lower-is-better.

### Step 6 — Composite metric formula

Default (preserved from CLAUDE.md): `min(val_primary, test_primary) - penalty * n_below_threshold_folds`.

Offer alternatives:
- `mean(val_primary, test_primary) - penalty * n_below_threshold_folds` (smoother)
- `test_primary - penalty * max(0, test_primary - val_primary)` (overfit-penalizing)
- custom formula string (sandbox-evaluated) OR custom callable path

Record `penalty_weight` and `below_threshold` (for fold-counting).

### Step 7 — Hardware constraints

- GPU VRAM (GB)
- CPU logical cores
- Cores reserved for the runner (default 4)
- Any banned cores (if hardware has known-bad cores, e.g. E-cores with WHEA errors)
- Time budget per experiment (seconds)
- Max training time per phase (wall-clock budget per backbone's 50-experiment cycle)

### Step 8 — Backbone list

Auto-generate from `templates/sota_catalog.yaml` filtered by task_type:

- **Tier 1** (classical baselines): linear/ridge/MLP, or softmax/MLP for classification
- **Tier 2** (2024-2026 SOTA): FT-Transformer/TabNet for tabular; PatchTST/iTransformer/xLSTM/Mamba/TimeMixer for time-series; foundation stubs for TimesFM/Chronos/MOMENT if time-series
- **Tier 3** (GBM trio — ALWAYS three separate backbones): xgboost, lightgbm, catboost

User can add/remove. The generated CLAUDE.md lists every backbone with its SOTA recipe.

### Step 9 — Starting hyperparameter recipes per backbone

The sota_catalog.yaml has full citations (author/year/venue/title/arXiv/relevance).
For each selected backbone, the wizard renders a row in the Tier-1/2/3 table in
CLAUDE.md. User confirms or edits.

Rule: every recipe MUST cite its originating paper per Citation Rigor.

### Step 10 — Artifacts plan

All MANDATORY artifacts (preserved from CLAUDE.md):
- Dashboard (dashboard/dashboard.html)
- reasoning_annotations.json (runner + Claude two-phase)
- research_journal.md
- experiment_summary.md
- memory/project_autoresearch_checkpoint.md
- trade_logs/exp<N>_predictions.csv (generalized per-prediction log)
- winners/<backbone>_expN_<desc>/ (archive template)
- audit_report.md (14 sections)
- Colab notebook per winner

Ask whether any are opt-out (none by default).

### Step 11 — Experiment protocol customization

- N experiments per backbone (default 50)
- Multi-seed policy (default: 3-seed median before declaring champion)
- Monotonic improvement enforcement (default ON)
- Winner archive triggers (default: any new global best; optional: only if delta > X)

### Step 12 — Review + write

Produce and write to disk:

1. **`<project_root>/CLAUDE.md`** — filled from `templates/CLAUDE_template.md` with every `{{...}}` placeholder resolved.
2. **`<project_root>/configs/project.yaml`** — runner config.
3. **Folder skeleton:**
   - `configs/`, `data/`, `memory/`, `autoresearch_results/`, `autoresearch_results/trade_logs/`, `autoresearch_results/winners/`, `code_versions/v1_original/`
4. **`<project_root>/autoresearch_results/dashboard.html`** (copy of `dashboard/dashboard.html`)
5. **`<project_root>/autoresearch_results/research_journal.md`** (seed with "## Project initialized {{setup_date}}")
6. **`<project_root>/memory/project_autoresearch_checkpoint.md`** — initial checkpoint with "session start" instructions + empty history.
7. **`<project_root>/autoresearch_setup_answers.json`** — full wizard answers for audit.

Show the user a diff preview before writing. After writing, tell them the exact
first-experiment command to run (using `generalized_ml_autoresearch.core.runner`
with their project.yaml config).

## Template filling details

- Read `generalized_ml_autoresearch/templates/CLAUDE_template.md`.
- Resolve every `{{placeholder}}` using the wizard answers and auto-derived values.
- For auto-derived: `{{sota_year_range}}` = "2024-2026"; `{{current_year}}` = current year;
  `{{setup_date}}` = today's ISO date; `{{python_exe}}` = the user's reported Python path.
- For the Tier-1/2/3 tables: filter `sota_catalog.yaml` by `task_type` and render rows.
- VRAM ceiling formulas (for the GPU Memory Constraint table):
  - `vram_params_gb = round(user_vram / 5.3)` → 3 GB at 16 GB; scales linearly
  - similar proportional scaling for other columns
  - Parameter ceilings: derived from the budget breakdown (roughly 30 M params per GB of VRAM for from-scratch FP32 training)

## Non-negotiable gates

Before writing CLAUDE.md, verify:

- [ ] Every `{{placeholder}}` is resolved (no `{{}}` left in output)
- [ ] The filled Tier-1/2/3 tables each cite at least one arXiv ID
- [ ] The composite formula's fingerprint is stored so Goodhart-rewrites are detectable
- [ ] The generated `reasoning_annotations.json` contains NO entries (clean start)
- [ ] The checkpoint file mentions the full 7-step process
- [ ] The checkpoint's "Next experiment" block says: "Author Exp1 pre-run reasoning in reasoning_annotations.json BEFORE launching"

## Failure modes to warn the user about

- "I don't have a dataset ready yet" → point them at sklearn.datasets examples; do NOT auto-generate synthetic data as a crutch.
- "I want to change the composite metric later" → warn: changing mid-project is a Goodhart risk; require a RULE_CHANGE entry in the checkpoint.
- "Can I skip the reasoning annotation floor?" → No. Runner refuses to launch without it. This is a hard rule.
- "Can I merge xgboost/lightgbm/catboost?" → No. Three separate backbones (see CLAUDE.md Tier-3).
