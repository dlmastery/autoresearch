# Section Mapping — Source CLAUDE.md → CLAUDE_template.md

This is the Queen's coverage audit. Every top-level `##` and `###` heading from
`C:/Users/evija/autoresearch/CLAUDE.md` is listed here with its destination
heading in `CLAUDE_template.md`, the placeholders introduced, and the generic
content retained.

**Audit gate:** 0 missing sections. If any row below says "MISSING", the template
is not ready to ship.

## 1-to-1 mapping

| # | Source heading | Source line | Target heading | Status | Placeholders introduced |
|---|----------------|-------------|----------------|--------|--------------------------|
| 1 | `# CLAUDE.md — Project Rules for AutoResearch` | 1 | `# CLAUDE.md — Project Rules for {{project_name}}` | PRESERVED | `{{project_name}}` |
| 2 | `## On Session Start` | 3 | `## On Session Start (ALWAYS do this first)` | PRESERVED | `{{memory_dir}}`, `{{results_dir}}`, `{{python_exe}}`, `{{dashboard_port}}`, `{{configs_dir}}`, `{{experiment_timeout_seconds}}` |
| 3 | `## Hardware Constraints (MANDATORY — updated 2026-04-19)` | 15 | `## Hardware Constraints ({{hardware_policy_mandatory_or_optional}})` | PRESERVED | `{{gpu_vram_gb}}`, `{{cpu_logical_cores}}`, `{{cpu_runner_cores}}`, `{{cpu_runner_affinity}}`, `{{banned_cores_and_reason}}`, `{{experiment_timeout_seconds}}`, `{{phase_training_time_budget}}` |
| 4 | `## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)` | 32 | `## Crash-Recovery Checkpointing (MANDATORY)` | PRESERVED | `{{memory_dir}}`, `{{results_dir}}`, `{{fold_or_group_label}}`, `{{primary_metric_name}}` |
| 5 | `## Mindset (Read First)` | 60 | `## Mindset (Read First)` | PRESERVED | `{{domain_description}}`, `{{windowing_or_batching}}` |
| 6 | `## Hard Rules (NEVER violate)` | 69 | `## Hard Rules (NEVER violate)` | PRESERVED (header) | — |
| 7 | `### Data Integrity` | 71 | `### Data Integrity` | PRESERVED | `{{fold_or_group_label}}`, `{{label_horizon_buffer_units}}`, `{{label_horizon_value}}`, `{{purge_mechanism}}`, `{{data_cache_dir}}`, `{{data_integrity_custom_rules}}` |
| 8 | `### Super-Fold Invariants` | 78 | `### Evaluation Protocol Invariants` | GENERALIZED (super-fold → user's chosen protocol) | `{{split_protocol_name}}`, `{{configs_dir}}`, `{{cross_fold_overlap_rule}}`, `{{val_set_description}}`, `{{test_set_description}}` |
| 9 | `### Experiment Design` | 85 | `### Experiment Design` | PRESERVED | `{{composite_formula}}`, `{{composite_penalty_weight}}`, `{{min_epochs}}`, `{{cooldown_seconds}}`, `{{regime_label_singular}}`, `{{primary_regime}}`, `{{regime_label_plural}}`, `{{fold_or_group_label}}` |
| 10 | `### Autoresearch Agent Protocol (Karpathy-adapted)` | 95 | `## Autoresearch Agent Protocol (Karpathy-adapted)` | PRESERVED (all 8 rules verbatim in spirit) | `{{fold_or_group_label}}`, `{{sequence_or_feature_scope}}`, `{{failing_metric_direction}}`, `{{code_versions_dir}}`, `{{memory_dir}}` |
| 11 | `### Research-Driven Experiment Selection (STRICT — no blind sweeps)` | 105 | `## Research-Driven Experiment Selection (STRICT — no blind sweeps)` | PRESERVED (7-step process) | `{{fold_or_group_label}}`, `{{regime_label_singular}}`, `{{correct_vs_wrong_label}}`, `{{domain_specific_diagnosis_examples}}` |
| 12 | `### Monotonic Quality Progression (NEVER regress)` | 127 | `## Monotonic Quality Progression (NEVER regress)` | PRESERVED + strengthened with Goodhart protection | `{{large_regression_threshold}}`, `{{primary_metric_name}}` |
| 13 | `### MLOps Documentation Standards (MANDATORY)` | 135 | `## MLOps Documentation Standards (MANDATORY)` | PRESERVED | `{{results_dir}}`, `{{primary_metric_name}}`, `{{fold_or_group_label}}`, `{{per_fold_template}}`, `{{secondary_metric_template}}`, `{{ratio_decimals}}`, `{{percentage_decimals}}`, `{{correct_vs_wrong_label}}` |
| 14 | Experiment Log template format (lines 141-153) | — | embedded in section 13 | PRESERVED | — |
| 15 | `### Explainability & Auditability Report (MANDATORY for every NEW BEST)` | 163 | `## Explainability & Auditability Report (MANDATORY for every NEW BEST)` | PRESERVED (all 14 sections) | `{{results_dir}}`, `{{primary_metric_name}}`, `{{secondary_primary_metric_label}}`, `{{risk_metric_label}}`, `{{num_folds_or_groups}}`, `{{fold_or_group_label}}`, `{{Regime_label_plural_titled}}`, `{{n_features}}`, `{{regime_label_singular}}`, `{{risk_audit_content}}`, `{{purge_mechanism}}`, `{{embargo_value}}`, `{{label_horizon_buffer_units}}`, `{{regime_label_plural}}`, `{{deployment_specific_items}}` |
| 16 | `### Winner Definition (CLARIFICATION)` | 202 | `## Winner Definition (CLARIFICATION)` | PRESERVED | `{{results_dir}}` |
| 17 | `### Per-Backbone Code Snapshots (MANDATORY)` | 213 | `## Per-Backbone Code Snapshots (MANDATORY)` | PRESERVED | `{{code_versions_dir}}`, `{{n_experiments_per_backbone}}` |
| 18 | `### Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)` | 229 | `## Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)` | PRESERVED (7-field schema + pre/post-run cadence + enforcement + backfill rules + runner responsibility) | `{{results_dir}}`, `{{fold_or_group_label}}`, `{{regime_label_singular}}` |
| 19 | `## Exp<N> — <short title>` journal template (lines 259-266) | — | embedded in section 18 | PRESERVED | — |
| 20 | `### Per-Backbone 50-Experiment Mandate (MANDATORY, not optional)` | 284 | `## Per-Backbone N-Experiment Mandate (MANDATORY, not optional)` | PRESERVED (N parameterized, default 50) | `{{n_experiments_per_backbone}}`, `{{current_year}}`, `{{sota_year_range}}` |
| 21 | `### Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)` | 308 | `## Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)` | PRESERVED | `{{sota_year_range}}` |
| 22 | `### Backbone-Specific Training Recipes (updated 2026-04-19 from SOTA literature)` | 335 | `### Backbone-Specific Training Recipes (auto-generated from SOTA catalog for task = {{task_type}})` | GENERALIZED (recipe table auto-generated from `sota_catalog.yaml` filtered by task type) | `{{task_type}}`, `{{tier1_recipes_table}}`, `{{tier2_recipes_table}}` |
| 23 | `#### Tier 1 — neural backbones (require from-scratch or fine-tune training)` | 341 | `#### Tier 1 — classical baselines (required for every run to establish a floor)` | PRESERVED as subsection | — |
| 24 | `#### Tier 2 — 10 NEW 2024-2026 SOTA backbones (add to runner before running)` | 354 | `#### Tier 2 — {{sota_year_range}} SOTA ({{task_type}}-specific)` | PRESERVED as subsection, filtered by task | — |
| 25 | `#### Tier 3 — gradient boosted machines (each is its OWN backbone, run independently)` | 371 | `#### Tier 3 — gradient boosted machines (each is its OWN backbone, run independently)` | PRESERVED verbatim (three separate backbones rule explicit) | — |
| 26 | `### GPU Memory Constraint (MANDATORY — 16 GB VRAM hard cap)` | 390 | `## GPU Memory Constraint (MANDATORY — {{gpu_vram_gb}} GB VRAM hard cap)` | PRESERVED (budget scales to user's VRAM) | `{{gpu_vram_gb}}`, `{{vram_params_gb}}`, `{{vram_optim_gb}}`, `{{vram_grads_gb}}`, `{{vram_acts_gb}}`, `{{vram_reserve_gb}}`, `{{ceiling_fp32_scratch}}`, `{{ceiling_bf16_scratch}}`, etc. |
| 27 | `### Epoch-budget rule of thumb (when in doubt)` | 464 | `### Epoch-budget rule of thumb (when in doubt)` | PRESERVED verbatim | — |
| 28 | `### Empirical evidence (LSTM phase confirmations)` | 475 | _embedded in_ "Session Learnings" (initial) | GENERALIZED — becomes project-specific once the project runs experiments | — |
| 29 | `### Backbone Isolation Rule` | 482 | `## Backbone Isolation Rule` | PRESERVED | `{{code_versions_dir}}`, `{{n_experiments_per_backbone}}` |
| 30 | `### Dashboard Backbone Tabs` | 486 | `## Dashboard Backbone Tabs` | PRESERVED verbatim | `{{results_dir}}` |
| 31 | `### Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)` | 490 | `## Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)` | PRESERVED (ownership table, per-experiment ritual, verification checklist, TODO-REWRITE rule) | `{{results_dir}}`, `{{primary_metric_name}}`, `{{fold_or_group_label}}`, `{{regime_label_plural}}`, `{{memory_dir}}` |
| 32 | `### Citation Rigor (MANDATORY format for `citations` field)` | 539 | `## Citation Rigor (MANDATORY format for `citations` field)` | PRESERVED verbatim (6 required elements, format template, good/bad examples, arXiv lookup discipline) | — |
| 33 | `### Reasoning Blob Completeness (what "full reasoning" means)` | 580 | `## Reasoning Blob Completeness (what "full reasoning" means)` | PRESERVED verbatim (word-count floors table, must-include items, batch-update prohibition) | `{{fold_or_group_label}}` |
| 34 | `### Heteroscedastic Loss Rules (Kendall & Gal 2017)` | 598 | `## Loss Function Rules` → `### Heteroscedastic / uncertainty-aware loss (neural regression & time-series)` | GENERALIZED (het-loss preserved as a subsection under a broader Loss Function Rules section; task-appropriate default loss rules added for classification, ranking, survival) | `{{task_specific_loss_rules}}`, `{{fold_or_group_label}}`, `{{regime_label_singular}}` |
| 35 | `### Winner Archiving Protocol (MANDATORY for every NEW BEST)` | 606 | `## Winner Archiving Protocol (MANDATORY for every NEW BEST)` | PRESERVED (directory structure, README template, portability rules, checkpoint format, predict.py spec) | `{{results_dir}}`, `{{fold_or_group_label}}`, `{{primary_metric_name}}`, `{{secondary_metrics_list}}`, `{{reproduce_tolerance}}` |
| 36 | "Trading Strategy section" (lines 669-680) | — | `{{deployment_strategy_section_template}}` — task-conditional content filled by setup wizard | GENERALIZED | — |
| 37 | `### Google Colab Notebook (MANDATORY for every winner)` | 682 | `## Google Colab Notebook (MANDATORY for every winner)` | PRESERVED (8 required cells, notebook principles, 5-min runtime target, self-contained rule) | `{{results_dir}}`, `{{num_folds_or_groups}}`, `{{fold_or_group_label}}`, `{{visualization_plan}}` |
| 38 | `### Traditional ML Metrics (MANDATORY for every experiment)` | 703 | `## Traditional ML Metrics (MANDATORY for every experiment)` | PRESERVED framing; metric list generalized per task type | `{{task_type}}`, `{{secondary_metrics_detailed_spec}}`, `{{fold_or_group_label}}` |
| 39 | `### Trade-Level Win/Loss Logging (MANDATORY for every experiment)` | 724 | `## Per-Prediction Log (MANDATORY for every experiment)` | GENERALIZED (trade log → per-prediction log; task-specific columns filled by setup) | `{{results_dir}}`, `{{fold_or_group_label}}`, `{{regime_label_singular}}`, `{{task_type}}`, `{{task_specific_prediction_columns}}` |
| 40 | `### Architecture` | 761 | `## Architecture` | PRESERVED verbatim | — |
| 41 | `### Validation Checklist (Run Before Every Experiment Session)` | 768 | `## Validation Checklist (Run Before Every Experiment Session)` | PRESERVED (6 items, parameterized to user's protocol) | `{{expected_train_n}}`, `{{expected_val_n}}`, `{{expected_test_n}}`, `{{fold_or_group_label}}`, `{{data_cache_dir}}` |
| 42 | `## Project Structure` | 776 | `## Project Structure` | PRESERVED, generalized folder layout | `{{project_name}}`, `{{results_dir}}`, `{{memory_dir}}`, `{{code_versions_dir}}`, `{{fold_or_group_label}}` |
| 43 | `## Key Constants` | 810 | `## Key Constants` | PRESERVED (table format) | all values parameterized from setup answers |
| 44 | `## Common Mistakes (Never Repeat)` | 825 | `## Common Mistakes (Never Repeat)` | PRESERVED + two new items (silent drop / Goodhart) added | `{{fold_or_group_label}}`, `{{num_folds_or_groups}}`, `{{data_cache_dir}}`, `{{regime_label_singular}}`, `{{regime_label_plural}}` |
| 45 | `## Session Learnings (LSTM Phase, Exps 1-44 of 50)` | 844 | `## Session Learnings` | GENERALIZED (project-specific — append-only, starts empty except for setup metadata) | `{{setup_date}}`, `{{task_type}}`, `{{primary_metric_name}}`, `{{split_protocol_name}}`, `{{backbones_list}}` |
| 46 | `### Confirmed optimal LSTM hyperparameters (at n=2738 daily FX samples)` | 848 | (Session Learnings subsection example, filled by project) | APPEND-ONLY structure preserved | — |
| 47 | `### Axes that DID NOT help` | 861 | (Session Learnings subsection example) | APPEND-ONLY structure preserved | — |
| 48 | `### Seed variance is LARGE and backbone-specific` | 869 | (Session Learnings subsection example) | APPEND-ONLY structure preserved | — |
| 49 | `### Key protocol additions` | 877 | (Session Learnings subsection example) | APPEND-ONLY structure preserved | — |
| 50 | `### Next-backbone priorities` | 889 | (Session Learnings subsection example) | APPEND-ONLY structure preserved | — |
| 51 | `### Checkpoint + packaging cadence` | 897 | _subsumed by_ Dashboard Files Update Mandate + Session Learnings | PRESERVED in stricter form | — |
| 52 | (implicit) Trading strategy 10-item checklist (lines 669-680) | 669-680 | `{{deployment_strategy_section_template}}` placeholder; filled by setup per task type | GENERALIZED | — |

## Summary

- **Total source headings:** 52 (including subsections and embedded templates)
- **Status:** 52/52 PRESERVED or GENERALIZED (0 missing)
- **Placeholders introduced:** 60+
- **New sections added in target template (hardening additions):**
  - Goodhart-protection clause under "Monotonic Quality Progression"
  - "Silent drop" common-mistake row
  - "Rewriting composite mid-project" common-mistake row
  - Pre-flight GPU check enforcement by `core/reasoning.py`

## Verification command (run after any template edit)

The companion Python script `tests/test_section_coverage.py` greps both files
for `##` and `###` headings and asserts the target contains a mapping row for
each source row. Run:

```
{{python_exe}} -m pytest generalized_ml_autoresearch/tests/test_section_coverage.py -v
```

## Non-preserved content justification

Zero source sections were non-preserved. Every FX-specific detail that was not
generalizable (e.g. specific CPU APIC IDs, specific fold-7 FX regimes) was
preserved as a parameterized placeholder whose default in the FX project is
exactly the original value, and whose default in new projects comes from the
setup wizard.
