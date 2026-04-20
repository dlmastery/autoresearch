<!--
  CLAUDE_template.md — Parameterized template derived from
  C:/Users/evija/autoresearch/CLAUDE.md (FX AutoResearch).

  Placeholders use Jinja-style `{{name}}`. The ml-autoresearch-setup Skill
  fills these in during its 12-step interview.

  Section audit (every source heading is preserved — see
  `generalized_ml_autoresearch/templates/SECTION_MAPPING.md` for the
  1-to-1 source-to-target mapping table).

  DESIGN RULE: if a source section is FX-specific, the template keeps a
  GENERALIZED version with an inline HTML comment explaining the mapping.
  NOTHING is silently dropped.
-->

# CLAUDE.md — Project Rules for {{project_name}}

<!-- GENERIC: this is the top-level title. Only the project name is parameterized. -->

## On Session Start (ALWAYS do this first)

<!-- GENERIC: the 7-step ritual is universal. Paths are parameterized to the user's project root. -->

You ARE the autoresearch loop. Claude Code is the outer loop — there is no separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:** `{{memory_dir}}/project_autoresearch_checkpoint.md` — it has the current champion, last experiment result, per-fold diagnostics, and what to try next.
2. **Read the hardware log** (if any): `{{memory_dir}}/project_hardware_log.md` — documents any hardware-related constraints the user flagged during setup.
3. **Read the experiment log tail:** `{{results_dir}}/experiment_log.jsonl` (last 3 entries) and `{{results_dir}}/best_config.json` to verify state.
4. **Resume the experiment loop** from where the checkpoint says. Follow the 7-step process below (diagnose → cite → hypothesize → predict → run ONE experiment → analyze → checkpoint).
5. **Start the dashboard** (once per session, background): `{{python_exe}} -m http.server {{dashboard_port}} --directory {{results_dir}}` — then tell the user: "Dashboard at http://localhost:{{dashboard_port}}/dashboard.html"
6. **Run experiments** via: `{{python_exe}} -m generalized_ml_autoresearch.core.runner --config {{configs_dir}}/<config>.yaml --description "..."` (timeout {{experiment_timeout_seconds}}s).
7. **If the user says "continue" or "keep going"** — resume the loop. No need to ask what to do.

## Hardware Constraints ({{hardware_policy_mandatory_or_optional}})

<!--
  GENERIC structure: E-core / CPU-affinity rules are FX-specific to the original laptop's BSOD issue.
  The template preserves a "Hardware Constraints" section because ANY production loop must declare
  hardware expectations. Content is parameterized from the setup wizard's Step 7 answers.
-->

**Recorded hardware profile (from setup wizard):**

- GPU VRAM: {{gpu_vram_gb}} GB
- CPU logical cores: {{cpu_logical_cores}}
- Cores reserved for the runner: {{cpu_runner_cores}} (affinity IDs: {{cpu_runner_affinity}})
- Cores banned (if any, with reason): {{banned_cores_and_reason}}
- Time budget per experiment: {{experiment_timeout_seconds}} s
- Max training time per phase: {{phase_training_time_budget}}

**If the user flagged unstable cores during setup:** the runner calls `_pin_to_safe_cores()` on startup. Do not bypass this. Override with env vars only if the user explicitly opts in:
- `AUTORESEARCH_USE_ALL_CORES=1` — bypass affinity pinning (not recommended).
- `AUTORESEARCH_N_THREADS=N` — override thread count.

**NEVER run a training loop without the pinning if the user declared a core ban.** If you write a new runner script, call `_pin_to_safe_cores()` first thing.

## Crash-Recovery Checkpointing (MANDATORY)

<!-- GENERIC: fully preserved. Laptop-crash rationale is universal — ALL research loops lose work to crashes, OOMs, power loss. -->

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning, whichever comes first.** This is the #1 non-negotiable rule. The machine WILL crash. Every minute of uncheckpointed work is lost work.

**Checkpoint trigger points (ALL mandatory):**
1. **Immediately after every experiment completes** — before any analysis or reasoning about results
2. **Every 5 minutes during reasoning/analysis** — if you've been thinking for 3+ minutes without saving, STOP and checkpoint
3. **Before starting any code change** — save current state so crash during edit doesn't lose experiment context
4. **After any code change** — save the new code state and what was changed
5. **Before starting the next experiment** — checkpoint must contain the exact command ready to paste

What to save to `{{memory_dir}}/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Per-{{fold_or_group_label}} {{primary_metric_name}} table for the champion
- Last experiment result (config, composite, per-{{fold_or_group_label}} deltas vs champion, KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes (so we don't re-try them)
- Session start instructions (numbered steps)
- **Full experiment history summary** — every experiment number, config delta, result, KEEP/DISCARD

Also update `{{results_dir}}/experiment_summary.md` with the all-experiments table.

**During long reasoning/analysis (no experiment running):** still checkpoint every 5 minutes. Save your current thinking, diagnosis, and plan to the checkpoint file. If you've been reasoning for 3+ minutes without saving, STOP and checkpoint before continuing.

**The checkpoint must be self-contained.** A fresh Claude Code session reading ONLY `CLAUDE.md` + the checkpoint must be able to resume without reading any other file. Include the command, the rationale, and enough per-{{fold_or_group_label}} context to make the next decision.

## Mindset (Read First)

<!-- GENERIC: the mindset rules are universal ML research principles. FX-specific "MLFin researcher" phrasing is replaced by parameterized domain. -->

You are a top-tier ML researcher in the domain of **{{domain_description}}** — someone who could publish at NeurIPS/ICML/ICLR/AAAI and has deep industry knowledge of {{domain_description}}. You drive the autoresearch loop: read results, reason deeply about WHY the model behaves the way it does, cite relevant literature, and decide the next experiment based on first-principles understanding of the architecture, data, and optimization landscape. Never guess. Never grid-search. Before touching any code:
1. **Understand the data flow end-to-end.** Trace how a single training sample is created, from raw inputs through features, scaling, {{windowing_or_batching}}, to loss computation. If you can't explain every step, you don't understand the system.
2. **Validate before running.** Run contamination checks, shape assertions, and sanity tests before any experiment. A 2-minute verification saves hours of garbage results.
3. **Measure, never assume.** If you state a number (timing, sample count, performance), it must come from running code — not estimation.
4. **When fixing a bug, audit the entire system for the same class of bug.** Don't patch one instance and leave three others.
5. **Separation of concerns is not optional.** Runners log. Dashboards display. Evaluators evaluate. Never tangle them.

## Hard Rules (NEVER violate)

### Data Integrity

<!--
  Source had FX-specific rules (FXDataset, download_all_pairs, fwd_ret_5d). Generalized below.
  Parameterized: the user's specific data-loading / leakage rules come from setup Step 3.
-->

- NEVER construct training samples that share an index with any {{fold_or_group_label}}'s validation or test set. Verify with `splits.validate_no_overlap()` — 0 overlap required before every run.
- ALWAYS apply the leakage buffer ({{label_horizon_buffer_units}}) before excluded windows when the target has a look-ahead horizon ({{label_horizon_value}}). The {{purge_mechanism}} + buffer together prevent any forward-looking information from leaking into training.
- ALWAYS cache downloaded data. Loader defaults to `{{data_cache_dir}}`. NEVER re-download mid-run.
- Load data ONCE at startup. Compute features/targets ONCE. Split ONCE. Reuse across all experiments in a loop.
- Additional project-specific integrity rules from setup Step 3:
{{data_integrity_custom_rules}}

### Evaluation Protocol Invariants

<!--
  Source was "Super-Fold Invariants" (FX-specific cross-regime fold union).
  Generalized to the user's chosen split protocol (holdout / k-fold / stratified / grouped / time-series /
  walk-forward / super-fold / bootstrap / leave-one-out). Invariants below apply to whichever
  protocol was selected in setup Step 4.
-->

The chosen evaluation protocol is **{{split_protocol_name}}** (see `{{configs_dir}}/splits.yaml` for parameters).

Invariants (enforced by `core/evaluation/splits.py::validate_no_overlap()`):
- Every {{fold_or_group_label}}'s training data contains ZERO overlap with its own validation or test set.
- {{cross_fold_overlap_rule}}
- Validation set: {{val_set_description}}
- Test set: {{test_set_description}}
- **Zero overlap** between train/val/test — verified programmatically before every run. Verification output is part of the experiment log.
- These invariants encode standard ML: train never sees val or test data.

### Experiment Design

<!-- GENERIC: preserved with parameterized metric names. -->

- **Composite metric for keep/revert:** `{{composite_formula}}`. Default: `min(test_primary, val_primary) - {{composite_penalty_weight}} * n_below_threshold_folds`. The model must do well on BOTH val and test across ALL {{fold_or_group_label}}s. The most important {{regime_label_singular}} is {{primary_regime}} but the model must NOT have large regressions in other {{regime_label_plural}}.
- Training is EPOCH-BOUND (minimum {{min_epochs}} epochs with early stopping for neural nets; iteration-bound with early stopping for GBMs). NOT wall-clock-bound.
- **{{cooldown_seconds}}-second cooldown after each experiment** to let compute cool (skip if the hardware policy says sandbox blocks sleep).
- ONE config change per experiment. Diagnose WHY before choosing what to change next.
- Report per-{{fold_or_group_label}} breakdown for BOTH val and test alongside aggregates.
- Dashboard shows train/val/test tabs for per-{{fold_or_group_label}} breakdown. Test is the default view.
- Every config parameter must be wired end-to-end. Dead params are bugs — remove them.
- Every hyperparameter choice must be justified by published papers, model developer guidelines, or prior empirical results from this project. Never choose arbitrary values.

## Autoresearch Agent Protocol (Karpathy-adapted)

<!-- GENERIC: all 8 rules preserved verbatim in spirit; "code changes" rule points to the user's code_versions dir. -->

1. **Always start from the current best config.** Every experiment modifies ONE thing from the best. If it improves, it becomes the new best. If it doesn't, revert and try a different direction. Never wander off from the best baseline.
2. **If you see consecutive discards, stop and rethink.** Multiple failures mean your hypothesis about what to change is wrong. Re-read the per-{{fold_or_group_label}} results. Look at which {{fold_or_group_label}}s are weak and WHY. Don't keep guessing.
3. **Explore around the best AND try radical changes.** Most experiments should be small tweaks around the champion. But occasionally try something bold (different architecture, very different {{sequence_or_feature_scope}}) to escape local optima.
4. **Cite your reasoning for every experiment.** "I'm trying X because {{fold_or_group_label}} Y has [problem] due to Z, and paper W suggests this fix." Not "let me try X and see."
5. **The agent never stops.** If out of ideas, research deeper: read the relevant SOTA tech reports, adapter papers, domain literature. Think harder. Try combining near-misses.
6. **Checkpoint reasoning to memory every few minutes.** The machine crashes often. After every experiment (or every ~3 minutes of reasoning), save the current state to `{{memory_dir}}/project_autoresearch_checkpoint.md`.
7. **Deep per-{{fold_or_group_label}} failure analysis every iteration.** For each {{fold_or_group_label}} with a {{failing_metric_direction}} {{primary_metric_name}}, explain WHY: what {{regime_label_singular}} it is, what conditions, what the uncertainty outputs reveal (high aleatoric = noisy data, high epistemic = model doesn't know, low confidence = skip signal). Use this to guide the next experiment.
8. **Code changes are allowed.** The agent may modify the Python codebase (model architecture, loss function, training loop, features, evaluation) if it has a principled reason. Save modified versions to `{{code_versions_dir}}/` with a version number. Code changes are the most powerful lever — hyperparams only go so far.

## Research-Driven Experiment Selection (STRICT — no blind sweeps)

<!-- GENERIC: 7-step process preserved verbatim. Examples updated to include task-agnostic techniques. -->

The experiment loop is NOT a grid search. It is a research process. Every single experiment must follow this exact sequence:

**Step 1 — Diagnose the champion's weakness.** Look at the per-{{fold_or_group_label}} test results. Which {{fold_or_group_label}}s are weakest? What {{regime_label_singular}} are they? What do the uncertainty metrics say? What does the {{correct_vs_wrong_label}} spread look like for those {{fold_or_group_label}}s? Identify the SPECIFIC failure mode.

**Step 2 — Search the literature.** Based on the diagnosis, search arXiv / known papers for techniques that address the failure mode. Examples (task-agnostic):
- Weak on high-variance subgroups → group-aware training, distributionally robust optimization (Sagawa et al. 2020)
- High epistemic in specific {{fold_or_group_label}}s → data augmentation, deep ensembles (Lakshminarayanan et al. 2017)
- Overfitting to majority class/regime → focal loss (Lin et al. 2017), re-weighting, class-balanced loss (Cui et al. 2019)
- Architecture ceiling hit → residual connections (He et al. 2016), attention (Vaswani et al. 2017)
- LR too high/low → cyclical LR (Smith 2017), warmup (Goyal et al. 2017)
- Calibration issue → temperature scaling (Guo et al. 2017), isotonic regression
- {{domain_specific_diagnosis_examples}}

**Step 3 — Form a hypothesis and predict the outcome.** Write down: "I hypothesize that [change X] will improve [metric Y] on [{{fold_or_group_label}} Z] because [paper/principle]. I predict composite will move from [current] to approximately [target]." If you can't write this sentence, you don't understand what you're doing. Stop and think more.

**Step 4 — Run ONE experiment.** Execute the change. ONE change only.

**Step 5 — Analyze against prediction.** Did the result match your prediction? If yes, why? If no, what does that tell you about your mental model? Update your understanding.

**Step 6 — Document everything.** Write the full cycle (diagnosis → literature → hypothesis → prediction → result → learning) into the experiment log and checkpoint.

**Step 7 — Checkpoint.** Ritual close: every output file listed in the "Dashboard Files Update Mandate" is up to date, then commit the next-experiment command to the checkpoint.

**The goal is monotonic improvement.** Every experiment should have a principled reason to believe it will improve composite score. If you're out of ideas for hyperparameters, the answer is almost always a CODE CHANGE — modify the architecture, loss function, or feature engineering.

## Monotonic Quality Progression (NEVER regress)

<!-- GENERIC: preserved verbatim. -->

The experiment loop must work towards monotonic increase in quality. This means:
- **Never run an experiment you can't justify.** Every experiment must have a written rationale citing literature or prior empirical evidence from this project.
- **Track the champion lineage.** Document the chain: Exp1 (baseline) → Exp5 (technique X, +ΔY) → Exp10 (tweak Z, +ΔW) → etc. Each link must explain WHY the improvement happened.
- **When you hit a plateau, go deeper.** If 3+ consecutive experiments are DISCARD, you're in a local optimum. The answer is NOT more hyperparameter tweaks — it's a structural change: different architecture, different loss, different features, different training procedure.
- **Protect gains.** When trying bold changes, if the result is far worse (composite drops > {{large_regression_threshold}}), investigate WHY before trying the next thing. Understanding failures is as valuable as finding improvements.
- **Quality ratchet:** once a metric improves, treat the new level as the floor. If a change improves test {{primary_metric_name}} but regresses val {{primary_metric_name}} below the previous champion, it's a DISCARD — both must improve or at least hold.
- **Goodhart protection (MANDATORY):** the agent MAY NOT rewrite the composite metric formula, the split protocol, the data integrity invariants, or the primary-metric definition mid-project. These are frozen at setup time. Changes require an explicit user sign-off (documented in the checkpoint as a `RULE_CHANGE` entry).

## MLOps Documentation Standards (MANDATORY)

<!-- GENERIC: preserved verbatim, with parameterized metric names. -->

You are a strong MLOps engineer. Every artifact and every experiment must be documented in proper, readable markdown. No exceptions.

**`{{results_dir}}/experiment_summary.md`** — the master experiment log. Updated after EVERY experiment. Format:

```markdown
## Experiment Log — [Backbone] Phase

### Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected composite change]
- **Result:** Composite [X] | Test {{primary_metric_name}} [Y] | Val {{primary_metric_name}} [Z] | [N]/[K] {{fold_or_group_label}}s above threshold
- **Per-{{fold_or_group_label}} test {{primary_metric_name}}:** {{per_fold_template}}
- **Classification/Regression metrics:** {{secondary_metric_template}}
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned, why result matched/differed from prediction]
- **{{correct_vs_wrong_label}}:** [summary — see per-prediction spreadsheet in trade_logs/]
```

**`{{results_dir}}/trade_logs/`** — per-experiment per-prediction detail (see Per-Prediction Log below).

**Key documentation principles:**
1. **Readable by a human who wasn't there.** Someone reading the experiment summary 6 months from now must understand WHY each experiment was run and WHAT was learned.
2. **No orphan artifacts.** Every file must be referenced from either the checkpoint, experiment summary, or winner README.
3. **Consistent formatting.** Same table format, same metric names, same precision ({{ratio_decimals}} decimal places for ratios, {{percentage_decimals}} for percentages).
4. **Append-only experiment log.** Never delete or rewrite experiment entries. If an experiment was wrong (e.g., bug found), add a note — don't erase history.

## Explainability & Auditability Report (MANDATORY for every NEW BEST)

<!--
  GENERIC: all 14 sections preserved. FX-specific bits (VIX, drawdown period, regime = GFC/post-crash)
  are generalized to the project's subgroup labels. Section numbering frozen.
-->

When a new champion is found, produce a full data-scientist-grade audit to `{{results_dir}}/winners/<exp_id>/audit_report.md`. This is not optional — a model without explainability is un-deployable.

**Required sections (all 14):**

1. **Executive summary** — Champion test {{primary_metric_name}}, {{secondary_primary_metric_label}}, {{risk_metric_label}}, all {{num_folds_or_groups}} {{fold_or_group_label}} metrics. {{Regime_label_plural_titled}}-by-{{regime_label_singular}} pass/fail.

2. **Feature importance (permutation method)** — For each of the {{n_features}} features, shuffle that column in the test set, re-evaluate, report the drop in test {{primary_metric_name}}. Rank features by importance. Cite: Breiman (2001) "Random Forests" section on variable importance. Save `feature_importance.csv` with columns `[feature_name, metric_drop, rank, domain_category]`.

3. **Top-N feature analysis** — For the top 10 most-impactful features, explain:
   - What the feature measures (from feature docstrings)
   - Why it matters substantively in the domain (domain knowledge)
   - Per-{{fold_or_group_label}} impact: is feature X strong in {{regime_label_singular}} A but weak in B?

4. **SHAP-style local explanations** — For 10 random test-set predictions, compute per-feature contribution. Use gradient × input as a cheap approximation for neural nets, or `shap.TreeExplainer` for GBMs. Save as `shap_local.csv`.

5. **Per-{{fold_or_group_label}} feature drift** — For each {{fold_or_group_label}}, compute mean/std of each feature vs the training set. Features with |Z| > 2 on a {{fold_or_group_label}} indicate distribution shift. Report top 5 drifted features per {{fold_or_group_label}} with explanation.

6. **Calibration analysis** — For regression: predicted quantile vs realized mean; ideal monotonic; report calibration error. For classification: reliability diagram, ECE. Cite: Guo et al. (2017) "On Calibration of Modern Neural Networks."

7. **Uncertainty sanity** — Plot aleatoric vs prediction |error|. Should be monotonic. Plot confidence vs correctness; bucket predictions by confidence decile; report accuracy per decile. Cite: Kendall & Gal (2017).

8. **Per-{{regime_label_singular}} prediction distribution** — For each {{fold_or_group_label}}, plot histogram of predictions. Identify if the model is systematically biased (e.g., always predicting the mean) vs appropriately reactive.

9. **Error attribution / top-N winners & losers** — For each test {{fold_or_group_label}}, report top-5 best-predicted examples and top-5 worst. Pattern analysis: are errors concentrated on specific groups?

10. **Risk audit** — {{risk_audit_content}} (for regression: residual skew/kurtosis, max |error| period, conditional VaR; for classification: per-class error rates, false-positive/negative cost; for time-series: max drawdown period when mapped to a downstream strategy).

11. **Data pipeline audit** — Reassert: zero train/val/test leakage, {{purge_mechanism}}, {{embargo_value}}, {{label_horizon_buffer_units}} label horizon buffer. Rerun `validate_no_overlap()` and include the output verbatim. No assumptions — MEASURE.

12. **Model config complete dump** — Every hyperparameter + the Python version + framework version (torch/sklearn/xgboost/etc.) + numpy version + random seed. For true reproducibility.

13. **Known limitations & risks** — What {{regime_label_plural}} has this model NEVER been tested on? Where will it most likely fail in production?

14. **Deployment checklist** — What monitoring is needed? What's the kill-switch criterion? What retraining cadence? {{deployment_specific_items}}

**Implementation:** `core/winner_archive.py::generate_audit_report(winner_dir)` produces the full report. Runner calls it automatically when `composite > prev_best`.

## Winner Definition (CLARIFICATION)

<!-- GENERIC: preserved verbatim. -->

**"Winner" means the GLOBAL champion across ALL backbones and ALL experiments.** Not per-backbone. The one single best model (by composite score) at any point in time.

Per-backbone best is tracked separately in the checkpoint but does NOT get archived to `winners/` unless it is also the global best.

When a new experiment beats the global composite:
1. Save artifacts to `{{results_dir}}/winners/<backbone>_exp<N>_<desc>/`
2. Include: README.md, config.json, model_checkpoint.pt (or `.joblib` for sklearn/GBM), code/ (frozen snapshot), inference/, reproduction/, audit_report.md (14 sections per audit rules), colab_train_and_infer.ipynb
3. Update `best_config.json` at repo root

## Per-Backbone Code Snapshots (MANDATORY)

<!-- GENERIC: preserved. Folder names are parameterized. -->

Before starting experiments on a new backbone, snapshot the CURRENT backbone code + training loop + runner to `{{code_versions_dir}}/<backbone>_start/` so you can diff what changed during that backbone's exploration. This prevents mixing backbone-X-specific changes into backbone-Y exploration.

```
{{code_versions_dir}}/
  v1_original/                 # pre-any-change snapshot
  <backbone1>_start/           # snapshot before <backbone1> experiments begin
  <backbone1>_final/           # snapshot after the 50-experiment cycle
  <backbone2>_start/
  ...
```

Rule: never modify code specific to backbone X while experiments on backbone Y are in progress. Finish one backbone's {{n_experiments_per_backbone}} experiments, snapshot, then move on.

## Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)

<!-- GENERIC: preserved verbatim. File paths parameterized. -->

**Every single experiment MUST have a complete reasoning record in `{{results_dir}}/reasoning_annotations.json` keyed by `experiment_num`. No experiment ships without one. Orphan entries or "auto-backfilled" placeholders are a bug.**

The entry is a JSON object with these REQUIRED fields (all non-empty strings):

| Field | Content | Source |
|-------|---------|--------|
| `diagnosis` | Why THIS experiment now: which champion weakness it targets, which {{fold_or_group_label}} is weakest and why ({{regime_label_singular}}, conditions, uncertainty profile), what prior experiments ruled out the alternatives | Authored by Claude BEFORE running |
| `citations` | Full author/year/venue string for every paper motivating the choice. Multiple papers semicolon-separated. Parenthetical-only tags are INSUFFICIENT — expand to full reference | Authored before running |
| `hypothesis` | Concrete mechanism: "parameter X = value Y will change metric Z via mechanism M (what the paper argues)". Not just "try X". | Authored before running |
| `prediction` | Numeric target: "composite should move from +X to +Y–Z; val {{fold_or_group_label}} K expected to improve from −A to −B/+C". Include ranges, not single numbers | Authored before running |
| `verdict` | KEEP / DISCARD / NEAR-MISS + composite achieved + delta vs global best + which {{fold_or_group_label}}s carried it | Written immediately after results |
| `learning` | What this result updates in the mental model: did the prediction hold? Which axis is now exhausted? Which variant should be tried next? | Written immediately after results |
| `_manual` | `true` if authored by Claude as part of the 7-step process; `false` only for purely mechanical variance-check runs that reuse a prior annotation template | Always set |

**Dashboard `dashboard.html` renders all 7 fields in the detail panel when a row is clicked.** If any field is missing, empty, or placeholder, that's a regression — fix it before the next experiment.

**Write cadence — two places on every run:**
1. **BEFORE the experiment command runs:** Claude adds the entry to `reasoning_annotations.json` with `diagnosis`, `citations`, `hypothesis`, `prediction`, `_manual: true`. The experiment is not launched until this entry exists. `core/reasoning.py::commit_pre_run()` enforces this.
2. **AFTER the experiment completes:** Claude appends `verdict` and `learning` to the same entry. The runner's auto-written fallback is only a placeholder with `TODO-REWRITE` sentinels.

**Enforcement:** At the start of every experiment cycle, Claude MUST check:
- Does `reasoning_annotations.json` already have a complete entry for the previous experiment? If no `verdict`/`learning`, write them before starting the next.
- Is the next experiment's pre-entry already authored? If no, write it now.
- Did the citation field survive any recent `backfill_reasoning.py` run? Check `_manual: true` is preserved.

**Parallel write to `research_journal.md`.** The same diagnosis/citations/hypothesis/prediction/verdict/learning narrative belongs in the research journal in markdown form, keyed by experiment number. Journal format:

```markdown
## Exp<N> — <short title>
**Diagnosis:** ...
**Citations:** ...
**Hypothesis:** ...
**Prediction:** ...
**Verdict:** ...
**Learning:** ...
```

The journal is the human-readable twin of the JSON; they must stay in sync. If they drift, the JSON is authoritative (runner-written), and the journal gets updated from it.

**`backfill_reasoning.py` rules:**
- Only runs on DEMAND — not automatically, not after every experiment
- Never overwrites entries with `_manual: true`
- Fills in only the fields that are empty AND whose experiment JSONL entry exists
- Logs every overwrite it makes
- Is NOT a substitute for authoring the annotation before the run

**Runner's responsibility (`core/runner.py`):**
- On every invocation, merge the user-visible description's citation tags + the CLI flag delta into the runtime `reasoning_annotations.json` entry — WITHOUT clobbering `_manual: true` fields
- Populate `verdict` and `learning` from the results automatically as a fallback
- Never emit placeholder strings; if it can't compute a field, leave it blank with `TODO-REWRITE` and `_needs_rewrite: true`, and log a warning so Claude knows to author it

**Why this matters:** the dashboard is the shared memory between sessions. A new Claude Code session resuming this project reads the dashboard reasoning panel to understand why a champion was chosen. Missing or shallow annotations mean lost institutional knowledge and wasted experiments that retry dead-end ideas.

## Per-Backbone N-Experiment Mandate (MANDATORY, not optional)

<!--
  GENERIC: source had 50-experiment mandate. Template parameterizes N per user's setup Step 11.
  Default is 50 unless user overrides.
-->

**Every backbone gets a full {{n_experiments_per_backbone}}-experiment exploration.** Do not stop early because "axes look exhausted." The mandate:

1. **{{n_experiments_per_backbone}} experiments per backbone** — no fewer. If standard HP sweeps plateau, explore:
   - Architectural variants from arXiv literature through {{current_year}}
   - Cross-variant combinations
   - Feature engineering changes (input projections, feature selection)
   - Multi-seed studies on the champion to characterize variance
   - Regularization beyond weight decay (label smoothing, mixup, stochastic depth)

2. **Research latest SOTA ({{sota_year_range}} arXiv papers) before declaring any backbone done.** See "Per-Backbone SOTA Training Recipes" below for the starter table — update as the literature evolves.

3. **Each experiment must cite its paper/source** — no "let me try X". Per Autoresearch Agent Protocol rule 4.

4. **Document all {{n_experiments_per_backbone}} in `research_journal.md`** — even DISCARDs. Negative results are informative.

5. **Only after {{n_experiments_per_backbone}} experiments** may a backbone be declared "done" and progression to the next backbone resume.

## Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)

<!--
  GENERIC: preserved. Recipes below are auto-generated per task-type in setup Step 8/9.
  The setup wizard populates this table from a curated 2024-2026 SOTA catalog in
  `templates/sota_catalog.yaml`, filtered to the user's task type.
-->

**Every backbone picks its OWN epochs, patience, learning rate, batch size, scheduler, and optimizer from the latest SOTA literature for THAT architecture. Never copy another backbone's config.**

**Before the first experiment on any new backbone, Claude MUST:**

1. **Pull the latest {{sota_year_range}} arXiv / NeurIPS / ICML / ICLR paper for the backbone family.** For each backbone, read the paper's experimental section and note:
   - Recommended epochs (and how they terminate — fixed vs early-stop)
   - Patience threshold (absolute vs relative to epochs)
   - Learning rate (and whether warmup is required)
   - Scheduler (cosine annealing, linear decay, plateau, ReduceLROnPlateau)
   - Optimizer (Adam vs AdamW vs Lion vs Adafactor vs SOAP)
   - Batch size (and whether it's effective-batch via grad accumulation)
   - Weight decay
   - Gradient clipping
   - Loss function

2. **Record the chosen recipe with a paper citation in the reasoning annotation** for Experiment 1 of that backbone.

3. **Justify the DELTA from the paper.** If our chosen epochs deviate from the paper's recommendation, the reasoning entry MUST explain why.

4. **Never assume "one recipe works for everything."** Empirical proof from prior projects:
   - Using one backbone's epoch count for another often costs 20% of peak performance.
   - Ignoring a paper's minimum sequence length produces uninterpretable negative results.

### Backbone-Specific Training Recipes (auto-generated from SOTA catalog for task = {{task_type}})

<!--
  The table below is populated by the setup wizard from
  `templates/sota_catalog.yaml`, filtered on the user's task type and hardware budget.
  For non-neural tasks (classical GBM-focused), Tier 2 is skipped.
-->

#### Tier 1 — classical baselines (required for every run to establish a floor)

{{tier1_recipes_table}}

#### Tier 2 — {{sota_year_range}} SOTA ({{task_type}}-specific)

{{tier2_recipes_table}}

#### Tier 3 — gradient boosted machines (each is its OWN backbone, run independently)

<!--
  GENERIC: preserved verbatim. GBMs are always three separate backbones.
-->

GBMs are fundamentally different from neural nets: no epochs, no LR schedule, no batch. Iterations are tree-count. Each GBM has its own paper, its own hyperparameter language, its own {{n_experiments_per_backbone}}-experiment exploration budget. **Do NOT bundle xgboost/lightgbm/catboost as "the GBM backbone" — they are three separate architectures with different splitting algorithms, different regularization mechanisms, and different category handling.** Explore each fully.

| Backbone | Key HP | Default Start | Regularization | Special feature | Paper (full citation) |
|----------|--------|---------------|----------------|------------------|----------------------|
| **xgboost** | n_estimators=1500, max_depth=6, lr=0.03, subsample=0.8, colsample_bytree=0.8, early_stop=50 | level-wise trees | reg_lambda=1.0, reg_alpha=0, min_child_weight=1, gamma=0 | 2nd-order Newton boosting; monotonic constraints; histogram method | Chen & Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting System' (arXiv:1603.02754) |
| **lightgbm** | n_estimators=2000, num_leaves=63, lr=0.03, feature_fraction=0.8, bagging_fraction=0.8, early_stop=50 | leaf-wise trees (GOSS) | reg_alpha, reg_lambda, min_data_in_leaf=20 | GOSS; EFB; native categorical | Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017 NeurIPS 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree' |
| **catboost** | iterations=2000, depth=6, lr=0.03, random_strength=1.0, early_stop=100 | symmetric oblivious trees | l2_leaf_reg=3, bagging_temperature=1.0 | Ordered boosting; native ordered target-stat for categoricals | Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin 2018 NeurIPS 'CatBoost: Unbiased Boosting with Categorical Features' (arXiv:1706.09516) |

**Why GBMs are 3 separate backbones:**
- **XGBoost** uses 2nd-order gradient info (Hessian) — effective on imbalanced targets, fast on GPU.
- **LightGBM** uses leaf-wise growth + GOSS — fastest wall-clock, handles large n well.
- **CatBoost** uses ordered boosting against prediction shift + best default categorical handling — slowest but often best out-of-box accuracy on tabular.

Each will rank different features as important and each has different failure modes. Do not skip any.

**Re-derive for EVERY new SOTA variant.** When a backbone family has multiple variants, each variant re-derives its recipe from its OWN paper.

## GPU Memory Constraint (MANDATORY — {{gpu_vram_gb}} GB VRAM hard cap)

<!-- GENERIC: preserved. Budget scales with the user's reported VRAM. -->

**This machine has {{gpu_vram_gb}} GB of GPU VRAM. Every backbone selection, every experiment, every fine-tuning run MUST fit within this budget with headroom. A model that OOMs mid-training is not a valid experiment — it's a wasted GPU cycle and a crash risk.**

**Memory budget breakdown ({{gpu_vram_gb}} GB total):**

| Component | Budget | Notes |
|-----------|--------|-------|
| Model parameters | ≤ {{vram_params_gb}} GB | FP32 weights; BF16/FP16 halves this |
| Optimizer state (AdamW) | ≤ {{vram_optim_gb}} GB | Adam stores 2 moments at FP32 → ≈ 2× param size |
| Gradients | ≤ {{vram_grads_gb}} GB | Same size as params; freed after step |
| Activations | ≤ {{vram_acts_gb}} GB | batch × seq × hidden, scales with bs and depth |
| Reserved / fragmentation | ≥ {{vram_reserve_gb}} GB | Framework caching allocator overhead |

**Practical parameter ceilings by training mode (derived from {{gpu_vram_gb}} GB):**

| Training mode | Max params @ FP32 | Max params @ BF16/FP16 | Max params w/ grad-ckpt + BF16 |
|---------------|-------------------|------------------------|-------------------------------|
| From-scratch train (Adam full states) | {{ceiling_fp32_scratch}} | {{ceiling_bf16_scratch}} | {{ceiling_bf16_ckpt_scratch}} |
| Full fine-tune | {{ceiling_fp32_ft}} | {{ceiling_bf16_ft}} | {{ceiling_bf16_ckpt_ft}} |
| Parameter-efficient FT (LoRA r=8) | {{ceiling_fp32_peft}} | {{ceiling_bf16_peft}} | {{ceiling_bf16_ckpt_peft}} |
| Frozen-backbone head-only FT | {{ceiling_fp32_head}} | {{ceiling_bf16_head}} | {{ceiling_bf16_ckpt_head}} |
| Inference only (no grads) | {{ceiling_fp32_infer}} | {{ceiling_bf16_infer}} | {{ceiling_bf16_infer}} |

**Mandatory pre-flight check for any new backbone.** Before launching Experiment 1 on ANY new backbone, include in the reasoning annotation:

```
Measured/estimated size: N million params
Training mode selected: [from-scratch | LoRA fine-tune | head-only FT | zero-shot]
Expected peak VRAM: <X> GB at bs=<Y>, seq=<Z>, precision=<FP32|BF16>
Headroom vs {{gpu_vram_gb}} GB: <{{gpu_vram_gb}} - X> GB
Fallback plan if OOM: [reduce bs | switch to BF16 | gradient checkpointing | adapter-only]
```

Without this entry, Experiment 1 does not launch. The same check applies whenever batch size or sequence length changes.

**Default protocol when adopting a new foundation model:**
1. Start with the SMALLEST published checkpoint of that family.
2. Run zero-shot first — measure composite without any training.
3. If zero-shot is promising, fine-tune (full or PEFT depending on size).
4. Scale up to larger checkpoint ONLY if smaller shows signal AND memory math works.

**Mixed precision note.** BF16 preferred over FP16 on modern GPUs — keeps dynamic range without loss-scaling. Use `torch.autocast(dtype=torch.bfloat16)` + `GradScaler` unset. LayerNorm/GroupNorm should stay FP32.

**Gradient checkpointing note.** Use `torch.utils.checkpoint.checkpoint_sequential` for any model > 200 M params being fine-tuned. Costs ~30% more FLOPs but cuts activation memory by 70-80%.

### Epoch-budget rule of thumb (when in doubt)

<!-- GENERIC: preserved verbatim. -->

If the paper's recipe is unclear, use this scaling heuristic:

- **Data scaling (Smith 2017):** `epochs ≈ paper_epochs × (paper_n / our_n)^0.5`.
- **Parameter scaling (Kaplan 2020):** `epochs ≈ base × (our_params / paper_params)^0.2`.
- **Patience as 15% of epochs** is a safe default when papers don't specify.
- **Warmup = 5-10% of total epochs** for transformer families.

These are starting heuristics; always iterate and checkpoint the actual convergence profile per backbone.

## Backbone Isolation Rule

<!-- GENERIC: preserved verbatim. Paths parameterized. -->

Before starting experiments on a new backbone, snapshot `core/backbones/<backbone>.py`, `core/runner.py`, and any modified training utilities to `{{code_versions_dir}}/<backbone>_start/`. Do NOT modify backbone code specific to backbone X while experiments on backbone Y are in progress. Complete one backbone's {{n_experiments_per_backbone}}-experiment cycle, snapshot as `<backbone>_final/`, then move to the next backbone.

## Dashboard Backbone Tabs

<!-- GENERIC: preserved verbatim. -->

Dashboard (`{{results_dir}}/dashboard.html`) renders a backbone tab bar above the experiment list. Default view shows "ALL". Tabs filter the scrollable experiment list to just that backbone's experiments. Click to switch.

## Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)

<!--
  GENERIC: preserved verbatim. Ownership table listed below — runner writes or Claude writes.
  TODO-REWRITE sentinels are enforced by core/reasoning.py.
-->

**Every single experiment updates ALL the following files. If any file is stale after an experiment completes, that's a regression — stop and fix before moving on. No "I'll batch-update at the end." No "It's just a variance check."**

**Ownership — who writes what:**

| File | Written by | When | Content |
|------|------------|------|---------|
| `{{results_dir}}/experiment_log.jsonl` | **runner (auto)** | every run, appended | full metrics: composite, test/val/train {{primary_metric_name}}, per-{{fold_or_group_label}} results, per-window classification/regression metrics, uncertainty, timing, config |
| `{{results_dir}}/best_config.json` | **runner (auto)** | only when new GLOBAL champion | overwritten with full champion entry |
| `{{results_dir}}/best_model.pt` | **runner (auto)** | only when new GLOBAL champion | weights + scaler + config + feature_columns + provenance (or `.joblib` for sklearn/GBM) |
| `{{results_dir}}/trade_logs/exp<N>_predictions.csv` | **runner (auto)** | every run | one row per test prediction (per-prediction log; see Per-Prediction Log section) |
| `{{results_dir}}/trade_logs/exp<N>_prediction_summary.json` | **runner (auto)** | every run | per-{{fold_or_group_label}} totals, correct, wrong, avg metric, confidence-stratified accuracy |
| `{{results_dir}}/reasoning_annotations.json` | **Claude BEFORE run + runner AFTER run** | every run, two-phase | diagnosis, citations, hypothesis, prediction (Claude); verdict, learning (runner fallback, Claude overrides) |
| `{{results_dir}}/research_journal.md` | **Claude** | every run, appended | markdown narrative of the full 7-step process |
| `{{results_dir}}/experiment_summary.md` | **Claude** | every run, appended | short tabular entry per experiment |
| `{{memory_dir}}/project_autoresearch_checkpoint.md` | **Claude** | every run | update champion, history table, next-command block |
| `{{results_dir}}/winners/<backbone>_exp<N>_<desc>/*` | **Claude + runner** | only when new GLOBAL champion | README.md, config.json, model_checkpoint (copy), code/ snapshot, inference/predict.py, per_fold_results.json, experiment_log_entry.json |
| `{{results_dir}}/winners/<backbone>_exp<N>_<desc>/audit_report.md` | **Claude (via `winner_archive.py`)** | only when new GLOBAL champion | 14-section audit per Explainability & Auditability Report spec |
| `{{results_dir}}/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb` | **Claude (via `winner_archive.py`)** | only when new GLOBAL champion | self-contained Colab notebook |
| `{{results_dir}}/dashboard.html` | **Claude (rarely)** | only when adding a new metric/tab | static HTML — reads JSONL + annotations live |

**Per-experiment ritual (repeat in order, every single run):**

1. **Before launch:** open `reasoning_annotations.json`, insert a new entry keyed by the upcoming `experiment_num` with `diagnosis`, `citations` (full reference), `hypothesis`, `prediction` (numeric target), `_manual: true`. If this entry isn't there, the experiment doesn't run. `core/reasoning.py` enforces this with the Citation Rigor + Reasoning Blob Completeness validators.
2. **Before launch:** append a matching section to `research_journal.md` with the same 4 fields in markdown.
3. **Launch:** run the CLI command.
4. **Runner auto-updates:** JSONL, best_config (if champion), best_model (if champion), trade_logs CSV + JSON, reasoning_annotations verdict/learning fallback.
5. **After completion:** Claude reads the runner output, overwrites the `verdict` and `learning` fields in `reasoning_annotations.json` with richer analysis (per-{{fold_or_group_label}} narrative, which {{regime_label_plural}} won/lost, uncertainty profile). Updates the corresponding section in `research_journal.md`.
6. **After completion:** Claude appends a row to `experiment_summary.md`.
7. **After completion:** Claude updates `{{memory_dir}}/project_autoresearch_checkpoint.md`.
8. **If new champion:** Claude archives to `winners/<backbone>_exp<N>_<desc>/`.

**Verification at the start of every experiment cycle:**

Before launching Experiment N+1, confirm all of these are CURRENT for Experiment N:

- [ ] `experiment_log.jsonl` has an entry for N (runner writes, verify)
- [ ] `reasoning_annotations.json[N]` has all 7 fields non-empty and non-placeholder
- [ ] `research_journal.md` has a section for N
- [ ] `experiment_summary.md` has a row for N
- [ ] `{{memory_dir}}/project_autoresearch_checkpoint.md` references N in its history table
- [ ] `trade_logs/expN_predictions.csv` and `expN_prediction_summary.json` exist
- [ ] If N set a new champion: `winners/<backbone>_expN_<desc>/` exists with all required files

If ANY checkbox is unchecked, stop and fix BEFORE launching N+1.

**Placeholder strings are a bug.** The runner refuses to fabricate pre-run content. If a pre-run entry is missing, the runner inserts `"TODO-REWRITE"` sentinel values and a `_needs_rewrite: true` flag — Claude MUST rewrite those entries before launching the next experiment.

## Citation Rigor (MANDATORY format for `citations` field)

<!-- GENERIC: preserved verbatim. Examples are domain-neutral. -->

**Every citation string MUST contain, for every paper referenced:**

1. **All authors' surnames** (not just first-author et al. unless > 6 authors)
2. **Year** of publication
3. **Venue** — journal name, conference abbreviation (NeurIPS, ICML, ICLR, AAAI, CVPR, KDD, etc.), or `arXiv` if preprint-only
4. **Full paper title** in single quotes
5. **arXiv ID** in the form `(arXiv:XXXX.YYYYY)` if available — mandatory for any paper posted to arXiv
6. **One-sentence relevance note** — why this paper motivates THIS experiment specifically

**Format template:**

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) — one-sentence note on why we cite it here.
```

**Multiple papers separated by semicolons + linebreak.** Minimum one primary citation per experiment; secondary citations encouraged when combining ideas.

**Examples of GOOD citations (copy this style):**

> Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 ICLR 'On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima' (arXiv:1609.04836) — motivates bs=16 as a flat-minima probe.

> Loshchilov & Hutter 2019 ICLR 'Decoupled Weight Decay Regularization' (arXiv:1711.05101) — AdamW wd acts as decoupled weight shrinkage, so perturbations must be log-scale.

**Examples of BAD citations (REJECTED — rewrite required):**

- `"Keskar 2017 flat minima"` — missing coauthors, venue, title, arXiv, relevance note
- `"(Keskar2017)"` — parenthetical tag only, useless
- `"Keskar et al."` — no year, no venue
- `"arxiv paper on batch size"` — no attribution
- `"(no citation tag)"` — confesses the author didn't do the work
- `"see research_journal.md"` — redirects instead of citing

**The goal:** anyone (including a future Claude Code session with zero project context) must be able to open the dashboard, click a row, read the `citations` field, and immediately know which paper to read and why.

**Arxiv ID lookup discipline.** If you know the paper but not its arXiv ID, fetch it via WebSearch or WebFetch (arxiv.org/abs search) before writing the entry. Authoring a citation without the arXiv ID is a partial job.

## Reasoning Blob Completeness (what "full reasoning" means)

<!-- GENERIC: preserved verbatim. -->

Each of the 7 fields in `reasoning_annotations.json` has a minimum content spec. Entries that fall below this spec must be rewritten. `core/reasoning.py::validate_reasoning_blob()` enforces these floors.

| Field | Minimum content | Word count floor | Must include |
|-------|-----------------|------------------|--------------|
| `diagnosis` | Why THIS experiment NOW; which champion weakness; which {{fold_or_group_label}} is worst and why | ≥ 60 words | Reference to at least one prior experiment by number OR a per-{{fold_or_group_label}} metric from the current champion |
| `citations` | Per the Citation Rigor spec above | ≥ 40 words for single paper, ≥ 80 for multi-paper | Author list + year + venue + title + arXiv ID + relevance note for each paper |
| `hypothesis` | The config change stated mechanistically — what parameter(s) move, what they do in the model, what the cited paper predicts | ≥ 50 words | The word "mechanism" or "because" or "per [paper]"; the specific parameter and value |
| `prediction` | Concrete numeric range on composite AND at least one sub-metric prediction | ≥ 25 words | A numeric range; a direction for at least one sub-metric |
| `verdict` | KEEP/DISCARD/NEAR-MISS + exact composite + delta vs global best + per-{{fold_or_group_label}} narrative | ≥ 30 words | Status label; composite to 4 decimals; mention of at least one per-{{fold_or_group_label}} result |
| `learning` | What this updates in the mental model; which axis is now closed/open; what to try next | ≥ 40 words | "Axis closed" / "axis open" language OR a concrete "next try: ..." |
| `_manual` | Boolean | — | `true` if Claude-authored; `false` only for mechanical reruns |

**When running a batch of variance checks** (same config, varying seed), the `_manual: true` entries can share templated `diagnosis` and `citations` across runs, but `verdict`/`learning` must always be per-run-specific.

**Batch updates are forbidden.** Don't do 5 experiments then update the journal/summary/checkpoint in one go — each experiment's state gets stale and crash-recovery breaks.

## Loss Function Rules

<!--
  Source was "Heteroscedastic Loss Rules (Kendall & Gal 2017)" — FX-specific.
  Generalized: each task type has its own default loss + tuning rules.
  Task-specific subsection is auto-filled from task_type.
-->

{{task_specific_loss_rules}}

### Heteroscedastic / uncertainty-aware loss (neural regression & time-series)

If using uncertainty-aware loss (mean + log-variance per prediction), loss = `exp(-s) * base_loss(mu, y) + 0.5 * s`:
- **Variance-branch dominance is the #1 failure mode.** If aleatoric > 0.2 of the target scale, the model is copping out. Fix: higher LR, more epochs, or clamp log_var.
- **Optimal aleatoric range: 0.05-0.15 of target scale.** Below 0.05 = overconfident. Above 0.20 = lazy variance.
- **Het-loss needs ~50% more epochs than plain base loss** to converge.
- **LR sweet spot shifts up slightly** — the `exp(-s)` weighting reduces effective gradient on mean.
- **Monitor uncertainty per {{fold_or_group_label}}:** high aleatoric = noisy data correctly identified; high epistemic = model needs more data from that {{regime_label_singular}}. Use `confidence < 0.8` as a "skip prediction" signal.

Cite: Kendall & Gal 2017 NeurIPS 'What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?' (arXiv:1703.04977).

## Winner Archiving Protocol (MANDATORY for every NEW BEST)

<!--
  GENERIC: preserved verbatim. Paths parameterized. FX-specific trading README section
  is generalized to a "Deployment Strategy" section with task-conditional content.
-->

Every time a new champion is found (status=KEEP and composite > previous best), archive ALL artifacts to a self-contained subdirectory. The archive must be fully portable — someone can copy the directory to another machine and reproduce + run inference without any external dependencies beyond the Python environment.

**Directory structure:** `{{results_dir}}/winners/<backbone>_exp<N>_<short_description>/`

```
winners/
  <backbone>_exp<N>_<desc>/
    README.md                    # Full description (see template below)
    config.json                  # Exact config that produced this winner
    model_checkpoint.pt          # Saved model weights (or .joblib for sklearn/GBM)
    experiment_log_entry.json    # The JSONL entry for this experiment
    per_fold_results.json        # Full per-{{fold_or_group_label}} val + test breakdown
    code/                        # Frozen snapshot of ALL source code at time of win
      backbones/
      evaluation/
      runner.py
      reasoning.py
      checkpoint.py
      winner_archive.py
    inference/
      predict.py                 # Standalone inference script with sample usage
      README_inference.md        # How to load model and run predictions
    reproduction/
      reproduce_log.txt          # Output from reproduction run
      seed_variance.json         # Cross-seed results if available
    audit_report.md              # 14-section explainability audit
    colab_train_and_infer.ipynb  # Self-contained Colab notebook
```

**README.md template for each winner:**
- Model name + experiment number
- Champion composite score, test {{primary_metric_name}}, val {{primary_metric_name}}
- Per-{{fold_or_group_label}} test {{primary_metric_name}} table
- Per-{{fold_or_group_label}} val {{primary_metric_name}} table
- Full hyperparameter config
- Architecture description
- Key insight: WHY this config won
- Training details: epochs/iterations run, early stopping epoch, training time
- Uncertainty metrics: aleatoric, epistemic, confidence per {{fold_or_group_label}}
- Task-specific secondary metrics ({{secondary_metrics_list}})
- Reproduction status: seeds tested, variance observed
- Sample inference code snippet
- **Deployment Strategy** section (task-conditional; see below)

**After archiving:** Rerun the winner to verify reproduction. The reproduction log goes into `reproduction/reproduce_log.txt`. If composite differs by > {{reproduce_tolerance}}, flag it and investigate before proceeding.

**Model checkpoint MUST be portable and self-contained:**
For neural nets (`.pt`): include in the torch.save dict:
- `model_state_dict`
- `config` — hyperparameters dict
- `scaler_mean`, `scaler_scale` — StandardScaler parameters (ndarray[n_features])
- `feature_columns` — list of feature names in order
- `target_columns` — list of target names
- `n_features` — int
- `composite`, `description`, `backbone`, `experiment_num` — provenance

For sklearn/GBM (`.joblib`): pickle the fitted `Backbone` wrapper (which includes the scaler, feature list, and target list as instance attributes).

**The `predict.py` inference script must:**
1. Load the model checkpoint
2. Accept raw feature input (or a dataset path)
3. Output: prediction, confidence, aleatoric uncertainty, epistemic uncertainty (when applicable)
4. Include a `__main__` block with a working example
5. Print results in a clear table format

**Deployment Strategy section (MANDATORY in every winner README.md):**
Task-conditional content. For {{task_type}}:

{{deployment_strategy_section_template}}

## Google Colab Notebook (MANDATORY for every winner)

<!-- GENERIC: preserved verbatim. Data-download cell is task-agnostic (use bundled CSV or sklearn.datasets). -->

For every archived winner, generate a self-contained Google Colab notebook at `{{results_dir}}/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb` that anyone can open in Colab and run end-to-end.

**The Colab notebook must contain:**
1. **Setup cell:** `!pip install` all dependencies
2. **Data cell:** load the project's dataset (bundled CSV preferred over network download for reproducibility)
3. **Feature engineering cell:** compute the project's features with clear explanations
4. **Training cell:** full training loop reproducing the winner config exactly. Print per-epoch loss + validation metrics.
5. **Evaluation cell:** evaluate on all {{num_folds_or_groups}} test {{fold_or_group_label}}s, print per-{{fold_or_group_label}} metric table, compute composite score
6. **Inference cell:** load trained model, accept a sample input, produce predictions with uncertainty bands
7. **Visualization cell:** plot {{visualization_plan}}
8. **Export cell:** save model weights + config for deployment

**Notebook principles:**
- Every cell must have a markdown header explaining what it does and WHY
- Include the champion config as a clearly visible dict at the top
- Use `torch.manual_seed()` and `np.random.seed()` for reproducibility
- Print all key metrics at the end in a summary table
- Target runtime: < 5 minutes on Colab free tier (T4 GPU or CPU)
- The notebook must be SELF-CONTAINED — no imports from the main package (inline all necessary code)

## Traditional ML Metrics (MANDATORY for every experiment)

<!--
  GENERIC: source had FX-specific "direction classification" framing.
  Generalized: each task type has its own traditional ML metric list.
-->

In addition to the primary composite metric, compute and log the task-type secondary metrics for every experiment.

**For task = {{task_type}}:**

{{secondary_metrics_detailed_spec}}

These must appear in:
1. `core/evaluation/metrics.py::full_report()` output
2. Per-{{fold_or_group_label}} results in JSONL log entries
3. Dashboard per-{{fold_or_group_label}} tables
4. Winner archive `per_fold_results.json`
5. Experiment summary markdown

## Per-Prediction Log (MANDATORY for every experiment)

<!--
  Source: "Trade-Level Win/Loss Logging" — FX-specific (pnl_bps, pair).
  Generalized: per-prediction CSV applies to every task type with task-specific columns.
-->

For EVERY experiment, produce a per-prediction spreadsheet on test data. This is critical for understanding WHERE the model makes and loses value — not just aggregate metrics.

**Output file:** `{{results_dir}}/trade_logs/exp<N>_predictions.csv`

**Generic columns (present for every task):**

| Column | Description |
|--------|-------------|
| index | Row index in test data (or date/timestamp if time-series) |
| fold | Which test {{fold_or_group_label}} |
| {{regime_label_singular}} | {{regime_label_singular}} label |
| prediction | Model point prediction |
| actual | Ground-truth label |
| confidence | Model confidence (1 − epistemic, when available) |
| aleatoric | Aleatoric uncertainty |
| epistemic | Epistemic uncertainty |
| primary_metric_value | Per-row contribution to the primary metric |

**Task-specific columns (added per {{task_type}}):**

{{task_specific_prediction_columns}}

**Per-{{fold_or_group_label}} summary in `exp<N>_prediction_summary.json`:**
- Total predictions, correct/wrong counts per {{fold_or_group_label}}
- Average winning magnitude, average losing magnitude
- Largest single correct, largest single wrong
- Correct/wrong ratio
- Streak analysis: max consecutive correct, max consecutive wrong
- Confidence-stratified accuracy: accuracy when confidence > 0.9 vs < 0.9

**This data enables:**
- Identifying specific examples/{{regime_label_plural}} where the model fails
- Confidence calibration analysis
- Downstream decision-rule research (threshold tuning, filter rules)

## Architecture

<!-- GENERIC: preserved verbatim. Paths parameterized. -->

- **Autoresearch loop = Claude agent.** Claude reads results, decides what to try, calls the runner, reads output. The intelligence is in the agent, NOT in Python code. No pre-baked experiment lists.
- Runner (`core/runner.py`) executes ONE experiment per call. Logs JSONL. That's it.
- Dashboard (`dashboard.html`) reads logs. DECOUPLED from runner.
- Save checkpoint after every experiment.
- Use relative imports (`from .backbones import ...`).

## Validation Checklist (Run Before Every Experiment Session)

<!-- GENERIC: items parameterized to user's split protocol. -->

1. `splits.validate_no_overlap()` passes — 0 violations
2. `splits.create_splits()` returns correct counts — train={{expected_train_n}}, val={{expected_val_n}}, test={{expected_test_n}}
3. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0
4. Any task-specific windowing (if applicable) produces the expected segment count
5. Each test {{fold_or_group_label}} processed individually has enough rows for the model's minimum input shape
6. Data loaded from `{{data_cache_dir}}` (not re-downloaded)

## Project Structure

<!-- GENERIC: preserved structure with parameterized paths. -->

```
{{project_name}}/                # project root
  CLAUDE.md                      # this file
  configs/
    project.yaml                 # one-time project config (produced by setup wizard)
    splits.yaml                  # split-protocol parameters
  data/                          # raw data / features
  core/                          # imported from generalized_ml_autoresearch
  {{results_dir}}/
    experiment_log.jsonl
    best_config.json
    dashboard.html
    experiment_summary.md
    research_journal.md
    reasoning_annotations.json
    trade_logs/
      exp<N>_predictions.csv
      exp<N>_prediction_summary.json
    winners/
      <backbone>_exp<N>_<desc>/
        README.md
        config.json
        model_checkpoint.pt (or .joblib)
        code/
        inference/
        reproduction/
        audit_report.md
        colab_train_and_infer.ipynb
  {{memory_dir}}/
    project_autoresearch_checkpoint.md
    project_hardware_log.md
  {{code_versions_dir}}/
    <backbone>_start/
    <backbone>_final/
```

## Key Constants

<!-- GENERIC: preserved. Values filled in from setup answers. -->

| Constant | Value | Location |
|----------|-------|----------|
| PRIMARY_METRIC | {{primary_metric_name}} | composite.py |
| SPLIT_PROTOCOL | {{split_protocol_name}} | splits.yaml |
| N_FOLDS_OR_GROUPS | {{num_folds_or_groups}} | splits.yaml |
| LABEL_HORIZON_BUFFER | {{label_horizon_buffer_units}} | splits.py |
| DEFAULT_LEARNING_RATE | {{default_lr}} | runner.py |
| DEFAULT_BATCH_SIZE | {{default_batch_size}} | runner.py |
| DEFAULT_EPOCHS | {{default_epochs}} | runner.py |
| DEFAULT_PATIENCE | {{default_patience}} | runner.py |
| DEFAULT_WEIGHT_DECAY | {{default_weight_decay}} | runner.py |
| N_EXPERIMENTS_PER_BACKBONE | {{n_experiments_per_backbone}} | CLAUDE.md (this file) |

## Common Mistakes (Never Repeat)

<!--
  GENERIC: source mistakes preserved, FX-specific ones generalized.
  New project-specific mistakes appended via Session Learnings.
-->

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Sliding windows across data gaps (time-series) | Garbage windows, meaningless predictions | Use `create_contiguous_datasets()` or equivalent per-segment splitting |
| Expanding window without hole-punching | Cross-{{fold_or_group_label}} contamination, inflated metrics | `splits.py::split_data()` punches ALL val/test from ALL {{fold_or_group_label}}s |
| Dead config params | Experiments with no effect, wasted compute | Wire every param end-to-end or remove it |
| Data re-downloading every run | Minutes wasted, flaky network dependency | Default `cache_dir={{data_cache_dir}}` |
| Grid sweep instead of diagnostic | Uninformed, 10× more experiments than needed | One change at a time, diagnose results first |
| Running all {{num_folds_or_groups}} {{fold_or_group_label}}s independently per experiment | Slower, unnecessary | Super-fold / union protocol: one train, one eval pass |
| Absolute imports in package | `ModuleNotFoundError` when run as `-m` | Always `from .module import ...` |
| Assuming timing/performance | Wrong estimates, wrong priorities | Measure with `time.time()`, log elapsed |
| Monolithic scripts | Can't debug, can't reuse, can't monitor | Runners log. Dashboard reads. Decoupled. |
| Fine-grained weight-decay sweeps | AdamW decouples wd from grads; tiny changes are no-ops | Use log-spaced sweeps (1e-4, 5e-4, 1e-3, 5e-3) |
| Smaller batch without seed plan | Smaller batch can double seed std | When trying bs < default, always multi-seed before declaring champion |
| Blaming model when problem is {{regime_label_singular}} | Some {{regime_label_plural}} are genuinely hard across all backbones | Don't chase per-{{regime_label_singular}} perfection; aim for ≥ threshold with acceptable std |
| Silently dropping a CLAUDE.md section | Rules drift; future runs violate invariants | Every source section preserved in this template; audit via `SECTION_MAPPING.md` |
| Rewriting composite metric mid-project | Goodhart's Law; meaningless improvement | Metric frozen at setup; changes require RULE_CHANGE entry |

## Session Learnings

<!-- Append-only. New session insights go at the bottom, date-stamped. -->

_Populated by Claude over the course of the project. Every backbone's confirmed optimal config, axes-that-did-not-help, seed-variance study results, and key protocol additions live here. Never delete — only append._

### Initial setup

- **Project created:** {{setup_date}}
- **Task type:** {{task_type}}
- **Primary metric:** {{primary_metric_name}}
- **Split protocol:** {{split_protocol_name}}
- **Backbones in scope:** {{backbones_list}}
