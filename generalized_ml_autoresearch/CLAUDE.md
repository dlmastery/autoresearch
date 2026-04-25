# CLAUDE.md — Project Rules for `generalized_ml_autoresearch` (meta-framework)

> **Scope.** This file governs Claude Code sessions whose CWD is
> `C:/Users/evija/autoresearch/generalized_ml_autoresearch/`. The work here is
> two-fold: (a) **maintain the framework** (templates, core/, dashboard/,
> tests/) and (b) **run the bundled examples** (`examples/regression_house_prices/`,
> `examples/classification_titanic/`, `examples/time_series_airline/`) when
> verifying changes end-to-end.
>
> **Inheritance.** This CLAUDE.md is a filled-in instance of
> [`templates/CLAUDE_template.md`](templates/CLAUDE_template.md), which is itself
> the parameterized generalization of `C:/Users/evija/autoresearch/CLAUDE.md`
> (the FX source of truth). All 52 source sections are present and accounted
> for — see [`templates/SECTION_MAPPING.md`](templates/SECTION_MAPPING.md).
> **Nothing has been silently dropped.** Defaults below mirror the FX project
> where the framework inherits a value, and add a thin meta-framework layer
> on top.

---

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no
separate Python agent. When a session starts:

1. **Read the architecture map:** [`ARCHITECTURE.md`](ARCHITECTURE.md) — the
   order-of-build plan and module I/O contracts.
2. **Read the section coverage audit:**
   [`templates/SECTION_MAPPING.md`](templates/SECTION_MAPPING.md). Audit gate
   must read **52/52 PRESERVED or GENERALIZED, 0 missing**. Any row marked
   `MISSING` is a release blocker.
3. **Read the framework's last checkpoint** (if any framework-level work is in
   progress): `memory/project_autoresearch_checkpoint.md`.
4. **Read the user-facing template:**
   [`templates/CLAUDE_template.md`](templates/CLAUDE_template.md). It is the
   public API of the framework — edits cascade to every downstream project.
5. **Read the SOTA catalog:**
   [`templates/sota_catalog.yaml`](templates/sota_catalog.yaml).
6. **Verify tests are green** before any change:
   ```
   "C:/Users/evija/anaconda3/python.exe" -m pytest generalized_ml_autoresearch/tests -v
   ```
7. **If running an example experiment:** start the dashboard (once per session,
   background):
   ```
   "C:/Users/evija/anaconda3/python.exe" -m http.server 8765 --directory generalized_ml_autoresearch/examples/<task>/autoresearch_results
   ```
   Tell the user: "Dashboard at http://localhost:8765/dashboard.html"
8. **Run an experiment** via:
   ```
   "C:/Users/evija/anaconda3/python.exe" -m generalized_ml_autoresearch.core.runner --config <path>/config.yaml --description "..."
   ```
   Timeout 600 s.
9. **If the user says "continue" or "keep going"** — resume the loop from the
   checkpoint. No need to ask what to do.

---

## Hardware Constraints (MANDATORY — inherited from FX project 2026-04-19)

**E-cores are BANNED.** Same Intel 14th-gen HX system as the FX project. WHEA-Logger
parity errors on CPU APIC IDs 16, 17, 24, 25 (E-cores). System BSODed 4 times
under sustained compute.

- **Use ONLY P-cores**: logical IDs 0-15. Even IDs (0, 2, 4, …, 14) are primary
  threads; odd IDs (1, 3, …, 15) are HT siblings.
- **Default**: 4 P-core threads via `torch.set_num_threads(4)` +
  `cpu_affinity([0, 2, 4, 6])`.
- **GPU does heavy compute**; CPU is coordination only.
- `core/runner.py::_pin_to_safe_cores()` handles this automatically.
- Override with env var `AUTORESEARCH_USE_ALL_CORES=1` (not recommended).
- Override thread count with `AUTORESEARCH_N_THREADS=N`.

**Recorded hardware profile (used by examples and framework dev):**

- GPU VRAM: **16 GB**
- CPU logical cores: **32** (P-cores 0–15 used; E-cores 16–31 banned)
- Cores reserved for the runner: **4** (affinity IDs `[0, 2, 4, 6]`)
- Cores banned: **16, 17, 24, 25** (WHEA parity errors)
- Time budget per experiment: **600 s**
- Max training time per phase: **24 h wall-clock per backbone**

**NEVER run a training loop without the pinning.** If you write a new runner
script, call `_pin_to_safe_cores()` first thing or the laptop will BSOD.

---

## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning,
whichever comes first.** This is the #1 non-negotiable rule. The laptop WILL
crash. Every minute of uncheckpointed work is lost work.

**Checkpoint trigger points (ALL mandatory):**
1. Immediately after every experiment completes — before any analysis.
2. Every 5 minutes during reasoning/analysis — if you've been thinking 3+
   minutes without saving, STOP and checkpoint.
3. Before starting any code change — save current state.
4. After any code change — save the new state and what changed.
5. Before starting the next experiment — checkpoint must contain the exact
   command ready to paste.

What to save to `memory/project_autoresearch_checkpoint.md` (or
`examples/<task>/memory/...` for example runs):

- Current champion config + composite score
- Per-fold primary-metric table for the champion
- Last experiment result (config, composite, per-fold deltas vs champion,
  KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes
- Session start instructions (numbered steps)
- **Full experiment history summary**

Also update `<results_dir>/experiment_summary.md` with the all-experiments
table.

The checkpoint must be **self-contained**. A fresh Claude Code session reading
ONLY `CLAUDE.md` + the checkpoint must be able to resume without reading any
other file.

---

## Mindset (Read First)

You are the maintainer of an autoresearch framework that any top-tier
ML researcher could pick up and extend. You drive the loop: read results,
reason deeply about WHY, cite relevant literature, and decide the next action
based on first-principles understanding of the architecture, data, and
optimization landscape. Never guess. Never grid-search. Before touching code:

1. **Understand the data flow end-to-end.** Trace how a single training sample
   is created, from raw input through features, scaling, batching, to loss
   computation. If you can't explain every step, you don't understand the
   system.
2. **Validate before running.** Run contamination checks, shape assertions, and
   sanity tests before any experiment. A 2-minute verification saves hours of
   garbage results.
3. **Measure, never assume.** If you state a number, it must come from running
   code — not estimation.
4. **When fixing a bug, audit the entire system for the same class of bug.**
   Don't patch one instance and leave three others.
5. **Separation of concerns is not optional.** Runners log. Dashboards display.
   Evaluators evaluate. Never tangle them.

---

## Hard Rules (NEVER violate)

### Data Integrity

- NEVER construct training samples that share an index with any fold's
  validation or test set. Verify with
  `core.evaluation.splits.validate_no_overlap()` — 0 overlap required before
  every run.
- ALWAYS apply the leakage buffer (task-specific, e.g. 10 days for time-series
  with a 5-day forward target) before excluded windows when the target has a
  look-ahead horizon.
- ALWAYS cache downloaded data. Loaders default to a `data_cache/` subfolder.
  NEVER re-download mid-run.
- Load data ONCE at startup. Compute features/targets ONCE. Split ONCE. Reuse
  across all experiments in a loop.

### Evaluation Protocol Invariants

The chosen evaluation protocol is **per-example** (see each example's
`config.yaml`). The framework supports: `holdout`, `kfold`, `stratified_kfold`,
`group_kfold`, `time_series`, `walk_forward`, `super_fold`.

Invariants (enforced by `core/evaluation/splits.py::validate_no_overlap()`):
- Every fold's training data contains ZERO overlap with its own validation or
  test set.
- For super-fold style: train sees no fold's val/test indices.
- Validation set: per protocol (e.g. union of all folds' val windows in
  super-fold).
- Test set: per protocol (e.g. union of all folds' test windows in super-fold).
- **Zero overlap** between train/val/test — verified programmatically before
  every run; output is part of the experiment log.

### Experiment Design

- **Composite metric for keep/revert:** default
  `min(test_primary, val_primary) - 0.1 * n_below_threshold_folds`. The model
  must do well on BOTH val and test across ALL folds.
- Training is EPOCH-BOUND (minimum 20 epochs with early stopping for neural
  nets; iteration-bound with early stopping for GBMs). NOT wall-clock-bound.
- **60-second cooldown after each experiment** to let compute cool (skip if the
  hardware policy says sandbox blocks sleep).
- ONE config change per experiment. Diagnose WHY before choosing what to
  change next.
- Report per-fold breakdown for BOTH val and test alongside aggregates.
- Dashboard shows train/val/test tabs. Test is the default view.
- Every config parameter must be wired end-to-end. Dead params are bugs —
  remove them.
- Every hyperparameter choice must be justified by published papers, model
  developer guidelines, or prior empirical results. Never choose arbitrary
  values.

---

## Autoresearch Agent Protocol (Karpathy-adapted)

1. **Always start from the current best config.** Every experiment modifies ONE
   thing from the best. If it improves, it becomes the new best. If it doesn't,
   revert and try a different direction. Never wander off from the best
   baseline.
2. **If you see consecutive discards, stop and rethink.** Multiple failures
   mean your hypothesis about what to change is wrong. Re-read the per-fold
   results. Look at which folds are weak and WHY. Don't keep guessing.
3. **Explore around the best AND try radical changes.** Most experiments
   should be small tweaks around the champion. But occasionally try something
   bold (different architecture, very different sequence length / feature
   scope) to escape local optima.
4. **Cite your reasoning for every experiment.** "I'm trying X because fold Y
   has [problem] due to Z, and paper W suggests this fix." Not "let me try X
   and see."
5. **The agent never stops.** If out of ideas, research deeper: read the
   relevant SOTA tech reports, adapter papers, domain literature. Think harder.
   Try combining near-misses.
6. **Checkpoint reasoning to memory every few minutes.** The laptop crashes
   often. After every experiment (or every ~3 minutes of reasoning), save the
   current state to `memory/project_autoresearch_checkpoint.md`.
7. **Deep per-fold failure analysis every iteration.** For each fold with a
   below-threshold primary metric, explain WHY: what regime/group it is, what
   conditions, what the uncertainty outputs reveal (high aleatoric = noisy
   data, high epistemic = model doesn't know, low confidence = skip signal).
   Use this to guide the next experiment.
8. **Code changes are allowed.** The agent may modify the Python codebase
   (model architecture, loss function, training loop, features, evaluation)
   if it has a principled reason. Save modified versions to `code_versions/`
   with a version number. Code changes are the most powerful lever — hyperparams
   only go so far.

---

## Research-Driven Experiment Selection (STRICT — no blind sweeps)

The experiment loop is NOT a grid search. It is a research process. Every
single experiment must follow this exact sequence:

**Step 1 — Diagnose the champion's weakness.** Look at per-fold test results.
Which folds are weakest? What regime/group? What do the uncertainty metrics
say? Identify the SPECIFIC failure mode (e.g. "fold 2 post-shift recovery has
low primary metric and high epistemic uncertainty — model hasn't seen enough
of this regime").

**Step 2 — Search the literature.** Examples (task-agnostic):

- Weak on high-variance subgroups → group-aware training, distributionally
  robust optimization (Sagawa et al. 2020 ICLR 'Distributionally Robust
  Neural Networks' arXiv:1911.08731)
- High epistemic in specific folds → data augmentation, deep ensembles
  (Lakshminarayanan, Pritzel, Blundell 2017 NeurIPS 'Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles' arXiv:1612.01474)
- Overfitting to majority class/regime → focal loss (Lin, Goyal, Girshick, He,
  Dollár 2017 ICCV 'Focal Loss for Dense Object Detection' arXiv:1708.02002),
  class-balanced loss (Cui, Jia, Lin, Song, Belongie 2019 CVPR 'Class-Balanced
  Loss Based on Effective Number of Samples' arXiv:1901.05555)
- Architecture ceiling hit → residual connections (He, Zhang, Ren, Sun 2016
  CVPR 'Deep Residual Learning for Image Recognition' arXiv:1512.03385),
  attention (Vaswani et al. 2017 NeurIPS 'Attention Is All You Need'
  arXiv:1706.03762)
- LR too high/low → cyclical LR (Smith 2017 WACV 'Cyclical Learning Rates for
  Training Neural Networks' arXiv:1506.01186), warmup (Goyal et al. 2017 arXiv
  'Accurate, Large Minibatch SGD' arXiv:1706.02677)
- Calibration issue → temperature scaling (Guo, Pleiss, Sun, Weinberger 2017
  ICML 'On Calibration of Modern Neural Networks' arXiv:1706.04599), isotonic
  regression

**Step 3 — Form a hypothesis and predict the outcome.** Write down: "I
hypothesize that [change X] will improve [metric Y] on [fold Z] because
[paper/principle]. I predict composite will move from [current] to approximately
[target]." If you can't write this sentence, you don't understand what you're
doing. Stop and think more.

**Step 4 — Run ONE experiment.** Execute the change. ONE change only.

**Step 5 — Analyze against prediction.** Did the result match? If yes, why? If
no, what does that tell you? Update your understanding.

**Step 6 — Document everything.** Write the full cycle into the experiment log
and checkpoint.

**Step 7 — Checkpoint.** Ritual close: every output file in the "Dashboard
Files Update Mandate" is up to date, then commit the next-experiment command
to the checkpoint.

**The goal is monotonic improvement.** Every experiment should have a
principled reason to believe it will improve composite score. Random guessing
wastes GPU and time. If you're out of ideas for hyperparameters, the answer is
almost always a CODE CHANGE — modify the architecture, loss function, or
feature engineering.

---

## Monotonic Quality Progression (NEVER regress)

The experiment loop must work towards monotonic increase in quality:

- **Never run an experiment you can't justify.**
- **Track the champion lineage.** Document the chain: Exp1 → Exp5 (technique X,
  +ΔY) → Exp10 (tweak Z, +ΔW) → … Each link explains WHY the improvement
  happened.
- **When you hit a plateau, go deeper.** If 3+ consecutive experiments are
  DISCARD, you're in a local optimum. The answer is NOT more hyperparameter
  tweaks — it's a structural change.
- **Protect gains.** When trying bold changes, if composite drops > 2.0,
  investigate WHY before trying the next thing.
- **Quality ratchet:** once a metric improves, treat the new level as the
  floor. If a change improves test primary metric but regresses val below the
  previous champion, it's a DISCARD — both must improve or at least hold.
- **Goodhart protection (MANDATORY):** the agent MAY NOT rewrite the composite
  metric formula, the split protocol, the data integrity invariants, or the
  primary-metric definition mid-project. These are frozen at setup time.
  Changes require an explicit user sign-off (documented in the checkpoint as
  a `RULE_CHANGE` entry). `core/evaluation/composite.py` enforces this with a
  fingerprint hash.

---

## MLOps Documentation Standards (MANDATORY)

You are a strong MLOps engineer. Every artifact and every experiment must be
documented in proper, readable markdown. No exceptions.

**`<results_dir>/experiment_summary.md`** — the master experiment log. Updated
after EVERY experiment. Format:

```markdown
## Experiment Log — [Backbone] Phase

### Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected composite change]
- **Result:** Composite [X] | Test primary [Y] | Val primary [Z] | [N]/[K] folds above threshold
- **Per-fold test:** F1=[X] F2=[X] … F_K=[X]
- **Secondary metrics:** [task-specific]
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned]
- **Per-prediction summary:** [see `trade_logs/exp<N>_predictions.csv`]
```

**`<results_dir>/trade_logs/`** — per-experiment per-prediction detail (see
"Per-Prediction Log" below).

**Key documentation principles:**
1. Readable by a human who wasn't there.
2. No orphan artifacts. Every file referenced from checkpoint, summary, or
   winner README.
3. Consistent formatting. Same table format, same metric names, same precision
   (4 decimal places for ratios, 2 for percentages).
4. Append-only experiment log. Never delete or rewrite entries — add a note
   if an experiment was wrong.

---

## Explainability & Auditability Report (MANDATORY for every NEW BEST)

When a new champion is found, produce a full data-scientist-grade audit to
`<results_dir>/winners/<exp_id>/audit_report.md`. This is not optional — a
model without explainability is un-deployable.

**Required sections (all 14):**

1. **Executive summary** — Champion test primary metric, secondary metrics,
   risk metric, all per-fold metrics. Pass/fail per regime/group.
2. **Feature importance (permutation method)** — For each feature, shuffle that
   column in the test set, re-evaluate, report the drop in test primary
   metric. Rank features by importance. Cite: Breiman 2001 'Random Forests'
   section on variable importance. Save `feature_importance.csv` with columns
   `[feature_name, metric_drop, rank, domain_category]`.
3. **Top-N feature analysis** — For the top 10 most-impactful features, explain
   what they measure, why they matter substantively, and per-fold impact.
4. **SHAP-style local explanations** — For 10 random test predictions, compute
   per-feature contribution. Use gradient × input for neural nets,
   `shap.TreeExplainer` for GBMs. Save as `shap_local.csv`.
5. **Per-fold feature drift** — For each fold, mean/std of each feature vs
   training. Top-5 drifted features (|Z|>2) per fold with explanation.
6. **Calibration analysis** — For regression: predicted-quantile vs realized
   mean (monotonicity, calibration error). For classification: reliability
   diagram, ECE. Cite Guo et al. 2017.
7. **Uncertainty sanity** — Aleatoric vs |error|, confidence vs correctness
   buckets, accuracy per confidence decile. Cite Kendall & Gal 2017.
8. **Per-regime prediction distribution** — Histograms per fold; identify
   systematic bias.
9. **Error attribution / top-N winners & losers** — Top-5 best-predicted and
   top-5 worst per fold. Pattern analysis.
10. **Risk audit** — Task-specific (residual skew/kurtosis for regression;
    per-class error rates for classification; max-drawdown for time-series).
11. **Data pipeline audit** — Reassert zero leakage, purge, embargo, label
    horizon buffer. Rerun `validate_no_overlap()` and include output verbatim.
12. **Model config complete dump** — Every hyperparameter + Python version +
    framework version + numpy version + random seed. For true reproducibility.
13. **Known limitations & risks** — Untested regimes/conditions; likely
    failure modes in production.
14. **Deployment checklist** — Monitoring, kill-switch criterion, retraining
    cadence, task-specific items.

**Implementation:** `core/winner_archive.py::generate_audit_report()` produces
the full report. Runner calls it automatically when `composite > prev_best`.

---

## Winner Definition (CLARIFICATION)

**"Winner" means the GLOBAL champion across ALL backbones and ALL experiments.**
Not per-backbone. The one single best model (by composite) at any point in time.

Per-backbone best is tracked separately in the checkpoint but does NOT get
archived to `winners/` unless it is also the global best.

When a new experiment beats the global composite:
1. Save artifacts to `<results_dir>/winners/<backbone>_exp<N>_<desc>/`
2. Include: README.md, config.json, model_checkpoint.pt (or `.joblib`), code/,
   inference/, reproduction/, audit_report.md (14 sections),
   colab_train_and_infer.ipynb
3. Update `best_config.json` at repo root.

---

## Per-Backbone Code Snapshots (MANDATORY)

Before starting experiments on a new backbone, snapshot the CURRENT
`core/backbones/<backbone>.py`, `core/runner.py`, and any modified training
utilities to `code_versions/<backbone>_start/` so you can diff what changed
during that backbone's exploration.

```
code_versions/
  v1_original/                 # pre-any-change snapshot
  <backbone1>_start/           # snapshot before <backbone1> experiments begin
  <backbone1>_final/           # snapshot after the 50-experiment cycle
  <backbone2>_start/
  ...
```

Rule: never modify code specific to backbone X while experiments on backbone Y
are in progress.

---

## Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)

**Every single experiment MUST have a complete reasoning record in
`<results_dir>/reasoning_annotations.json` keyed by `experiment_num`. No
experiment ships without one. Orphan entries or "auto-backfilled" placeholders
are a bug.**

The entry has these REQUIRED fields (all non-empty strings unless noted):

| Field | Content | Source |
|-------|---------|--------|
| `diagnosis` | Why THIS experiment now: which champion weakness, which fold weakest and why (regime, conditions, uncertainty profile), what prior experiments ruled out | Authored by Claude BEFORE running |
| `citations` | Full author/year/venue per Citation Rigor; multiple papers semicolon-separated | Authored before running |
| `hypothesis` | Concrete mechanism: "parameter X = value Y will change metric Z via mechanism M" | Authored before running |
| `prediction` | Numeric range: "composite from +X to +Y–Z; val fold K from −A to ±B" | Authored before running |
| `verdict` | KEEP / DISCARD / NEAR-MISS + composite + delta vs global best + per-fold narrative | Written immediately after results |
| `learning` | What this updates in the mental model; which axis closed/open; what to try next | Written immediately after results |
| `_manual` | `true` if Claude-authored; `false` only for mechanical variance reruns | Always set |

**Dashboard renders all 7 fields in the detail panel.** Missing/empty/placeholder
strings are a regression — fix before the next experiment.

**Write cadence — two phases per run:**
1. **BEFORE the experiment:** Claude inserts diagnosis/citations/hypothesis/
   prediction with `_manual: true`. The runner's
   `core/reasoning.py::commit_pre_run()` enforces this. The experiment is not
   launched until the entry exists and passes both Citation Rigor and Reasoning
   Blob Completeness validators.
2. **AFTER the experiment:** Claude appends verdict and learning. The runner's
   auto-written fallback only emits `TODO-REWRITE` sentinels with
   `_needs_rewrite: true` — Claude must rewrite.

**Enforcement:** at the start of every experiment cycle, Claude MUST check:
- Does the previous experiment's entry have non-empty verdict/learning? If not,
  write them now.
- Is the next experiment's pre-entry authored? If not, write it now.
- Did `_manual: true` survive any backfill run?

**Parallel write to `research_journal.md`:**

```markdown
## Exp<N> — <short title>
**Diagnosis:** ...
**Citations:** ...
**Hypothesis:** ...
**Prediction:** ...
**Verdict:** ...
**Learning:** ...
```

Journal and JSON must stay in sync. JSON is authoritative if they drift.

**`backfill_reasoning.py` rules:**
- Only runs on DEMAND.
- Never overwrites entries with `_manual: true`.
- Fills only empty fields whose JSONL entry exists.
- Logs every overwrite.
- Is NOT a substitute for authoring annotations before the run.

**Runner's responsibility:**
- Merge CLI flag delta into the runtime entry — without clobbering
  `_manual: true` fields.
- Populate verdict/learning from results as a fallback.
- Never emit content placeholders; use `TODO-REWRITE` + `_needs_rewrite: true`
  if a field can't be computed, and log a warning.

**Why this matters:** the dashboard is shared memory between sessions.

---

## Per-Backbone N-Experiment Mandate (MANDATORY, not optional)

**Every backbone gets a full 50-experiment exploration.** Do not stop early
because "axes look exhausted." The mandate:

1. **50 experiments per backbone** — no fewer. If standard HP sweeps plateau,
   explore architectural variants from arXiv literature through 2026,
   cross-variant combinations, feature engineering changes, multi-seed studies,
   regularization beyond weight decay (label smoothing, mixup, stochastic
   depth).
2. **Research latest SOTA (2024-2026 arXiv papers)** before declaring any
   backbone done. See "Per-Backbone SOTA Training Recipes" below. Update as the
   literature evolves.
3. **Each experiment must cite its paper/source** — no "let me try X". Per
   Autoresearch Agent Protocol rule 4.
4. **Document all 50 in `research_journal.md`** — even DISCARDs. Negative
   results are informative.
5. **Only after 50 experiments** may a backbone be declared "done" and
   progression to the next backbone resume.

For the framework's bundled examples, the budget per backbone may be reduced
(e.g. 5–10) to keep CI runtimes sane, but downstream projects use the full 50.

---

## Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)

**Every backbone picks its OWN epochs, patience, learning rate, batch size,
scheduler, and optimizer from the latest SOTA literature for THAT
architecture. Never copy another backbone's config.**

**Before the first experiment on any new backbone, Claude MUST:**

1. **Pull the latest 2024-2026 paper** for the backbone family. Note:
   recommended epochs and termination, patience, LR (and warmup), scheduler,
   optimizer (Adam / AdamW / Lion / Adafactor / SOAP), batch size (and whether
   effective via grad accumulation), weight decay, gradient clipping, loss.
2. **Record the recipe + paper citation** in the reasoning annotation for
   Experiment 1 of that backbone.
3. **Justify the DELTA from the paper.** Different n? Different domain?
   Different precision? Per Smith 2017 scaling rule: `epochs ≈ paper_epochs ×
   (paper_n / our_n)^0.5`.
4. **Never assume "ep=50 works for everything."** Concrete proof from the FX
   project: LSTM Exp3 (ep=100 pat=15) beat LSTM Exp1 (ep=50 pat=10) by **+0.94
   composite** — wrong epoch count costs 20% of peak performance.

### Backbone-Specific Training Recipes (auto-generated from SOTA catalog)

The full recipe table is in `templates/sota_catalog.yaml`. The framework's
recipes are filtered per task type at setup time. The catalog includes:

#### Tier 1 — classical baselines (required floor for every project)

| Backbone | Task types | Source |
|----------|------------|--------|
| `mlp` | regression, binary-classification, multiclass-classification, time-series-forecasting | Gu, Kelly, Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning' |
| `lstm` | regression, time-series-forecasting | Fischer & Krauss 2018 EJOR 'Deep learning with LSTMs for financial market predictions' |
| `ft_transformer` | regression, classification (tabular) | Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting Deep Learning Models for Tabular Data' (arXiv:2106.11189) |

#### Tier 2 — 2024-2026 SOTA (task-specific, drawn from `sota_catalog.yaml`)

For time-series: PatchTST (Nie et al. 2023 ICLR), iTransformer (Liu et al.
2024 ICLR), TimesFM (Das et al. 2024 ICML), Chronos-Bolt (Ansari et al. 2024
TMLR), Moirai (Woo et al. 2024 ICML), MOMENT (Goswami et al. 2024 ICML), TiRex
(Auer et al. 2025), Sundial (Liu et al. 2025), Time-MoE (Shi et al. 2024
ICLR'25), TimeMixer (Wang et al. 2024 ICLR), TimesNet (Wu et al. 2023 ICLR),
xLSTM (Beck et al. 2024 NeurIPS), Mamba (Gu & Dao 2024 COLM), MambaTS (Cai et
al. 2024 NeurIPS).

For tabular regression/classification: TabPFN (Hollmann et al. 2023 ICLR),
SAINT (Somepalli et al. 2021), NODE (Popov et al. 2020), TabTransformer
(Huang et al. 2020).

#### Tier 3 — gradient boosted machines (each is its OWN backbone, run independently)

GBMs are fundamentally different from neural nets: no epochs, no LR schedule,
no batch. Iterations are tree-count. Each GBM has its own paper, its own
hyperparameter language, its own 50-experiment exploration budget. **Do NOT
bundle xgboost/lightgbm/catboost as "the GBM backbone" — they are three
separate architectures with different splitting algorithms, different
regularization mechanisms, and different category handling.** Explore each
fully.

| Backbone | Key HP | Default Start | Regularization | Special feature | Paper (full citation) |
|----------|--------|---------------|----------------|------------------|----------------------|
| **xgboost** | n_estimators=1500, max_depth=6, lr=0.03, subsample=0.8, colsample_bytree=0.8, early_stop=50 | level-wise trees | reg_lambda=1.0, reg_alpha=0, min_child_weight=1, gamma=0 | 2nd-order Newton boosting; monotonic constraints; histogram method | Chen & Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting System' (arXiv:1603.02754) |
| **lightgbm** | n_estimators=2000, num_leaves=63, lr=0.03, feature_fraction=0.8, bagging_fraction=0.8, early_stop=50 | leaf-wise trees (GOSS) | reg_alpha, reg_lambda, min_data_in_leaf=20 | GOSS; EFB; native categorical | Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017 NeurIPS 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree' |
| **catboost** | iterations=2000, depth=6, lr=0.03, random_strength=1.0, early_stop=100 | symmetric oblivious trees | l2_leaf_reg=3, bagging_temperature=1.0 | Ordered boosting; native ordered target-stat for categoricals | Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin 2018 NeurIPS 'CatBoost: Unbiased Boosting with Categorical Features' (arXiv:1706.09516) |

**Why GBMs are 3 separate backbones:**
- **XGBoost** uses 2nd-order gradient info (Hessian) — effective on imbalanced
  targets, fast on GPU.
- **LightGBM** uses leaf-wise growth + GOSS — fastest wall-clock, handles large
  n well.
- **CatBoost** uses ordered boosting against prediction shift + best default
  categorical handling — slowest but often best out-of-box accuracy on tabular.

Each will rank different features as important and each has different failure
modes. Do not skip any.

**Re-derive for EVERY new SOTA variant.** When a backbone family has multiple
variants, each variant re-derives its recipe from its OWN paper.

---

## GPU Memory Constraint (MANDATORY — 16 GB VRAM hard cap)

**This machine has 16 GB of GPU VRAM. Every backbone selection, every
experiment, every fine-tuning run MUST fit within this budget with headroom. A
model that OOMs mid-training is not a valid experiment — it's a wasted GPU
cycle and a crash risk.**

**Memory budget breakdown (16 GB total):**

| Component | Budget | Notes |
|-----------|--------|-------|
| Model parameters | ≤ 3 GB | FP32 weights; BF16/FP16 halves this |
| Optimizer state (AdamW) | ≤ 6 GB | Adam stores 2 moments at FP32 even with BF16 weights → ≈ 2× param size |
| Gradients | ≤ 3 GB | Same size as params; freed after step |
| Activations | ≤ 3 GB | batch × seq × hidden, scales with bs and depth |
| Reserved / fragmentation | ≥ 1 GB | PyTorch caching allocator overhead |

**Practical parameter ceilings by training mode:**

| Training mode | Max params @ FP32 | Max params @ BF16/FP16 | Max params w/ grad-ckpt + BF16 |
|---------------|-------------------|------------------------|-------------------------------|
| From-scratch train (Adam full states) | ~500 M | ~1.0 B | ~2.0 B |
| Full fine-tune | ~500 M | ~1.0 B | ~2.0 B |
| Parameter-efficient FT (LoRA r=8) | ~1.0 B | ~3.0 B | ~5.0 B |
| Frozen-backbone head-only FT | ~1.5 B | ~4.0 B | ~7.0 B |
| Inference only (no grads) | ~4.0 B | ~8.0 B | ~8.0 B |

**Mandatory pre-flight check for any new backbone.** Before launching
Experiment 1, include in the reasoning annotation:

```
Measured/estimated size: N million params
Training mode selected: [from-scratch | LoRA fine-tune | head-only FT | zero-shot]
Expected peak VRAM: <X> GB at bs=<Y>, seq=<Z>, precision=<FP32|BF16>
Headroom vs 16 GB: <16 - X> GB
Fallback plan if OOM: [reduce bs to 16 | switch to BF16 | gradient checkpointing | adapter-only]
```

Without this entry, Experiment 1 does not launch. The same check applies any
time we change batch size or sequence length during a backbone's cycle.

**Default protocol when adopting a new foundation model:**
1. Start with the SMALLEST published checkpoint of that family.
2. Run zero-shot first — measure composite without any training.
3. If zero-shot is promising, fine-tune (full or PEFT depending on size).
4. Scale up to a larger checkpoint ONLY if smaller shows signal AND memory math
   works.

**Mixed precision note.** BF16 preferred over FP16 — keeps dynamic range without
loss-scaling. Use `torch.autocast(dtype=torch.bfloat16)` + `GradScaler` unset.
LayerNorm/GroupNorm should stay FP32.

**Gradient checkpointing note.** Use `torch.utils.checkpoint.checkpoint_sequential`
for any model > 200 M params being fine-tuned. Costs ~30% more FLOPs but cuts
activation memory by 70-80%, unlocking bs=32 at 500 M – 1 B params.

### Epoch-budget rule of thumb (when in doubt)

If the paper's recipe is unclear:

- **Data scaling (Smith 2017):** `epochs ≈ paper_epochs × (paper_n / our_n)^0.5`.
- **Parameter scaling (Kaplan 2020):** holding data fixed, larger models need
  more epochs. `epochs ≈ base × (our_params / paper_params)^0.2`.
- **Patience as 15% of epochs** is a safe default when papers don't specify.
- **Warmup = 5-10% of total epochs** for transformer families (required by
  layer-norm stability).

These are starting heuristics; always iterate and checkpoint the actual
convergence profile per backbone.

---

## Backbone Isolation Rule

Before starting experiments on a new backbone, snapshot
`core/backbones/<backbone>.py`, `core/runner.py`, and any modified training
utilities to `code_versions/<backbone>_start/`. Do NOT modify backbone code
specific to backbone X while experiments on backbone Y are in progress. Complete
one backbone's 50-experiment cycle, snapshot as `<backbone>_final/`, then move
to the next backbone.

---

## Dashboard Backbone Tabs

Dashboard (`dashboard/dashboard.html`, copied to each project's
`<results_dir>/`) renders a backbone tab bar above the experiment list.
Default view shows "ALL". Tabs filter the scrollable experiment list to just
that backbone's experiments. Click to switch.

---

## Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)

**Every single experiment updates ALL the following files. If any file is
stale after an experiment completes, that's a regression — stop and fix
before moving on. No "I'll batch-update at the end." No "It's just a variance
check."**

**Ownership — who writes what:**

| File | Written by | When | Content |
|------|------------|------|---------|
| `<results_dir>/experiment_log.jsonl` | **runner (auto)** | every run, appended | full metrics: composite, test/val/train primary metric, per-fold results, per-window secondary metrics, uncertainty, timing, config |
| `<results_dir>/best_config.json` | **runner (auto)** | only when new GLOBAL champion | overwritten with full champion entry |
| `<results_dir>/best_model.pt` | **runner (auto)** | only when new GLOBAL champion | weights + scaler + config + feature_columns + provenance (or `.joblib` for sklearn/GBM) |
| `<results_dir>/trade_logs/exp<N>_predictions.csv` | **runner (auto)** | every run | one row per test prediction |
| `<results_dir>/trade_logs/exp<N>_prediction_summary.json` | **runner (auto)** | every run | per-fold totals, correct/wrong, avg metric, confidence-stratified accuracy |
| `<results_dir>/reasoning_annotations.json` | **Claude BEFORE run + runner AFTER run** | every run, two-phase | diagnosis, citations, hypothesis, prediction (Claude); verdict, learning (runner fallback, Claude overrides) |
| `<results_dir>/research_journal.md` | **Claude** | every run, appended | markdown narrative of the 7-step process |
| `<results_dir>/experiment_summary.md` | **Claude** | every run, appended | short tabular entry per experiment |
| `memory/project_autoresearch_checkpoint.md` | **Claude** | every run | update champion, history table, next-command block |
| `<results_dir>/winners/<backbone>_exp<N>_<desc>/*` | **Claude + runner** | only when new GLOBAL champion | README.md, config.json, model_checkpoint (copy), code/ snapshot, inference/predict.py, per_fold_results.json, experiment_log_entry.json |
| `<results_dir>/winners/<backbone>_exp<N>_<desc>/audit_report.md` | **Claude (via `winner_archive.py`)** | only when new GLOBAL champion | 14-section audit per Explainability rules |
| `<results_dir>/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb` | **Claude (via `winner_archive.py`)** | only when new GLOBAL champion | self-contained Colab notebook |
| `<results_dir>/dashboard.html` | **Claude (rarely)** | only when adding a new metric/tab | static HTML — reads JSONL + annotations live |

**Per-experiment ritual (repeat in order, every single run):**

1. **Before launch:** open `reasoning_annotations.json`, insert a new entry
   keyed by the upcoming `experiment_num` with `diagnosis`, `citations`,
   `hypothesis`, `prediction`, `_manual: true`. If this entry isn't there, the
   experiment doesn't run. `core/reasoning.py` enforces this with the Citation
   Rigor + Reasoning Blob Completeness validators.
2. **Before launch:** append a matching section to `research_journal.md`.
3. **Launch:** run the CLI command.
4. **Runner auto-updates:** JSONL, best_config (if champion), best_model (if
   champion), trade_logs CSV + JSON, reasoning_annotations verdict/learning
   fallback.
5. **After completion:** Claude reads the runner output, overwrites verdict
   and learning with richer analysis (per-fold narrative, which regimes
   won/lost, uncertainty profile). Updates the corresponding journal section.
6. **After completion:** Claude appends a row to `experiment_summary.md`.
7. **After completion:** Claude updates the checkpoint.
8. **If new champion:** Claude archives to `winners/<backbone>_exp<N>_<desc>/`.

**Verification at the start of every experiment cycle:**

Before launching Experiment N+1, confirm all of these are CURRENT for Experiment N:

- [ ] `experiment_log.jsonl` has an entry for N (runner writes, verify)
- [ ] `reasoning_annotations.json[N]` has all 7 fields non-empty and non-placeholder
- [ ] `research_journal.md` has a section for N
- [ ] `experiment_summary.md` has a row for N
- [ ] `memory/project_autoresearch_checkpoint.md` references N in its history table
- [ ] `trade_logs/expN_predictions.csv` and `expN_prediction_summary.json` exist
- [ ] If N set a new champion: `winners/<backbone>_expN_<desc>/` exists with all required files

If ANY checkbox is unchecked, stop and fix BEFORE launching N+1.

**Placeholder strings are a bug.** The runner refuses to fabricate pre-run
content. If a pre-run entry is missing, the runner inserts `"TODO-REWRITE"`
sentinel values and a `_needs_rewrite: true` flag — Claude MUST rewrite those
entries before launching the next experiment.

---

## Citation Rigor (MANDATORY format for `citations` field)

**Every citation string MUST contain, for every paper referenced:**

1. **All authors' surnames** (not just first-author et al. unless > 6 authors)
2. **Year** of publication
3. **Venue** — journal name, conference abbreviation (NeurIPS, ICML, ICLR,
   AAAI, CVPR, KDD, etc.), or `arXiv` if preprint-only
4. **Full paper title** in single quotes
5. **arXiv ID** in the form `(arXiv:XXXX.YYYYY)` if available — mandatory for
   any paper posted to arXiv
6. **One-sentence relevance note** — why this paper motivates THIS experiment
   specifically

**Format template:**

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) — one-sentence note on why we cite it here.
```

**Multiple papers separated by semicolons + linebreak.** Minimum one primary
citation per experiment; secondary citations encouraged when combining ideas.

**Examples of GOOD citations (copy this style):**

> Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 ICLR 'On Large-Batch
> Training for Deep Learning: Generalization Gap and Sharp Minima'
> (arXiv:1609.04836) — motivates bs=16 as a flat-minima probe.

> Loshchilov & Hutter 2019 ICLR 'Decoupled Weight Decay Regularization'
> (arXiv:1711.05101) — AdamW wd acts as decoupled weight shrinkage, so
> perturbations must be log-scale.

> Nie, Nguyen, Sinthong, Kalagnanam 2023 ICLR 'A Time Series is Worth 64
> Words' (arXiv:2211.14730) — requires seq_len ≥ 60 for attention heads to
> have enough patches.

**Examples of BAD citations (REJECTED — rewrite required):**

- `"Keskar 2017 flat minima"` — missing coauthors, venue, title, arXiv,
  relevance note
- `"(Keskar2017)"` — parenthetical tag only, useless
- `"Keskar et al."` — no year, no venue
- `"arxiv paper on batch size"` — no attribution
- `"(no citation tag)"` — confesses the author didn't do the work
- `"see research_journal.md"` — redirects instead of citing

**The goal:** anyone (including a future Claude Code session with zero project
context) must be able to open the dashboard, click a row, read the `citations`
field, and immediately know which paper to read and why. Citations are
institutional memory.

**Arxiv ID lookup discipline.** If you know the paper but not its arXiv ID,
fetch it via WebSearch or WebFetch (arxiv.org/abs search) before writing the
entry. Authoring a citation without the arXiv ID is a partial job.

---

## Reasoning Blob Completeness (what "full reasoning" means)

Each of the 7 fields in `reasoning_annotations.json` has a minimum content
spec. Entries below this spec must be rewritten.
`core/reasoning.py::validate_reasoning_blob()` enforces these floors.

| Field | Minimum content | Word count floor | Must include |
|-------|-----------------|------------------|--------------|
| `diagnosis` | Why THIS experiment NOW; which champion weakness; which fold is worst and why | ≥ 60 words | Reference to at least one prior experiment by number OR a per-fold metric from the current champion |
| `citations` | Per Citation Rigor spec | ≥ 40 words for single paper, ≥ 80 for multi-paper | Author list + year + venue + title + arXiv ID + relevance note for each paper |
| `hypothesis` | Mechanism: what parameter(s) move, what they do, what the cited paper predicts | ≥ 50 words | The word "mechanism" or "because" or "per [paper]"; the specific parameter and value |
| `prediction` | Concrete numeric range on composite AND at least one sub-metric | ≥ 25 words | A numeric range; a direction for at least one sub-metric |
| `verdict` | KEEP/DISCARD/NEAR-MISS + exact composite + delta vs global best + per-fold narrative | ≥ 30 words | Status label; composite to 4 decimals; mention of at least one per-fold result |
| `learning` | What this updates in the mental model; which axis closed/open; what to try next | ≥ 40 words | "Axis closed" / "axis open" language OR a concrete "next try: ..." |
| `_manual` | Boolean | — | `true` if Claude-authored; `false` only for mechanical reruns |

**When running a batch of variance checks** (same config, varying seed), the
`_manual: true` entries can share templated `diagnosis` and `citations` across
runs, but `verdict`/`learning` must always be per-run-specific.

**Batch updates are forbidden.** Don't do 5 experiments then update the
journal/summary/checkpoint in one go — each experiment's state gets stale and
crash-recovery breaks.

---

## Loss Function Rules

The default loss is task-specific:

- **Regression:** Huber (δ=1) by default; switch to MSE if residual scale is
  small enough that Huber is effectively MSE; switch to Quantile loss if the
  user wants distributional output.
- **Binary classification:** BCEWithLogitsLoss with class-weighting if
  imbalanced (`pos_weight = n_neg / n_pos`).
- **Multiclass classification:** CrossEntropyLoss with `label_smoothing=0.1`
  default (Müller, Kornblith, Hinton 2019 NeurIPS 'When does label smoothing
  help?' arXiv:1906.02629).
- **Time-series forecasting:** MSE for short horizon; Quantile / Pinball for
  probabilistic; flow-matching for foundation models that support it.
- **Ranking:** ListMLE or NDCG-loss surrogate (per the task's evaluation
  metric).
- **Survival:** negative partial log-likelihood (Cox) or NLL on parametric
  distributions.

### Heteroscedastic / uncertainty-aware loss (neural regression & time-series)

If using uncertainty-aware loss (mean + log-variance per prediction), loss =
`exp(-s) * base_loss(mu, y) + 0.5 * s`:

- **Variance-branch dominance is the #1 failure mode.** If aleatoric > 0.2 of
  the target scale, the model is copping out. Fix: higher LR, more epochs, or
  clamp log_var.
- **Optimal aleatoric range: 0.05–0.15 of target scale.** Below 0.05 =
  overconfident. Above 0.20 = lazy variance.
- **Het-loss needs ~50% more epochs than plain base loss.**
- **LR sweet spot shifts up slightly** — `exp(-s)` weighting reduces effective
  gradient on mean.
- **Monitor uncertainty per fold:** high aleatoric = noisy data correctly
  identified; high epistemic = model needs more data from that regime. Use
  `confidence < 0.8` as a "skip prediction" signal.

Cite: Kendall & Gal 2017 NeurIPS 'What Uncertainties Do We Need in Bayesian
Deep Learning for Computer Vision?' (arXiv:1703.04977).

---

## Winner Archiving Protocol (MANDATORY for every NEW BEST)

Every time a new champion is found (status=KEEP and composite > previous best),
archive ALL artifacts to a self-contained subdirectory. The archive must be
fully portable.

**Directory structure:** `<results_dir>/winners/<backbone>_exp<N>_<short_description>/`

```
winners/
  <backbone>_exp<N>_<desc>/
    README.md                    # Full description (template below)
    config.json                  # Exact config that produced this winner
    model_checkpoint.pt          # Saved weights (or .joblib for sklearn/GBM)
    experiment_log_entry.json    # The JSONL entry for this experiment
    per_fold_results.json        # Full per-fold val + test breakdown
    code/                        # Frozen snapshot of ALL source code at win time
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
- Champion composite, test primary metric, val primary metric
- Per-fold test primary metric table
- Per-fold val primary metric table
- Full hyperparameter config
- Architecture description
- Key insight: WHY this config won (what change from previous champion)
- Training details: epochs/iterations run, early stopping epoch, training time
- Uncertainty metrics per fold: aleatoric, epistemic, confidence
- Task-specific secondary metrics
- Reproduction status: seeds tested, variance observed
- Sample inference code snippet
- **Deployment Strategy** section (task-conditional, see below)

**After archiving:** Rerun the winner to verify reproduction. The reproduction
log goes into `reproduction/reproduce_log.txt`. If composite differs by > 0.5,
flag and investigate before proceeding.

**Model checkpoint MUST be portable and self-contained:**

For neural nets (`.pt`): include in the torch.save dict:
- `model_state_dict`
- `config` — hyperparameters dict
- `scaler_mean`, `scaler_scale` — StandardScaler parameters
- `feature_columns` — list of feature names in order
- `target_columns` — list of target names
- `n_features` — int
- `composite`, `description`, `backbone`, `experiment_num` — provenance

For sklearn/GBM (`.joblib`): pickle the fitted `Backbone` wrapper.

**The `predict.py` inference script must:**
1. Load the model checkpoint.
2. Accept raw feature input (or a dataset path).
3. Output: prediction, confidence, aleatoric uncertainty, epistemic uncertainty
   (when applicable).
4. Include a `__main__` block with a working example.
5. Print results in a clear table format.

**Deployment Strategy section (MANDATORY in every winner README.md):**

Task-conditional content. Generic template (each example fills in):

1. Signal generation — inputs, outputs, MC Dropout usage
2. Decision rules / thresholds
3. Resource sizing / cost
4. Exit / refresh / retraining cadence
5. Per-regime performance table
6. Risk controls / monitoring
7. Expected performance (metric estimates pre/post cost)
8. Caveats and warnings (seed variance, distribution shift, feature
   dependencies)
9. Reference to inference code

---

## Google Colab Notebook (MANDATORY for every winner)

For every archived winner, generate a self-contained Google Colab notebook at
`<results_dir>/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb`.

**Required cells:**

1. **Setup cell:** `!pip install` all dependencies.
2. **Data cell:** load the project's dataset (bundled CSV preferred over network
   download for reproducibility).
3. **Feature engineering cell:** compute the project's features with clear
   explanations.
4. **Training cell:** full training loop reproducing the winner config exactly
   — including split, contiguous datasets, loss, optimizer, early stopping.
   Print per-epoch loss + validation metrics.
5. **Evaluation cell:** evaluate on all test folds, print per-fold metric
   table, compute composite score.
6. **Inference cell:** load trained model, accept a sample input, produce
   predictions with uncertainty bands. Show a sample prediction table.
7. **Visualization cell:** plot per-fold metric curves, prediction vs actual
   scatter (regression) / confusion matrix (classification), uncertainty
   calibration.
8. **Export cell:** save model weights + config for deployment.

**Notebook principles:**
- Every cell has a markdown header explaining what it does and WHY.
- Include the champion config as a clearly visible dict at the top.
- Use `torch.manual_seed()` and `np.random.seed()` for reproducibility.
- Print all key metrics at the end in a summary table.
- Target runtime: < 5 minutes on Colab free tier.
- The notebook must be SELF-CONTAINED — no imports from the main package.

---

## Traditional ML Metrics (MANDATORY for every experiment)

In addition to the primary composite metric, compute and log task-type
secondary metrics for every experiment.

**For task = regression:** RMSE, MAE, R², MAPE, residual skew/kurtosis, max
absolute error.

**For task = binary-classification:** AUROC, AUPRC, accuracy, precision,
recall, F1, F2 (recall-weighted), MCC, ECE, confusion matrix counts (TP, FP,
TN, FN).

**For task = multiclass-classification:** macro-F1, micro-F1, weighted-F1,
top-k accuracy, per-class AUROC, per-class precision/recall, confusion matrix.

**For task = time-series-forecasting:** RMSE, MAE, MAPE, sMAPE, Pinball loss
(if quantile output), CRPS (if probabilistic), directional-accuracy.

**For task = ranking:** NDCG@k, MAP, MRR, Spearman.

**For task = survival:** concordance index (C-index), Brier score, integrated
Brier score.

These must appear in:
1. `core/evaluation/metrics.py::full_report()` output
2. Per-fold results in JSONL log entries
3. Dashboard per-fold tables
4. Winner archive `per_fold_results.json`
5. Experiment summary markdown

---

## Per-Prediction Log (MANDATORY for every experiment)

For EVERY experiment, produce a per-prediction CSV on test data. This is
critical for understanding WHERE the model makes and loses value.

**Output file:** `<results_dir>/trade_logs/exp<N>_predictions.csv`

**Generic columns (present for every task):**

| Column | Description |
|--------|-------------|
| index | Row index in test data (or date/timestamp if time-series) |
| fold | Which test fold |
| regime | Regime/group label |
| prediction | Model point prediction |
| actual | Ground-truth label |
| confidence | Model confidence (1 − epistemic, when available) |
| aleatoric | Aleatoric uncertainty |
| epistemic | Epistemic uncertainty |
| primary_metric_value | Per-row contribution to the primary metric |

**Task-specific columns** (added per task):

- regression: `error`, `abs_error`, `pct_error`
- binary-classification: `prob_positive`, `pred_label`, `correct`
- multiclass: `top1_pred`, `top1_prob`, `top3_correct`
- time-series: `horizon`, `quantile_10`, `quantile_50`, `quantile_90`,
  `directional_correct`
- ranking: `query_id`, `rank_pred`, `rank_true`, `dcg_contribution`
- survival: `time_to_event`, `event_indicator`, `risk_score`

**Per-fold summary in `exp<N>_prediction_summary.json`:**
- Total predictions, correct/wrong counts per fold
- Average winning magnitude, average losing magnitude
- Largest single correct, largest single wrong
- Streak analysis: max consecutive correct, max consecutive wrong
- Confidence-stratified accuracy: > 0.9 vs < 0.9

**This data enables:** identifying specific examples/regimes where the model
fails; confidence calibration analysis; downstream decision-rule research
(threshold tuning, filter rules).

---

## Architecture

- **Autoresearch loop = Claude agent.** Claude reads results, decides what to
  try, calls the runner, reads output. The intelligence is in the agent, NOT
  in Python code. No pre-baked experiment lists.
- Runner (`core/runner.py`) executes ONE experiment per call. Logs JSONL.
  That's it.
- Dashboard (`dashboard/dashboard.html`) reads logs. DECOUPLED from runner.
- Save checkpoint after every experiment (JSONL append + best_config.json
  overwrite).
- Use relative imports (`from .backbones import ...`).

---

## Validation Checklist (Run Before Every Experiment Session)

1. `core.evaluation.splits.validate_no_overlap()` passes — 0 violations.
2. `splits.create_splits()` returns expected counts (per the example's
   `config.yaml`).
3. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0.
4. Any task-specific windowing (if applicable) produces the expected segment
   count.
5. Each test fold processed individually has enough rows for the model's
   minimum input shape.
6. Data loaded from the configured `data_cache/` (not re-downloaded).

---

## Project Structure

```
generalized_ml_autoresearch/        # framework root
  CLAUDE.md                         # this file
  README.md                         # user guide
  ARCHITECTURE.md                   # order-of-build, I/O contracts
  core/
    __init__.py
    runner.py                       # one experiment per invocation
    reasoning.py                    # Citation Rigor + Reasoning Blob validators
    checkpoint.py                   # crash-recovery checkpoint manager
    winner_archive.py               # 14-section audit + Colab generator
    backbones/
      base.py                       # Backbone ABC
      registry.py                   # @register_backbone decorator
      mlp.py
      lstm.py
      gbm.py                        # 3 separate registry entries
      tabular_transformer.py        # FT-Transformer
      foundation_models.py          # stubs: TimesFM/Chronos/MOMENT/Moirai/...
    evaluation/
      splits.py                     # 7 split protocols + validate_no_overlap()
      metrics.py                    # registry + per-task metrics
      composite.py                  # frozen-fingerprint composite metric
      uncertainty.py                # MC Dropout, deep ensembles, quantile
  templates/
    CLAUDE_template.md              # parameterized 52-section template
    SECTION_MAPPING.md              # source heading → target heading audit
    sota_catalog.yaml               # curated 2024-2026 SOTA recipes
  skills/
    ml-autoresearch-setup/SKILL.md  # 12-step setup wizard
  dashboard/
    dashboard.html                  # static HTML+JS, reads JSONL
  examples/
    regression_house_prices/
    classification_titanic/
    time_series_airline/
  tests/
    test_smoke.py
    test_runner_e2e.py
    test_section_coverage.py
```

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| PRIMARY_METRIC | per-task (rmse / auc / mae / ndcg / cindex) | each example's `composite.py` config |
| SPLIT_PROTOCOL | per-task (kfold / stratified / time_series) | each example's `config.yaml` |
| N_FOLDS_OR_GROUPS | per-task (5 default for k-fold; 7 for super-fold) | each example's `config.yaml` |
| LABEL_HORIZON_BUFFER | 10 (time-series default) | `splits.py` |
| DEFAULT_LEARNING_RATE | 3e-4 | `runner.py` |
| DEFAULT_BATCH_SIZE | 32 | `runner.py` |
| DEFAULT_EPOCHS | 50 | `runner.py` |
| DEFAULT_PATIENCE | 10 | `runner.py` |
| DEFAULT_WEIGHT_DECAY | 1e-5 | `runner.py` |
| N_EXPERIMENTS_PER_BACKBONE | 50 (downstream); 5–10 (CI) | this file |
| GPU_VRAM_GB | 16 | `core/runner.py` (pre-flight check) |
| CPU_RUNNER_AFFINITY | [0, 2, 4, 6] | `runner.py::_pin_to_safe_cores()` |
| BANNED_CORES | [16, 17, 24, 25] | hardware policy |

---

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Sliding windows across data gaps (time-series) | Garbage windows, meaningless predictions | Use `create_contiguous_datasets()` or equivalent per-segment splitting |
| Expanding window without hole-punching | Cross-fold contamination, inflated metrics | `splits.py::split_data()` punches ALL val/test from ALL folds |
| Dead config params | Experiments with no effect, wasted compute | Wire every param end-to-end or remove it |
| Data re-downloading every run | Minutes wasted, flaky network dependency | Default `cache_dir=data_cache/` |
| Grid sweep instead of diagnostic | Uninformed, 10× more experiments than needed | One change at a time, diagnose results first |
| Running all folds independently per experiment | Slower, unnecessary | Super-fold / union protocol: one train, one eval pass |
| Absolute imports in package | `ModuleNotFoundError` when run as `-m` | Always `from .module import ...` |
| Assuming timing/performance | Wrong estimates, wrong priorities | Measure with `time.time()`, log elapsed |
| Monolithic scripts | Can't debug, can't reuse, can't monitor | Runners log. Dashboard reads. Decoupled. |
| Fine-grained weight-decay sweeps | AdamW decouples wd from grads; tiny changes are no-ops | Use log-spaced sweeps (1e-4, 5e-4, 1e-3, 5e-3) |
| Smaller batch without seed plan | bs<default can double seed std | Always multi-seed before declaring champion |
| Blaming model when problem is regime | Some regimes are genuinely hard across all backbones | Don't chase per-regime perfection; aim for ≥ threshold with acceptable std |
| Silently dropping a CLAUDE.md section | Rules drift; future runs violate invariants | Every source section preserved in this template; audit via `SECTION_MAPPING.md` |
| Rewriting composite metric mid-project | Goodhart's Law; meaningless improvement | Metric frozen at setup; changes require RULE_CHANGE entry |
| Editing `CLAUDE_template.md` without updating `SECTION_MAPPING.md` | Audit gate fails; downstream projects drift | Update mapping in same commit; CI runs `test_section_coverage.py` |
| Lowering a Reasoning Blob word-count floor | Project drift toward "just try X" guessing | Floors only go up; require RULE_CHANGE entry |
| Adding a runner bypass without `_needs_rewrite: true` | Citation discipline collapses | Bypasses always tag the entry as needing rewrite |
| Bundling xgboost+lightgbm+catboost | Tier-3 rule violated | Three separate registry entries, three separate budgets |
| Modifying `dashboard.html` to write files | Decoupling broken | Dashboard reads only |
| Skipping a backbone's pre-flight VRAM check | OOM at experiment 1 | Runner requires the VRAM block in the reasoning annotation |

---

## Session Learnings

_Append-only. New session insights go at the bottom, date-stamped. Never delete
— only append._

### Initial setup

- **Project created:** 2026-04-19 (framework v1.0)
- **CLAUDE.md added:** 2026-04-24 (this file — closes the missing-CLAUDE.md gap
  flagged when bootstrapping the `AUTORESEARCHIMAGE` (WILDS-Camelyon17) sister
  project)
- **Task type:** meta — varies per bundled example
- **Primary metric:** per-task (RMSE for regression, AUROC for classification,
  MAE for time-series)
- **Split protocol:** per-task (k-fold for regression/classification,
  time-series for forecasting)
- **Backbones in scope:** mlp, lstm, ft_transformer, xgboost, lightgbm,
  catboost (Tier 1+3), plus stubs for TimesFM, Chronos, MOMENT, Moirai, TiRex,
  Sundial, Time-MoE (Tier 2 foundation models, time-series only)

### Inherited from FX project (parent)

- **Residual skip connection** drove a 5× improvement (Sharpe +0.82 → +4.77) on
  the FX MLP — confirms the principle that low-capacity nonlinear branches
  should learn corrections to a linear baseline rather than fight it.
- **Capacity reduction** (512 → 128 hidden units) improved generalization
  (params/sample 428 → 121) — confirms simplicity bias on low-signal data.
- **Heteroscedastic loss is a liability on small datasets** (< 3k samples) —
  variance branch steals capacity. Default to plain Huber unless n is large.
- **Seed variance dominates architecture variance for similarly-sized models**
  — declare champions only after a 3-seed median check.
- **From-scratch MLPs need 50 epochs** with cosine annealing; foundation-model
  fine-tunes need 20 epochs (paper-derived, not transferable).
- **Foundation models are not free wins on low-n problems** — LFM2-350M
  underperformed a 300k-param MLP on FX (median Sharpe +1.40 vs +4.76).

### Framework-development meta-rules (this CLAUDE.md only)

- **No section is silently dropped.** Every source CLAUDE.md heading maps to
  a row in `SECTION_MAPPING.md`. CI test enforces.
- **Strengthening is allowed; weakening is not.** Goodhart-protection clauses
  may be added; Citation Rigor / Reasoning Blob floors may go up; Tier-3
  GBM-as-three-backbones rule cannot be relaxed.
- **The runner has no bypass.** `--bypass-reasoning-gate` exists for
  emergencies but always tags the entry as `_needs_rewrite: true` and surfaces
  red in the dashboard.
- **Three examples are the smoke test.** Every framework PR must keep
  regression / classification / time-series examples end-to-end-green.
- **The 7-step process applies to framework changes too.** Diagnose user
  pain → cite design pattern → hypothesize fix → predict success criterion →
  one PR per change → analyze post-merge → checkpoint via CHANGELOG /
  release notes.

---

## Cross-references

| Document | Purpose |
|----------|---------|
| `C:/Users/evija/autoresearch/CLAUDE.md` | The original FX-project CLAUDE.md. Source of truth for every section. |
| `templates/CLAUDE_template.md` | Parameterized version. Public API of the framework. |
| `templates/SECTION_MAPPING.md` | Audit log: source heading → target heading + status. 52/52 must be PRESERVED or GENERALIZED. |
| `templates/sota_catalog.yaml` | Curated 2024-2026 SOTA recipes by family + task type. |
| `ARCHITECTURE.md` | Order-of-build, module I/O, invariants. |
| `README.md` | User-facing quick-start + comparison table. |
| `skills/ml-autoresearch-setup/SKILL.md` | The 12-step setup wizard. |
| `tests/test_section_coverage.py` | The 52-section audit gate. |

---

## License

Inherits the parent `autoresearch` repository's MIT license.

## Credits

- FX AutoResearch methodology (the source CLAUDE.md) — Evija Ranti.
- Generalized framework — Claude (hierarchical coordinator), 2026-04-19.
- This filled-in CLAUDE.md — Claude, 2026-04-24, closing the missing-CLAUDE.md
  gap flagged during the WILDS-Camelyon17 sister-project bootstrap.
