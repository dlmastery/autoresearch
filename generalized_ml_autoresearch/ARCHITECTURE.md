# Architecture — Generalized ML AutoResearch

## Purpose

This document is the order-of-build plan and the I/O contract map. Anyone (the Queen coordinator, a new contributor, or a fresh Claude Code session) should be able to read this file and understand which module calls which, what each module writes to disk, and what invariants cross module boundaries.

## High-level diagram

```
                          USER (via slash command /ml-autoresearch-setup)
                                          |
                                          v
                   skills/ml-autoresearch-setup/SKILL.md  (12-step wizard)
                                          |
                                          v
       templates/CLAUDE_template.md + templates/runner_template.py
                                          |
                                 [generates]
                                          v
              <project_root>/CLAUDE.md, <project_root>/configs/*.yaml,
              <project_root>/memory/*, <project_root>/autoresearch_results/
                                          |
                                          v
                            core/runner.py  (one experiment per invocation)
                                          |
                 +------------------------+------------------------+
                 |                        |                        |
                 v                        v                        v
     core/backbones/*.py    core/evaluation/*.py        core/reasoning.py
    (Backbone interface,   (splits, metrics, composite, (validates citations +
     MLP/LSTM/GBM/Tx)      uncertainty)                  reasoning floors)
                 |                        |                        |
                 +------------------------+------------------------+
                                          |
                                          v
                              writes JSONL + CSV + JSON
                                          |
                                          v
           core/checkpoint.py (append history, refresh next-command)
                                          |
                                          v
           core/winner_archive.py (only when new global champion)
                                          |
                                          v
           dashboard/dashboard.html (static — reads JSONL + annotations)
```

## Module boundaries and I/O contracts

### `core/runner.py`

- **Input**: config dict (or CLI flags mapped to dict) describing one experiment (backbone name, hyperparameters, split protocol name, metric name, epochs, patience, seed, description).
- **Output**:
  - appended row in `autoresearch_results/experiment_log.jsonl`
  - appended row in `autoresearch_results/trade_logs/exp<N>_predictions.csv` (per-prediction log; generalized from trade log)
  - `autoresearch_results/trade_logs/exp<N>_prediction_summary.json`
  - auto-fallback entry in `autoresearch_results/reasoning_annotations.json` if a pre-run entry is missing (inserts `TODO-REWRITE` sentinels so the human knows to rewrite — never fakes reasoning)
  - only on GLOBAL champion: `autoresearch_results/best_config.json`, `autoresearch_results/best_model.pt`
- **Invariants enforced**:
  - refuses to launch if the pre-run reasoning annotation is missing OR fails the Citation Rigor + Reasoning Blob Completeness validators in `core/reasoning.py`
  - refuses to launch if the split protocol produces any train/val/test overlap
  - refuses to launch if the pre-flight GPU memory check (for neural backbones) was not logged in the reasoning annotation
- **One change per experiment**: the runner itself doesn't enforce this — Claude does, per the Research-Driven Experiment Selection rule. But the runner does log the full config so diffing is trivial.

### `core/backbones/`

- **`base.py`** defines the abstract `Backbone` class:
  ```python
  class Backbone(ABC):
      name: ClassVar[str]
      task_types: ClassVar[set[str]]  # {"regression", "binary-classification", ...}

      def build(self, config: dict, input_shape, n_outputs) -> None: ...
      def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict: ...
      def predict_with_uncertainty(self, X, n_samples=30) -> PredictionBundle: ...
      def save(self, path: str | Path) -> None: ...
      @classmethod
      def load(cls, path: str | Path) -> "Backbone": ...
      def gpu_memory_estimate_mb(self, batch_size: int) -> float: ...
  ```
- **`registry.py`** maintains the `BACKBONE_REGISTRY` dict and the `@register_backbone` decorator. `create_model(name, config)` is the factory used by the runner.
- Neural backbones (`mlp.py`, `lstm.py`, `tabular_transformer.py`) use PyTorch. GBM backbones (`gbm.py`) wrap XGBoost / LightGBM / CatBoost as **three separate** registry entries.
- `foundation_models.py` is a stub wiring table — users can add TimesFM, Chronos, MOMENT, etc.

### `core/evaluation/`

- **`splits.py`**: `HoldoutSplit`, `KFoldSplit`, `StratifiedKFoldSplit`, `GroupKFoldSplit`, `TimeSeriesSplit`, `WalkForwardSplit`, `SuperFoldSplit`. All implement `split(df) -> list[FoldAssignment]` returning indexer objects. Include `validate_no_overlap()` assertion.
- **`metrics.py`**: pluggable registry of named metric functions (regression, classification, ranking, survival). Every metric receives `(y_true, y_pred, uncertainty=None, groups=None)`.
- **`composite.py`**: generic composite-metric calculator. Default formula: `min(val_primary, test_primary) - lambda * n_below_threshold_folds`. User-specified formulas accepted as a small expression DSL or a Python callable path.
- **`uncertainty.py`**: MC Dropout + deep-ensemble for neural nets; quantile regression for trees. Returns `(mean, aleatoric, epistemic, confidence)`.

### `core/reasoning.py`

- `ReasoningEntry` pydantic model with 7 fields (diagnosis, citations, hypothesis, prediction, verdict, learning, _manual).
- `validate_citation_rigor(citations_text) -> list[str]` returns violations.
- `validate_reasoning_blob(entry) -> list[str]` returns violations (word-count floors, required keywords).
- `commit_pre_run(exp_num, entry)` writes to `reasoning_annotations.json` after passing validation.
- `commit_post_run(exp_num, verdict, learning)` appends runner verdict and Claude's learning.

### `core/checkpoint.py`

- `CheckpointManager` handles:
  - `refresh(experiment_record, next_command)` — overwrites `memory/project_autoresearch_checkpoint.md`.
  - `append_to_summary(experiment_record)` — appends a row to `autoresearch_results/experiment_summary.md`.
  - `reasoning_time_checkpoint(current_thinking)` — 5-minute rule checkpoint.
- Checkpoint file format is human-readable markdown, self-contained: champion snapshot, last experiment, exact next command, history table, key learnings, session-start instructions.

### `core/winner_archive.py`

- `archive_winner(experiment_record, global_best_before, repo_root)`:
  1. only runs when `composite > global_best_before`
  2. creates `winners/<backbone>_exp<N>_<desc>/` tree
  3. copies model checkpoint, code snapshot, writes filled-in README, config.json, inference/predict.py
  4. generates the 14-section `audit_report.md` (see `generate_audit_report`)
  5. generates `colab_train_and_infer.ipynb` (see `generate_colab_notebook`)
  6. reruns the winner to produce `reproduction/reproduce_log.txt`
- `generate_audit_report` implements all 14 sections (executive summary, feature importance via permutation, top-N features, SHAP-style local, per-fold drift, calibration, uncertainty sanity, per-class/per-regime distribution, error attribution, risk audit, data-pipeline audit, config dump, limitations, deployment checklist).

### `skills/ml-autoresearch-setup/SKILL.md`

- Invokable as `/ml-autoresearch-setup`.
- 12-step AskUserQuestion flow. Persists answers to `<project_root>/autoresearch_setup_answers.json`.
- Emits filled-in `CLAUDE.md`, folder skeleton, starter runner, starter dashboard.

### `dashboard/dashboard.html`

- Pure-HTML + inline JS.
- Reads `experiment_log.jsonl` and `reasoning_annotations.json` via `fetch()` (served locally via `python -m http.server`).
- Backbone tabs, per-fold breakdown, reasoning detail panel.

## Order of build

| Phase | Artifact | Depends on |
|-------|----------|------------|
| 1 | `ARCHITECTURE.md` (this file) | — |
| 1 | `templates/CLAUDE_template.md` | source CLAUDE.md section inventory |
| 2 | `core/reasoning.py` | CLAUDE.md (Citation Rigor + Reasoning Blob Completeness) |
| 2 | `core/evaluation/splits.py`, `metrics.py`, `composite.py`, `uncertainty.py` | — |
| 2 | `core/backbones/base.py`, `registry.py`, `mlp.py`, `lstm.py`, `gbm.py`, `tabular_transformer.py`, `foundation_models.py` | — |
| 2 | `core/runner.py` | reasoning, evaluation, backbones, checkpoint |
| 2 | `core/checkpoint.py` | — |
| 2 | `core/winner_archive.py` | runner, backbones |
| 3 | `skills/ml-autoresearch-setup/SKILL.md` | templates |
| 3 | `dashboard/dashboard.html` | reasoning annotation schema |
| 4 | `examples/regression_house_prices/` | all core code |
| 4 | `examples/classification_titanic/` | all core code |
| 4 | `examples/time_series_airline/` | all core code |
| 5 | `tests/` (integration tests) | all of the above |
| 6 | `README.md` | everything referenced |

## Parallelism opportunities

- Phase 2 modules are mostly independent — `splits.py`, `metrics.py`, `reasoning.py`, `checkpoint.py` can be written in any order.
- Examples are independent of each other and all pull the same core, so they can be built after Phase 2 completes.

## Non-negotiable rules (from CLAUDE.md)

1. No section from source `CLAUDE.md` is dropped. Template must contain a generalized version of every heading, with a comment explaining what stays vs what is parameterized.
2. Citation Rigor floor: author/year/venue/title/arXiv/relevance note. `core/reasoning.py` refuses entries that violate.
3. Reasoning Blob Completeness floor: word counts per field. `core/reasoning.py` refuses entries that violate.
4. 7-step process: diagnose → cite → hypothesize → predict → execute → analyze → checkpoint.
5. Dashboard Files Update Mandate: runner writes JSONL + trade_logs + fallback annotation; Claude writes pre-run annotation + journal + summary + checkpoint + winner archive. Ownership table is preserved verbatim in the generated CLAUDE.md.
6. SOTA recipes drawn from the originating paper (not inherited across backbones).
7. Tier-3 GBMs remain THREE separate backbones (xgboost, lightgbm, catboost).
8. Winner archive is fully self-contained: README + config.json + model_checkpoint + code snapshot + inference + reproduction + audit report (14 sections) + Colab notebook.
9. Monotonic quality progression with Goodhart protections: the runner refuses to let the agent rewrite the composite metric or the data-integrity invariants mid-run.
10. GPU Memory Constraint: size-class ceilings and pre-flight check required as a reasoning annotation field.
