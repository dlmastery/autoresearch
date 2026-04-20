# Generalized ML AutoResearch

Domain-agnostic successor to the FX-specific AutoResearch loop at
`C:/Users/evija/autoresearch/`. Preserves **every rule** from the source
`CLAUDE.md` (see [`templates/SECTION_MAPPING.md`](templates/SECTION_MAPPING.md)
for the 1-to-1 audit) while making the system work for any supervised ML task —
regression, classification (binary / multiclass), time-series forecasting,
ranking, and survival.

## What you get

A Claude-driven autoresearch loop with:

- **Automatic reasoning validation**: every experiment must pass Citation Rigor +
  Reasoning Blob Completeness gates before it launches. The runner refuses to
  run an experiment whose pre-run reasoning annotation is missing or shallow.
- **Pluggable backbones**: MLP, LSTM, FT-Transformer, XGBoost, LightGBM, CatBoost,
  plus stubs for 2024-2026 foundation models (TimesFM, Chronos, MOMENT, Moirai,
  TiRex, Sundial, Time-MoE).
- **Pluggable splits**: HoldoutSplit, KFold, StratifiedKFold, GroupKFold,
  TimeSeriesSplit, WalkForwardSplit, SuperFoldSplit.
- **Composite-metric discipline**: `min(val, test) − penalty × n_below_threshold`
  with a frozen fingerprint (detects Goodhart-style mid-project rewrites).
- **Full MLOps artifacts** per experiment: JSONL log, per-prediction CSV,
  reasoning annotation entry, dashboard-ready data.
- **Winner archive**: self-contained `winners/<backbone>_exp<N>_<desc>/` with
  14-section audit report, inference script, Colab notebook, code snapshot,
  reproduction log — everything a downstream engineer needs to deploy.
- **Crash-recovery checkpointing**: laptop-crash-resistant `memory/project_autoresearch_checkpoint.md`
  written after every experiment.
- **Interactive setup wizard**: a `/ml-autoresearch-setup` Skill that walks users
  through 12 steps and produces a filled-in CLAUDE.md.

## Installation

```bash
pip install numpy pandas scikit-learn torch xgboost lightgbm catboost pyyaml
```

Optional: `psutil` for CPU-affinity pinning; `scipy` for softmax/sigmoid in the MLP
and FT-Transformer backbones.

## Quick start

### Option A — interactive wizard

```
/ml-autoresearch-setup
```

The Skill walks you through 12 steps and produces a ready-to-run project at a
location you choose.

### Option B — copy a worked example

The fastest way to understand the framework is to run the three bundled examples:

```bash
python generalized_ml_autoresearch/examples/regression_house_prices/run_example.py
python generalized_ml_autoresearch/examples/classification_titanic/run_example.py
python generalized_ml_autoresearch/examples/time_series_airline/run_example.py
```

Each writes a complete `autoresearch_results/` tree plus a dashboard you can open via
`python -m http.server 8765 --directory <dir>`.

### Option C — write your own config

1. Create a `configs/my_project.yaml` with this shape:
   ```yaml
   paths:
     results_dir: /path/to/results
   task_type: regression  # or binary_classification / time_series_forecasting / ...
   primary_metric: rmse
   backbone: mlp
   backbone_config:
     hidden: [256, 128, 64]
     dropout: 0.2
     epochs: 50
     patience: 10
     lr: 3.0e-4
     batch_size: 32
     seed: 0
   data:
     format: csv
     path: /path/to/features.csv
     target_columns: [target]
   split:
     name: kfold
     n_splits: 5
   composite:
     higher_is_better: false
     penalty_weight: 0.1
   description: "baseline MLP"
   seed: 0
   ```
2. Author a pre-run reasoning entry in `<results_dir>/reasoning_annotations.json`
   keyed by `"1"` (see `examples/*/seed_reasoning.json` for the shape and the
   minimum word counts).
3. Run:
   ```bash
   python -m generalized_ml_autoresearch.core.runner --config configs/my_project.yaml
   ```

## Full workflow walkthrough

The intended operating mode is: **Claude Code is the outer loop**. Every
experiment follows the 7-step Research-Driven Experiment Selection process
from CLAUDE.md:

1. **Diagnose** the current champion's weakness (per-fold analysis).
2. **Cite** a paper that addresses the diagnosed weakness (full author/year/venue/title/arXiv + relevance note).
3. **Hypothesize** the mechanistic change.
4. **Predict** the numeric outcome range.
5. **Execute** ONE experiment via the runner.
6. **Analyze** against prediction.
7. **Checkpoint** — every output file ([Dashboard Files Update Mandate](templates/CLAUDE_template.md)) is refreshed.

The framework enforces this with hard gates:
- The runner refuses to launch without a valid pre-run reasoning entry.
- The composite-metric fingerprint is stored — mid-project rewrites are detected.
- The winner archive is auto-generated on every new global champion.

## How to extend

### Adding a new backbone

```python
# in your project or a new file in core/backbones/
from generalized_ml_autoresearch.core.backbones import register_backbone, Backbone, PredictionBundle

@register_backbone("my_model")
class MyBackbone(Backbone):
    name = "my_model"
    task_types = {"regression", "binary_classification"}

    def build(self, config, input_shape, n_outputs):
        ...
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        ...
    def predict_with_uncertainty(self, X, n_samples=30) -> PredictionBundle:
        ...
    def save(self, path):
        ...
    @classmethod
    def load(cls, path):
        ...
```

Then set `backbone: my_model` in your config.

### Adding a new metric

```python
from generalized_ml_autoresearch.core.evaluation.metrics import register_metric

@register_metric("smape")
def smape(y_true, y_pred, **_):
    import numpy as np
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom))
```

Then set `primary_metric: smape` in your config.

### Adding a new split protocol

Subclass `core.evaluation.splits._BaseSplitter` and register in `SPLIT_REGISTRY`.

## FAQ

**Q: Why does the runner refuse to launch?**
A: Your pre-run reasoning annotation (`reasoning_annotations.json` entry for the
upcoming experiment number) is either missing or fails the validators (Citation
Rigor or Reasoning Blob Completeness). Read the error message — it lists every
violation.

**Q: The word-count floors feel pedantic. Can I disable them?**
A: No. They are the mechanism that prevents drift into "let me try X" guessing.
If you really must bypass in an emergency, pass `--bypass-reasoning-gate`, but
the runner will insert `TODO-REWRITE` sentinels and the dashboard will flag them.

**Q: Can I merge xgboost/lightgbm/catboost into one backbone?**
A: No. They are three separate architectures per the CLAUDE.md Tier-3 rule —
different splitting algorithms, regularization mechanisms, categorical handling.
Each gets its own 50-experiment budget.

**Q: What's different from the FX AutoResearch at `C:/Users/evija/autoresearch/`?**
See the comparison table below.

## Comparison vs existing FX AutoResearch

| Dimension | FX AutoResearch | Generalized ML AutoResearch |
|-----------|-----------------|-----------------------------|
| **Task types** | FX return prediction (regression, time-series) | Any supervised ML (regression / classification / time-series / ranking / survival / multi-label) |
| **Data loader** | Hard-coded `download_all_pairs()` | `format: csv / parquet / numpy / sklearn / callable` |
| **Features** | 104 hand-engineered FX features | User-defined; no assumption |
| **Splits** | `SuperFoldSplit` over 7 regime windows | 7 split protocols (holdout / kfold / stratified / grouped / TS / walk-forward / super-fold) |
| **Primary metric** | Sharpe | Any registered metric; user chooses at setup |
| **Composite formula** | `min(val, test) − 0.1 × n_negative_folds` | Same default; user-overridable (sandboxed formula or callable) |
| **Backbones** | 8 bespoke (mlp, lstm, patchtst, lfm2, etc.) | 6+ generic (mlp, lstm, ft_transformer, xgboost, lightgbm, catboost) + 7 foundation-model stubs |
| **Trade log** | `exp<N>_trades.csv` with pnl_bps, pair | `exp<N>_predictions.csv` with task-conditional columns |
| **Hardware** | 16 GB VRAM cap, E-core ban hard-coded | Setup wizard captures user's hardware profile; budget scales |
| **Winner archive** | FX-specific trading README | Task-conditional deployment-strategy template |
| **Setup** | One-off shell commands | 12-step `/ml-autoresearch-setup` Skill |
| **CLAUDE.md sections** | 52 sections | 52/52 preserved (see [`templates/SECTION_MAPPING.md`](templates/SECTION_MAPPING.md)) |
| **Reasoning floor** | Citation Rigor + Reasoning Blob Completeness (manual enforcement) | Same floors, **programmatically enforced** by `core/reasoning.py` |
| **Goodhart protection** | Informal | Composite fingerprint stored with every experiment; rewrites detectable |

## Deliverable map

| Deliverable | Path |
|---|---|
| User guide (this file) | `README.md` |
| Architecture DAG | `ARCHITECTURE.md` |
| Setup Skill | `skills/ml-autoresearch-setup/SKILL.md` |
| CLAUDE.md template | `templates/CLAUDE_template.md` |
| Section-coverage audit | `templates/SECTION_MAPPING.md` |
| SOTA backbone catalog | `templates/sota_catalog.yaml` |
| Runner | `core/runner.py` |
| Backbones | `core/backbones/{base,registry,mlp,lstm,gbm,tabular_transformer,foundation_models}.py` |
| Evaluation | `core/evaluation/{splits,metrics,composite,uncertainty}.py` |
| Reasoning validator | `core/reasoning.py` |
| Checkpoint manager | `core/checkpoint.py` |
| Winner archive + audit + Colab | `core/winner_archive.py` |
| Dashboard | `dashboard/dashboard.html` |
| Example — regression | `examples/regression_house_prices/` |
| Example — classification | `examples/classification_titanic/` |
| Example — time-series | `examples/time_series_airline/` |
| Tests | `tests/test_smoke.py`, `tests/test_runner_e2e.py`, `tests/test_section_coverage.py` |

## Testing

```bash
python -m pytest generalized_ml_autoresearch/tests -v
```

Three test files; 10 total tests. All pass on Python 3.12 + PyTorch 2.x.

## Licensing

Follows the parent `autoresearch` repository's license.

## Credits

- FX AutoResearch methodology (the source CLAUDE.md) — Evija Ranti.
- Generalized framework — Claude (hierarchical coordinator) delivering on the Queen's mission of 2026-04-19.
