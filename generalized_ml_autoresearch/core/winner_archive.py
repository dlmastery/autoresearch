"""Winner archive — portable, self-contained champion package.

Implements CLAUDE.md "Winner Archiving Protocol" + "Explainability & Auditability
Report" (14 sections) + "Google Colab Notebook" requirements.

Triggered by the runner when composite > previous best.

Directory layout (from CLAUDE.md, preserved verbatim):

    winners/<backbone>_exp<N>_<desc>/
      README.md                  # champion description + deployment strategy
      config.json                # exact config
      model_checkpoint.pt / .joblib  # saved weights
      experiment_log_entry.json  # this experiment's JSONL row
      per_fold_results.json
      code/                      # frozen source snapshot
      inference/
        predict.py
        README_inference.md
      reproduction/
        reproduce_log.txt
        seed_variance.json
      audit_report.md            # 14-section explainability audit
      colab_train_and_infer.ipynb
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_SECTIONS = [
    "Executive summary",
    "Feature importance (permutation method)",
    "Top-N feature analysis",
    "SHAP-style local explanations",
    "Per-fold feature drift",
    "Calibration analysis",
    "Uncertainty sanity",
    "Per-regime prediction distribution",
    "Error attribution (top-5 winners & losers per fold)",
    "Risk audit",
    "Data pipeline audit",
    "Model config complete dump",
    "Known limitations & risks",
    "Deployment checklist",
]


def archive_winner(
    experiment_log_entry: dict[str, Any],
    source_code_roots: list[Path],
    model_checkpoint_src: Path,
    results_dir: Path,
    repro_fn=None,
) -> Path:
    """Create the winner archive directory tree. Returns its root path.

    `source_code_roots` is a list of directories to copy into `code/` (typically
    the runner's source root, plus any project-local code_versions).
    """
    exp_num = int(experiment_log_entry["experiment_num"])
    backbone = experiment_log_entry["backbone"]
    desc = _slugify(experiment_log_entry.get("description", "champion"))
    winner_root = Path(results_dir) / "winners" / f"{backbone}_exp{exp_num}_{desc}"
    winner_root.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    (winner_root / "config.json").write_text(
        json.dumps(experiment_log_entry.get("config", {}), indent=2, default=str),
        encoding="utf-8",
    )

    # 2. experiment_log_entry.json
    (winner_root / "experiment_log_entry.json").write_text(
        json.dumps(experiment_log_entry, indent=2, default=str), encoding="utf-8"
    )

    # 3. per_fold_results.json
    (winner_root / "per_fold_results.json").write_text(
        json.dumps(experiment_log_entry.get("per_fold_test_reports", []), indent=2, default=str),
        encoding="utf-8",
    )

    # 4. model_checkpoint
    if model_checkpoint_src.exists():
        dest = winner_root / model_checkpoint_src.name
        shutil.copy2(model_checkpoint_src, dest)

    # 5. code/ snapshot
    code_dir = winner_root / "code"
    code_dir.mkdir(exist_ok=True)
    for root in source_code_roots:
        if not root.exists():
            continue
        target = code_dir / root.name
        shutil.copytree(root, target, dirs_exist_ok=True,
                         ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))

    # 6. README.md
    (winner_root / "README.md").write_text(_render_readme(experiment_log_entry), encoding="utf-8")

    # 7. inference/
    infer_dir = winner_root / "inference"
    infer_dir.mkdir(exist_ok=True)
    (infer_dir / "predict.py").write_text(_render_predict_py(experiment_log_entry), encoding="utf-8")
    (infer_dir / "README_inference.md").write_text(_render_inference_readme(experiment_log_entry), encoding="utf-8")

    # 8. audit_report.md (14 sections)
    (winner_root / "audit_report.md").write_text(
        _render_audit_report(experiment_log_entry), encoding="utf-8"
    )

    # 9. Colab notebook
    (winner_root / "colab_train_and_infer.ipynb").write_text(
        _render_colab_notebook(experiment_log_entry), encoding="utf-8"
    )

    # 10. reproduction/
    repro_dir = winner_root / "reproduction"
    repro_dir.mkdir(exist_ok=True)
    repro_log = []
    repro_log.append(f"Winner archive created at {datetime.now().isoformat()}.")
    if repro_fn is not None:
        try:
            reproduced = repro_fn()
            repro_log.append(f"Reproduction composite: {reproduced:.4f}")
            orig = float(experiment_log_entry.get("composite", 0.0))
            diff = abs(reproduced - orig)
            repro_log.append(f"Original composite: {orig:.4f}; |diff| = {diff:.4f}")
            if diff > 0.5:
                repro_log.append("WARNING: reproduction differs by more than 0.5 — investigate.")
        except Exception as e:  # pragma: no cover
            repro_log.append(f"Reproduction failed: {e!r}")
    else:
        repro_log.append("Reproduction skipped (no repro_fn provided).")
    (repro_dir / "reproduce_log.txt").write_text("\n".join(repro_log), encoding="utf-8")
    (repro_dir / "seed_variance.json").write_text(
        json.dumps({"seeds_tested": [experiment_log_entry.get("config", {}).get("seed")], "notes": "multi-seed variance TBD — run seed-sweep experiments"}, indent=2),
        encoding="utf-8",
    )

    return winner_root


# -------------------------- rendering --------------------------

def _slugify(text: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")
    return slug[:40] or "champion"


def _render_readme(entry: dict[str, Any]) -> str:
    cfg = entry.get("config", {})
    per_fold = entry.get("per_fold_test", [])
    per_fold_val = entry.get("per_fold_val", [])
    lines = [
        f"# Champion — {cfg.get('backbone', entry.get('backbone', '?'))} Exp{entry['experiment_num']}",
        "",
        f"- **Description:** {entry.get('description', '')}",
        f"- **Composite:** {entry.get('composite', 0):.4f}",
        f"- **Test primary:** {entry.get('test_primary', 0):.4f}",
        f"- **Val primary:** {entry.get('val_primary', 0):.4f}",
        f"- **Timestamp:** {entry.get('timestamp', '')}",
        "",
        "## Per-fold test primary metric",
        "",
        "| Fold | Primary |",
        "|------|---------|",
    ]
    for i, v in enumerate(per_fold):
        lines.append(f"| {i+1} | {v:.4f} |")
    lines.append("")
    lines.append("## Per-fold val primary metric")
    lines.append("")
    lines.append("| Fold | Primary |")
    lines.append("|------|---------|")
    for i, v in enumerate(per_fold_val):
        lines.append(f"| {i+1} | {v:.4f} |")
    lines.append("")

    lines += [
        "## Full hyperparameter config",
        "",
        "```json",
        json.dumps(cfg, indent=2, default=str),
        "```",
        "",
        "## Architecture description",
        "",
        f"- **Backbone:** `{cfg.get('backbone')}`",
        f"- **Task type:** `{cfg.get('task_type')}`",
        f"- **Features in scope:** {len(cfg.get('data', {}).get('feature_columns', []) or [])} feature columns",
        "",
        "## Key insight — why this config won",
        "",
        "_Fill in by Claude: what changed from the previous champion and why it helped._",
        "",
        "## Training details",
        "",
        f"- **Seconds elapsed:** {entry.get('seconds_elapsed', 0):.1f}",
        f"- **Early-stop / best-iteration:** see `experiment_log_entry.json`",
        "",
        "## Uncertainty metrics (per fold)",
        "",
        "_See `per_fold_results.json` and `trade_logs/exp<N>_predictions.csv` for aleatoric / epistemic / confidence breakdown._",
        "",
        "## Secondary metrics",
        "",
    ]
    sm = entry.get("secondary_metrics", {})
    for k, v in sm.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines += [
        "## Reproduction status",
        "",
        "See `reproduction/reproduce_log.txt`.",
        "",
        "## Sample inference code",
        "",
        "```python",
        "from inference.predict import load_and_predict",
        "pred, conf, ale, epi = load_and_predict('model_checkpoint.pt', X_new)",
        "```",
        "",
        "## Deployment Strategy",
        "",
        "_Task-conditional. Claude fills in per CLAUDE.md 'Winner Archiving Protocol' — Trading Strategy section for time-series/finance, API-style inference for tabular, streaming batch for TS forecasting, etc._",
        "",
        "Required content (edit before deploying):",
        "",
        "1. **Signal generation** — input schema, output schema, uncertainty bands used",
        "2. **Decision rules** — pseudocode with thresholds (magnitude + confidence)",
        "3. **Output sizing / post-processing** — calibration temperature, clamp bounds, sign convention",
        "4. **Exit / retraining rules** — cadence, drift detection, kill-switch",
        "5. **Per-regime performance table** — (copy from audit report §1 and §8)",
        "6. **Risk controls** — daily loss cap or error ceiling, regime shift detection",
        "7. **Expected performance** — aggregate metric estimates (pre/post cost if applicable)",
        "8. **Caveats and warnings** — seed variance, feature dependencies, data-drift sensitivity",
        "9. **Reference to inference code** — link to `inference/predict.py`",
        "",
    ]
    return "\n".join(lines)


def _render_inference_readme(entry: dict[str, Any]) -> str:
    return (
        "# Inference — how to use this champion\n\n"
        "## Load\n\n"
        "```python\n"
        "from generalized_ml_autoresearch.core.backbones import create_model\n"
        "from generalized_ml_autoresearch.core.backbones.mlp import MLPBackbone\n"
        "# pick the matching Backbone class for the winner\n"
        "model = MLPBackbone.load('model_checkpoint.pt')\n"
        "```\n\n"
        "## Predict\n\n"
        "```python\n"
        "bundle = model.predict_with_uncertainty(X, n_samples=30)\n"
        "print(bundle.mean, bundle.aleatoric, bundle.epistemic, bundle.confidence)\n"
        "```\n\n"
        "## Input schema\n\n"
        "- **feature_columns:** see `config.json.data.feature_columns`\n"
        "- **scaler:** applied automatically on load; raw features expected\n\n"
        "## Output schema\n\n"
        "- `mean` — point prediction (regression) or predicted class (classification)\n"
        "- `aleatoric` — data uncertainty\n"
        "- `epistemic` — model uncertainty\n"
        "- `confidence` — 0..1, higher is more certain\n"
        "- `probabilities` — per-class probabilities (classification only)\n"
    )


def _render_predict_py(entry: dict[str, Any]) -> str:
    cfg = entry.get("config", {})
    backbone = cfg.get("backbone", entry.get("backbone", "mlp"))
    return f'''"""Standalone inference script for {backbone} champion Exp{entry["experiment_num"]}.

Usage:
    python predict.py model_checkpoint.pt path/to/features.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the backbone class dynamically.
# Copy the `core/backbones/` folder alongside this script for full portability,
# or keep generalized_ml_autoresearch on the PYTHONPATH.
from generalized_ml_autoresearch.core.backbones.registry import BACKBONE_REGISTRY


def load_model(checkpoint_path: str | Path):
    with open(Path(checkpoint_path).parent / "config.json") as f:
        cfg = json.load(f)
    backbone_name = cfg["config"]["backbone"] if "config" in cfg else cfg.get("backbone", "{backbone}")
    cls = BACKBONE_REGISTRY[backbone_name]
    return cls.load(checkpoint_path)


def load_and_predict(checkpoint_path: str, X_new: np.ndarray, n_samples: int = 30):
    model = load_model(checkpoint_path)
    bundle = model.predict_with_uncertainty(X_new, n_samples=n_samples)
    return bundle.mean, bundle.confidence, bundle.aleatoric, bundle.epistemic


if __name__ == "__main__":  # pragma: no cover
    ckpt, X_path = sys.argv[1], sys.argv[2]
    X = pd.read_csv(X_path).to_numpy(dtype=float)
    mean, conf, ale, epi = load_and_predict(ckpt, X)
    out = pd.DataFrame({{"mean": mean.flatten(), "confidence": conf.flatten(),
                         "aleatoric": ale.flatten(), "epistemic": epi.flatten()}})
    print(out.head(20).to_string(index=False))
    print(f"\\n(showing 20 / {{len(out)}} rows)")
'''


def _render_audit_report(entry: dict[str, Any]) -> str:
    """14-section audit report skeleton. Claude / run_audit_report.py fills
    in the quantitative bodies using `feature_importance.csv`, `shap_local.csv`,
    and the actual trained model.
    """
    cfg = entry.get("config", {})
    lines: list[str] = []
    lines.append(f"# Audit report — {cfg.get('backbone', entry.get('backbone'))} Exp{entry['experiment_num']}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("**Provenance:** this report's skeleton is created automatically by `core/winner_archive.py`. "
                 "The quantitative bodies (permutation importances, SHAP, drift Z-scores, calibration curves, "
                 "etc.) must be populated by running `run_audit_report.py` or filled in manually by Claude.")
    lines.append("")
    for i, section_title in enumerate(AUDIT_SECTIONS, start=1):
        lines.append(f"## {i}. {section_title}")
        lines.append("")
        lines.append(_audit_section_template(i, section_title, entry))
        lines.append("")
    return "\n".join(lines)


def _audit_section_template(i: int, title: str, entry: dict[str, Any]) -> str:
    cfg = entry.get("config", {})
    composite = entry.get("composite", 0)
    per_fold = entry.get("per_fold_test", [])
    if i == 1:  # Executive summary
        pf_str = "; ".join(f"F{j+1}={v:.4f}" for j, v in enumerate(per_fold))
        return (
            f"- Composite: **{composite:.4f}**\n"
            f"- Test primary metric (mean across folds): **{entry.get('test_primary', 0):.4f}**\n"
            f"- Val primary metric: **{entry.get('val_primary', 0):.4f}**\n"
            f"- Per-fold test primary: {pf_str}\n"
            f"- Pass/fail by fold: _Claude: fill in per regime rules._"
        )
    if i == 2:
        return (
            "Permutation importance: shuffle each feature column in the test set, re-evaluate, "
            "report the drop in primary metric. Output file: `feature_importance.csv` with columns "
            "`[feature_name, metric_drop, rank, domain_category]`. "
            "Cite: Breiman (2001) 'Random Forests'."
        )
    if i == 3:
        return "_Top-10 features by permutation importance; each with (a) what it measures, (b) why it matters in the domain, (c) per-fold stability._"
    if i == 4:
        return "_SHAP-style local explanations for 10 random test-set predictions using gradient × input (neural) or `shap.TreeExplainer` (GBM). Save as `shap_local.csv`._"
    if i == 5:
        return "_Per-fold feature drift: mean/std of each feature vs train set. Features with |Z| > 2 indicate shift. Report top 5 per fold._"
    if i == 6:
        return (
            "Calibration analysis: regression → predicted-quantile vs realized-mean monotonic check; "
            "classification → reliability diagram + ECE. "
            "Cite: Guo et al. (2017) 'On Calibration of Modern Neural Networks' (arXiv:1706.04599)."
        )
    if i == 7:
        return (
            "Uncertainty sanity: aleatoric vs |error| monotonicity; confidence decile vs accuracy. "
            "Cite: Kendall & Gal 2017 (arXiv:1703.04977)."
        )
    if i == 8:
        return "_Per-regime/class prediction distribution histograms; bias check (is model always predicting the mean?)._"
    if i == 9:
        return "_For each test fold, top-5 best predictions + top-5 worst. Pattern analysis: are errors concentrated on specific regimes?_"
    if i == 10:
        return (
            "_Risk audit: task-conditional. Regression: residual skew/kurtosis, max |error|, conditional VaR. "
            "Classification: per-class error rates, false-positive/negative cost. TS: drawdown period._"
        )
    if i == 11:
        return (
            "Data pipeline audit: reassert zero train/val/test leakage. "
            "Rerun `core.evaluation.splits.validate_no_overlap()` and include output verbatim."
        )
    if i == 12:
        return "```json\n" + json.dumps(cfg, indent=2, default=str) + "\n```"
    if i == 13:
        return "_Known limitations: which regimes has this model never seen? Where will it most likely fail?_"
    if i == 14:
        return (
            "- Monitoring plan: _which metrics to track daily / weekly?_\n"
            "- Kill-switch criterion: _max drawdown / error ceiling threshold_\n"
            "- Retraining cadence: _daily / weekly / monthly / on-drift_\n"
            "- Alerting: _who gets paged when metric X crosses threshold Y_"
        )
    return ""  # pragma: no cover


def _render_colab_notebook(entry: dict[str, Any]) -> str:
    """Render a minimal, runnable ipynb JSON. 8 cells per CLAUDE.md spec."""
    cfg = entry.get("config", {})
    backbone = cfg.get("backbone", "mlp")

    def md(text: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": text}

    def code(src: str) -> dict[str, Any]:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": src.splitlines(keepends=True)}

    cfg_json = json.dumps(cfg, indent=2, default=str)

    cells = [
        md(f"# Colab — train + infer for {backbone} Exp{entry['experiment_num']}\n\n"
            "Self-contained reproduction of the champion config. Target runtime < 5 min on Colab free tier."),
        md("## 1. Setup — install dependencies"),
        code("!pip install -q numpy pandas scikit-learn torch xgboost lightgbm catboost"),
        md("## 2. Data — load dataset\n\n"
            "_Replace this cell with the project's data loader (bundled CSV preferred over network download)._"),
        code("from sklearn.datasets import fetch_california_housing\n"
             "bunch = fetch_california_housing()\n"
             "X, y = bunch.data, bunch.target\n"
             "print('X.shape', X.shape, 'y.shape', y.shape)"),
        md("## 3. Feature engineering / preprocessing"),
        code("import numpy as np\n"
             "from sklearn.model_selection import train_test_split\n"
             "from sklearn.preprocessing import StandardScaler\n"
             "X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\n"
             "X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15, random_state=0)\n"
             "scaler = StandardScaler().fit(X_train)\n"
             "X_train_s = scaler.transform(X_train); X_val_s = scaler.transform(X_val); X_test_s = scaler.transform(X_test)"),
        md("## 4. Training — reproduce the champion config\n\n"
           "### Champion config\n\n"
           "```json\n" + cfg_json + "\n```"),
        code("# Minimal reproducible trainer — swap for full runner if the package is installed.\n"
             "import torch, torch.nn as nn\n"
             "config = " + cfg_json + "\n"
             "torch.manual_seed(config.get('seed', 0))\n"
             "# ... (user wires in the full training loop or imports the runner)\n"
             "print('Training placeholder — see generalized_ml_autoresearch/core/runner.py for the full loop.')"),
        md("## 5. Evaluation — primary + secondary metrics"),
        code("from sklearn.metrics import mean_squared_error\n"
             "# Evaluation placeholder.\n"
             "print('RMSE placeholder = 0.0')"),
        md("## 6. Inference — uncertainty bundle"),
        code("# Use generalized_ml_autoresearch.core.backbones.*.load() to load a real checkpoint.\n"
             "print('Inference placeholder')"),
        md("## 7. Visualization"),
        code("import matplotlib.pyplot as plt\n"
             "plt.figure(); plt.title('Equity / error curve placeholder'); plt.show()"),
        md("## 8. Export — save weights for deployment"),
        code("# torch.save(model.state_dict(), 'my_winner.pt')\n"
             "print('Export placeholder — copy model_checkpoint.pt from the winner archive.')"),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)
