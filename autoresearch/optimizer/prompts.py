"""Prompt templates for the autonomous optimizer loop.

Contains category definitions and two prompt templates used by the
Claude-powered experiment agent: one for brainstorming ideas and one
for generating code modifications.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment categories the agent can explore
# ---------------------------------------------------------------------------
CATEGORIES = [
    "feature_engineering",
    "model_architecture",
    "training_hyperparams",
    "head_design",
    "regularization",
    "data_preprocessing",
    "ensemble",
]

# ---------------------------------------------------------------------------
# Brainstorm prompt — proposes 3 experiment ideas as JSON
# ---------------------------------------------------------------------------
BRAINSTORM_PROMPT = """\
You are an expert quantitative researcher working on an FX return prediction \
system. Your goal is to improve the model's annualized Sharpe ratio.

The system predicts 1-day and 5-day forward returns for 6 currency pairs using \
an LFM2.5-350M backbone with walk-forward cross-validation.

Experiment categories you may explore:
{categories}

Current Sharpe ratio: {current_sharpe:.4f}
Best Sharpe ratio so far: {best_sharpe:.4f}

Past experiments (most recent first):
{past_experiments}

=== Current model code (model/backbone.py) ===
{model_code}

=== Current feature code (data/features.py) ===
{feature_code}

Propose exactly 3 diverse experiment ideas. Each idea should target a \
different category and offer a clear hypothesis for why it would improve \
the Sharpe ratio.

Return ONLY a JSON array (no markdown fences, no commentary) with this schema:
[
  {{
    "id": 1,
    "category": "<one of the categories above>",
    "description": "<concise description of the change and the hypothesis>",
    "risk": "<low|medium|high>"
  }},
  ...
]
"""

# ---------------------------------------------------------------------------
# Modify prompt — generates complete modified file
# ---------------------------------------------------------------------------
MODIFY_PROMPT = """\
You are a senior Python engineer. Apply the following experiment to the code \
below. Output ONLY the complete modified Python file — no explanations, no \
markdown fences, no commentary before or after.

Rules:
- Do NOT change existing function signatures or class names.
- Do NOT modify train/val/test splits or evaluation logic.
- Keep changes focused and minimal — only what the experiment requires.
- Preserve all existing imports that are still needed.
- The code must be valid Python 3.10+.

=== Experiment description ===
{experiment_description}

=== Current code ===
{current_code}
"""
