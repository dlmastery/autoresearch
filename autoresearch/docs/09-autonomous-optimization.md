# 09 - Autonomous Optimization (Claude API Agent Loop)

**SWEBoK Knowledge Area:** KA3 — Software Construction (Automated Code Generation)
**Google SWE Reference:** Ch. 20 — "Static Analysis" (automated code quality)

---

## 1. Concept

The optimizer implements an autonomous experiment loop where Claude API acts as both researcher and engineer:
1. **Brainstorms** experiment ideas given current performance and past results
2. **Selects** the most promising low-risk idea
3. **Generates** complete modified source code
4. **Validates** syntax before execution
5. **Evaluates** the modified system on the full 7-fold walk-forward
6. **Keeps or reverts** based on Sharpe ratio comparison

This is a meta-learning loop: the AI improves the ML system's code, not just its weights.

## 2. Architecture

```
                    ┌──────────────────────────┐
                    │    Optimizer State        │
                    │  optimizer_state.json     │
                    │  - iteration              │
                    │  - current_sharpe         │
                    │  - best_sharpe            │
                    │  - experiments[]           │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │    Brainstorm Phase       │
                    │  Claude API + BRAINSTORM_ │
                    │  PROMPT                   │
                    │  → 3 experiment ideas     │
                    │  → JSON [{id, category,  │
                    │     description, risk}]   │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │    Selection Phase        │
                    │  Pick first low/medium    │
                    │  risk idea                │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │    Code Generation        │
                    │  Claude API + MODIFY_     │
                    │  PROMPT                   │
                    │  → Complete modified file  │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
     ┌────────▼────┐  ┌───────▼──────┐  ┌──────▼───────┐
     │ Syntax Check │  │ Backup Files  │  │ Apply Change │
     │ py_compile   │  │ .optimizer_   │  │ write_file() │
     │              │  │ backups/      │  │              │
     └──────┬──────┘  └──────────────┘  └──────┬──────┘
            │ fail                              │ pass
            │                          ┌───────▼──────┐
     ┌──────▼──────┐                  │   Evaluate    │
     │   Revert    │                  │  run_baseline  │
     │   + Log     │                  │  (7 folds)    │
     └─────────────┘                  └───────┬──────┘
                                              │
                                     ┌────────▼────────┐
                                     │   Compare       │
                                     │ new > current?  │
                                     └──┬──────────┬──┘
                                   yes │          │ no
                              ┌────────▼──┐  ┌───▼────────┐
                              │   Keep    │  │   Revert   │
                              │ Update    │  │ Restore    │
                              │ State     │  │ Backups    │
                              └───────────┘  └────────────┘
```

## 3. Experiment Categories

| Category | Description | Examples |
|----------|-------------|---------|
| `feature_engineering` | Add/modify/remove input features | New technical indicators, feature selection |
| `model_architecture` | Change backbone or model structure | Unfreeze layers, add attention, change hidden size |
| `training_hyperparams` | Adjust optimizer/scheduler/loss | Learning rate, batch size, epochs, scheduler |
| `head_design` | Modify prediction heads | Deeper heads, skip connections, different activations |
| `regularization` | Add/modify regularization | Dropout, weight decay, label smoothing |
| `data_preprocessing` | Change scaling/normalization | RobustScaler, log-transform, winsorization |
| `ensemble` | Combine multiple models | Fold-level ensembling, backbone blending |

## 4. Prompt Engineering

### 4.1 BRAINSTORM_PROMPT

**Inputs:** categories, current_sharpe, best_sharpe, past_experiments, model_code, feature_code

**Requirements for Claude:**
- Propose exactly 3 diverse ideas (different categories)
- Each idea: `{id, category, description, risk: "low"|"medium"|"high"}`
- Output pure JSON (no markdown, no commentary)
- Consider what has been tried before (avoid repeats)
- Consider current performance level

### 4.2 MODIFY_PROMPT

**Inputs:** experiment_description, current_code (full file contents)

**Requirements for Claude:**
- Output the COMPLETE modified file
- Keep changes minimal and targeted
- Don't change function signatures (interface compatibility)
- Valid Python 3.10+
- No markdown fences, no explanations

## 5. Safety Mechanisms

### 5.1 Pre-Experiment Backups

```python
MODIFIABLE_FILES = {
    "model": "model/backbone.py",
    "features": "data/features.py",
    "training": "model/train.py",
}
```

Before each experiment:
1. Copy all modifiable files to `.optimizer_backups/`
2. On failure: restore from backups
3. On revert: restore from backups

### 5.2 Syntax Validation

```python
def validate_syntax(code: str) -> (bool, str):
    # Write to temp file, run py_compile
    # Returns (True, "OK") or (False, "SyntaxError: ...")
```

No code is applied unless it passes `py_compile`.

### 5.3 Risk Assessment

The selection phase prioritizes:
1. Low-risk experiments first
2. Medium-risk if no low-risk available
3. High-risk only as last resort

### 5.4 State Persistence

```json
{
  "iteration": 5,
  "current_sharpe": 0.234,
  "best_sharpe": 0.234,
  "experiments": [
    {
      "id": 1,
      "category": "feature_engineering",
      "description": "Add momentum z-scores",
      "risk": "low",
      "sharpe": 0.234,
      "kept": true,
      "status": "completed"
    }
  ]
}
```

## 6. Modifiable Scope

The optimizer can ONLY modify these files:
- `model/backbone.py` — Model architecture
- `data/features.py` — Feature engineering
- `model/train.py` — Training loop

It CANNOT modify:
- `data/splits.py` — Split definitions (prevents data snooping)
- `evaluation/metrics.py` — Metric computation (prevents metric gaming)
- `baseline.py` — Evaluation framework (prevents evaluation shortcuts)
- Test files — Test integrity

This constraint ensures the optimizer improves the model, not the evaluation.

## 7. Entry Points

```bash
# Full optimizer loop (12 experiments)
python run_optimizer.py --max-experiments 12

# Baseline only (no optimization)
python run_optimizer.py --baseline-only

# Different Claude model
python run_optimizer.py --model claude-sonnet-4-20250514

# Overnight full pipeline (baseline + 10 experiments)
python run_overnight.py
```

## 8. Experiment Log Example

```
Experiment 1: feature_engineering (low risk)
  Description: Add RSI divergence features
  Result: Sharpe 0.12 → 0.18 (+0.06) ✓ KEPT

Experiment 2: training_hyperparams (low risk)
  Description: Increase learning rate to 3e-4
  Result: Sharpe 0.18 → 0.15 (-0.03) ✗ REVERTED

Experiment 3: model_architecture (medium risk)
  Description: Unfreeze last 2 LFM2.5 layers
  Result: SyntaxError in generated code ✗ SKIPPED

Experiment 4: regularization (low risk)
  Description: Add label smoothing (0.1)
  Result: Sharpe 0.18 → 0.22 (+0.04) ✓ KEPT
```
