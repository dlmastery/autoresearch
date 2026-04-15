# AutoResearch Process — Claude as Expert MLFin Researcher

## Karpathy's Original Principle (github.com/karpathy/autoresearch)

Karpathy's autoresearch: modify → train (5 min) → check if improved → keep/discard → repeat.
"Everything is fair game: architecture, hyperparameters, optimizer, batch size."
The agent runs autonomously until interrupted. One file modified. One metric to beat.

## Our Adaptation for FX Prediction

Same keep/discard loop, but with THREE critical differences:

1. **NEVER deviate far from the winner.** Karpathy's agents can try wild changes
   because training is 5 minutes. Ours is ~3 minutes but the evaluation is across
   7 regime windows. Small modifications around the champion. Build on what works.
   The winner config is sacred — every experiment starts from it.

2. **Claude IS the expert researcher.** Karpathy's agent just tries things.
   Our agent must DIAGNOSE per-fold, cite literature, form hypotheses. The quality
   of reasoning determines the quality of experiments. No blind exploration.

3. **Epoch-bound, not time-bound.** 20 epochs with early stopping. The model
   trains until convergence, not for a fixed wall-clock budget.

## Core Invariant

**Always start from the current best config. Modify ONE thing. Keep if composite
improves. Revert if not. Never wander off.**

## The Loop (Every Iteration)

### Step 1: Read Results
- Load `experiment_log.jsonl` — the full history
- Load `best_config.json` — the current champion
- Examine per-window breakdown for BOTH val and test
- Note: which folds improved/degraded vs. prior experiments?

### Step 2: Diagnose (This Is Where The Work Happens)
- **Per-fold forensics**: Which fold windows are weak? WHY?
  - Is it a regime problem? (GFC is fundamentally different from low-vol plateau)
  - Is it a sample-size problem? (fold_1 has 53 test samples vs fold_4's 118)
  - Is it a train-test distribution shift? (features that matter in 2008 vs 2025)
  - Is it overfitting? (train Sharpe >> test Sharpe for that window?)
  - Is it underfitting? (model predicting near-zero for all samples?)
- **Train-test gap analysis**: How much does the model overfit?
  - gap < 0.5 → healthy generalisation
  - gap 0.5-1.5 → mild overfitting, regularisation may help
  - gap > 2.0 → severe overfitting, need structural changes
- **Val-test consistency**: Do val and test agree on direction?
  - If val >> test → model found patterns that don't persist out-of-sample
  - If test >> val → lucky on recent data, val is more honest
  - If both positive → signal is real across regimes
- **Composite decomposition**: What's driving the score?
  - `min(test_sharpe, val_sharpe)` — which side is the bottleneck?
  - `0.1 * n_negative_folds` — which specific windows are negative?
- **Trajectory analysis**: Are we making progress or cycling?
  - Plot composite over experiments — is there a trend?
  - Are we seeing diminishing returns from HP tuning?
  - Have we exhausted the current architecture's capacity?

### Step 3: Research (Go Deep When Stuck)
- Read relevant literature for the specific problem diagnosed in Step 2:
  - Foundation model fine-tuning: Hu et al. 2022 (LoRA), Devlin et al. 2019 (BERT)
  - FX microstructure: Menkhoff et al. 2012 (carry trade), Lustig et al. 2011
  - Time series forecasting: Nie et al. 2023 (PatchTST), Das et al. 2024 (LFM)
  - Robust training: Zhang et al. 2017 (mixup), Müller et al. 2019 (label smoothing)
  - Walk-forward evaluation: Lopez de Prado 2018 (AFML)
- Check LFM2 technical report for recommended fine-tuning practices
- Look at what the adapter/PEFT literature says about the specific failure mode

### Step 4: Hypothesize
Write a SPECIFIC, FALSIFIABLE hypothesis:
- "Fold 7 test is negative because [specific reason]. [Paper X] suggests
  [specific technique] addresses this because [mechanism]. I predict this
  will improve fold 7 from -0.44 to positive while maintaining fold 4-6
  performance."
- NOT: "Let me try warmup and see what happens."
- NOT: "Maybe a different learning rate will work."

### Step 5: Design ONE Experiment
- Change exactly ONE thing from the current best config
- The change must directly test the hypothesis from Step 4
- Justify every parameter value with literature, guidelines, or prior results
- Predict what you expect to see (so you can evaluate the hypothesis)

### Step 6: Run & Analyse
- Run the experiment via `run_autoresearch.py`
- Compare result to prediction:
  - Did the fold you targeted improve?
  - Did anything else break?
  - Does the composite improve?
- Update understanding based on what happened
- If composite improved → new best. If not → revert, hypothesis was wrong.

### Step 7: Decide Next Direction
- If KEEP: small tweaks around the new best (explore locally)
- If REVERT after 1-2 tries: try a different direction for the same problem
- If REVERT after 3+ tries on the same problem: rethink the diagnosis
  - Maybe the problem isn't what you thought
  - Maybe it's architectural, not hyperparameter-level
  - Maybe the fold is fundamentally unpredictable (GFC with 53 samples)
- Occasionally try RADICAL changes to escape local optima

## Anti-Patterns (Never Do These)

| Anti-Pattern | What To Do Instead |
|---|---|
| "Let me try X and see" | "I'm trying X because [diagnosis] and [paper] suggests [mechanism]" |
| Grid search (try 5 values of lr) | Diagnose → hypothesize → test ONE value with justification |
| Changing 2+ things at once | ONE change. If both matter, test them in sequence. |
| Ignoring per-fold breakdown | The aggregate hides regime-specific failures. Always read per-fold. |
| Repeating a failed direction | 3 failures on the same axis → rethink the problem, not the values |
| Arbitrary parameter values | Every number must be justified: paper, guidelines, or empirical history |
| Running without diagnosing | NEVER run an experiment without first writing the diagnosis |
| Assuming stochasticity | Measure it. Same config, different seed. Quantify the noise floor. |

## What Makes This Different From Hyperopt/Optuna

Hyperopt explores a parameter space blindly. This process:
1. Uses domain knowledge to ELIMINATE most of the space
2. Forms testable hypotheses about WHY the model behaves as it does
3. Reads literature for solutions to DIAGNOSED problems
4. Makes predictions about experimental outcomes
5. Learns from failures (updates the mental model, not just the search bounds)
6. Knows when to stop tuning and make structural changes

The researcher's advantage over Optuna is understanding. Optuna can't read a
paper. It can't notice that fold 7's negative Sharpe might be because the
projection layer is learning a degenerate mapping. It can't reason about
regime shifts. Claude can.

## State Files
- `autoresearch_results/experiment_log.jsonl` — append-only experiment log
- `autoresearch_results/best_config.json` — current champion config + full results
- `autoresearch_results/dashboard.html` — visual dashboard (reads logs, decoupled)
- `autoresearch_results/running.json` — transient signal while experiment runs
