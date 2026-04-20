# 09 - Autonomous Optimization (Claude Agent Loop)

**SWEBoK Knowledge Area:** KA3 — Software Construction (Automated Code Generation)
**Google SWE Reference:** Ch. 20 — "Static Analysis" (automated code quality)

---

## Executive Summary

AutoResearch implements an autonomous ML research loop where **Claude Code itself is the outer loop** -- there is no separate Python agent. Claude reads experiment results, reasons about WHY the model behaves a certain way (citing relevant literature), formulates a hypothesis, predicts the expected outcome, runs ONE experiment, analyzes the results, and checkpoints its state for crash recovery. This is inspired by Andrej Karpathy's approach to hyperparameter tuning: always start from the best config, change one thing, keep or revert, repeat.

Over 90 experiments, the system evolved from a baseline LFM2-350M foundation model (test Sharpe +1.40) to a residual MLP champion (test Sharpe +6.21) -- a 4.4x improvement. Key breakthroughs came not from hyperparameter tuning but from architectural insight: adding residual skip connections (He et al. 2016) produced a 5x improvement in a single experiment.

```
  The AutoResearch Agent Loop
  ===========================

  +-----------+     +-----------+     +-------------+     +-----------+
  |  1. Diag- |---->|  2. Cite  |---->| 3. Hypothe- |---->| 4. Pre-   |
  |  nose     |     |  Litera-  |     |    size      |     |    dict   |
  |           |     |  ture     |     |              |     |           |
  | "Fold 2   |     | "He 2016: |     | "Add skip    |     | "Expect   |
  |  is weak  |     |  residual |     |  connection  |     |  fold 2   |
  |  due to   |     |  connec-  |     |  to stabi-   |     |  Sharpe   |
  |  gradient |     |  tions    |     |  lize gradi- |     |  goes     |
  |  instab"  |     |  fix this"|     |  ent flow"   |     |  >+0.5"   |
  +-----------+     +-----------+     +-------------+     +-----------+
       ^                                                       |
       |                                                       v
  +-----------+     +-----------+                         +-----------+
  | 7. Check- |<----| 6. Ana-   |<------------------------| 5. Run    |
  |    point  |     |    lyze   |                         |    ONE    |
  |           |     |           |                         |    exper- |
  | Save to   |     | "Fold 2:  |                         |    iment  |
  | memory/   |     |  +0.23 ->  |                         |           |
  | project_  |     |  +1.17    |                         | ~36 sec   |
  | checkpoint|     |  KEEP"    |                         | on CPU    |
  +-----------+     +-----------+                         +-----------+
```

---

## 1. Concept

The system implements two levels of autonomous optimization:

**Level 1: Claude Code as Research Agent (Current -- Production)**
Claude Code IS the outer loop. It reads results, reasons about architecture/hyperparameters, and calls `run_autoresearch.py` for each experiment. The intelligence is in the agent, not in Python code. This is the system that produced the 90-experiment champion.

**Level 2: Claude API Code Generation Agent (Legacy -- Experimental)**
An earlier approach where the Claude API brainstormed experiment ideas, generated modified source code, validated syntax, and ran evaluations. This approach is documented below for reference but has been superseded by Level 1.

**Why Level 1 is better:**
- The agent has the full conversation context (all prior experiments, literature knowledge, fold-level diagnostics)
- It can make architectural decisions (adding skip connections) not just hyperparameter changes
- It can modify code directly, test it, and revert if needed
- Crash recovery is simpler (checkpoint file + JSONL log)

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

## 3. The 7-Step Experiment Protocol (Karpathy-Adapted)

Every experiment follows this exact protocol. Skipping steps leads to wasted experiments.

### Step 1: DIAGNOSE -- Understand the Current State

Before choosing what to change, deeply analyze the per-fold results:
- Which folds have negative or weak test Sharpe?
- Is the weakness in val too (overfitting) or just test (regime mismatch)?
- What is the aleatoric/epistemic uncertainty per fold?
- Are there patterns across folds (e.g., all high-volatility folds are weak)?

**Example from the project:**
```
After experiment 58 (plain MLP), fold 2 (post-crash recovery) had Sharpe +0.23
while other folds averaged +1.8. Diagnosis: fold 2 has extreme variance in
daily returns (post-GFC), causing gradient instability in the plain MLP.
```

### Step 2: CITE -- Ground the Diagnosis in Literature

Every diagnosis must be supported by published work or prior project results:
- He et al. (2016): Residual connections solve gradient degradation in deep networks
- Gu, Kelly & Xiu (2020): Smaller networks generalize better for empirical asset pricing
- Smith & Le (2018): Batch size and learning rate are coupled; larger batch = higher LR
- Kendall & Gal (2017): Heteroscedastic uncertainty separates aleatoric from epistemic noise

**Example:**
```
He et al. (2016) showed that skip connections allow gradients to flow directly
through the network, preventing the gradient vanishing that causes deep networks
to perform WORSE than shallow ones. Our MLP has this exact symptom: deeper
hidden layers hurt performance.
```

### Step 3: HYPOTHESIZE -- Formulate a Testable Prediction

State the exact change and WHY it should help, based on the diagnosis and literature:
- ONE change per experiment (strict)
- Must be falsifiable: "I expect fold X Sharpe to improve from Y to Z"
- Must explain the mechanism: "because skip connections stabilize gradients through the high-variance fold 2 data"

### Step 4: PREDICT -- Quantitative Expected Outcome

Before running, write down:
- Expected composite score (range)
- Expected per-fold changes (especially the weak fold)
- What would cause you to KEEP vs DISCARD

### Step 5: RUN -- Execute ONE Experiment

```bash
cd C:/Users/evija/autoresearch && "C:/Users/evija/anaconda3/python.exe" \
  -m autoresearch.run_autoresearch --backbone lfm2-350m \
  [config flags] \
  --description "add residual skip connection (He 2016) to stabilize fold 2 gradients"
```

- Timeout: 600 seconds (10 minutes)
- 60-second cooldown after each experiment
- Log output is parsed for per-fold results

### Step 6: ANALYZE -- Compare Results to Prediction

- Was the hypothesis correct? Did the predicted fold improve?
- Did any OTHER folds regress? (Common: fixing fold 2 breaks fold 4)
- Is the composite score above the current best?
- KEEP if composite > previous best; DISCARD otherwise.
- If DISCARD: understand WHY the hypothesis was wrong before trying something else.

### Step 7: CHECKPOINT -- Save State for Crash Recovery

Save the complete state to `memory/project_autoresearch_checkpoint.md`. This file must be self-contained: a fresh Claude Code session reading ONLY `CLAUDE.md` + the checkpoint must be able to resume the loop.

### Real Experiment Decision Examples

**Example 1: Residual Skip Connection (Experiment 29 -- the breakthrough)**
```
DIAGNOSE:  Plain MLP composite +1.01. Fold 2 Sharpe +0.23. Fold 1 Sharpe +0.48.
           Deep MLP (3 layers) was WORSE than shallow MLP (2 layers).
           Gradient degradation problem identified.

CITE:      He et al. 2016 "Deep Residual Learning for Image Recognition"
           -- skip connections fix gradient degradation, enabling deeper
           networks to at least match shallow ones.

HYPOTHESIZE: Adding y = f(x) + x skip connection will stabilize gradients,
             especially on high-variance folds (1, 2). The MLP hidden layer
             output is added to the input projection, creating a residual path.

PREDICT:   Composite from +1.01 to at least +2.0. Fold 2 from +0.23 to >+0.5.

RUN:       --backbone mlp --hidden_dim 128 --head_dim 64 --residual True

RESULT:    Composite +5.50 (5x improvement!). Fold 2: +1.17. ALL 7 folds positive.
           The skip connection didn't just fix fold 2 -- it improved EVERY fold.

ANALYSIS:  Hypothesis confirmed beyond expectations. The gradient flow benefit
           was universal, not fold-specific. This is now the champion.
```

**Example 2: BatchNorm Addition (Experiment 35 -- a revert)**
```
DIAGNOSE:  Champion residual MLP composite +5.50. Looking for incremental gains.
           Feature scale variation across folds might benefit from normalization.

CITE:      Ioffe & Szegedy 2015 "Batch Normalization: Accelerating Deep
           Network Training by Reducing Internal Covariate Shift"

HYPOTHESIZE: Adding BatchNorm after the hidden layer will normalize activations,
             reducing internal covariate shift and improving generalization.

PREDICT:   Small improvement, composite from +5.50 to ~+5.70.

RUN:       --backbone mlp --hidden_dim 128 --residual True --batchnorm True

RESULT:    Composite +4.20. Fold 3 dropped from +9.76 to +5.2.

ANALYSIS:  BatchNorm HURT. Hypothesis: BatchNorm removes regime-scale information
           that the model uses to distinguish market regimes. FX returns have
           regime-dependent scale (high-vol GFC vs low-vol 2019), and normalizing
           this away removes a useful signal. REVERTED. Key learning: do not
           normalize away regime information in financial data.
```

**Example 3: Consecutive Discards Leading to Rethink (Experiments 40-43)**
```
DIAGNOSE:  4 consecutive discards trying to push past +5.50 with hyperparameter
           tweaks (LR=7e-4, wd=1e-3, seq=20, epochs=100). All within +/-0.3 of
           champion but none exceeding it.

RETHINK:   The Karpathy protocol says: "If you see consecutive discards, stop
           and rethink." Hyperparameters are exhausted near the optimum. Need
           a fundamentally different approach.

NEW DIRECTION: Switched from hyperparameter tuning to cross-seed verification
               (seeds 0, 42, 99) to confirm the champion is robust, then moved
               to exploring batch size reductions as the next architectural lever.
```

---

## 4. Crash-Recovery Checkpointing

The development laptop crashes frequently (thermal, power, Windows updates). The checkpoint system ensures zero progress is lost.

### 4.1 Checkpoint Frequency

- **After EVERY experiment completion** (mandatory)
- **Every 5 minutes of wall clock time** (even during analysis/reasoning)
- **Before starting any long-running operation** (experiment run, data download)

### 4.2 Checkpoint File Format

The checkpoint is saved to `memory/project_autoresearch_checkpoint.md` and contains:

```
## Session Recovery
1. Read this checkpoint
2. Read JSONL tail (last 3) + best_config.json
3. Start dashboard
4. Resume from next experiment below

## Current Champion
Config: [exact CLI flags]
Per-fold test Sharpe table (7 rows)
Composite: [score]

## Last Experiment
Config: [what was tried]
Result: [composite, per-fold deltas vs champion]
Status: KEEP or DISCARD
Why: [one-sentence explanation]

## Next Experiment
Command: [copy-pasteable bash command]
Rationale: [diagnosis + literature cite + hypothesis]

## Exhausted Axes
[List of parameters already explored with results]

## Key Learnings
[Findings that should not be re-discovered]
```

### 4.3 Recovery Procedure

A fresh Claude Code session reads exactly two files:
1. `CLAUDE.md` -- project rules and architecture
2. `memory/project_autoresearch_checkpoint.md` -- current state

From these two files, the agent can resume the experiment loop exactly where it left off, without reading the full JSONL log, best_config.json, or any source code.

---

## 5. Experiment Categories

| Category | Description | Examples |
|----------|-------------|---------|
| `feature_engineering` | Add/modify/remove input features | New technical indicators, feature selection |
| `model_architecture` | Change backbone or model structure | Unfreeze layers, add attention, change hidden size |
| `training_hyperparams` | Adjust optimizer/scheduler/loss | Learning rate, batch size, epochs, scheduler |
| `head_design` | Modify prediction heads | Deeper heads, skip connections, different activations |
| `regularization` | Add/modify regularization | Dropout, weight decay, label smoothing |
| `data_preprocessing` | Change scaling/normalization | RobustScaler, log-transform, winsorization |
| `ensemble` | Combine multiple models | Fold-level ensembling, backbone blending |

## 6. Prompt Engineering (Legacy Claude API Agent)

### 6.1 BRAINSTORM_PROMPT

**Inputs:** categories, current_sharpe, best_sharpe, past_experiments, model_code, feature_code

**Requirements for Claude:**
- Propose exactly 3 diverse ideas (different categories)
- Each idea: `{id, category, description, risk: "low"|"medium"|"high"}`
- Output pure JSON (no markdown, no commentary)
- Consider what has been tried before (avoid repeats)
- Consider current performance level

### 6.2 MODIFY_PROMPT

**Inputs:** experiment_description, current_code (full file contents)

**Requirements for Claude:**
- Output the COMPLETE modified file
- Keep changes minimal and targeted
- Don't change function signatures (interface compatibility)
- Valid Python 3.10+
- No markdown fences, no explanations

## 7. Safety Mechanisms

### 7.1 Pre-Experiment Backups

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

### 7.2 Syntax Validation

```python
def validate_syntax(code: str) -> (bool, str):
    # Write to temp file, run py_compile
    # Returns (True, "OK") or (False, "SyntaxError: ...")
```

No code is applied unless it passes `py_compile`.

### 7.3 Risk Assessment

The selection phase prioritizes:
1. Low-risk experiments first
2. Medium-risk if no low-risk available
3. High-risk only as last resort

### 7.4 State Persistence

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

## 8. Modifiable Scope

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

## 9. Entry Points

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

## 10. Experiment Log Example

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

## 11. Progress Summary: 90 Experiments

### 11.1 Phase 1: LFM2 Foundation Model (Experiments 1-50)

The first 50 experiments explored the LFM2-350M foundation model (frozen backbone, ~500K trainable params). Median test Sharpe reached +1.40 after extensive tuning.

Key findings:
- LFM2 required warmup epochs to stabilize randomly-initialized projection layers
- Higher LR (3e-5) with het-loss compensated for exp(-s) gradient attenuation
- The foundation model's representations, pretrained on general time series data, were not well-suited for daily FX returns

### 11.2 Phase 2: MLP From Scratch (Experiments 51-90)

Pivoted to training simple models from scratch. The breakthrough came at experiment 58 (residual MLP).

```
  Performance Trajectory (Composite Score)
  =========================================

  +6 |                                          *****  champion (+5.50)
     |                                         *
  +5 |                                        *
     |                                       *
  +4 |
     |
  +3 |
     |
  +2 |
     |                                      *  residual skip added (exp 58)
  +1 |            *****  LFM2 plateau       *
     |           *       (+1.40)           *
   0 |----------*---------*---------*------*---------->
     0         10        20        30     50   60  90   experiment
     |<---- LFM2 phase ---->|              |<- MLP ->|
```

### 11.3 Exhausted Optimization Axes

| Axis | Values Tried | Best | Notes |
|------|-------------|:----:|-------|
| Architecture | plain MLP, **residual MLP** | residual | 5x improvement from skip connection |
| Hidden dim | 512, **128** | 128 | Smaller = better (Gu, Kelly & Xiu 2020) |
| Head dim | 256, **64** | 64 | Matches hidden reduction |
| Learning rate | 3e-4, **5e-4**, 7e-4 | 5e-4 | Higher LR enabled by residual stability |
| Epochs | 20, **50**, 100 | 50 | Diminishing returns above 50 |
| Head dropout | 0.1, **0.15**, 0.2 | 0.15 | Balances fold 2 vs other folds |
| Huber delta | **0.5**, 1.0 | 0.5 | Tighter works better for residual arch |
| Seq len | **10**, 20 | 10 | Longer windows add noise, not signal |
| Weight decay | **1e-5**, 1e-3 | 1e-5 | 1e-3 was dead weight on MLP |
| BatchNorm | **off**, on | off | Removes regime-scale info |
| Seeds | **0**, 42, 99 | 0 | All seeds positive; seed 0 is best |

### 11.4 Key Lessons Learned

1. **Architecture > hyperparameters.** The single biggest improvement (5x) came from adding a skip connection, not from tuning LR or batch size. All hyperparameter experiments combined produced less than 2x improvement.

2. **Foundation models are not always better.** LFM2-350M with 350M pretrained parameters (500K trainable) was decisively beaten by a 301K-parameter residual MLP trained from scratch on domain-specific FX data. This aligns with the Re(Visiting) TSFMs paper (2025) and the Category Error position paper (2026).

3. **Smaller networks generalize better on small financial datasets.** Hidden dim 128 beat 512. Gu, Kelly & Xiu (2020) predicted this: with only 2,478 training samples, large networks memorize rather than generalize.

4. **BatchNorm is harmful for regime-aware financial models.** It normalizes away the scale differences between market regimes (high-vol GFC vs low-vol 2019), removing information the model needs.

5. **Per-fold diagnosis is essential.** Aggregate metrics hide fold-level problems. The champion has fold 2 Sharpe +1.17 (weakest) vs fold 6 Sharpe +9.95 (strongest). Understanding WHY fold 2 is weak (post-crash recovery, extreme variance) prevents wasting experiments on the wrong hypothesis.

6. **Consecutive discards mean the hypothesis is wrong.** After 4 failed hyperparameter tweaks, the correct action was to stop tuning and try a structural change.
