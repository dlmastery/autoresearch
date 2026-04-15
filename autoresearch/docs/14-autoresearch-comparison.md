# 14 - Autonomous ML Research: Landscape Comparison & Adaptation Plan

**Date:** 2026-04-06
**Scope:** Comparison of our AutoResearch optimizer with Karpathy's autoresearch and the broader autonomous ML research ecosystem
**Goal:** Identify concrete improvements ranked by impact for our FX prediction system

---

## 1. Executive Summary

Our autonomous optimizer already implements a solid single-agent hill-climbing loop for FX Sharpe improvement. Karpathy's autoresearch (March 2026) demonstrates that even a minimal three-file architecture with git-based version control can drive meaningful improvements (11% faster time-to-GPT-2-quality in 700 experiments). The broader ecosystem -- Sakana AI's AI Scientist, DeepMind's FunSearch, AIDE's tree search, OpenAI's MLE-bench, and AutoML-Zero -- reveals several high-impact patterns we can adopt: **tree-structured search** (not just greedy hill-climbing), **population-based exploration**, **failure analysis feedback**, and **structured experiment logging**. This document maps each system's architecture, identifies what we do well, and provides a ranked roadmap for improvements.

---

## 2. System-by-System Overview

### 2.1 Karpathy's autoresearch (March 2026)

**Repository:** [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)

**What it is:** A minimal autonomous ML experimentation framework where an AI coding agent (Claude, Codex, etc.) iteratively modifies a single training script (`train.py`) for a GPT-style language model, runs a fixed 5-minute training experiment, and keeps or discards changes based on a single metric (validation bits-per-byte).

**Architecture:**
- **Three files:** `prepare.py` (immutable constants, data loading, evaluation), `train.py` (agent-editable model + optimizer + training loop), `program.md` (human-written instructions for the agent)
- **Metric:** `val_bpb` (validation bits-per-byte) -- lower is better, vocabulary-size-independent
- **Version control:** Git-based -- commit each change, `git reset --hard HEAD~1` on failure
- **Logging:** `results.tsv` with commit hash, val_bpb, peak VRAM, status, description
- **Loop:** Agent reads code -> forms hypothesis -> modifies train.py -> commits -> runs 5-min training -> checks metric -> keep or revert -> repeat indefinitely

**Key results:**
- ~700 experiments over 2 days, ~20 genuine improvements
- 11% reduction in time-to-GPT-2-quality (2.02h -> 1.80h)
- With 16 parallel GPUs (SkyPilot): 910 experiments in 8 hours, val_bpb from 1.003 to 0.974 (2.87% improvement), 9x faster than sequential
- Shopify CEO Tobi Lutke: 19% improvement in model quality after 37 experiments overnight

**Strengths:**
- Extreme simplicity (3 files, one Markdown orchestration document)
- Fixed time budget makes all experiments directly comparable
- Git-based version control gives complete experiment history
- Agent-agnostic (works with any coding LLM -- Claude, Codex, etc.)
- "NEVER STOP" instruction keeps the agent running indefinitely

**Limitations:**
- Single-file modification (only train.py)
- Greedy hill-climbing (no tree search or backtracking)
- No structured failure analysis or learning from errors
- Single-agent, single-GPU by default (parallel scaling requires external orchestration)
- No formal hypothesis tracking or experiment categorization

---

### 2.2 Sakana AI's AI Scientist (v1: Aug 2024, v2: Apr 2025)

**Repository:** [github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) / [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)

**What it is:** A fully automated pipeline that generates research ideas, implements experiments, runs them, analyzes results, writes LaTeX papers, and reviews them -- producing complete ML conference-style manuscripts at ~$15/paper.

**Architecture (v2):**
1. **Idea Generation:** Brainstorms research directions from code templates, uses Semantic Scholar for novelty verification
2. **Experimentation:** Implements proposed algorithms via code generation, runs experiments, generates visualizations
3. **Paper Writing:** Produces LaTeX manuscripts with literature citations
4. **Automated Review:** LLM-powered peer review using conference standards
5. **v2 addition:** Progressive agentic tree-search managed by a dedicated experiment manager agent

**Key results:**
- Papers reach "Weak Accept" standards at ML conferences using frontier LLMs
- v2 produced the first entirely AI-generated peer-review-accepted workshop paper
- Cost: ~$6-15 per complete paper

**Limitations (from independent evaluation -- arxiv:2502.14297):**
- 42% of experiments failed due to coding errors
- Poor novelty assessment (misclassifies established concepts as novel)
- Median 5 citations per paper, mostly outdated
- Code modifications averaged only 8% additional characters per iteration
- Structural errors: missing figures, repeated sections, placeholder text
- Quality resembles "a rushed undergraduate paper"

---

### 2.3 DeepMind's FunSearch (Dec 2023)

**Paper:** [Nature (2023)](https://www.nature.com/articles/s41586-023-06924-6) | [Blog](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)

**What it is:** An evolutionary program search system that pairs an LLM (PaLM 2) with an automated evaluator to discover novel programs for mathematical and combinatorial problems.

**Architecture:**
1. **LLM (Creative Generator):** Generates candidate programs in code form
2. **Automated Evaluator:** Tests correctness, filters hallucinations
3. **Population Pool:** Maintains a set of high-performing programs
4. **Island Model:** Multiple independent populations evolving in parallel, with periodic migration of top programs between islands -- prevents premature convergence

**Key design decisions:**
- Searches the space of *programs* (how to solve), not *solutions* (what the solution is)
- Favors compact, low-complexity solutions (Kolmogorov complexity bias)
- Parallel execution across islands for diversity
- Outputs are human-readable code, not opaque neural network weights

**Key achievements:**
- Discovered largest cap sets in 20 years (longstanding open mathematical problem)
- Outperformed established bin-packing heuristics
- Dec 2024 update: human-AI collaboration surpassed top-percentile competitive programmers

---

### 2.4 AIDE (Weco AI, Feb 2025)

**Paper:** [arxiv:2502.13138](https://arxiv.org/abs/2502.13138)

**What it is:** An AI-driven exploration agent that treats ML engineering as a tree-search problem, systematically refining solutions at the code level rather than within predefined search spaces.

**Architecture:**
- **Solution Tree:** Each node is a complete Python script; edges represent improvement attempts
- **Three operations:** Draft (new solutions from scratch), Debug (repair broken solutions), Improve (atomic refinements to working solutions)
- **Selection policy:** Explore diverse initial solutions first, then intensify on best performer; constrain debug attempts for broken branches
- **Summarization operator:** Extracts performance metrics, hyperparameters, and debugging hints rather than appending full history (manages context window)

**Key distinction from hill climbing:** Maintains the complete solution tree and can backtrack to different branches, preventing entrapment in local optima.

**Key results:**
- Weco-Kaggle Lite: 51.38% "Exceeds % of humans" vs. H2O AutoML's 35.34%
- MLE-Bench with o1-preview: medals in 16.9% of Kaggle competitions (4x the next agent)
- Pass@8 doubles medal rate to 34.1% (benefits from multiple attempts)
- Outperformed human ML experts on Triton kernel optimization under time constraints

---

### 2.5 OpenAI's MLE-bench (Oct 2024)

**Paper:** [arxiv:2410.07095](https://arxiv.org/abs/2410.07095) | [GitHub](https://github.com/openai/mle-bench)

**What it is:** A benchmark for evaluating how well AI agents perform at ML engineering, curating 75 Kaggle competitions as test tasks.

**Key findings:**
- Best result: o1-preview + AIDE scaffolding achieves Kaggle bronze in 16.9% of competitions
- Performance doubles with multiple attempts (pass@1 16.9% -> pass@8 34.1%)
- Success rates span 100% (established datasets) to 0% (recent Kaggle challenges)
- Key challenges: long-term planning and reducing hallucination

**Relevance to us:** Demonstrates that code-level search (AIDE) significantly outperforms traditional AutoML, and that multiple attempts per problem dramatically improve success rates.

---

### 2.6 MLAgentBench (Stanford, Oct 2023)

**Paper:** [arxiv:2310.03302](https://arxiv.org/abs/2310.03302) | [GitHub](https://github.com/snap-stanford/MLAgentBench)

**What it is:** A benchmark suite of 13 end-to-end ML experimentation tasks where agents autonomously develop or improve ML models.

**Key findings:**
- Claude v3 Opus: 37.5% average success rate (best at publication time)
- Success varies enormously by task familiarity
- Key failure modes: long-term planning and hallucination
- Environment directly resembles what human researchers see (file system, compute, analysis)

---

### 2.7 Google's AutoML-Zero (ICML 2020)

**Paper:** [arxiv:2003.03384](https://arxiv.org/abs/2003.03384) | [GitHub](https://github.com/google-research/google-research/tree/master/automl_zero)

**What it is:** An evolutionary system that discovers complete ML algorithms from scratch using only basic mathematical operations as building blocks.

**Architecture:**
- Starts from empty programs (no human-designed templates)
- Uses evolutionary methods (mutation, crossover, selection) to evolve complete ML algorithms
- Search space: sequences of basic math operations (add, multiply, dot product, etc.)
- Evaluation: task performance on classification/regression problems

**Key discoveries:**
- Rediscovered linear regression with gradient descent
- Rediscovered 2-layer neural networks with backpropagation
- Found algorithms that surpass hand-designed baselines of comparable complexity

**Relevance to us:** Demonstrates that the search space itself matters enormously -- starting from minimal building blocks can yield surprising discoveries, but requires massive compute. Constraining the search space (as we do with 3 modifiable files) is a pragmatic tradeoff.

---

### 2.8 Traditional AutoML (AutoKeras, AutoGluon, NAS)

**AutoKeras** ([arxiv:1806.10282](https://arxiv.org/abs/1806.10282)): Neural architecture search using Bayesian optimization. Efficient but limited to predefined architecture families. Faster training but lower performance on complex datasets.

**AutoGluon** ([auto.gluon.ai](https://auto.gluon.ai/)): Multi-layer stacking of models across tabular, image, and text. Best overall balance of accuracy and efficiency in 2025 evaluations. Automates feature engineering, model selection, and hyperparameter tuning.

**NAS (Neural Architecture Search):** Automates architecture design using reinforcement learning, evolutionary methods, or differentiable search. Optimizes for user-defined metrics (accuracy, model size, inference time). Requires significant compute but can discover architectures that outperform human designs.

**Relevance to us:** These operate at a different level of abstraction -- they search architecture/hyperparameter spaces, not code spaces. Our optimizer is more flexible (it can modify arbitrary code) but less structured in its search.

---

### 2.9 AgentRxiv / Agent Laboratory (2025)

**Website:** [agentrxiv.github.io](https://agentrxiv.github.io/)

**What it is:** A multi-agent framework that coordinates specialized LLM agents through Literature Review, Experimentation, and Report Writing phases.

**Architecture:**
- **Specialized agents:** PhD, Postdoc, ML Engineer, Professor roles
- **Tools:** mle-solver (experiment execution), paper-solver (manuscript writing)
- **Collaborative research:** Agents build on each other's outputs iteratively
- **Results:** Improved MATH-500 accuracy from 70.2% to 78.2% baseline through iterative research

**Relevance to us:** Demonstrates the value of multi-agent collaboration with specialized roles -- a pattern we could adapt for brainstorming, code generation, and analysis phases.

---

## 3. Comparison Matrix

| Feature | **Our System** | **Karpathy** | **AI Scientist** | **FunSearch** | **AIDE** |
|---------|---------------|-------------|-----------------|-------------|---------|
| **Search strategy** | Greedy hill-climb | Greedy hill-climb | Tree search (v2) | Population evolution + islands | Tree search + backtracking |
| **Modifiable scope** | 3 files (backbone, features, train) | 1 file (train.py) | Template-based code | Single function | Full Python scripts |
| **Metric** | Avg Sharpe (7-fold walk-forward) | val_bpb (5-min training) | Paper quality / experiment results | Task-specific evaluator | Kaggle leaderboard / validation |
| **Version control** | JSON state + file backups | Git commits + git reset | Not specified | Population pool | Solution tree |
| **Failure handling** | Revert + log error type | Git reset + log to TSV | Retry / skip | Evaluator filters | Debug operation (dedicated) |
| **Learning from failures** | Past experiment summaries in prompt | Agent reads results.tsv | Reviews inform next cycle | Low-scoring programs discarded | Tree history + summarization |
| **Experiment categorization** | 7 categories (feature eng, architecture, etc.) | Unstructured (agent decides) | Research domain templates | N/A (math problems) | Draft/Debug/Improve operations |
| **Risk assessment** | Low/medium/high per idea | None | None | None | None |
| **Multi-agent** | No | No (single agent) | Yes (v2: experiment manager) | Yes (island model) | No (single agent) |
| **Parallel execution** | No | No (but SkyPilot scaling shown) | Not specified | Yes (parallel islands) | No (sequential tree expansion) |
| **LLM used** | Claude Sonnet 4 | Agent-agnostic (Claude, Codex) | GPT-4, Claude | PaLM 2 | GPT-4o, o1-preview |
| **Orchestration** | Python (agent_loop.py) | Markdown (program.md) | Python pipeline | Custom evolutionary loop | Python framework |
| **Time per experiment** | ~5-30 min (7-fold eval) | Fixed 5 min | ~hours per paper | Seconds (evaluator) | Minutes per node |
| **Syntax validation** | py_compile before execution | Agent handles errors | Runtime error handling | Evaluator rejects invalid | Debug operation |
| **Crash recovery** | JSON state persistence | Git history | Not specified | Population persistence | Solution tree persistence |
| **Experiment count** | ~10-12 per run | ~100 overnight, ~700 in 2 days | ~dozens per run | Thousands | ~dozens per tree |

---

## 4. What Our System Already Does Well

### 4.1 Rigorous Evaluation Framework (Major Strength)
Our 7-fold regime-aware walk-forward evaluation with 90-day purge gaps and 21-day embargo windows is far more rigorous than any other system in this comparison. Karpathy uses a single 5-minute training run. AIDE uses single train/val splits. The AI Scientist's evaluations are often flawed (42% failure rate). Our framework follows Lopez de Prado (2018) best practices -- this is a genuine competitive advantage for financial applications.

### 4.2 Protected Evaluation Boundary
The strict separation between modifiable files (backbone.py, features.py, train.py) and immutable files (splits.py, metrics.py, baseline.py) is well-designed. This is similar to Karpathy's prepare.py/train.py split but more granular -- we protect the evaluation metric computation AND the data splits, not just the data loading.

### 4.3 Structured Experiment Categories
Our 7 experiment categories (feature_engineering, model_architecture, training_hyperparams, head_design, regularization, data_preprocessing, ensemble) provide more structure than Karpathy's unstructured approach. The brainstorm prompt explicitly asks for diverse ideas across categories, which reduces search redundancy.

### 4.4 Risk Assessment
The low/medium/high risk classification with preference for low-risk experiments first is unique to our system. This is a pragmatic choice for a financial domain where catastrophic regressions are expensive to evaluate (each evaluation runs 7 full training folds).

### 4.5 Past Experiment Context
Feeding the last 10 experiments into the brainstorm prompt helps avoid repeating failed approaches. This is similar to what AIDE does with its summarization operator but less structured.

### 4.6 Multi-Backbone Architecture
Our 11-backbone ablation capability (MLP, LSTM, LFM2.5, PatchTST, Mamba2, Informer, XGBoost, etc.) is significantly richer than Karpathy's single GPT model. The optimizer can modify architecture code across multiple backbone types.

### 4.7 Comprehensive Metrics
We track Sharpe, Sortino, PSR, DSR, IC, hit rate, max drawdown, VaR, CVaR, omega ratio, and more. Karpathy tracks a single metric (val_bpb). Our multi-metric reporting enables much richer analysis of what the optimizer is actually improving.

---

## 5. Gaps and Improvement Opportunities

### 5.1 Greedy Hill-Climbing (Critical Gap)

**Problem:** Our optimizer is a strict greedy hill-climber -- it only keeps changes that improve the current Sharpe. If experiment A reduces Sharpe by 0.01 but enables experiment B that improves it by 0.05, we will never discover this because we revert A.

**What others do:**
- **AIDE:** Maintains a solution tree with backtracking -- can explore multiple branches and return to promising earlier states
- **FunSearch:** Population-based evolution with island model -- maintains diverse solutions and prevents premature convergence
- **Karpathy (scaled):** Parallel factorial grids test multiple hypotheses simultaneously

**Impact:** HIGH. This is the single most impactful architectural improvement we could make.

---

### 5.2 No Failure Analysis Feedback (Major Gap)

**Problem:** When an experiment fails (syntax error, evaluation crash, or regression), we log the status but do not feed the error details back to Claude for the next brainstorm. The brainstorm prompt only shows past experiment descriptions and whether they were kept/reverted -- not WHY they failed.

**What others do:**
- **AIDE:** Has a dedicated "Debug" operation that inspects error logs and execution traces to fix broken solutions
- **Karpathy:** Agent reads `tail -n 50 run.log` after crashes and can diagnose issues
- **AI Scientist v2:** Reviews inform the next research cycle

**Impact:** HIGH. Many experiment failures contain valuable signal about what NOT to try.

---

### 5.3 No Structured Experiment Log / TSV Tracking (Moderate Gap)

**Problem:** Our experiment history is stored in a JSON state file with minimal structure. Karpathy's `results.tsv` format with commit hash, metric, VRAM usage, status, and description enables much richer post-hoc analysis.

**What others do:**
- **Karpathy:** `results.tsv` with standardized columns, git commit hashes for full reproducibility
- **AIDE:** Solution tree with performance metrics at each node
- **FunSearch:** Population fitness scores with full evolutionary history

**Impact:** MEDIUM. Better logging enables pattern discovery across experiments.

---

### 5.4 No Git-Based Version Control for Experiments (Moderate Gap)

**Problem:** We use file-level backups (.optimizer_backups/) instead of git commits. This means we cannot easily compare, diff, or replay specific experiments. If we want to return to a previous experiment's code, we have no mechanism to do so.

**What Karpathy does:** Every modification is a git commit. Successful experiments stay in history; failures get `git reset --hard HEAD~1`. This creates a clean, browsable experiment history.

**Impact:** MEDIUM. Git-based tracking provides free experiment reproducibility and diffing.

---

### 5.5 Single-Turn LLM Calls (Moderate Gap)

**Problem:** We make two single-turn Claude calls per experiment (brainstorm + code generation). There is no iterative refinement -- if the generated code has a subtle bug that passes py_compile but fails at runtime, we just discard the experiment.

**What others do:**
- **AIDE:** Draft -> Debug -> Improve cycle with multiple refinement passes
- **Karpathy:** Agent can read error logs and retry with fixes
- **AI Scientist v2:** Iterative refinement through tree search

**Impact:** MEDIUM. Adding a debug/retry step would rescue experiments that fail due to minor coding errors.

---

### 5.6 No train.py in Brainstorm Context (Minor Gap)

**Problem:** The brainstorm prompt includes backbone.py and features.py code, but not train.py code. Yet the optimizer CAN modify train.py (it is in the MODIFIABLE_FILES map). This means Claude proposes training hyperparameter changes without seeing the current training configuration.

**Impact:** MEDIUM. Easy fix that immediately improves code generation quality for training-related experiments.

---

### 5.7 No Multi-File Coordinated Changes (Minor Gap)

**Problem:** Each experiment modifies exactly one file. But some improvements require coordinated changes -- e.g., adding a new feature in features.py AND adjusting the model head in backbone.py to use it. Our current architecture cannot do this in a single experiment.

**What others do:**
- **AI Scientist:** Can modify multiple files per experiment
- **AIDE:** Each node is a complete script (single-file by design, but no constraint on what changes)

**Impact:** LOW-MEDIUM. Most improvements can be expressed as single-file changes, but coordinated changes would unlock feature-architecture co-optimization.

---

### 5.8 No Population Diversity / Exploration-Exploitation Balance (Moderate Gap)

**Problem:** We always build on the current best solution. There is no mechanism to maintain a population of diverse solutions or to balance exploration vs. exploitation.

**What others do:**
- **FunSearch:** Island model with multiple populations evolving independently, periodic migration
- **AutoML-Zero:** Full evolutionary approach with mutation, crossover, selection
- **AIDE:** Tree structure naturally maintains diversity through branching

**Impact:** MEDIUM. Especially relevant for our problem where FX markets have regime-dependent optimal strategies.

---

### 5.9 No Curriculum / Progressive Difficulty (Minor Gap)

**Problem:** We jump straight to full 7-fold evaluation for every experiment. A quick smoke-test on 1-2 folds would filter out bad ideas 3-5x faster.

**What Karpathy does:** Fixed 5-minute budget means every experiment is equally cheap. Our 7-fold evaluation is much more expensive, making screening even more valuable.

**Impact:** MEDIUM. A two-stage evaluation (quick screen + full eval) could triple our experiment throughput.

---

## 6. Specific Improvements Ranked by Impact

### Tier 1: High Impact, Moderate Effort

| # | Improvement | Effort | Expected Impact | Inspired By |
|---|-----------|--------|----------------|------------|
| 1 | **Two-stage evaluation** (quick 2-fold screen + full 7-fold for promising ideas) | 2-3 hours | 3x experiment throughput | Karpathy's 5-min budget philosophy |
| 2 | **Failure analysis feedback** (include error details, stack traces, and failed code diffs in brainstorm context) | 1-2 hours | 30-50% reduction in repeated failure modes | AIDE's Debug operation, Karpathy's log reading |
| 3 | **Include train.py in brainstorm context** | 30 min | Immediate improvement in training-related experiments | Common sense gap |
| 4 | **Debug/retry loop** (on runtime errors, feed error back to Claude for one fix attempt before discarding) | 2-3 hours | Rescue 20-40% of currently-discarded experiments | AIDE's Debug operation |

### Tier 2: High Impact, Higher Effort

| # | Improvement | Effort | Expected Impact | Inspired By |
|---|-----------|--------|----------------|------------|
| 5 | **Solution tree with backtracking** (maintain top-3 solution branches, not just current best) | 1-2 days | Escape local optima, discover synergistic changes | AIDE's tree search |
| 6 | **Git-based experiment tracking** (commit each experiment, tag results, enable diffing) | 3-4 hours | Full reproducibility, richer post-hoc analysis | Karpathy's git workflow |
| 7 | **Structured experiment log** (TSV/CSV with metric breakdown, code diff size, category, hypothesis, result, error type) | 2-3 hours | Pattern discovery, meta-learning about what works | Karpathy's results.tsv |

### Tier 3: Medium Impact, Significant Effort

| # | Improvement | Effort | Expected Impact | Inspired By |
|---|-----------|--------|----------------|------------|
| 8 | **Multi-file coordinated changes** (allow modifying backbone.py + features.py in same experiment) | 3-4 hours | Unlock feature-architecture co-optimization | AI Scientist's multi-file edits |
| 9 | **Population-based search** (maintain 3-5 diverse solution variants, evolve in parallel) | 2-3 days | Better exploration of solution space | FunSearch's island model |
| 10 | **Experiment hypothesis formalization** (require testable hypothesis + expected effect size before running) | 2-3 hours | More disciplined search, better learning signal | Scientific method best practices |

### Tier 4: Lower Impact or Exploratory

| # | Improvement | Effort | Expected Impact | Inspired By |
|---|-----------|--------|----------------|------------|
| 11 | **Multi-agent roles** (separate brainstorm agent, code gen agent, analysis agent) | 1-2 days | Specialized prompts for each phase | AgentRxiv's role specialization |
| 12 | **Semantic Scholar integration** (search for relevant papers before proposing ideas) | 1 day | Better-grounded hypotheses | AI Scientist's literature search |
| 13 | **Markdown-based orchestration** (replace Python agent loop with a program.md-style instruction set) | 1 day | Agent-agnostic, easier iteration on research strategy | Karpathy's program.md |
| 14 | **Cross-experiment meta-analysis** (periodically analyze all experiments to find patterns) | 3-4 hours | Learn which categories/changes tend to succeed | FunSearch's population analysis |

---

## 7. Implementation Roadmap

### Phase 1: Quick Wins (Week 1)

**Goal:** 3x experiment throughput + better code generation quality

**1a. Add train.py to brainstorm context**
```
# In optimizer/prompts.py BRAINSTORM_PROMPT, add:
=== Current training code (model/train.py) ===
{train_code}
```

**1b. Two-stage evaluation**
```
Stage 1: Run 2 folds (quickest regimes) -> if Sharpe drops > 20%, skip
Stage 2: Run full 7 folds -> keep/revert as now
```
This lets us screen ~3x more ideas in the same wall-clock time.

**1c. Failure analysis in brainstorm prompt**
```
# Enhance past_experiments format:
[REVERTED] Add momentum features (sharpe=0.12, reason: no improvement)
[SYNTAX_ERROR] Unfreeze layers (error: IndentationError line 45)
[EVAL_ERROR] Custom loss function (error: RuntimeError: shape mismatch)
```

**1d. Debug/retry loop**
```
On runtime error:
  1. Feed error + code to Claude with "fix this error" prompt
  2. Validate syntax of fix
  3. If valid, retry evaluation once
  4. If still fails, revert and log
```

### Phase 2: Structural Improvements (Week 2)

**Goal:** Escape local optima + full reproducibility

**2a. Git-based experiment tracking**
```
Before each experiment:
  git add -A && git commit -m "pre-experiment-{N} baseline"
After code modification:
  git add backbone.py features.py train.py
  git commit -m "experiment-{N}: {description}"
On revert:
  git revert HEAD --no-edit
On keep:
  git tag "kept-{N}-sharpe-{value}"
```

**2b. Solution tree (simplified)**
```python
class SolutionNode:
    code_snapshot: dict[str, str]  # file -> content
    sharpe: float
    parent: Optional[SolutionNode]
    children: list[SolutionNode]
    experiment_id: int

class SolutionTree:
    nodes: list[SolutionNode]
    
    def select_branch(self) -> SolutionNode:
        """Select node to expand: best performer or least-explored branch"""
        # UCB1-style: balance exploitation (high Sharpe) and exploration (fewer children)
        ...
    
    def backtrack(self, node: SolutionNode):
        """Restore code from a previous node"""
        for path, content in node.code_snapshot.items():
            write_file(path, content)
```

**2c. Structured experiment log (CSV)**
```csv
id,timestamp,category,description,hypothesis,risk,target_file,code_diff_lines,
sharpe_new,sharpe_delta,stage1_sharpe,status,error_type,error_msg,kept,
branch_id,parent_id,git_commit
```

### Phase 3: Advanced Search (Week 3-4)

**Goal:** Population diversity + meta-learning

**3a. Population-based search (3 branches)**
```
Maintain 3 independent solution branches:
  Branch A: Current best (exploitation)
  Branch B: Best from different category than A (diversity)
  Branch C: Random restart from baseline (exploration)

Each experiment cycle:
  1. Select branch (round-robin or UCB1)
  2. Run experiment on selected branch
  3. If improvement: update branch
  4. Every 5 experiments: cross-pollinate (try Branch A's features on Branch B's architecture)
```

**3b. Experiment meta-analysis (every 10 experiments)**
```
Feed all experiment results to Claude:
"Analyze these N experiments. Which categories tend to improve Sharpe?
 Which changes tend to cause regressions? What patterns do you see?
 Based on this analysis, propose the next experiment strategy."
```

**3c. Multi-file coordinated changes**
```python
# Allow MODIFY_PROMPT to specify multiple files:
MODIFY_PROMPT_MULTI = """
You may modify up to 2 files in a single experiment.
Return a JSON object with file paths as keys and complete file contents as values:
{
  "model/backbone.py": "...",
  "data/features.py": "..."
}
"""
```

### Phase 4: Exploration (Month 2+)

- Multi-agent roles (brainstorm specialist + code gen specialist + analysis specialist)
- Regime-aware optimization (different strategies for crisis vs. plateau periods)
- Formal hypothesis testing with power analysis
- Integration with literature search (Semantic Scholar API)
- Overnight scheduling with email/Slack notifications on breakthroughs

---

## 8. Adaptation Recommendations for FX Prediction

### 8.1 Search Space Design

**Current:** 7 experiment categories across 3 files
**Recommended additions:**
- **Loss function engineering:** Custom asymmetric losses (penalize false signals more than missed signals)
- **Ensemble strategies:** Cross-backbone ensembles (e.g., LFM2.5 + XGBoost blend)
- **Sequence length exploration:** Varying the 60-day lookback window
- **Target engineering:** Alternative targets beyond raw forward returns (e.g., risk-adjusted returns, binary direction, quantile targets)
- **Feature selection:** Automated feature importance + pruning low-signal features

### 8.2 Experiment Prioritization

Adopt a modified UCB1 (Upper Confidence Bound) strategy:
```
priority(category) = avg_improvement(category) + C * sqrt(ln(N) / n_category)
```
Where `C` balances exploitation (categories that have worked) vs. exploration (under-explored categories). Start with `C = 1.0` and tune.

### 8.3 Code Generation Quality

**Current issues:** Single-turn generation, no iterative refinement
**Recommendations:**
1. Include ALL modifiable file contents in the code generation prompt (not just the target file)
2. Add a "self-review" step: ask Claude to verify the generated code against the experiment description before applying
3. Add runtime validation: beyond py_compile, try importing the module and running a quick shape check
4. Provide explicit examples of well-structured modifications in the prompt

### 8.4 Result Analysis and Learning from Failures

**Failure taxonomy to track:**
- `syntax_error`: Code doesn't compile (LLM coding quality issue)
- `import_error`: Missing dependency or circular import
- `shape_mismatch`: Tensor dimension errors (common with architecture changes)
- `nan_loss`: Training diverged (learning rate too high, unstable architecture)
- `oom_error`: Out of memory (model too large)
- `regression`: Code works but Sharpe decreased
- `marginal`: Sharpe change < 0.001 (noise, not signal)

Feed this taxonomy into the brainstorm prompt so Claude can avoid known failure patterns.

### 8.5 Multi-Agent Collaboration

For FX-specific adaptation:
- **Quant Researcher Agent:** Proposes hypotheses grounded in FX market microstructure
- **ML Engineer Agent:** Implements the hypothesis as code changes
- **Risk Analyst Agent:** Reviews proposed changes for overfitting risk, regime sensitivity
- **Meta-Analyst Agent:** Periodically reviews all experiments and suggests strategic pivots

### 8.6 Curriculum Learning

Order experiments by complexity and risk:
1. **Phase 1 (Low risk):** Hyperparameter tuning (learning rate, batch size, dropout)
2. **Phase 2 (Medium risk):** Feature engineering (new indicators, feature selection)
3. **Phase 3 (Medium-high risk):** Architecture changes (head design, attention patterns)
4. **Phase 4 (High risk):** Backbone changes (unfreezing layers, model swaps)

### 8.7 Population-Based Approaches for FX

FX markets exhibit regime-dependent behavior. A population approach naturally maps to this:
- **Branch per regime:** One solution optimized for crisis periods, one for trending, one for range-bound
- **Ensemble at inference:** Combine predictions from regime-specialized branches
- **Migration between branches:** If a crisis-optimized feature also helps in trending markets, propagate it

### 8.8 Formal Hypothesis Testing

Before each experiment, require:
1. **Null hypothesis:** "This change will not improve average Sharpe"
2. **Expected effect size:** "Expected Sharpe improvement of 0.02-0.05"
3. **Mechanism:** "Why this should work given FX market properties"
4. **Risk:** "What could go wrong and how likely"

After evaluation, record:
1. **Actual effect:** Sharpe delta with confidence interval
2. **PSR change:** Statistical significance of improvement
3. **Regime breakdown:** Did it help in all regimes or just some?

---

## 9. Key Takeaways

1. **Our evaluation framework is our biggest advantage.** No other system in this space has regime-aware walk-forward validation with purge/embargo. Protect this.

2. **Karpathy's key insight: simplicity scales.** His three-file architecture is deliberately minimal. We should resist adding complexity unless it provides measurable improvement.

3. **AIDE's tree search is the most impactful architectural pattern we could adopt.** Moving from greedy hill-climbing to tree-based search with backtracking would fundamentally improve our exploration capability.

4. **FunSearch's island model solves our regime problem.** Maintaining multiple independent solution populations naturally maps to FX regime diversity.

5. **Two-stage evaluation is the highest-ROI quick win.** A fast 2-fold screen before full 7-fold evaluation would approximately triple our experiment throughput at minimal implementation cost.

6. **Failure analysis is free information we are currently discarding.** Every failed experiment tells us something about the search space boundary.

7. **The field is converging on code-level search as the right abstraction.** All recent systems (Karpathy, AIDE, FunSearch, AI Scientist) search in code space, not parameter space. Our approach is well-aligned with this trend.

---

## 10. Sources

- [Karpathy autoresearch - GitHub](https://github.com/karpathy/autoresearch)
- [Karpathy autoresearch - program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)
- [Karpathy announcement - X/Twitter](https://x.com/karpathy/status/2030371219518931079)
- [Karpathy on collaborative agents - X/Twitter](https://x.com/karpathy/status/2030705271627284816)
- [Scaling Autoresearch with SkyPilot](https://blog.skypilot.co/scaling-autoresearch/)
- [Karpathy autoresearch - DataCamp Guide](https://www.datacamp.com/tutorial/guide-to-autoresearch)
- [Karpathy autoresearch - DeepWiki](https://deepwiki.com/karpathy/autoresearch)
- [Karpathy autoresearch - Data Science Dojo](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/)
- [Sakana AI - The AI Scientist](https://sakana.ai/ai-scientist/)
- [AI Scientist v2 - GitHub](https://github.com/SakanaAI/AI-Scientist-v2)
- [AI Scientist v2 - arxiv:2504.08066](https://arxiv.org/abs/2504.08066)
- [Evaluating AI Scientist - arxiv:2502.14297](https://arxiv.org/abs/2502.14297)
- [FunSearch - DeepMind Blog](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
- [FunSearch - Nature (2023)](https://www.nature.com/articles/s41586-023-06924-6)
- [FunSearch - GitHub](https://github.com/google-deepmind/funsearch)
- [AIDE - arxiv:2502.13138](https://arxiv.org/abs/2502.13138)
- [MLE-bench - OpenAI](https://openai.com/index/mle-bench/)
- [MLE-bench - arxiv:2410.07095](https://arxiv.org/abs/2410.07095)
- [MLAgentBench - arxiv:2310.03302](https://arxiv.org/abs/2310.03302)
- [MLAgentBench - GitHub](https://github.com/snap-stanford/MLAgentBench)
- [AutoML-Zero - arxiv:2003.03384](https://arxiv.org/abs/2003.03384)
- [AutoML-Zero - GitHub](https://github.com/google-research/google-research/tree/master/automl_zero)
- [Auto-Keras - arxiv:1806.10282](https://arxiv.org/abs/1806.10282)
- [AgentRxiv](https://agentrxiv.github.io/)
- [AutoML NAS Overview](https://www.automl.org/nas-overview/)
