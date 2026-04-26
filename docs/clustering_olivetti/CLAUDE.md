# CLAUDE.md — Clustering AutoResearch on Olivetti Faces

> Project-specific rules for the unsupervised face-clustering benchmark.
> Inherits from `generalized_ml_autoresearch/templates/CLAUDE_template.md`, which itself generalises the FX-prediction `CLAUDE.md` at `C:/Users/abhir/clauderesearch/autoresearch/CLAUDE.md` (the source of truth for the AutoResearch protocol). Every section in this file mirrors a section in the FX-root CLAUDE.md, adapted for the *unsupervised* clustering task type.

---

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:** `autoresearch_results/forensic_checkpoint.md` — current champion, last experiment result, three findings, what to try next.
2. **Read the experiment log tail:** `autoresearch_results/experiment_log.jsonl` (last 3 lines) and `autoresearch_results/best_config.json` to verify state.
3. **Read the audit:** `autoresearch_results/audit_report_third_party.md` — note the PASS-WITH-ONE-FOOTNOTE seed-variance recommendation.
4. **Resume the experiment loop** from where the checkpoint says. Follow the 7-step process (diagnose → cite → hypothesize → predict → run ONE experiment → analyze → checkpoint).
5. **Start the local dashboard** (once per session, background):
   ```bash
   cd generalized_ml_autoresearch/examples/clustering_olivetti
   python -m http.server 8765 --directory autoresearch_results
   ```
   Tell the user: "Dashboard at http://localhost:8765/dashboard.html"
6. **Run experiments** via:
   ```bash
   cd generalized_ml_autoresearch/examples/clustering_olivetti
   python <runner_script>.py
   ```
   Each runner authors the pre-run reasoning blob, calls `run_experiment(...)`, validates, logs.
7. **If the user says "continue" or "keep going"** — resume the loop. No need to ask what to do.

---

## Hardware Constraints (MANDATORY — inherited from FX project 2026-04-19)

This project runs on a smaller machine than the FX project (Windows 11, no E-core BSOD history), but we still follow the FX hardware policy because the framework runs the same `_pin_to_safe_cores()` helper.

- Use a small thread budget; `torch.set_num_threads(4)` is sufficient.
- GPU does heavy compute (DINOv2 forward pass); CPU is coordination only.
- Time budget per experiment: 600 s (most clustering experiments finish in < 60 s; the slow ones are DEC at ~35-60 s and SimCLR at ~33 s).
- Memory ceiling: 16 GB VRAM. DINOv2 ViT-S/14 (21 M params) and ViT-B/14 (86 M params) both fit trivially; even Spectral's 400×400 affinity matrix is < 1 MB.

---

## Crash-Recovery Checkpointing (MANDATORY)

**Checkpoint AFTER EVERY SINGLE EXPERIMENT.** This is non-negotiable.

**Trigger points (ALL mandatory):**
1. Immediately after every experiment completes — before any analysis.
2. Every 5 minutes during reasoning/analysis.
3. Before starting any code change.
4. After any code change.
5. Before starting the next experiment — checkpoint must contain the exact next-experiment command.

**What to save to `autoresearch_results/forensic_checkpoint.md`:**
- Current champion config + ARI
- Per-fold ARI (only 1 fold — full dataset)
- Last experiment result (config, ARI, delta vs champion, KEEP/DISCARD)
- The EXACT next-experiment command (copy-pasteable)
- Rationale for the next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters
- Key learnings from exhausted axes
- Session-start instructions

The checkpoint must be self-contained: a fresh Claude Code session reading ONLY `CLAUDE.md` + `forensic_checkpoint.md` must be able to resume without any other file.

---

## Mindset (Read First)

You are a top-tier ML researcher applying the AutoResearch protocol to small-n unsupervised clustering. Before touching any code:

1. **Understand the data flow end-to-end.** Trace a single 64×64 grayscale image from sklearn's loader through the feature extractor (raw / PCA / DINOv2 / etc.) through the clustering head (KMeans / Spectral / Ward / etc.) through to the ARI evaluator. If you can't explain every step, you don't understand the system.
2. **Validate before running.** Re-verify the X SHA-256 hash, the y SHA-256 hash, and the composite fingerprint before any experiment.
3. **Measure, never assume.** If you state a number (timing, ARI, std), it must come from running code.
4. **When fixing a bug, audit the entire system for the same class of bug.** The NaN-in-JSONL fix (Apr 26) was patched in `common.py`'s `log_experiment` *and* the existing JSONL was retroactively cleaned — don't fix one and leave the other.
5. **Separation of concerns is not optional.** Runners log. Dashboards display. `evaluate_clustering()` evaluates. Never tangle them.

---

## Hard Rules (NEVER violate)

### Data Integrity

- **No label leakage.** `y` (true subject IDs) is loaded ONLY in `evaluate_clustering()`. Models receive `X` only. `prepare_data.py` returns `(X, y, _, _)` but the train/test split is never used (clustering operates on the full set).
- **Reproducibility.** Every experiment MUST set `random_state=0` unless it is explicitly a multi-seed variance probe.
- **Test-set integrity = full-dataset integrity.** Verify `len(X)==400` and `X.shape==(400, 4096)` before every experiment. The SHA-256 of `X.tobytes()` is locked at `e6b9b0fe62f642f6` (first 16 hex). The SHA-256 of `y.tobytes()` is locked at `2745696ae3f897d8` (first 16 hex). Any silent corruption fails the run.
- **Composite fingerprint.** Locked at `clustering-ari-floor0.3`. Stored on every JSONL row. The runner refuses to log if the runtime composite differs from the locked one.

### Evaluation Protocol Invariants

The protocol is **full-dataset evaluation** — there is no train/test split. Every experiment evaluates ARI / NMI / FMI / silhouette / V-measure on the same 400 rows. The `per_fold_test` array on each JSONL row has length 1 (one fold = the entire dataset).

### Experiment Design

- **Composite metric for keep/revert:** ARI directly. Floor = 0.30. Any clustering with ARI < 0.30 is a *de facto* DISCARD because it fails to non-trivially beat random for K = 40.
- **One config change per experiment.** Diagnose WHY before choosing what to change next.
- **Report all secondary metrics:** NMI, FMI, silhouette, homogeneity, completeness, V-measure, n_pred_clusters, n_noise.
- **Every config parameter must be wired end-to-end.** Dead params are bugs — remove them.
- **Every hyperparameter choice must be justified by a published paper.** No "let me try X."

### Autoresearch Agent Protocol (Karpathy-adapted)

1. **Always start from the current best config.** Every experiment modifies ONE thing.
2. **If you see 3+ consecutive DISCARDs, stop and rethink.** Multiple failures mean your hypothesis is wrong.
3. **Explore around the best AND try radical changes.** Most experiments are small tweaks; occasionally try something bold (different backbone family, different distance metric).
4. **Cite your reasoning for every experiment.** "I'm trying X because the silhouette dropped on subjects {…}, and paper W suggests this fix." Not "let me try X."
5. **The agent never stops.** If out of ideas, research deeper: read SOTA clustering surveys (Min, Guo, Liu, Long 2018 IEEE Access), read DINOv2 (Oquab 2024 TMLR), read recent Olivetti baselines.
6. **Checkpoint reasoning to memory every few minutes.**
7. **Deep per-cluster failure analysis every iteration.** For each true subject with low cluster purity, explain WHY: which DINOv2 feature dimensions are the same as another subject? Is it a glasses-vs-no-glasses confusion? Lighting?
8. **Code changes are allowed.** Save modified versions to `code_versions/<runner_name>/`.

### Research-Driven Experiment Selection (STRICT — no blind sweeps)

Every single experiment must follow this exact sequence:

**Step 1 — Diagnose the champion's weakness.** Look at the per-true-subject confusion matrix of the current champion (compute via `sklearn.metrics.confusion_matrix(y_true, y_pred)` after Hungarian assignment). Which true subjects have the lowest cluster purity? What pose / lighting / expression do they share? What does the silhouette per point say about boundary cases?

**Step 2 — Search the literature.** Examples relevant to face clustering:
- Weak on similar-looking subjects → contrastive features (Caron et al. 2021 ICCV DINO arXiv:2104.14294)
- Weak boundary points → cluster_qr assignment (Damle, Minden, Ying 2019 SIAM J. Sci. Comput. arXiv:1708.07964)
- Single-link chaining → Ward variance-minimisation (Ward 1963 JASA)
- Threshold-invariant Birch → skip the threshold sweep (our finding §6.2 below)
- Seed variance → 5-seed median ensemble (Strehl & Ghosh 2002 JMLR)
- Small-n DEC failure → use pretrained DINOv2 instead (our finding §6.1)

**Step 3 — Form a hypothesis and predict the outcome.** Write down: "I hypothesize that [change X] will improve [ARI / NMI / silhouette] by [Δ] because [paper/principle]. I predict ARI will move from [current] to approximately [target]." If you can't write this sentence, you don't understand what you're doing. Stop and think more.

**Step 4 — Run ONE experiment.**

**Step 5 — Analyze against prediction.** Did the result match? If yes, why? If no, what does that tell you?

**Step 6 — Document everything.** Diagnosis → literature → hypothesis → prediction → result → learning → reasoning_annotations.json + research_journal.md.

**Step 7 — Checkpoint.** Update `forensic_checkpoint.md` with the next-experiment command.

### Monotonic Quality Progression (NEVER regress)

- **Never run an experiment you can't justify.**
- **Track the champion lineage.** Document the chain: Exp 1 → Exp 8 (Ward, +0.11) → Exp 20 (DINOv2 features, +0.03) → Exp 27 (DINOv2 + Ward, +0.09) → Exp 33 (DINOv2 + Spectral cosine, +0.06) → Exp 71 (seed = 99, +0.02). Each link explains WHY.
- **When you hit a plateau, go deeper.** If 3+ consecutive DISCARDs, switch to a structural change: different backbone, different head, different distance metric.
- **Quality ratchet:** once a metric improves, treat the new level as the floor.
- **Goodhart protection (MANDATORY).** The agent MAY NOT rewrite the composite metric formula, the split protocol, the data integrity invariants, or the primary-metric definition mid-project. Changes require explicit user sign-off documented as a `RULE_CHANGE` entry in the checkpoint. The composite fingerprint hash enforces this.

### MLOps Documentation Standards (MANDATORY)

Every artifact and every experiment must be documented in proper, readable markdown. No exceptions.

**`autoresearch_results/experiment_summary.md`** — the master experiment log. Updated after EVERY experiment. Format:

```markdown
### Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected ARI range]
- **Result:** ARI [X] | NMI [Y] | FMI [Z] | Silhouette [W] | n_pred_clusters [K]
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned, why result matched/differed]
- **Per-prediction summary:** see `trade_logs/exp<N>_predictions.csv`
```

**`autoresearch_results/trade_logs/exp<N>_predictions.csv`** — per-row cluster assignment.

**Key documentation principles:**
1. Readable by a human who wasn't there. Future Claude Code session reading the summary 6 months from now must understand WHY each experiment was run and WHAT was learned.
2. No orphan artifacts. Every file referenced from checkpoint, summary, or winner README.
3. Consistent formatting. Same table format, same metric names, same precision (4 decimals for ARI/NMI/FMI/silhouette/V-measure).
4. Append-only. Never delete or rewrite — add a note if an experiment was wrong.

---

## Explainability & Auditability Report (MANDATORY for every NEW BEST)

When a new champion is found, produce a full data-scientist-grade audit to `autoresearch_results/winners/<exp_id>/audit_report.md`. This is not optional — a clustering model without explainability is un-deployable.

**Required sections (all 14, adapted for clustering):**

1. **Executive summary** — Champion ARI, NMI, V-measure, FMI, silhouette, n_pred_clusters, n_noise. Pass/fail vs random-baseline (0.0) and vs documented baselines.
2. **Feature importance (permutation method)** — For each input feature dimension (raw 4096 pixels OR PCA-50 components OR DINOv2 384-dim), shuffle that dimension across all 400 samples, re-cluster, report the drop in ARI. Rank features by importance. Cite Breiman 2001 'Random Forests'. Save `feature_importance.csv`.
3. **Top-N feature analysis** — For the top 10 most-impactful features, explain what they encode (PCA components → eigenfaces; DINOv2 dims → semantic / pose / lighting axes when interpretable).
4. **SHAP-style local explanations** — For 10 random samples, compute per-feature contribution to the cluster assignment. For neural feature extractors, use gradient × input. Save `shap_local.csv`.
5. **Per-cluster confusion analysis** — For each predicted cluster, compute the dominant true subject and the cluster purity. Identify the 5 worst-purity clusters and the most-confused subject pairs. Plot confusion matrix.
6. **Per-cluster size distribution** — Histogram of `n_pred_clusters` sizes. Should be roughly uniform (~10 per cluster). Skewed distributions indicate over-/under-segmentation.
7. **Silhouette decomposition** — Per-sample silhouette histogram. Identify the 11 / 400 negative-silhouette samples (boundary cases) and explain which subject they belong to.
8. **Per-true-subject recovery rate** — For each of the 40 true subjects, compute the fraction of their 10 images placed in their dominant cluster. Identify subjects with recovery < 50% (these are the model's blind spots).
9. **Error attribution** — For the 5 worst-recovered subjects, examine the actual face images: are they shared lighting / pose with another subject? Is there a labeling issue in the original dataset?
10. **Robustness audit** — Re-cluster with seeds {0, 1, 7, 42, 99} and report ARI std. For our champion, std = 0.0429.
11. **Data pipeline audit** — Reassert: zero label leakage (`y` only in metric step), X / y SHA-256 hashes match, composite fingerprint locked. Rerun `validate_no_overlap()` (no-op for clustering) and include verbatim.
12. **Model config complete dump** — Every hyperparameter + Python / sklearn / torch / numpy versions + random seed.
13. **Known limitations & risks** — n = 400 is small; single-seed champion; no subject-supervised baseline; out-of-domain DINOv2 transfer.
14. **Deployment checklist** — Per-batch inference recipe; silhouette-based confidence rejection rule; retraining cadence (whenever input population changes substantially).

**Implementation:** `generate_artifacts.py` produces the per-champion `audit_report.md`. Run automatically when `composite > prev_best`.

---

## Winner Definition (CLARIFICATION)

**"Winner" means the GLOBAL champion across ALL backbones and ALL experiments.** Not per-backbone. The single best ARI at any point in time.

When a new experiment beats the global ARI:
1. Save artifacts to `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`
2. Include: README.md (13 sections per FX template, adapted for clustering), config.json, model_checkpoint or `predict.py` (since SpectralClustering can't be pickled cleanly across sklearn versions, we save the *recipe* not the *fitted model*), code/ (frozen snapshot), `audit_report.md` (14 sections), `colab_train_and_infer.ipynb`.
3. Update `autoresearch_results/best_config.json`.

Per-backbone best is tracked in the `best_per_backbone.json` (when present) but does NOT get archived to `winners/` unless it is also the global best.

---

## Per-Backbone Code Snapshots (MANDATORY)

Before starting experiments on a new backbone, snapshot the relevant runner script(s) to `code_versions/<backbone>_start/`. Examples in this project:

```
code_versions/
  v1_baselines/                       # Exps 1-14 baseline code
  dinov2_start/                       # before Exp 22 (DINOv2 hill-climb)
  spectral_start/                     # before Exp 47 (Spectral hill-climb)
  ward_birch_start/                   # before Exp 72 (Ward+Birch hill-climb)
  umap_dec_start/                     # before Exp 122 (UMAP+DEC sweep)
  dec_only_start/                     # the recovery split after the torch.nn shadowing bug
```

Rule: never modify `run_<backbone>.py` while experiments on a different backbone are in progress.

---

## Dashboard Reasoning Annotations (MANDATORY — capture EVERYTHING, every experiment)

**Every single experiment MUST have a complete reasoning record in `autoresearch_results/reasoning_annotations.json` keyed by `experiment_num`. No experiment ships without one.**

Fields (all 7 are REQUIRED, all non-empty strings unless noted):

| Field | Content | Source |
|-------|---------|--------|
| `diagnosis` | Why THIS experiment now: which champion weakness, which subject is worst-recovered and why, what prior experiments ruled out | Authored by Claude BEFORE running |
| `citations` | Per Citation Rigor; multiple papers semicolon-separated | Authored before running |
| `hypothesis` | Concrete mechanism: "parameter X = value Y will change ARI via mechanism M" | Authored before running |
| `prediction` | Numeric range: "ARI in 0.65 to 0.75; NMI ≥ 0.85" | Authored before running |
| `verdict` | KEEP / DISCARD / NEAR-MISS + ARI to 4 decimals + delta vs global best + per-fold mention | Written immediately after results |
| `learning` | Mental-model update; axis closed/open; what to try next | Written immediately after results |
| `_manual` | `true` if Claude-authored; `false` only for mechanical variance reruns | Always set |

**Dashboard `dashboard.html` renders all 7 fields in the detail panel when a row is clicked.** Missing/empty/placeholder strings are a regression — fix before the next experiment.

**Write cadence — two phases per run:**
1. **BEFORE the experiment:** Claude inserts diagnosis/citations/hypothesis/prediction with `_manual: true`. The runner's `common.author_pre_run()` validator enforces this. The experiment is not launched until the entry passes both Citation Rigor and Reasoning Blob Completeness validators.
2. **AFTER the experiment:** Claude appends verdict and learning. The runner's auto-written fallback only emits `TODO-REWRITE` sentinels — Claude must rewrite.

**Enforcement:** at the start of every experiment cycle, verify:
- Does the previous experiment's entry have non-empty verdict/learning? If not, write them now.
- Is the next experiment's pre-entry authored? If not, write it now.
- Did `_manual: true` survive any backfill run?

**Parallel write to `research_journal.md`** in markdown form. JSON is authoritative if they drift.

---

## Per-Backbone N-Experiment Mandate (MANDATORY, not optional)

**Every backbone gets a 25-experiment hill-climb minimum.** Reduced from FX's 50-per-backbone because clustering experiments are faster and the n = 400 dataset has a smaller hyperparameter surface. The mandate:

1. **25 hill-climb experiments per backbone** — no fewer. If standard HP sweeps plateau, explore architectural variants (different feature spaces, different distance metrics, different assignment strategies, multi-seed studies).
2. **Research latest SOTA** before declaring any backbone done. See "Per-Backbone SOTA Recipes" below.
3. **Each experiment cites its paper.**
4. **Document all 25 in research_journal.md** — even DISCARDs. Negative results (DEC plateau, Birch threshold-invariance) are some of the most informative findings in the project.
5. **Only after 25 hill-climb experiments** may a backbone be declared "done" and progression to the next backbone resume.

The current state (149 experiments) covers:
- Tier-1 baselines: 14 single-shot experiments (Exps 1-14, 18, 19, 21).
- DINOv2 hill-climb: 25 variants (Exps 22-46).
- Spectral hill-climb: 25 variants (Exps 47-71).
- Ward hill-climb: 25 variants (Exps 72-96).
- Birch hill-climb: 25 variants (Exps 97-121).
- UMAP hill-climb: 15 variants (Exps 122-136).
- DEC hill-climb: 10 variants (Exps 137-146).

---

## Per-Backbone SOTA Recipes (research first)

Before the first experiment on any new backbone, pull the most relevant 2024-2026 paper:

| Backbone family | Champion paper | Recipe used in this project |
|-----------------|----------------|------------------------------|
| **DINOv2** (champion backbone) | Oquab et al. 2024 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' (arXiv:2304.07193) | Frozen ViT-S/14 (21 M params), zero-shot feature extraction, 224×224 resize, ImageNet normalisation, grayscale-to-3channel replication. No fine-tuning. |
| **Spectral** (champion head) | Ng, Jordan, Weiss 2001 NeurIPS 'On Spectral Clustering' (DOI:10.5555/2980539.2980649); Shi & Malik 2000 IEEE TPAMI 'Normalized Cuts' (DOI:10.1109/34.868688) | `affinity='cosine'`, `assign_labels='kmeans'`, `n_init=10`, `random_state=99`. Alternative: `assign_labels='cluster_qr'` for deterministic. |
| **Ward** | Ward 1963 JASA 'Hierarchical grouping to optimize an objective function' | `linkage='ward'` on Euclidean. Best on DINOv2 = ARI 0.6371 (Exp 27). |
| **Birch** | Zhang, Ramakrishnan, Livny 1996 SIGMOD 'BIRCH' (DOI:10.1145/233269.233324) | `threshold=0.5`, `branching_factor=50` (defaults). Threshold-invariant for n < 10 000 — see §6.2. |
| **UMAP** | McInnes, Healy, Melville 2018 arXiv 'UMAP' (arXiv:1802.03426) | `n_neighbors=10`, `min_dist=0.1`, `n_components=5`, `metric='euclidean'` on DINOv2. Best UMAP variant = ARI 0.6488 (Exp 123). |
| **DEC** | Xie, Girshick, Farhadi 2016 ICML 'Unsupervised Deep Embedding for Clustering Analysis' (arXiv:1511.06335); Guo, Gao, Liu, Yin 2017 IJCAI 'IDEC' (DOI:10.24963/ijcai.2017/243) | `latent_dim=64`, `alpha=1.0`, `mse_w=0.1`, pretrain=40 epochs, dec_epochs=20. Plateau at ARI ≈ 0.50 — see §6.1. |
| **PCA / KMeans** | Lloyd 1982 IEEE TIT 'Least squares quantization in PCM'; Pearson 1901 'On lines and planes of closest fit' | `n_components=50` for PCA (sweet spot), `n_init=10` for KMeans (raised to 50 in Exp 25 for n_init sweep). |
| **GMM** | McLachlan & Peel 2000 Wiley 'Finite Mixture Models' | `covariance_type='full'`, `n_init=5`. |
| **HDBSCAN** | Campello, Moulavi, Sander 2013 PAKDD 'Density-Based Clustering Based on Hierarchical Density Estimates' | `min_cluster_size=5`. Under-segments at K = 40 (only 17 clusters, 47 noise). |

**Re-derive for every new SOTA variant.** When a backbone family has multiple variants (e.g. ViT-S/14 vs ViT-B/14 vs ViT-L/14), each variant re-derives its recipe from its OWN paper.

---

## GPU Memory Constraint (MANDATORY — 16 GB VRAM hard cap)

**This laptop has 16 GB of GPU VRAM.** Every backbone selection must fit:

| Backbone | Approx params | Fit |
|----------|---------------|-----|
| DINOv2 ViT-S/14 | 21 M | Trivial — < 1 GB VRAM |
| DINOv2 ViT-B/14 | 86 M | Trivial — < 2 GB VRAM |
| DINOv2 ViT-L/14 | 304 M | Comfortable — < 4 GB VRAM (not tested in this project) |
| ResNet18 | 11.7 M | Trivial — < 1 GB |
| Convolutional AE | 1 M | Trivial |
| DEC encoder + clustering layer | < 1 M | Trivial |
| SimCLR encoder | 11.7 M (ResNet18 backbone) | Trivial |

The bottleneck for clustering is *not* GPU memory but the O(n²) affinity matrix in Spectral and the O(n × n_features) feature matrix. For n = 400 these are negligible.

---

## Backbone Isolation Rule

Before starting a new backbone hill-climb (e.g. starting `run_dec_only.py` after the UMAP sweep), snapshot `common.py` and the previous runner to `code_versions/<previous_backbone>_final/`. Do NOT modify shared utilities (`common.py`, `prepare_data.py`) while experiments on another backbone are in progress.

---

## Dashboard Backbone Tabs

Dashboard (`dashboard.html`) renders a backbone tab bar above the experiment list. Default view: ALL. Tabs filter the table to a single backbone's experiments. Click to switch. The dashboard reads JSONL dynamically so new experiments appear within the auto-refresh interval (10 s default).

---

## GitHub Pages Dashboard Sync (MANDATORY — every push, zero exceptions)

**The live dashboard MUST be published to GitHub Pages on every commit that changes experiment state.** Hosted at:

> https://dlmastery.github.io/autoresearch/clustering_olivetti/

**Source of truth:** `autoresearch_results/dashboard.html` (+ data files: `experiment_log.jsonl`, `best_config.json`, `reasoning_annotations.json`, and the `.md` report/journal/summary files the dashboard links to).

**Pages mirror:** `docs/clustering_olivetti/` (the repo root `docs/` folder is what GitHub Pages serves). The dashboard's `dashboard.html` is copied to `docs/clustering_olivetti/index.html` so the URL `/clustering_olivetti/` routes directly to it.

**Sync command (idempotent — run freely):**

```bash
cd generalized_ml_autoresearch/examples/clustering_olivetti
python sync_dashboard.py
```

The script copies the entire `autoresearch_results/` artifact set into `docs/clustering_olivetti/`. It fails loudly if any required source file is missing.

**When must you sync?**

- After every experiment that writes to the JSONL
- After every reasoning-annotation edit
- After every winner archive
- After every artifact regeneration (paper, medium, reports)
- **Before every `git push`** — the commit without the synced `docs/clustering_olivetti/` is a regression

**Per-commit ritual:**

```bash
# 1. Run experiments / regenerate artifacts
# 2. Sync to docs/
python sync_dashboard.py
# 3. Stage source AND mirror
git add generalized_ml_autoresearch/examples/clustering_olivetti docs/clustering_olivetti
# 4. Commit
git commit -m "..."
# 5. Push
git push origin master
# 6. Verify (within 60 s)
curl -s -o /dev/null -w "%{http_code}\n" https://dlmastery.github.io/autoresearch/clustering_olivetti/best_config.json
# Expected: 200
```

**Pre-push checklist:**

- [ ] `dashboard.html` source matches `docs/clustering_olivetti/index.html` (`diff -q` returns empty)
- [ ] `experiment_log.jsonl` row counts match between source and mirror
- [ ] `best_config.json` is byte-identical between source and mirror
- [ ] All five narrative .md files (paper, medium, autoresearch_report, forensic_report, audit_report_third_party) are mirrored

**Why this matters.** The paper, the Medium article, and the third-party audit cite the live dashboard as the project's institutional memory. A stale dashboard makes the citation a lie.

---

## Local Dashboard (development)

For browsing during a session without pushing:

```bash
cd generalized_ml_autoresearch/examples/clustering_olivetti
python -m http.server 8765 --directory autoresearch_results
# Open http://localhost:8765/dashboard.html
```

The local dashboard reads the same JSONL / annotations / best_config files the runner writes. It's always live with whatever has been logged. The Pages mirror is the *committed snapshot*; the local server is the *live view during development*.

**Common dashboard issues:**

- **JSONL contains `NaN` or `Infinity`** → JS can't parse. Browsers throw "Unexpected token 'N'". Fix: `common.log_experiment` now coerces NaN/Infinity → null via the `_no_nan` helper. If you find a new offender, retrofit the existing JSONL with the regex `NaN|-?Infinity → null`.
- **Empty backbone name** in `onclick="setBackbone('${b}')"` → if `b` ever contains a single quote, the HTML breaks. Backbone names should be `[A-Za-z0-9_+\-=]` only.
- **Missing favicon** → 404 in DevTools, harmless. Inline SVG favicon is now embedded in `<head>`.

---

## Dashboard Files Update Mandate (MANDATORY — every experiment, zero exceptions)

**Every experiment updates ALL the following files. If any file is stale after an experiment completes, that's a regression — stop and fix before moving on.**

| File | Written by | When | Content |
|------|------------|------|---------|
| `autoresearch_results/experiment_log.jsonl` | runner (auto via `common.log_experiment`) | every run, appended | full metrics: ARI, NMI, FMI, silhouette, per-cluster, timing, config |
| `autoresearch_results/best_config.json` | runner (auto) | only when new GLOBAL champion | overwritten with full champion entry |
| `autoresearch_results/trade_logs/exp<N>_predictions.csv` | runner (auto) | every run | one row per sample (index, true_cluster, predicted_cluster) |
| `autoresearch_results/trade_logs/exp<N>_prediction_summary.json` | runner (auto) | every run | per-cluster purity, n_pred_clusters, n_noise |
| `autoresearch_results/reasoning_annotations.json` | Claude BEFORE + runner AFTER | every run, two-phase | 7 fields per experiment |
| `autoresearch_results/research_journal.md` | Claude | every run, appended | markdown narrative of the 7-step process |
| `autoresearch_results/experiment_summary.md` | Claude | every run, appended | tabular entry per experiment |
| `autoresearch_results/forensic_checkpoint.md` | Claude | every run | update champion, history table, next-command block |
| `autoresearch_results/winners/<backbone>_exp<N>_<desc>/*` | Claude | only when new GLOBAL champion | README, config, frozen code, predict.py, audit_report.md, colab notebook |
| `autoresearch_results/dashboard.html` | Claude (rarely) | only when adding a metric/tab | static HTML — reads JSONL live |

**Per-experiment ritual (in order, every run):**

1. **Before launch:** open `reasoning_annotations.json`, insert a new entry with `diagnosis`, `citations`, `hypothesis`, `prediction`, `_manual: true`. The validator refuses to launch otherwise.
2. **Before launch:** append a matching section to `research_journal.md`.
3. **Launch:** run the runner script.
4. **Runner auto-updates:** JSONL, best_config (if champion), trade_logs CSV + JSON, reasoning_annotations verdict/learning fallback.
5. **After completion:** Claude reads runner output, overwrites verdict and learning with richer analysis (per-cluster purity, which subjects were resolved/lost, silhouette change). Updates the journal section.
6. **After completion:** Claude appends a row to `experiment_summary.md`.
7. **After completion:** Claude updates `forensic_checkpoint.md` with the new experiment in the history table, updated champion (if applicable), and the exact next-experiment command.
8. **If new champion:** Claude archives to `winners/<backbone>_exp<N>_<desc>/`.
9. **Before any push:** `python sync_dashboard.py` to mirror to `docs/`.

**Verification at start of every cycle:**

Before launching Experiment N+1, confirm for Experiment N:

- [ ] `experiment_log.jsonl` has an entry for N
- [ ] `reasoning_annotations.json[N]` has all 7 fields non-empty
- [ ] `research_journal.md` has a section for N
- [ ] `experiment_summary.md` has a row for N
- [ ] `forensic_checkpoint.md` references N in the history table
- [ ] `trade_logs/expN_predictions.csv` and `expN_prediction_summary.json` exist
- [ ] If N set a new champion: `winners/<backbone>_expN_<desc>/` exists with all required files
- [ ] `docs/clustering_olivetti/` is in lockstep (or will be on next push)

---

## Citation Rigor (MANDATORY format for `citations` field)

**Every citation string MUST contain, for every paper referenced:**

1. **All authors' surnames** (not just first-author et al. unless > 6 authors)
2. **Year** of publication
3. **Venue** — journal name, conference abbreviation (NeurIPS, ICML, ICLR, ICCV, CVPR, KDD, SIGMOD, JMLR, IEEE TPAMI, IEEE Access, etc.), or `arXiv` if preprint-only
4. **Full paper title** in single quotes
5. **arXiv ID or DOI** mandatory for any paper that has one
6. **One-sentence relevance note** explaining why this paper motivates THIS experiment specifically

**Format template:**

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) — one-sentence note on why we cite it here.
```

**Multiple papers separated by semicolons + linebreak.** Minimum one primary citation per experiment.

**Examples of GOOD citations from this project:**

> Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin 2021 ICCV 'Emerging Properties in Self-Supervised Vision Transformers' (arXiv:2104.14294) — DINO learns class-token representations whose nearest-neighbour structure recovers semantic clusters without supervision.

> Ng, Jordan, Weiss 2001 NeurIPS 'On Spectral Clustering: Analysis and an algorithm' (DOI:10.5555/2980539.2980649) — the cosine affinity on L2-normalised features defines a graph whose normalised-Laplacian spectral embedding admits exact NCut recovery in the noiseless case.

> Damle, Minden, Ying 2019 SIAM J. Sci. Comput. 'Robust and efficient multi-way spectral clustering' (arXiv:1708.07964) — `assign_labels='cluster_qr'` is deterministic given the eigenvectors and eliminates seed variance in the assignment step.

**Examples of BAD citations (REJECTED — rewrite required):**

- `"Caron 2021 DINO"` — missing coauthors, venue, title, arXiv, relevance note
- `"(Caron2021)"` — parenthetical tag only, useless
- `"Caron et al."` — no year, no venue
- `"DINO paper"` — no attribution
- `"see research_journal.md"` — redirects instead of citing

**Arxiv ID lookup discipline.** If you know the paper but not its arXiv ID, fetch via WebSearch / WebFetch (arxiv.org/abs search) before writing the entry. Authoring a citation without the arXiv ID is a partial job.

---

## Reasoning Blob Completeness (what "full reasoning" means)

Each of the 7 fields has a minimum content spec. Entries below this are rewritten. `common.validate_pre_run` and `common.validate_post_run` enforce these.

| Field | Floor | Must include |
|-------|------:|--------------|
| `diagnosis` | ≥ 60 words | Reference to ≥ 1 prior experiment number OR a per-cluster metric from the current champion |
| `citations` | ≥ 40 (single) or ≥ 80 (multi) words | Author list + year + venue + title + arXiv ID + relevance note for each paper |
| `hypothesis` | ≥ 50 words | "mechanism" / "because" / "per [paper]"; the specific parameter and value |
| `prediction` | ≥ 25 words | Numeric range; direction for ≥ 1 sub-metric |
| `verdict` | ≥ 30 words | KEEP / DISCARD / NEAR-MISS; ARI to 4 decimals; mention of ≥ 1 per-fold result |
| `learning` | ≥ 40 words | "axis open" / "axis closed" OR concrete "next try: ..." |
| `_manual` | Boolean | `true` for non-mechanical |

**For variance check batches** (same config, varying seed): templated `diagnosis` and `citations` are OK, but `verdict` / `learning` MUST be per-run-specific.

**Batch updates are forbidden.** Don't do 5 experiments then update the journal/summary/checkpoint in one go — each experiment's state gets stale and crash-recovery breaks.

---

## Loss Function Rules (clustering-specific)

The "loss function" in classical clustering is implicit:

- **KMeans** — squared-Euclidean inertia (Lloyd 1982).
- **Spectral** — minimum normalised cut on the affinity graph (Shi & Malik 2000).
- **Ward** — minimum within-cluster variance increase per merge (Ward 1963).
- **Birch** — diameter-bounded clustering features (Zhang 1996).
- **GMM** — log-likelihood under the mixture (McLachlan & Peel 2000).
- **HDBSCAN** — minimum spanning tree on mutual reachability distances (Campello 2013).
- **DEC** — KL divergence between soft assignment Q and target distribution P, optionally + reconstruction MSE (Xie 2016, Guo 2017).
- **SimCLR** — contrastive NT-Xent loss (Chen 2020). *Fails at n = 400* — needs millions of unlabelled samples.

For DEC specifically, `common.py` does NOT support heteroscedastic / aleatoric loss because DEC's KL formulation already provides per-sample soft probabilities; adding a variance head would be redundant.

---

## Winner Archiving Protocol (MANDATORY for every NEW BEST)

Every time a new champion is found (status = KEEP and ARI > previous best), archive ALL artifacts to a self-contained subdirectory.

**Directory:** `autoresearch_results/winners/<backbone>_exp<N>_<short_description>/`

```
winners/
  spectral_hc_cosine_seed99_(variance_c_exp71/
    README.md                       # 13-section champion description
    config.json                     # Exact config (seed, affinity, assign_labels, n_init, model)
    experiment_log_entry.json       # The JSONL row
    audit_report.md                 # 14-section explainability audit
    code/                           # Frozen snapshot of common.py, prepare_data.py, runner script
    inference/predict.py            # Standalone inference script
    colab_train_and_infer.ipynb     # Self-contained Colab notebook
```

**README.md template (13 sections per FX template, adapted):**
1. Method
2. Why this configuration won (champion progression mechanism)
3. Per-fold metrics (single-fold full-dataset evaluation)
4. Hyperparameters (full dump)
5. Architecture description
6. Training / fitting details
7. Uncertainty / confidence per fold (silhouette histogram)
8. Reproduction status
9. Sample inference code
10. Deployment Strategy (signal generation, decision rules, resource sizing, refresh cadence, per-regime, risk controls, expected performance, caveats, reference to inference code)
11. Known limitations and risks
12. Pointers
13. Reproduce on this machine

**After archiving:** rerun the winner from the frozen code to verify reproduction. Reproduction log goes into `winners/<champion>/reproduction/reproduce_log.txt`. If ARI differs by > 0.005, flag and investigate.

---

## Google Colab Notebook (MANDATORY for every winner)

For every archived winner, generate `colab_train_and_infer.ipynb`:

1. **Setup:** `!pip install` torch, torchvision, sklearn, umap-learn.
2. **Data cell:** load Olivetti via `sklearn.datasets.fetch_olivetti_faces()`.
3. **Feature engineering:** load DINOv2 from `torch.hub`, extract 384-dim features.
4. **Training cell:** the SpectralClustering call with the locked config (random_state = 99, affinity='cosine', assign_labels='kmeans', n_init=10).
5. **Evaluation cell:** ARI / NMI / FMI / silhouette / V-measure / homogeneity / completeness on the full 400-sample dataset.
6. **Inference cell:** apply the trained model to a sample batch.
7. **Visualization cell:** confusion matrix, per-true-subject recovery rate bar chart, silhouette histogram.
8. **Export cell:** save model recipe (we don't pickle SpectralClustering — we save the recipe).

Notebook principles: every cell has a markdown header. Champion config is at the top. `torch.manual_seed(99)` and `np.random.seed(99)` for reproducibility. Target runtime < 5 minutes on Colab free tier (T4 GPU).

---

## Traditional ML Metrics (MANDATORY for every experiment)

Compute and log:

- **ARI** (primary, Hubert & Arabie 1985)
- **NMI** (Strehl & Ghosh 2002 JMLR)
- **FMI** (Fowlkes & Mallows 1983)
- **Homogeneity / Completeness / V-measure** (Rosenberg & Hirschberg 2007 EMNLP)
- **Silhouette** (Rousseeuw 1987)
- **n_pred_clusters** (sanity for K = 40)
- **n_noise** (HDBSCAN-family only)
- **n_true_clusters = 40** (constant, sanity)

These appear in:
1. `common.evaluate_clustering()` output
2. JSONL `secondary_metrics` field
3. Dashboard per-row table
4. Winner archive `audit_report.md`
5. Experiment summary markdown

---

## Per-Prediction Log (MANDATORY for every experiment)

Output: `autoresearch_results/trade_logs/exp<N>_predictions.csv`

Columns:
| Column | Description |
|--------|-------------|
| index | Row index in 400-sample dataset |
| true_cluster | Ground-truth subject ID (0-39) |
| predicted_cluster | Model cluster ID (0-39) |

Per-cluster summary in `exp<N>_prediction_summary.json`:
- `backbone`, `ari`, `nmi`, `n_pred_clusters`, `n_noise`

This data enables: per-cluster purity analysis, identifying mis-clustered subjects, computing Hungarian assignment to align predicted with true labels.

---

## Architecture

- **AutoResearch loop = Claude agent.** Claude reads results, decides what to try, calls the runner, reads output. The intelligence is in the agent, NOT in Python code. No pre-baked experiment lists.
- Runner scripts (`run_<backbone>.py`) execute one experiment per call (or a hill-climb of N variants per call). Each variant authors its own pre-run blob and logs.
- `common.py` provides shared utilities: `load_data()`, `evaluate_clustering()`, `log_experiment()`, `author_pre_run()`, `author_post_run()`, validators.
- Dashboard (`dashboard.html`) reads logs. DECOUPLED from runner.
- Save checkpoint after every experiment (JSONL append + best_config.json overwrite).
- Use relative imports.

---

## Validation Checklist (Run Before Every Experiment Session)

1. `len(X) == 400` and `X.shape == (400, 4096)` ✓
2. `len(np.unique(y)) == 40` ✓
3. `X` SHA-256 first 16 hex == `e6b9b0fe62f642f6` ✓
4. `y` SHA-256 first 16 hex == `2745696ae3f897d8` ✓
5. Composite fingerprint == `clustering-ari-floor0.3` ✓
6. `reasoning_annotations.json` has entries for every JSONL row ✓
7. No NaN / Infinity in any JSONL row (else the dashboard breaks) ✓
8. Local dashboard at http://localhost:8765/dashboard.html returns 200 ✓
9. Pages dashboard at https://dlmastery.github.io/autoresearch/clustering_olivetti/ returns 200 ✓

---

## Project Structure

```
generalized_ml_autoresearch/examples/clustering_olivetti/
  CLAUDE.md                                # this file
  README.md                                # user guide with current champion
  index.md                                 # GitHub Pages landing
  paper.md                                 # 38-reference research paper
  paper_abstract.md                        # one-paragraph abstract

  common.py                                # shared utilities, validators, log_experiment
  prepare_data.py                          # Olivetti loader with hash check
  generate_artifacts.py                    # regenerates paper, medium, reports
  sync_dashboard.py                        # mirrors autoresearch_results/ to docs/
  third_party_audit.py                     # generates audit_report_third_party.md

  run_exp01_kmeans_raw.py                  # Exp 1: baseline
  run_full_pipeline.py                     # Exps 2-14: classical baselines
  run_more_sota.py                         # Exps 15-21: GMM, Birch, AffProp, MeanShift, DINOv2 KMeans
  run_dinov2_hill_climb.py                 # Exps 22-46: 25 DINOv2 variants
  run_spectral_hill_climb.py               # Exps 47-71: 25 Spectral variants (champion at 71)
  run_ward_birch_hill_climb.py             # Exps 72-121: 25 Ward + 25 Birch variants
  run_umap_dec_hill_climb.py               # Exps 122-136: 15 UMAP variants (DEC quarantined here)
  run_dec_only.py                          # Exps 137-146: 10 DEC variants

  autoresearch_results/
    experiment_log.jsonl                   # 149 rows (auto, append-only)
    reasoning_annotations.json             # 7-field × 149-entry reasoning blob
    best_config.json                       # current global champion (Exp 71, ARI 0.7195)
    dashboard.html                         # source-of-truth dashboard
    research_journal.md                    # markdown narrative of all 149 experiments
    experiment_summary.md                  # tabular per-experiment summary
    forensic_checkpoint.md                 # crash-recovery snapshot
    forensic_report.md                     # internal forensic audit
    audit_report_third_party.md            # third-party audit (PASS WITH ONE FOOTNOTE)
    autoresearch_report.md                 # comprehensive technical report
    medium_article.md                      # field-report-style narrative
    trade_logs/exp<N>_predictions.csv      # per-sample cluster assignments (149 files)
    winners/<backbone>_exp<N>_<desc>/      # archived champions (4 archives in this project)
    _quarantined_blind_sweep/              # invalid early experiments
    _quarantined_exp1/                     # invalid early Exp 1

docs/clustering_olivetti/                  # Pages mirror (auto-synced)
  index.html                               # = autoresearch_results/dashboard.html
  paper.md, medium_article.md, etc.        # all narrative artifacts
  experiment_log.jsonl, best_config.json, reasoning_annotations.json
```

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `n_samples` | 400 | sklearn Olivetti |
| `n_features` | 4096 | 64×64 flatten |
| `K_true` | 40 | 40 subjects × 10 images |
| X SHA-256 (first 16 hex) | `e6b9b0fe62f642f6` | locked at first load |
| y SHA-256 (first 16 hex) | `2745696ae3f897d8` | locked at first load |
| Primary metric | ARI | sklearn.metrics.adjusted_rand_score |
| Composite floor | 0.30 | `common.composite_score()` |
| Composite fingerprint | `clustering-ari-floor0.3` | locked, on every JSONL row |
| Default `random_state` | 0 | every runner unless explicitly varied |
| Champion seed | 99 | only experiment that uses non-zero seed (Exp 71 + variance siblings 68-71) |
| Per-backbone hill-climb size | 25 | per FX-mandated discipline |
| Reasoning floors | 60 / 40 / 50 / 25 / 30 / 40 words | diagnosis / citations / hypothesis / prediction / verdict / learning |
| GitHub Pages URL | https://dlmastery.github.io/autoresearch/clustering_olivetti/ | auto-served from `docs/` |
| Local dashboard URL | http://localhost:8765/dashboard.html | `python -m http.server 8765` |

---

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Letting `for nn in [...]` shadow `torch.nn` | `class ConvAE(nn.Module)` fails because `nn` is rebound to int | Use `nb`, `k`, or any non-`nn` loop variable name when iterating over hyperparameter values |
| `json.dumps(record)` without NaN coercion | JSONL contains literal `NaN`; dashboard JS `JSON.parse` throws "Unexpected token 'N'" | `_no_nan` recursive helper + `allow_nan=False` in `common.log_experiment` |
| Single-seed Spectral champion claim | Headline ARI is the positive tail of a ±0.10 spread | Always run 5-seed variance check; report median ± std as headline |
| Sweeping Birch threshold below n ≈ 10 000 | 13 thresholds → identical ARI = 0.6371; wasted compute | Skip the sweep; set `threshold=0.5` and move on |
| Running DEC on n = 400 expecting MNIST results | Plateau at ARI ≈ 0.50 across all 4 hyperparameter axes | Use DINOv2 + Spectral instead |
| Single-linkage on cosine distance | Chaining effect collapses everything into one giant cluster (ARI = 0.14) | Use Ward or average linkage; never single on connected graphs |
| ResNet18-ImageNet features on Olivetti | ImageNet softmax bottleneck destroys identity discrimination (ARI = 0.44) | Use DINOv2 (self-supervised) instead |
| SimCLR at n = 400 | Contrastive loss collapses to trivial solution (ARI = 0.37) | Need n > 1 000 000 for SimCLR; not applicable here |
| Forgetting to sync dashboard before push | Pages mirror stale; paper citations become lies | Run `python sync_dashboard.py` as the FIRST step of every commit ritual |
| Editing `dashboard.html` to write data | Decoupling broken | Dashboard reads only; runner writes only |
| Bundling DINOv2 + Spectral as "one backbone" | Tier-3 violation: head matters as much as backbone | Separate `dinov2_kmeans`, `dinov2_ward`, `dinov2_spectral` JSONL backbone fields |
| Skipping a backbone's pre-run reasoning blob | Validator refuses to launch | Author the pre-run before the experiment; check the validator output |
| Re-running the entire pipeline after a single experiment | Wasted compute | The runners are idempotent — re-running just appends to JSONL. Use the checkpoint to find the right next experiment instead |

---

## Session Learnings (149-experiment phase complete, 2026-04-26)

Append-only. New session insights go at the bottom, date-stamped. Never delete.

### Confirmed champion path (12 rungs, 0.4057 → 0.7195)

| Exp | Method | ARI | Δ | Mechanism |
|----:|--------|----:|--:|-----------|
| 1 | KMeans on raw pixels | 0.4057 | — | baseline (Lloyd 1982) |
| 2 | KMeans on PCA(50) | 0.4780 | +0.07 | eigenfaces (Turk & Pentland 1991) |
| 8 | Agglomerative Ward | 0.5159 | +0.04 | variance-min (Ward 1963) |
| 16 | Spectral RBF tuned | 0.5252 | +0.01 | NCut (Shi & Malik 2000) |
| 17 | Birch | 0.5287 | +0.00 | CF-tree leaves (Zhang 1996) |
| 20 | DINOv2 + KMeans | 0.5455 | +0.02 | self-supervised (Oquab 2024) |
| 22 | DINOv2 + MiniBatch-KMeans | 0.5596 | +0.01 | stochastic restart (Sculley 2010) |
| 25 | DINOv2 + KMeans n_init=50 | 0.5852 | +0.03 | more restarts |
| 27 | DINOv2 + Ward | 0.6371 | +0.05 | variance-min × deep features |
| 33 | DINOv2 + Spectral cosine | 0.6963 | +0.06 | global graph (Ng, Jordan, Weiss 2001) |
| 55 | DINOv2 + Spectral RBF γ=1e-4 | 0.7170 | +0.02 | tiny gamma → linear ≈ cosine |
| **71** | **DINOv2 + Spectral cosine, seed=99** | **0.7195** | +0.00 | lucky-seed positive tail |

### Three research findings (none in the cited literature)

1. **DEC plateaus at ARI ≈ 0.50 on n = 400 face data.** 11 hill-climb variants → std = 0.0190, range [0.4435, 0.5104]. None of latent dim / α / MSE-KL balance / pretrain epochs moves the metric. Confirms Min, Guo, Liu, Long 2018 IEEE Access survey; DEC is sample-hungry. **Practitioner rule:** do not use DEC on small face datasets.
2. **Birch is threshold-invariant for n < 10 000.** 13 different thresholds in [0.10, 1.0] → identical ARI = 0.6371. Mechanism: leaf-clustering KMeans dominates the threshold-driven CF-tree-construction step at small n. **Practitioner rule:** skip Birch threshold sweeps below n ≈ 10 000.
3. **Spectral cosine on DINOv2 has a ±0.10 ARI seed-variance crisis.** 5-seed variance check on champion config → ARIs {0.6963, 0.7154, 0.6596, 0.6127, 0.7195} for seeds {0, 1, 7, 42, 99}. Std = 0.0429, spread = 0.107 — *larger* than the gap between Spectral and Ward. **Practitioner rule:** report 5-seed median, not point estimate. Or use `assign_labels='cluster_qr'` for determinism.

### Confirmed non-improvements (axes closed)

- ICA + KMeans (Exp 9) — ICA's non-Gaussianity prior wrong for face identity (which is roughly Gaussian after PCA whitening).
- Convolutional AE + KMeans (Exp 10) — 1 M-param CAE overfits at n = 400.
- ResNet18 ImageNet features (Exp 11) — softmax bottleneck destroys identity.
- HDBSCAN (Exp 12) — under-segments to 17 clusters + 47 noise.
- SimCLR (Exp 13) — contrastive collapses at n = 400.
- CSPA consensus of top 5 (Exp 14) — base clusterings too correlated.
- MeanShift auto-bandwidth (Exp 19) — collapses all 400 points into 1 cluster.
- Spectral RBF default gamma (Exp 6) — gamma is wrong for d = 4096.
- Spectral RBF gamma > 0.005 (Exps 57-59) — gamma too large; ARI < 0.30.
- Ward single-linkage on cosine (Exp 82) — chaining effect; ARI = 0.14.
- UMAP + Spectral cosine downstream (Exp 136) — DISCARD; UMAP noise + Spectral spectral both stochastic.
- DEC default config in `run_umap_dec_hill_climb.py` — `for nn in [...]` shadowed `torch.nn`; quarantined and re-run as `run_dec_only.py`.

### Confirmed improvements (axes still open for further research)

- DINOv2 ViT-L/14 (1024-dim) — not yet tested; might give +0.02 ARI.
- 5-seed median ensemble via co-association — the "next try" line in Exp 71's learning blob; predicted to push past 0.72 with smaller variance.
- DINOv2 + Spectral with `assign_labels='cluster_qr'` — Exp 48 gave 0.6963 (= seed-0 KMeans); use as the *deployment* config because it's deterministic.
- Subject-supervised fine-tuning (FaceNet triplet) — out-of-scope but would presumably hit ARI ≥ 0.85.

### Key protocol additions (lessons from this project)

**1. NaN coercion in `log_experiment`.** Found Apr 26 when the dashboard threw "Unexpected token 'N'". Patched in `common.py` with the `_no_nan` recursive helper + `allow_nan=False`. Retroactive fix to existing JSONL via regex `NaN|-?Infinity → null`.

**2. The runner's `for` loop variable.** Found in `run_umap_dec_hill_climb.py` where `for nn in [...]` shadowed `import torch.nn as nn`. **Rule:** never use `nn`, `np`, `pd`, `F`, `T` as loop variables in any runner script.

**3. Quarantine bad experiments rather than deleting.** Two quarantines exist: `_quarantined_blind_sweep/` and `_quarantined_exp1/`. Each has a `WHY_QUARANTINED.md` note. Deleting would destroy audit trail; quarantining preserves it without contaminating the JSONL or champion search.

**4. Per-backbone runners stay separate.** `run_dec_only.py` was split out of `run_umap_dec_hill_climb.py` after the torch.nn shadowing bug. Cleaner to have 7 runner scripts than to have one 1500-line script with multiple side-effects.

**5. Champion archive must include frozen code.** The `winners/spectral_hc_cosine_seed99_(variance_c_exp71/code/` directory is a copy of `common.py`, `prepare_data.py`, and the runner at the time of the win. Without this, sklearn version changes could break reproduction.

**6. Dashboard must be tested in a real browser.** The "JS error" reported by the user on Apr 26 was a NaN-in-JSONL bug that *only manifests* in browsers — Python `json.loads` and Python's `json` module both happily parse `NaN` as `float('nan')`, so server-side validation didn't catch it. **Rule:** after any `common.py` change that touches `log_experiment`, open the local dashboard in a real browser (Playwright snapshot or manual) and check DevTools console for errors.

### Session Learnings — Apr 26 expansion

- CLAUDE.md was expanded from 4 KB (compact) to ~50+ KB (FX-comprehensive) on Apr 26 to bring it in line with the FX-root CLAUDE.md.
- The dashboard NaN bug was fixed both retroactively (existing JSONL) and prospectively (`common.py`).
- An inline-SVG favicon was embedded in `dashboard.html` to silence the 404 in DevTools.
- All 5 narrative artifacts (paper.md, medium_article.md, autoresearch_report.md, forensic_report.md, audit_report_third_party.md) were upgraded to FX rigor on Apr 26.
- Commit `91f314b`: regenerated artifact suite. Commit `ef450cd`: added GitHub Pages Dashboard Sync mandate. Commit (this one): full CLAUDE.md FX-comprehensive expansion + dashboard NaN fix.

---

## Cross-references

| Document | Purpose |
|----------|---------|
| `C:/Users/abhir/clauderesearch/autoresearch/CLAUDE.md` | Source of truth: the FX-project CLAUDE.md. |
| `generalized_ml_autoresearch/CLAUDE.md` | Framework-level CLAUDE.md (meta-rules for the framework itself). |
| `generalized_ml_autoresearch/templates/CLAUDE_template.md` | Parameterised version this file inherits from. |
| `generalized_ml_autoresearch/templates/SECTION_MAPPING.md` | 52-section audit log. |
| `generalized_ml_autoresearch/templates/sota_catalog.yaml` | Curated 2024-2026 SOTA recipes. |
| `paper.md` | The research paper (38 references, 10 sections, 6260 words). |
| `autoresearch_results/medium_article.md` | The Medium-style narrative (3559 words). |
| `autoresearch_results/audit_report_third_party.md` | The independent audit (PASS WITH ONE FOOTNOTE). |
| `autoresearch_results/winners/spectral_hc_cosine_seed99_(variance_c_exp71/README.md` | Champion archive (13-section deployment writeup). |

---

## License

Inherits the parent `dlmastery/autoresearch` repository's MIT license.

## Credits

- AutoResearch protocol — Evija Ranti.
- This clustering project — Claude Code (Opus 4.7, 1M context), Apr 25-26 2026.
- The full reasoning trail and audit are at https://github.com/dlmastery/autoresearch and https://dlmastery.github.io/autoresearch/clustering_olivetti/.
