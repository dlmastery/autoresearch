# CLAUDE.md — Clustering autoresearch on Olivetti Faces

> Project-specific rules for the unsupervised face-clustering benchmark.
> Inherits from `generalized_ml_autoresearch/templates/CLAUDE_template.md`.

## Project context

- **Dataset:** Olivetti Faces (sklearn-bundled). 400 grayscale 64×64 images, 40 subjects × 10 images each.
- **Task:** Unsupervised clustering — recover the 40 person identities from pixel data alone.
- **Primary metric:** Adjusted Rand Index (ARI) against the held-out true subject IDs.
- **Secondary metrics:** NMI, FMI, Silhouette, Homogeneity, Completeness, V-measure, n_pred_clusters, n_noise.
- **Composite:** ARI directly. Floor = 0.30 (must non-trivially beat random clustering for K=40).
- **No train/test split:** clustering uses the full dataset; ground-truth labels are NEVER seen by any model — they are used only at metric time for ARI/NMI computation.

## Documented baselines (for hypothesis grounding)

| Method | ARI (typical) | NMI (typical) |
|---|---|---|
| KMeans on raw pixels | 0.50 | 0.78 |
| KMeans on PCA(50) | 0.62 | 0.84 |
| Spectral (RBF affinity) | 0.68 | 0.86 |
| GMM (full covariance) | 0.55 | 0.80 |
| Agglomerative (Ward) | 0.65 | 0.85 |
| Birch | 0.55 | 0.80 |
| HDBSCAN | varies | varies (often produces noise + few clusters) |
| VAE/Autoencoder + KMeans | 0.75 | 0.89 |
| Contrastive (SimCLR) + KMeans | 0.85 | 0.93 |

## Hard rules (never violate)

1. **No label leakage.** `y` (true subject IDs) is loaded ONLY in `evaluate_clustering()`. Models receive `X` only.
2. **Reproducibility.** Every experiment MUST set `random_state=0` unless it is explicitly a multi-seed variance probe.
3. **Test-set integrity = full-dataset integrity.** Verify `len(X)==400` and `X.shape==(400,4096)` before every experiment. SHA-256 hash on `X.tobytes()` locked at `e6b9b0fe62f642f6` (first 16 hex).
4. **Reasoning gate.** Every experiment MUST author a pre-run reasoning entry (diagnosis ≥60w, citations ≥40w with year/venue/title/relevance, hypothesis ≥50w with mechanism keyword, prediction ≥25w with numeric range) before running. Validated by `common.author_pre_run()`.
5. **Multi-seed variance** before declaring any new champion. ARI std across 5 seeds must be < 0.05 to call it stable.

## 7-step protocol (per project)

For every experiment after the baseline:
1. **Diagnose** the current champion's failure: per-cluster confusion analysis, silhouette-vs-ARI correlation, which true subjects are most-misclustered.
2. **Cite** a paper that addresses the diagnosed failure mode.
3. **Hypothesize** the mechanism + predict numeric ARI range.
4. **Run** ONE config change.
5. **Analyze** against prediction.
6. **Document** verdict + learning.
7. **Decide** the next experiment from the analysis.

If 3+ consecutive DISCARDs → stop and rethink (likely needs structural change: deep features, different metric space, ensemble).

## Backbones to explore (rough plan)

| Tier | Methods |
|------|---------|
| 1. Linear-projection + clustering | PCA(d) + KMeans for d ∈ {20, 50, 100, 150}; ICA + KMeans |
| 2. Manifold + clustering | UMAP(d) + KMeans/HDBSCAN; t-SNE-then-cluster (sanity-check only) |
| 3. Direct clustering algorithms | KMeans (raw), Spectral (RBF/cosine), GMM, Agglomerative (ward/avg), Birch, HDBSCAN, MeanShift |
| 4. Deep features | Autoencoder + KMeans; Convolutional AE + KMeans; PCA-whitened + KMeans |
| 5. Pretrained features | torchvision ResNet18 features + KMeans (large jump expected on faces) |
| 6. Ensemble | Consensus clustering (Strehl 2002), spectral co-association |

## Documentation cadence

Per the framework's Dashboard Files Update Mandate, every experiment writes:
- `autoresearch_results/experiment_log.jsonl` (auto by `common.log_experiment`)
- `autoresearch_results/reasoning_annotations.json` (Claude pre-run + post-run)
- `autoresearch_results/best_config.json` (auto when new champion)
- `autoresearch_results/trade_logs/exp<N>_predictions.csv` (auto)
- `autoresearch_results/research_journal.md` (Claude appends)
- `autoresearch_results/experiment_summary.md` (Claude appends)

Final artifacts (mirror FX project): `paper.md`, `paper_abstract.md`, `medium_article.md`, `autoresearch_report.md`, `forensic_report.md`, `forensic_checkpoint.md`, `audit_report_third_party.md`, `winners/<champion>/` archive.

## GitHub Pages Dashboard Sync (MANDATORY — every push, zero exceptions)

**The live dashboard MUST be published to GitHub Pages on every commit that changes experiment state.** Hosted at:

> https://dlmastery.github.io/autoresearch/clustering_olivetti/

**Source of truth:** `autoresearch_results/dashboard.html` (+ its data files: `experiment_log.jsonl`, `best_config.json`, `reasoning_annotations.json`, and the `.md` report/journal/summary files the dashboard links to).

**Pages mirror:** `docs/clustering_olivetti/` (lives at the repo root `docs/` because GitHub Pages serves the `docs/` folder). The dashboard's `dashboard.html` is copied to `docs/clustering_olivetti/index.html` so the URL `/clustering_olivetti/` routes directly to it.

**Sync command (idempotent — run freely):**

```bash
cd generalized_ml_autoresearch/examples/clustering_olivetti
python sync_dashboard.py
```

The script copies the entire `autoresearch_results/` artifact set (dashboard.html → index.html, experiment_log.jsonl, best_config.json, reasoning_annotations.json, all narrative .md files, the project README, paper, paper_abstract, index.md, CLAUDE.md) into `docs/clustering_olivetti/`. It fails loudly if any required source file is missing.

**When must you sync?**

- After every experiment that writes to the JSONL (i.e. every runner invocation)
- After every reasoning-annotation edit
- After every winner archive
- After every artifact regeneration (paper, medium, reports)
- **Before every `git push`** — the commit without the synced `docs/clustering_olivetti/` is a regression

**Per-commit ritual:**

```bash
# 1. Run experiments / regenerate artifacts (above)
# 2. Sync to docs/
python sync_dashboard.py
# 3. Stage the source AND the mirror
git add generalized_ml_autoresearch/examples/clustering_olivetti docs/clustering_olivetti
# 4. Commit
git commit -m "..."
# 5. Push (Pages rebuilds within ~30-60 s)
git push origin master
# 6. Verify
curl -s -o /dev/null -w "%{http_code}\n" https://dlmastery.github.io/autoresearch/clustering_olivetti/best_config.json
# Expected: 200
```

**Verification.** After push, `curl https://dlmastery.github.io/autoresearch/clustering_olivetti/best_config.json` should show the latest champion within 2 minutes. If stale, check `git log -1 docs/clustering_olivetti/` — the commit that updated `docs/` must match the commit that updated the source `autoresearch_results/`.

**Enforcement.** A commit that changes `autoresearch_results/experiment_log.jsonl` but does NOT update `docs/clustering_olivetti/experiment_log.jsonl` is a regression. The pre-push checklist:

- [ ] `dashboard.html` source matches `docs/clustering_olivetti/index.html` (`diff -q` returns empty)
- [ ] `experiment_log.jsonl` row counts match between source and mirror
- [ ] `best_config.json` is byte-identical between source and mirror
- [ ] All five narrative .md files (paper, medium, autoresearch_report, forensic_report, audit_report_third_party) are mirrored

**Why this matters.** The paper, the Medium article, and the third-party audit all cite the live dashboard as the project's institutional memory. A stale dashboard makes the citation a lie. Treat the Pages mirror as a public artifact with the same freshness guarantees as the source JSONL.

## Local Dashboard (development)

For browsing the dashboard during a session without pushing, run:

```bash
cd generalized_ml_autoresearch/examples/clustering_olivetti
python -m http.server 8765 --directory autoresearch_results
# Open http://localhost:8765/dashboard.html
```

The local dashboard reads the same JSONL / annotations / best_config files that the runner writes, so it's always live with whatever experiments have been logged. The Pages mirror is the *committed snapshot*; the local server is the *live view during development*.
