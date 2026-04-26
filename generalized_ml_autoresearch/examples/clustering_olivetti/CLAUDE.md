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
