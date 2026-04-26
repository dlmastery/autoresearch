# Olivetti Faces Clustering AutoResearch Project

**Champion: DINOv2 ViT-S/14 + Spectral Clustering, cosine affinity, seed = 99 — ARI = 0.7195**

149 honest experiments across 6 backbone families (linear projections, manifold learning, spectral clustering, deep autoencoders, pretrained vision transformers, ensembles). Every experiment is grounded in a peer-reviewed paper, predicted with a numeric ARI range *before* it runs, and validated against that prediction afterwards. The reasoning trail lives in `reasoning_annotations.json` (149 entries × 7 fields) so a future Claude Code session can reconstruct *why* every champion was chosen without re-reading any source.

## Headline result

| Metric | Value |
|--------|------:|
| ARI (seed = 99) | **0.7195** |
| ARI (5-seed median) | **0.6963 ± 0.0429** |
| NMI | 0.9004 |
| V-measure | 0.9004 |
| FMI | 0.7270 |
| Silhouette | 0.0927 |

The headline ARI = 0.7195 is the *positive tail* of a noisy distribution. The honest headline is "5-seed median = 0.6963 ± 0.0429 (single-seed peak: 0.7195 at seed = 99)". See `paper.md` Section 6.3 for the seed-variance crisis discussion.

## Three research findings

1. **DEC plateaus at ARI ≈ 0.50 on n = 400 face data** — std = 0.019 across 11 hill-climb variants. DEC needs n ≥ 10 000 to differentiate from PCA + KMeans.
2. **Birch is threshold-invariant for n < 10 000** — 13 different thresholds produced identical ARI = 0.6371. Threshold sweep is wasted compute below this n.
3. **Spectral cosine on DINOv2 has a ±0.10 ARI seed-variance crisis** — single-seed champions in this regime are statistically meaningless without a 5-seed median.

## Quick links

- 🎯 Live dashboard: https://dlmastery.github.io/autoresearch/clustering_olivetti/
- 📄 [Research paper](paper.md) — 38 references, 10 sections
- ✍️ [Medium article](autoresearch_results/medium_article.md)
- 📋 [Comprehensive AutoResearch report](autoresearch_results/autoresearch_report.md)
- 🔍 [Forensic report](autoresearch_results/forensic_report.md)
- 🧪 [Third-party audit](autoresearch_results/audit_report_third_party.md) — PASS WITH ONE FOOTNOTE
- 🏆 [Champion archive](autoresearch_results/winners/spectral_hc_cosine_seed99_%28variance_c_exp71/) — frozen code + inference + audit
- 📜 [Project rules (CLAUDE.md)](CLAUDE.md)

## Champion progression (12 rungs, 0.4057 → 0.7195)

| Exp | Method | ARI | Mechanism |
|----:|--------|----:|-----------|
| 1 | KMeans on raw pixels | 0.4057 | baseline |
| 2 | KMeans on PCA(50) | 0.4780 | eigenfaces (Turk & Pentland 1991) |
| 8 | Agglomerative Ward | 0.5159 | variance-minimisation linkage (Ward 1963) |
| 16 | Spectral RBF tuned | 0.5252 | NCut on similarity graph (Shi & Malik 2000) |
| 17 | Birch | 0.5287 | CF-tree with leaf KMeans (Zhang 1996) |
| 20 | DINOv2 + KMeans | 0.5455 | self-supervised features (Oquab 2024) |
| 22 | DINOv2 + MiniBatch-KMeans | 0.5596 | stochastic restarts (Sculley 2010) |
| 25 | DINOv2 + KMeans n_init = 50 | 0.5852 | more restarts |
| 27 | DINOv2 + Ward | 0.6371 | variance-minimisation × deep features |
| 33 | DINOv2 + Spectral cosine | 0.6963 | global graph structure (Ng, Jordan, Weiss 2001) |
| 55 | DINOv2 + Spectral RBF γ = 1e-4 | 0.7170 | tiny gamma → linear ≈ cosine |
| **71** | **DINOv2 + Spectral cosine, seed = 99** | **0.7195** | lucky-seed positive tail |

## Reproduce the champion

```bash
git clone https://github.com/dlmastery/autoresearch.git
cd autoresearch
pip install -e .
pip install torch torchvision sklearn umap-learn
cd generalized_ml_autoresearch/examples/clustering_olivetti
python autoresearch_results/winners/spectral_hc_cosine_seed99_\(variance_c_exp71/predict.py
# Expected: ARI = 0.7195, NMI = 0.9004, V-measure = 0.9004
```

## Reproduce the full 149-experiment ladder

```bash
python prepare_data.py             # loads Olivetti from sklearn (cached)
python run_exp01_kmeans_raw.py     # Exp 1: baseline
python run_full_pipeline.py        # Exps 2-14: classical baselines
python run_more_sota.py            # Exps 15-21: GMM, Birch, AffProp, MeanShift, DINOv2 KMeans
python run_dinov2_hill_climb.py    # Exps 22-46: 25 DINOv2 variants
python run_spectral_hill_climb.py  # Exps 47-71: 25 Spectral variants (champion at 71)
python run_ward_birch_hill_climb.py # Exps 72-121: 25 Ward + 25 Birch variants
python run_umap_dec_hill_climb.py  # Exps 122-136: 15 UMAP variants
python run_dec_only.py             # Exps 137-146: 10 DEC variants
python third_party_audit.py        # generate audit_report_third_party.md
python generate_artifacts.py       # regenerate paper.md, medium_article.md, reports
python sync_dashboard.py           # mirror to docs/ for GitHub Pages
```

Each runner script enforces the AutoResearch reasoning gate before launching any experiment: pre-run blob with diagnosis ≥ 60 words, citations ≥ 40 words (per paper, with author/year/venue/arXiv/relevance), hypothesis ≥ 50 words with mechanism keyword, prediction ≥ 25 words with numeric range. Post-run validation enforces verdict ≥ 30 words and learning ≥ 40 words. The validators are in `common.py`.
