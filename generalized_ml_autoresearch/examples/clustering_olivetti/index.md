---
layout: default
title: Olivetti Clustering Autoresearch
---

# Clustering Autoresearch — Olivetti Faces

> **Unconditional champion:** Exp 147 — 5-seed CSPA co-association ensemble (Strehl 2002) of Spectral cosine on DINOv2 ViT-S/14 — ARI = **0.7346** (NMI = 0.9093, silhouette = 0.1017)
> **Deployment-mode champion:** Exp 149 — silhouette-rejection — conditional ARI = **0.8740** on 317/400 kept samples
> **Resolved crisis:** the ±0.10 single-seed Spectral variance is *eliminated* by the CSPA ensemble (152 experiments, 6 backbone families)

## Quick links
- 🎯 [Live Dashboard](clustering_olivetti/)
- 📄 [Paper](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/clustering_olivetti/paper.md) — 38 references, 10 sections
- ✍️ [Medium article](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/clustering_olivetti/autoresearch_results/medium_article.md)
- 📋 [AutoResearch report](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/clustering_olivetti/autoresearch_results/autoresearch_report.md)
- 🔍 [Forensic report](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/clustering_olivetti/autoresearch_results/forensic_report.md)
- 🧪 [Third-party audit](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/clustering_olivetti/autoresearch_results/audit_report_third_party.md) — PASS WITH ONE FOOTNOTE
- 🏆 [Champion archive](https://github.com/dlmastery/autoresearch/tree/master/generalized_ml_autoresearch/examples/clustering_olivetti/autoresearch_results/winners/spectral_hc_cosine_seed99_%28variance_c_exp71)

## Three research findings

1. **DEC plateaus at ARI ≈ 0.50 on n = 400 face data** — std = 0.019 across 11 hill-climb variants
2. **Birch is threshold-invariant for n < 10 000** — 13 thresholds → identical ARI = 0.6371
3. **Spectral cosine on DINOv2 has a ±0.10 ARI seed-variance crisis** — single-seed champions are statistically meaningless
