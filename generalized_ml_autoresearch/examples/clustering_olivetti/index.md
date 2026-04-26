---
layout: default
title: Olivetti Clustering Autoresearch
---

# Clustering Autoresearch — Olivetti Faces

> **Champion:** DINOv2 ViT-S/14 + Spectral Clustering, cosine affinity, seed = 99 — ARI = **0.7195**
> **Honest headline:** 5-seed median ARI = **0.6963 ± 0.0429** (149 experiments, 6 backbone families)

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
