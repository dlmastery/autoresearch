---
layout: default
title: AutoResearch on FDB fraudecom
---

# AutoResearch on FDB `fraudecom`

> **Champion: XGBoost test AUC 0.6097** on the FDB-exact chronological 80/20 protocol — beats every published FDB open-source AutoML baseline (AutoGluon, H2O, Auto-sklearn) by +0.088 to +0.095, lands 0.026 below the proprietary AFD-TFI ceiling (0.636).

## Quick links

- 🎯 [Live Dashboard](fraud_ecommerce/) — interactive leaderboard with reasoning panel for every experiment
- 📄 [Research Paper](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/fraud_ecommerce/paper.md) — 5,500-word academic write-up
- ✍️ [Medium Article](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/fraud_ecommerce/autoresearch_results/medium_article.md) — narrative blog
- 📋 [Comprehensive Report](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/fraud_ecommerce/autoresearch_results/autoresearch_report.md)
- 🔍 [Forensic Audit](https://github.com/dlmastery/autoresearch/blob/master/generalized_ml_autoresearch/examples/fraud_ecommerce/autoresearch_results/forensic_report.md)
- 🥇 [Winner Archive](https://github.com/dlmastery/autoresearch/tree/master/generalized_ml_autoresearch/examples/fraud_ecommerce/autoresearch_results/winners) — model checkpoint, code, inference, Colab

## What this is

This is a complete autoresearch project applying the `generalized_ml_autoresearch` framework to the most-unsaturated dataset in the Amazon Science Fraud Dataset Benchmark (FDB; Grover et al. 2023, arXiv:2208.14417).

28 honest experiments across 8 model families (XGBoost, LightGBM, CatBoost, MLP, Energy-Based Model, Autoencoder anomaly, Contrastive SimCLR-tabular, Explainable Boosting Machine), driven by Claude Code as the research agent following the 7-step diagnose → cite → hypothesize → predict → run → analyze → checkpoint protocol.

Four protocol bugs surfaced during the loop:
1. Stratified CV inflated AUC by 0.27 on time-ordered fraud data
2. Velocity feature computation leaked val rows into their own predictor values (val_AUC 0.9988 vs test_AUC 0.5297)
3. Framework silently dropped `scale_pos_weight` parameter
4. A "recency improvement" was actually 100% reward hacking (changed test set size)

All four are now Hard Rules in the framework's CLAUDE.md template.

## See also

- [Repository](https://github.com/dlmastery/autoresearch) — full source
- [FX AutoResearch Dashboard](dashboard/) — the parent project (FX prediction)
