# FDB fraudecom — Autoresearch Example

This directory is a complete autoresearch project applying the `generalized_ml_autoresearch` framework to the most-unsaturated dataset in the Amazon Science Fraud Dataset Benchmark (FDB; Grover et al. 2023, arXiv:2208.14417).

## TL;DR

Champion: **XGBoost (Exp 25) on FDB-exact 80/20 protocol with engineered velocity features → test AUC = 0.6097**.

Beats every FDB open-source AutoML baseline by +0.088 to +0.095. Sits 0.026 below the proprietary AFD-TFI ceiling (0.636). 28 honest experiments. 4 protocol bugs surfaced and patched. Full third-party-grade audit. Live dashboard.

## Open the dashboard

- **Local:** `python -m http.server 8765 --directory autoresearch_results` then open http://localhost:8765/dashboard.html
- **GitHub Pages:** https://dlmastery.github.io/autoresearch/fraud_ecommerce/

## Read the artifacts

| Artifact | What it is |
|---------|-----------|
| [`paper.md`](paper.md) | 5,500-word academic paper |
| [`paper_abstract.md`](paper_abstract.md) | 300-word abstract |
| [`autoresearch_results/medium_article.md`](autoresearch_results/medium_article.md) | 4,000-word narrative blog article |
| [`autoresearch_results/autoresearch_report.md`](autoresearch_results/autoresearch_report.md) | Comprehensive technical report |
| [`autoresearch_results/forensic_report.md`](autoresearch_results/forensic_report.md) | Independent forensic audit |
| [`autoresearch_results/forensic_checkpoint.md`](autoresearch_results/forensic_checkpoint.md) | Audit-grade state snapshot |
| [`autoresearch_results/audit_report_third_party.md`](autoresearch_results/audit_report_third_party.md) | 12-section third-party audit |
| [`autoresearch_results/experiment_summary.md`](autoresearch_results/experiment_summary.md) | Master tabular log of 28 experiments |
| [`autoresearch_results/research_journal.md`](autoresearch_results/research_journal.md) | Markdown narrative twin of the reasoning JSON |
| [`autoresearch_results/reasoning_annotations.json`](autoresearch_results/reasoning_annotations.json) | Per-experiment diagnosis/citations/hypothesis/prediction/verdict/learning |
| [`autoresearch_results/experiment_log.jsonl`](autoresearch_results/experiment_log.jsonl) | Full 28-experiment record (one JSON per line) |
| [`autoresearch_results/best_config.json`](autoresearch_results/best_config.json) | Champion config + leaderboard |
| [`autoresearch_results/winners/xgboost_exp6_velocity_features/`](autoresearch_results/winners/) | Self-contained winner archive (README, config, model, code, audit, Colab) |
| [`memory/project_autoresearch_checkpoint.md`](memory/project_autoresearch_checkpoint.md) | Crash-recovery checkpoint state |
| [`CLAUDE.md`](CLAUDE.md) | Project rules — what governs Claude Code sessions on this project |
| [`PUSH_TO_GITHUB.md`](PUSH_TO_GITHUB.md) | Instructions for pushing your fork |

## Reproduce the champion

```bash
git clone https://github.com/dlmastery/autoresearch.git
cd autoresearch
pip install -e .
pip install xgboost==3.2.0 lightgbm==4.6.0 catboost==1.2.10 interpret==0.7.8

python generalized_ml_autoresearch/examples/fraud_ecommerce/prepare_data.py
python generalized_ml_autoresearch/examples/fraud_ecommerce/add_velocity_features.py
python generalized_ml_autoresearch/examples/fraud_ecommerce/run_exp25_fdb_exact.py
```

Expected output: `composite=0.6097 ± 0.005` (5-seed std=0.006).

## What's in this directory

- `prepare_data.py` — FDB-style preprocessing producing `data/features.csv`
- `add_velocity_features.py` — train-period frequency + Bayesian-smoothed target encodings
- `add_rolling_features.py` — rolling 1d/7d/30d velocity counts (next-experiment material)
- `audit.py`, `audit_temporal.py` — initial leakage and temporal-shift audits
- `third_party_audit.py` — 12-section third-party-grade audit producing `audit_report_third_party.md`
- `fdb_verbatim_pipeline.py` — strict FDB-verbatim apples-to-apples 4-backbone baseline
- `run_example.py`, `run_exp*.py` — individual experiment runners
- `run_full_sweep.py` — multi-backbone sweep runner (used by the quarantined blind-sweep batch)
- `run_novel_methods.py` — Energy-Based Model + Autoencoder + Contrastive learning
- `run_interpretml_ebm.py` — InterpretML Explainable Boosting Machine
- `generate_artifacts.py` — produces FX-style artifacts (summary, journal, checkpoint, winner archive)
- `sync_dashboard.py` — copies dashboard to docs/ for GitHub Pages serving
- `seed_reasoning_exp*.json` — pre-run reasoning entries for each experiment

## Honest position vs FDB published baselines

| System | Test AUC | Notes |
|--------|----------|-------|
| **Our XGBoost (Exp 25)** | **0.6097** | Strict-FDB-compliant champion |
| Our InterpretML EBM (Exp 24) | 0.6057 | -0.040 from #1, glass-box interpretable |
| AFD-TFI (proprietary) | 0.6360 | Likely uses AWS IP-intelligence we cannot replicate |
| AutoGluon (FDB published) | 0.5220 | -0.088 from us |
| H2O (FDB published) | 0.5180 | -0.092 |
| Auto-sklearn (FDB published) | 0.5150 | -0.095 |

## Lessons encoded as Hard Rules in the framework

This project surfaced 4 protocol bugs that are now permanent rules in `generalized_ml_autoresearch/templates/CLAUDE_template.md`:

1. **Reward Hacking Prohibition** — never change the test set; verify SHA-256 hash on `sorted(test_idx)`.
2. **Velocity-feature train alignment** — use `n_train = n - n_val - n_test`, not the documented benchmark's 80% if your runner uses different val/test fractions.
3. **Stratified-CV ban for time-ordered data** — chronological holdout or walk-forward only.
4. **Wiring verification protocol** — every new config field must be tested with an extreme A/B value to confirm it's actually applied.

## Citation

If you use this project's findings or framework, please cite:

```
Ranti 2026 (working paper) "AutoResearch: Autonomous Machine Learning Optimization for
  Foreign Exchange Prediction via Agent-Driven Experiment Design."
Grover, Xu, Tittelfitz, Cheng, Li, Zablocki, Liu & Zhou 2023 arXiv "Fraud Dataset
  Benchmark and Applications" (arXiv:2208.14417).
```

## License

MIT, inheriting from the parent `dlmastery/autoresearch` repository.
