# Olivetti Faces Clustering Autoresearch Project

**Champion: Agglomerative Ward on PCA(50) — ARI = 0.7195**

14 honest experiments. 8 model families. Classical methods beat deep clustering on this small-n benchmark.

## Quick links

- 🎯 Live dashboard: https://dlmastery.github.io/autoresearch/clustering_olivetti/
- 📄 [Research paper](paper.md)
- ✍️ [Medium article](autoresearch_results/medium_article.md)
- 📋 [Comprehensive report](autoresearch_results/autoresearch_report.md)
- 🔍 [Forensic audit](autoresearch_results/forensic_report.md)
- 🧪 [Third-party audit](autoresearch_results/audit_report_third_party.md)
- 📜 [Project rules (CLAUDE.md)](CLAUDE.md)

## Reproduce

```bash
git clone https://github.com/dlmastery/autoresearch.git
cd autoresearch
pip install -e .
pip install torch torchvision interpret
cd generalized_ml_autoresearch/examples/clustering_olivetti
python prepare_data.py        # loads Olivetti from sklearn
python run_exp01_kmeans_raw.py  # baseline
python run_full_pipeline.py   # Exps 2-14
python third_party_audit.py   # audit
python generate_artifacts.py  # paper, medium, reports
```
