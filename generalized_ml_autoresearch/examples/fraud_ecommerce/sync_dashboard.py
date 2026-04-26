"""Sync dashboard + data files to docs/ for GitHub Pages serving.

Usage: run after every batch of experiments before `git push`.
  python sync_dashboard.py

Mirrors:
  autoresearch_results/dashboard.html             → docs/fraud_ecommerce/index.html
  autoresearch_results/experiment_log.jsonl       → docs/fraud_ecommerce/experiment_log.jsonl
  autoresearch_results/reasoning_annotations.json → docs/fraud_ecommerce/reasoning_annotations.json
  autoresearch_results/best_config.json           → docs/fraud_ecommerce/best_config.json
"""
from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
# HERE = examples/fraud_ecommerce/  →  parents[2] = autoresearch/ (the cloned dlmastery/autoresearch repo root)
REPO_ROOT = HERE.parents[2]
SRC = HERE / "autoresearch_results"
DST = REPO_ROOT / "docs" / "fraud_ecommerce"


def main():
    if not SRC.exists():
        raise SystemExit(f"source directory not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    # dashboard.html → index.html so the GitHub Pages URL routes directly
    pairs = [
        ("dashboard.html", "index.html"),
        ("experiment_log.jsonl", "experiment_log.jsonl"),
        ("reasoning_annotations.json", "reasoning_annotations.json"),
        ("best_config.json", "best_config.json"),
        ("medium_article.md", "medium_article.md"),
        ("autoresearch_report.md", "autoresearch_report.md"),
        ("forensic_report.md", "forensic_report.md"),
        ("forensic_checkpoint.md", "forensic_checkpoint.md"),
        ("audit_report_third_party.md", "audit_report_third_party.md"),
        ("experiment_summary.md", "experiment_summary.md"),
        ("research_journal.md", "research_journal.md"),
    ]
    for src_name, dst_name in pairs:
        src = SRC / src_name
        if not src.exists():
            print(f"  skip (not present): {src_name}")
            continue
        dst = DST / dst_name
        shutil.copy2(src, dst)
        print(f"  copied {src.name} -> {dst.relative_to(REPO_ROOT)}  "
              f"({dst.stat().st_size/1024:.1f} KB)")

    print(f"\nGitHub Pages mirror updated at: {DST.relative_to(REPO_ROOT)}")
    print(f"After pushing, the dashboard will be at:")
    print(f"  https://<your-github-username>.github.io/<your-repo-name>/fraud_ecommerce/")


if __name__ == "__main__":
    main()
