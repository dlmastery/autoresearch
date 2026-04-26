"""Sync clustering dashboard + artifacts to docs/clustering_olivetti/ for GitHub Pages."""
from __future__ import annotations
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # autoresearch/ repo root
SRC = HERE / "autoresearch_results"
DST = REPO_ROOT / "docs" / "clustering_olivetti"

def main():
    DST.mkdir(parents=True, exist_ok=True)
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
        if src.exists():
            shutil.copy2(src, DST / dst_name)
            print(f"  {src_name} -> {DST.relative_to(REPO_ROOT)}/{dst_name} ({src.stat().st_size/1024:.1f} KB)")
    # Top-level project files
    for top in ["paper.md", "paper_abstract.md", "README.md", "CLAUDE.md"]:
        src = HERE / top
        if src.exists():
            shutil.copy2(src, DST / top)
            print(f"  {top} -> {DST.relative_to(REPO_ROOT)}/{top}")
    # index.md (Pages landing)
    if (HERE / "index.md").exists():
        shutil.copy2(HERE / "index.md", DST / "INDEX.md")
    print(f"\nMirror updated: {DST.relative_to(REPO_ROOT)}")
    print("Pages URL after push: https://dlmastery.github.io/autoresearch/clustering_olivetti/")


if __name__ == "__main__":
    main()
