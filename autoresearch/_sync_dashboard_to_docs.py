"""Sync the live dashboard + data files into docs/dashboard/ so GitHub Pages
serves the latest state at https://dlmastery.github.io/autoresearch/dashboard/

MUST run before every git commit that touches experiment state. See CLAUDE.md
'Dashboard Files Update Mandate' and the 'GitHub Pages Dashboard Sync' rule.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # autoresearch repo root
SRC = ROOT / "autoresearch" / "autoresearch_results"
DST = ROOT / "docs" / "dashboard"
DST.mkdir(parents=True, exist_ok=True)

# Files the dashboard fetches at runtime (keep this list aligned with
# dashboard.html fetch() calls)
REQUIRED = ["dashboard.html", "experiment_log.jsonl", "best_config.json",
            "reasoning_annotations.json"]
# Optional docs the dashboard links to (via <a href="...">)
OPTIONAL = ["experiment_summary.md", "autoresearch_report.md",
            "research_journal.md", "medium_article.md"]

for name in REQUIRED:
    src = SRC / name
    if not src.exists():
        raise FileNotFoundError(
            f"REQUIRED dashboard file missing: {src}. "
            f"Cannot sync to docs/dashboard/ — fix the runner first."
        )
    # dashboard.html lands as index.html so GitHub Pages routes
    # /dashboard/ directly to it
    tgt_name = "index.html" if name == "dashboard.html" else name
    shutil.copy2(src, DST / tgt_name)

for name in OPTIONAL:
    src = SRC / name
    if src.exists():
        shutil.copy2(src, DST / name)

# Copy trade_logs/ so the dashboard can fetch per-trade daily data for
# real equity curves (not just stepped per-fold endpoints).
trade_src = SRC / "trade_logs"
trade_dst = DST / "trade_logs"
if trade_src.exists():
    trade_dst.mkdir(exist_ok=True)
    n_copied = 0
    for csv in trade_src.glob("exp*_trades.csv"):
        shutil.copy2(csv, trade_dst / csv.name)
        n_copied += 1
    print(f"  [trade_logs] copied {n_copied} per-trade CSV files")

# Add / refresh a lightweight landing pointer at docs/index.md so the
# github.io home surfaces the dashboard
index = ROOT / "docs" / "index.md"
if index.exists():
    content = index.read_text(encoding="utf-8")
    marker = "[Live dashboard]"
    if marker not in content:
        addition = (
            "\n\n---\n\n## Live experiment dashboard\n\n"
            "[Live dashboard](./dashboard/) — every experiment, per-fold "
            "Sharpe/IC/hit-rate, click a row to see its arXiv-cited reasoning. "
            "Auto-synced from `autoresearch/autoresearch_results/` on every "
            "commit per CLAUDE.md 'GitHub Pages Dashboard Sync' rule.\n"
        )
        index.write_text(content + addition, encoding="utf-8")

# Regenerate the all-experiments Excel download (with embedded chart)
try:
    import subprocess
    subprocess.run([
        "python",
        str(Path(__file__).parent / "_export_equity_excel.py")
    ], check=True, capture_output=True)
    print("  [excel] autoresearch_equity.xlsx refreshed")
except Exception as e:
    print(f"  [excel] WARN: failed to regenerate xlsx: {e}")

total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
n_files = sum(1 for f in DST.rglob("*") if f.is_file())
print(f"Synced {n_files} files to docs/dashboard/ "
      f"({total / 1024 / 1024:.2f} MB). Ready for git commit + push.")
