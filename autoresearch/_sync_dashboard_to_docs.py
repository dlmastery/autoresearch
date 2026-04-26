"""Sync the live dashboard + data files into docs/dashboard/ so GitHub Pages
serves the latest state at https://dlmastery.github.io/autoresearch/dashboard/

MUST run before every git commit that touches experiment state. See CLAUDE.md
'Dashboard Files Update Mandate' and the 'GitHub Pages Dashboard Sync' rule.
"""
import json
import re
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
    # Per-experiment daily CSVs
    n_csv = 0
    for csv in trade_src.glob("exp*_trades.csv"):
        shutil.copy2(csv, trade_dst / csv.name)
        n_csv += 1
    # Ensemble-winner daily CSVs (mega_ensemble, ensemble_3way_gbm, ...)
    n_ens = 0
    for csv in trade_src.glob("*_trades.csv"):
        if csv.name.startswith("exp"):
            continue
        shutil.copy2(csv, trade_dst / csv.name)
        n_ens += 1
    # Per-experiment & ensemble JSON summaries (so the dashboard can show
    # per-fold totals if it wants to; also useful as standalone artefacts).
    n_sum = 0
    for js in trade_src.glob("*_trade_summary.json"):
        shutil.copy2(js, trade_dst / js.name)
        n_sum += 1

    # Build a manifest of which experiments and which ensembles have daily
    # CSVs on disk. The dashboard fetches this once on each refresh and
    # uses it to gate the Trades column link per row.
    EXP_RE = re.compile(r"^exp(\d+)_trades\.csv$")
    experiments: list[int] = []
    ensembles: list[dict] = []
    for csv in sorted(trade_dst.glob("*_trades.csv")):
        m = EXP_RE.match(csv.name)
        if m:
            experiments.append(int(m.group(1)))
            continue
        # Ensemble winner — pull headline metrics from its summary JSON if
        # present so the dashboard can show "Sharpe +X.YYYY" on the chip.
        stem = csv.stem.replace("_trades", "")
        sum_path = trade_dst / f"{stem}_trade_summary.json"
        sharpe = rows = wr = ret_pct = None
        if sum_path.exists():
            try:
                s = json.loads(sum_path.read_text(encoding="utf-8"))
                sharpe = s.get("test_sharpe")
                rows = s.get("total_trades")
                wr = s.get("overall_win_rate")
                ret_pct = s.get("total_return_pct")
            except Exception:  # pragma: no cover — manifest is best-effort
                pass
        # Friendly label for known ensembles
        label_map = {
            "mega_ensemble": "Mega-Ensemble (GBM3 + LSTM1)",
            "ensemble_3way_gbm": "3-Way GBM (rank-avg, seq=60)",
        }
        ensembles.append({
            "name": stem,
            "label": label_map.get(stem, stem.replace("_", " ").title()),
            "file": csv.name,
            "summary_file": sum_path.name if sum_path.exists() else None,
            "sharpe": sharpe,
            "rows": rows,
            "win_rate": wr,
            "return_pct": ret_pct,
        })

    manifest = {
        "experiments": sorted(set(experiments)),
        "ensembles": ensembles,
        "n_experiment_csvs": len(set(experiments)),
        "n_ensemble_csvs": len(ensembles),
    }
    # Write the manifest BOTH to the source (so the local dashboard works)
    # and to the docs mirror (so GitHub Pages serves it).
    for target_dir in (trade_src, trade_dst):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(f"  [trade_logs] copied {n_csv} per-experiment CSVs, "
          f"{n_ens} ensemble CSVs, {n_sum} summary JSONs; "
          f"manifest: {len(set(experiments))} exps + {len(ensembles)} ensembles")

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
