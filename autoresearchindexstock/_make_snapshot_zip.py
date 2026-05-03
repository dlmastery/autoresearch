"""Build a comprehensive snapshot zip for upload to Drive.

Includes everything needed to reproduce the QQQ project state:
- autoresearchindexstock/ (the QQQ project itself, full)
- docs/index_stock_dashboard/ (the live dashboard mirror)
- autoresearch/model + data + evaluation + run_autoresearch.py + CLAUDE.md

Excludes large/regenerable junk: .git, __pycache__, .data_cache, *.pyc, etc.
"""
import zipfile
import os
import datetime
from pathlib import Path

ROOT = Path("C:/Users/evija/autoresearch")
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M")
ZIP_PATH = ROOT.parent / f"autoresearch_qqq_snapshot_{TS}.zip"

EXCLUDE_DIRS = {".git", "__pycache__", ".data_cache", ".pytest_cache", ".idea",
                ".vscode", "node_modules", ".mypy_cache", ".ipynb_checkpoints",
                "venv", ".venv"}
EXCLUDE_EXT = {".pyc", ".pyo"}

INCLUDE_PATHS = [
    "autoresearchindexstock",
    "docs/index_stock_dashboard",
    "autoresearch/model",
    "autoresearch/data",
    "autoresearch/evaluation",
    "autoresearch/run_autoresearch.py",
    "autoresearch/CLAUDE.md",
]


def main():
    count = 0
    total_size = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for inc in INCLUDE_PATHS:
            p = ROOT / inc
            if not p.exists():
                print(f"[skip] {inc} does not exist")
                continue
            if p.is_file():
                arcname = inc.replace(os.sep, "/")
                zf.write(p, arcname)
                count += 1
                total_size += p.stat().st_size
                continue
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for f in files:
                    if any(f.endswith(e) for e in EXCLUDE_EXT):
                        continue
                    fp = Path(root) / f
                    arcname = str(fp.relative_to(ROOT)).replace(os.sep, "/")
                    try:
                        zf.write(fp, arcname)
                        count += 1
                        total_size += fp.stat().st_size
                    except Exception as e:
                        print(f"[skip] {arcname}: {e}")

    print(f"Wrote {ZIP_PATH}")
    print(f"  Files: {count}")
    print(f"  Uncompressed: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Compressed:   {ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    return ZIP_PATH


if __name__ == "__main__":
    main()
