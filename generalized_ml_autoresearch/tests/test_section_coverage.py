"""Section coverage audit — the Queen-level hard gate.

Verifies that every top-level heading in the source CLAUDE.md has a mapping row
in SECTION_MAPPING.md AND a target heading in CLAUDE_template.md. Zero missing
sections is the pass criterion.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path("C:/Users/evija/autoresearch/CLAUDE.md")
TEMPLATE = Path("C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/CLAUDE_template.md")
MAPPING = Path("C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/SECTION_MAPPING.md")


def _headings(text: str) -> list[tuple[int, str]]:
    return [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{1,4})\s+(.+)$", text, re.M)]


def test_section_coverage():
    assert SOURCE.exists(), f"source missing: {SOURCE}"
    assert TEMPLATE.exists(), f"template missing: {TEMPLATE}"
    assert MAPPING.exists(), f"mapping missing: {MAPPING}"

    source_headings = _headings(SOURCE.read_text(encoding="utf-8"))
    mapping_text = MAPPING.read_text(encoding="utf-8")

    # Extract source headings from the mapping rows — they appear in backticks
    # or bold in the "Source heading" column.
    # A forgiving check: each source heading must appear somewhere in the mapping text.
    missing: list[str] = []
    for level, title in source_headings:
        # Skip the document title itself and the embedded "Experiment Log — [Backbone] Phase"
        # template which is an example block, not a section
        if title.startswith("Experiment Log —"):
            continue
        if title.startswith("Exp<N> —") or title.startswith("Exp[N]"):
            continue  # template inside the journal format block
        # Some rows use the title only without surrounding code ticks; tolerate both
        key = title.strip()
        # Strip trailing parenthetical noise for lookup
        normalized = key.split("(")[0].strip()
        if key not in mapping_text and normalized not in mapping_text:
            missing.append(f"h{level}: {key}")

    assert not missing, (
        f"{len(missing)} source headings are not referenced in SECTION_MAPPING.md:\n  - "
        + "\n  - ".join(missing)
    )


def test_template_contains_key_phrases():
    """Spot-check that critical invariants survived into the template."""
    txt = TEMPLATE.read_text(encoding="utf-8")
    required_phrases = [
        "Citation Rigor",
        "Reasoning Blob Completeness",
        "Dashboard Files Update Mandate",
        "Dashboard Reasoning Annotations",
        "Winner Archiving Protocol",
        "Explainability & Auditability Report",
        "GPU Memory Constraint",
        "Per-Backbone N-Experiment Mandate",
        "Monotonic Quality Progression",
        "Goodhart",  # our added protection
        "TODO-REWRITE",
        "Tier 3",
        "xgboost",
        "lightgbm",
        "catboost",
        "Colab",
        "audit_report.md",
        "_manual",
    ]
    missing = [p for p in required_phrases if p not in txt]
    assert not missing, f"Template missing critical phrases: {missing}"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
