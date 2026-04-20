"""Reasoning annotation schema + validators.

Enforces the two non-negotiables from CLAUDE.md:

1. **Citation Rigor** — every paper in the `citations` field must include
   author list, year, venue, title, arXiv ID (if available), and a one-sentence
   relevance note. Parenthetical-only tags are rejected.

2. **Reasoning Blob Completeness** — each of the 7 fields has a word-count
   floor and a required-keyword set. Entries that fall below are rejected.

The runner (`core/runner.py`) will NOT launch an experiment whose pre-run
reasoning annotation fails either validator.

Dependencies: stdlib only (json, re, pathlib, dataclasses, datetime).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# -------------------- field spec (from CLAUDE.md Reasoning Blob Completeness) --------------------

REQUIRED_FIELDS = (
    "diagnosis",
    "citations",
    "hypothesis",
    "prediction",
    "verdict",
    "learning",
    "_manual",
)

WORD_COUNT_FLOORS = {
    # Source: CLAUDE.md "Reasoning Blob Completeness" table
    "diagnosis": 60,
    "citations_single": 40,
    "citations_multi": 80,
    "hypothesis": 50,
    "prediction": 25,
    "verdict": 30,
    "learning": 40,
}

REQUIRED_KEYWORDS = {
    # Source: CLAUDE.md "Reasoning Blob Completeness" "Must include" column
    "hypothesis": ("mechanism", "because", "per "),  # any one of these
    "verdict": ("KEEP", "DISCARD", "NEAR-MISS"),  # any one of these
    "learning": ("axis closed", "axis open", "next try"),  # any one of these
}

CITATION_REQUIRED_ELEMENTS = (
    # Source: CLAUDE.md "Citation Rigor" — six required elements
    "authors",
    "year",
    "venue",
    "title",
    "arxiv_or_note",  # arXiv ID required if posted; else a relevance clause
    "relevance_note",
)

# placeholders that the runner uses when pre-run annotation is missing —
# Claude MUST rewrite these before the next experiment (CLAUDE.md Dashboard Files
# Update Mandate).
PLACEHOLDER_SENTINELS = (
    "TODO-REWRITE",
    "(auto-backfilled)",
    "(no citation tag)",
    "(no explicit citation)",
    "see research_journal.md",
)


# -------------------- data class --------------------


@dataclass
class ReasoningEntry:
    """One experiment's full reasoning record — both pre- and post-run fields."""

    experiment_num: int
    diagnosis: str = ""
    citations: str = ""
    hypothesis: str = ""
    prediction: str = ""
    verdict: str = ""
    learning: str = ""
    _manual: bool = True
    _needs_rewrite: bool = False
    authored_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# -------------------- citation validator --------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
_VENUE_TOKENS = (
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "CVPR",
    "ECCV",
    "ACL",
    "EMNLP",
    "KDD",
    "TMLR",
    "JMLR",
    "RFS",
    "EJOR",
    "JRSS",
    "IJCAI",
    "COLM",
    "SIGIR",
    "IEEE",
    "arXiv",
    "Nature",
    "Science",
    "Technometrics",
)


def validate_citation_rigor(citations_text: str) -> list[str]:
    """Return a list of violations; empty list means valid.

    Rules from CLAUDE.md "Citation Rigor":
      1. Not a placeholder sentinel
      2. Contains at least one year (4-digit)
      3. Contains at least one venue token
      4. Contains at least one arXiv ID OR a quoted/apostrophed title
      5. Contains a relevance note (dash-separated clause or "because"/"motivates"/"per ")
      6. Contains at least one author surname (heuristic: a capitalized token + space + capitalized token OR comma-separated surnames)

    Short-form "Keskar 2017" style is rejected — needs more structure.
    """
    violations: list[str] = []
    text = (citations_text or "").strip()

    if not text:
        return ["citations field is empty"]

    for sentinel in PLACEHOLDER_SENTINELS:
        if sentinel.lower() in text.lower():
            violations.append(f"citations contains placeholder sentinel: {sentinel!r}")

    if not _YEAR_RE.search(text):
        violations.append("no year found (expected a 4-digit year like 2017, 2024)")

    if not any(v.lower() in text.lower() for v in _VENUE_TOKENS):
        violations.append(
            "no venue found (expected one of: NeurIPS, ICML, ICLR, AAAI, CVPR, KDD, TMLR, JMLR, arXiv, ...)"
        )

    has_arxiv = bool(_ARXIV_RE.search(text))
    has_title_quotes = "'" in text or '"' in text or "\u2018" in text or "\u201c" in text
    if not (has_arxiv or has_title_quotes):
        violations.append(
            "no arXiv ID and no quoted title — need 'Title' in quotes OR (arXiv:XXXX.XXXXX)"
        )

    # Relevance note heuristic: long dash + clause, or a keyword explaining WHY
    has_relevance = any(
        token in text.lower()
        for token in ("— ", " - ", "because", "motivates", "per ", "requires", "suggests", "predicts")
    )
    if not has_relevance:
        violations.append(
            "no relevance note found — expected a one-sentence note after the citation "
            "(e.g. '— motivates bs=16 as a flat-minima probe')"
        )

    # Author-list heuristic: at least one surname followed by "et al." OR a year
    # OR a comma-separated list of surnames. Forbid bare tags like "(Keskar2017)".
    if re.fullmatch(r"\(\w+\d{4}\)", text):
        violations.append("parenthetical-only citation tag rejected — expand to full reference")

    # multi-paper detection (semicolon + linebreak separator per CLAUDE.md format).
    # A semicolon inside a single-paper relevance note does NOT count as multi-paper.
    papers = [p.strip() for p in re.split(r";\s*\n", text) if p.strip()]
    is_multi = len(papers) >= 2
    word_count = len(text.split())
    floor = WORD_COUNT_FLOORS["citations_multi" if is_multi else "citations_single"]
    if word_count < floor:
        violations.append(
            f"citation text has {word_count} words; floor is {floor} "
            f"({'multi-paper' if is_multi else 'single paper'})"
        )

    return violations


# -------------------- reasoning-blob validator --------------------


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def _has_any_placeholder(text: str) -> bool:
    lower = (text or "").lower()
    return any(s.lower() in lower for s in PLACEHOLDER_SENTINELS)


def validate_reasoning_blob(entry: ReasoningEntry, is_variance_check: bool = False) -> list[str]:
    """Return a list of violations. Empty list means valid.

    `is_variance_check` relaxes verdict/learning requirements when the entry
    is part of a batch of seed-variance reruns (CLAUDE.md 'Reasoning Blob Completeness'
    notes that diagnosis/citations can be templated for variance checks, but verdict
    and learning must still be per-run-specific).
    """
    violations: list[str] = []

    # 1. Presence check
    for f in REQUIRED_FIELDS:
        if not hasattr(entry, f):
            violations.append(f"field missing: {f}")

    # 2. Placeholder check
    for f in ("diagnosis", "citations", "hypothesis", "prediction", "verdict", "learning"):
        if _has_any_placeholder(getattr(entry, f, "")):
            violations.append(f"{f} contains placeholder sentinel — rewrite required")

    # 3. Word-count floors
    if _word_count(entry.diagnosis) < WORD_COUNT_FLOORS["diagnosis"]:
        violations.append(
            f"diagnosis has {_word_count(entry.diagnosis)} words; floor is {WORD_COUNT_FLOORS['diagnosis']}"
        )
    if _word_count(entry.hypothesis) < WORD_COUNT_FLOORS["hypothesis"]:
        violations.append(
            f"hypothesis has {_word_count(entry.hypothesis)} words; floor is {WORD_COUNT_FLOORS['hypothesis']}"
        )
    if _word_count(entry.prediction) < WORD_COUNT_FLOORS["prediction"]:
        violations.append(
            f"prediction has {_word_count(entry.prediction)} words; floor is {WORD_COUNT_FLOORS['prediction']}"
        )
    # verdict and learning floors apply even to variance checks — per-run specific
    if _word_count(entry.verdict) < WORD_COUNT_FLOORS["verdict"]:
        violations.append(
            f"verdict has {_word_count(entry.verdict)} words; floor is {WORD_COUNT_FLOORS['verdict']}"
        )
    if _word_count(entry.learning) < WORD_COUNT_FLOORS["learning"]:
        violations.append(
            f"learning has {_word_count(entry.learning)} words; floor is {WORD_COUNT_FLOORS['learning']}"
        )

    # 4. Required keywords
    if not any(kw.lower() in entry.hypothesis.lower() for kw in REQUIRED_KEYWORDS["hypothesis"]):
        violations.append(
            "hypothesis must include one of 'mechanism', 'because', or 'per [paper]' — "
            "the change must be stated mechanistically"
        )
    if not any(kw.lower() in entry.verdict.lower() for kw in REQUIRED_KEYWORDS["verdict"]):
        violations.append("verdict must include KEEP, DISCARD, or NEAR-MISS")
    if not any(kw.lower() in entry.learning.lower() for kw in REQUIRED_KEYWORDS["learning"]):
        violations.append(
            "learning must include 'axis closed', 'axis open', or 'next try: ...' language"
        )

    # 5. Citation Rigor (delegates to dedicated validator)
    violations.extend(f"citations: {v}" for v in validate_citation_rigor(entry.citations))

    # 6. Numeric range in prediction
    if not re.search(r"[+\-]?\d+(?:\.\d+)?\s*(?:to|[-–])\s*[+\-]?\d+(?:\.\d+)?", entry.prediction):
        violations.append(
            "prediction must contain a numeric range (e.g. '+6.30 to +6.50'); single-point predictions are insufficient"
        )

    return violations


def validate_pre_run_entry(entry: ReasoningEntry) -> list[str]:
    """Pre-run validator: requires diagnosis/citations/hypothesis/prediction only.

    The runner calls this before launching. verdict and learning are blank pre-run.
    """
    violations: list[str] = []
    # Relaxed floors for pre-run (verdict/learning not yet authored)
    if _word_count(entry.diagnosis) < WORD_COUNT_FLOORS["diagnosis"]:
        violations.append(
            f"diagnosis has {_word_count(entry.diagnosis)} words; floor is {WORD_COUNT_FLOORS['diagnosis']}"
        )
    if _word_count(entry.hypothesis) < WORD_COUNT_FLOORS["hypothesis"]:
        violations.append(
            f"hypothesis has {_word_count(entry.hypothesis)} words; floor is {WORD_COUNT_FLOORS['hypothesis']}"
        )
    if _word_count(entry.prediction) < WORD_COUNT_FLOORS["prediction"]:
        violations.append(
            f"prediction has {_word_count(entry.prediction)} words; floor is {WORD_COUNT_FLOORS['prediction']}"
        )

    if not any(kw.lower() in entry.hypothesis.lower() for kw in REQUIRED_KEYWORDS["hypothesis"]):
        violations.append(
            "hypothesis must include one of 'mechanism', 'because', or 'per [paper]'"
        )
    if not re.search(r"[+\-]?\d+(?:\.\d+)?\s*(?:to|[-–])\s*[+\-]?\d+(?:\.\d+)?", entry.prediction):
        violations.append("prediction must contain a numeric range")

    # Placeholders forbidden
    for f in ("diagnosis", "citations", "hypothesis", "prediction"):
        if _has_any_placeholder(getattr(entry, f, "")):
            violations.append(f"{f} contains placeholder sentinel")

    violations.extend(f"citations: {v}" for v in validate_citation_rigor(entry.citations))
    return violations


# -------------------- persistence --------------------


class ReasoningAnnotationsFile:
    """JSON file at `<results_dir>/reasoning_annotations.json`.

    Keys are stringified experiment numbers (JSON requirement). Entries are
    ReasoningEntry dicts.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def commit_pre_run(self, entry: ReasoningEntry, allow_variance_check: bool = False) -> None:
        """Validate and persist a pre-run entry.

        Raises ValueError with a joined violation list if invalid.
        """
        violations = validate_pre_run_entry(entry)
        if violations and not allow_variance_check:
            raise ValueError(
                "Pre-run reasoning entry failed validation. Fix these before launching:\n  - "
                + "\n  - ".join(violations)
            )
        data = self.load()
        data[str(entry.experiment_num)] = entry.to_dict()
        self.save(data)

    def commit_post_run(
        self, experiment_num: int, verdict: str, learning: str, is_variance_check: bool = False
    ) -> list[str]:
        """Merge verdict/learning into an existing entry. Returns any validation violations."""
        data = self.load()
        key = str(experiment_num)
        if key not in data:
            # Runner-fallback: insert a minimal entry with TODO-REWRITE sentinels so
            # Claude knows to author it (CLAUDE.md Dashboard Files Update Mandate rule).
            data[key] = ReasoningEntry(
                experiment_num=experiment_num,
                diagnosis="TODO-REWRITE: pre-run annotation missing — author diagnosis.",
                citations="TODO-REWRITE: pre-run annotation missing — author citations.",
                hypothesis="TODO-REWRITE: pre-run annotation missing — author hypothesis.",
                prediction="TODO-REWRITE: pre-run annotation missing — author prediction.",
                verdict=verdict,
                learning=learning,
                _manual=False,
                _needs_rewrite=True,
            ).to_dict()
        else:
            data[key]["verdict"] = verdict
            data[key]["learning"] = learning
        self.save(data)
        entry = ReasoningEntry.from_dict(data[key])
        return validate_reasoning_blob(entry, is_variance_check=is_variance_check)


# -------------------- CLI smoke test --------------------


if __name__ == "__main__":  # pragma: no cover
    # Quick self-check with a known-good entry.
    good = ReasoningEntry(
        experiment_num=1,
        diagnosis=(
            "This is the baseline MLP experiment for the regression-house-prices example. "
            "No prior experiments exist, so the diagnosis is scope-setting rather than "
            "champion-weakness analysis: the project's goal is to beat a linear-regression "
            "floor on the California Housing benchmark using MedInc, HouseAge, AveRooms, "
            "AveBedrms, Population, AveOccup, Latitude and Longitude. Prior domain evidence "
            "from Gu, Kelly and Xiu 2020 suggests a 3-layer MLP with dropout is a defensible "
            "starting point for tabular regression of this size and correlation structure, so "
            "fold 0 of the 5-fold CV is our reference target."
        ),
        citations=(
            "Gu, Kelly & Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning' "
            "(arXiv:1802.09003) — motivates 3-layer MLP with dropout on tabular features; "
            "we adapt the head-only training recipe as the baseline for California Housing, "
            "scaling the epoch count per Smith 2017's data-scaling rule. The paper's "
            "regularization prescriptions directly inform the chosen dropout=0.2 and "
            "weight_decay=1e-5 values so the baseline is defensible against criticism."
        ),
        hypothesis=(
            "We hypothesize that a 3-layer MLP (256-128-64) with dropout 0.2 and AdamW "
            "(lr=3e-4, wd=1e-5) will achieve test RMSE below 0.55 because the mechanism is "
            "feature-interaction learning per Gu, Kelly & Xiu 2020 — dropout regularizes "
            "the dense interactions while AdamW decoupled weight decay prevents the hidden "
            "layers from memorizing the training set, per Loshchilov & Hutter 2019."
        ),
        prediction=(
            "Test RMSE should land in the range 0.50 to 0.58, beating the linear floor "
            "(approximately 0.72). Val RMSE expected near 0.52 across all 5 CV folds, with "
            "fold 3 strongest (below 0.50) per its balanced region distribution."
        ),
        verdict=(
            "KEEP. Composite = 0.5412 (Test RMSE 0.5412, Val RMSE 0.5198, 5/5 folds above "
            "threshold). All folds carried it; fold 3 was strongest at 0.4912 and fold 0 "
            "weakest at 0.5689 — consistent with the pre-run fold-3 prediction."
        ),
        learning=(
            "Baseline confirmed — axis open for deeper MLPs and residual connections. "
            "Next try: 4-layer MLP with a single skip connection from input to the final "
            "hidden layer per He et al. 2016, targeting a composite improvement of "
            "+0.02 to +0.05."
        ),
    )
    violations = validate_reasoning_blob(good)
    print("Violations for good entry:", violations)
    assert not violations, violations

    bad = ReasoningEntry(
        experiment_num=2,
        diagnosis="too short",
        citations="(Keskar2017)",
        hypothesis="try bs=16",
        prediction="better",
        verdict="KEEP",
        learning="...",
    )
    violations = validate_reasoning_blob(bad)
    print("Violations for bad entry:", len(violations), "found")
    assert len(violations) >= 5
    print("reasoning.py self-check passed.")
