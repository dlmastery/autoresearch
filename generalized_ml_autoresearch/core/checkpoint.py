"""Checkpoint manager — implements Crash-Recovery Checkpointing rules.

Every experiment + every 5 minutes of reasoning saves a self-contained markdown
file at `<memory_dir>/project_autoresearch_checkpoint.md` with:

  - Current champion snapshot (config + composite + per-fold table)
  - Last experiment result
  - EXACT next command to paste
  - Full experiment history summary
  - Key learnings / exhausted axes
  - Session start instructions

Also maintains `<results_dir>/experiment_summary.md` (master experiment log).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    experiment_num: int
    backbone: str
    description: str
    config: dict[str, Any]
    composite: float
    val_primary: float
    test_primary: float
    per_fold_test: list[float] = field(default_factory=list)
    per_fold_val: list[float] = field(default_factory=list)
    status: str = "DISCARD"  # KEEP / DISCARD / NEAR-MISS
    seconds_elapsed: float = 0.0
    timestamp: str = ""
    secondary_metrics: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, memory_dir: str | Path, results_dir: str | Path,
                 project_name: str, python_exe: str = "python"):
        self.memory_dir = Path(memory_dir)
        self.results_dir = Path(results_dir)
        self.project_name = project_name
        self.python_exe = python_exe
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._last_reasoning_save = time.time()

    @property
    def checkpoint_path(self) -> Path:
        return self.memory_dir / "project_autoresearch_checkpoint.md"

    @property
    def summary_path(self) -> Path:
        return self.results_dir / "experiment_summary.md"

    # ---------------- experiment-level checkpoint ----------------

    def refresh(
        self,
        record: ExperimentRecord,
        champion: ExperimentRecord | None,
        next_command: str,
        next_rationale: str,
        history: list[ExperimentRecord],
        exhausted_axes: list[str],
    ) -> None:
        """Overwrite the checkpoint with the latest state.

        Called after EVERY experiment (per CLAUDE.md Crash-Recovery rule).
        """
        text = self._render_checkpoint(record, champion, next_command, next_rationale,
                                         history, exhausted_axes)
        self.checkpoint_path.write_text(text, encoding="utf-8")

    def append_to_summary(self, record: ExperimentRecord) -> None:
        """Append a human-readable row to experiment_summary.md."""
        block = self._render_summary_block(record)
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(block)

    def reasoning_time_checkpoint(self, current_thinking: str) -> bool:
        """Called during long reasoning phases. Writes to checkpoint if > 5 min since last save.
        Returns True if a save happened.
        """
        now = time.time()
        if now - self._last_reasoning_save < 300:
            return False
        append_path = self.checkpoint_path
        if append_path.exists():
            existing = append_path.read_text(encoding="utf-8")
        else:
            existing = ""
        stamp = datetime.now().isoformat(timespec="seconds")
        new = existing + f"\n\n### Reasoning checkpoint at {stamp}\n\n{current_thinking}\n"
        append_path.write_text(new, encoding="utf-8")
        self._last_reasoning_save = now
        return True

    # ---------------- rendering ----------------

    def _render_checkpoint(
        self, record, champion, next_command, next_rationale, history, exhausted_axes
    ) -> str:
        lines: list[str] = []
        lines.append(f"# Autoresearch checkpoint — {self.project_name}")
        lines.append(f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n")

        lines.append("## Session start instructions\n")
        lines.append("1. Read this checkpoint (you are here).")
        lines.append("2. Read the hardware log (if present).")
        lines.append(f"3. Tail `{self.results_dir}/experiment_log.jsonl` and `{self.results_dir}/best_config.json`.")
        lines.append("4. Resume the 7-step experiment loop from the 'Next experiment' block below.")
        lines.append(f"5. Start the dashboard: `{self.python_exe} -m http.server 8765 --directory {self.results_dir}`.")
        lines.append("")

        lines.append("## Current champion (global best)\n")
        if champion is not None:
            lines.append(f"- **Backbone:** `{champion.backbone}` (Exp {champion.experiment_num})")
            lines.append(f"- **Composite:** {champion.composite:.4f}")
            lines.append(f"- **Test primary:** {champion.test_primary:.4f}")
            lines.append(f"- **Val primary:** {champion.val_primary:.4f}")
            lines.append(f"- **Description:** {champion.description}")
            lines.append("")
            lines.append("### Champion per-fold test (primary metric)\n")
            for i, v in enumerate(champion.per_fold_test):
                lines.append(f"- F{i+1}: {v:.4f}")
            lines.append("")
        else:
            lines.append("_No champion yet._\n")

        lines.append("## Last experiment result\n")
        lines.append(f"- **Exp {record.experiment_num} — `{record.backbone}`:** {record.description}")
        lines.append(f"- **Composite:** {record.composite:.4f}")
        lines.append(f"- **Test / Val primary:** {record.test_primary:.4f} / {record.val_primary:.4f}")
        lines.append(f"- **Status:** `{record.status}`")
        if record.per_fold_test:
            lines.append("- **Per-fold test:** " + ", ".join(f"F{i+1}={v:.4f}" for i, v in enumerate(record.per_fold_test)))
        lines.append("")

        lines.append("## Next experiment (exact command)\n")
        lines.append("```bash")
        lines.append(next_command)
        lines.append("```\n")
        lines.append("### Rationale\n")
        lines.append(next_rationale or "_author the rationale (diagnosis + citation + hypothesis) before launching_")
        lines.append("")

        lines.append("## Experiment history summary\n")
        lines.append("| # | Backbone | Description | Composite | Status |")
        lines.append("|---|----------|-------------|-----------|--------|")
        for h in history[-200:]:
            lines.append(
                f"| {h.experiment_num} | {h.backbone} | {h.description[:60]} | "
                f"{h.composite:.4f} | {h.status} |"
            )
        lines.append("")

        lines.append("## Exhausted axes (do not re-try without new hypothesis)\n")
        if exhausted_axes:
            for a in exhausted_axes:
                lines.append(f"- {a}")
        else:
            lines.append("_none recorded yet_")
        lines.append("")

        return "\n".join(lines)

    def _render_summary_block(self, r: ExperimentRecord) -> str:
        per_fold = " ".join(f"F{i+1}={v:.4f}" for i, v in enumerate(r.per_fold_test))
        secondary = ", ".join(f"{k}={v}" for k, v in r.secondary_metrics.items() if not isinstance(v, dict))
        return (
            f"\n### Exp{r.experiment_num} ({r.backbone}): {r.description}\n"
            f"- **Timestamp:** {r.timestamp}\n"
            f"- **Config:** {json.dumps(r.config, default=str, sort_keys=True)}\n"
            f"- **Result:** Composite {r.composite:.4f} | "
            f"Test {r.test_primary:.4f} | Val {r.val_primary:.4f}\n"
            f"- **Per-fold test:** {per_fold}\n"
            f"- **Secondary metrics:** {secondary}\n"
            f"- **Status:** {r.status}\n"
            f"- **Seconds elapsed:** {r.seconds_elapsed:.1f}\n"
        )
