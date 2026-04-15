"""Autonomous optimizer loop driven by Claude API.

Iteratively brainstorms experiment ideas, applies code changes, evaluates
them via the baseline pipeline, and keeps improvements while reverting
regressions.
"""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import tempfile
import textwrap
from pathlib import Path

import anthropic

from .prompts import BRAINSTORM_PROMPT, CATEGORIES, MODIFY_PROMPT

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODIFIABLE_FILES = {
    "model_architecture": PROJECT_ROOT / "model" / "backbone.py",
    "head_design": PROJECT_ROOT / "model" / "backbone.py",
    "training_hyperparams": PROJECT_ROOT / "model" / "train.py",
    "regularization": PROJECT_ROOT / "model" / "train.py",
    "feature_engineering": PROJECT_ROOT / "data" / "features.py",
    "data_preprocessing": PROJECT_ROOT / "data" / "features.py",
    "ensemble": PROJECT_ROOT / "model" / "backbone.py",
}

STATE_PATH = PROJECT_ROOT / "optimizer_state.json"
BACKUP_DIR = PROJECT_ROOT / ".optimizer_backups"

# Files that should be backed up before every experiment
_BACKUP_TARGETS = [
    PROJECT_ROOT / "model" / "backbone.py",
    PROJECT_ROOT / "data" / "features.py",
    PROJECT_ROOT / "model" / "train.py",
]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_file(path: str | Path) -> str:
    """Read a text file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write_file(path: str | Path, content: str) -> None:
    """Write *content* to a text file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Syntax validation
# ---------------------------------------------------------------------------

def validate_syntax(code: str) -> tuple[bool, str]:
    """Write *code* to a temp file and check it compiles.

    Returns
    -------
    (ok, message) where *ok* is True on success and *message* contains
    the error detail on failure.
    """
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write(code)
        tmp.close()
        py_compile.compile(tmp.name, doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as exc:
        return False, str(exc)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(path: str | Path = STATE_PATH) -> dict:
    """Load optimizer state from JSON, or return a fresh default."""
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "iteration": 0,
        "current_sharpe": 0.0,
        "best_sharpe": 0.0,
        "experiments": [],
    }


def save_state(state: dict, path: str | Path = STATE_PATH) -> None:
    """Persist optimizer state as pretty-printed JSON."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


# ---------------------------------------------------------------------------
# Backup / restore helpers
# ---------------------------------------------------------------------------

def _backup_files() -> None:
    """Copy modifiable source files into the backup directory."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in _BACKUP_TARGETS:
        if src.exists():
            dst = BACKUP_DIR / src.relative_to(PROJECT_ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _restore_files() -> None:
    """Restore modifiable source files from the backup directory."""
    for src in _BACKUP_TARGETS:
        backup = BACKUP_DIR / src.relative_to(PROJECT_ROOT)
        if backup.exists():
            shutil.copy2(backup, src)


# ---------------------------------------------------------------------------
# Claude API helpers
# ---------------------------------------------------------------------------

def _call_claude(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int = 8192,
) -> str:
    """Send a single-turn message to Claude and return the text response."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _strip_markdown_fences(text: str) -> str:
    """Remove optional ```python / ``` fences wrapping the response."""
    text = text.strip()
    # Handle ```python ... ``` or ```json ... ```
    if text.startswith("```"):
        # Remove first line (```python or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


# ---------------------------------------------------------------------------
# Main optimizer loop
# ---------------------------------------------------------------------------

def run_optimizer(
    max_experiments: int = 12,
    model_name: str = "claude-sonnet-4-20250514",
) -> None:
    """Run the autonomous experiment loop.

    Parameters
    ----------
    max_experiments : int
        Maximum number of experiments to attempt.
    model_name : str
        Anthropic model identifier for both brainstorm and code-gen calls.
    """
    # Lazy import so baseline.py can be created in parallel
    from ..baseline import run_baseline  # type: ignore[import-untyped]

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    state = load_state()

    # ── Initial baseline evaluation ──────────────────────────────────
    if state["iteration"] == 0:
        print("=" * 60)
        print("Running initial baseline evaluation ...")
        print("=" * 60)
        result = run_baseline()
        initial_sharpe = result.get("avg_sharpe", 0.0)
        state["current_sharpe"] = initial_sharpe
        state["best_sharpe"] = initial_sharpe
        state["iteration"] = 1
        save_state(state)
        print(f"Baseline Sharpe: {initial_sharpe:.4f}")

    # ── Experiment loop ──────────────────────────────────────────────
    for exp_num in range(1, max_experiments + 1):
        print()
        print("=" * 60)
        print(f"Experiment {exp_num}/{max_experiments}  "
              f"(iteration {state['iteration']})")
        print("=" * 60)

        # 1. Back up modifiable files
        _backup_files()

        # 2. Read current source files
        model_code = read_file(PROJECT_ROOT / "model" / "backbone.py")
        feature_code = read_file(PROJECT_ROOT / "data" / "features.py")

        # 3. Format past experiments summary
        past_lines = []
        for exp in reversed(state["experiments"][-10:]):
            status = "KEPT" if exp.get("kept") else "REVERTED"
            past_lines.append(
                f"  [{status}] {exp.get('description', 'N/A')} "
                f"(sharpe={exp.get('sharpe', 'N/A')})"
            )
        past_experiments = "\n".join(past_lines) if past_lines else "(none yet)"

        # 4. Brainstorm ideas via Claude
        brainstorm_system = BRAINSTORM_PROMPT.format(
            categories=", ".join(CATEGORIES),
            current_sharpe=state["current_sharpe"],
            best_sharpe=state["best_sharpe"],
            past_experiments=past_experiments,
            model_code=model_code,
            feature_code=feature_code,
        )

        print("Brainstorming experiment ideas ...")
        brainstorm_response = _call_claude(
            client,
            system_prompt=brainstorm_system,
            user_message="Propose 3 experiment ideas now.",
            model=model_name,
        )

        # 5. Parse JSON response
        brainstorm_text = _strip_markdown_fences(brainstorm_response)
        try:
            ideas = json.loads(brainstorm_text)
        except json.JSONDecodeError:
            print(f"Failed to parse brainstorm JSON, skipping this round.")
            print(f"Response was:\n{brainstorm_text[:500]}")
            continue

        if not isinstance(ideas, list) or len(ideas) == 0:
            print("Brainstorm returned empty or invalid list, skipping.")
            continue

        # 6. Pick the first low/medium risk idea
        chosen = None
        for idea in ideas:
            risk = idea.get("risk", "high").lower()
            if risk in ("low", "medium"):
                chosen = idea
                break
        if chosen is None:
            # Fallback: pick the first idea regardless of risk
            chosen = ideas[0]

        category = chosen.get("category", "model_architecture")
        description = chosen.get("description", "No description")
        risk = chosen.get("risk", "unknown")

        print(f"Selected idea: [{category}] ({risk} risk)")
        print(f"  {description}")

        # 7. Map category to target file
        target_file = MODIFIABLE_FILES.get(
            category,
            PROJECT_ROOT / "model" / "backbone.py",
        )
        current_code = read_file(target_file)

        # 8. Generate modified code via Claude
        print(f"Generating code modification for {target_file.name} ...")
        modify_system = MODIFY_PROMPT.format(
            experiment_description=description,
            current_code=current_code,
        )

        modified_response = _call_claude(
            client,
            system_prompt=modify_system,
            user_message="Generate the modified file now.",
            model=model_name,
        )

        # 9. Strip markdown fences if present
        modified_code = _strip_markdown_fences(modified_response)

        # 10. Validate syntax
        ok, msg = validate_syntax(modified_code)
        if not ok:
            print(f"Syntax error in generated code: {msg}")
            print("Skipping this experiment.")
            _restore_files()
            state["experiments"].append({
                "id": exp_num,
                "category": category,
                "description": description,
                "risk": risk,
                "status": "syntax_error",
                "error": msg,
                "kept": False,
            })
            save_state(state)
            continue

        # 11. Apply the change
        print("Applying change and running evaluation ...")
        write_file(target_file, modified_code)

        # 12. Evaluate
        try:
            result = run_baseline()
            new_sharpe = result.get("avg_sharpe", 0.0)
        except Exception as exc:
            print(f"Evaluation failed: {exc}")
            _restore_files()
            state["experiments"].append({
                "id": exp_num,
                "category": category,
                "description": description,
                "risk": risk,
                "status": "eval_error",
                "error": str(exc),
                "kept": False,
            })
            save_state(state)
            continue

        print(f"New Sharpe: {new_sharpe:.4f}  "
              f"(current: {state['current_sharpe']:.4f}, "
              f"best: {state['best_sharpe']:.4f})")

        # 13. Keep or revert
        if new_sharpe > state["current_sharpe"]:
            print(">> Improvement! Keeping change.")
            state["current_sharpe"] = new_sharpe
            if new_sharpe > state["best_sharpe"]:
                state["best_sharpe"] = new_sharpe
            kept = True
        else:
            print(">> No improvement. Reverting.")
            _restore_files()
            kept = False

        # 14. Record experiment and save state
        state["experiments"].append({
            "id": exp_num,
            "category": category,
            "description": description,
            "risk": risk,
            "sharpe": new_sharpe,
            "kept": kept,
            "status": "completed",
        })
        state["iteration"] += 1
        save_state(state)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Optimizer loop complete.")
    print(f"  Total experiments: {len(state['experiments'])}")
    kept_count = sum(1 for e in state["experiments"] if e.get("kept"))
    print(f"  Kept:     {kept_count}")
    print(f"  Reverted: {len(state['experiments']) - kept_count}")
    print(f"  Best Sharpe: {state['best_sharpe']:.4f}")
    print("=" * 60)
