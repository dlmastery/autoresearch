"""Composite metric calculator — enforces the keep/revert gate.

Default formula (from CLAUDE.md Experiment Design):
    composite = min(test_primary, val_primary) - penalty_weight * n_below_threshold_folds

For regression where lower is better (RMSE, MAE), we internally negate so that
the same `>` comparison means "better". The sign-handling is set via
`higher_is_better` in the constructor.

User can supply:
  - a custom formula string (evaluated in a restricted sandbox)
  - OR a direct Python callable

Goodhart protection: the formula is frozen at project-setup time. Changing it
mid-project requires a RULE_CHANGE entry in the checkpoint (enforced outside
this module — the runner checks the config hash against the original).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CompositeCalculator:
    primary_metric_name: str
    higher_is_better: bool = True
    penalty_weight: float = 0.1
    below_threshold: float = 0.0  # a fold is "bad" if primary < threshold (higher-is-better)
    formula: str | None = None  # optional custom formula string
    custom_fn: Callable[..., float] | None = None  # optional callable

    def _fold_primary(self, fold_metric: float) -> float:
        """Return a 'higher is better' view of the fold's primary metric."""
        return fold_metric if self.higher_is_better else -fold_metric

    def compute(
        self,
        val_primary: float,
        test_primary: float,
        per_fold_test: list[float],
    ) -> float:
        """Canonical composite = min(val, test) - penalty * n_below_threshold."""
        if self.custom_fn is not None:
            return float(
                self.custom_fn(val_primary, test_primary, per_fold_test, self)
            )
        if self.formula is not None:
            return float(self._eval_formula(val_primary, test_primary, per_fold_test))

        v = self._fold_primary(val_primary)
        t = self._fold_primary(test_primary)
        fold_vals = [self._fold_primary(x) for x in per_fold_test]
        n_below = sum(1 for x in fold_vals if x < self.below_threshold)
        return float(min(v, t) - self.penalty_weight * n_below)

    def _eval_formula(self, val_primary, test_primary, per_fold_test) -> float:
        """Tiny sandbox evaluator for simple formulas.

        Available names: val, test, per_fold_test, min, max, sum, n_folds,
        n_below (count of fold values below `below_threshold`), penalty (self.penalty_weight).
        """
        v = self._fold_primary(val_primary)
        t = self._fold_primary(test_primary)
        folds = [self._fold_primary(x) for x in per_fold_test]
        n_below = sum(1 for x in folds if x < self.below_threshold)
        safe_globals = {
            "__builtins__": {},
            "min": min, "max": max, "sum": sum, "abs": abs, "len": len,
            "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        }
        safe_locals = {
            "val": v, "test": t, "per_fold_test": folds, "n_folds": len(folds),
            "n_below": n_below, "penalty": self.penalty_weight,
            "threshold": self.below_threshold,
        }
        return eval(self.formula, safe_globals, safe_locals)  # noqa: S307 — sandboxed

    def fingerprint(self) -> str:
        """Stable hash of the formula config — used by runner to detect Goodhart rewrites."""
        raw = f"{self.primary_metric_name}|{self.higher_is_better}|{self.penalty_weight}|{self.below_threshold}|{self.formula or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
