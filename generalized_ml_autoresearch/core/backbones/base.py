"""Abstract Backbone class — the contract every backbone implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


@dataclass
class PredictionBundle:
    mean: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    confidence: np.ndarray
    probabilities: np.ndarray | None = None  # for classification


class Backbone(ABC):
    """Abstract interface for all backbones — neural and classical alike.

    Subclasses must set class-level attributes:
      - `name` — unique registry key
      - `task_types` — set of supported task types
        ({"regression", "binary_classification", "multiclass_classification",
          "time_series_forecasting", "ranking", "survival", "multi_label"})

    And implement:
      - `build(config, input_shape, n_outputs)` — construct the model
      - `fit(X_train, y_train, X_val, y_val)` — train; returns training history dict
      - `predict_with_uncertainty(X, n_samples)` — returns PredictionBundle
      - `save(path)` / `load(path)` — portable serialization (classmethod on load)
      - `gpu_memory_estimate_mb(batch_size)` — pre-flight memory estimate
    """

    name: ClassVar[str] = "abstract"
    task_types: ClassVar[set[str]] = set()

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = []
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        self._model: Any = None

    @abstractmethod
    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        ...

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        ...

    @abstractmethod
    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Backbone":
        ...

    def gpu_memory_estimate_mb(self, batch_size: int) -> float:
        """Estimate peak VRAM. Classical backbones can return 0.0.

        Subclasses should override if they're neural to return a measured estimate.
        """
        return 0.0

    def summary(self) -> dict[str, Any]:
        """Small summary dict for logging."""
        return {
            "name": self.name,
            "task_types": sorted(self.task_types),
            "n_features": len(self.feature_columns),
            "n_targets": len(self.target_columns),
        }
