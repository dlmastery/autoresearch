"""Foundation-model backbone stubs.

These are wiring skeletons for users who want to plug in TimesFM, Chronos, MOMENT,
Moirai, TiRex, Sundial, Time-MoE, etc. Each stub documents:
  - the paper to consult
  - the HuggingFace / official checkpoint
  - the expected memory class (per CLAUDE.md GPU Memory Constraint table)

The stubs raise `NotImplementedError` on `build()` — users replace the body with
their actual integration. This is deliberate: we do not ship vendored copies of
foundation-model code, and we force the user to make a conscious decision about
fine-tune mode (zero-shot / head-only / LoRA / full) per the GPU Memory Constraint
pre-flight-check rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backbone, PredictionBundle
from .registry import register_backbone


class _FoundationStub(Backbone):
    """Base stub — forces user to implement before use."""

    task_types = {"time_series_forecasting"}
    paper_citation = "See subclass paper_citation attribute"

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        raise NotImplementedError(
            f"{self.name}: foundation-model backbone is a stub.\n"
            f"Paper: {self.paper_citation}\n"
            f"Before use:\n"
            f"  1. Decide fine-tune mode (zero-shot / head-only / LoRA / full) per GPU memory budget.\n"
            f"  2. Install the official package (e.g. `pip install timesfm`, `pip install chronos-forecasting`).\n"
            f"  3. Replace this stub with the integration in a subclass.\n"
            f"  4. Document the memory pre-flight check in the pre-run reasoning annotation."
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:  # pragma: no cover
        raise NotImplementedError

    def save(self, path: str | Path) -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "Backbone":  # pragma: no cover
        raise NotImplementedError


@register_backbone("timesfm")
class TimesFMBackbone(_FoundationStub):
    name = "timesfm"
    paper_citation = (
        "Das, Kong, Sen & Zhou 2024 ICML 'A Decoder-Only Foundation Model for "
        "Time-Series Forecasting' (arXiv:2310.10688). "
        "Size class: 200M–500M params → PEFT or head-only FT."
    )


@register_backbone("chronos_bolt")
class ChronosBoltBackbone(_FoundationStub):
    name = "chronos_bolt"
    paper_citation = (
        "Ansari, Stella, Turkmen, Zhang, Mercado, Shen, Shchur, Rangapuram, Pineda Arango, "
        "Kapoor, Zschiegner, Maddix, Mahoney, Torkkola, Wilson, Bohlke-Schneider & Wang "
        "2024 TMLR 'Chronos: Learning the Language of Time Series' (arXiv:2403.07815)."
    )


@register_backbone("moment")
class MomentBackbone(_FoundationStub):
    name = "moment"
    paper_citation = (
        "Goswami, Szafer, Choudhry, Cai, Li & Dubrawski 2024 ICML "
        "'MOMENT: A Family of Open Time-series Foundation Models' (arXiv:2402.03885)."
    )


@register_backbone("moirai")
class MoiraiBackbone(_FoundationStub):
    name = "moirai"
    paper_citation = (
        "Woo, Liu, Kumar, Xiong, Savarese & Sahoo 2024 ICML 'Unified Training of Universal "
        "Time Series Forecasting Transformers' (arXiv:2402.02592); "
        "Moirai-MoE (arXiv:2410.10469); Moirai 2.0 (arXiv:2511.11698)."
    )


@register_backbone("tirex")
class TiRexBackbone(_FoundationStub):
    name = "tirex"
    paper_citation = (
        "Auer, Pöppel, Pflüger, Brandstetter & Hochreiter 2025 "
        "'TiRex: Zero-Shot Forecasting with Recurrent xLSTM Backbones' (NXAI/JKU)."
    )


@register_backbone("sundial")
class SundialBackbone(_FoundationStub):
    name = "sundial"
    paper_citation = (
        "Liu, Zhang, Wu & Long 2025 'Sundial: A Family of Highly Capable Time Series "
        "Foundation Models' (arXiv:2502.00816)."
    )


@register_backbone("time_moe")
class TimeMoEBackbone(_FoundationStub):
    name = "time_moe"
    paper_citation = (
        "Shi, Wang, Yang, Wang, Yang, Wang, Li, Li, Sun, Gao & Li 2024/2025 ICLR "
        "'Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts' "
        "(arXiv:2409.16040)."
    )
