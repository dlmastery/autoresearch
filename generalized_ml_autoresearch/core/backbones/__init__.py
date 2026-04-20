"""Backbone subsystem — pluggable model implementations.

Every backbone subclasses `Backbone` from `base.py` and registers itself via
`@register_backbone` in `registry.py`. The runner uses `create_model(name, config)`
to instantiate.

The GBM Tier-3 entries xgboost, lightgbm, catboost are THREE separate backbones
(per CLAUDE.md rule). Never merge them.
"""

from .base import Backbone, PredictionBundle
from .registry import (
    BACKBONE_REGISTRY,
    register_backbone,
    create_model,
    list_backbones,
)

# Import modules so their registrations execute.
from . import mlp  # noqa: F401
from . import lstm  # noqa: F401
from . import gbm  # noqa: F401
from . import tabular_transformer  # noqa: F401
from . import foundation_models  # noqa: F401

__all__ = [
    "Backbone",
    "PredictionBundle",
    "BACKBONE_REGISTRY",
    "register_backbone",
    "create_model",
    "list_backbones",
]
