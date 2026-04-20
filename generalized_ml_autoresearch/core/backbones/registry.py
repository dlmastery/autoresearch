"""Backbone registry + factory."""

from __future__ import annotations

from typing import Type

from .base import Backbone


BACKBONE_REGISTRY: dict[str, Type[Backbone]] = {}


def register_backbone(name: str | None = None):
    """Class decorator — registers a Backbone subclass.

    If `name` is not given, uses the class's `name` attribute.
    """
    def _wrap(cls: Type[Backbone]):
        key = name or cls.name
        if not key or key == "abstract":
            raise ValueError(f"Backbone {cls.__name__} must set a non-empty .name attribute")
        if key in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone {key!r} is already registered")
        BACKBONE_REGISTRY[key] = cls
        return cls
    return _wrap


def create_model(name: str, config: dict | None = None) -> Backbone:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone {name!r}. Registered: {sorted(BACKBONE_REGISTRY)}"
        )
    cls = BACKBONE_REGISTRY[name]
    inst = cls()
    if config is not None:
        inst.config = dict(config)
    return inst


def list_backbones() -> list[str]:
    return sorted(BACKBONE_REGISTRY)
