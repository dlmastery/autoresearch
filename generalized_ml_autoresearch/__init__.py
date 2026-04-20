"""Generalized ML AutoResearch framework.

Domain-agnostic successor to the FX-specific AutoResearch loop. Preserves every
rule from the source CLAUDE.md (see `templates/SECTION_MAPPING.md`) while making
the system work for any supervised ML task — regression, classification,
time-series forecasting, ranking, survival, multi-label.

Entry points:
- `generalized_ml_autoresearch.core.runner` — one experiment per invocation.
- `generalized_ml_autoresearch.skills.ml-autoresearch-setup` — slash command
  `/ml-autoresearch-setup` that walks users through the 12-step setup wizard.
- `generalized_ml_autoresearch.dashboard.dashboard.html` — read-only dashboard.
"""

__version__ = "0.1.0"
