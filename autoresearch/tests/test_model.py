"""Tests for model.backbone -- CurrencyLFM wrapper around LFM2.5-350M."""

import pytest
import torch

from autoresearch.model.backbone import CurrencyLFM, N_PAIRS, HORIZONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load CurrencyLFM once for the whole test module (slow on CPU)."""
    return CurrencyLFM(n_input_features=10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCurrencyLFM:

    def test_model_creates(self, model):
        """CurrencyLFM(n_input_features=10) creates without error."""
        assert isinstance(model, CurrencyLFM)

    def test_model_forward_shape(self, model):
        """Input (2, 30, 10) -> output dicts with correct shapes."""
        x = torch.randn(2, 30, 10)
        with torch.no_grad():
            out = model(x)

        assert isinstance(out, dict)
        for horizon in HORIZONS:
            assert horizon in out, f"Missing key: {horizon}"
            assert out[horizon].shape == (2, N_PAIRS), (
                f"{horizon}: expected (2, {N_PAIRS}), got {out[horizon].shape}"
            )
            # Uncertainty outputs
            log_var_key = f"{horizon}_log_var"
            assert log_var_key in out, f"Missing key: {log_var_key}"
            assert out[log_var_key].shape == (2, N_PAIRS), (
                f"{log_var_key}: expected (2, {N_PAIRS}), got {out[log_var_key].shape}"
            )

    def test_backbone_frozen(self, model):
        """All backbone params have requires_grad=False."""
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param {name} has requires_grad=True"
            )

    def test_trainable_params_exist(self, model):
        """Projection + heads params have requires_grad=True."""
        # Projection layer
        for name, param in model.projection.named_parameters():
            assert param.requires_grad, (
                f"Projection param {name} should be trainable"
            )

        # Head layers
        for head_name, head in model.heads.items():
            for name, param in head.named_parameters():
                assert param.requires_grad, (
                    f"Head {head_name} param {name} should be trainable"
                )
