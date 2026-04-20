"""Model backbones for FX return prediction.

Provides multiple architectures for ablation study spanning:
  - Classical ML: XGBoost, LightGBM, CatBoost
  - Recurrent: Bidirectional LSTM
  - Feedforward: MLP baseline
  - Transformers: PatchTST
  - MLP-Mixer: PatchTSMixer
  - Foundation: LFM2.5-350M (Liquid AI)

All neural models share the same interface:
  forward(x: Tensor[batch, seq_len, n_features]) -> {"ret_1d": Tensor, "ret_5d": Tensor}

Gradient boosting wrappers (XGBoost, LightGBM, CatBoost) use a separate
sklearn-compatible interface — see GBMWrapper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PAIRS: int = 6
HORIZONS: list[str] = ["ret_1d", "ret_5d"]

BACKBONE_REGISTRY: dict[str, str] = {
    "mlp": "Simple MLP (no pretrained weights)",
    "lstm": "Bidirectional LSTM baseline",
    "lfm2-350m": "LiquidAI/LFM2.5-350M-Base (frozen)",
    "patchtst": "PatchTST — Patch Time Series Transformer (Nie et al., ICLR 2023)",
    "patchtsmixer": "PatchTSMixer — MLP-Mixer for time series (Google, NeurIPS 2023)",
    "xgboost": "XGBoost gradient boosting (Chen & Guestrin, 2016)",
    "lightgbm": "LightGBM gradient boosting (Ke et al., NeurIPS 2017)",
    "catboost": "CatBoost gradient boosting (Prokhorenkova et al., NeurIPS 2018)",
}

DEFAULT_BACKBONE = "lfm2-350m"

# Per-backbone default sequence lengths (business days per input window).
# LFM2.5 benefits from long context (foundation model pre-trained on long sequences).
# All other backbones use 10 business days (~2 weeks), the industry standard
# for short-term FX prediction with 1d/5d horizons.
_LFM_SEQ_LEN: int = 60
_DEFAULT_SEQ_LEN: int = 10

BACKBONE_SEQ_LEN: dict[str, int] = {
    "lfm2-350m": _LFM_SEQ_LEN,
}


def get_seq_len(backbone: str) -> int:
    """Return the default sequence length for a backbone."""
    return BACKBONE_SEQ_LEN.get(backbone, _DEFAULT_SEQ_LEN)


# Models that use the GBM wrapper instead of nn.Module
GBM_BACKBONES = {"xgboost", "lightgbm", "catboost"}


def is_gbm(backbone: str) -> bool:
    """Check if a backbone uses gradient boosting (non-PyTorch)."""
    return backbone in GBM_BACKBONES


# ---------------------------------------------------------------------------
# Shared prediction heads (for neural models)
# ---------------------------------------------------------------------------
def _fit_patch_length(seq_len: int, patch_length: int) -> int:
    """Return the largest divisor of seq_len <= patch_length that yields >= 2 patches."""
    candidates = [d for d in range(2, patch_length + 1)
                  if seq_len % d == 0 and seq_len // d >= 2]
    return max(candidates) if candidates else max(1, seq_len // 2)


def _make_heads(hidden_size: int, dropout: float = 0.1, het_loss: bool = True, head_hidden: int = 256) -> nn.ModuleDict:
    """Prediction heads with optional aleatoric uncertainty output.

    When het_loss=True (default): outputs 2 * N_PAIRS (mean + log_var)
    When het_loss=False: outputs N_PAIRS (mean only, uncertainty via MC Dropout)
    """
    out_dim = 2 * N_PAIRS if het_loss else N_PAIRS
    return nn.ModuleDict(
        {
            horizon: nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_dim),
            )
            for horizon in HORIZONS
        }
    )


def _split_mean_logvar(
    raw: torch.Tensor,
    clamp_min: float = -6.0,
    clamp_max: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split head output into mean and clamped log-variance.

    Clamping prevents two failure modes (Stirn et al., 2023):
      - log_var too low (<-6 → var<0.0025): overconfident, wrong predictions
      - log_var too high (>2 → var>7.4): lazy variance, model skips learning mean
    """
    mean, log_var = raw[:, :N_PAIRS], raw[:, N_PAIRS:]
    log_var = torch.clamp(log_var, min=clamp_min, max=clamp_max)
    return mean, log_var


# ---------------------------------------------------------------------------
# MLP Backbone
# ---------------------------------------------------------------------------
def _forward_heads(heads: nn.ModuleDict, hidden: torch.Tensor, het_loss: bool) -> dict[str, torch.Tensor]:
    """Shared forward for all model heads. Handles both het and plain modes."""
    out = {}
    for name, head in heads.items():
        raw = head(hidden)
        if het_loss:
            mean, log_var = _split_mean_logvar(raw)
            out[name] = mean
            out[f"{name}_log_var"] = log_var
        else:
            out[name] = raw
    return out


class CurrencyMLP(nn.Module):
    """Residual MLP: learns correction to linear projection (He et al. 2016).

    For low-SNR financial data, the signal is a small perturbation on a
    linear baseline. The skip connection lets the nonlinear layers focus
    on learning the residual correction rather than the full mapping.
    """
    def __init__(self, n_input_features: int, seq_len: int = 10, hidden_size: int = 128,
                 head_dropout: float = 0.1, het_loss: bool = True):
        super().__init__()
        self.het_loss = het_loss
        head_hidden = min(hidden_size, 64)
        input_dim = n_input_features * seq_len

        # Linear shortcut: input -> hidden_size (the "baseline" prediction)
        self.shortcut = nn.Linear(input_dim, hidden_size)

        # Nonlinear residual branch: learns correction to the linear baseline
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss, head_hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.size(0), -1)
        hidden = self.shortcut(flat) + self.residual(flat)
        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# LSTM Backbone
# ---------------------------------------------------------------------------
class CurrencyLSTM(nn.Module):
    def __init__(self, n_input_features: int, hidden_size: int = 128, num_layers: int = 2, head_dropout: float = 0.1, het_loss: bool = True):
        super().__init__()
        self.het_loss = het_loss
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.heads = _make_heads(hidden_size * 2, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        hidden = lstm_out[:, -1, :]
        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# LFM2.5 Backbone (350M)
# ---------------------------------------------------------------------------
class CurrencyLFM(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        hidden_size: int = 1024,
        freeze_backbone: bool = True,
        model_id: str = "LiquidAI/LFM2.5-350M-Base",
        head_dropout: float = 0.1,
        het_loss: bool = True,
    ) -> None:
        super().__init__()
        self.het_loss = het_loss
        from transformers import Lfm2Model

        self.projection = nn.Linear(n_input_features, hidden_size)
        self.backbone = Lfm2Model.from_pretrained(model_id, torch_dtype=torch.float32)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.projection(x)
        outputs = self.backbone(inputs_embeds=projected)
        hidden = outputs.last_hidden_state[:, -1, :]
        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# PatchTST Backbone (Nie et al., ICLR 2023)
# ---------------------------------------------------------------------------
class CurrencyPatchTST(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        seq_len: int = 10,
        patch_length: int = 5,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        head_dropout: float = 0.1,
        het_loss: bool = True,
    ):
        super().__init__()
        self.het_loss = het_loss
        from transformers import PatchTSTConfig, PatchTSTModel

        patch_length = _fit_patch_length(seq_len, patch_length)

        self._hidden_size = hidden_size
        config = PatchTSTConfig(
            num_input_channels=n_input_features,
            context_length=seq_len,
            patch_length=patch_length,
            stride=patch_length,
            d_model=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            prediction_length=1,
            head_dropout=head_dropout,
            dropout=0.1,
        )
        self.backbone = PatchTSTModel(config)
        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(past_values=x)
        hidden = outputs.last_hidden_state
        if hidden.dim() == 4:
            hidden = hidden.mean(dim=(1, 2))
        elif hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# PatchTSMixer Backbone (Google, NeurIPS 2023)
# ---------------------------------------------------------------------------
class CurrencyPatchTSMixer(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        seq_len: int = 10,
        patch_length: int = 5,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        head_dropout: float = 0.1,
        het_loss: bool = True,
    ):
        super().__init__()
        self.het_loss = het_loss
        from transformers import PatchTSMixerConfig, PatchTSMixerModel

        patch_length = _fit_patch_length(seq_len, patch_length)

        config = PatchTSMixerConfig(
            num_input_channels=n_input_features,
            context_length=seq_len,
            patch_length=patch_length,
            patch_stride=patch_length,
            d_model=hidden_size,
            num_layers=num_hidden_layers,
            prediction_length=1,
            dropout=0.1,
        )
        self.backbone = PatchTSMixerModel(config)
        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(past_values=x)
        hidden = outputs.last_hidden_state
        if hidden.dim() == 4:
            hidden = hidden.mean(dim=(1, 2))
        elif hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# Gradient Boosting Wrapper (XGBoost / LightGBM / CatBoost)
# ---------------------------------------------------------------------------
class GBMWrapper:
    """Unified wrapper for gradient boosting models.

    These are NOT nn.Module — they use sklearn-compatible fit/predict.
    The ablation runner handles them separately from neural models.
    """

    def __init__(self, gbm_type: str, n_targets: int = 2):
        self.gbm_type = gbm_type
        self.n_targets = n_targets
        self.models: list = []  # one per target column
        self._fitted = False

    def _create_estimator(self):
        if self.gbm_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7,
                min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                tree_method="hist", device="cuda",
                verbosity=0,
            )
        elif self.gbm_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7,
                num_leaves=31, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0,
                device="gpu", verbose=-1,
            )
        elif self.gbm_type == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=500, depth=4, learning_rate=0.03,
                l2_leaf_reg=3.0, subsample=0.7,
                bootstrap_type="Bernoulli",
                task_type="GPU", verbose=0,
            )
        else:
            raise ValueError(f"Unknown GBM type: {self.gbm_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit one estimator per target column.

        X: (n_samples, seq_len * n_features) — flattened sequences
        y: (n_samples, n_targets)
        """
        self.models = []
        for col in range(y.shape[1]):
            est = self._create_estimator()
            est.fit(X, y[:, col])
            self.models.append(est)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all targets. Returns (n_samples, n_targets)."""
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds


# ---------------------------------------------------------------------------
# MC Dropout uncertainty estimation (Gal & Ghahramani, 2016)
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_mc_samples: int = 20,
    horizon: str = "ret_1d",
    pair_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """Run MC Dropout to decompose prediction uncertainty.

    Performs *n_mc_samples* stochastic forward passes with dropout enabled,
    then decomposes total uncertainty into aleatoric and epistemic components
    per Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian
    Deep Learning for Computer Vision?"

    Returns per-sample tensors on CPU:
      - mean: expected prediction (average over MC samples)
      - aleatoric: data uncertainty (mean of predicted variances)
      - epistemic: model uncertainty (variance of predicted means)
      - total_uncertainty: aleatoric + epistemic
      - confidence: sigmoid(-log(total_uncertainty)), in [0, 1]
      - pred_std: sqrt(total_uncertainty) — useful for bands/intervals
      - lower_1s, upper_1s: mean +/- 1 sigma band
      - lower_2s, upper_2s: mean +/- 2 sigma band (≈95% interval)
    """
    was_training = model.training
    het_mode = hasattr(model, 'het_loss') and model.het_loss
    model.train()  # enable dropout for stochastic passes

    mc_means = []
    mc_log_vars = []

    with torch.no_grad():
        for _ in range(n_mc_samples):
            out = model(x)
            mc_means.append(out[horizon][:, pair_idx])
            if het_mode:
                mc_log_vars.append(out[f"{horizon}_log_var"][:, pair_idx])

    mc_means = torch.stack(mc_means)      # (T, B)

    # Mean prediction across MC samples
    mean = mc_means.mean(dim=0)

    # Epistemic: variance of predicted means (model disagreement)
    epistemic = mc_means.var(dim=0)

    # Aleatoric: from het head or estimated from MC residual variance
    if het_mode and mc_log_vars:
        mc_log_vars = torch.stack(mc_log_vars)
        aleatoric = torch.exp(mc_log_vars).mean(dim=0)
    else:
        # Plain mode: estimate aleatoric as total MC variance minus epistemic
        # (rough approximation — epistemic dominates in MC Dropout)
        aleatoric = mc_means.var(dim=0) * 0.5  # split total variance

    # Total
    total = aleatoric + epistemic
    pred_std = torch.sqrt(total)

    # Confidence: high when uncertainty is low
    # Scale so typical FX uncertainties map to ~0.3-0.9 range
    confidence = torch.sigmoid(-torch.log(total + 1e-8))

    if not was_training:
        model.eval()

    return {
        "mean": mean.cpu(),
        "aleatoric": aleatoric.cpu(),
        "epistemic": epistemic.cpu(),
        "total_uncertainty": total.cpu(),
        "confidence": confidence.cpu(),
        "pred_std": pred_std.cpu(),
        "lower_1s": (mean - pred_std).cpu(),
        "upper_1s": (mean + pred_std).cpu(),
        "lower_2s": (mean - 2 * pred_std).cpu(),
        "upper_2s": (mean + 2 * pred_std).cpu(),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_model(
    backbone: str,
    n_input_features: int,
    seq_len: int | None = None,
    freeze_backbone: bool = True,
    head_dropout: float = 0.1,
    het_loss: bool = True,
    hidden_size: int | None = None,
) -> nn.Module | GBMWrapper:
    """Create a model by backbone name.

    Args:
        het_loss: If True, heads output mean + log_var for heteroscedastic loss.
                  If False, heads output mean only; uncertainty via MC Dropout.

    Returns nn.Module for neural backbones, GBMWrapper for tree-based.
    """
    if seq_len is None:
        seq_len = get_seq_len(backbone)
    if backbone == "mlp":
        kwargs = dict(seq_len=seq_len, head_dropout=head_dropout, het_loss=het_loss)
        if hidden_size is not None:
            kwargs["hidden_size"] = hidden_size
        return CurrencyMLP(n_input_features, **kwargs)
    elif backbone == "lstm":
        kwargs = dict(head_dropout=head_dropout, het_loss=het_loss)
        if hidden_size is not None:
            kwargs["hidden_size"] = hidden_size
        return CurrencyLSTM(n_input_features, **kwargs)
    elif backbone == "lfm2-350m":
        return CurrencyLFM(
            n_input_features, freeze_backbone=freeze_backbone,
            model_id="LiquidAI/LFM2.5-350M-Base",
            head_dropout=head_dropout, het_loss=het_loss,
        )
    elif backbone == "patchtst":
        return CurrencyPatchTST(n_input_features, seq_len=seq_len, head_dropout=head_dropout, het_loss=het_loss)
    elif backbone == "patchtsmixer":
        return CurrencyPatchTSMixer(n_input_features, seq_len=seq_len, head_dropout=head_dropout, het_loss=het_loss)
    elif backbone in GBM_BACKBONES:
        return GBMWrapper(backbone, n_targets=2)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )
