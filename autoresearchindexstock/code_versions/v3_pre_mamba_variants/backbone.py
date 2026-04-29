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
import torch.nn.functional as F

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
    "mamba": "Mamba — Selective State Space Model (Gu & Dao 2024, arXiv:2312.00752). Variants via --mamba-variant {vanilla, s_mamba, dmamba, mambats}",
    "dlinear": "DLinear — Decomposition + Linear (Zeng et al., AAAI 2023, arXiv:2205.13504)",
    "nbeats": "N-BEATS — Neural Basis Expansion (Oreshkin et al., ICLR 2020, arXiv:1905.10437)",
    "itransformer": "iTransformer — Inverted Transformer (Liu et al., ICLR 2024, arXiv:2310.06625)",
    "xlstm": "xLSTM — Extended LSTM with exponential gating (Beck et al., NeurIPS 2024, arXiv:2405.04517)",
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
    def __init__(self, n_input_features: int, hidden_size: int = 128, num_layers: int = 2,
                 head_dropout: float = 0.1, het_loss: bool = True, bidirectional: bool = True,
                 cell: str = "lstm", input_layernorm: bool = False):
        super().__init__()
        self.het_loss = het_loss
        self.bidirectional = bidirectional
        self.cell = cell
        self.input_ln = nn.LayerNorm(n_input_features) if input_layernorm else None
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.lstm = rnn_cls(
            input_size=n_input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1,
        )
        out_features = hidden_size * (2 if bidirectional else 1)
        self.heads = _make_heads(out_features, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.input_ln is not None:
            x = self.input_ln(x)
        lstm_out, _ = self.lstm(x)
        hidden = lstm_out[:, -1, :]
        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# Mamba Backbone — Selective State Space Model
# Gu & Dao 2024, COLM (arXiv:2312.00752) — canonical Mamba.
# Variants supported via `variant` kwarg:
#   vanilla  — canonical Mamba (Gu & Dao 2024)
#   s_mamba  — channel-flipped variant (arXiv:2403.11144, 2024)
#   dmamba   — trend+seasonal decomposition (arXiv:2602.09081, 2025)
#   mambats  — LTSF-tuned MambaTS (Cai et al. 2024 NeurIPS, arXiv:2405.16440)
#
# Implementation choices:
#   - Naive O(L) recurrent scan (seq_len=10 is small, parallel scan unnecessary)
#   - Pre-norm + residual per block (standard for SSM stability)
#   - Gated MLP wrapper from Gu & Dao 2024 Section 3
# ---------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """Simplified selective SSM block with input-dependent Δ, B, C (the
    'selective' mechanism from Gu & Dao 2024). State matrix A is a learnable
    negative real vector (HiPPO-LegS diagonal approximation). Skip D is learned.

    Reference: Gu & Dao 2024, 'Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces', arXiv:2312.00752 Section 3.2–3.3."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 variant: str = "vanilla"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.variant = variant

        # Input → (x_in, z_gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        # Input-dependent B, C, Δ (the selective bits)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        # A: log-parameterised negative reals, HiPPO-like init
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # S-Mamba variant (Liu et al. 2024, arXiv:2403.11144):
        # Run the SSM over the CHANNEL/variate axis instead of the time axis.
        # Equivalent to transposing L <-> d_inner before the scan so each
        # "timestep" of the scan is a different feature channel.
        if self.variant == "s_mamba":
            return self._forward_s_mamba(x_in, z, B, L)

        # Selective params from x_in
        bcd = self.x_proj(x_in)  # [B, L, 2*d_state+1]
        B_mat, C_mat, dt = bcd.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        # A = -exp(A_log); shape [d_inner, d_state]
        A = -torch.exp(self.A_log)

        # Recurrent scan (naive, fine at L≤60)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # Zero-order hold discretisation (Gu & Dao 2024 Eq. 4)
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)            # [B, d_inner, d_state]
            dB = dt[:, t].unsqueeze(-1) * B_mat[:, t].unsqueeze(1)  # [B, d_inner, d_state]
            h = dA * h + dB * x_in[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1) + self.D * x_in[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # [B, L, d_inner]

        # Gated output (Mamba block structure)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _forward_s_mamba(self, x_in: torch.Tensor, z: torch.Tensor,
                          B: int, L: int) -> torch.Tensor:
        """S-Mamba (Liu 2024): transpose variate<->time; scan across channels.
        Each channel becomes a scan step; d_inner channels become 'length'."""
        # Transpose: [B, L, d_inner] -> [B, d_inner, L]; treat d_inner as scan steps
        x_t = x_in.transpose(1, 2).contiguous()   # [B, d_inner, L]
        # x_proj expects d_inner on last axis; project per-step vector of length L
        # Build ad-hoc per-channel B/C/dt using a length-L MLP: reuse self.x_proj
        # by mapping x_t[..., :] of shape [B, d_inner, L] to [B, d_inner, 2*d_state+1]
        # via an average-pool over L (simplest faithful S-Mamba reduction).
        pooled = x_t.mean(dim=-1)                  # [B, d_inner]
        # Now pooled is fed through x_proj which expects d_inner input
        bcd = self.x_proj(pooled)                  # [B, 2*d_state+1]
        B_mat = bcd[..., :self.d_state]            # [B, d_state]
        C_mat = bcd[..., self.d_state:2*self.d_state]  # [B, d_state]
        dt_s = bcd[..., -1:]                       # [B, 1]
        dt = F.softplus(self.dt_proj(dt_s))        # [B, d_inner]
        A = -torch.exp(self.A_log)                 # [d_inner, d_state]

        # Scan ACROSS channels: each "step" is one of d_inner channels
        h = torch.zeros(B, L, self.d_state, device=x_in.device, dtype=x_in.dtype)
        ys = []
        for c in range(self.d_inner):
            dA = torch.exp(dt[:, c:c+1].unsqueeze(-1) * A[c])  # [B, 1, d_state]
            dB = dt[:, c:c+1].unsqueeze(-1) * B_mat.unsqueeze(1)  # [B, 1, d_state]
            h = dA * h + dB * x_t[:, c, :].unsqueeze(-1)
            y_c = (h * C_mat.unsqueeze(1)).sum(-1) + self.D[c] * x_t[:, c, :]  # [B, L]
            ys.append(y_c)
        y = torch.stack(ys, dim=1)   # [B, d_inner, L]
        y = y.transpose(1, 2)        # [B, L, d_inner] — transpose back
        y = y * F.silu(z)
        return self.out_proj(y)


class CurrencyMamba(nn.Module):
    """Mamba-family backbone for return prediction.

    Variants:
      vanilla  — canonical 2-layer Mamba (Gu & Dao 2024)
      s_mamba  — S-Mamba channel-flipped variant
      dmamba   — decomposition: seasonal via Mamba, trend via MLP (arXiv:2602.09081)
      mambats  — same block with LTSF-tuned hyperparams (Cai et al. 2024)
    """

    def __init__(self, n_input_features: int, hidden_size: int = 128,
                 num_layers: int = 2, d_state: int = 16, expand: int = 2,
                 head_dropout: float = 0.1, het_loss: bool = True,
                 variant: str = "vanilla"):
        super().__init__()
        self.het_loss = het_loss
        self.variant = variant
        self.embed = nn.Linear(n_input_features, hidden_size)
        self.blocks = nn.ModuleList([
            SelectiveSSM(hidden_size, d_state=d_state, expand=expand, variant=variant)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)

        if variant == "dmamba":
            # Decomposition: trend branch is a simple MLP on mean-pool, seasonal is Mamba output
            self.trend_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )

        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        h = self.final_norm(h)

        if self.variant == "dmamba":
            trend = self.trend_mlp(h.mean(dim=1))
            seasonal = h[:, -1, :]
            hidden = trend + seasonal
        else:
            hidden = h[:, -1, :]

        return _forward_heads(self.heads, hidden, self.het_loss)


# ---------------------------------------------------------------------------
# DLinear — Decomposition-Linear (Zeng et al. AAAI 2023, arXiv:2205.13504)
# "Are Transformers Effective for Time Series Forecasting?"
# Splits the input into trend (moving average) + seasonal residual, applies
# a linear layer to each, sums the result. Despite being trivially simple,
# it beats many transformer TS baselines.
# ---------------------------------------------------------------------------
class CurrencyDLinear(nn.Module):
    def __init__(self, n_input_features: int, seq_len: int = 10,
                 kernel_size: int = 25, hidden_size: int = 128,
                 head_dropout: float = 0.1, het_loss: bool = True):
        super().__init__()
        self.het_loss = het_loss
        self.seq_len = seq_len
        # Moving-average pooling for trend component; odd kernel for symmetric padding
        k = min(kernel_size, seq_len) if seq_len >= 3 else max(3, seq_len)
        if k % 2 == 0:
            k -= 1
        self.kernel = k
        self.pool = nn.AvgPool1d(kernel_size=k, stride=1, padding=(k - 1) // 2)
        flat = seq_len * n_input_features
        self.trend_lin = nn.Linear(flat, hidden_size)
        self.seasonal_lin = nn.Linear(flat, hidden_size)
        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, L, D]
        x_t = x.transpose(1, 2)               # [B, D, L]
        trend = self.pool(x_t).transpose(1, 2)  # [B, L, D]
        seasonal = x - trend
        B = x.size(0)
        h = self.trend_lin(trend.reshape(B, -1)) + self.seasonal_lin(seasonal.reshape(B, -1))
        h = F.gelu(h)
        return _forward_heads(self.heads, h, self.het_loss)


# ---------------------------------------------------------------------------
# N-BEATS — Neural Basis Expansion (Oreshkin et al. ICLR 2020, arXiv:1905.10437)
# Stack of MLP blocks that jointly predict a backcast (reconstructing the
# input) + forecast; each block subtracts its backcast from the residual so
# subsequent blocks refine the remainder. Here we use a generic (not
# trend/seasonality-interpretable) version for regression.
# ---------------------------------------------------------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: int, theta_b: int, theta_f: int,
                 n_layers: int = 4):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.b_proj = nn.Linear(hidden, theta_b)
        self.f_proj = nn.Linear(hidden, theta_f)

    def forward(self, x):
        h = self.trunk(x)
        return self.b_proj(h), self.f_proj(h)


class CurrencyNBeats(nn.Module):
    def __init__(self, n_input_features: int, seq_len: int = 10,
                 n_stacks: int = 3, blocks_per_stack: int = 3, hidden: int = 256,
                 head_dropout: float = 0.1, het_loss: bool = True):
        super().__init__()
        self.het_loss = het_loss
        self.input_dim = seq_len * n_input_features
        self.blocks = nn.ModuleList()
        for _ in range(n_stacks):
            for _ in range(blocks_per_stack):
                self.blocks.append(NBeatsBlock(
                    self.input_dim, hidden,
                    theta_b=self.input_dim, theta_f=hidden,
                ))
        self.heads = _make_heads(hidden, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.size(0)
        res = x.reshape(B, -1)  # [B, seq_len*n_features]
        forecast = torch.zeros(B, self.blocks[0].f_proj.out_features,
                                device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            b, f = blk(res)
            res = res - b
            forecast = forecast + f
        return _forward_heads(self.heads, forecast, self.het_loss)


# ---------------------------------------------------------------------------
# iTransformer — Inverted Transformer (Liu et al. ICLR 2024, arXiv:2310.06625)
# Inverts the attention: treats each FEATURE as a token (not each timestep).
# For multivariate with many variates, variate-attention captures cross-feature
# structure that temporal attention misses. Our 104 features × 10+ timesteps
# is exactly the regime this paper targets.
# ---------------------------------------------------------------------------
class CurrencyITransformer(nn.Module):
    def __init__(self, n_input_features: int, seq_len: int = 10,
                 hidden_size: int = 128, num_heads: int = 4, num_layers: int = 2,
                 head_dropout: float = 0.1, het_loss: bool = True):
        super().__init__()
        self.het_loss = het_loss
        # Each feature becomes a token; its "embedding" is its seq_len-long
        # history (linear projection: seq_len -> hidden_size).
        self.feat_embed = nn.Linear(seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pool over features then predict
        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, L, D] -> invert to [B, D, L] so each feature is a token
        x_t = x.transpose(1, 2)                # [B, D, L]
        tokens = self.feat_embed(x_t)          # [B, D, hidden]
        enc = self.encoder(tokens)             # [B, D, hidden]
        pooled = enc.mean(dim=1)               # [B, hidden]  — mean over variates
        return _forward_heads(self.heads, pooled, self.het_loss)


# ---------------------------------------------------------------------------
# xLSTM — Extended LSTM (Beck et al. NeurIPS 2024, arXiv:2405.04517)
# Replaces sigmoid gates with exponential gates; adds a normalizer state
# n_t = f_t · n_{t-1} + i_t so output is c_t / max(|n_t|, 1). This is the
# sLSTM variant (scalar state). mLSTM (matrix state) is heavier; our input
# is small enough that sLSTM is the right choice.
# ---------------------------------------------------------------------------
class sLSTMCell(nn.Module):
    """Single scalar-state xLSTM cell (Beck 2024 Section 2, Eq 6-9)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Standard 4 gates + normaliser (5 projections per direction)
        self.W = nn.Linear(input_size, 5 * hidden_size, bias=True)
        self.U = nn.Linear(hidden_size, 5 * hidden_size, bias=False)

    def forward(self, x_seq):
        # x_seq: [B, L, input_size]
        B, L, _ = x_seq.shape
        H = self.hidden_size
        h = torch.zeros(B, H, device=x_seq.device, dtype=x_seq.dtype)
        c = torch.zeros(B, H, device=x_seq.device, dtype=x_seq.dtype)
        n = torch.ones(B, H, device=x_seq.device, dtype=x_seq.dtype)  # normaliser
        m = torch.zeros(B, H, device=x_seq.device, dtype=x_seq.dtype)  # stabilizer
        outputs = []
        for t in range(L):
            gates = self.W(x_seq[:, t]) + self.U(h)
            i, f, z, o, _unused = gates.chunk(5, dim=-1)
            # Exponential gating (Beck 2024 Eq 7)
            m_new = torch.maximum(f + m, i)
            i_g = torch.exp(i - m_new)
            f_g = torch.exp(f + m - m_new)
            c = f_g * c + i_g * torch.tanh(z)
            n = f_g * n + i_g
            h = torch.sigmoid(o) * (c / torch.clamp(torch.abs(n), min=1.0))
            m = m_new
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # [B, L, H]


class CurrencyxLSTM(nn.Module):
    def __init__(self, n_input_features: int, hidden_size: int = 128,
                 num_layers: int = 2, head_dropout: float = 0.1,
                 het_loss: bool = True):
        super().__init__()
        self.het_loss = het_loss
        self.cells_fwd = nn.ModuleList()
        in_dim = n_input_features
        for _ in range(num_layers):
            self.cells_fwd.append(sLSTMCell(in_dim, hidden_size))
            in_dim = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.heads = _make_heads(hidden_size, dropout=head_dropout, het_loss=het_loss)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = x
        for cell in self.cells_fwd:
            h = cell(h)
        hidden = self.norm(h[:, -1, :])
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

    # Backbone-default SOTA hyperparameters per CLAUDE.md Tier-3 recipe table.
    # These are starting points; users override via runner CLI flags which are
    # routed through `hp_overrides` at __init__ time.
    _DEFAULTS = {
        "xgboost": dict(
            n_estimators=1500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, gamma=0.0,
            reg_alpha=0.0, reg_lambda=1.0,
            tree_method="hist",
        ),
        "lightgbm": dict(
            n_estimators=2000, num_leaves=63, max_depth=-1,
            learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, min_data_in_leaf=20,
            reg_alpha=0.1, reg_lambda=1.0,
        ),
        "catboost": dict(
            iterations=2000, depth=6, learning_rate=0.03,
            l2_leaf_reg=3.0, random_strength=1.0,
            bagging_temperature=1.0, bootstrap_type="Bayesian",
        ),
    }

    def __init__(self, gbm_type: str, n_targets: int = 2,
                 hp_overrides: dict | None = None):
        self.gbm_type = gbm_type
        self.n_targets = n_targets
        self.models: list = []  # one per target column
        self._fitted = False
        # Merge user overrides onto the backbone-default recipe
        base = dict(self._DEFAULTS.get(gbm_type, {}))
        if hp_overrides:
            base.update({k: v for k, v in hp_overrides.items() if v is not None})
        self.hp = base

    def _create_estimator(self):
        if self.gbm_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(verbosity=0, **self.hp)
        elif self.gbm_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(verbose=-1, **self.hp)
        elif self.gbm_type == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(verbose=0, **self.hp)
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

    # No-op torch.nn.Module compatibility shims so the runner's
    # _evaluate_per_window / predict helpers can call .eval() / .train()
    # on any model without branching.
    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    @property
    def training(self) -> bool:
        return False

    def __call__(self, x):
        """Torch-compatible forward: accepts [B, L, D] tensor, flattens to
        [B, L*D] and predicts, returns a dict shaped like neural heads so the
        runner's _evaluate_per_window can treat GBMs and nets uniformly.

        Output schema matches _make_heads output: each head key maps to a
        [B, N_PAIRS] tensor. Since our GBMs are trained to predict
        EUR/USD scalar per horizon, the prediction is placed at column 0
        (the pair_idx the evaluator reads) and zeros fill the remaining
        pair slots.
        """
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            X_np = x.detach().cpu().numpy()
        else:
            X_np = np.asarray(x)
        if X_np.ndim == 3:
            B, L, D = X_np.shape
            X_flat = X_np.reshape(B, L * D)
        else:
            B = X_np.shape[0]
            X_flat = X_np
        preds = self.predict(X_flat)  # [B, n_targets] — ret_1d, ret_5d
        out = {}
        horizons = ["ret_1d", "ret_5d"]
        for col, horizon in enumerate(horizons[:self.n_targets]):
            arr = np.zeros((B, N_PAIRS), dtype=np.float32)
            arr[:, 0] = preds[:, col]
            out[horizon] = _torch.from_numpy(arr)
        return out


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
    bidirectional: bool | None = None,
    num_layers: int | None = None,
    rnn_cell: str | None = None,
    input_layernorm: bool = False,
    mamba_variant: str | None = None,
    mamba_d_state: int | None = None,
    mamba_expand: int | None = None,
    gbm_hp_overrides: dict | None = None,
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
        if bidirectional is not None:
            kwargs["bidirectional"] = bidirectional
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        if rnn_cell is not None:
            kwargs["cell"] = rnn_cell
        if input_layernorm:
            kwargs["input_layernorm"] = True
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
    elif backbone == "mamba":
        kwargs = dict(head_dropout=head_dropout, het_loss=het_loss)
        if hidden_size is not None:
            kwargs["hidden_size"] = hidden_size
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        if mamba_variant is not None:
            kwargs["variant"] = mamba_variant
        if mamba_d_state is not None:
            kwargs["d_state"] = mamba_d_state
        if mamba_expand is not None:
            kwargs["expand"] = mamba_expand
        return CurrencyMamba(n_input_features, **kwargs)
    elif backbone == "dlinear":
        kwargs = dict(seq_len=seq_len, head_dropout=head_dropout, het_loss=het_loss)
        if hidden_size is not None:
            kwargs["hidden_size"] = hidden_size
        return CurrencyDLinear(n_input_features, **kwargs)
    elif backbone == "nbeats":
        kwargs = dict(seq_len=seq_len, head_dropout=head_dropout, het_loss=het_loss)
        return CurrencyNBeats(n_input_features, **kwargs)
    elif backbone == "itransformer":
        kwargs = dict(seq_len=seq_len, head_dropout=head_dropout, het_loss=het_loss)
        if hidden_size is not None:
            kwargs["hidden_size"] = hidden_size
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        return CurrencyITransformer(n_input_features, **kwargs)
    elif backbone == "xlstm":
        kwargs = dict(head_dropout=head_dropout, het_loss=het_loss)
        if hidden_size is not None:
            kwargs["hidden_size"] = hidden_size
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        return CurrencyxLSTM(n_input_features, **kwargs)
    elif backbone in GBM_BACKBONES:
        return GBMWrapper(backbone, n_targets=2, hp_overrides=gbm_hp_overrides)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )
