"""FT-Transformer-style tabular transformer (task-agnostic).

Reference: Gorishniy, Rubachev, Khrulkov & Babenko 2021 NeurIPS
'Revisiting Deep Learning Models for Tabular Data' (arXiv:2106.11189).

Minimal implementation that works for small tabular datasets. Users upgrade by
replacing this module with the full `rtdl` package or HuggingFace implementation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

from .base import Backbone, PredictionBundle
from .registry import register_backbone


class _TabTransformerModule(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int,
                 n_outputs: int, dropout: float = 0.1, norm_first: bool = False,
                 ff_factor: float = 2.0, num_embedding: str = "linear",
                 n_frequencies: int = 16, pooling: str = "cls",
                 feature_tokenizer: str = "shared", activation: str = "gelu"):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.num_embedding = num_embedding
        self.pooling = pooling
        self.feature_tokenizer = feature_tokenizer
        if num_embedding == "periodic":
            # Gorishniy et al. 2022 periodic embeddings (arXiv:2203.05556).
            n_freq = int(n_frequencies)
            self.frequencies = nn.Parameter(torch.normal(0.0, 0.01, (n_features, n_freq)))
            self.periodic_weight = nn.Parameter(torch.empty(n_features, 2 * n_freq, d_model))
            nn.init.xavier_uniform_(self.periodic_weight.reshape(-1, d_model))
            self.periodic_bias = nn.Parameter(torch.zeros(n_features, d_model))
            self.feature_embed = None
            self.feature_weight = None
            self.feature_bias = None
        elif feature_tokenizer == "per_feature":
            # Gorishniy et al. 2021 Feature Tokenizer: T_j(x_j) = b_j + x_j W_j.
            self.feature_embed = None
            self.feature_weight = nn.Parameter(torch.empty(n_features, d_model))
            self.feature_bias = nn.Parameter(torch.empty(n_features, d_model))
            nn.init.kaiming_uniform_(self.feature_weight, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(1)
            nn.init.uniform_(self.feature_bias, -bound, bound)
        else:
            # Shared Linear(1, d) across columns (legacy default).
            self.feature_embed = nn.Linear(1, d_model)
            self.feature_weight = None
            self.feature_bias = None
        self.pos_embed = nn.Parameter(torch.randn(1, n_features + 1, d_model))
        dim_ff = max(n_heads, int(round(d_model * float(ff_factor))))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation=activation,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        b, n = x.shape
        if self.num_embedding == "periodic":
            v = 2.0 * math.pi * self.frequencies.unsqueeze(0) * x.unsqueeze(-1)
            p = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
            tokens = torch.einsum("bnk,nkd->bnd", p, self.periodic_weight) + self.periodic_bias
        elif self.feature_tokenizer == "per_feature":
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        else:
            tokens = self.feature_embed(x.unsqueeze(-1))  # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : n + 1, :]
        enc = self.encoder(tokens)
        pooled = enc.mean(dim=1) if self.pooling == "mean" else enc[:, 0, :]
        pooled = self.dropout(pooled)
        return self.head(pooled), None


@register_backbone("ft_transformer")
class FTTransformerBackbone(Backbone):
    name = "ft_transformer"
    task_types = {"regression", "binary_classification", "multiclass_classification",
                  "time_series_forecasting"}

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for FT-Transformer backbone")
        self.config = dict(config)
        self._n_features = int(np.prod(input_shape))
        self._n_outputs = n_outputs
        d_model = int(config.get("d_model", 64))
        n_heads = int(config.get("n_heads", 4))
        n_layers = int(config.get("n_layers", 3))
        dropout = float(config.get("dropout", 0.1))
        norm_first = bool(config.get("norm_first", False))
        ff_factor = float(config.get("ff_factor", 2.0))
        num_embedding = str(config.get("num_embedding", "linear"))
        n_frequencies = int(config.get("n_frequencies", 16))
        pooling = str(config.get("pooling", "cls"))
        feature_tokenizer = str(config.get("feature_tokenizer", "shared"))
        activation = str(config.get("activation", "gelu"))
        self._task_type = config.get("task_type", "regression")
        self._model = _TabTransformerModule(self._n_features, d_model, n_heads, n_layers,
                                             n_outputs, dropout, norm_first=norm_first,
                                             ff_factor=ff_factor, num_embedding=num_embedding,
                                             n_frequencies=n_frequencies, pooling=pooling,
                                             feature_tokenizer=feature_tokenizer,
                                             activation=activation)
        self._device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu")
                                     else "cpu")
        self._model.to(self._device)

    def _loss(self, mean, y):
        t = self._task_type
        if t in ("regression", "time_series_forecasting"):
            return torch.nn.functional.smooth_l1_loss(mean, y, beta=1.0)
        if t == "binary_classification":
            return torch.nn.functional.binary_cross_entropy_with_logits(mean.squeeze(-1), y.float())
        return torch.nn.functional.cross_entropy(mean, y.long())

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        cfg = self.config
        epochs = int(cfg.get("epochs", 100))
        patience = int(cfg.get("patience", 15))
        batch_size = int(cfg.get("batch_size", 64))
        lr = float(cfg.get("lr", 1e-4))
        wd = float(cfg.get("weight_decay", 1e-5))
        warmup = int(cfg.get("warmup", 10))
        seed = int(cfg.get("seed", 0))
        torch.manual_seed(seed)

        X_t = torch.tensor(np.asarray(X_train).reshape(len(X_train), -1), dtype=torch.float32)
        task_type = self._task_type
        y_t = torch.tensor(np.asarray(y_train),
                           dtype=torch.long if task_type == "multiclass_classification" else torch.float32)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        if X_val is not None:
            X_val_t = torch.tensor(np.asarray(X_val).reshape(len(X_val), -1),
                                     dtype=torch.float32, device=self._device)
        else:
            X_val_t = None
        y_val_np = np.asarray(y_val) if y_val is not None else None

        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return 0.5 * (1.0 + np.cos(np.pi * (step - warmup) / max(1, epochs - warmup)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        history: dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val, best_state, patience_ctr = float("inf"), None, 0
        for epoch in range(epochs):
            self._model.train()
            batch_losses = []
            for xb, yb in loader:
                xb = xb.to(self._device); yb = yb.to(self._device)
                opt.zero_grad()
                mean, _ = self._model(xb)
                y_target = yb.view_as(mean) if (task_type in ("regression", "time_series_forecasting") and mean.ndim > 1) else yb
                loss = self._loss(mean, y_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                batch_losses.append(float(loss.item()))
            sched.step()
            history["train_loss"].append(float(np.mean(batch_losses)))
            if X_val_t is not None:
                self._model.eval()
                with torch.no_grad():
                    mean, _ = self._model(X_val_t)
                    if task_type in ("regression", "time_series_forecasting"):
                        yv = torch.tensor(y_val_np, dtype=torch.float32, device=self._device)
                        yv = yv.view_as(mean) if mean.ndim > 1 else yv
                        val_loss = float(self._loss(mean, yv).item())
                    elif task_type == "binary_classification":
                        yv = torch.tensor(y_val_np, dtype=torch.float32, device=self._device)
                        val_loss = float(self._loss(mean, yv).item())
                    else:
                        yv = torch.tensor(y_val_np, dtype=torch.long, device=self._device)
                        val_loss = float(self._loss(mean, yv).item())
                history["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        history["epochs_run"] = epoch + 1
        history["best_val_loss"] = best_val
        return history

    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:
        X = np.asarray(X).reshape(len(X), -1)
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        self._model.train()
        preds = []
        with torch.no_grad():
            for _ in range(max(1, n_samples)):
                mean, _ = self._model(X_t)
                preds.append(mean.cpu().numpy())
        self._model.eval()
        arr = np.stack(preds, axis=0)
        mean_out = arr.mean(axis=0)
        epi = arr.std(axis=0)
        if self._task_type == "binary_classification":
            import scipy.special as sp
            probs = sp.expit(mean_out).squeeze(-1) if mean_out.ndim > 1 else sp.expit(mean_out)
            mean_out = (probs > 0.5).astype(int)
            eps_norm = epi / (epi.max() + 1e-9)
            return PredictionBundle(mean=mean_out, aleatoric=np.zeros_like(mean_out, dtype=float),
                                     epistemic=np.asarray(epi).mean(axis=-1) if epi.ndim > 1 else epi,
                                     confidence=np.abs(probs - 0.5) * 2.0, probabilities=probs)
        if self._task_type == "multiclass_classification":
            import scipy.special as sp
            probs = sp.softmax(mean_out, axis=-1)
            mean_out = probs.argmax(axis=-1)
            eps_norm = epi.mean(axis=-1)
            return PredictionBundle(mean=mean_out, aleatoric=np.zeros_like(mean_out, dtype=float),
                                     epistemic=eps_norm, confidence=probs.max(axis=-1),
                                     probabilities=probs)
        eps_norm = epi / (epi.max() + 1e-9)
        return PredictionBundle(mean=mean_out, aleatoric=np.zeros_like(mean_out),
                                 epistemic=epi, confidence=np.clip(1.0 - eps_norm, 0.0, 1.0))

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(),
            "config": self.config,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "n_features": self._n_features,
            "n_outputs": self._n_outputs,
            "task_type": self._task_type,
            "backbone_name": self.name,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "FTTransformerBackbone":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        inst = cls()
        inst.config = payload["config"]
        inst.feature_columns = payload.get("feature_columns", [])
        inst.target_columns = payload.get("target_columns", [])
        inst.scaler_mean = payload.get("scaler_mean")
        inst.scaler_scale = payload.get("scaler_scale")
        inst._n_features = payload["n_features"]
        inst._n_outputs = payload["n_outputs"]
        inst._task_type = payload["task_type"]
        inst.build(inst.config, (inst._n_features,), inst._n_outputs)
        inst._model.load_state_dict(payload["state_dict"])
        return inst
