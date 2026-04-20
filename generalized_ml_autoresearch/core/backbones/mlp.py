"""MLP backbone — works for regression and classification via task_type config.

Task-agnostic design: the final layer's output dimension and the loss function
are set from `config['task_type']` and `n_outputs`. Includes dropout for MC-dropout
uncertainty.

Default recipe (overrideable via config):
  - hidden: [256, 128, 64]
  - dropout: 0.2
  - lr: 3e-4
  - epochs: 50
  - patience: 10
  - batch: 32
  - weight_decay: 1e-5
  - optimizer: AdamW
  - scheduler: cosine

Cite: Gu, Kelly & Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning'
(arXiv:1802.09003) — fixed-depth MLP with dropout; competitive floor for tabular.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

from .base import Backbone, PredictionBundle
from .registry import register_backbone


class _MLPModule(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, hidden: list[int], n_outputs: int,
                 dropout: float = 0.2, task_type: str = "regression",
                 hetero_loss: bool = False):
        super().__init__()
        layers: list[Any] = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_outputs)
        self.log_var_head: nn.Module | None = None
        if hetero_loss:
            self.log_var_head = nn.Linear(in_dim, n_outputs)
        self.task_type = task_type
        self.hetero_loss = hetero_loss

    def forward(self, x):
        feat = self.trunk(x)
        mean = self.head(feat)
        if self.log_var_head is not None:
            log_var = self.log_var_head(feat)
            return mean, log_var
        return mean, None


@register_backbone("mlp")
class MLPBackbone(Backbone):
    name = "mlp"
    task_types = {"regression", "binary_classification", "multiclass_classification",
                  "time_series_forecasting"}

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLP backbone")
        self.config = dict(config)
        self._n_features = int(np.prod(input_shape))
        self._n_outputs = n_outputs
        hidden = list(config.get("hidden", [256, 128, 64]))
        dropout = float(config.get("dropout", 0.2))
        hetero_loss = bool(config.get("hetero_loss", False))
        task_type = str(config.get("task_type", "regression"))
        self._model = _MLPModule(self._n_features, hidden, n_outputs, dropout, task_type, hetero_loss)
        self._device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu")
                                     else "cpu")
        self._model.to(self._device)

    def _loss(self, mean, log_var, y):
        task_type = self.config.get("task_type", "regression")
        if task_type == "regression" or task_type == "time_series_forecasting":
            if self.config.get("hetero_loss", False) and log_var is not None:
                # Kendall & Gal 2017 heteroscedastic regression loss
                precision = torch.exp(-log_var)
                base = torch.nn.functional.smooth_l1_loss(mean, y, reduction="none", beta=1.0)
                return (precision * base + 0.5 * log_var).mean()
            return torch.nn.functional.smooth_l1_loss(mean, y, beta=1.0)
        if task_type == "binary_classification":
            return torch.nn.functional.binary_cross_entropy_with_logits(mean.squeeze(-1), y.float())
        if task_type == "multiclass_classification":
            return torch.nn.functional.cross_entropy(mean, y.long())
        raise ValueError(f"Unknown task_type {task_type!r}")

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        cfg = self.config
        epochs = int(cfg.get("epochs", 50))
        patience = int(cfg.get("patience", 10))
        batch_size = int(cfg.get("batch_size", 32))
        lr = float(cfg.get("lr", 3e-4))
        wd = float(cfg.get("weight_decay", 1e-5))
        grad_clip = float(cfg.get("grad_clip", 1.0))
        seed = int(cfg.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        task_type = cfg.get("task_type", "regression")
        if task_type in ("multiclass_classification",):
            y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.long)
        else:
            y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        ds = TensorDataset(X_train, y_train_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        if X_val is not None:
            X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self._device)
            y_val_t_np = np.asarray(y_val)
        else:
            X_val_t = None
            y_val_t_np = None

        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        history: dict[str, list] = {"train_loss": [], "val_loss": []}

        best_val = float("inf")
        best_state = None
        patience_ctr = 0
        for epoch in range(epochs):
            self._model.train()
            batch_losses = []
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                opt.zero_grad()
                out = self._model(xb)
                mean, log_var = out
                if task_type in ("regression", "time_series_forecasting"):
                    y_target = yb.view_as(mean) if mean.ndim > 1 else yb
                else:
                    y_target = yb
                loss = self._loss(mean, log_var, y_target)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                opt.step()
                batch_losses.append(float(loss.item()))
            sched.step()
            mean_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            history["train_loss"].append(mean_train_loss)
            val_loss = float("nan")
            if X_val_t is not None:
                val_loss = self._val_loss(X_val_t, y_val_t_np)
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

    def _val_loss(self, X_val_t, y_val_np) -> float:
        task_type = self.config.get("task_type", "regression")
        self._model.eval()
        with torch.no_grad():
            mean, log_var = self._model(X_val_t)
            if task_type in ("regression", "time_series_forecasting"):
                y_target = torch.tensor(y_val_np, dtype=torch.float32, device=self._device)
                y_target = y_target.view_as(mean) if mean.ndim > 1 else y_target
                loss = self._loss(mean, log_var, y_target)
            elif task_type == "binary_classification":
                y_target = torch.tensor(y_val_np, dtype=torch.float32, device=self._device)
                loss = self._loss(mean, None, y_target)
            else:
                y_target = torch.tensor(y_val_np, dtype=torch.long, device=self._device)
                loss = self._loss(mean, None, y_target)
        return float(loss.item())

    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:
        task_type = self.config.get("task_type", "regression")
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=self._device)

        # MC dropout: keep model in train mode
        self._model.train()
        preds = []
        log_vars = []
        with torch.no_grad():
            for _ in range(max(1, n_samples)):
                mean, log_var = self._model(X_t)
                preds.append(mean.cpu().numpy())
                if log_var is not None:
                    log_vars.append(log_var.cpu().numpy())
        self._model.eval()
        preds_arr = np.stack(preds, axis=0)  # (S, N, D) or (S, N)
        mean_out = preds_arr.mean(axis=0)
        epistemic = preds_arr.std(axis=0)
        if log_vars:
            lv = np.stack(log_vars, axis=0)
            aleatoric = np.sqrt(np.exp(lv).mean(axis=0))
        else:
            aleatoric = np.zeros_like(mean_out)
        eps_norm = epistemic / (epistemic.max() + 1e-9)
        confidence = np.clip(1.0 - eps_norm, 0.0, 1.0)

        # For classification tasks, compute probabilities
        probabilities: np.ndarray | None = None
        if task_type == "binary_classification":
            import scipy.special as sp_special
            probabilities = sp_special.expit(mean_out)
            # For classification, "prediction" should be the hard label
            mean_out = (probabilities > 0.5).astype(int) if probabilities.ndim <= 1 else (probabilities > 0.5).astype(int)
        elif task_type == "multiclass_classification":
            import scipy.special as sp_special
            probabilities = sp_special.softmax(mean_out, axis=-1)
            mean_out = probabilities.argmax(axis=-1)

        return PredictionBundle(
            mean=np.asarray(mean_out),
            aleatoric=np.asarray(aleatoric),
            epistemic=np.asarray(epistemic),
            confidence=np.asarray(confidence),
            probabilities=probabilities,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._model.state_dict(),
            "config": self.config,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "n_features": self._n_features,
            "n_outputs": self._n_outputs,
            "backbone_name": self.name,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPBackbone":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        inst = cls()
        inst.config = payload["config"]
        inst.feature_columns = payload.get("feature_columns", [])
        inst.target_columns = payload.get("target_columns", [])
        inst.scaler_mean = payload.get("scaler_mean")
        inst.scaler_scale = payload.get("scaler_scale")
        inst._n_features = payload["n_features"]
        inst._n_outputs = payload["n_outputs"]
        inst.build(inst.config, (inst._n_features,), inst._n_outputs)
        inst._model.load_state_dict(payload["state_dict"])
        return inst

    def gpu_memory_estimate_mb(self, batch_size: int) -> float:
        hidden = list(self.config.get("hidden", [256, 128, 64]))
        n_features = self._n_features if hasattr(self, "_n_features") else 100
        params = n_features * hidden[0]
        for i in range(len(hidden) - 1):
            params += hidden[i] * hidden[i + 1]
        params += hidden[-1] * self._n_outputs if hasattr(self, "_n_outputs") else hidden[-1]
        param_mb = params * 4 / 1e6  # FP32
        adam_mb = param_mb * 2  # two moments
        grads_mb = param_mb
        act_mb = batch_size * sum(hidden) * 4 / 1e6
        return param_mb + adam_mb + grads_mb + act_mb + 100  # +100 MB framework overhead
