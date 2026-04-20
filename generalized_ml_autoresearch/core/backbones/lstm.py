"""LSTM backbone — time-series focused (and regression on sequence-shaped inputs).

Default recipe:
  - hidden_size: 128
  - num_layers: 2, bidirectional
  - head_dropout: 0.25
  - epochs: 100, patience: 15 (per Fischer & Krauss 2018)
  - lr: 1e-3, wd: 7e-4, batch: 16

Cite: Fischer & Krauss 2018 EJOR 'Deep learning with long short-term memory networks
for financial market predictions'.
"""

from __future__ import annotations

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


class _LSTMModule(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 n_outputs: int, bidirectional: bool, head_dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=head_dropout if num_layers > 1 else 0.0,
        )
        head_in = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(head_in, n_outputs)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        last = out[:, -1, :]   # last timestep
        last = self.dropout(last)
        mean = self.head(last)
        return mean, None  # no variance head by default


@register_backbone("lstm")
class LSTMBackbone(Backbone):
    name = "lstm"
    task_types = {"regression", "time_series_forecasting", "binary_classification",
                  "multiclass_classification"}

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM backbone")
        self.config = dict(config)
        # input_shape expected: (seq_len, n_features)
        if len(input_shape) == 1:
            # caller forgot to reshape: assume seq_len=1
            self._seq_len = 1
            self._n_features = input_shape[0]
        else:
            self._seq_len, self._n_features = input_shape[-2], input_shape[-1]
        self._n_outputs = n_outputs
        hidden_size = int(config.get("hidden_size", 128))
        num_layers = int(config.get("num_layers", 2))
        bidirectional = bool(config.get("bidirectional", True))
        head_dropout = float(config.get("head_dropout", 0.25))
        self._model = _LSTMModule(self._n_features, hidden_size, num_layers, n_outputs,
                                   bidirectional, head_dropout)
        self._device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu")
                                     else "cpu")
        self._model.to(self._device)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        cfg = self.config
        epochs = int(cfg.get("epochs", 100))
        patience = int(cfg.get("patience", 15))
        batch_size = int(cfg.get("batch_size", 16))
        lr = float(cfg.get("lr", 1e-3))
        wd = float(cfg.get("weight_decay", 7e-4))
        grad_clip = float(cfg.get("grad_clip", 1.0))
        seed = int(cfg.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_train_np = np.asarray(X_train)
        if X_train_np.ndim == 2:
            X_train_np = X_train_np[:, None, :]  # add seq_len dim
        X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        X_val_t = None
        y_val_np = None
        if X_val is not None:
            Xv = np.asarray(X_val)
            if Xv.ndim == 2:
                Xv = Xv[:, None, :]
            X_val_t = torch.tensor(Xv, dtype=torch.float32, device=self._device)
            y_val_np = np.asarray(y_val, dtype=float)

        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        history: dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val, best_state, patience_ctr = float("inf"), None, 0

        for epoch in range(epochs):
            self._model.train()
            batch_losses = []
            for xb, yb in loader:
                xb = xb.to(self._device); yb = yb.to(self._device)
                opt.zero_grad()
                mean, _ = self._model(xb)
                y_target = yb.view_as(mean) if mean.ndim > 1 else yb
                loss = torch.nn.functional.smooth_l1_loss(mean, y_target, beta=1.0)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                opt.step()
                batch_losses.append(float(loss.item()))
            sched.step()
            history["train_loss"].append(float(np.mean(batch_losses)))
            if X_val_t is not None:
                self._model.eval()
                with torch.no_grad():
                    mean, _ = self._model(X_val_t)
                    y_target = torch.tensor(y_val_np, dtype=torch.float32, device=self._device)
                    y_target = y_target.view_as(mean) if mean.ndim > 1 else y_target
                    val_loss = float(torch.nn.functional.smooth_l1_loss(mean, y_target, beta=1.0).item())
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
        X_np = np.asarray(X)
        if X_np.ndim == 2:
            X_np = X_np[:, None, :]
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self._device)
        self._model.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(max(1, n_samples)):
                mean, _ = self._model(X_t)
                preds.append(mean.cpu().numpy())
        self._model.eval()
        preds_arr = np.stack(preds, axis=0)
        mean_out = preds_arr.mean(axis=0)
        epistemic = preds_arr.std(axis=0)
        aleatoric = np.zeros_like(mean_out)
        eps_norm = epistemic / (epistemic.max() + 1e-9)
        confidence = np.clip(1.0 - eps_norm, 0.0, 1.0)
        return PredictionBundle(mean=mean_out, aleatoric=aleatoric, epistemic=epistemic, confidence=confidence)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(),
            "config": self.config,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "n_features": self._n_features,
            "seq_len": self._seq_len,
            "n_outputs": self._n_outputs,
            "backbone_name": self.name,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMBackbone":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        inst = cls()
        inst.config = payload["config"]
        inst.feature_columns = payload.get("feature_columns", [])
        inst.target_columns = payload.get("target_columns", [])
        inst.scaler_mean = payload.get("scaler_mean")
        inst.scaler_scale = payload.get("scaler_scale")
        inst.build(inst.config, (payload["seq_len"], payload["n_features"]), payload["n_outputs"])
        inst._model.load_state_dict(payload["state_dict"])
        return inst

    def gpu_memory_estimate_mb(self, batch_size: int) -> float:
        hidden = self.config.get("hidden_size", 128)
        n_layers = self.config.get("num_layers", 2)
        n_features = getattr(self, "_n_features", 100)
        seq = getattr(self, "_seq_len", 10)
        bidir = 2 if self.config.get("bidirectional", True) else 1
        params = (4 * hidden * (n_features + hidden)) * n_layers * bidir
        param_mb = params * 4 / 1e6
        return param_mb * 4 + batch_size * seq * hidden * bidir * 4 / 1e6 + 100
