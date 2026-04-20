"""Uncertainty estimation — MC Dropout + deep ensembles for neural nets; quantile regression for trees.

Returns `(mean, aleatoric, epistemic, confidence)` per prediction.

Cite: Kendall & Gal 2017 NeurIPS 'What Uncertainties Do We Need in Bayesian Deep Learning
for Computer Vision?' (arXiv:1703.04977) — aleatoric + epistemic decomposition;
Gal & Ghahramani 2016 ICML 'Dropout as a Bayesian Approximation: Representing Model
Uncertainty in Deep Learning' (arXiv:1506.02142) — MC dropout theory;
Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS 'Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles' (arXiv:1612.01474) — deep ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


@dataclass
class PredictionBundle:
    mean: np.ndarray
    aleatoric: np.ndarray  # data uncertainty — noise the model explicitly models
    epistemic: np.ndarray  # model uncertainty — variance of ensemble / MC dropout samples
    confidence: np.ndarray  # 1 - epistemic (normalized to 0..1 clamp)


def mc_dropout_predict(
    forward_fn: Callable[[Any], tuple[np.ndarray, np.ndarray | None]],
    x: Any,
    n_samples: int = 30,
) -> PredictionBundle:
    """MC-dropout inference. `forward_fn` must:
      - run the model in train mode (dropout active)
      - return (mean, log_variance) OR (mean, None) if the model has no variance head
    """
    means = []
    log_vars = []
    for _ in range(n_samples):
        mu, logv = forward_fn(x)
        means.append(np.asarray(mu))
        if logv is not None:
            log_vars.append(np.asarray(logv))
    means = np.stack(means, axis=0)  # (n_samples, N, [D])
    mean = means.mean(axis=0)
    epistemic = means.std(axis=0)
    if log_vars:
        log_vars = np.stack(log_vars, axis=0)
        aleatoric = np.sqrt(np.exp(log_vars).mean(axis=0))
    else:
        aleatoric = np.zeros_like(mean)
    # normalize confidence to [0,1]
    eps_norm = epistemic / (epistemic.max() + 1e-9)
    confidence = np.clip(1.0 - eps_norm, 0.0, 1.0)
    return PredictionBundle(mean=mean, aleatoric=aleatoric, epistemic=epistemic, confidence=confidence)


def ensemble_predict(models_and_xforms: list[tuple[Callable, Any]]) -> PredictionBundle:
    """Deep ensemble — average predictions from independently trained seeds."""
    preds = []
    log_vars = []
    for fwd, x in models_and_xforms:
        out = fwd(x)
        if isinstance(out, tuple):
            mu, logv = out
        else:
            mu, logv = out, None
        preds.append(np.asarray(mu))
        if logv is not None:
            log_vars.append(np.asarray(logv))
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    epistemic = preds.std(axis=0)
    if log_vars:
        lv = np.stack(log_vars, axis=0)
        aleatoric = np.sqrt(np.exp(lv).mean(axis=0))
    else:
        aleatoric = np.zeros_like(mean)
    eps_norm = epistemic / (epistemic.max() + 1e-9)
    confidence = np.clip(1.0 - eps_norm, 0.0, 1.0)
    return PredictionBundle(mean=mean, aleatoric=aleatoric, epistemic=epistemic, confidence=confidence)


def tree_quantile_predict(
    point_preds: np.ndarray,
    q_low_preds: np.ndarray,
    q_high_preds: np.ndarray,
) -> PredictionBundle:
    """Quantile-regression uncertainty for tree models.
    `q_low_preds` is the 5th percentile model prediction, `q_high_preds` the 95th.
    Aleatoric = (q_high - q_low) / 2 (Gaussian approx); epistemic = 0 unless bagged.
    """
    spread = np.abs(q_high_preds - q_low_preds) / 2.0
    zeros = np.zeros_like(point_preds)
    conf = np.clip(1.0 - spread / (spread.max() + 1e-9), 0.0, 1.0)
    return PredictionBundle(
        mean=point_preds, aleatoric=spread, epistemic=zeros, confidence=conf
    )
