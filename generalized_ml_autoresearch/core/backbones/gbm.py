"""GBM backbones — three SEPARATE entries per CLAUDE.md rule.

xgboost / lightgbm / catboost each register as their own backbone. Do NOT merge.

Why three separate backbones:
  - XGBoost: 2nd-order gradient info (Hessian); histogram method.
  - LightGBM: leaf-wise growth (GOSS); fastest wall-clock.
  - CatBoost: ordered boosting; best default categorical handling.

Each has its own paper, hyperparameter language, and 50-experiment exploration budget.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import Backbone, PredictionBundle
from .registry import register_backbone

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    xgb = None
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    lgb = None
    _LGB_AVAILABLE = False

try:
    from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier, Pool
    _CAT_AVAILABLE = True
except ImportError:  # pragma: no cover
    CatBoost = CatBoostRegressor = CatBoostClassifier = Pool = None
    _CAT_AVAILABLE = False


def _gbm_objective(task_type: str, framework: str) -> str:
    """Map task_type to framework-specific objective name."""
    mapping = {
        ("regression", "xgboost"): "reg:squarederror",
        ("time_series_forecasting", "xgboost"): "reg:squarederror",
        ("binary_classification", "xgboost"): "binary:logistic",
        ("multiclass_classification", "xgboost"): "multi:softprob",
        ("regression", "lightgbm"): "regression",
        ("time_series_forecasting", "lightgbm"): "regression",
        ("binary_classification", "lightgbm"): "binary",
        ("multiclass_classification", "lightgbm"): "multiclass",
        ("regression", "catboost"): "RMSE",
        ("time_series_forecasting", "catboost"): "RMSE",
        ("binary_classification", "catboost"): "Logloss",
        ("multiclass_classification", "catboost"): "MultiClass",
    }
    key = (task_type, framework)
    if key not in mapping:
        raise ValueError(f"GBM objective not defined for {key}")
    return mapping[key]


# ------------------------------------------------------------------ XGBoost

@register_backbone("xgboost")
class XGBoostBackbone(Backbone):
    name = "xgboost"
    task_types = {"regression", "binary_classification", "multiclass_classification",
                  "time_series_forecasting", "ranking"}

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        self.config = dict(config)
        self._n_features = int(np.prod(input_shape))
        self._n_outputs = n_outputs
        self._task_type = config.get("task_type", "regression")
        params = {
            "objective": _gbm_objective(self._task_type, "xgboost"),
            "n_estimators": int(config.get("n_estimators", 1500)),
            "max_depth": int(config.get("max_depth", 6)),
            "learning_rate": float(config.get("learning_rate", config.get("lr", 0.03))),
            "subsample": float(config.get("subsample", 0.8)),
            "colsample_bytree": float(config.get("colsample_bytree", 0.8)),
            "reg_lambda": float(config.get("reg_lambda", 1.0)),
            "reg_alpha": float(config.get("reg_alpha", 0.0)),
            "min_child_weight": float(config.get("min_child_weight", 1.0)),
            "gamma": float(config.get("gamma", 0.0)),
            "scale_pos_weight": float(config.get("scale_pos_weight", 1.0)),
            "random_state": int(config.get("seed", 0)),
            "tree_method": config.get("tree_method", "hist"),
            "n_jobs": int(config.get("n_jobs", 4)),
            "verbosity": 0,
        }
        if self._task_type == "multiclass_classification":
            params["num_class"] = n_outputs
        self._params = params
        if self._task_type in ("binary_classification", "multiclass_classification"):
            self._model = xgb.XGBClassifier(**params)
        else:
            self._model = xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        early = int(self.config.get("early_stopping_rounds", 50))
        X_train = np.asarray(X_train).reshape(len(X_train), -1)
        y_train = np.asarray(y_train)
        eval_set = None
        if X_val is not None:
            X_val = np.asarray(X_val).reshape(len(X_val), -1)
            y_val = np.asarray(y_val)
            eval_set = [(X_val, y_val)]
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            # XGBoost newer API uses `early_stopping_rounds` on the estimator constructor.
            try:
                self._model.set_params(early_stopping_rounds=early)
            except Exception:  # pragma: no cover
                fit_kwargs["early_stopping_rounds"] = early
        self._model.fit(X_train, y_train, **fit_kwargs)
        best_iter = getattr(self._model, "best_iteration", None)
        return {"best_iteration": best_iter or 0, "epochs_run": self._params["n_estimators"]}

    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:
        X = np.asarray(X).reshape(len(X), -1)
        if self._task_type in ("binary_classification",):
            probs = self._model.predict_proba(X)[:, 1]
            mean = (probs > 0.5).astype(int)
            return PredictionBundle(mean=mean, aleatoric=np.zeros_like(mean, dtype=float),
                                     epistemic=np.zeros_like(mean, dtype=float),
                                     confidence=np.abs(probs - 0.5) * 2.0,
                                     probabilities=probs)
        if self._task_type == "multiclass_classification":
            probs = self._model.predict_proba(X)
            mean = probs.argmax(axis=-1)
            conf = probs.max(axis=-1)
            return PredictionBundle(mean=mean, aleatoric=np.zeros_like(mean, dtype=float),
                                     epistemic=np.zeros_like(mean, dtype=float),
                                     confidence=conf, probabilities=probs)
        pred = self._model.predict(X)
        return PredictionBundle(
            mean=pred, aleatoric=np.zeros_like(pred), epistemic=np.zeros_like(pred),
            confidence=np.ones_like(pred) * 0.5,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "config": self.config,
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
                "scaler_mean": self.scaler_mean,
                "scaler_scale": self.scaler_scale,
                "n_features": self._n_features,
                "n_outputs": self._n_outputs,
                "task_type": self._task_type,
                "backbone_name": self.name,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostBackbone":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        inst = cls()
        inst.config = payload["config"]
        inst.feature_columns = payload.get("feature_columns", [])
        inst.target_columns = payload.get("target_columns", [])
        inst.scaler_mean = payload.get("scaler_mean")
        inst.scaler_scale = payload.get("scaler_scale")
        inst._n_features = payload["n_features"]
        inst._n_outputs = payload["n_outputs"]
        inst._task_type = payload["task_type"]
        inst._model = payload["model"]
        return inst


# ------------------------------------------------------------------ LightGBM

@register_backbone("lightgbm")
class LightGBMBackbone(Backbone):
    name = "lightgbm"
    task_types = {"regression", "binary_classification", "multiclass_classification",
                  "time_series_forecasting"}

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm not installed")
        self.config = dict(config)
        self._n_features = int(np.prod(input_shape))
        self._n_outputs = n_outputs
        self._task_type = config.get("task_type", "regression")
        params = {
            "objective": _gbm_objective(self._task_type, "lightgbm"),
            "n_estimators": int(config.get("n_estimators", 2000)),
            "num_leaves": int(config.get("num_leaves", 63)),
            "learning_rate": float(config.get("learning_rate", config.get("lr", 0.03))),
            "feature_fraction": float(config.get("feature_fraction", 0.8)),
            "bagging_fraction": float(config.get("bagging_fraction", 0.8)),
            "min_data_in_leaf": int(config.get("min_data_in_leaf", 20)),
            "reg_alpha": float(config.get("reg_alpha", 0.0)),
            "reg_lambda": float(config.get("reg_lambda", 0.0)),
            "random_state": int(config.get("seed", 0)),
            "n_jobs": int(config.get("n_jobs", 4)),
            "verbose": -1,
        }
        if self._task_type == "multiclass_classification":
            params["num_class"] = n_outputs
        self._params = params
        if self._task_type in ("binary_classification", "multiclass_classification"):
            self._model = lgb.LGBMClassifier(**params)
        else:
            self._model = lgb.LGBMRegressor(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        X_train = np.asarray(X_train).reshape(len(X_train), -1)
        y_train = np.asarray(y_train)
        callbacks = None
        eval_set = None
        if X_val is not None:
            X_val = np.asarray(X_val).reshape(len(X_val), -1)
            y_val = np.asarray(y_val)
            eval_set = [(X_val, y_val)]
            callbacks = [lgb.early_stopping(int(self.config.get("early_stopping_rounds", 50)),
                                             verbose=False)]
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks
        self._model.fit(X_train, y_train, **fit_kwargs)
        best = getattr(self._model, "best_iteration_", None)
        return {"best_iteration": best or 0, "epochs_run": self._params["n_estimators"]}

    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:
        X = np.asarray(X).reshape(len(X), -1)
        if self._task_type == "binary_classification":
            probs = self._model.predict_proba(X)[:, 1]
            mean = (probs > 0.5).astype(int)
            return PredictionBundle(mean=mean, aleatoric=np.zeros_like(mean, dtype=float),
                                     epistemic=np.zeros_like(mean, dtype=float),
                                     confidence=np.abs(probs - 0.5) * 2.0, probabilities=probs)
        if self._task_type == "multiclass_classification":
            probs = self._model.predict_proba(X)
            mean = probs.argmax(axis=-1)
            conf = probs.max(axis=-1)
            return PredictionBundle(mean=mean, aleatoric=np.zeros_like(mean, dtype=float),
                                     epistemic=np.zeros_like(mean, dtype=float),
                                     confidence=conf, probabilities=probs)
        pred = self._model.predict(X)
        return PredictionBundle(
            mean=pred, aleatoric=np.zeros_like(pred), epistemic=np.zeros_like(pred),
            confidence=np.ones_like(pred) * 0.5,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model, "config": self.config,
                "feature_columns": self.feature_columns, "target_columns": self.target_columns,
                "scaler_mean": self.scaler_mean, "scaler_scale": self.scaler_scale,
                "n_features": self._n_features, "n_outputs": self._n_outputs,
                "task_type": self._task_type, "backbone_name": self.name,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMBackbone":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        inst = cls()
        inst.config = payload["config"]
        inst.feature_columns = payload.get("feature_columns", [])
        inst.target_columns = payload.get("target_columns", [])
        inst.scaler_mean = payload.get("scaler_mean")
        inst.scaler_scale = payload.get("scaler_scale")
        inst._n_features = payload["n_features"]
        inst._n_outputs = payload["n_outputs"]
        inst._task_type = payload["task_type"]
        inst._model = payload["model"]
        return inst


# ------------------------------------------------------------------ CatBoost

@register_backbone("catboost")
class CatBoostBackbone(Backbone):
    name = "catboost"
    task_types = {"regression", "binary_classification", "multiclass_classification",
                  "time_series_forecasting"}

    def build(self, config: dict[str, Any], input_shape: tuple[int, ...], n_outputs: int) -> None:
        if not _CAT_AVAILABLE:
            raise ImportError("catboost not installed")
        self.config = dict(config)
        self._n_features = int(np.prod(input_shape))
        self._n_outputs = n_outputs
        self._task_type = config.get("task_type", "regression")
        params = {
            "iterations": int(config.get("iterations", config.get("n_estimators", 2000))),
            "depth": int(config.get("depth", 6)),
            "learning_rate": float(config.get("learning_rate", config.get("lr", 0.03))),
            "random_strength": float(config.get("random_strength", 1.0)),
            "l2_leaf_reg": float(config.get("l2_leaf_reg", 3.0)),
            "bagging_temperature": float(config.get("bagging_temperature", 1.0)),
            "random_seed": int(config.get("seed", 0)),
            "thread_count": int(config.get("n_jobs", 4)),
            "verbose": False,
            "loss_function": _gbm_objective(self._task_type, "catboost"),
            "early_stopping_rounds": int(config.get("early_stopping_rounds", 100)),
        }
        self._params = params
        if self._task_type in ("binary_classification", "multiclass_classification"):
            self._model = CatBoostClassifier(**params)
        else:
            self._model = CatBoostRegressor(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, Any]:
        X_train = np.asarray(X_train).reshape(len(X_train), -1)
        y_train = np.asarray(y_train)
        eval_set = None
        if X_val is not None:
            X_val = np.asarray(X_val).reshape(len(X_val), -1)
            y_val = np.asarray(y_val)
            eval_set = (X_val, y_val)
        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        best = getattr(self._model, "best_iteration_", None)
        return {"best_iteration": best or 0, "epochs_run": self._params["iterations"]}

    def predict_with_uncertainty(self, X, n_samples: int = 30) -> PredictionBundle:
        X = np.asarray(X).reshape(len(X), -1)
        if self._task_type == "binary_classification":
            probs = self._model.predict_proba(X)[:, 1]
            mean = (probs > 0.5).astype(int)
            return PredictionBundle(mean=mean, aleatoric=np.zeros_like(mean, dtype=float),
                                     epistemic=np.zeros_like(mean, dtype=float),
                                     confidence=np.abs(probs - 0.5) * 2.0, probabilities=probs)
        if self._task_type == "multiclass_classification":
            probs = self._model.predict_proba(X)
            mean = probs.argmax(axis=-1)
            conf = probs.max(axis=-1)
            return PredictionBundle(mean=mean, aleatoric=np.zeros_like(mean, dtype=float),
                                     epistemic=np.zeros_like(mean, dtype=float),
                                     confidence=conf, probabilities=probs)
        pred = self._model.predict(X).flatten()
        return PredictionBundle(
            mean=pred, aleatoric=np.zeros_like(pred), epistemic=np.zeros_like(pred),
            confidence=np.ones_like(pred) * 0.5,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model, "config": self.config,
                "feature_columns": self.feature_columns, "target_columns": self.target_columns,
                "scaler_mean": self.scaler_mean, "scaler_scale": self.scaler_scale,
                "n_features": self._n_features, "n_outputs": self._n_outputs,
                "task_type": self._task_type, "backbone_name": self.name,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostBackbone":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        inst = cls()
        inst.config = payload["config"]
        inst.feature_columns = payload.get("feature_columns", [])
        inst.target_columns = payload.get("target_columns", [])
        inst.scaler_mean = payload.get("scaler_mean")
        inst.scaler_scale = payload.get("scaler_scale")
        inst._n_features = payload["n_features"]
        inst._n_outputs = payload["n_outputs"]
        inst._task_type = payload["task_type"]
        inst._model = payload["model"]
        return inst
