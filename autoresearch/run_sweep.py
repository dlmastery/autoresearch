"""Optuna hyperparameter sweep with walk-forward aware tuning.

Supports two evaluation modes:
  - quick:  2-fold screen (fold_5 + fold_7) for fast iteration
  - full:   all 7 folds for final evaluation

Usage:
    python run_sweep.py --backbone mlp --n-trials 50
    python run_sweep.py --backbone lstm --n-trials 30 --mode full
    python run_sweep.py --backbone xgboost --n-trials 100
    python run_sweep.py --backbone lfm2-350m --n-trials 20
    python run_sweep.py --list-backbones
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .data.download import download_all_pairs, download_macro_signals
from .data.features import compute_all_features, compute_targets
from .data.splits import FOLDS, split_data, validate_purge_embargo
from .model.train import find_contiguous_segments
from .evaluation.metrics import (
    sharpe_ratio,
    average_sharpe_across_folds,
    weighted_sharpe_across_folds,
    trading_report,
    information_coefficient,
)
from .model.backbone import (
    create_model,
    BACKBONE_REGISTRY,
    is_gbm,
    GBMWrapper,
)
from .model.train import create_dataset, FXDataset

logger = logging.getLogger(__name__)

SWEEP_DIR = Path(__file__).resolve().parent / "sweep_results"

# Quick screening folds: fold_5 (low-vol plateau) + fold_7 (recent mixed)
# These two cover diverse regimes and have the most training data.
QUICK_FOLDS = [f for f in FOLDS if f["name"] in ("fold_5", "fold_7")]


# ---------------------------------------------------------------------------
# Per-backbone hyperparameter search spaces
# ---------------------------------------------------------------------------

def _suggest_shared_neural(trial: optuna.Trial) -> dict:
    """Shared hyperparameters for non-LFM neural backbones (short window)."""
    return {
        "seq_len": trial.suggest_categorical("seq_len", [5, 10, 15, 20]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 3, 15),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "patience": trial.suggest_int("patience", 2, 6),
    }


def suggest_mlp(trial: optuna.Trial) -> dict:
    hp = _suggest_shared_neural(trial)
    hp["hidden_size"] = trial.suggest_categorical("hidden_size", [256, 512, 1024])
    hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    return hp


def suggest_lstm(trial: optuna.Trial) -> dict:
    hp = _suggest_shared_neural(trial)
    hp["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 256, 512])
    hp["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
    return hp


def suggest_patchtst(trial: optuna.Trial) -> dict:
    hp = _suggest_shared_neural(trial)
    hp["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 256, 512])
    hp["num_attention_heads"] = trial.suggest_categorical("num_attention_heads", [2, 4, 8])
    hp["num_hidden_layers"] = trial.suggest_int("num_hidden_layers", 1, 4)
    hp["patch_length"] = trial.suggest_categorical("patch_length", [2, 3, 5, 10])
    # Constraint: seq_len must be divisible by patch_length with >= 2 patches
    if hp["seq_len"] % hp["patch_length"] != 0 or hp["seq_len"] // hp["patch_length"] < 2:
        raise optuna.TrialPruned("seq_len not divisible by patch_length or < 2 patches")
    return hp


def suggest_patchtsmixer(trial: optuna.Trial) -> dict:
    hp = _suggest_shared_neural(trial)
    hp["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 256, 512])
    hp["num_hidden_layers"] = trial.suggest_int("num_hidden_layers", 2, 6)
    hp["patch_length"] = trial.suggest_categorical("patch_length", [2, 3, 5, 10])
    if hp["seq_len"] % hp["patch_length"] != 0 or hp["seq_len"] // hp["patch_length"] < 2:
        raise optuna.TrialPruned("seq_len not divisible by patch_length or < 2 patches")
    return hp


def suggest_lfm2(trial: optuna.Trial) -> dict:
    """LFM2.5 — limited sweep since backbone is frozen."""
    return {
        "seq_len": trial.suggest_categorical("seq_len", [20, 40, 60, 90]),
        "lr": trial.suggest_float("lr", 5e-6, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 3, 10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 3.0),
        "patience": trial.suggest_int("patience", 2, 5),
    }


def suggest_gbm(trial: optuna.Trial, gbm_type: str) -> dict:
    hp = {
        "seq_len": trial.suggest_categorical("seq_len", [5, 10, 15, 20]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    if gbm_type == "lightgbm":
        hp["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
        hp["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
    elif gbm_type == "catboost":
        hp["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
    elif gbm_type == "xgboost":
        hp["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        hp["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
    return hp


SUGGEST_FN = {
    "mlp": suggest_mlp,
    "lstm": suggest_lstm,
    "patchtst": suggest_patchtst,
    "patchtsmixer": suggest_patchtsmixer,
    "lfm2-350m": suggest_lfm2,
    "xgboost": lambda t: suggest_gbm(t, "xgboost"),
    "lightgbm": lambda t: suggest_gbm(t, "lightgbm"),
    "catboost": lambda t: suggest_gbm(t, "catboost"),
}


# ---------------------------------------------------------------------------
# Model creation with hyperparameters
# ---------------------------------------------------------------------------

def _create_model_with_hp(backbone: str, n_features: int, hp: dict):
    """Create a model using suggested hyperparameters."""
    seq_len = hp.get("seq_len", 60)

    if backbone == "mlp":
        from .model.backbone import CurrencyMLP
        return CurrencyMLP(n_features, seq_len=seq_len, hidden_size=hp.get("hidden_size", 512))
    elif backbone == "lstm":
        from .model.backbone import CurrencyLSTM
        return CurrencyLSTM(
            n_features,
            hidden_size=hp.get("hidden_size", 256),
            num_layers=hp.get("num_layers", 2),
        )
    elif backbone.startswith("lfm2"):
        return create_model(backbone, n_features, seq_len=seq_len, freeze_backbone=True)
    elif backbone == "patchtst":
        from .model.backbone import CurrencyPatchTST
        return CurrencyPatchTST(
            n_features, seq_len=seq_len,
            patch_length=hp.get("patch_length", 12),
            hidden_size=hp.get("hidden_size", 256),
            num_attention_heads=hp.get("num_attention_heads", 4),
            num_hidden_layers=hp.get("num_hidden_layers", 3),
        )
    elif backbone == "patchtsmixer":
        from .model.backbone import CurrencyPatchTSMixer
        return CurrencyPatchTSMixer(
            n_features, seq_len=seq_len,
            patch_length=hp.get("patch_length", 12),
            hidden_size=hp.get("hidden_size", 256),
            num_hidden_layers=hp.get("num_hidden_layers", 4),
        )
    elif is_gbm(backbone):
        return _create_gbm_with_hp(backbone, hp)
    else:
        return create_model(backbone, n_features, seq_len=seq_len)


def _create_gbm_with_hp(backbone: str, hp: dict) -> GBMWrapper:
    """Create a GBM wrapper with custom hyperparameters."""
    wrapper = GBMWrapper(backbone, n_targets=2)

    # Override _create_estimator to inject trial hyperparameters
    original_create = wrapper._create_estimator

    def custom_create():
        est = original_create()
        for key in ("n_estimators", "max_depth", "learning_rate", "subsample",
                     "colsample_bytree", "num_leaves", "min_child_samples",
                     "min_child_weight", "gamma", "l2_leaf_reg"):
            if key in hp:
                try:
                    setattr(est, key, hp[key])
                except AttributeError:
                    pass  # Not all params apply to all GBM types
        return est

    wrapper._create_estimator = custom_create
    return wrapper


# ---------------------------------------------------------------------------
# Custom training with hp
# ---------------------------------------------------------------------------

def _train_neural_fold_with_hp(
    model, train_feat, train_tgt, val_feat, val_tgt,
    scaler, hp,
):
    """Train one neural fold with custom hyperparameters."""
    seq_len = hp.get("seq_len", 60)
    lr = hp.get("lr", 1e-4)
    batch_size = hp.get("batch_size", 64)
    epochs = hp.get("epochs", 5)
    weight_decay = hp.get("weight_decay", 1e-5)
    grad_clip = hp.get("grad_clip", 1.0)
    patience = hp.get("patience", 3)

    # Scale
    train_s = pd.DataFrame(
        scaler.transform(train_feat.values),
        index=train_feat.index, columns=train_feat.columns,
    )
    val_s = pd.DataFrame(
        scaler.transform(val_feat.values),
        index=val_feat.index, columns=val_feat.columns,
    )

    train_ds = create_dataset(train_s, train_tgt, seq_len)
    val_ds = create_dataset(val_s, val_tgt, seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        return float("inf")

    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=use_cuda, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=use_cuda, num_workers=0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    import copy
    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            pred_stack = torch.stack([preds["ret_1d"][:, 0], preds["ret_5d"][:, 0]], dim=1)
            loss = criterion(pred_stack, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                pred_stack = torch.stack([preds["ret_1d"][:, 0], preds["ret_5d"][:, 0]], dim=1)
                loss = criterion(pred_stack, y)
                val_sum += loss.item() * x.size(0)
                val_n += x.size(0)

        avg_val = val_sum / max(val_n, 1)
        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(
    backbone: str,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    folds: list[dict],
):
    """Create an Optuna objective that evaluates avg Sharpe across folds."""
    n_features = features.shape[1]

    def objective(trial: optuna.Trial) -> float:
        suggest_fn = SUGGEST_FN.get(backbone)
        if suggest_fn is None:
            raise ValueError(f"No suggest function for backbone '{backbone}'")

        try:
            hp = suggest_fn(trial)
        except optuna.TrialPruned:
            raise

        seq_len = hp.get("seq_len", 60)
        fold_sharpes = []

        for i, fold in enumerate(folds):
            train_feat, val_feat, test_feat = split_data(features, fold)
            train_tgt, val_tgt, test_tgt = split_data(targets, fold)

            if (len(train_feat) < seq_len or len(val_feat) < seq_len
                    or len(test_feat) < seq_len):
                continue

            scaler = StandardScaler()
            scaler.fit(train_feat.values)

            if is_gbm(backbone):
                # GBM path — contiguous windowing to avoid spanning punched-out gaps
                train_s = scaler.transform(train_feat.values)
                test_s = scaler.transform(test_feat.values)

                # Training: per-segment windows
                segments = find_contiguous_segments(train_feat.index)
                X_parts, y_parts = [], []
                for seg_start, seg_end in segments:
                    seg = train_s[seg_start:seg_end]
                    seg_tgt = train_tgt.iloc[seg_start:seg_end]
                    if len(seg) <= seq_len:
                        continue
                    X = np.array([seg[j:j+seq_len].ravel() for j in range(len(seg) - seq_len)])
                    y = seg_tgt.values[seq_len:][:len(X)]
                    X_parts.append(X[:len(y)])
                    y_parts.append(y)
                X_train = np.concatenate(X_parts) if X_parts else None
                y_train = np.concatenate(y_parts) if y_parts else None

                # Test: single contiguous window
                def _windows(feat, tgt, sl):
                    if feat.shape[0] <= sl:
                        return None, None
                    X = np.array([feat[j:j+sl].ravel() for j in range(feat.shape[0] - sl)])
                    y = tgt.values[sl:][:len(X)]
                    return X[:len(y)], y
                X_test, y_test = _windows(test_s, test_tgt, seq_len)

                if X_train is None or X_test is None or len(X_test) == 0:
                    continue

                model = _create_gbm_with_hp(backbone, hp)
                try:
                    model.fit(X_train, y_train)
                except Exception:
                    # GPU fallback handled in GBMWrapper
                    continue

                preds = model.predict(X_test)
                preds_1d = preds[:, 0]
                actuals_1d = y_test[:, 0]
            else:
                # Neural path
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = _create_model_with_hp(backbone, n_features, hp).to(device)

                _train_neural_fold_with_hp(
                    model, train_feat, train_tgt, val_feat, val_tgt, scaler, hp,
                )

                test_s = pd.DataFrame(
                    scaler.transform(test_feat.values),
                    index=test_feat.index, columns=test_feat.columns,
                )
                test_ds = create_dataset(test_s, test_tgt, seq_len)
                if len(test_ds) == 0:
                    continue

                test_loader = torch.utils.data.DataLoader(
                    test_ds, batch_size=256, shuffle=False,
                )

                model.eval()
                preds_list, actuals_list = [], []
                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(device)
                        out = model(x)
                        preds_list.append(out["ret_1d"][:, 0].cpu().numpy())
                        actuals_list.append(y[:, 0].numpy())

                preds_1d = np.concatenate(preds_list)
                actuals_1d = np.concatenate(actuals_list)

            # Strategy returns
            strategy_returns = np.sign(preds_1d) * actuals_1d
            fold_sharpe = sharpe_ratio(strategy_returns)
            fold_sharpes.append(fold_sharpe)

            # Optuna pruning: report intermediate result per fold
            trial.report(np.mean(fold_sharpes), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_sharpes:
            return -999.0

        return float(np.mean(fold_sharpes))

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep")
    parser.add_argument("--backbone", required=True, choices=list(BACKBONE_REGISTRY.keys()))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick=2 folds (fast), full=7 folds (thorough)")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name (default: sweep_{backbone})")
    parser.add_argument("--list-backbones", action="store_true")
    args = parser.parse_args()

    if args.list_backbones:
        for name, desc in BACKBONE_REGISTRY.items():
            print(f"  {name:15s}  {desc}")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Validate splits
    violations = validate_purge_embargo()
    if violations:
        raise RuntimeError(f"Purge/embargo violations: {violations}")

    # Download + prepare data
    print("Preparing data...")
    all_pairs = download_all_pairs()
    macro_data = download_macro_signals()
    features = compute_all_features(all_pairs, macro_data)
    targets = compute_targets(all_pairs["EURUSD=X"])
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    print(f"Features: {features.shape}  |  Targets: {targets.shape}")

    # Select folds
    folds = QUICK_FOLDS if args.mode == "quick" else FOLDS
    print(f"Mode: {args.mode}  |  Folds: {len(folds)}  |  Trials: {args.n_trials}")

    # Create Optuna study with persistent SQLite storage
    SWEEP_DIR.mkdir(exist_ok=True)
    study_name = args.study_name or f"sweep_{args.backbone}"
    storage = f"sqlite:///{SWEEP_DIR / 'sweeps.db'}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    objective = create_objective(args.backbone, features, targets, folds)

    print(f"\nStarting sweep: {args.backbone} ({BACKBONE_REGISTRY[args.backbone]})")
    print(f"Study: {study_name}  |  Previous trials: {len(study.trials)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {args.backbone}")
    print(f"{'='*60}")
    print(f"  Total trials:    {len(study.trials)}")
    print(f"  Best avg Sharpe: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Save results
    result = {
        "backbone": args.backbone,
        "mode": args.mode,
        "n_folds": len(folds),
        "n_trials": len(study.trials),
        "best_sharpe": study.best_value,
        "best_params": study.best_params,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "top_10": [],
    }

    # Top 10 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    for t in sorted_trials[:10]:
        result["top_10"].append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
        })

    result_path = SWEEP_DIR / f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Results saved to {result_path}")

    # Print top 5
    print(f"\nTop 5 trials:")
    for t in sorted_trials[:5]:
        print(f"  #{t.number}: Sharpe={t.value:.4f}  params={t.params}")


if __name__ == "__main__":
    main()
