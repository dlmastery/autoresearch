"""Baseline runner: walk-forward evaluation of CurrencyLFM across 7 folds.

Produces an average Sharpe ratio and per-fold details, saving results to
``baseline_results.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.download import download_all_pairs, download_macro_signals
from .data.features import compute_all_features, compute_targets
from .data.splits import FOLDS, split_data, validate_purge_embargo
from .evaluation.metrics import (
    sharpe_ratio, average_sharpe_across_folds, weighted_sharpe_across_folds,
    trading_report, information_coefficient, probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
)
from .model.backbone import create_model, BACKBONE_REGISTRY, DEFAULT_BACKBONE, is_gbm, GBMWrapper, get_seq_len
from .model.train import create_dataset, find_contiguous_segments, train_one_fold, SEQ_LEN

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "baseline_results.json"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "baseline_checkpoint.json"


def _run_neural_fold(
    backbone, train_feat, train_tgt, val_feat, val_tgt,
    test_feat, test_tgt, scaler, seq_len, n_features, epochs,
):
    """Train and evaluate a neural model on one fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        backbone=backbone, n_input_features=n_features,
        seq_len=seq_len, freeze_backbone=True,
    ).to(device)

    result = train_one_fold(
        model, train_feat, train_tgt, val_feat, val_tgt,
        scaler=scaler, epochs=epochs, seq_len=seq_len,
    )
    best_val_loss = result["best_val_loss"]

    test_feat_scaled = pd.DataFrame(
        scaler.transform(test_feat.values),
        index=test_feat.index, columns=test_feat.columns,
    )
    test_ds = create_dataset(test_feat_scaled, test_tgt, seq_len)
    if len(test_ds) == 0:
        return None, None, 0.0

    from torch.utils.data import DataLoader as DL
    test_loader = DL(test_ds, batch_size=256, shuffle=False)

    model.eval()
    preds_list, actuals_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds_list.append(out["ret_1d"][:, 0].cpu().numpy())
            actuals_list.append(y[:, 0].numpy())

    return np.concatenate(preds_list), np.concatenate(actuals_list), best_val_loss


def _run_gbm_fold(
    backbone, train_feat, train_tgt, test_feat, test_tgt,
    scaler, seq_len, n_features,
):
    """Train and evaluate a GBM model on one fold.

    GBMs use flattened sliding windows as features (seq_len * n_features).
    Training data may be non-contiguous (hole-punched); windows are created
    per contiguous segment to avoid spanning date gaps.
    """
    # Scale features
    train_s = scaler.transform(train_feat.values)
    test_s = scaler.transform(test_feat.values)

    # --- Training windows: per-contiguous-segment to avoid gap-spanning ---
    segments = find_contiguous_segments(train_feat.index)
    X_parts, y_parts = [], []
    for seg_start, seg_end in segments:
        seg_feat = train_s[seg_start:seg_end]
        seg_tgt = train_tgt.iloc[seg_start:seg_end]
        n = len(seg_feat)
        if n <= seq_len:
            continue
        X = np.array([seg_feat[i:i + seq_len].ravel() for i in range(n - seq_len)])
        y = seg_tgt.values[seq_len:][:len(X)]
        X_parts.append(X[:len(y)])
        y_parts.append(y)

    if not X_parts:
        return None, None, 0.0
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)

    # --- Test windows: single contiguous block (no gaps) ---
    n_test = test_s.shape[0]
    if n_test <= seq_len:
        return None, None, 0.0
    X_test = np.array([test_s[i:i + seq_len].ravel() for i in range(n_test - seq_len)])
    y_test = test_tgt.values[seq_len:][:len(X_test)]

    if len(X_test) == 0:
        return None, None, 0.0

    model = create_model(backbone, n_features, seq_len=seq_len)
    assert isinstance(model, GBMWrapper)

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  GBM GPU failed ({e}), falling back to CPU")
        model = GBMWrapper(backbone, n_targets=2)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds_1d = preds[:, 0]  # ret_1d predictions
    actuals_1d = y_test[:, 0]

    return preds_1d, actuals_1d, 0.0


def run_baseline(epochs: int = 20, seq_len: int | None = None, backbone: str = DEFAULT_BACKBONE) -> dict:
    """Run walk-forward baseline evaluation.

    Parameters
    ----------
    epochs : int
        Training epochs per fold.
    seq_len : int or None
        Sequence length for the sliding-window dataset.  If None, uses the
        per-backbone default (60 for LFM2.5, 10 for all others).

    Returns
    -------
    dict
        ``{"avg_sharpe": float, "folds": [...], "n_features": int}``
    """
    # Resolve per-backbone seq_len
    if seq_len is None:
        seq_len = get_seq_len(backbone)
    print(f"Sequence length: {seq_len} (backbone: {backbone})")

    # ------------------------------------------------------------------
    # 0. Validate purge + embargo constraints
    # ------------------------------------------------------------------
    violations = validate_purge_embargo()
    if violations:
        print("PURGE/EMBARGO VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        raise RuntimeError("Fix fold definitions before running baseline")
    print("Purge + embargo validation: PASSED")

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    all_pairs = download_all_pairs()
    macro_data = download_macro_signals()

    # ------------------------------------------------------------------
    # 2. Compute features and targets
    # ------------------------------------------------------------------
    features = compute_all_features(all_pairs, macro_data)
    targets = compute_targets(all_pairs["EURUSD=X"])

    # ------------------------------------------------------------------
    # 3. Align features and targets on common index
    # ------------------------------------------------------------------
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    n_features = features.shape[1]
    print(f"Features shape: {features.shape}  |  Targets shape: {targets.shape}")

    # ------------------------------------------------------------------
    # 4. Walk-forward evaluation across 7 folds (with checkpoint resume)
    # ------------------------------------------------------------------
    fold_details: list[dict] = []
    fold_returns_list: list[np.ndarray] = []
    fold_train_sizes: list[int] = []
    completed_folds: set[str] = set()

    # Resume from checkpoint if one exists and matches current backbone
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        if ckpt.get("backbone") == backbone:
            fold_details = ckpt.get("folds", [])
            fold_returns_list = [np.array(r) for r in ckpt.get("fold_returns", [])]
            fold_train_sizes = ckpt.get("fold_train_sizes", [])
            # Validate checkpoint consistency
            if len(fold_returns_list) != len(fold_train_sizes):
                print(f"  Checkpoint corrupt (returns={len(fold_returns_list)}, "
                      f"train_sizes={len(fold_train_sizes)}), starting fresh")
                fold_details, fold_returns_list, fold_train_sizes = [], [], []
            else:
                completed_folds = {d["fold"] for d in fold_details}
                print(f"Resuming from checkpoint: {len(completed_folds)} folds already done")
        else:
            print(f"  Checkpoint is for backbone '{ckpt.get('backbone')}', "
                  f"current is '{backbone}' — starting fresh")
            CHECKPOINT_PATH.unlink()

    for fold in FOLDS:
        fold_name = fold["name"]
        if fold_name in completed_folds:
            print(f"\nSkipping {fold_name} (already in checkpoint)")
            continue

        print(f"\n{'='*60}")
        print(f"Fold: {fold_name}  |  Regime: {fold['regime']}")
        print(f"{'='*60}")

        # Split features and targets
        train_feat, val_feat, test_feat = split_data(features, fold)
        train_tgt, val_tgt, test_tgt = split_data(targets, fold)

        print(f"  Split sizes: train={len(train_feat)}, val={len(val_feat)}, test={len(test_feat)}")

        # Skip fold if any split has fewer rows than seq_len
        if (
            len(train_feat) < seq_len
            or len(val_feat) < seq_len
            or len(test_feat) < seq_len
        ):
            print(f"  Skipping {fold_name}: insufficient data")
            fold_details.append({
                "fold": fold_name,
                "regime": fold["regime"],
                "skipped": True,
                "reason": "insufficient data",
            })
            continue

        if len(train_feat) < 500:
            print(f"  WARNING: small training set ({len(train_feat)} rows) — results may be noisy")

        # ----------------------------------------------------------
        # Branching: Neural vs GBM training paths
        # ----------------------------------------------------------
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(train_feat.values)

        if is_gbm(backbone):
            preds_arr, actuals_arr, best_val_loss = _run_gbm_fold(
                backbone, train_feat, train_tgt, test_feat, test_tgt,
                scaler, seq_len, n_features,
            )
        else:
            preds_arr, actuals_arr, best_val_loss = _run_neural_fold(
                backbone, train_feat, train_tgt, val_feat, val_tgt,
                test_feat, test_tgt, scaler, seq_len, n_features, epochs,
            )

        if preds_arr is None:
            print(f"  Skipping {fold_name}: empty test dataset after windowing")
            fold_details.append({
                "fold": fold_name, "regime": fold["regime"],
                "skipped": True, "reason": "empty test dataset",
            })
            continue

        # d. Strategy returns = sign(predicted) * actual
        strategy_returns = np.sign(preds_arr) * actuals_arr

        # e. Compute full trading report + information coefficient
        report = trading_report(strategy_returns, initial_capital=1000.0)
        ic = information_coefficient(preds_arr, actuals_arr)
        fold_returns_list.append(strategy_returns)
        fold_train_sizes.append(len(train_feat))

        print(f"  Test samples: {len(strategy_returns)}  |  "
              f"Sharpe: {report['sharpe']:.4f}  |  "
              f"Sortino: {report['sortino']:.4f}  |  "
              f"PSR: {report['psr']:.4f}  |  "
              f"Return: {report['total_return_pct']:+.2f}%  |  "
              f"MaxDD: {report['max_drawdown_pct']:.2f}%  |  "
              f"IC: {ic['ic_spearman']:.4f} (p={ic['ic_pvalue']:.4f})  |  "
              f"Hit: {ic['hit_rate']:.1f}%  |  "
              f"Omega: {report['omega_ratio']:.3f}")

        fold_details.append({
            "fold": fold_name,
            "regime": fold["regime"],
            "skipped": False,
            "best_val_loss": best_val_loss,
            "train_samples": len(train_feat),
            "val_samples": len(val_feat),
            "test_samples": len(strategy_returns),
            **report,
            **ic,
        })

        # Checkpoint after each fold for crash recovery
        ckpt_data = {
            "backbone": backbone,
            "folds": fold_details,
            "fold_returns": [r.tolist() for r in fold_returns_list],
            "fold_train_sizes": fold_train_sizes,
            "n_features": n_features,
        }
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(ckpt_data, f, indent=2, default=str)
        print(f"  Checkpoint saved ({len(fold_details)} folds done)")

    # ------------------------------------------------------------------
    # 5. Average Sharpe across folds
    # ------------------------------------------------------------------
    if fold_returns_list:
        avg_sharpe = average_sharpe_across_folds(fold_returns_list)
        wtd_sharpe = weighted_sharpe_across_folds(fold_returns_list, fold_train_sizes)
    else:
        avg_sharpe = 0.0
        wtd_sharpe = 0.0

    # ------------------------------------------------------------------
    # 6. Print summary and save results
    # ------------------------------------------------------------------
    # Combined equity across all folds
    if fold_returns_list:
        all_returns = np.concatenate(fold_returns_list)
        overall_report = trading_report(all_returns, initial_capital=1000.0)
    else:
        overall_report = trading_report(np.array([]), initial_capital=1000.0)

    r = overall_report

    # Compute aggregate PSR and DSR (n_trials=1 for baseline, more for optimizer)
    overall_psr = r["psr"]
    overall_dsr = deflated_sharpe_ratio(all_returns, n_trials=1) if fold_returns_list else 0.0

    # Aggregate IC across all folds
    all_preds = []
    all_actuals = []
    for fd in fold_details:
        if not fd.get("skipped", True) and "ic_spearman" in fd:
            all_preds.append(fd.get("ic_spearman", 0))
    avg_ic = float(np.mean(all_preds)) if all_preds else 0.0

    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY ($1000 initial capital)")
    print(f"{'='*60}")
    print(f"  Folds evaluated:     {len(fold_returns_list)} / {len(FOLDS)}")
    print(f"  N features:          {n_features}")
    print(f"  ---  Risk-Adjusted Performance  ---")
    print(f"  Average Sharpe:      {avg_sharpe:.4f}")
    print(f"  Weighted Sharpe:     {wtd_sharpe:.4f}  (by training size)")
    print(f"  Overall Sharpe:      {r['sharpe']:.4f}")
    print(f"  Sortino Ratio:       {r['sortino']:.4f}")
    print(f"  PSR (vs 0):          {overall_psr:.4f}  {'*' if overall_psr > 0.95 else '(not significant)'}")
    print(f"  DSR (1 trial):       {overall_dsr:.4f}  {'*' if overall_dsr > 0.95 else '(not significant)'}")
    print(f"  Calmar Ratio:        {r['calmar_ratio']:.3f}")
    print(f"  Omega Ratio:         {r['omega_ratio']:.3f}")
    print(f"  ---  Predictive Quality  ---")
    print(f"  Avg IC (Spearman):   {avg_ic:.4f}")
    print(f"  ---  Returns  ---")
    print(f"  Total Return:        {r['total_return_pct']:+.2f}%")
    print(f"  Annualized Return:   {r['annualized_return_pct']:+.2f}%")
    print(f"  Final Equity:        ${r['final_equity']:.2f}")
    print(f"  Profit:              ${r['profit']:.2f}")
    print(f"  Mean Daily:          {r['mean_daily_bps']:+.2f} bps")
    print(f"  Median Daily:        {r['median_daily_bps']:+.2f} bps")
    print(f"  Daily Vol:           {r['daily_vol_bps']:.2f} bps")
    print(f"  ---  Risk & Tail  ---")
    print(f"  Max Drawdown:        {r['max_drawdown_pct']:.2f}%")
    print(f"  VaR 95%:             {r['var_95']:.4f}%")
    print(f"  CVaR 95%:            {r['cvar_95']:.4f}%")
    print(f"  VaR 99%:             {r['var_99']:.4f}%")
    print(f"  CVaR 99%:            {r['cvar_99']:.4f}%")
    print(f"  Skewness:            {r['skewness']:.4f}")
    print(f"  Excess Kurtosis:     {r['excess_kurtosis']:.4f}")
    print(f"  Tail Ratio:          {r['tail_ratio']:.3f}")
    print(f"  ---  Trade Stats  ---")
    print(f"  Win Rate:            {r['win_rate']:.1f}%")
    print(f"  Profit Factor:       {r['profit_factor']:.3f}")
    print(f"  Avg Win:             {r['avg_win']:.4f} bps")
    print(f"  Avg Loss:            {r['avg_loss']:.4f} bps")
    print(f"  Max Consec Wins:     {r['max_consec_wins']}")
    print(f"  Max Consec Losses:   {r['max_consec_losses']}")
    print(f"  Recovery Factor:     {r['recovery_factor']:.3f}")

    results = {
        "avg_sharpe": avg_sharpe,
        "weighted_sharpe": wtd_sharpe,
        "psr": overall_psr,
        "dsr": overall_dsr,
        "avg_ic": avg_ic,
        "overall": overall_report,
        "folds": fold_details,
        "n_features": n_features,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS_PATH}")

    # Clean up checkpoint after successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("  Checkpoint cleaned up")

    # ------------------------------------------------------------------
    # 7. Return results dict
    # ------------------------------------------------------------------
    return results


def main():
    """CLI entry point for baseline evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Baseline walk-forward evaluation")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE,
                        choices=list(BACKBONE_REGISTRY.keys()),
                        help=f"Model backbone (default: {DEFAULT_BACKBONE})")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"Backbone: {args.backbone} — {BACKBONE_REGISTRY[args.backbone]}")
    results = run_baseline(epochs=args.epochs, seq_len=args.seq_len, backbone=args.backbone)
    print(f"\nFinal avg Sharpe: {results['avg_sharpe']:.4f}")


if __name__ == "__main__":
    main()
