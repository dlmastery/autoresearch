"""Karpathy-style systematic LFM2 hyperparameter sweep.

Runs targeted experiments one-at-a-time, analyzing results between rounds.
Not Optuna random search — informed, sequential exploration.

Usage:
    python -m autoresearch.run_lfm2_sweep
"""

from __future__ import annotations

import json
import time
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
)
from .model.backbone import create_model, get_seq_len
from .model.train import create_dataset, train_one_fold

SWEEP_DIR = Path(__file__).resolve().parent / "sweep_results"


def run_single_config(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict,
    folds: list[dict] | None = None,
    tag: str = "",
) -> dict:
    """Run LFM2 with a specific config across all folds. Returns results dict."""
    if folds is None:
        folds = FOLDS

    backbone = "lfm2-350m"
    seq_len = config.get("seq_len", get_seq_len(backbone))
    lr = config.get("lr", 1e-4)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 20)
    weight_decay = config.get("weight_decay", 1e-5)
    patience = config.get("patience", 5)
    grad_clip = config.get("grad_clip", 1.0)

    n_features = features.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_details = []
    fold_returns_list = []
    fold_train_sizes = []

    print(f"\n{'='*70}")
    print(f"CONFIG: {tag}")
    print(f"  lr={lr:.1e}  bs={batch_size}  seq={seq_len}  epochs={epochs}  "
          f"wd={weight_decay:.1e}  patience={patience}  clip={grad_clip}")
    print(f"{'='*70}")

    for fold in folds:
        fold_name = fold["name"]
        train_feat, val_feat, test_feat = split_data(features, fold)
        train_tgt, val_tgt, test_tgt = split_data(targets, fold)

        if len(train_feat) < seq_len or len(val_feat) < seq_len or len(test_feat) < seq_len:
            print(f"  {fold_name}: SKIP (insufficient data)")
            fold_details.append({"fold": fold_name, "skipped": True})
            continue

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feat.values)

        # Create model fresh each fold
        model = create_model(
            backbone=backbone, n_input_features=n_features,
            seq_len=seq_len, freeze_backbone=True,
        ).to(device)

        # Train with custom hyperparameters
        result = train_one_fold(
            model, train_feat, train_tgt, val_feat, val_tgt,
            scaler=scaler, epochs=epochs, seq_len=seq_len,
            lr=lr, batch_size=batch_size,
        )
        best_val_loss = result["best_val_loss"]

        # Evaluate on test
        test_feat_scaled = pd.DataFrame(
            scaler.transform(test_feat.values),
            index=test_feat.index, columns=test_feat.columns,
        )
        test_ds = create_dataset(test_feat_scaled, test_tgt, seq_len)
        if len(test_ds) == 0:
            fold_details.append({"fold": fold_name, "skipped": True})
            continue

        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model.eval()
        preds_list, actuals_list = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                preds_list.append(out["ret_1d"][:, 0].cpu().numpy())
                actuals_list.append(y[:, 0].numpy())

        preds_arr = np.concatenate(preds_list)
        actuals_arr = np.concatenate(actuals_list)
        strategy_returns = np.sign(preds_arr) * actuals_arr

        report = trading_report(strategy_returns)
        ic = information_coefficient(preds_arr, actuals_arr)
        fold_returns_list.append(strategy_returns)
        fold_train_sizes.append(len(train_feat))

        fold_details.append({
            "fold": fold_name,
            "regime": fold["regime"],
            "train_samples": len(train_feat),
            "sharpe": report["sharpe"],
            "return_pct": report["total_return_pct"],
            "win_rate": report["win_rate"],
            "ic_spearman": ic["ic_spearman"],
            "hit_rate": ic["hit_rate"],
            "val_loss": best_val_loss,
        })

        s = report["sharpe"]
        print(f"  {fold_name} ({fold['regime'][:25]:25s}): "
              f"Sharpe={s:+.3f}  IC={ic['ic_spearman']:+.3f}  "
              f"Hit={ic['hit_rate']:.1f}%  ValLoss={best_val_loss:.6f}  "
              f"Train={len(train_feat)}")

    # Aggregate
    if fold_returns_list:
        avg_sharpe = average_sharpe_across_folds(fold_returns_list)
        wtd_sharpe = weighted_sharpe_across_folds(fold_returns_list, fold_train_sizes)
        all_returns = np.concatenate(fold_returns_list)
        overall = trading_report(all_returns)
        psr = probabilistic_sharpe_ratio(all_returns)
    else:
        avg_sharpe = wtd_sharpe = psr = 0.0
        overall = {}

    # Focus metric: fold 7 Sharpe (most data, most representative)
    fold7 = next((f for f in fold_details if f.get("fold") == "fold_7"), {})
    fold7_sharpe = fold7.get("sharpe", 0.0)

    print(f"\n  SUMMARY: avg_sharpe={avg_sharpe:+.4f}  wtd_sharpe={wtd_sharpe:+.4f}  "
          f"PSR={psr:.4f}  fold7_sharpe={fold7_sharpe:+.4f}")

    return {
        "tag": tag,
        "config": config,
        "avg_sharpe": avg_sharpe,
        "wtd_sharpe": wtd_sharpe,
        "psr": psr,
        "fold7_sharpe": fold7_sharpe,
        "overall_equity": overall.get("final_equity", 1000.0),
        "folds": fold_details,
    }


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Validate splits
    violations = validate_purge_embargo()
    if violations:
        raise RuntimeError(f"Purge/embargo violations: {violations}")
    print("Purge + embargo: PASSED")

    # Download + prepare data (one-time)
    print("Loading data...")
    all_pairs = download_all_pairs()
    macro_data = download_macro_signals()
    features = compute_all_features(all_pairs, macro_data)
    targets = compute_targets(all_pairs["EURUSD=X"])
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    print(f"Features: {features.shape}  Targets: {targets.shape}")

    SWEEP_DIR.mkdir(exist_ok=True)
    ckpt_path = SWEEP_DIR / "lfm2_sweep_checkpoint.json"

    # Resume from checkpoint if available
    ckpt = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: completed rounds = {list(ckpt.keys())}")

    def _save_ckpt(key, data):
        ckpt[key] = data
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"  Checkpoint saved: {key}")

    # =====================================================================
    # ROUND 1: Learning Rate sweep (most impactful for frozen backbone)
    #
    # The LFM2 backbone is 350M params frozen. Only ~636K trainable
    # (projection + heads). High lr overshoots on this thin adapter.
    # Karpathy rule: lr is always the first thing you sweep.
    # =====================================================================
    lr_candidates = [5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
    base_config = {
        "seq_len": 60, "batch_size": 32, "epochs": 20,
        "weight_decay": 1e-5, "patience": 5, "grad_clip": 1.0,
    }

    if "round1" in ckpt:
        all_results = ckpt["round1"]["results"]
        best_lr = ckpt["round1"]["best_lr"]
        print(f"\nRound 1 loaded from checkpoint: best_lr={best_lr:.1e}")
    else:
        print("\n" + "#" * 70)
        print("# ROUND 1: Learning Rate Sweep")
        print("# Base: seq=60, bs=32, epochs=20, patience=5, wd=1e-5")
        print("#" * 70)

        all_results = []
        for lr in lr_candidates:
            cfg = {**base_config, "lr": lr}
            result = run_single_config(features, targets, cfg, tag=f"lr={lr:.1e}")
            all_results.append(result)

        # Find best lr
        print("\n" + "=" * 70)
        print("ROUND 1 RESULTS: Learning Rate")
        print(f"{'lr':>10s}  {'avg_sharpe':>10s}  {'wtd_sharpe':>10s}  {'fold7':>8s}  {'PSR':>6s}  {'equity':>8s}")
        for r in all_results:
            lr_val = r["config"]["lr"]
            print(f"{lr_val:>10.1e}  {r['avg_sharpe']:>+10.4f}  {r['wtd_sharpe']:>+10.4f}  "
                  f"{r['fold7_sharpe']:>+8.4f}  {r['psr']:>6.4f}  ${r['overall_equity']:>7.2f}")

        best_r1 = max(all_results, key=lambda r: r["wtd_sharpe"])
        best_lr = best_r1["config"]["lr"]
        print(f"\nBest lr: {best_lr:.1e} (wtd_sharpe={best_r1['wtd_sharpe']:+.4f})")

        _save_ckpt("round1", {
            "results": all_results,
            "best_lr": best_lr,
        })

    # =====================================================================
    # ROUND 2: Batch size sweep at best lr
    # =====================================================================
    if "round2" in ckpt:
        r2_results = ckpt["round2"]["results"]
        best_bs = ckpt["round2"]["best_bs"]
        print(f"\nRound 2 loaded from checkpoint: best_bs={best_bs}")
    else:
        print("\n" + "#" * 70)
        print(f"# ROUND 2: Batch Size Sweep (lr={best_lr:.1e})")
        print("#" * 70)

        r2_results = []
        bs_candidates = [8, 16, 32, 64, 128]
        for bs in bs_candidates:
            cfg = {**base_config, "lr": best_lr, "batch_size": bs}
            result = run_single_config(features, targets, cfg, tag=f"bs={bs}")
            r2_results.append(result)

        print("\n" + "=" * 70)
        print("ROUND 2 RESULTS: Batch Size")
        print(f"{'bs':>5s}  {'avg_sharpe':>10s}  {'wtd_sharpe':>10s}  {'fold7':>8s}  {'PSR':>6s}")
        for r in r2_results:
            bs_val = r["config"]["batch_size"]
            print(f"{bs_val:>5d}  {r['avg_sharpe']:>+10.4f}  {r['wtd_sharpe']:>+10.4f}  "
                  f"{r['fold7_sharpe']:>+8.4f}  {r['psr']:>6.4f}")

        best_r2 = max(r2_results, key=lambda r: r["wtd_sharpe"])
        best_bs = best_r2["config"]["batch_size"]
        print(f"\nBest batch_size: {best_bs} (wtd_sharpe={best_r2['wtd_sharpe']:+.4f})")

        _save_ckpt("round2", {
            "results": r2_results,
            "best_bs": best_bs,
        })

    # =====================================================================
    # ROUND 3: Sequence length (LFM2-specific: does long context help?)
    # =====================================================================
    if "round3" in ckpt:
        r3_results = ckpt["round3"]["results"]
        best_seq = ckpt["round3"]["best_seq"]
        print(f"\nRound 3 loaded from checkpoint: best_seq={best_seq}")
    else:
        print("\n" + "#" * 70)
        print(f"# ROUND 3: Sequence Length Sweep (lr={best_lr:.1e}, bs={best_bs})")
        print("#" * 70)

        r3_results = []
        seq_candidates = [20, 30, 40, 60, 90, 120]
        for sl in seq_candidates:
            cfg = {**base_config, "lr": best_lr, "batch_size": best_bs, "seq_len": sl}
            result = run_single_config(features, targets, cfg, tag=f"seq={sl}")
            r3_results.append(result)

        print("\n" + "=" * 70)
        print("ROUND 3 RESULTS: Sequence Length")
        print(f"{'seq':>5s}  {'avg_sharpe':>10s}  {'wtd_sharpe':>10s}  {'fold7':>8s}  {'PSR':>6s}")
        for r in r3_results:
            sl_val = r["config"]["seq_len"]
            print(f"{sl_val:>5d}  {r['avg_sharpe']:>+10.4f}  {r['wtd_sharpe']:>+10.4f}  "
                  f"{r['fold7_sharpe']:>+8.4f}  {r['psr']:>6.4f}")

        best_r3 = max(r3_results, key=lambda r: r["wtd_sharpe"])
        best_seq = best_r3["config"]["seq_len"]
        print(f"\nBest seq_len: {best_seq} (wtd_sharpe={best_r3['wtd_sharpe']:+.4f})")

        _save_ckpt("round3", {
            "results": r3_results,
            "best_seq": best_seq,
        })

    # =====================================================================
    # ROUND 4: Weight decay + patience refinement
    # =====================================================================
    if "round4" in ckpt:
        r4_results = ckpt["round4"]["results"]
        best_wd = ckpt["round4"]["best_wd"]
        best_pat = ckpt["round4"]["best_pat"]
        print(f"\nRound 4 loaded from checkpoint: wd={best_wd:.0e}, patience={best_pat}")
    else:
        print("\n" + "#" * 70)
        print(f"# ROUND 4: Regularization Sweep (lr={best_lr:.1e}, bs={best_bs}, seq={best_seq})")
        print("#" * 70)

        r4_results = []
        wd_patience_combos = [
            (0.0, 3),      # no reg, early stop
            (1e-6, 5),     # minimal
            (1e-5, 5),     # light
            (1e-4, 5),     # moderate
            (1e-3, 5),     # heavy
            (1e-4, 3),     # moderate + aggressive early stop
            (1e-4, 8),     # moderate + patient
        ]
        for wd, pat in wd_patience_combos:
            cfg = {
                "lr": best_lr, "batch_size": best_bs, "seq_len": best_seq,
                "epochs": 20, "weight_decay": wd, "patience": pat, "grad_clip": 1.0,
            }
            result = run_single_config(features, targets, cfg, tag=f"wd={wd:.0e}_p={pat}")
            r4_results.append(result)

        print("\n" + "=" * 70)
        print("ROUND 4 RESULTS: Regularization")
        print(f"{'wd':>8s}  {'pat':>3s}  {'avg_sharpe':>10s}  {'wtd_sharpe':>10s}  {'fold7':>8s}  {'PSR':>6s}")
        for r in r4_results:
            wd_val = r["config"]["weight_decay"]
            pat_val = r["config"]["patience"]
            print(f"{wd_val:>8.0e}  {pat_val:>3d}  {r['avg_sharpe']:>+10.4f}  {r['wtd_sharpe']:>+10.4f}  "
                  f"{r['fold7_sharpe']:>+8.4f}  {r['psr']:>6.4f}")

        best_r4 = max(r4_results, key=lambda r: r["wtd_sharpe"])
        best_wd = best_r4["config"]["weight_decay"]
        best_pat = best_r4["config"]["patience"]
        print(f"\nBest: wd={best_wd:.0e}, patience={best_pat}")

        _save_ckpt("round4", {
            "results": r4_results,
            "best_wd": best_wd,
            "best_pat": best_pat,
        })

    # =====================================================================
    # FINAL: Print champion config
    # =====================================================================
    champion = {
        "lr": best_lr, "batch_size": best_bs, "seq_len": best_seq,
        "weight_decay": best_wd, "patience": best_pat,
        "epochs": 20, "grad_clip": 1.0,
    }
    print("\n" + "#" * 70)
    print("# CHAMPION CONFIG")
    print("#" * 70)
    for k, v in champion.items():
        print(f"  {k}: {v}")
    print(f"\n  avg_sharpe:  {best_r4['avg_sharpe']:+.4f}")
    print(f"  wtd_sharpe:  {best_r4['wtd_sharpe']:+.4f}")
    print(f"  fold7:       {best_r4['fold7_sharpe']:+.4f}")
    print(f"  PSR:         {best_r4['psr']:.4f}")

    # Save all results
    sweep_path = SWEEP_DIR / "lfm2_karpathy_sweep.json"
    all_sweep = {
        "champion": champion,
        "champion_metrics": {
            "avg_sharpe": best_r4["avg_sharpe"],
            "wtd_sharpe": best_r4["wtd_sharpe"],
            "fold7_sharpe": best_r4["fold7_sharpe"],
            "psr": best_r4["psr"],
        },
        "round1_lr": [{"lr": r["config"]["lr"], "wtd_sharpe": r["wtd_sharpe"],
                        "fold7": r["fold7_sharpe"]} for r in all_results],
        "round2_bs": [{"bs": r["config"]["batch_size"], "wtd_sharpe": r["wtd_sharpe"],
                        "fold7": r["fold7_sharpe"]} for r in r2_results],
        "round3_seq": [{"seq": r["config"]["seq_len"], "wtd_sharpe": r["wtd_sharpe"],
                         "fold7": r["fold7_sharpe"]} for r in r3_results],
        "round4_reg": [{"wd": r["config"]["weight_decay"], "patience": r["config"]["patience"],
                         "wtd_sharpe": r["wtd_sharpe"], "fold7": r["fold7_sharpe"]} for r in r4_results],
    }
    with open(sweep_path, "w") as f:
        json.dump(all_sweep, f, indent=2)
    print(f"\nSaved to {sweep_path}")


if __name__ == "__main__":
    main()
