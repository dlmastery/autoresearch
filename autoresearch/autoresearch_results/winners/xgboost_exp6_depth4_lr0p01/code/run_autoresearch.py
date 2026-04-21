"""Single-experiment executor for autoresearch.

The AGENT (Claude) drives the loop — reads results, decides what to try next.
This script just executes ONE experiment and logs the result.

Usage:
    python -m autoresearch.run_autoresearch --backbone lfm2-350m \
        --lr 3e-5 --batch-size 32 --seq-len 60 --epochs 20 \
        --weight-decay 1e-5 --patience 5 --grad-clip 1.0 \
        --description "lr=3e-5 baseline"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _pin_to_safe_cores(n_threads: int = 4):
    """Pin process to a small subset of P-cores to minimize CPU stress.

    GPU does the heavy compute; CPU is coordination only. Using fewer cores
    reduces thermal load and avoids failing E-cores (APIC 16,17,24,25 on
    this Intel 14th-gen HX showed WHEA Internal parity errors on 2026-04-15).

    Default: 4 P-core logical threads (even-numbered, avoid HT siblings)
    Override with AUTORESEARCH_USE_ALL_CORES=1 or AUTORESEARCH_N_THREADS=N.
    """
    if os.environ.get("AUTORESEARCH_USE_ALL_CORES"):
        return
    try:
        import psutil
        n = int(os.environ.get("AUTORESEARCH_N_THREADS", n_threads))
        proc = psutil.Process(os.getpid())
        logical = psutil.cpu_count(logical=True)
        if logical and logical >= 32:  # Intel hybrid
            # Even logical IDs 0,2,4,... are primary P-core threads (no HT
            # sibling contention). Use just the first N of those.
            safe_cores = [2 * i for i in range(min(n, 8))]
            proc.cpu_affinity(safe_cores)
            torch.set_num_threads(n)
            os.environ["OMP_NUM_THREADS"] = str(n)
            os.environ["MKL_NUM_THREADS"] = str(n)
            print(f"[CPU SAFETY] Pinned to {n} P-core threads {safe_cores} "
                  f"(WHEA errors on E-cores 16,17,24,25). "
                  f"Override with AUTORESEARCH_USE_ALL_CORES=1.")
    except ImportError:
        pass
    except Exception as e:
        print(f"[CPU SAFETY] Could not pin affinity: {e}")


# Pin at import time so every run benefits
_pin_to_safe_cores()
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .data.download import download_all_pairs, download_macro_signals
from .data.features import compute_all_features, compute_targets
from .data.splits import FOLDS, split_superfold, validate_purge_embargo, get_fold_dates
from .evaluation.metrics import (
    sharpe_ratio, trading_report, information_coefficient, classification_metrics,
    probabilistic_sharpe_ratio,
)
from .model.backbone import create_model, get_seq_len, BACKBONE_REGISTRY, is_gbm, GBMWrapper, predict_with_uncertainty
from .model.train import create_dataset, create_contiguous_datasets, find_contiguous_segments, train_one_fold

RESULTS_DIR = Path(__file__).resolve().parent / "autoresearch_results"


def _evaluate_per_window(
    model, scaler, data_feat, data_tgt, seq_len, device, window_type="test",
    n_mc_samples: int = 20,
):
    """Evaluate on each fold's window separately, aggregate with full metrics.

    For val/test windows on neural models, also computes per-prediction
    uncertainty via MC Dropout (Gal & Ghahramani, 2016):
      - aleatoric: learned data noise (heteroscedastic variance head)
      - epistemic: model uncertainty (MC Dropout variance of means)
      - confidence: sigmoid(-log(total_uncertainty))
    """
    all_preds, all_actuals, per_window = [], [], []
    all_aleatoric, all_epistemic, all_confidence = [], [], []
    use_mc = isinstance(model, torch.nn.Module) and window_type != "train"
    trade_rows = []  # Per-trade records for CSV logging

    if window_type == "train":
        data_scaled = pd.DataFrame(
            scaler.transform(data_feat.values),
            index=data_feat.index, columns=data_feat.columns,
        )
        concat_ds = create_contiguous_datasets(data_scaled, data_tgt, seq_len)
        if len(concat_ds) > 0:
            loader = DataLoader(concat_ds, batch_size=256, shuffle=False)
            model.eval()
            preds_list, actuals_list = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    out = model(x)
                    preds_list.append(out["ret_1d"][:, 0].cpu().numpy())
                    actuals_list.append(y[:, 0].numpy())
            preds = np.concatenate(preds_list)
            actuals = np.concatenate(actuals_list)
            returns = np.sign(preds) * actuals
            rpt = trading_report(returns)
            ic = information_coefficient(preds, actuals)
            cm = classification_metrics(preds, actuals)
            per_window.append({
                "fold": "all_train", "regime": "Training data",
                "sharpe": round(sharpe_ratio(returns), 4),
                "sortino": rpt["sortino"], "return_pct": rpt["total_return_pct"],
                "equity": rpt["final_equity"], "max_dd": rpt["max_drawdown_pct"],
                "win_rate": rpt["win_rate"], "profit_factor": rpt["profit_factor"],
                "ic": ic["ic_spearman"], "hit": ic["hit_rate"], "n": len(returns),
                "precision": cm["precision"], "recall": cm["recall"],
                "f1": cm["f1"], "f2": cm["f2"], "mcc": cm["mcc"],
                "accuracy": cm["accuracy"],
                "tp": cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
            })
            all_preds.append(preds)
            all_actuals.append(actuals)
    else:
        for fold in FOLDS:
            d = get_fold_dates(fold)
            w_start, w_end = (d["val_start"], d["val_end"]) if window_type == "val" else (d["test_start"], d["test_end"])
            wf = data_feat.loc[w_start:w_end]
            wt = data_tgt.loc[w_start:w_end]
            if len(wf) < seq_len + 1:
                per_window.append({"fold": fold["name"], "regime": fold["regime"], "sharpe": 0.0, "skipped": True, "n": 0})
                continue
            ws = pd.DataFrame(scaler.transform(wf.values), index=wf.index, columns=wf.columns)
            ds = create_dataset(ws, wt, seq_len)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=256, shuffle=False)

            # Collect predictions and uncertainties
            pl, al = [], []
            w_aleatoric, w_epistemic, w_confidence = [], [], []

            for x, y in loader:
                x = x.to(device)
                if use_mc:
                    unc = predict_with_uncertainty(model, x, n_mc_samples=n_mc_samples)
                    pl.append(unc["mean"].numpy())
                    w_aleatoric.append(unc["aleatoric"].numpy())
                    w_epistemic.append(unc["epistemic"].numpy())
                    w_confidence.append(unc["confidence"].numpy())
                else:
                    model.eval()
                    with torch.no_grad():
                        out = model(x)
                    pl.append(out["ret_1d"][:, 0].cpu().numpy())
                al.append(y[:, 0].numpy())

            preds = np.concatenate(pl)
            actuals = np.concatenate(al)
            returns = np.sign(preds) * actuals
            rpt = trading_report(returns)
            ic = information_coefficient(preds, actuals) if len(returns) > 3 else {"ic_spearman": 0.0, "hit_rate": 0.0}
            cm = classification_metrics(preds, actuals)

            window_entry = {
                "fold": fold["name"], "regime": fold["regime"],
                "sharpe": round(sharpe_ratio(returns), 4),
                "sortino": rpt["sortino"], "return_pct": rpt["total_return_pct"],
                "equity": rpt["final_equity"], "max_dd": rpt["max_drawdown_pct"],
                "win_rate": rpt["win_rate"], "profit_factor": rpt["profit_factor"],
                "ic": ic["ic_spearman"], "hit": ic["hit_rate"], "n": len(returns),
                "precision": cm["precision"], "recall": cm["recall"],
                "f1": cm["f1"], "f2": cm["f2"], "mcc": cm["mcc"],
                "accuracy": cm["accuracy"],
                "tp": cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
            }

            if use_mc and w_aleatoric:
                ale = np.concatenate(w_aleatoric)
                epi = np.concatenate(w_epistemic)
                conf = np.concatenate(w_confidence)
                window_entry["aleatoric_mean"] = round(float(ale.mean()), 6)
                window_entry["epistemic_mean"] = round(float(epi.mean()), 6)
                window_entry["confidence_mean"] = round(float(conf.mean()), 4)
                window_entry["aleatoric_std"] = round(float(ale.std()), 6)
                window_entry["epistemic_std"] = round(float(epi.std()), 6)
                all_aleatoric.append(ale)
                all_epistemic.append(epi)
                all_confidence.append(conf)

            # Per-trade CSV records — dates align with the target position (idx+seq_len-1)
            fold_dates = wf.index[seq_len - 1 : seq_len - 1 + len(preds)]
            cum_ret = np.cumprod(1.0 + returns) - 1.0
            for i, dt in enumerate(fold_dates):
                ale_i = float(ale[i]) if use_mc and w_aleatoric else None
                epi_i = float(epi[i]) if use_mc and w_aleatoric else None
                conf_i = float(conf[i]) if use_mc and w_aleatoric else None
                trade_rows.append({
                    "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "fold": fold["name"],
                    "regime": fold["regime"],
                    "prediction": float(preds[i]),
                    "pred_direction": int(np.sign(preds[i])) or 1,
                    "actual_return": float(actuals[i]),
                    "actual_direction": int(np.sign(actuals[i])) or 1,
                    "strategy_return": float(returns[i]),
                    "cumulative_return": float(cum_ret[i]),
                    "confidence": conf_i,
                    "aleatoric": ale_i,
                    "epistemic": epi_i,
                    "correct": int(np.sign(preds[i]) == np.sign(actuals[i])),
                    "pnl_bps": float(returns[i] * 10000),
                })

            per_window.append(window_entry)
            all_preds.append(preds)
            all_actuals.append(actuals)

    if all_preds:
        ap = np.concatenate(all_preds)
        aa = np.concatenate(all_actuals)
        ar = np.sign(ap) * aa
        rpt = trading_report(ar)
        ic = information_coefficient(ap, aa)
        cm = classification_metrics(ap, aa)
        result = {
            "sharpe": round(sharpe_ratio(ar), 4), "psr": round(probabilistic_sharpe_ratio(ar), 4),
            "equity": round(rpt["final_equity"], 2), "sortino": rpt["sortino"],
            "return_pct": rpt["total_return_pct"], "max_dd": rpt["max_drawdown_pct"],
            "win_rate": rpt["win_rate"], "profit_factor": rpt["profit_factor"],
            "ic": ic["ic_spearman"], "hit": ic["hit_rate"],
            "precision": cm["precision"], "recall": cm["recall"],
            "f1": cm["f1"], "f2": cm["f2"], "mcc": cm["mcc"],
            "accuracy": cm["accuracy"],
            "n": len(ar), "per_window": per_window,
        }
        if all_aleatoric:
            agg_ale = np.concatenate(all_aleatoric)
            agg_epi = np.concatenate(all_epistemic)
            agg_conf = np.concatenate(all_confidence)
            result["aleatoric_mean"] = round(float(agg_ale.mean()), 6)
            result["epistemic_mean"] = round(float(agg_epi.mean()), 6)
            result["confidence_mean"] = round(float(agg_conf.mean()), 4)
        result["trade_rows"] = trade_rows
        return result
    return {"sharpe": 0.0, "psr": 0.0, "equity": 1000.0, "sortino": 0.0,
            "return_pct": 0.0, "max_dd": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "ic": 0.0, "hit": 0.0, "n": 0, "per_window": per_window}


def run_single_experiment(backbone, config, description):
    """Load data, train, evaluate, log, print results. Returns result dict."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Write running signal for dashboard
    RESULTS_DIR.mkdir(exist_ok=True)
    running_path = RESULTS_DIR / "running.json"
    with open(running_path, "w") as f:
        json.dump({
            "backbone": backbone, "config": config,
            "description": description,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    try:
        return _run_experiment_inner(backbone, config, description)
    finally:
        # Clear running signal
        if running_path.exists():
            running_path.unlink()


def _set_seed(seed: int):
    """Set all random seeds for reproducible training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_experiment_inner(backbone, config, description):
    """Actual experiment execution."""
    seed = config.get("seed", None)
    if seed is not None:
        _set_seed(seed)

    violations = validate_purge_embargo()
    if violations:
        raise RuntimeError(f"Purge/embargo violations: {violations}")

    # Load data (cached)
    all_pairs = download_all_pairs()
    macro_data = download_macro_signals()
    features = compute_all_features(all_pairs, macro_data)
    targets = compute_targets(all_pairs["EURUSD=X"])
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    train_feat, val_feat, test_feat = split_superfold(features)
    train_tgt, val_tgt, test_tgt = split_superfold(targets)

    seq_len = config["seq_len"]
    n_features = train_feat.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    scaler = StandardScaler()
    scaler.fit(train_feat.values)

    if is_gbm(backbone):
        # Route CLI-provided GBM hyperparameters through the config
        gbm_hp = {k: v for k, v in {
            "n_estimators": config.get("n_estimators"),
            "max_depth": config.get("max_depth"),
            "learning_rate": config.get("gbm_lr"),
            "subsample": config.get("subsample"),
            "colsample_bytree": config.get("colsample_bytree"),
            "reg_lambda": config.get("reg_lambda"),
            "reg_alpha": config.get("reg_alpha"),
            "min_child_weight": config.get("min_child_weight"),
            "gamma": config.get("gamma"),
            "num_leaves": config.get("num_leaves"),
            "feature_fraction": config.get("feature_fraction"),
            "bagging_fraction": config.get("bagging_fraction"),
            "min_data_in_leaf": config.get("min_data_in_leaf"),
            "iterations": config.get("iterations"),
            "depth": config.get("depth"),
            "l2_leaf_reg": config.get("l2_leaf_reg"),
            "random_strength": config.get("random_strength"),
            "bagging_temperature": config.get("bagging_temperature"),
            "bootstrap_type": config.get("bootstrap_type"),
        }.items() if v is not None}
        # Wire --seed through as GBM random_state (XGBoost/LightGBM/CatBoost all accept it)
        if config.get("seed") is not None:
            gbm_hp["random_state"] = config["seed"]
        model = create_model(backbone, n_features, seq_len=seq_len,
                              gbm_hp_overrides=gbm_hp or None)
        train_s = scaler.transform(train_feat.values)
        # Contiguous-segment windowing to avoid spanning punched-out gaps
        segments = find_contiguous_segments(train_feat.index)
        X_parts, y_parts = [], []
        for seg_start, seg_end in segments:
            seg = train_s[seg_start:seg_end]
            seg_tgt = train_tgt.iloc[seg_start:seg_end]
            if len(seg) <= seq_len:
                continue
            # Target alignment MUST match FXDataset/evaluator convention:
            # window [i..i+seq_len-1] predicts target AT i+seq_len-1 (the last
            # timestep of the window). Earlier version used target at i+seq_len
            # which caused an off-by-one mismatch between training and eval,
            # producing anti-predictive GBMs with negative train-set Sharpe.
            X = np.array([seg[i:i+seq_len].ravel() for i in range(len(seg) - seq_len + 1)])
            y = seg_tgt.values[seq_len-1:][:len(X)]
            X_parts.append(X[:len(y)])
            y_parts.append(y)
        if X_parts:
            model.fit(np.concatenate(X_parts), np.concatenate(y_parts))
        best_val_loss = 0.0
    else:
        model = create_model(backbone=backbone, n_input_features=n_features,
                             seq_len=seq_len, freeze_backbone=True,
                             head_dropout=config.get("head_dropout", 0.1),
                             het_loss=config.get("het_loss", False),
                             hidden_size=config.get("hidden_size"),
                             bidirectional=config.get("bidirectional"),
                             num_layers=config.get("num_layers"),
                             rnn_cell=config.get("rnn_cell"),
                             input_layernorm=config.get("input_layernorm", False),
                             mamba_variant=config.get("mamba_variant"),
                             mamba_d_state=config.get("mamba_d_state"),
                             mamba_expand=config.get("mamba_expand")).to(device)
        result = train_one_fold(
            model, train_feat, train_tgt, val_feat, val_tgt,
            scaler=scaler, epochs=config["epochs"], seq_len=seq_len,
            lr=config["lr"], batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            warmup_epochs=config.get("warmup_epochs", 0),
            huber_delta=config.get("huber_delta", 1.0),
            patience=config.get("patience", 5),
            grad_clip=config.get("grad_clip", 1.0),
        )
        best_val_loss = result["best_val_loss"]

    # Evaluate train/val/test
    train_eval = _evaluate_per_window(model, scaler, train_feat, train_tgt, seq_len, device, "train")
    val_eval = _evaluate_per_window(model, scaler, val_feat, val_tgt, seq_len, device, "val")
    test_eval = _evaluate_per_window(model, scaler, test_feat, test_tgt, seq_len, device, "test")

    elapsed = time.time() - t0

    # Composite score
    s, vs = test_eval["sharpe"], val_eval["sharpe"]
    tw = [w for w in test_eval["per_window"] if not w.get("skipped")]
    vw = [w for w in val_eval["per_window"] if not w.get("skipped")]
    tp = sum(1 for w in tw if w["sharpe"] > 0)
    vp = sum(1 for w in vw if w["sharpe"] > 0)
    n_neg = (len(tw) - tp) + (len(vw) - vp)
    composite = min(s, vs) - 0.1 * n_neg

    # Build result
    entry = {
        "backbone": backbone, "description": description,
        "config": config, "composite": round(composite, 4),
        "test_pos_folds": tp, "val_pos_folds": vp,
        "val_loss": round(best_val_loss, 6), "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Flatten all eval metrics with prefixes
    for prefix, ev in [("", test_eval), ("val_", val_eval), ("train_", train_eval)]:
        for k in ["sharpe", "psr", "equity", "sortino", "return_pct", "max_dd",
                   "win_rate", "profit_factor", "ic", "hit",
                   "aleatoric_mean", "epistemic_mean", "confidence_mean",
                   "precision", "recall", "f1", "f2", "mcc", "accuracy"]:
            if k in ev:
                entry[prefix + k] = ev[k]
        entry[prefix + "per_window"] = ev.get("per_window", [])

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULT: {backbone} — {description}")
    print(f"{'='*70}")
    print(f"  Composite: {composite:+.4f}  (min(test={s:+.4f}, val={vs:+.4f}) - 0.1*{n_neg})")
    print(f"  TEST  ({tp}/{len(tw)} pos):  Sharpe={s:+.4f}  Sortino={test_eval.get('sortino',0):+.4f}  "
          f"Return={test_eval.get('return_pct',0):+.2f}%  PSR={test_eval['psr']:.4f}  ${test_eval['equity']:.0f}")
    for w in tw:
        unc_str = ""
        if "confidence_mean" in w:
            unc_str = f"  Conf={w['confidence_mean']:.3f}  Ale={w['aleatoric_mean']:.1e}  Epi={w['epistemic_mean']:.1e}"
        print(f"    {w['fold']:10s} {w['regime'][:25]:25s}  Sharpe={w['sharpe']:+.4f}  "
              f"Ret={w.get('return_pct',0):+.2f}%  IC={w['ic']:+.3f}  Hit={w['hit']:.1f}%{unc_str}")
    if test_eval.get("confidence_mean") is not None:
        print(f"  TEST UNCERTAINTY:  Confidence={test_eval['confidence_mean']:.4f}  "
              f"Aleatoric={test_eval['aleatoric_mean']:.6f}  Epistemic={test_eval['epistemic_mean']:.6f}")
    print(f"  VAL   ({vp}/{len(vw)} pos):  Sharpe={vs:+.4f}  Sortino={val_eval.get('sortino',0):+.4f}")
    for w in vw:
        unc_str = ""
        if "confidence_mean" in w:
            unc_str = f"  Conf={w['confidence_mean']:.3f}  Ale={w['aleatoric_mean']:.1e}  Epi={w['epistemic_mean']:.1e}"
        print(f"    {w['fold']:10s} {w['regime'][:25]:25s}  Sharpe={w['sharpe']:+.4f}  "
              f"Ret={w.get('return_pct',0):+.2f}%  IC={w['ic']:+.3f}  Hit={w['hit']:.1f}%{unc_str}")
    if val_eval.get("confidence_mean") is not None:
        print(f"  VAL UNCERTAINTY:   Confidence={val_eval['confidence_mean']:.4f}  "
              f"Aleatoric={val_eval['aleatoric_mean']:.6f}  Epistemic={val_eval['epistemic_mean']:.6f}")
    print(f"  TRAIN: Sharpe={train_eval['sharpe']:+.4f}")
    print(f"  Time: {elapsed:.0f}s  ValLoss: {best_val_loss:.6f}")

    # Assign experiment number (auto-increment from existing JSONL)
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "experiment_log.jsonl"
    prev_num = 0
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    prev_num = max(prev_num, json.loads(line).get("experiment_num", prev_num))
                except (json.JSONDecodeError, ValueError):
                    pass
    entry["experiment_num"] = prev_num + 1

    # Update best if improved
    best_path = RESULTS_DIR / "best_config.json"
    # best_config.json tracks GLOBAL champion across all backbones (per CLAUDE.md Winner Definition)
    prev_best = -999.0
    prev_best_backbone = None
    if best_path.exists():
        with open(best_path) as f:
            saved = json.load(f)
        prev_best = saved.get("composite", -999.0)
        prev_best_backbone = saved.get("backbone")

    # KEEP status = beat global best
    entry["status"] = "KEEP" if composite > prev_best else "DISCARD"

    # Write trade-level CSV (per CLAUDE.md Trade-Level Win/Loss Logging)
    trade_dir = RESULTS_DIR / "trade_logs"
    trade_dir.mkdir(exist_ok=True)
    test_trades = test_eval.get("trade_rows", [])
    if test_trades:
        import csv as _csv
        csv_path = trade_dir / f"exp{entry['experiment_num']}_trades.csv"
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(test_trades[0].keys()))
            writer.writeheader()
            writer.writerows(test_trades)
        # Summary stats
        summary = {"exp": entry["experiment_num"], "total_trades": len(test_trades),
                   "wins": sum(1 for r in test_trades if r["correct"]),
                   "losses": sum(1 for r in test_trades if not r["correct"]),
                   "total_pnl_bps": sum(r["pnl_bps"] for r in test_trades),
                   "per_fold": {}}
        from collections import defaultdict
        by_fold = defaultdict(list)
        for r in test_trades:
            by_fold[r["fold"]].append(r)
        for fn, rows in by_fold.items():
            wins = [r["pnl_bps"] for r in rows if r["correct"]]
            losses = [r["pnl_bps"] for r in rows if not r["correct"]]
            summary["per_fold"][fn] = {
                "n": len(rows), "wins": len(wins), "losses": len(losses),
                "avg_win_bps": round(float(np.mean(wins)), 2) if wins else 0.0,
                "avg_loss_bps": round(float(np.mean(losses)), 2) if losses else 0.0,
                "max_win_bps": round(float(np.max(wins)), 2) if wins else 0.0,
                "max_loss_bps": round(float(np.min(losses)), 2) if losses else 0.0,
                "win_rate": round(len(wins)/len(rows)*100, 2) if rows else 0.0,
            }
        with open(trade_dir / f"exp{entry['experiment_num']}_trade_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Remove trade_rows from entry before logging JSONL (too large)
    for ev_key in ["per_window", "val_per_window", "train_per_window"]:
        pass  # per_window already in entry, no trade_rows in those
    entry_to_log = {k: v for k, v in entry.items() if k != "trade_rows"}

    # Log to JSONL (append)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry_to_log) + "\n")

    # Reasoning annotation: the runner ONLY writes verdict + learning (post-run
    # results-derived). diagnosis / citations / hypothesis / prediction MUST be
    # authored by Claude BEFORE launch and merged into this file with _manual=True.
    # If those pre-run fields are missing, that's a protocol violation — the
    # runner refuses to fill them with placeholders. See CLAUDE.md "Dashboard
    # Files Update Mandate" and "Dashboard Reasoning Annotations".
    ann_path = RESULTS_DIR / "reasoning_annotations.json"
    try:
        annotations = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    except Exception:
        annotations = {}
    exp_key = str(entry["experiment_num"])
    existing = annotations.get(exp_key, {})
    is_manual = bool(existing.get("_manual"))

    # Always refresh verdict + learning from actual results (even for _manual entries).
    new_verdict = (
        f"{entry.get('status','?')} — composite {composite:+.4f}, test Sharpe {entry.get('sharpe',0):+.4f}"
        + (f" (new global best, previous {prev_best:+.4f} on {prev_best_backbone})"
           if composite > prev_best
           else f" (global best remains {prev_best:+.4f} on {prev_best_backbone})")
    )
    new_learning = (
        f"Test Sharpe {entry.get('sharpe',0):+.4f} | Val Sharpe {entry.get('val_sharpe',0):+.4f} | "
        f"Train Sharpe {entry.get('train_sharpe',0):+.4f} | Return {entry.get('return_pct',0):+.2f}% | "
        f"Val loss {entry.get('val_loss',0):.6f}"
    )

    if is_manual and existing.get("diagnosis") and existing.get("citations") and existing.get("hypothesis") and existing.get("prediction"):
        # Claude pre-authored the pre-run fields — just update verdict/learning.
        existing["verdict"] = new_verdict
        existing["learning"] = new_learning
        annotations[exp_key] = existing
    else:
        # No pre-run entry found. Write a skeleton with explicit TODO markers
        # so Claude is forced to rewrite them before moving to the next experiment.
        # NEVER emit fake-looking placeholders — they corrupt the dashboard.
        desc = description or ""
        change_bits = []
        for k in ("lr", "batch_size", "seq_len", "epochs", "weight_decay", "patience",
                  "grad_clip", "huber_delta", "head_dropout", "warmup_epochs",
                  "hidden_size", "bidirectional", "num_layers", "rnn_cell",
                  "input_layernorm", "het_loss"):
            if k in config and config[k] is not None:
                change_bits.append(f"{k}={config[k]}")
        config_delta = "; ".join(change_bits) if change_bits else desc
        annotations[exp_key] = {
            "diagnosis": f"TODO-REWRITE: {backbone} experiment #{exp_key}. Description tag: {desc}. "
                         f"Claude must replace this with: why THIS experiment now (champion weakness, "
                         f"weakest fold, regime, uncertainty profile, what prior experiments ruled out).",
            "citations": "TODO-REWRITE: Insert full author(s) + year + venue + arXiv ID for every paper "
                         "motivating this experiment. Parenthetical-only tags are insufficient.",
            "hypothesis": f"TODO-REWRITE: mechanistic hypothesis. Config delta on this run: {config_delta}. "
                          f"Claude must explain the MECHANISM from the cited paper that justifies this change.",
            "prediction": "TODO-REWRITE: numeric prediction range (composite, per-fold Sharpe, uncertainty) "
                          "authored BEFORE running. Placeholder here means the 7-step process was skipped.",
            "verdict": new_verdict,
            "learning": new_learning,
            "_manual": False,
            "_needs_rewrite": True,
        }
    ann_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    if annotations[exp_key].get("_needs_rewrite"):
        print(f"  WARNING: reasoning_annotations.json[{exp_key}] needs manual rewrite — "
              f"pre-run diagnosis/citations/hypothesis/prediction were not authored.")

    if composite > prev_best:
        with open(best_path, "w") as f:
            json.dump(entry, f, indent=2)
        # Save model + scaler for reuse (portable, reloadable)
        if not is_gbm(backbone) and hasattr(model, 'state_dict'):
            weights_path = RESULTS_DIR / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "composite": composite,
                "description": description,
                "backbone": backbone,
                "n_features": n_features,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "feature_columns": list(train_feat.columns),
                "target_columns": list(train_tgt.columns),
                "experiment_num": entry["experiment_num"],
            }, weights_path)
            print(f"\n  >>> NEW GLOBAL CHAMPION ({backbone}): {composite:+.4f} (previous: {prev_best:+.4f} on {prev_best_backbone})  [weights+scaler saved to {weights_path}]")
    else:
        print(f"\n  >>> composite={composite:+.4f} vs global best={prev_best:+.4f} ({prev_best_backbone}) — not improved")

    return entry


def main():
    parser = argparse.ArgumentParser(description="Run ONE autoresearch experiment")
    parser.add_argument("--backbone", required=True, choices=list(BACKBONE_REGISTRY.keys()))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden size for MLP/LSTM backbone")
    parser.add_argument("--unidirectional", action="store_true", help="LSTM only: use unidirectional instead of default bidirectional")
    parser.add_argument("--num-layers", type=int, default=None, help="LSTM only: number of stacked LSTM layers (default 2)")
    parser.add_argument("--rnn-cell", type=str, default=None, choices=["lstm","gru"], help="LSTM backbone: use 'lstm' (default) or 'gru' cell")
    parser.add_argument("--input-layernorm", action="store_true", help="LSTM backbone: apply LayerNorm over input features per timestep (Ba 2016)")
    parser.add_argument("--mamba-variant", type=str, default=None, choices=["vanilla","s_mamba","dmamba","mambats"], help="Mamba backbone: select variant (Gu&Dao 2024 / arXiv 2403.11144 / 2602.09081 / 2405.16440)")
    parser.add_argument("--mamba-d-state", type=int, default=None, help="Mamba backbone: state dimension d_state (default 16)")
    parser.add_argument("--mamba-expand", type=int, default=None, help="Mamba backbone: inner expansion factor (default 2)")
    # GBM backbone hyperparameters (XGBoost / LightGBM / CatBoost per Chen 2016, Ke 2017, Prokhorenkova 2018)
    parser.add_argument("--n-estimators", type=int, default=None, help="XGBoost/LightGBM: number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=None, help="XGBoost/LightGBM: max tree depth (XGBoost default 6)")
    parser.add_argument("--gbm-lr", type=float, default=None, help="GBM learning_rate (default 0.03 per CLAUDE.md)")
    parser.add_argument("--subsample", type=float, default=None, help="XGBoost: row subsample (default 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=None, help="XGBoost: column subsample by tree (default 0.8)")
    parser.add_argument("--reg-lambda", type=float, default=None, help="XGBoost/LightGBM: L2 regularisation on leaf weights (default 1.0)")
    parser.add_argument("--reg-alpha", type=float, default=None, help="XGBoost/LightGBM: L1 regularisation on leaf weights (default 0/0.1)")
    parser.add_argument("--min-child-weight", type=float, default=None, help="XGBoost: minimum sum of instance weight per child (default 1)")
    parser.add_argument("--gamma", type=float, default=None, help="XGBoost: minimum loss reduction to make a split (default 0)")
    parser.add_argument("--num-leaves", type=int, default=None, help="LightGBM: max number of leaves (default 63)")
    parser.add_argument("--feature-fraction", type=float, default=None, help="LightGBM: feature subsample ratio (default 0.8)")
    parser.add_argument("--bagging-fraction", type=float, default=None, help="LightGBM: row subsample ratio (default 0.8)")
    parser.add_argument("--min-data-in-leaf", type=int, default=None, help="LightGBM: min samples per leaf (default 20)")
    parser.add_argument("--iterations", type=int, default=None, help="CatBoost: number of boosting iterations (default 2000)")
    parser.add_argument("--depth", type=int, default=None, help="CatBoost: tree depth (default 6)")
    parser.add_argument("--l2-leaf-reg", type=float, default=None, help="CatBoost: L2 regularisation on leaves (default 3.0)")
    parser.add_argument("--random-strength", type=float, default=None, help="CatBoost: randomisation strength for splits (default 1.0)")
    parser.add_argument("--bagging-temperature", type=float, default=None, help="CatBoost: Bayesian bootstrap temperature (default 1.0)")
    parser.add_argument("--bootstrap-type", type=str, default=None, help="CatBoost: bootstrap type ('Bayesian', 'Bernoulli', 'No')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--het-loss", action="store_true", default=False, help="Use heteroscedastic loss (default: plain Huber)")
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    backbone = args.backbone
    config = {
        "seq_len": args.seq_len or get_seq_len(backbone),
        "lr": args.lr or (3e-5 if backbone.startswith("lfm2") else 3e-4),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "warmup_epochs": args.warmup_epochs,
        "huber_delta": args.huber_delta,
        "head_dropout": args.head_dropout,
    }
    if args.seed is not None:
        config["seed"] = args.seed
    if args.hidden_size is not None:
        config["hidden_size"] = args.hidden_size
    if args.unidirectional:
        config["bidirectional"] = False
    if args.num_layers is not None:
        config["num_layers"] = args.num_layers
    if args.rnn_cell is not None:
        config["rnn_cell"] = args.rnn_cell
    if args.input_layernorm:
        config["input_layernorm"] = True
    if args.mamba_variant is not None:
        config["mamba_variant"] = args.mamba_variant
    if args.mamba_d_state is not None:
        config["mamba_d_state"] = args.mamba_d_state
    if args.mamba_expand is not None:
        config["mamba_expand"] = args.mamba_expand
    # GBM hyperparameters (only set if user provided)
    for _name, _val in [
        ("n_estimators", args.n_estimators), ("max_depth", args.max_depth),
        ("gbm_lr", args.gbm_lr), ("subsample", args.subsample),
        ("colsample_bytree", args.colsample_bytree),
        ("reg_lambda", args.reg_lambda), ("reg_alpha", args.reg_alpha),
        ("min_child_weight", args.min_child_weight), ("gamma", args.gamma),
        ("num_leaves", args.num_leaves),
        ("feature_fraction", args.feature_fraction),
        ("bagging_fraction", args.bagging_fraction),
        ("min_data_in_leaf", args.min_data_in_leaf),
        ("iterations", args.iterations), ("depth", args.depth),
        ("l2_leaf_reg", args.l2_leaf_reg),
        ("random_strength", args.random_strength),
        ("bagging_temperature", args.bagging_temperature),
        ("bootstrap_type", args.bootstrap_type),
    ]:
        if _val is not None:
            config[_name] = _val
    config["het_loss"] = args.het_loss

    run_single_experiment(backbone, config, args.description)


if __name__ == "__main__":
    main()
