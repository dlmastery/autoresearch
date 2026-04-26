"""Generate per-day win/loss CSVs for the two ensemble winners:
 - mega_ensemble (3 GBMs at seq=60 + LSTM Exp35 at seq=10) — global champion +9.7071
 - ensemble_3way_gbm (3 GBMs at seq=60) — +9.4708

Schema matches autoresearch_results/trade_logs/exp<N>_trades.csv:
 date, fold, regime, prediction, pred_direction, actual_return,
 actual_direction, strategy_return, cumulative_return, confidence,
 aleatoric, epistemic, correct, pnl_bps

Reuses the rank-avg recipe from _emtsf_mega_ensemble.py — the version that
produced the +9.7071 README headline number.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/evija/autoresearch")

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from autoresearch.run_autoresearch import compute_all_features, compute_targets
from autoresearch.data.download import download_all_pairs, download_macro_signals
from autoresearch.data.splits import split_superfold, FOLDS, get_fold_dates
from autoresearch.model.backbone import GBMWrapper, create_model
from autoresearch.evaluation.metrics import sharpe_ratio, trading_report

ROOT = Path("C:/Users/evija/autoresearch")
RESULTS = ROOT / "autoresearch" / "autoresearch_results"
WINNERS = RESULTS / "winners"
TRADE_LOGS = RESULTS / "trade_logs"
TRADE_LOGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("[ensemble-trade-logs] loading data + features ...")
    pairs = download_all_pairs()
    macro = download_macro_signals()
    feats = compute_all_features(pairs, macro)
    targets = compute_targets(pairs["EURUSD=X"])
    common = feats.index.intersection(targets.index)
    feats = feats.loc[common]
    targets = targets.loc[common]
    train_feat, _, test_feat = split_superfold(feats)
    _, _, test_tgt = split_superfold(targets)

    n_features = train_feat.shape[1]

    # Load 3 GBM bundles
    gbm_paths = [
        WINNERS / "xgboost_exp203_maxdepth4_gbmlr0.01_seq60" / "xgboost_model.pkl",
        WINNERS / "lightgbm_exp235_maxdepth4_gbmlr0.01_seq60" / "lightgbm_model.pkl",
        WINNERS / "catboost_exp236_gbmlr0.01_depth4_seq60" / "catboost_model.pkl",
    ]
    gbm_bundles = [pickle.load(open(p, "rb")) for p in gbm_paths if p.exists()]
    print(f"[ensemble-trade-logs] loaded {len(gbm_bundles)} GBM bundles")

    # Load LSTM champion checkpoint
    lstm_ckpt_path = WINNERS / "lstm_exp35_wd7e4_bs16_seed42" / "model_checkpoint.pt"
    lstm_bundle = None
    if lstm_ckpt_path.exists():
        ckpt = torch.load(lstm_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        seq_len = 10
        model = create_model(
            backbone="lstm",
            n_input_features=n_features,
            seq_len=seq_len,
            freeze_backbone=True,
            head_dropout=cfg.get("head_dropout", 0.25),
            het_loss=cfg.get("het_loss", False),
            hidden_size=cfg.get("hidden_size"),
            bidirectional=cfg.get("bidirectional"),
            num_layers=cfg.get("num_layers"),
            rnn_cell=cfg.get("rnn_cell"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        lstm_bundle = {
            "model": model,
            "seq_len": seq_len,
            "scaler_mean": np.asarray(ckpt["scaler_mean"]),
            "scaler_scale": np.asarray(ckpt["scaler_scale"]),
        }
        print(f"[ensemble-trade-logs] loaded LSTM (seq={seq_len})")
    else:
        print("[ensemble-trade-logs] WARN: LSTM checkpoint not found")

    # Inference helpers ------------------------------------------------------
    def gbm_predict(bundle: dict, wf: pd.DataFrame, wt: pd.DataFrame):
        seq = bundle["seq_len"]
        ws = (wf.values - bundle["scaler_mean"]) / bundle["scaler_scale"]
        if len(ws) < seq + 1:
            return None, None, None
        X = np.array([ws[i:i + seq].ravel() for i in range(len(ws) - seq + 1)])
        y = wt.values[seq - 1:][:len(X), 0]
        dates = wt.index[seq - 1:][:len(X)]
        preds = bundle["gbm_wrapper"].predict(X)[:, 0]
        return dates, preds, y

    def lstm_predict(bundle: dict, wf: pd.DataFrame, wt: pd.DataFrame):
        seq = bundle["seq_len"]
        ws = (wf.values - bundle["scaler_mean"]) / bundle["scaler_scale"]
        if len(ws) < seq + 1:
            return None, None, None

        class _DS(Dataset):
            def __init__(self, f, t, L):
                self.f = torch.tensor(f, dtype=torch.float32)
                self.t = torch.tensor(t, dtype=torch.float32)
                self.L = L

            def __len__(self):
                return len(self.f) - self.L + 1

            def __getitem__(self, i):
                return self.f[i:i + self.L], self.t[i + self.L - 1]

        ds = _DS(ws, wt.values, seq)
        loader = DataLoader(ds, batch_size=256)
        preds_list = []
        with torch.no_grad():
            for x, _ in loader:
                out = bundle["model"](x)
                preds_list.append(out["ret_1d"][:, 0].numpy())
        preds = np.concatenate(preds_list) if preds_list else np.array([])
        y = wt.values[seq - 1:][:len(preds), 0]
        dates = wt.index[seq - 1:][:len(preds)]
        return dates, preds, y

    def rank_avg(arr: np.ndarray) -> np.ndarray:
        ranks = np.column_stack([rankdata(arr[:, c]) for c in range(arr.shape[1])])
        return ranks.mean(axis=1) - (len(arr) + 1) / 2

    # Build daily-row tables -------------------------------------------------
    rows_mega: list[dict] = []
    rows_gbm3: list[dict] = []

    for fold in FOLDS:
        d = get_fold_dates(fold)
        wf = test_feat.loc[d["test_start"]:d["test_end"]]
        wt = test_tgt.loc[d["test_start"]:d["test_end"]]
        if len(wf) < 61:  # need seq=60 + 1
            continue

        per_model: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for b in gbm_bundles:
            dt, p, y = gbm_predict(b, wf, wt)
            if p is not None:
                per_model.append(("gbm_" + b["backbone"], dt, p, y))
        if lstm_bundle is not None:
            dt, p, y = lstm_predict(lstm_bundle, wf, wt)
            if p is not None:
                per_model.append(("nn_lstm", dt, p, y))

        if not per_model:
            continue

        # Align on the LATEST start date so every model has a prediction
        latest_start = max(m[1][0] for m in per_model)
        aligned = []
        for name, dt, p, y in per_model:
            mask = dt >= latest_start
            aligned.append((name, dt[mask], p[mask], y[mask]))
        min_n = min(len(a[1]) for a in aligned)
        aligned = [(n, dt[:min_n], p[:min_n], y[:min_n]) for n, dt, p, y in aligned]
        if min_n == 0:
            continue

        dates = aligned[0][1]
        y_true = aligned[0][3]

        gbm_arr = np.column_stack([a[2] for a in aligned if a[0].startswith("gbm_")])
        mega_arr = np.column_stack([a[2] for a in aligned])

        gbm3_score = rank_avg(gbm_arr)
        mega_score = rank_avg(mega_arr)

        def append_rows(rows: list[dict], score: np.ndarray) -> None:
            cum = 0.0
            for i, dt in enumerate(dates):
                pred = float(score[i])
                pred_dir = 1 if pred > 0 else (-1 if pred < 0 else 0)
                act_ret = float(y_true[i])
                act_dir = 1 if act_ret > 0 else (-1 if act_ret < 0 else 0)
                strat_ret = pred_dir * act_ret
                cum += strat_ret
                correct = 1 if pred_dir == act_dir and pred_dir != 0 else 0
                rows.append({
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "fold": fold["name"],
                    "regime": fold["regime"],
                    "prediction": pred,
                    "pred_direction": pred_dir,
                    "actual_return": act_ret,
                    "actual_direction": act_dir,
                    "strategy_return": strat_ret,
                    "cumulative_return": cum,
                    "confidence": "",
                    "aleatoric": "",
                    "epistemic": "",
                    "correct": correct,
                    "pnl_bps": strat_ret * 10000.0,
                })

        append_rows(rows_mega, mega_score)
        append_rows(rows_gbm3, gbm3_score)

    # Write CSVs + summaries -------------------------------------------------
    fieldnames = [
        "date", "fold", "regime", "prediction", "pred_direction",
        "actual_return", "actual_direction", "strategy_return",
        "cumulative_return", "confidence", "aleatoric", "epistemic",
        "correct", "pnl_bps",
    ]

    def write(rows: list[dict], stem: str, label: str) -> dict:
        csv_path = TRADE_LOGS / f"{stem}_trades.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        # Summary
        per_fold: dict[str, dict] = {}
        for r in rows:
            f_ = r["fold"]
            d = per_fold.setdefault(f_, {"n": 0, "wins": 0, "losses": 0,
                                         "win_pnls": [], "loss_pnls": []})
            d["n"] += 1
            if r["correct"] == 1:
                d["wins"] += 1
                d["win_pnls"].append(r["pnl_bps"])
            else:
                d["losses"] += 1
                d["loss_pnls"].append(r["pnl_bps"])
        for f_, d in per_fold.items():
            d["avg_win_bps"] = round(float(np.mean(d["win_pnls"])), 2) if d["win_pnls"] else 0.0
            d["avg_loss_bps"] = round(float(np.mean(d["loss_pnls"])), 2) if d["loss_pnls"] else 0.0
            d["max_win_bps"] = round(float(np.max(d["win_pnls"])), 2) if d["win_pnls"] else 0.0
            d["max_loss_bps"] = round(float(np.min(d["loss_pnls"])), 2) if d["loss_pnls"] else 0.0
            d["win_rate"] = round(100.0 * d["wins"] / d["n"], 2) if d["n"] else 0.0
            d.pop("win_pnls"); d.pop("loss_pnls")

        rets = np.array([r["strategy_return"] for r in rows])
        rpt = trading_report(rets) if len(rets) else {"total_return_pct": 0, "win_rate": 0}
        sh = float(sharpe_ratio(rets)) if len(rets) else 0.0

        summary = {
            "ensemble": label,
            "total_trades": len(rows),
            "wins": sum(1 for r in rows if r["correct"] == 1),
            "losses": sum(1 for r in rows if r["correct"] == 0),
            "total_pnl_bps": float(sum(r["pnl_bps"] for r in rows)),
            "test_sharpe": round(sh, 4),
            "total_return_pct": round(rpt["total_return_pct"], 2),
            "overall_win_rate": round(rpt["win_rate"], 2),
            "per_fold": per_fold,
        }
        sum_path = TRADE_LOGS / f"{stem}_trade_summary.json"
        sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[ensemble-trade-logs] {label}: wrote {csv_path.name} ({len(rows)} rows) + summary  "
              f"Sharpe={sh:+.4f}  Ret={rpt['total_return_pct']:+.2f}%  WR={rpt['win_rate']:.1f}%")
        return summary

    write(rows_mega, "mega_ensemble", "MEGA (3 GBM + 1 LSTM, rank-avg)")
    write(rows_gbm3, "ensemble_3way_gbm", "3-WAY GBM (rank-avg, seq=60)")

    print("[ensemble-trade-logs] done")


if __name__ == "__main__":
    main()
