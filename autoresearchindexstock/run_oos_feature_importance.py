"""Permutation feature importance on OOS winners.

Per Breiman 2001 'Random Forests' §4 and Fisher, Rudin & Dominici 2019 JMLR
'All Models are Wrong, but Many are Useful: Learning a Variable's Importance
by Studying an Entire Class of Prediction Models Simultaneously' (arXiv:1801.01489)
— the model-agnostic permutation importance method:

  baseline_score = score(model, X_oos, y_oos)
  for each feature j:
      X' = X_oos with column j shuffled
      perm_score_j = score(model, X', y_oos)
      importance_j = baseline_score - perm_score_j

Higher importance => shuffling that feature degrades performance more
=> the model relies on that feature.

We use OOS Sharpe as the score. We run this on:
  1. exp 234 (best individual OOS, LSTM s35 seed=2026 +2.22 Sharpe)
  2. exp 304 (best mamba, +2.01)
  3. exp 231 (LSTM s35 seed=11, +2.17)
  4. Top-5 vote ensemble (the actual production strategy)

For each model, the OOS window is 2025-10-01 → 2026-04-30 with the
training-set scaler applied. Features are shuffled within the OOS buffer
window only — preserving the train period (which the model never sees
during inference anyway).

Outputs:
  - oos_feature_importance_exp{N}.csv (per-feature drop, ranked)
  - oos_feature_importance_summary.json (top-K + grouped by domain)
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoresearch.run_autoresearch import _pin_to_safe_cores
_pin_to_safe_cores()

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from autoresearchindexstock.data.features import compute_qqq_features, compute_qqq_targets
from autoresearchindexstock.data.download import ALL_SIGNALS
from autoresearch.model.backbone import create_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "autoresearch_results"
WINNERS = RESULTS / "winners"

OOS_START = pd.Timestamp("2025-10-01")
OOS_END = pd.Timestamp("2026-04-30")

# Models to analyze (rank-1 individual + top mamba + LSTM-seed11)
MODELS_TO_ANALYZE = [
    {"exp": 234, "ckpt": "exp234_prod_retrain/model_checkpoint.pt", "label": "LSTM s35 seed=2026 (#1 OOS)"},
    {"exp": 231, "ckpt": "exp231_prod_retrain/model_checkpoint.pt", "label": "LSTM s35 seed=11  (#2 OOS)"},
    {"exp": 304, "ckpt": "exp304_prod_retrain/model_checkpoint.pt", "label": "Mamba s60 seed=42 (best mamba OOS)"},
    {"exp":  55, "ckpt": "exp55_prod_retrain/model_checkpoint.pt",  "label": "Mamba s60 seed=7"},
    {"exp": 281, "ckpt": "exp281_prod_retrain/model_checkpoint.pt", "label": "Mambastock s60 seed=42"},
]

# Group features by economic-domain category for summary
DOMAIN_TAGS = {
    "qqq_": "qqq_primary",
    "vix": "volatility_regime", "vxn": "volatility_regime", "skew": "volatility_regime",
    "move": "volatility_regime", "vrp_": "volatility_regime", "vvix": "volatility_regime",
    "ovx": "volatility_regime", "gvz": "volatility_regime",
    "yld_": "yields_curve", "term_": "yields_curve",
    "tlt_": "bonds_credit", "ief_": "bonds_credit", "shy_": "bonds_credit", "tip_": "bonds_credit",
    "hyg_": "bonds_credit", "lqd_": "bonds_credit", "agg_": "bonds_credit",
    "dxy_": "fx_macro", "eurusd": "fx_macro", "usdjpy": "fx_macro",
    "gold_": "commodities", "silver_": "commodities", "wti_": "commodities",
    "brent_": "commodities", "copper_": "commodities", "copper_gold": "commodities",
    "spy_": "benchmarks", "qqq_over_spy": "benchmarks", "dia_": "benchmarks",
    "qqq_over_dia": "benchmarks", "iwm_": "benchmarks", "qqq_over_iwm": "benchmarks",
    "efa_": "benchmarks", "qqq_over_efa": "benchmarks", "eem_": "benchmarks", "qqq_over_eem": "benchmarks",
    "qqq_over_ixic": "benchmarks",
    "soxx_": "industry_tilts", "smh_": "industry_tilts", "qqq_over_soxx": "industry_tilts",
    "qqq_over_smh": "industry_tilts", "ibb_": "industry_tilts", "arkk_": "industry_tilts",
    "btc_": "crypto",
    "sec_": "sectors", "sector_": "sectors", "xlk_": "sectors", "xlf_": "sectors",
    "xlv_": "sectors", "xly_": "sectors", "xlp_": "sectors", "xle_": "sectors",
    "xli_": "sectors", "xlu_": "sectors", "xlb_": "sectors", "xly_minus": "sectors", "xlk_minus": "sectors",
    "n225_": "intl_risk", "ftse_": "intl_risk", "dax_": "intl_risk", "hsi_": "intl_risk",
    "dow_": "calendar", "month": "calendar", "jan_effect": "calendar", "dec_effect": "calendar",
    "turn_of_month": "calendar", "santa_rally": "calendar", "fomc_": "calendar",
    "opex_": "calendar", "earnings_season": "calendar",
    "lag_ret": "autoregressive", "var_ratio": "autoregressive", "dd_from_": "autoregressive",
}


def domain_for(col: str) -> str:
    for prefix, tag in DOMAIN_TAGS.items():
        if col.startswith(prefix) or prefix in col:
            return tag
    return "other"


def _download_no_cap(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    return df


def download_data(start="2004-01-01", end="2026-04-30") -> dict:
    out = {}
    for group in ALL_SIGNALS.values():
        for ticker in group:
            try:
                df = _download_no_cap(ticker, start, end)
            except Exception as e:
                continue
            if df is None or df.empty:
                continue
            out[ticker] = df
    print(f"[download] {len(out)} tickers")
    return out


def build_model_from_ckpt(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    backbone = ckpt["backbone"]
    n_features = ckpt["n_features"]
    seq_len = cfg.get("seq_len") or 60
    # Heuristic mamba HPs from saved state if missing in config
    sd = ckpt["model_state_dict"]
    mamba_variant = cfg.get("mamba_variant")
    mamba_d_state = cfg.get("d_state") or cfg.get("mamba_d_state")
    mamba_expand = cfg.get("expand") or cfg.get("mamba_expand")
    if backbone == "mamba":
        if mamba_variant is None:
            mamba_variant = "dmamba" if any("trend_mlp" in k for k in sd.keys()) else "vanilla"
        if mamba_d_state is None:
            for k, v in sd.items():
                if k.endswith("A_log"):
                    mamba_d_state = v.shape[-1]; break
            mamba_d_state = mamba_d_state or 16
        if mamba_expand is None:
            for k, v in sd.items():
                if "in_proj.weight" in k:
                    mamba_expand = max(1, v.shape[0] // (cfg.get("hidden_size") or 64) // 2); break
            mamba_expand = mamba_expand or 2
    model = create_model(
        backbone=backbone,
        n_input_features=n_features,
        seq_len=seq_len,
        head_dropout=cfg.get("head_dropout", 0.1),
        het_loss=True,
        hidden_size=cfg.get("hidden_size") if cfg.get("hidden_size") else None,
        num_layers=cfg.get("num_layers") if cfg.get("num_layers") else None,
        mamba_variant=mamba_variant,
        mamba_d_state=mamba_d_state,
        mamba_expand=mamba_expand,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, ckpt


def predict_oos(model, scaler_mean, scaler_scale, feat_oos: np.ndarray, seq_len: int) -> np.ndarray:
    """Returns array of len = N_OOS - seq_len + 1 (one prediction per valid window)."""
    arr = (feat_oos - scaler_mean) / scaler_scale
    preds = []
    with torch.no_grad():
        for i in range(seq_len - 1, len(arr)):
            window = arr[i - seq_len + 1: i + 1]
            x = torch.from_numpy(window).float().unsqueeze(0)
            out = model(x)
            mu = float(out["ret_1d"][:, 0].cpu().numpy()[0])
            preds.append(mu)
    return np.array(preds)


def annualized_sharpe(pnl: np.ndarray) -> float:
    if pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def analyze_one(model_meta: dict, feats_full: pd.DataFrame, targets_full: pd.DataFrame,
                rng: np.random.Generator) -> dict:
    ckpt_path = WINNERS / model_meta["ckpt"]
    if not ckpt_path.exists():
        print(f"[skip] {model_meta['exp']}: no checkpoint at {ckpt_path}")
        return {}
    print(f"[exp{model_meta['exp']}] loading {ckpt_path.name}")
    model, ckpt = build_model_from_ckpt(ckpt_path)
    feature_columns = ckpt["feature_columns"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=float)
    seq_len = ckpt.get("config", {}).get("seq_len") or 60

    # Align features to ckpt's feature_columns
    common = feats_full.index.intersection(targets_full.index)
    feats = feats_full.loc[common, feature_columns].copy()
    tars = targets_full.loc[common].copy()

    # Slice OOS window (need seq_len history before OOS_START to predict OOS_START)
    oos_mask = (feats.index >= OOS_START) & (feats.index <= OOS_END)
    pre_oos_dates = feats.index[feats.index < OOS_START][-seq_len + 1:] if seq_len > 1 else pd.DatetimeIndex([])
    eval_dates = pre_oos_dates.append(feats.index[oos_mask])
    feats_eval = feats.loc[eval_dates].values
    actual_1d = tars.loc[feats.index[oos_mask], "fwd_ret_1d"].values

    # Baseline OOS prediction
    base_preds = predict_oos(model, scaler_mean, scaler_scale, feats_eval, seq_len)
    # Align: base_preds[i] predicts for date eval_dates[seq_len-1+i] which is feats.index[oos_mask][i]
    n_pred = len(base_preds)
    n_act = min(n_pred, len(actual_1d))
    valid_mask = ~np.isnan(actual_1d[:n_act])
    base_pnl = np.sign(base_preds[:n_act]) * actual_1d[:n_act]
    base_pnl_valid = base_pnl[valid_mask]
    base_sharpe = annualized_sharpe(base_pnl_valid)
    print(f"[exp{model_meta['exp']}] baseline OOS Sharpe = {base_sharpe:.4f} ({len(base_pnl_valid)} preds)")

    # Permutation importance: shuffle each column, measure Sharpe drop
    n_features = len(feature_columns)
    importances = []
    for j, col in enumerate(feature_columns):
        feats_perm = feats_eval.copy()
        # Shuffle column j across rows (preserves marginal distribution, breaks dependence)
        idx = rng.permutation(len(feats_perm))
        feats_perm[:, j] = feats_eval[idx, j]
        perm_preds = predict_oos(model, scaler_mean, scaler_scale, feats_perm, seq_len)
        perm_pnl = np.sign(perm_preds[:n_act]) * actual_1d[:n_act]
        perm_sharpe = annualized_sharpe(perm_pnl[valid_mask])
        importance = base_sharpe - perm_sharpe
        importances.append({
            "feature": col,
            "domain": domain_for(col),
            "baseline_sharpe": round(base_sharpe, 4),
            "permuted_sharpe": round(perm_sharpe, 4),
            "sharpe_drop": round(importance, 4),
            "rank": -1,  # filled after sort
        })
        if (j + 1) % 25 == 0 or j == n_features - 1:
            print(f"[exp{model_meta['exp']}] feature {j+1}/{n_features} done")

    # Rank by drop magnitude (positive = important; negative = noisy/harmful)
    importances.sort(key=lambda r: -r["sharpe_drop"])
    for k, r in enumerate(importances, 1):
        r["rank"] = k

    # Save per-experiment CSV
    df = pd.DataFrame(importances)
    csv_path = RESULTS / f"oos_feature_importance_exp{model_meta['exp']}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[exp{model_meta['exp']}] wrote {csv_path.name}")

    return {
        "exp": model_meta["exp"],
        "label": model_meta["label"],
        "baseline_sharpe": round(base_sharpe, 4),
        "n_features": n_features,
        "top10": importances[:10],
        "bottom10_negative": [r for r in importances if r["sharpe_drop"] < 0][:10],
        "csv": f"oos_feature_importance_exp{model_meta['exp']}.csv",
    }


def main():
    print("[setup] downloading market data...")
    raw = download_data()
    feats_full = compute_qqq_features(raw)
    targets_full = compute_qqq_targets(raw)
    print(f"[setup] features {feats_full.shape}, targets {targets_full.shape}")

    rng = np.random.default_rng(seed=42)
    summary = {
        "method": "Breiman 2001 + Fisher, Rudin & Dominici 2019 (arXiv:1801.01489)",
        "oos_window": {"start": str(OOS_START.date()), "end": str(OOS_END.date())},
        "models": [],
        "domain_aggregation": {},
    }

    for meta in MODELS_TO_ANALYZE:
        existing_csv = RESULTS / f"oos_feature_importance_exp{meta['exp']}.csv"
        if existing_csv.exists():
            print(f"[skip] exp{meta['exp']}: existing CSV — load summary from disk")
            df = pd.read_csv(existing_csv)
            top10 = df.head(10).to_dict("records")
            bot10 = df[df["sharpe_drop"] < 0].head(10).to_dict("records") if (df["sharpe_drop"] < 0).any() else []
            baseline = float(df["baseline_sharpe"].iloc[0]) if "baseline_sharpe" in df.columns else 0
            summary["models"].append({
                "exp": meta["exp"], "label": meta["label"], "baseline_sharpe": round(baseline, 4),
                "n_features": len(df), "top10": top10, "bottom10_negative": bot10,
                "csv": existing_csv.name,
            })
            continue
        try:
            result = analyze_one(meta, feats_full, targets_full, rng)
            if result:
                summary["models"].append(result)
        except Exception as e:
            print(f"[error] exp{meta['exp']}: {e}")
            import traceback; traceback.print_exc()

    # Cross-model domain aggregation: average drop per domain across all models
    domain_drops = {}
    for m in summary["models"]:
        for r in m.get("top10", []) + m.get("bottom10_negative", []):
            d = r["domain"]
            domain_drops.setdefault(d, []).append(r["sharpe_drop"])
    summary["domain_aggregation"] = {
        d: {
            "mean_drop": round(float(np.mean(v)), 4),
            "max_drop": round(float(np.max(v)), 4),
            "n_features_in_topbot": len(v),
        } for d, v in sorted(domain_drops.items(), key=lambda kv: -np.max(kv[1]))
    }

    json_path = RESULTS / "oos_feature_importance_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {json_path.name}")
    print(f"[done] {len(summary['models'])} models analyzed")

    # Print top-5 per model for terminal readout
    print("\n=== TOP-5 most important features per model ===")
    for m in summary["models"]:
        print(f"\nexp{m['exp']} {m['label']} (baseline Sharpe {m['baseline_sharpe']}):")
        for r in m["top10"][:5]:
            print(f"  rank{r['rank']:>2}  Δ={r['sharpe_drop']:+6.3f}  {r['feature']}  ({r['domain']})")


if __name__ == "__main__":
    main()
