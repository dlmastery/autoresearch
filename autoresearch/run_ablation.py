"""Ablation study: compare all backbone architectures on the same data/splits.

Runs each backbone through the full walk-forward evaluation and generates
a comprehensive Markdown report with per-model and per-fold metrics.

Usage:
    python run_ablation.py                    # all backbones
    python run_ablation.py --backbones mlp patchtst lstm
    python run_ablation.py --epochs 3         # faster iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from .baseline import run_baseline
from .model.backbone import BACKBONE_REGISTRY, DEFAULT_BACKBONE

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
RESULTS_DIR = Path(__file__).resolve().parent / "ablation_results"


def run_ablation(
    backbones: list[str],
    epochs: int = 5,
) -> dict[str, dict]:
    """Run baseline evaluation for each backbone and collect results.

    Each backbone uses its own default seq_len (60 for LFM2.5, 10 for others).
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for i, backbone in enumerate(backbones, 1):
        print(f"\n{'#'*70}")
        print(f"# ABLATION [{i}/{len(backbones)}]: {backbone}")
        print(f"# {BACKBONE_REGISTRY.get(backbone, 'Unknown')}")
        print(f"{'#'*70}")

        t0 = time.time()
        try:
            results = run_baseline(epochs=epochs, backbone=backbone)
            results["elapsed_sec"] = round(time.time() - t0, 1)
            results["backbone"] = backbone
            results["status"] = "ok"
        except Exception as e:
            logger.error(f"Backbone {backbone} failed: {e}")
            results = {
                "backbone": backbone,
                "status": "error",
                "error": str(e),
                "elapsed_sec": round(time.time() - t0, 1),
            }

        all_results[backbone] = results

        # Save per-backbone result
        result_path = RESULTS_DIR / f"{backbone}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {result_path}")

    return all_results


def generate_report(all_results: dict[str, dict]) -> str:
    """Generate a comprehensive Markdown ablation report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("# FX Prediction Ablation Study")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Backbones tested:** {len(all_results)}")
    lines.append(f"**Evaluation:** 7-fold regime-aware walk-forward with 90-day purge + 21-day embargo")
    lines.append(f"**Target:** EUR/USD 1-day forward return, sign-based strategy")
    lines.append(f"**Metric:** Annualized Sharpe ratio (risk-adjusted)")
    lines.append("")

    # ── Summary table ────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| Backbone | Avg Sharpe | Wtd Sharpe | PSR | Overall Sharpe | Sortino | Return % | MaxDD % | Win Rate | Omega | Avg IC | Time (s) |")
    lines.append("|----------|-----------|-----------|-----|---------------|---------|----------|---------|----------|-------|--------|----------|")

    ranked = sorted(
        all_results.items(),
        key=lambda kv: kv[1].get("avg_sharpe", -999),
        reverse=True,
    )

    for backbone, res in ranked:
        if res.get("status") != "ok":
            lines.append(f"| {backbone} | ERROR | — | — | — | — | — | — | — | — | — | {res.get('elapsed_sec', '?')} |")
            continue
        o = res.get("overall", {})
        lines.append(
            f"| **{backbone}** "
            f"| {res.get('avg_sharpe', 0):.4f} "
            f"| {res.get('weighted_sharpe', 0):.4f} "
            f"| {res.get('psr', 0):.3f} "
            f"| {o.get('sharpe', 0):.4f} "
            f"| {o.get('sortino', 0):.4f} "
            f"| {o.get('total_return_pct', 0):+.2f} "
            f"| {o.get('max_drawdown_pct', 0):.2f} "
            f"| {o.get('win_rate', 0):.1f} "
            f"| {o.get('omega_ratio', 0):.3f} "
            f"| {res.get('avg_ic', 0):.4f} "
            f"| {res.get('elapsed_sec', 0):.0f} |"
        )

    lines.append("")

    # ── Per-backbone detail ──────────────────────────────────────────────
    lines.append("## Per-Backbone Detail")

    for backbone, res in ranked:
        if res.get("status") != "ok":
            lines.append(f"\n### {backbone} — FAILED")
            lines.append(f"```\n{res.get('error', 'Unknown error')}\n```")
            continue

        lines.append(f"\n### {backbone}")
        lines.append(f"*{BACKBONE_REGISTRY.get(backbone, '')}*")
        lines.append("")

        o = res.get("overall", {})
        lines.append("**Risk-Adjusted Performance**")
        lines.append(f"- Average Sharpe: **{res.get('avg_sharpe', 0):.4f}**")
        lines.append(f"- Weighted Sharpe (by train size): {res.get('weighted_sharpe', 0):.4f}")
        lines.append(f"- PSR (vs 0): {res.get('psr', 0):.4f} {'✓' if res.get('psr', 0) > 0.95 else '(not significant)'}")
        lines.append(f"- DSR: {res.get('dsr', 0):.4f}")
        lines.append(f"- Sortino: {o.get('sortino', 0):.4f}")
        lines.append(f"- Calmar: {o.get('calmar_ratio', 0):.3f}")
        lines.append(f"- Omega: {o.get('omega_ratio', 0):.3f}")
        lines.append("")

        lines.append("**Returns**")
        lines.append(f"- Total return: {o.get('total_return_pct', 0):+.2f}%")
        lines.append(f"- Annualized: {o.get('annualized_return_pct', 0):+.2f}%")
        lines.append(f"- Mean daily: {o.get('mean_daily_bps', 0):+.2f} bps")
        lines.append(f"- Daily vol: {o.get('daily_vol_bps', 0):.2f} bps")
        lines.append("")

        lines.append("**Risk**")
        lines.append(f"- Max drawdown: {o.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"- VaR 95%: {o.get('var_95', 0):.4f}%")
        lines.append(f"- CVaR 95%: {o.get('cvar_95', 0):.4f}%")
        lines.append(f"- Skewness: {o.get('skewness', 0):.4f}")
        lines.append(f"- Excess kurtosis: {o.get('excess_kurtosis', 0):.4f}")
        lines.append("")

        # Per-fold table
        folds = res.get("folds", [])
        if folds:
            lines.append("**Per-Fold Results**")
            lines.append("")
            lines.append("| Fold | Regime | Train | Sharpe | PSR | IC | Hit % | Return % | MaxDD % |")
            lines.append("|------|--------|-------|--------|-----|-----|-------|----------|---------|")
            for fd in folds:
                if fd.get("skipped"):
                    lines.append(f"| {fd['fold']} | {fd['regime']} | — | skipped | — | — | — | — | — |")
                else:
                    lines.append(
                        f"| {fd['fold']} | {fd['regime']} "
                        f"| {fd.get('train_samples', '?')} "
                        f"| {fd.get('sharpe', 0):.4f} "
                        f"| {fd.get('psr', 0):.3f} "
                        f"| {fd.get('ic_spearman', 0):.4f} "
                        f"| {fd.get('hit_rate', 0):.1f} "
                        f"| {fd.get('total_return_pct', 0):+.2f} "
                        f"| {fd.get('max_drawdown_pct', 0):.2f} |"
                    )
            lines.append("")

    # ── Conclusions ──────────────────────────────────────────────────────
    lines.append("## Conclusions")
    lines.append("")

    ok_results = [(b, r) for b, r in ranked if r.get("status") == "ok"]
    if ok_results:
        best_name, best_res = ok_results[0]
        worst_name, worst_res = ok_results[-1]

        lines.append(f"**Best backbone:** `{best_name}` with avg Sharpe = {best_res.get('avg_sharpe', 0):.4f}")
        lines.append(f"**Worst backbone:** `{worst_name}` with avg Sharpe = {worst_res.get('avg_sharpe', 0):.4f}")
        lines.append("")

        # Statistical significance
        any_significant = any(r.get("psr", 0) > 0.95 for _, r in ok_results)
        if any_significant:
            sig_models = [b for b, r in ok_results if r.get("psr", 0) > 0.95]
            lines.append(f"**Statistically significant (PSR > 0.95):** {', '.join(sig_models)}")
        else:
            lines.append("**No model achieved statistical significance (PSR > 0.95).** "
                         "This is expected for a baseline without hyperparameter tuning on a "
                         "notoriously efficient market (FX).")
        lines.append("")

        # IC analysis
        positive_ic = [(b, r.get("avg_ic", 0)) for b, r in ok_results if r.get("avg_ic", 0) > 0]
        if positive_ic:
            lines.append("**Positive information coefficient:**")
            for b, ic in sorted(positive_ic, key=lambda x: x[1], reverse=True):
                lines.append(f"- `{b}`: IC = {ic:.4f}")
        else:
            lines.append("No model achieved positive average IC across folds.")
        lines.append("")

        # Regime analysis
        lines.append("### Regime Analysis")
        lines.append("")
        lines.append("Models generally perform differently across market regimes. "
                      "The most informative comparison is performance in high-volatility "
                      "(GFC, EUR crisis) vs. low-volatility (plateau) periods.")
        lines.append("")

    lines.append("### Methodology Notes")
    lines.append("")
    lines.append("- **Walk-forward**: expanding training window, no future data leakage")
    lines.append("- **Purge**: 90-day gap between train/val and val/test splits")
    lines.append("- **Embargo**: 21-day gap between consecutive fold boundaries")
    lines.append("- **Scaler**: StandardScaler fit on training data only")
    lines.append("- **Loss**: Huber loss (robust to fat tails)")
    lines.append("- **Optimizer**: AdamW with cosine annealing + gradient clipping")
    lines.append("- **Early stopping**: patience = 3 epochs")
    lines.append("- **Strategy**: sign(predicted_return) * actual_return (directional trading)")
    lines.append("- **Metrics**: Lopez de Prado (2018) — PSR, DSR, IC, CVaR")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ablation study across model backbones")
    parser.add_argument(
        "--backbones", nargs="+",
        default=list(BACKBONE_REGISTRY.keys()),
        choices=list(BACKBONE_REGISTRY.keys()),
        help="Backbones to evaluate",
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print(f"Ablation study: {len(args.backbones)} backbones")
    for b in args.backbones:
        print(f"  - {b}: {BACKBONE_REGISTRY[b]}")

    all_results = run_ablation(args.backbones, epochs=args.epochs)

    # Generate and save report
    REPORT_DIR.mkdir(exist_ok=True)
    report = generate_report(all_results)
    report_path = REPORT_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Also save combined JSON
    combined_path = RESULTS_DIR / "ablation_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
