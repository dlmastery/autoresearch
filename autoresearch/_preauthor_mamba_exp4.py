"""Pre-author Mamba Exp4 (JSONL 155): dmamba decomposition variant."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["155"] = {
    "diagnosis": (
        "Mamba family Exp4 — first principled architectural variant after "
        "ruling out s_mamba. Vanilla Mamba (Exp1) gave +5.27 with val fold 2 "
        "+1.37 breakthrough. dmamba decomposes the input into (a) a slow "
        "trend learned via a 2-layer MLP on the time-mean of features, plus "
        "(b) a seasonal residual processed by the canonical Mamba block. "
        "For FX, the trend captures macro regime drift (USD bull/bear over "
        "years) and the seasonal residual captures short-term noise + intra-"
        "regime fluctuation. Hypothesis: separating these signals lets each "
        "branch specialise, lifting test fold 2 (post-crash recovery is a "
        "trend-shift regime by definition) without the variate-axis loss."
    ),
    "citations": (
        "Liu et al. 2025 'DMamba: Decomposition-Enhanced Mamba for Time "
        "Series Forecasting' (arXiv:2602.09081) — introduces trend+seasonal "
        "decomposition with Mamba on the seasonal component and a simple "
        "MLP on the trend. Beats vanilla Mamba and DLinear on standard "
        "LTSF benchmarks.\n"
        "Wu, Xu, Wang, Long 2021 NeurIPS 'Autoformer' (arXiv:2106.13008) — "
        "earliest principled use of seasonal-trend decomposition inside a "
        "deep model; shows that explicit decomposition beats end-to-end on "
        "non-stationary time series.\n"
        "Zeng, Chen, Zhang, Xu 2023 AAAI 'DLinear: Are Transformers "
        "Effective for Time Series Forecasting?' (arXiv:2205.13504) — shows "
        "that a trivial linear trend predictor beats most transformer "
        "baselines, motivating the dmamba 'trend-MLP' branch.\n"
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) — underlying SSM "
        "block used for the seasonal branch."
    ),
    "hypothesis": (
        "Run Mamba Exp1 config (d_model=128, d_state=16, expand=2, 2-layer, "
        "lr=5e-4, bs=32, wd=0.1, warmup=10, ep=100, pat=20, seed=42) with "
        "--mamba-variant dmamba. The decomposition is implemented in "
        "CurrencyMamba.forward: trend = MLP(h.mean(dim=time)); seasonal = "
        "h[:, -1, :] (last-step Mamba output); hidden = trend + seasonal. "
        "Mechanism: trend MLP has ~33K params and learns the slow USD/EUR "
        "drift; the Mamba branch has ~330K params and focuses on regime-"
        "specific deviations. Predicted effect: test fold 2 lifts toward "
        "or above 0; other folds within ±0.5 of vanilla."
    ),
    "prediction": (
        "Composite +5.0 to +5.8. Probability of beating vanilla Mamba +5.27: "
        "55% (decomposition is well-supported by recent literature). "
        "Probability of new global champion (>+6.42): 8%. Test fold 2 lift "
        "from −0.98 toward [−0.3, +0.5] is the most informative metric. "
        "If composite < 5.0, decomposition adds noise rather than signal at "
        "our small n."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp4 (155). Total: {len(ann)}")
