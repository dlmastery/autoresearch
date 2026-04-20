"""Pre-author Mamba Exp6 (157): dmamba d_state=8 (half state, regularise)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["157"] = {
    "diagnosis": (
        "d_state axis: tested {16 (champion +5.36), 32 (over-capacity, "
        "+4.20)}. Both endpoints suggest 16 is at or above the optimum. "
        "Try d_state=8 to test the regularisation direction. If smaller "
        "state strictly hurts (under-capacity), d_state=16 confirmed optimal "
        "and axis closes. If smaller state matches or beats 16, the SSM "
        "was over-parameterised and we have a ratchet to even tighter "
        "config (d_state=4 next). Hypothesis priors lean toward 16 being "
        "best per Cai et al. 2024 MambaTS recipe matching our default."
    ),
    "citations": (
        "Cai, Jiang, Wu, Zhang, Wang 2024 NeurIPS 'MambaTS: Improved "
        "Selective State Space Models for Long-Term Time Series Forecasting' "
        "(arXiv:2405.16440) — uses d_state=16 for short-horizon LTSF; "
        "explicitly notes d_state ∈ [8, 32] as the practical range.\n"
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) Section 4.5 — "
        "ablation on d_state shows monotonic improvement up to ~64 on "
        "synthetic copying, but financial TS does not have the same "
        "long-range dependency structure.\n"
        "Liu et al. 2025 'DMamba' (arXiv:2602.09081) — confirms d_state=16 "
        "default for the seasonal Mamba branch; trend-MLP branch absorbs "
        "much of the long-range dependency that would otherwise need state."
    ),
    "hypothesis": (
        "Run dmamba (current Mamba family champion) with --mamba-d-state 8. "
        "All other params identical: d_model=128, expand=2, 2-layer, "
        "lr=5e-4, bs=32, wd=0.1, warmup=10, ep=100, pat=20, seed=42. "
        "Mechanism: 8 SSM states halve the seasonal-branch capacity. If "
        "d_state=16 was the right amount, this regresses uniformly. If "
        "d_state=16 had wasted capacity (overfit), d_state=8 acts as "
        "regulariser and improves test."
    ),
    "prediction": (
        "Composite +4.8 to +5.5. Probability of beating dmamba +5.36: "
        "30%. Probability of new global champion (>+6.42): 4%. Most "
        "informative: train Sharpe — if it stays at +7.5 (not below) "
        "while test improves, capacity reduction was clean regularisation."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp6 (157). Total: {len(ann)}")
