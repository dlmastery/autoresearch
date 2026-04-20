"""Pre-author Mamba Exp7 (158): dmamba expand=4 (double inner dim)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["158"] = {
    "diagnosis": (
        "d_state axis closed at 16. Now probe `expand` axis. Champion "
        "uses expand=2 (inner_dim = d_model × 2 = 256). Try expand=4 "
        "(inner_dim=512) — 2× wider per-block computation but same state "
        "size. The expand factor controls the MLP-like projection inside "
        "each Mamba block (in_proj: d_model → 2×inner; out_proj: inner → "
        "d_model). Wider inner allows more nonlinear feature mixing per "
        "step. At our 104-feature input, more inner capacity might better "
        "compose multi-feature interactions; or it might just overfit."
    ),
    "citations": (
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) Section 3.4 — "
        "default expand=2 follows the original Mamba paper; ablations "
        "with expand=4 not reported in the main text but referenced in "
        "the appendix as 'similar performance, more compute'.\n"
        "Tan & Le 2019 ICML 'EfficientNet' (arXiv:1905.11946) — compound "
        "scaling principle: width and depth should scale together. We're "
        "deliberately scaling ONLY width here to isolate its effect.\n"
        "Wang et al. 2024 'Is Mamba Effective for Time Series?' "
        "(arXiv:2403.11144) — uses expand=2 throughout; doesn't ablate."
    ),
    "hypothesis": (
        "Run dmamba (champion) with --mamba-expand 4. All other params "
        "identical: d_model=128, d_state=16, 2-layer, lr=5e-4, bs=32, "
        "wd=0.1, warmup=10, ep=100, pat=20, seed=42. Mechanism: doubles "
        "the Mamba block parameter count (mainly in_proj and out_proj). "
        "If our task benefits from richer per-step feature mixing, "
        "expand=4 lifts composite. If we're already at the data-limited "
        "ceiling, more parameters just overfit and val/test regress."
    ),
    "prediction": (
        "Composite +4.5 to +5.5. Probability of beating dmamba +5.36: "
        "30%. Probability of new global champion (>+6.42): 4%. Most "
        "informative: train-test gap. If train Sharpe rises to +8.5+ "
        "while test stays ~+5, expand=4 wasted on memorisation — confirms "
        "data-limited regime."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp7 (158). Total: {len(ann)}")
