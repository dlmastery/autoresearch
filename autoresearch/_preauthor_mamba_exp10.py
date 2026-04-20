"""Pre-author Mamba Exp10 (161): dmamba expand=4 num_layers=1 (shallow)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["161"] = {
    "diagnosis": (
        "Probe depth axis. Champion uses 2 Mamba blocks (num_layers=2). "
        "LSTM phase showed: 1-layer LSTM was −2.85 worse than 2-layer "
        "(+3.57 vs +6.42), 3-layer was catastrophic (+1.64). Mamba may "
        "be different — single Mamba block is more expressive than "
        "single LSTM layer because of the selective-scan + gated-MLP "
        "wrapper. Test if 1-layer Mamba (with dmamba decomposition) "
        "matches 2-layer at half the parameters."
    ),
    "citations": (
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) Section 4.6 — "
        "uses 2-12 Mamba blocks for language modelling. For TS, MambaTS "
        "(Cai 2024) uses 2 layers as default.\n"
        "Liu et al. 2025 'DMamba' (arXiv:2602.09081) — 2 layers default; "
        "ablates 1, 2, 3 with 2 winning at most TS scales.\n"
        "Empirical evidence from this project's LSTM phase: depth=2 was "
        "the local optimum for daily FX at n=2738 (1-layer underfit, "
        "3-layer overfit catastrophically)."
    ),
    "hypothesis": (
        "Run dmamba expand=4 with --num-layers 1. Halves parameter count "
        "of the seasonal-Mamba branch. Trend-MLP unchanged. Mechanism: "
        "fewer layers → less hierarchical temporal abstraction. If our "
        "fold-2 gains came from one layer's worth of selective routing "
        "and the second layer was redundant, 1-layer matches. If both "
        "layers contributed, 1-layer regresses."
    ),
    "prediction": (
        "Composite +4.5 to +5.5. Probability of beating dmamba expand=4 "
        "+5.60: 20% (depth typically helps a bit on noisy regression). "
        "Most informative: per-fold consistency — if 1-layer maintains "
        "all-14-positive but at lower magnitude, the second layer is "
        "purely about magnitude not direction."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp10 (161). Total: {len(ann)}")
