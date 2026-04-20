"""Pre-author reasoning entry for Mamba Exp2 (s_mamba variant), JSONL 153."""
import json
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

ann["153"] = {
    "diagnosis": (
        "Mamba Exp1 (vanilla) got +5.27 composite with 7/7 positive val but "
        "test fold 2 at −0.98. The vanilla Mamba applies SSM over the TIME "
        "axis (each timestep's hidden state evolves). For 104-feature FX data "
        "we might benefit from applying SSM over the FEATURE (variate) axis "
        "instead — i.e. let the model selectively route WHICH features matter "
        "per timestep rather than which timesteps matter per feature. This is "
        "the S-Mamba hypothesis (Liu et al. 2024). Target: lift test fold 2 "
        "from −0.98 without sacrificing the +1.37 val fold 2 we just earned."
    ),
    "citations": (
        "Liu, Wang, Tang, Zhou, Wang 2024 'Is Mamba Effective for Time Series "
        "Forecasting?' (arXiv:2403.11144) — introduces S-Mamba (Simplified "
        "Mamba) which transposes variate↔time axes before SSM block. Shows "
        "consistent gains on LTSF benchmarks where variate count > time steps. "
        "Our 104 features × 10 timesteps is exactly this regime (10× more "
        "variates than timesteps).\n"
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) — underlying block.\n"
        "Liu, Wu, Zhang et al. 2024 ICLR 'iTransformer: Inverted Transformers "
        "Are Effective for Time Series Forecasting' (arXiv:2310.06625) — "
        "parallel motivation for variate-axis attention; supports the "
        "variate-axis-first intuition for multi-feature TS."
    ),
    "hypothesis": (
        "Run Mamba Exp1 config (d_model=128, d_state=16, expand=2, 2-layer, "
        "lr=5e-4, bs=32, wd=0.1) with --mamba-variant s_mamba. Mechanism "
        "per Liu 2024: swapping variate↔time before the SSM pass means the "
        "state matrix evolves across FEATURES rather than TIMESTEPS — "
        "implicitly learning which macro signals (VIX, DXY, yield curve) "
        "matter in each regime. Because our test fold 2 is a crisis recovery "
        "period, the feature-importance ranking should shift (VIX rises in "
        "importance), and a model with variate-selective state can adapt. "
        "If the implementation is correct, we expect test fold 2 to improve "
        "toward ≥ 0 while retaining val fold 2 ≥ +1."
    ),
    "prediction": (
        "Composite +4.5 to +5.8. Probability of new global champion (> +6.4242): "
        "8%. Probability of beating Mamba Exp1 vanilla (+5.27): 45%. Expected "
        "test fold 2 lift: −0.98 → [−0.3, +0.8]. Key risk: naive S-Mamba "
        "(our placeholder implementation transposes but doesn't re-embed) may "
        "not actually deliver the paper's benefit — if composite drops far "
        "below +4.5, we revert and treat this as a negative signal."
    ),
    "_manual": True,
}

p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp2 (entry 153). Total: {len(ann)}")
