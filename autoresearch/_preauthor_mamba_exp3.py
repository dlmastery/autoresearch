"""Pre-author Mamba Exp3 (JSONL 154): real s_mamba re-run."""
import json
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

ann["154"] = {
    "diagnosis": (
        "Exp2 s_mamba was a null experiment — no-op placeholder in the branch. "
        "Fixed SelectiveSSM._forward_s_mamba to actually scan across CHANNELS "
        "(d_inner axis) instead of time. For our 104-feature × 10-timestep "
        "input, the d_inner=256 channel axis is much longer than L=10, so "
        "channel-axis SSM routing has more expressivity. Hypothesis from "
        "Exp1: vanilla Mamba fixes val fold 2 because feature-regime "
        "selectivity matters. S-Mamba doubles down on this: state evolves "
        "across features, with content-dependent gating. Target: beat "
        "vanilla Mamba +5.27 by improving the currently-weakest test fold 2 "
        "(−0.98) via better variate-axis state routing."
    ),
    "citations": (
        "Liu, Wang, Tang, Zhou, Wang 2024 'Is Mamba Effective for Time Series "
        "Forecasting?' (arXiv:2403.11144) — S-Mamba: scan across variates "
        "instead of time. Shows consistent gains on LTSF where #variates > "
        "#timesteps.\n"
        "Liu et al. 2024 ICLR 'iTransformer' (arXiv:2310.06625) — parallel "
        "motivation: inverted attention across variates beats temporal "
        "attention for multivariate forecasting with limited seq length.\n"
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) — underlying "
        "selective-state block.\n"
        "Implementation note: our _forward_s_mamba pools over time via mean "
        "before computing selective params (simplified; paper uses a per-"
        "channel linear). We accept a small fidelity loss in exchange for "
        "minimal added parameters — if results are good, we can upgrade "
        "later."
    ),
    "hypothesis": (
        "Run Exp1 config with --mamba-variant s_mamba now that the branch "
        "executes a real variate-axis scan. Mechanism: the SSM hidden state "
        "h is now [B, L, d_state] and evolves as the scan iterates through "
        "d_inner=256 channels, applying per-channel gates (dA, dB from "
        "mean-pooled features). This should learn feature-importance "
        "routing — e.g., downweight low-signal channels during crisis "
        "regimes. Prediction: composite improves or degrades by ±0.5. If it "
        "improves, S-Mamba is a real signal; if flat, the simplified pool "
        "kills the Liu 2024 benefit."
    ),
    "prediction": (
        "Composite +4.5 to +5.8. Probability of beating vanilla Mamba +5.27: "
        "40%. Probability of new global champion (>+6.42): 5%. Per-fold: test "
        "fold 2 most informative — if it stays ≤ −0.5, the S-Mamba variate-"
        "axis hypothesis doesn't hold for our data."
    ),
    "_manual": True,
}

p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp3 (entry 154). Total: {len(ann)}")
