"""Pre-author Mamba Exp11 (162): dmamba expand=4 num_layers=3."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["162"] = {
    "diagnosis": (
        "Depth at 1 and 2 layers tied (+5.60). Test depth=3 to either "
        "open or close the axis. LSTM at depth=3 collapsed catastrophically "
        "(+1.64 from +6.42). Mamba may handle depth better because each "
        "block has a residual connection (pre-norm + skip). Hypothesis: "
        "depth=3 either neutral (axis stays open) or worse (axis closes "
        "definitively at 1-2)."
    ),
    "citations": (
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) — uses 24+ layers "
        "for language; depth scaling is well-behaved due to residual "
        "connections.\n"
        "He, Zhang, Ren, Sun 2016 CVPR 'Deep Residual Learning for Image "
        "Recognition' (arXiv:1512.03385) — residual connections enable "
        "deep stacking without degradation.\n"
        "Empirical: LSTM 3-layer collapsed to +1.64 at our n=2738 — "
        "without residual connections in the LSTM stack. Mamba HAS "
        "residuals so should not collapse the same way."
    ),
    "hypothesis": (
        "Run dmamba expand=4 with --num-layers 3. 3 stacked Mamba blocks "
        "with pre-norm residuals. Mechanism: deeper temporal abstraction. "
        "Risk at our small n: 3× the seasonal-branch parameters may "
        "overfit. Residuals should prevent train collapse but not "
        "generalisation collapse."
    ),
    "prediction": (
        "Composite +4.5 to +5.7. Probability of new champion (>+5.60): "
        "20%. Probability of catastrophic collapse like LSTM 3-layer "
        "(<+3): 10% (much lower because residuals)."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp11 (162). Total: {len(ann)}")
