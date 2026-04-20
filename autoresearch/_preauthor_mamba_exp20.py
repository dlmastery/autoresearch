"""Pre-author Mamba Exp20 (171): dmamba expand=4 wd=0.05."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["171"] = {
    "diagnosis": (
        "lr axis closed at 5e-4. Probe wd axis. Champion uses wd=0.1 "
        "(Mamba paper default). Try wd=0.05 (half) — less explicit "
        "regularisation, may help if model is currently over-regularised."
    ),
    "citations": (
        "Loshchilov & Hutter 2019 ICLR (AdamW arXiv:1711.05101) — "
        "decoupled wd; perturbations should be log-spaced.\n"
        "Gu & Dao 2024 (Mamba arXiv:2312.00752) — wd=0.1 default for "
        "language; for TS, MambaTS (Cai 2024 arXiv:2405.16440) uses "
        "wd in [0.01, 0.1] depending on horizon."
    ),
    "hypothesis": (
        "Run dmamba expand=4 with --wd 0.05 (half champion). Mechanism: "
        "weaker L2 shrinkage allows model to retain larger weights → "
        "more expressive features. Risk: overfit if our n=2738 is too "
        "small for reduced regularisation."
    ),
    "prediction": (
        "Composite +5.0 to +5.7. Probability of beating champion: 30%. "
        "Most informative: train-test gap. Higher gap means overfit; "
        "neutral or smaller gap means clean win."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp20 (171). Total: {len(ann)}")
