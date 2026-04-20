"""Pre-author Mamba Exp18 (169): dmamba expand=4 lr=3e-4 (lower than 5e-4)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["169"] = {
    "diagnosis": (
        "Architectural axes (variant, d_state, expand, num_layers, bs) "
        "all closed. Multi-seed variance characterised: champion seed=42 "
        "is +1.4 sigma above 7-seed mean. Now probe optimization axes "
        "starting with lr. Champion uses lr=5e-4 (Mamba paper default). "
        "Try lr=3e-4 (60% of champion) — slower convergence but per "
        "Lewkowycz 2020 may find flatter minima with better val/test "
        "behavior on hard folds."
    ),
    "citations": (
        "Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari 2020 ICML 'The "
        "Large Learning Rate Phase of Deep Learning' (arXiv:2003.02218) "
        "— low LR finds flatter basins.\n"
        "Smith 2017 'A disciplined approach to neural network "
        "hyper-parameters: Part 1 -- learning rate, batch size, momentum, "
        "and weight decay' (arXiv:1803.09820) — LR range tests.\n"
        "Loshchilov & Hutter 2019 ICLR 'Decoupled Weight Decay "
        "Regularization' (arXiv:1711.05101) — AdamW lr scaling."
    ),
    "hypothesis": (
        "Run dmamba expand=4 with --lr 3e-4 (vs 5e-4 default). All other "
        "params identical: d_model=128, d_state=16, num_layers=2, bs=32, "
        "wd=0.1, warmup=10, ep=100, pat=20, seed=42. Mechanism: 60% "
        "smaller step finds flatter basin per Lewkowycz 2020. May "
        "regularise mildly without losing convergence at our 100 epoch "
        "budget."
    ),
    "prediction": (
        "Composite +5.0 to +5.7. Probability of beating champion +5.5996: "
        "30%. Most informative: train Sharpe — if it stays at +7.0+ "
        "while test holds, lr=3e-4 is a clean win."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp18 (169). Total: {len(ann)}")
