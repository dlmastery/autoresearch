"""Pre-author Mamba Exp9 (160): dmamba expand=4 + bs=16 (Keskar trick)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["160"] = {
    "diagnosis": (
        "Architectural axes (variant, d_state, expand) explored. Champion: "
        "dmamba expand=4 +5.60. Now turn to optimization axes. The LSTM "
        "phase showed bs=16 (vs bs=32) gave +0.05 composite via the "
        "Keskar 2017 flat-minima effect. Test if the same trick lifts "
        "Mamba. Mamba's selective scan is naturally noisier than LSTM's "
        "matrix multiply, so the implicit-regularisation gain may be "
        "smaller — but worth one experiment to characterise."
    ),
    "citations": (
        "Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 ICLR 'On Large-"
        "Batch Training for Deep Learning: Generalization Gap and Sharp "
        "Minima' (arXiv:1609.04836) — small batches find flat minima.\n"
        "Smith, Kindermans, Ying, Le 2018 ICLR 'Don't Decay the Learning "
        "Rate, Increase the Batch Size' (arXiv:1711.00489) — lr/bs noise "
        "scale framework. Our lr=5e-4 / bs=16 gives noise scale 3.1e-5, "
        "well within stable regime for 2-layer Mamba.\n"
        "Empirical evidence from this project: LSTM Exp29 bs=16 +6.37 "
        "beat bs=32 baseline by +0.013 composite (and started a chain "
        "that ended at LSTM Exp35 +6.42 champion)."
    ),
    "hypothesis": (
        "Run dmamba expand=4 (current Mamba champ) with --batch-size 16. "
        "All other params identical: d_model=128, d_state=16, 2-layer, "
        "lr=5e-4, wd=0.1, warmup=10, ep=100, pat=20, seed=42. Mechanism: "
        "halving batch doubles gradient noise per step → flatter "
        "convergence basin → marginally better test generalisation. "
        "Risk at small Mamba: noise may destabilise the SSM scan; "
        "watch for fold 2 regression as in Exp30 (LSTM bs=8 destabilised)."
    ),
    "prediction": (
        "Composite +5.4 to +5.9. Probability of beating dmamba expand=4 "
        "(+5.60): 50%. Probability of new global champion (>+6.42): 10% "
        "(this is the most likely path to overtake LSTM at this point). "
        "Most informative: per-fold balance — does small-batch maintain "
        "the all-14-positive property?"
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp9 (160). Total: {len(ann)}")
