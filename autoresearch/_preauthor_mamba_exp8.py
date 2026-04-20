"""Pre-author Mamba Exp8 (159): dmamba expand=8 — push capacity higher."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["159"] = {
    "diagnosis": (
        "expand=4 just set the Mamba family champion (+5.60). The expand "
        "axis is the OPPOSITE of d_state — more inner-dim helps, more "
        "state hurts. Test if the trend continues with expand=8 (4× "
        "original inner-dim, 1024 inner). If it does, we have a clean "
        "monotonic axis to climb. If it doesn't, expand=4 is the local "
        "optimum and we close the axis."
    ),
    "citations": (
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) — original paper "
        "ablates expand ∈ {1, 2, 3} but not higher; appendix notes expand "
        "is mostly compute-vs-quality tradeoff.\n"
        "Tan & Le 2019 ICML 'EfficientNet' (arXiv:1905.11946) — width "
        "scaling exponent ≈ 1.1 for image; analogous for SSMs would "
        "predict diminishing returns past ~4× base width.\n"
        "Hoffmann et al. 2022 NeurIPS 'Chinchilla' (arXiv:2203.15556) — "
        "compute-optimal scaling; at fixed data, more parameters help "
        "only if optimization budget keeps up. Our ep=100 + cosine "
        "schedule should be sufficient at 1024 inner-dim."
    ),
    "hypothesis": (
        "Run dmamba (current Mamba champ) with --mamba-expand 8. All "
        "other params identical to Exp7: d_model=128, d_state=16, "
        "2-layer, lr=5e-4, bs=32, wd=0.1, warmup=10, ep=100, pat=20, "
        "seed=42. Mechanism: 1024 inner-dim allows even richer per-step "
        "feature mixing. Risk: may overfit on n=2738 if width grows "
        "faster than data; train-test gap will widen if so."
    ),
    "prediction": (
        "Composite +5.0 to +5.9. Probability of beating expand=4 (+5.60): "
        "35%. Probability of new global champion (>+6.42): 7%. Most "
        "informative: train-test gap. Train Sharpe at expand=4 was "
        "+7.16 (close to test +5.60). If expand=8 train rises >+8 with "
        "test ≤+5.5, we've hit the overfit cliff."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp8 (159). Total: {len(ann)}")
