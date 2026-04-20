"""Pre-author Mamba Exp19 (170): dmamba expand=4 lr=1e-3 (LSTM-style high)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["170"] = {
    "diagnosis": (
        "lr=3e-4 hurt (+4.14). Test the upper end with lr=1e-3 (LSTM "
        "champion's LR). Mamba paper recommends 5e-4 but the LSTM "
        "champion at lr=1e-3 was clearly better than at 5e-4, so it's "
        "worth one experiment to see if Mamba behaves similarly."
    ),
    "citations": (
        "Gu & Dao 2024 COLM 'Mamba' (arXiv:2312.00752) — recommends "
        "5e-4 default but uses 1e-3 for some larger models.\n"
        "Loshchilov & Hutter 2019 ICLR (AdamW arXiv:1711.05101) — lr "
        "scaling principles.\n"
        "Empirical: LSTM champion at lr=1e-3 (Fischer & Krauss 2018 "
        "default), but Mamba and LSTM have different gradient scales "
        "due to selective scan vs matrix multiply."
    ),
    "hypothesis": (
        "Run dmamba expand=4 with --lr 1e-3 (2× champion default). "
        "Mechanism: faster convergence, possibly different basin. Risk: "
        "instability from large gradient updates through the cumulative "
        "scan. With wd=0.1 and warmup=10, should be manageable."
    ),
    "prediction": (
        "Composite +4.5 to +5.6. Probability of beating champion: 25%. "
        "Risk: optimization divergence (early NaN losses)."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp19 (170). Total: {len(ann)}")
