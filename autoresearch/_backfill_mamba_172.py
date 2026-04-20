"""Backfill rich annotation for Exp172 (Mamba#21, wd=0.2)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["172"] = {
    "diagnosis": "wd sweep continuation. wd=0.05 hurt (+5.30); now test wd=0.2 (double champion). Symmetric exploration to characterise wd axis curvature.",
    "citations": "Loshchilov & Hutter 2019 ICLR 'Decoupled Weight Decay Regularization' (AdamW arXiv:1711.05101). Gu & Dao 2024 (Mamba arXiv:2312.00752) wd=0.1 default.",
    "hypothesis": "Run dmamba expand=4 with --wd 0.2 (2x champion). More explicit shrinkage; if model is currently under-regularised, this helps; otherwise hurts.",
    "prediction": "Composite +5.0 to +5.6.",
    "verdict": "DISCARD. Composite +5.1657 vs champion +5.5996 (-0.43). Test 7/7 positive but val fold 2 -1.45. wd axis closed: {0.05:+5.30, 0.1:+5.60 CHAMP, 0.2:+5.17}. Symmetric peak at 0.1.",
    "learning": "wd axis CLOSED. Mamba paper default wd=0.1 confirmed optimal. Next: head_dropout sweep, then warmup, then ensemble work.",
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Backfilled 172.")
