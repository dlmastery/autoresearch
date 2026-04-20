"""Finalize Mamba Exp9 (160) — bs=16 trick."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("160", {})
entry["verdict"] = (
    "DISCARD (within noise). Composite +5.5669 vs champion +5.5996 — "
    "delta −0.033, well within seed noise. ALL 14 FOLDS POSITIVE retained. "
    "Test fold 2 +2.65 (down from +3.76 at bs=32 — capacity reallocated). "
    "Val fold 6 dropped +11.43 → +7.04 (notable). Train Sharpe +6.31 "
    "(down from +7.16) — small batch slows train convergence as expected. "
    "No clear win, no clear loss."
)
entry["learning"] = (
    "Keskar 2017 small-batch trick gives ~0 effect on Mamba (LSTM gain "
    "was +0.013 — also tiny). Mechanistic interpretation: Mamba's "
    "selective scan introduces input-dependent noise via the dt gate "
    "which already implicitly regularises; doubling SGD noise via "
    "smaller batch is redundant. The bs axis is approximately CLOSED "
    "for Mamba — bs=32 (default per recipe) is fine, bs=16 acceptable "
    "but slower. Skip bs=8 (LSTM showed it destabilises). Next: depth "
    "axis (num_layers ∈ {1, 3})."
)
entry["_manual"] = True
ann["160"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp9 (160) finalized.")
