"""Backfill Exp173 verdict + learning."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
e = ann["173"]
e["verdict"] = (
    "DISCARD. Composite +3.0213 vs champion +5.5996 (-2.58 — large)."
    " Test 6/7 positive; val 5/7 (folds 1, 2 negative). Doubling head "
    "dropout broke the model."
)
e["learning"] = (
    "head_dropout axis CLOSED at 0.1. Mamba is much more sensitive to "
    "head dropout than LSTM (LSTM peak 0.25, Mamba peak 0.1). All HP "
    "axes now confirmed at Mamba paper defaults. Champion held: dmamba "
    "expand=4 nl=2 d_state=16 lr=5e-4 wd=0.1 bs=32 hd=0.1 +5.5996."
)
e["_manual"] = True
ann["173"] = e
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Backfilled 173.")
