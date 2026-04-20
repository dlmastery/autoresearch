"""Finalize Mamba Exp19 (170)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
e = ann.get("170", {})
e["verdict"] = (
    "DISCARD. Composite +4.8050 vs champion +5.5996 (−0.79). All 14 "
    "folds positive (rare!). Test fold 2 = +3.61 (second-best ever for "
    "this regime). Train Sharpe +6.72 — converged but to a different "
    "basin that has higher per-fold mean but lower min-fold-Sharpe."
)
e["learning"] = (
    "lr axis CLOSED. Sweep: {3e-4: +4.14, 5e-4: +5.60 champ, 1e-3: "
    "+4.81}. Symmetric degradation around 5e-4 — peak narrow. Mamba's "
    "selective scan is more lr-sensitive than LSTM's matrix multiply. "
    "Next: wd sweep ({0.05, 0.2}); head_dropout sweep ({0.05, 0.2}); "
    "then move to multi-seed ensemble work which is the highest-value "
    "remaining experiment cluster."
)
e["_manual"] = True
ann["170"] = e
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp19 (170) finalized.")
