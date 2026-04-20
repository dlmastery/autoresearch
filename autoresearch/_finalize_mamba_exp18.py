"""Finalize Mamba Exp18 (169) — lr=3e-4."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
e = ann.get("169", {})
e["verdict"] = (
    "DISCARD. Composite +4.1448 vs champion +5.5996 (−1.45). Test 6/7 "
    "positive (fold 1 −0.40), val 5/7 positive (folds 1, 2 negative). "
    "Train Sharpe +7.53 (close to champion +7.16) — TRAIN didn't drop "
    "but val/test did, ruling out under-convergence. Lewkowycz 2020 "
    "flat-minima hypothesis NOT confirmed at our scale: lower LR found "
    "different basin but worse generalisation."
)
e["learning"] = (
    "lr axis lower bound: 3e-4 hurts substantially. Sweep so far: "
    "{3e-4: +4.14, 5e-4: +5.60 champ}. Try lr=1e-3 next (LSTM-style "
    "high LR) to map the upper bound. If 1e-3 also hurts, lr axis "
    "closes at 5e-4 (Mamba paper default)."
)
e["_manual"] = True
ann["169"] = e
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp18 (169) finalized.")
