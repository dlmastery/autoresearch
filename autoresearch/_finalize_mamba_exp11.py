"""Finalize Mamba Exp11 (162) — num_layers=3."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("162", {})
entry["verdict"] = (
    "DISCARD. Composite +4.8264 (test +5.27, val +4.93). −0.77 vs "
    "champion. No catastrophic collapse like LSTM 3-layer (+1.64) — "
    "residuals saved us. Test 7/7 positive but val 6/7 (fold 2 −1.51). "
    "Train Sharpe +6.21 (down from +7.16 at 2L) — train regressed too, "
    "indicating optimization difficulty rather than pure overfit."
)
entry["learning"] = (
    "Depth axis FULLY CLOSED: 1L (+5.60), 2L (+5.60), 3L (+4.83). "
    "Mamba is depth-flat between 1-2 (trend-MLP absorbs the work) and "
    "degrades at 3. Quite different from LSTM (1L underfit, 2L peak, "
    "3L collapse). Pattern: dmamba's effective capacity is dominated "
    "by the trend-MLP + first Mamba layer; additional Mamba layers "
    "don't help and can hurt. ARCHITECTURAL AXES ALL CLOSED for "
    "Mamba family: variant=dmamba, d_state=16, expand=4, num_layers∈{1,2}, "
    "bs=32, lr=5e-4, wd=0.1. Champion is robust. Next: multi-seed "
    "variance study (seeds 0, 99, 7, 2024, 13, 77) to characterise "
    "the noise floor before any further HP changes can be trusted."
)
entry["_manual"] = True
ann["162"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp11 (162) finalized. Architectural axes ALL CLOSED.")
