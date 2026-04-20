"""Finalize Mamba Exp6 (157)."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("157", {})
entry["verdict"] = (
    "DISCARD. Composite +4.5319 (test +4.63, val +4.83). −0.83 vs dmamba "
    "d_state=16 champion. Test 6/7 positive (fold 1 = −0.03). Val 7/7 "
    "positive (fold 1 +0.02 marginal). Test fold 4 collapsed +9.65 → +6.15 "
    "and fold 3 +5.45 → +5.00. Train Sharpe +7.03 vs +7.62 — train also "
    "regressed, confirming under-capacity rather than 'cleanly regularised'."
)
entry["learning"] = (
    "d_state axis CLOSED. d_state=16 is the optimum at our n=2738 / "
    "seq_len=10 / 104 features. Both 8 (under) and 32 (over) hurt by "
    "≥−0.83 composite. Total composite range across d_state ∈ {8, 16, 32}: "
    "1.16. d_state in dmamba is highly sensitive — more so than in many "
    "transformer ablations where 2× capacity is usually neutral. Likely "
    "because the seasonal-Mamba branch already runs on residuals (after "
    "trend-MLP subtraction), so its capacity demand is small. Next: "
    "explore expand axis {1, 4} to see if inner-dim sensitivity is similar."
)
entry["_manual"] = True
ann["157"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp6 (157) finalized.")
