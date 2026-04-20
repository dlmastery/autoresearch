"""Finalize Mamba Exp5 (156) — dmamba d_state=32."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("156", {})
entry["verdict"] = (
    "DISCARD. Composite +4.1995 (test +4.50, val +5.45). −1.16 vs dmamba "
    "d_state=16 champion. Test 6/7 positive but test fold 1 collapsed "
    "(−0.67) and test fold 7 weakened (+2.85 vs +8.89). However test "
    "fold 2 IMPROVED to +2.03 — best fold-2 result for any Mamba so far. "
    "Val 5/7 positive (folds 1, 2 negative). Train Sharpe +6.73 vs +7.62 "
    "at d_state=16 — even train regressed, suggesting optimization is "
    "harder with more state, not just generalization."
)
entry["learning"] = (
    "d_state=32 over-capacity at n=2738 / seq_len=10. Pattern: model "
    "concentrates capacity on the hardest fold (2) but gives up on easy "
    "ones. Net effect negative. Axis insight: SSM state size needs to "
    "match the ratio (effective seq_len × n_modes). Our seq_len is only "
    "10 — there are not 32 distinct temporal modes to capture. Next: try "
    "d_state=8 (half), expect either marginal gain (regularisation) or "
    "marginal loss (under-capacity). Mamba family standings: dmamba "
    "d_state=16 (+5.36) > vanilla d_state=16 (+5.27) > s_mamba (+5.19) "
    "> dmamba d_state=32 (+4.20)."
)
entry["_manual"] = True
ann["156"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp5 (156) finalized.")
