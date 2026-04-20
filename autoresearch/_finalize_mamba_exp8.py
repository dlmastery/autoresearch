"""Finalize Mamba Exp8 (159) — dmamba expand=8 over-capacity."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("159", {})
entry["verdict"] = (
    "DISCARD. Composite +4.1225 (test +4.32, val +4.43). −1.48 vs "
    "expand=4 champion. Test 7/7 positive but all folds weakened — "
    "fold 2 collapsed +3.76 → +0.08, fold 1 +3.40 → +0.57. Val 5/7 "
    "positive (folds 1, 2 negative). Train Sharpe +6.93 (down from "
    "+7.16 at expand=4) — train ALSO regressed, indicating optimization "
    "instability rather than pure overfit."
)
entry["learning"] = (
    "expand axis CLOSED at 4. Sweep: {2: +5.36, 4: +5.60, 8: +4.12}. "
    "Sharp peak at 4. Doubling beyond 4 destabilises optimization (loss "
    "landscape gets harder, AdamW can't navigate within 100 epochs). "
    "Total range across expand: 1.48 composite — high sensitivity. "
    "Mamba family standings: dmamba expand=4 +5.60 > dmamba expand=2 "
    "+5.36 > vanilla +5.27 > s_mamba +5.19 > dmamba d_state=8 +4.53 > "
    "dmamba d_state=32 +4.20 > dmamba expand=8 +4.12. Next: try the "
    "Keskar 2017 small-batch trick (bs=16) on the dmamba expand=4 "
    "champion — this gave +0.05 to the LSTM champion, may give a similar "
    "boost here."
)
entry["_manual"] = True
ann["159"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp8 (159) finalized.")
