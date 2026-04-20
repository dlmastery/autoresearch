"""Enrich Exp151 verdict + learning after the run."""
import json
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

entry = ann.get("151", {})
entry["verdict"] = (
    "DISCARD — composite +5.5699 (−0.854 vs champion +6.4242, global best remains +6.4242 on lstm). "
    "Test 7/7 positive, test Sharpe +6.0367. Val 6/7 positive; val fold 1 at −1.96 was the killer — "
    "much worse than any other seed's val fold 1 result on this config. Val fold 2 actually positive "
    "(+0.51), which is unusual. Fold-level outcome is very different from seed=42 champion, confirming "
    "bs=16 seeds produce heterogeneous per-fold profiles."
)
entry["learning"] = (
    "Six-seed composite distribution at champion config (wd=7e-4 bs=16): "
    "{42:+6.42, 2024:+6.01, 13:+3.84, 77:+5.57} + adjacent wd=1e-3 seeds {0:+4.24, 99:+5.44}. "
    "Mean ≈ +5.25, std ≈ 0.93, median +5.51. Champion seed=42 is +1.26σ above mean — "
    "significant but not an outlier. The key insight: the 3-seed median of the wd=7e-4 champion "
    "config is ~+5.57, which is LOWER than the bs=32 wd=1e-3 champion's 3-seed median of +5.99. "
    "Purely on the median metric, bs=32 beats bs=16. The bs=16 champion wins only on peak, not "
    "on expected value. Updated deployment policy: use bs=32 for production (lower variance) but "
    "bs=16 in competitive leaderboard mode (higher ceiling). Next: one more seed (123) to hit 6 "
    "samples, then either ensemble (Lakshminarayanan 2017) or pivot to PatchTST."
)
entry["_manual"] = True
ann["151"] = entry

p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Exp151 verdict + learning enriched.")
