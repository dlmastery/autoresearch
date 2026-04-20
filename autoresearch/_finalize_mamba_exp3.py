"""Finalize Mamba Exp3 (JSONL 154) — real s_mamba variate-axis scan."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("154", {})
entry["verdict"] = (
    "DISCARD. Composite +5.1861 (test Sharpe +5.29, val Sharpe +6.36). "
    "Worse than vanilla Mamba Exp1 (+5.27, test Sharpe +5.37) by ~0.09 "
    "composite — within seed noise but directionally negative. Test 6/7 "
    "positive, val 7/7 positive. Test fold 2 went from −0.98 (vanilla) to "
    "−1.84 (s_mamba) — the variate-axis scan made the post-crash regime "
    "WORSE, not better. Val fold 1 lifted +0.89 → +1.08 and val fold 2 "
    "lifted +1.37 → +1.71 — gains on val but degradation on test. Test "
    "fold 1 also gained: +2.88 → +3.10. Mixed signal."
)
entry["learning"] = (
    "Liu 2024 S-Mamba hypothesis (variate-axis scan beats time-axis when "
    "n_features > seq_len) is NOT confirmed at our scale. Mechanism "
    "interpretation: with our simplified mean-pool over time before "
    "computing per-channel B/C/dt, we lose the per-timestep signal that "
    "vanilla Mamba uses. The full Liu 2024 implementation uses a "
    "learnable per-channel projection, not a pool — that's the fidelity "
    "loss. Verdict: DON'T pursue s_mamba further at our n=2738; revert "
    "to vanilla as the Mamba family base for HP tuning. Axis CLOSED. "
    "Next experiments will explore: (a) dmamba decomposition (already "
    "implemented properly), (b) d_state sweep on vanilla, (c) expand "
    "sweep on vanilla. Open question: would S-Mamba help if we had "
    "more variates (say 500-1000) and fewer training samples? Likely "
    "yes per Liu 2024 — but that's a different problem."
)
entry["_manual"] = True
ann["154"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp3 (154) finalized.")
