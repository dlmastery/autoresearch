"""Finalize Mamba Exp10 (161) — num_layers=1."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("161", {})
entry["verdict"] = (
    "ESSENTIALLY TIED with champion (DISCARD by 0.0007). Composite "
    "+5.5989 vs +5.5996. Test 7/7 positive (test Sharpe +5.88 — "
    "actually higher than 2-layer's +5.60!). Val 5/7 positive (folds "
    "1, 2 marginally negative). Half the parameters, 35% faster training "
    "(173s vs 263s). For deployment: 1-layer is the better pick on cost-"
    "performance grounds. For composite leaderboard: 2-layer wins by "
    "robustness margin."
)
entry["learning"] = (
    "Depth axis: dmamba is essentially flat between 1 and 2 layers — "
    "very different from LSTM (1L was −2.85 worse). Mechanistic "
    "interpretation: dmamba's trend-MLP branch absorbs much of what a "
    "second seasonal-Mamba layer would otherwise capture. Practical "
    "implication: the trend-MLP is doing more work than expected. "
    "OPEN QUESTION: would deepening the trend-MLP (3-layer instead of "
    "2-layer) lift composite? Worth a code change at some point. "
    "Next: num_layers=3 for upper bound; then move to multi-seed "
    "variance (most informative remaining experiment cluster)."
)
entry["_manual"] = True
ann["161"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp10 (161) finalized.")
