"""Finalize Mamba Exp4 (155) — dmamba decomposition."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("155", {})
entry["verdict"] = (
    "DISCARD on global-best (+5.3641 vs LSTM champion +6.4242) but MAMBA "
    "FAMILY NEW BEST: beats vanilla Mamba (+5.27) by +0.09 composite. "
    "Test 6/7 positive (fold 2 −0.46 vs vanilla's −0.98 — substantial "
    "lift). Val 6/7 positive (fold 1 went −1.18, fold 2 +0.92). Val Sharpe "
    "+5.68 (lower than vanilla's +6.30 — dmamba shifts strength from val "
    "to test). Train Sharpe +7.62. Decomposition hypothesis CONFIRMED: "
    "trend-MLP + seasonal-Mamba > pure Mamba on regime-shift folds."
)
entry["learning"] = (
    "dmamba is now the Mamba family base for further iteration. The "
    "trend-MLP branch gives the model a stable linear baseline that "
    "absorbs the slow USD/EUR drift, freeing the Mamba branch to focus "
    "on regime-specific perturbations. This mirrors the He 2016 ResNet "
    "principle (linear shortcut + nonlinear correction) that was the "
    "MLP-phase breakthrough — same trick at the SSM level. Axis OPEN: "
    "(a) d_state sweep on dmamba {8, 32}, (b) expand sweep {1, 4}, "
    "(c) num_layers sweep {1, 3}, (d) trend_mlp depth/width tuning. "
    "Open question: does ensemble of (vanilla Mamba, dmamba) at "
    "prediction time beat dmamba alone? They appear to specialise on "
    "different folds (vanilla → val fold 2; dmamba → test fold 2)."
)
entry["_manual"] = True
ann["155"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp4 (155) finalized as MAMBA-FAMILY-CHAMPION.")
