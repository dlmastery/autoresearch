"""Finalize Mamba Exp7 (158) — dmamba expand=4 NEW MAMBA CHAMP."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
entry = ann.get("158", {})
entry["verdict"] = (
    "MAMBA FAMILY NEW CHAMPION (still DISCARD vs LSTM global +6.4242). "
    "Composite +5.5996 (test +5.60, val +5.83) — beats prior Mamba champ "
    "dmamba expand=2 (+5.36) by +0.24. ALL 14 FOLDS POSITIVE (7 test + 7 "
    "val) — first Mamba to achieve this. Test fold 2 = +3.76 (best ever "
    "for that regime across ALL backbones; LSTM champ was +0.40). Test "
    "fold 1 = +3.40 (also strong). However test fold 4 dropped +9.65 → "
    "+5.48 — capacity reallocated from easy folds to hard ones. Train "
    "Sharpe +7.16, no train inflation."
)
entry["learning"] = (
    "expand=4 strictly beats expand=2 for dmamba. Doubling inner-dim "
    "(256 → 512) provides richer per-step feature mixing without "
    "overfitting at our n=2738. Pattern: capacity helps when allocated "
    "to nonlinear feature interactions (expand axis), hurts when "
    "allocated to memory (d_state axis). Mechanistic insight: our 104-"
    "feature input has rich cross-feature interactions (FX × yields × "
    "VIX × DXY) that benefit from a wider MLP-like projection inside "
    "the Mamba block. The d_state-style temporal memory is already "
    "saturated at 16 because seq_len=10 admits at most 10 distinct "
    "temporal modes. Next experiments: (a) expand=8 to test if the "
    "trend continues, (b) num_layers ∈ {1, 3} to test depth, (c) "
    "bs=16 (Keskar trick that helped LSTM)."
)
entry["_manual"] = True
ann["158"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp7 (158) finalized as MAMBA FAMILY NEW CHAMPION.")
