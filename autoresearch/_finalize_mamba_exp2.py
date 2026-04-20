"""Finalize Mamba Exp2 (153) honestly — the s_mamba branch was a no-op placeholder."""
import json
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

entry = ann.get("153", {})
entry["verdict"] = (
    "NULL-EXPERIMENT. Composite +5.2714 BIT-IDENTICAL to Exp1 vanilla. Root "
    "cause: my s_mamba branch in SelectiveSSM is currently a no-op placeholder "
    "(x_in.transpose(1,2).transpose(1,2) is identity). The correct S-Mamba "
    "implementation per Liu 2024 arXiv:2403.11144 requires transposing the "
    "variate↔time axes BEFORE the SSM pass (so state evolves across features) "
    "and transposing back BEFORE the output projection. Also needs a per-variate "
    "embedding. Not implemented here. This is a process failure, not a "
    "science result — I should have either implemented s_mamba properly or "
    "not exposed the flag until I did."
)
entry["learning"] = (
    "Process lesson: don't expose a CLI flag whose backend is a no-op. "
    "Either gate the flag with NotImplementedError or complete the "
    "implementation. Going forward: (a) the SelectiveSSM s_mamba branch "
    "needs real variate-axis swap + re-embedding; (b) for now mark the "
    "variant as NOT-YET-IMPLEMENTED in the runner help string; (c) move to "
    "dmamba variant (Exp3) which DOES have a real implementation (trend MLP "
    "+ seasonal Mamba). Science-wise: Exp2 contributes nothing beyond Exp1. "
    "No axis opened or closed. Treat the Mamba-50 budget as advanced by 0."
)
entry["_manual"] = True
ann["153"] = entry
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print("Mamba Exp2 (153) annotation finalized as NULL-EXPERIMENT.")
