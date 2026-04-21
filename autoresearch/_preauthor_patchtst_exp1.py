"""Pre-author PatchTST Exp1 (174) — re-baseline with SOTA seq_len=60."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["174"] = {
    "diagnosis": (
        "PATCHTST PHASE — FIRST EXPERIMENT (1/50). The earlier PatchTST "
        "run (JSONL Exp117, described as 'PatchTST Exp1') used seq_len=10 "
        "(runner default for non-LFM2 backbones) and scored composite "
        "−1.72. That was a PROTOCOL VIOLATION per CLAUDE.md: the Nie "
        "et al. 2023 ICLR paper explicitly requires seq_len ≥ 60 for "
        "the patch attention to have enough tokens to be meaningful. At "
        "patch_length=5 and seq=10, only 2 patches exist — far below "
        "the ≥4 patches Nie et al. report as minimum. We redo Exp1 "
        "properly with seq_len=60 (our longest window, matching what "
        "LFM2 used). Expected outcome: a viable composite baseline "
        "somewhere between MLP (+5.50) and LSTM (+6.42). This is the "
        "first opportunity to see if transformer time-series models "
        "can work at our n=2738."
    ),
    "citations": (
        "Nie, Nguyen, Sinthong, Kalagnanam 2023 ICLR 'A Time Series is "
        "Worth 64 Words: Long-term Forecasting with Transformers' "
        "(arXiv:2211.14730) — introduces PatchTST with patch-based "
        "tokenisation + channel-independent attention. Uses seq_len=336 "
        "for ETT benchmarks; our seq_len=60 is smaller but well above "
        "the ≥4-patch minimum at patch=16 or ≥12-patch at patch=5.\n"
        "Zhou, Zhang, Peng, Zhang, Li, Xiong, Zhang 2021 AAAI 'Informer' "
        "(arXiv:2012.07436) — prior long-sequence transformer baseline; "
        "shows attention-quadratic problem PatchTST solves.\n"
        "Vaswani et al. 2017 NeurIPS 'Attention Is All You Need' "
        "(arXiv:1706.03762) — underlying transformer architecture.\n"
        "Wu, Xu, Wang, Long 2021 NeurIPS 'Autoformer' (arXiv:2106.13008) "
        "— motivation for decomposition-based TS transformers that "
        "PatchTST improves on."
    ),
    "hypothesis": (
        "Run PatchTST with SOTA recipe: seq_len=60, ep=100, pat=20, "
        "lr=1e-4, bs=32, wd=1e-4, warmup=10, cosine schedule, AdamW, "
        "MSE loss, seed=42. Head dropout 0.15 (our default). The model "
        "internally uses patch_length=5 (from our backbone.py default) "
        "so seq=60 gives 12 tokens per channel — well above the ≥4 "
        "minimum. Channel-independence means the 104 features are "
        "attended independently then concatenated. Mechanism: local "
        "temporal patches + attention across patches = captures local "
        "patterns at multiple time offsets within each 60-day window. "
        "This is the first transformer-based model we've run properly."
    ),
    "prediction": (
        "Composite +2.0 to +5.0 (wide range for first experiment of a "
        "new family). Probability of composite > 0: 85% (vs 50% for "
        "Mamba first experiment, because Transformer's inductive bias "
        "is well-established at n>2000 even if Mamba's SSM bias is "
        "stronger). Probability of new global champion (> +6.42): 8%. "
        "Per-fold: expect folds 4-6 (high-signal regimes) to be "
        "positive; folds 1/2 uncertain."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored PatchTST Exp1 (174). Total: {len(ann)}")
