"""Pre-author reasoning_annotations.json entry for UPCOMING Mamba Exp1
(JSONL exp will be #152). Must run BEFORE launching the experiment.
"""
import json
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

ann["152"] = {
    "diagnosis": (
        "MAMBA BACKBONE — FIRST EXPERIMENT (1/50). Transition from LSTM phase "
        "(halted at 46/50 at user direction with champion Exp35 wd=7e-4 bs=16 "
        "seed=42 composite +6.4242 test Sharpe +6.5242). The LSTM axes are "
        "exhausted: HP perturbations around champion yielded only noise, seed "
        "variance dominates, and the 3-seed median (+5.57) is below the bs=32 "
        "3-seed median (+5.99). Next structural lever is a fundamentally "
        "different sequence-modelling family: state-space models (SSMs). "
        "Mamba is the 2024 standard-bearer of the SSM resurgence and has "
        "demonstrated Pareto-dominance over Transformers on long-context "
        "benchmarks. At our n=2738 / seq_len=10 the computational advantage "
        "of Mamba is irrelevant — we test purely the inductive bias hypothesis: "
        "does selective state mixing handle regime-shift data better than the "
        "recurrent gating of LSTMs? Per-fold weakness the LSTM champion has "
        "(val fold 1 ≈ +0.46, val fold 2 ≈ 0.00) is our benchmark to beat. "
        "First-experiment goal: establish a literature-recipe baseline we can "
        "iterate from, not to immediately beat the champion."
    ),
    "citations": (
        "Gu & Dao 2024 COLM 'Mamba: Linear-Time Sequence Modeling with "
        "Selective State Spaces' (arXiv:2312.00752) — the canonical paper. "
        "Introduces input-dependent Δ, B, C that make the SSM 'selective', "
        "enabling content-aware state routing. Uses HiPPO-inspired A init "
        "(Gu et al. 2020 arXiv:2008.07669). Our implementation in "
        "model/backbone.py:SelectiveSSM follows Eq. 4 (ZOH discretisation) "
        "with the gated-MLP block from Section 3.\n"
        "Gu, Goel, Ré 2022 ICLR 'Efficiently Modeling Long Sequences with "
        "Structured State Spaces' (S4, arXiv:2111.00396) — theoretical "
        "foundation for diagonal SSMs.\n"
        "Dao & Gu 2024 ICML 'Transformers are SSMs: Generalized Models and "
        "Efficient Algorithms Through Structured State Space Duality' "
        "(Mamba-2, arXiv:2405.21060) — algebraic duality between SSMs and "
        "attention; motivates Mamba as a Transformer alternative.\n"
        "Cai, Jiang, Wu, Zhang, Wang 2024 NeurIPS 'MambaTS: Improved "
        "Selective State Space Models for Long-Term Time Series Forecasting' "
        "(arXiv:2405.16440) — shows Mamba beats PatchTST/FEDformer on 8 LTSF "
        "benchmarks; provides our starting recipe.\n"
        "Wang, Wu et al. 2024 'Is Mamba Effective for Time Series Forecasting?' "
        "(arXiv:2403.11144) — introduces S-Mamba channel-flipped variant that "
        "applies Mamba across variates instead of time. Motivates our "
        "`--mamba-variant s_mamba` flag.\n"
        "Liu et al. 2025 'DMamba: Decomposition-enhanced Mamba for Time Series "
        "Forecasting' (arXiv:2602.09081) — trend+seasonal decomposition. "
        "Motivates `--mamba-variant dmamba` flag."
    ),
    "hypothesis": (
        "Run canonical 2-layer Mamba (vanilla variant) on our champion recipe "
        "inputs: 104 features × seq_len=10, d_model=128, d_state=16, expand=2 "
        "(inner dim 256), with SOTA training config per CLAUDE.md recipe table "
        "(ep=100, pat=20, lr=5e-4, bs=32, wd=0.1, warmup=10, cosine schedule, "
        "AdamW, MSE loss, seed=42). Mechanism: the input-dependent Δ gate lets "
        "the model 'forget' information that is irrelevant to the current "
        "regime (Gu & Dao 2024 Sec 3.2), which is exactly the failure mode "
        "our LSTM struggles with on val fold 1/2 (crisis-era regime shifts). "
        "If selective state mixing is a better inductive bias than recurrent "
        "gating for our data, we expect first-experiment composite to land "
        "between PatchTST (−1.72, failed due to seq=10) and the LSTM "
        "champion (+6.42). If Mamba is as overfit-prone at small n as its "
        "Transformer cousins, we see a negative composite and must pivot "
        "to smaller d_state / fewer params."
    ),
    "prediction": (
        "Composite +1.0 to +4.5 (wide range because this is Experiment 1/50 "
        "of a new family). Probability of composite > 0: 70%. Probability of "
        "new global champion (> +6.4242): 5%. Per-fold: expect test folds 3-6 "
        "(high-signal regimes) to be positive; test fold 1/2 likely borderline. "
        "Train Sharpe likely high due to Mamba's expressivity; val-test gap "
        "probably wide — we'll diagnose overfit in Exp 2-3."
    ),
    "_manual": True,
}

p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp1 (entry 152). Total annotations: {len(ann)}")
