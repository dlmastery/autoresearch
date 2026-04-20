"""One-shot: merge hand-curated reasoning entries into reasoning_annotations.json
with _manual=True so backfill won't overwrite them."""
import json
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

curated = {
    "3": {
        "diagnosis": "Early LR sweep on LFM2 (frozen backbone + head fine-tuning).",
        "citations": "Devlin et al. 2019 (BERT fine-tuning LR conventions); Hu et al. 2022 (LoRA — low LR for adapters).",
        "hypothesis": "Lower LR than default 1e-4 improves head-only fine-tuning stability on small financial data.",
        "prediction": "Composite > +1.0.",
        "verdict": "KEEP — pre-residual era champion (+1.61).",
        "learning": "lr=3e-5 is the sweet spot for plain-Huber LFM2 head fine-tuning.",
        "_manual": True,
    },
    "20": {
        "diagnosis": "Fine-grained LR search around 3e-5 champion.",
        "citations": "Loshchilov & Hutter 2019 (AdamW); Smith 2017 (LR range tests).",
        "hypothesis": "lr=2e-5 slightly lower may better fit regime transitions.",
        "prediction": "Small improvement in composite.",
        "verdict": "KEEP-but-unreproducible: single-seed +1.77 yielded champion; multi-seed median ~+0.11.",
        "learning": "Seed variance dominates single-run results on LFM2. Established need for multi-seed protocol for future champions.",
        "_manual": True,
    },
    "66": {
        "diagnosis": "First residual skip-connection MLP experiment.",
        "citations": "He et al. 2016 (Deep Residual Learning / ResNet).",
        "hypothesis": "A linear shortcut + nonlinear residual branch dominates on small financial data; the linear path anchors a regime-agnostic baseline, letting the nonlinear branch learn corrections.",
        "prediction": "Composite > +2.0.",
        "verdict": "BREAKTHROUGH +4.67 — 4-line code change jumped composite from +0.78 to +4.67.",
        "learning": "Architectural changes dominate HP tuning on low-SNR financial data. One skip connection beat 47 prior MLP HP experiments combined.",
        "_manual": True,
    },
    "96": {
        "diagnosis": "MLP residual champion reproduction with enhanced metrics (precision/recall/F1/MCC + trade-level CSVs).",
        "citations": "He et al. 2016 (architecture); Srivastava et al. 2014 (hd=0.15).",
        "hypothesis": "Reproducible exact match with champion Exp32 config.",
        "prediction": "Composite +5.499, test Sharpe +6.2113.",
        "verdict": "KEEP — pre-LSTM global champion, exact reproduction.",
        "learning": "Deterministic seed=0 reproduces MLP champion to 4 decimal places.",
        "_manual": True,
    },
    "108": {
        "diagnosis": "First LSTM autoresearch baseline run (previous LSTM data from ablation only, 5-epoch non-SOTA).",
        "citations": "Fischer & Krauss 2018 EJOR (canonical financial LSTM, SOTA lr=1e-3, stacked 2-layer, bidirectional).",
        "hypothesis": "Classical financial-LSTM recipe with our proven training discipline (ep=50, pat=10, hd=0.15) should give +0.5 to +2.5 baseline.",
        "prediction": "Composite +0.5 to +2.5.",
        "verdict": "KEEP — LSTM backbone baseline +4.12 (exceeded expectations).",
        "learning": "LSTM's temporal inductive bias is strong for FX returns even at seq=10.",
        "_manual": True,
    },
    "110": {
        "diagnosis": "LSTM with proper SOTA epoch count per Fischer & Krauss.",
        "citations": "Fischer & Krauss 2018 EJOR — their experiments ran 100+ epochs with patience 15.",
        "hypothesis": "Exp108's early-stop at epoch 25 was premature; more patience may find better minimum.",
        "prediction": "Composite +4.5 to +5.5.",
        "verdict": "KEEP — LSTM improved to +5.06 via epoch bump alone.",
        "learning": "Per-backbone epoch counts matter: MLP 50 != LSTM 100. Literature SOTA epoch is backbone-specific.",
        "_manual": True,
    },
    "111": {
        "diagnosis": "LSTM with higher head dropout — regularizing prediction head while keeping recurrent capacity.",
        "citations": "Srivastava et al. 2014 JMLR (Dropout) — recommends 0.2-0.5 for dense layers.",
        "hypothesis": "hd=0.25 breaks the val/test tradeoff seen in MLP (regularizing heads fixes fold 2 without hurting other folds).",
        "prediction": "Composite > +5.06.",
        "verdict": "GLOBAL CHAMPION (+6.07) — beat MLP residual (+5.499).",
        "learning": "LSTM recurrent bias + head dropout combine synergistically (not redundantly). Fixed fold 2 test -1.75 -> +1.66 WITHOUT sacrificing fold 7.",
        "_manual": True,
    },
    "113": {
        "diagnosis": "LSTM with 10x stronger weight decay on top of champion.",
        "citations": "Zaremba et al. 2014 (Recurrent NN Regularization — L2 for LSTMs).",
        "hypothesis": "Slight L2 bump smooths the optimization landscape further without changing capacity.",
        "prediction": "Composite > +6.07.",
        "verdict": "GLOBAL CHAMPION (+6.10) — current global best, fold 5 test improved +10.31 -> +10.53.",
        "learning": "Marginal improvement confirms we are at a local Pareto optimum; wd=1e-4 locked in.",
        "_manual": True,
    },
    "114": {
        "diagnosis": "Half the LR for flatter minima seeking.",
        "citations": "Lewkowycz et al. 2020 ICML (The Large Learning Rate Phase) — lower LR finds flatter basins that generalize better.",
        "hypothesis": "lr=5e-4 finds flatter basin -> improved val fold 2 (the consistent weakness).",
        "prediction": "Val fold 2 > 0.",
        "verdict": "DISCARD — composite +4.95; val fully 7/7 positive (+6.88 best ever) but test dropped to 5/7.",
        "learning": "Flat minima help val metrics but can hurt test when val/test regimes diverge. Lewkowycz framework doesn't apply cleanly to regime-shifted financial time series.",
        "_manual": True,
    },
    "115": {
        "diagnosis": "Unidirectional (causal) LSTM — process only past->present.",
        "citations": "Qin et al. 2017 (DA-RNN dual-stage attention); Graves & Schmidhuber 2005 (original bidirectional context).",
        "hypothesis": "Causal matches live-trading direction and may reduce overfit to within-window future-to-past correlations.",
        "prediction": "Better val fold 2; test uncertain.",
        "verdict": "DISCARD — composite +5.00; val fold 2 hit best-ever +2.02, but test lost fold 2.",
        "learning": "Bidirectional LSTM wins on test; unidirectional wins on val. Classic val/test split — bidirectional captures more context.",
        "_manual": True,
    },
    "116": {
        "diagnosis": "Longer sequence for more temporal context.",
        "citations": "Bao et al. 2017 (Financial Time Series Prediction with seq=20).",
        "hypothesis": "Fold 2 (post-crash recovery) benefits from longer memory of pre-crash patterns.",
        "prediction": "Test fold 2 > 0.",
        "verdict": "DISCARD — composite +4.25; longer context hurt overall (loses training windows).",
        "learning": "seq=10 optimal for our n=2738; Bao's seq=20 requires substantially larger dataset.",
        "_manual": True,
    },
    "117": {
        "diagnosis": "First PatchTST SOTA baseline (transformer-based time series).",
        "citations": "Nie et al. 2023 ICLR (PatchTST).",
        "hypothesis": "SOTA config (lr=1e-4, ep=100, pat=20, hd=0.15) gives reasonable baseline.",
        "prediction": "Composite +1 to +4.",
        "verdict": "DISCARD — composite -1.72; transformer attention under-scaled at seq=10.",
        "learning": "At patch_length=5 and seq=10, only 2 attention tokens — far below the 4+ tokens Nie et al. report as minimum. Need seq>=60 for PatchTST to shine.",
        "_manual": True,
    },
    "118": {
        "diagnosis": "Stacked 3-layer bidirectional LSTM for multi-scale temporal modeling.",
        "citations": "Graves et al. 2013 (Speech Recognition with Deep RNNs).",
        "hypothesis": "Depth captures multi-scale temporal patterns not accessible to 2-layer.",
        "prediction": "Composite > +6.10.",
        "verdict": "DISCARD — composite +1.64; depth + small data mismatch.",
        "learning": "3 layers overfits on n=2738. 2-layer champion confirmed optimal for our data size. Graves's depth advantage emerges only at larger n.",
        "_manual": True,
    },
    "119": {
        "diagnosis": "GRU cell comparison — 2 gates vs LSTM's 3.",
        "citations": "Cho et al. 2014 (GRU); Chung et al. 2014 (empirical gated RNN evaluation).",
        "hypothesis": "GRU's simpler gating has fewer params, may reduce overfit on small data.",
        "prediction": "Composite near LSTM champion (~+6.10).",
        "verdict": "DISCARD — composite +4.59; GRU achieved 7/7 test positive but weaker Sharpe.",
        "learning": "LSTM's 3-gate architecture retains more signal at n=2738. GRU's parameter savings don't translate to gains here.",
        "_manual": True,
    },
    "120": {
        "diagnosis": "Upcoming: LayerNorm on LSTM input (post-standardization features).",
        "citations": "Ba et al. 2016 (Layer Normalization).",
        "hypothesis": "Normalizing feature statistics per-timestep may stabilize LSTM gate activations.",
        "prediction": "Small positive or neutral effect.",
        "verdict": "(pending)",
        "learning": "(pending)",
        "_manual": True,
    },
}

for k, v in curated.items():
    ann[k] = v

p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Annotations now: {len(ann)}, with {sum(1 for v in ann.values() if v.get('_manual'))} manually curated.")
