"""Pre-author Mamba Exp22 (173): head_dropout=0.2."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["173"] = {
    "diagnosis": (
        "Probe head_dropout axis. Champion uses hd=0.1 (Mamba paper default "
        "for the prediction head). LSTM phase showed hd=0.25 is sharply "
        "optimal — but Mamba's intrinsic noise from selective scan + "
        "cosine-schedule training is different. Try hd=0.2 (2x default) to "
        "see if extra head regularisation helps."
    ),
    "citations": (
        "Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov 2014 "
        "JMLR 'Dropout' (arXiv:1207.0580) — head dropout 0.1-0.5 typical.\n"
        "Gal & Ghahramani 2016 ICML 'Dropout as Bayesian Approximation' "
        "(arXiv:1506.02142) — MC Dropout interpretation; dropout level "
        "affects both regularisation and uncertainty.\n"
        "Empirical from this project: LSTM phase showed hd=0.25 strictly "
        "best (0.20 -1.0 composite worse, 0.30 -0.4 composite worse)."
    ),
    "hypothesis": (
        "Run dmamba expand=4 with --head-dropout 0.2. Mechanism: 2x "
        "stronger head regularisation. May help if Mamba's learned "
        "features are over-specialised on training distribution."
    ),
    "prediction": (
        "Composite +5.0 to +5.7. Probability of beating champion: 30%."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored Mamba Exp22 (173). Total: {len(ann)}")
