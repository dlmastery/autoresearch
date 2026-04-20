"""Backfill reasoning_annotations.json from experiment_log.jsonl.

Parses descriptions (which contain citations in parentheses) and composes
a best-effort {diagnosis, citations, hypothesis, prediction, verdict, learning}
entry for every experiment. Entries that already have manual annotations are
preserved.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

RESULTS = Path(__file__).parent / "autoresearch_results"
JSONL = RESULTS / "experiment_log.jsonl"
OUT = RESULTS / "reasoning_annotations.json"


CITATION_HINTS = {
    # Papers → plain descriptions
    "He2016": "He et al. 2016 (Deep Residual Learning / ResNet) — skip connections for small-data generalization",
    "Fischer&Krauss2018": "Fischer & Krauss 2018 EJOR (financial LSTM, SOTA lr=1e-3, ep=100, pat=15)",
    "Fischer&Krauss2018 proper": "Fischer & Krauss 2018 EJOR — full SOTA recipe (ep=100+, patience 15+)",
    "Srivastava2014": "Srivastava et al. 2014 JMLR (Dropout) — 0.2-0.5 for dense layers",
    "Zaremba2014": "Zaremba et al. 2014 (Recurrent NN Regularization) — LSTM L2/dropout",
    "Gu2020": "Gu, Kelly & Xiu 2020 RFS (Empirical Asset Pricing via ML) — financial MLP/LSTM recipes",
    "Lewkowycz2020": "Lewkowycz et al. 2020 ICML (Large LR Phase) — flat minima generalize",
    "Qin2017": "Qin et al. 2017 (DA-RNN Dual-Stage Attention) — causal / single-direction for trading",
    "Bao2017": "Bao et al. 2017 (Financial TS Prediction) — seq=20",
    "Huber1964": "Huber 1964 — robust loss for fat tails",
    "Kendall&Gal2017": "Kendall & Gal 2017 NeurIPS (Heteroscedastic uncertainty)",
    "Ioffe2015": "Ioffe & Szegedy 2015 (Batch Normalization)",
    "Goyal2017": "Goyal et al. 2017 (Accurate Large Minibatch SGD) — warmup stabilization",
    "Graves2013": "Graves et al. 2013 (Deep RNNs for speech) — stacked multi-layer",
    "Cho2014": "Cho et al. 2014 (GRU — fewer gates than LSTM)",
    "Chung2014": "Chung et al. 2014 (empirical gated RNN evaluation)",
    "Ba2016": "Ba et al. 2016 (Layer Normalization)",
    "Nie2023": "Nie et al. 2023 ICLR (PatchTST) — patch-based self-attention for TS",
    "Stirn2023": "Stirn et al. 2023 (Faithful Heteroscedastic Regression) — log-var clamping",
    "Smith2018": "Smith et al. 2018 (Don't Decay LR, Increase BS) — smoother grads",
    "Tan2019": "Tan & Le 2019 EfficientNet — width scaling",
    "Hu2022": "Hu et al. 2022 (LoRA) — adapter fine-tuning",
    "Devlin2019": "Devlin et al. 2019 (BERT) — fine-tuning conventions",
    "LopezDePrado2018": "Lopez de Prado 2018 (Advances in Financial ML) — purged CV, regime shifts",
    "LiquidAI2024": "Liquid AI 2024 (LFM2 technical report)",
}


def extract_citations(desc: str) -> str:
    """Pull references from description. Handles (Author YEAR) and bare mentions."""
    # Matches in parens
    parens = re.findall(r"\(([^)]+)\)", desc)
    hits = set()
    for p in parens:
        for tag, full in CITATION_HINTS.items():
            if tag.lower().replace(" ", "") in p.lower().replace(" ", ""):
                hits.add(full)
    # Also scan the whole description
    for tag, full in CITATION_HINTS.items():
        if tag.lower().replace(" ", "") in desc.lower().replace(" ", ""):
            hits.add(full)
    if hits:
        return "; ".join(sorted(hits))
    # Fallback: raw parens content
    if parens:
        return "; ".join(parens)
    return ""


def guess_hypothesis(desc: str) -> str:
    """Best-effort hypothesis phrasing from description keywords."""
    d = desc.lower()
    out = []
    # Extract the "change" from the description
    if "lr=" in d:
        m = re.search(r"lr=([0-9.e-]+)", d)
        if m:
            out.append(f"Learning rate = {m.group(1)}")
    if "hd=" in d:
        m = re.search(r"hd=([0-9.]+)", d)
        if m:
            out.append(f"Head dropout = {m.group(1)}")
    if "wd=" in d:
        m = re.search(r"wd=([0-9.e-]+)", d)
        if m:
            out.append(f"Weight decay = {m.group(1)}")
    if "bs=" in d:
        m = re.search(r"bs=([0-9]+)", d)
        if m:
            out.append(f"Batch size = {m.group(1)}")
    if "ep=" in d or "epochs=" in d:
        m = re.search(r"ep(?:ochs)?=([0-9]+)", d)
        if m:
            out.append(f"Epochs = {m.group(1)}")
    if "seq=" in d or "seq_len=" in d:
        m = re.search(r"seq(?:_len)?=([0-9]+)", d)
        if m:
            out.append(f"Sequence length = {m.group(1)}")
    if "huber=" in d:
        m = re.search(r"huber=([0-9.]+)", d)
        if m:
            out.append(f"Huber delta = {m.group(1)}")
    if "hidden=" in d:
        m = re.search(r"hidden=([0-9]+)", d)
        if m:
            out.append(f"Hidden size = {m.group(1)}")
    if "seed=" in d:
        m = re.search(r"seed=([0-9]+)", d)
        if m:
            out.append(f"Seed = {m.group(1)}")
    if "warmup" in d:
        out.append("With LR warmup")
    if "residual" in d:
        out.append("Residual skip-connection architecture")
    if "bidirectional" in d:
        out.append("Bidirectional")
    if "unidirectional" in d or "causal" in d:
        out.append("Unidirectional / causal")
    if "gru" in d:
        out.append("GRU cell (2-gate instead of LSTM 3-gate)")
    if "layernorm" in d:
        out.append("LayerNorm input normalization")
    if "stacked" in d or "3-layer" in d or "num_layers" in d:
        out.append("Stacked multi-layer RNN")
    if "het_loss" in d or "heteroscedastic" in d:
        out.append("Heteroscedastic loss (Kendall & Gal)")
    if "reproduce" in d.lower() or "rerun" in d.lower() or "repro" in d.lower():
        out.append("Reproduction check of prior champion at different seed")
    return "; ".join(out) if out else desc.split(":", 1)[-1].strip()[:200]


def verdict_line(status: str, composite: float, global_best: float) -> str:
    if composite > global_best - 1e-9 and status == "KEEP":
        return f"GLOBAL CHAMPION (composite {composite:+.3f})"
    if status == "KEEP":
        return f"KEEP — within-backbone best at time of run (composite {composite:+.3f})"
    return f"DISCARD — composite {composite:+.3f} did not beat then-current best"


def main():
    existing = {}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    annotations = dict(existing)
    entries = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    global_best = -1e9
    for e in entries:
        exp_num = str(e.get("experiment_num", ""))
        if not exp_num:
            continue
        composite = float(e.get("composite", 0) or 0)
        global_best = max(global_best, composite)
        # Don't clobber manual annotations
        if exp_num in existing and existing[exp_num].get("_manual"):
            continue
        desc = e.get("description", "")
        ann = {
            "diagnosis": f"{e.get('backbone','?')} experiment #{exp_num}: {desc}",
            "citations": extract_citations(desc) or "(no explicit citation — starting-point or ad hoc)",
            "hypothesis": guess_hypothesis(desc),
            "prediction": "(auto-backfilled: specific prediction not recorded)",
            "verdict": verdict_line(e.get("status", "DISCARD"), composite, global_best),
            "learning": f"Result: composite {composite:+.4f}, test Sharpe {float(e.get('sharpe',0) or 0):+.4f}. Status: {e.get('status', '?')}.",
        }
        annotations[exp_num] = ann

    OUT.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"Wrote {len(annotations)} annotations to {OUT}")
    print(f"Top global composite across all experiments: {global_best:+.4f}")


if __name__ == "__main__":
    main()
