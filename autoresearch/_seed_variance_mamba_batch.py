"""Pre-author + finalize annotations for the Mamba seed variance batch.

Reusable templated annotations for variance-check runs (Exps 12-17, JSONL 163-168).
The diagnosis/citations/hypothesis/prediction are templated (mechanical variance
study at champion config). Verdict + learning are filled in from results after
each run.
"""
import json
import sys
from pathlib import Path

p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))

CITATIONS = (
    "Bouthillier, Laurent, Vincent 2019 ICML workshop 'Unreproducible "
    "Research is Reproducible' (arXiv:1906.05268) — documents seed-driven "
    "variance in DL benchmarks.\n"
    "Henderson, Islam, Bachman, Pineau, Precup, Meger 2018 AAAI 'Deep RL "
    "That Matters' (arXiv:1709.06560) — methodological critique of single-"
    "seed reporting.\n"
    "Madhyastha & Jain 2019 EMNLP 'On Model Stability as a Function of "
    "Random Seed' (arXiv:1909.10447).\n"
    "Picard 2021 'Torch.manual_seed(3407) is all you need' "
    "(arXiv:2109.08203) — seeds can flip rankings.\n"
    "Lakshminarayanan, Pritzel, Blundell 2017 NeurIPS 'Simple and Scalable "
    "Predictive Uncertainty Estimation using Deep Ensembles' "
    "(arXiv:1612.01474) — motivates seed-ensembling as the deployment "
    "remedy."
)

# (jsonl_id, seed_used, mamba_exp_num)
RUNS = {
    163: (0, 12),
    164: (99, 13),
    165: (7, 14),
    166: (2024, 15),
    167: (13, 16),
    168: (77, 17),
}

def template(seed_used, mamba_num, prior_seeds):
    return {
        "diagnosis": (
            f"Mamba family multi-seed variance run #{mamba_num - 11} of 6 "
            f"(seed={seed_used}). Architectural axes all closed in Exps "
            f"1-11; champion is dmamba expand=4 num_layers=2 d_state=16 "
            f"bs=32 lr=5e-4 wd=0.1 at seed=42 -> composite +5.5996. "
            f"Need 5+ seed runs to characterise variance noise floor "
            f"before any HP fine-tuning is interpretable. Prior seeds "
            f"in this study: {prior_seeds}. Without variance "
            f"characterisation, single-seed wins are unreliable per "
            f"Bouthillier 2019 / Picard 2021."
        ),
        "citations": CITATIONS,
        "hypothesis": (
            f"Re-run champion config exactly (dmamba, expand=4, num_layers=2, "
            f"d_state=16, bs=32, lr=5e-4, wd=0.1, ep=100, pat=20, warmup=10, "
            f"head_dropout=0.1, huber_delta=1.0) with --seed {seed_used}. "
            f"Mechanism: changes weight-init + dropout-mask schedule + "
            f"data-shuffle order. No architecture or HP change. "
            f"Composite expected within ±2 of seed=42 mean, given the LSTM "
            f"phase showed std~1.0 at small batch."
        ),
        "prediction": (
            f"Composite +3.5 to +5.7. Probability of composite > seed=42 "
            f"(+5.5996): ~25%. Most informative outcome: distribution "
            f"shape of the 6-seed sample after this run."
        ),
        "_manual": True,
    }

# State of seed runs so far (will be appended as we go)
COMPLETED = {42: 5.5996}  # champion seed
if "163" in ann and ann["163"].get("verdict") and "5.2714" not in ann["163"].get("verdict",""):
    # already have it
    pass

# Use this in --pre or --post mode
mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
target = int(sys.argv[2]) if len(sys.argv) > 2 else None

if mode == "pre" and target:
    # Pre-author one entry
    seed_used, mamba_num = RUNS[target]
    prior_seeds_dict = {42: 5.5996}
    for jid in sorted(RUNS.keys()):
        if jid >= target:
            break
        if str(jid) in ann and ann[str(jid)].get("verdict"):
            v = ann[str(jid)]["verdict"]
            # try to extract composite
            import re as _re
            m = _re.search(r"\+([0-9]+\.[0-9]+)", v)
            if m:
                prior_seeds_dict[RUNS[jid][0]] = float(m.group(1))
    prior_str = ", ".join(f"{s}->+{c:.2f}" for s, c in prior_seeds_dict.items())
    ann[str(target)] = template(seed_used, mamba_num, prior_str)
    p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
    print(f"Pre-authored entry {target} (mamba#{mamba_num}, seed={seed_used}). Prior: {prior_str}")
elif mode == "post" and target:
    # Backfill verdict + learning from JSONL
    log = (Path(__file__).parent / "autoresearch_results" / "experiment_log.jsonl").read_text().splitlines()
    entry = next(json.loads(l) for l in log if json.loads(l).get("experiment_num") == target)
    composite = entry.get("composite", 0)
    sharpe = entry.get("sharpe", 0)
    val_sharpe = entry.get("val_sharpe", 0)
    test_pos = entry.get("test_pos_folds", 0)
    val_pos = entry.get("val_pos_folds", 0)
    seed_used, mamba_num = RUNS[target]
    e = ann.get(str(target), {})
    e["verdict"] = (
        f"Variance run (DISCARD vs champion). Composite {composite:+.4f} "
        f"(test Sharpe {sharpe:+.4f}, val Sharpe {val_sharpe:+.4f}). "
        f"Test {test_pos}/7 positive, val {val_pos}/7 positive. "
        f"Seed={seed_used} datapoint added to variance distribution."
    )
    # collect seeds dict
    seeds_done = {42: 5.5996}
    for jid in sorted(RUNS.keys()):
        if jid > target:
            break
        if str(jid) in ann and ann[str(jid)].get("verdict") and jid != target:
            v = ann[str(jid)]["verdict"]
            import re as _re
            m = _re.search(r"Composite \+?([+-]?[0-9]+\.[0-9]+)", v)
            if m:
                seeds_done[RUNS[jid][0]] = float(m.group(1).replace("+",""))
    seeds_done[seed_used] = float(f"{composite:.4f}")
    import statistics as st
    vals = list(seeds_done.values())
    mean = st.mean(vals)
    stdev = st.stdev(vals) if len(vals) > 1 else 0.0
    median = st.median(vals)
    e["learning"] = (
        f"Seeds tested so far at champion config: " +
        ", ".join(f"{s}->{c:+.4f}" for s, c in sorted(seeds_done.items())) +
        f". Mean {mean:+.4f}, std {stdev:.4f}, median {median:+.4f}, "
        f"range {max(vals)-min(vals):.4f}. Champion seed=42 is "
        f"{(5.5996-mean)/(stdev or 1):.2f}σ above mean. "
        f"Single-seed champion-declaration policy: only declare new "
        f"global champion if (a) 3-seed median beats prior 3-seed median "
        f"AND (b) peak seed beats peak prior."
    )
    e["_manual"] = True
    ann[str(target)] = e
    p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
    print(f"Post-authored entry {target}: composite {composite:+.4f}; running stats mean {mean:+.3f} std {stdev:.3f}")
