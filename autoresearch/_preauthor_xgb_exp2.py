"""Pre-author XGBoost Exp2 (JSONL 176): seed=0 variance check on champion config."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
ann["176"] = {
    "diagnosis": (
        "XGBoost Exp2 — first variance check on the Exp1 champion config "
        "(+7.1686 composite with SOTA recipe). Exp1 used seed=42; the "
        "initial --seed 0 attempt gave a BIT-IDENTICAL result because "
        "the runner's --seed flag wasn't plumbed through to the GBM "
        "wrapper's random_state. That plumbing is now fixed in "
        "run_autoresearch.py (if config['seed'] is not None, it is "
        "added as 'random_state' to gbm_hp overrides before create_model). "
        "With the plumbing fixed, seed=0 will genuinely alter "
        "XGBoost's subsample and colsample RNG streams and should "
        "produce a different (though probably similar) result."
    ),
    "citations": (
        "Chen & Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting "
        "System' (arXiv:1603.02754) -- random_state parameter controls "
        "the RNG used for subsample (row) and colsample_bytree (column) "
        "sampling.\n"
        "Bouthillier, Laurent, Vincent 2019 ICML workshop "
        "(arXiv:1906.05268) -- documents seed-driven variance in ML. "
        "XGBoost is relatively seed-stable vs deep nets because most "
        "of the model is deterministic given data + hparams, but "
        "subsample/colsample introduce real seed sensitivity.\n"
        "Picard 2021 (arXiv:2109.08203) -- seeds can flip rankings."
    ),
    "hypothesis": (
        "Same champion config (n_estimators=1500, max_depth=6, lr=0.03, "
        "subsample=0.8, colsample=0.8, reg_lambda=1.0, tree_method=hist, "
        "seq_len=10) with random_state=0. Mechanism: changes the RNG "
        "draws for the 20% row-hold-out and 20% column-hold-out at each "
        "tree split. Given the model converges early (shown by "
        "HP-insensitivity test at n_estimators=500 and max_depth=2), "
        "seed variance should be < 0.3 composite."
    ),
    "prediction": (
        "Composite +6.5 to +7.5. Probability of composite > seed=42 champion "
        "+7.17: 30-40% (XGBoost is close to deterministic). Probability "
        "of composite < +5: near-zero (unlike LSTM's seed variance of 1.0+). "
        "Most informative: if delta from seed=42 is < 0.2, XGBoost is a "
        "deployment-friendly stable champion. If delta > 0.5, the Exp1 "
        "win was partly seed luck."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored XGBoost Exp2 (176). Total: {len(ann)}")
