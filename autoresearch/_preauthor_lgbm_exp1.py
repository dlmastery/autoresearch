"""Pre-author LightGBM Exp1 — SOTA baseline."""
import json, sys
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
# Find next available JSONL index — we'll look up the next experiment_num
# from the log and use it as key (runner writes the correct key on its own).
log = (Path(__file__).parent / "autoresearch_results" / "experiment_log.jsonl").read_text().splitlines()
next_id = int(json.loads(log[-1])["experiment_num"]) + 1

ann[str(next_id)] = {
    "diagnosis": (
        "LIGHTGBM PHASE — FIRST EXPERIMENT (1/15). Per CLAUDE.md 'each GBM "
        "is its own backbone'. XGBoost Exp6 champion at composite +7.7601 "
        "(depth=4 lr=0.01) sets the bar. LightGBM uses a DIFFERENT "
        "splitting algorithm (leaf-wise best-first growth + GOSS sampling) "
        "vs XGBoost's level-wise. Key mechanistic differences: "
        "(a) leaf-wise can produce deeper asymmetric trees for the same "
        "num_leaves, which is more aggressive on complex feature "
        "interactions; (b) GOSS keeps all large-gradient samples + random "
        "subset of small-gradient samples, replacing uniform subsample; "
        "(c) histogram binning pre-discretizes features — potentially "
        "different exploitation of our 104 continuous features vs "
        "XGBoost's tree_method=hist which is similar in spirit. Hypothesis: "
        "LightGBM within +/-0.3 composite of XGBoost; if significantly "
        "better, the leaf-wise algorithm is finding structure XGBoost "
        "misses; if worse, the GOSS sampling is hurting on our small n."
    ),
    "citations": (
        "Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017 NeurIPS 'LightGBM: "
        "A Highly Efficient Gradient Boosting Decision Tree'. Introduces "
        "GOSS (Gradient-based One-Side Sampling) + EFB (Exclusive Feature "
        "Bundling). Distinctive from XGBoost: leaf-wise growth with "
        "num_leaves=63 default (depth-unlimited by default) vs XGBoost's "
        "level-wise with max_depth=6 default.\n"
        "Grinsztajn, Oyallon, Varoquaux 2022 NeurIPS 'Why do tree-based "
        "models still outperform deep learning on tabular data?' "
        "(arXiv:2207.08815) — discusses leaf-wise vs level-wise tradeoffs.\n"
        "Prokhorenkova et al. 2018 NeurIPS 'CatBoost' (arXiv:1706.09516) — "
        "comparative benchmark across the three GBMs."
    ),
    "hypothesis": (
        "Run LightGBM with CLAUDE.md SOTA recipe: n_estimators=2000, "
        "num_leaves=63, learning_rate=0.03, feature_fraction=0.8, "
        "bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=20, "
        "reg_alpha=0.1, reg_lambda=1.0, seq_len=10, seed=42. Same 1040-"
        "feature flattened windowed tabular input as XGBoost. No GPU "
        "(removed in our wrapper rewrite); CPU-only."
    ),
    "prediction": (
        "Composite +6.5 to +7.8. Probability of beating XGBoost +7.76: "
        "25% (XGBoost is hard to beat). Probability of positive composite: "
        "95%. Most informative: if LightGBM wins on the hardest val folds "
        "(1, 2) but XGBoost wins overall, they're complementary and "
        "should be ensembled in phase (b)."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored LGBM Exp1 (JSONL {next_id}).")
