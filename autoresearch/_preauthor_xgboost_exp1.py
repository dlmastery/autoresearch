"""Pre-author XGBoost Exp1 (JSONL 174): SOTA recipe baseline."""
import json
from pathlib import Path
p = Path(__file__).parent / "autoresearch_results" / "reasoning_annotations.json"
ann = json.loads(p.read_text(encoding="utf-8"))
# Previous 174 (PatchTST pre-author) was killed before launch; overwrite.
ann["174"] = {
    "diagnosis": (
        "XGBOOST PHASE — FIRST EXPERIMENT (1/50). Per user plan "
        "2026-04-20 ('c seems best' + 'try patchtst last'), GBMs run "
        "before the remaining neural backbones. Baseline recipe follows "
        "Chen & Guestrin 2016 and its 2024-era best practices: "
        "n_estimators=1500 with early-stopping, max_depth=6, lr=0.03, "
        "subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0. The GBM "
        "wrapper flattens the (seq_len=10, n_features=104) window into "
        "a 1040-dim tabular feature vector per sample and fits ONE "
        "estimator per target column (ret_1d, ret_5d). At n=2738 this "
        "is well within XGBoost's sweet spot for tabular regression "
        "(row/col ratio ~2.6, boosting rounds << n)."
    ),
    "citations": (
        "Chen & Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting "
        "System' (arXiv:1603.02754) — canonical reference. Introduces "
        "2nd-order Newton boosting (Hessian), sparsity-aware split "
        "finding, out-of-core computation.\n"
        "Friedman 2001 Annals of Statistics 'Greedy Function "
        "Approximation: A Gradient Boosting Machine' — foundational "
        "GBM theory.\n"
        "Gu, Kelly, Xiu 2020 RFS 'Empirical Asset Pricing via Machine "
        "Learning' — shows tree ensembles competitive vs deep nets on "
        "financial cross-section; recommends n_estimators ~ 1000-2000.\n"
        "Prokhorenkova et al. 2018 NeurIPS 'CatBoost' (arXiv:1706.09516) "
        "— comparative discussion of XGBoost vs CatBoost bias variance.\n"
        "Grinsztajn, Oyallon, Varoquaux 2022 NeurIPS 'Why do tree-based "
        "models still outperform deep learning on tabular data?' "
        "(arXiv:2207.08815) — explains why GBMs often beat DL on "
        "low-SNR tabular with n<10k, directly relevant to our n=2738."
    ),
    "hypothesis": (
        "Run XGBoost with SOTA recipe: n_estimators=1500, max_depth=6, "
        "lr=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, "
        "reg_alpha=0, min_child_weight=1, gamma=0, tree_method=hist, "
        "seq_len=10. Mechanism: 2nd-order gradient boosting on flattened "
        "windowed features captures nonlinear feature×lag interactions "
        "that a linear model misses. Risk: boosting rounds may overfit "
        "at n=2738 if we don't early-stop — we rely on XGBoost's "
        "internal train/val split via the runner's val set. Expected "
        "training time: <60s (tabular, no GPU needed)."
    ),
    "prediction": (
        "Composite +2.0 to +5.0. Probability of composite > 0: 90% "
        "(tree boosting has strong inductive bias for tabular). "
        "Probability of beating MLP residual (+5.50): 25%. Probability "
        "of new global champion (>+6.42): 5%. Per-fold: expect folds "
        "3-6 positive; folds 1/2 uncertain as before."
    ),
    "_manual": True,
}
p.write_text(json.dumps(ann, indent=2), encoding="utf-8")
print(f"Pre-authored XGBoost Exp1 (174). Total: {len(ann)}")
