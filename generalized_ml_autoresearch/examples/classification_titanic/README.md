# Example: binary classification (Titanic surrogate)

Demonstrates the framework on a binary-classification task using the xgboost backbone.

## Why sklearn breast-cancer instead of actual Titanic?

To avoid network dependencies and external data files, this example uses
`sklearn.datasets.load_breast_cancer` — a bundled binary-classification benchmark
(569 rows, 30 features). The runner's task-type = `binary_classification`,
primary metric = `f1`, and split protocol = `stratified_kfold` are all identical
to how a real Titanic pipeline would look.

## What this demonstrates

- A task-type different from example 1 (regression → classification)
- A backbone different from example 1 (MLP → xgboost, i.e. Tier-3 GBM)
- Task-specific metrics: precision, recall, F1, F2, MCC, AUC-ROC, AUC-PR, log-loss
- Stratified k-fold (preserves class balance per fold)

Zero changes to `core/` between examples — only `config.yaml` and the seed reasoning
entry differ. This is the generality audit in action.

## Run

```bash
python run_example.py
```

Expected: composite ~ 0.95+, status KEEP.

## Files produced

Same layout as example 1, but `exp1_predictions.csv` has the classification columns
(`prediction`, `actual`, `correct`, `confidence`, `aleatoric`, `epistemic`).
