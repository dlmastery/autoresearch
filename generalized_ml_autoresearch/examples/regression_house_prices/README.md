# Example: California Housing regression

Demonstrates the generalized ML autoresearch framework on a pure regression task.

## What this example shows

- How to point the runner at a `sklearn.datasets` loader instead of a CSV
- How to author a pre-run reasoning annotation that passes the validators
- How the runner writes JSONL + per-prediction CSV + best_config / best_model
- How the dashboard renders the experiment + reasoning panel

## Run

```bash
python run_example.py
```

On first run you should see output like:

```
[runner] Exp1 (mlp) done in ~30s — composite=-0.59 status=KEEP
```

The composite here is negated RMSE (lower-is-better internally flipped so `>` means "better").

## Generated artifacts

```
autoresearch_results/
  experiment_log.jsonl           # 1 row
  best_config.json               # KEEP promoted Exp1
  best_model.pt                  # saved MLP weights
  reasoning_annotations.json     # pre-run entry + runner-fallback verdict/learning
  trade_logs/
    exp1_predictions.csv         # per-prediction rows for fold 0+1+2 test sets
    exp1_prediction_summary.json # per-fold totals
  dashboard.html                 # open via python -m http.server 8765
```

## Open the dashboard

```bash
cd autoresearch_results
python -m http.server 8765
# browse to http://localhost:8765/dashboard.html
```

Click the Exp1 row to see:
- Composite / test / val metrics
- Per-fold RMSE stats
- Secondary metrics (MAE, MAPE, R², IC)
- Reasoning annotation (diagnosis, citations, hypothesis, prediction, verdict, learning)

## Next steps

Follow the 7-step process in CLAUDE.md. Author Exp 2's pre-run reasoning by:

1. Reading Exp 1's per-fold RMSE — which fold was weakest?
2. Finding a paper that addresses the weakness (e.g. deeper MLP → He et al. 2016 residual; regularization → Ioffe & Szegedy 2015 batch norm).
3. Writing the diagnosis / citations / hypothesis / prediction fields in `reasoning_annotations.json` under key `"2"`.
4. Running `python run_example.py` with an updated `config.yaml`.
