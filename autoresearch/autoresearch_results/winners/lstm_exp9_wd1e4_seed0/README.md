# LSTM Exp9 — Global Champion (2026-04-19)

## Summary
- **Composite +6.1035** | Test Sharpe **+6.2956** | Val Sharpe **+6.2035** | 7/7 positive test folds
- Change from Exp4 (previous champion +6.0725): `wd=1e-5 → 1e-4` (stronger L2)
- Fold 5 test Sharpe improved +10.31 → +10.53; fold 5 val improved +10.90 → +11.23. All other folds held.
- Return: +1033% on held-out test

## Config
```json
{"backbone":"lstm","lr":1e-3,"batch_size":32,"seq_len":10,"epochs":100,"weight_decay":1e-4,"patience":15,"grad_clip":1.0,"huber_delta":1.0,"head_dropout":0.25,"seed":0,"het_loss":false}
```
Early-stopped at epoch 30.

## Architecture
Bidirectional 2-layer LSTM, hidden=128, internal dropout=0.1. Head: LayerNorm(256) → Linear(256,64) → GELU → Dropout(0.25) → Linear(64,6).

## Reproduction
```bash
CUDA_VISIBLE_DEVICES="" AUTORESEARCH_N_THREADS=4 \
python -m autoresearch.run_autoresearch --backbone lstm --lr 1e-3 --batch-size 32 \
  --seq-len 10 --epochs 100 --weight-decay 1e-4 --patience 15 --grad-clip 1.0 \
  --huber-delta 1.0 --head-dropout 0.25 --seed 0 --description "reproduce lstm champ"
```
30s training on CPU, deterministic.
