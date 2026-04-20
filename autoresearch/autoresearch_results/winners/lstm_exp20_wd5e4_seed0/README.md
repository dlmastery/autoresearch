# LSTM Exp20 — New Global Champion (2026-04-19)

## Summary
- **Composite +6.1312** | **Test Sharpe +6.3363** | Val Sharpe +6.2312 | 7/7 positive test folds
- Change from Exp9 (+6.1035): wd=1e-4 -> 5e-4 (5x stronger L2)
- Return: +1048% on held-out test (vs +1033% for Exp9)

## Config
- Bidirectional 2-layer LSTM h=128, lr=1e-3, bs=32, seq=10, ep=100, pat=15
- huber=1.0, hd=0.25, **wd=5e-4**, seed=0
- Early-stopped epoch 30

## Per-Fold Test
| Fold | Sharpe | Return | IC | WR |
|------|--------|--------|-----|-----|
| 1 Pre-crisis/GFC | +2.07 | +16.37% | +0.157 | 55.3% |
| 2 Post-crash | +1.66 | +7.97% | +0.110 | 57.0% |
| 3 Eurozone | +11.26 | +38.34% | +0.685 | 81.1% |
| 4 Strong USD | +8.41 | +77.49% | +0.741 | 73.2% |
| 5 Low-vol | +10.53 | +33.86% | +0.738 | 75.3% |
| 6 EUR crisis | +12.23 | +83.52% | +0.776 | 77.0% |
| 7 Recent | +7.82 | +51.53% | +0.656 | 71.0% |

## Reproduction
```bash
python -m autoresearch.run_autoresearch --backbone lstm --lr 1e-3 --batch-size 32 \
  --seq-len 10 --epochs 100 --weight-decay 5e-4 --patience 15 --grad-clip 1.0 \
  --huber-delta 1.0 --head-dropout 0.25 --seed 0 \
  --description "reproduce lstm champ exp20"
```
31s training on CPU (60% cap), deterministic.

## Cite
Fischer & Krauss 2018 (LSTM for finance); Srivastava 2014 (dropout); Zaremba 2014 (LSTM L2).
