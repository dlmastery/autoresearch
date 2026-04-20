# LSTM Exp24 Champion (seed=42)

Composite +6.3571 | Test Sharpe +6.4571 | Return +1095% | 7/7 test pos.
Config: identical to Exp21 (BiLSTM h=128, lr=1e-3, bs=32, seq=10, ep=100, pat=15, wd=1e-3, huber=1.0, hd=0.25) but seed=42.

Seed variance study: seed=0 composite +6.1938, seed=42 composite +6.3571. Difference +0.16 confirms meaningful seed variance.
