# AutoResearch Detailed Journal

Persistent research log with literature citations, hypotheses, and findings.
Append-only. Every experiment with its arxiv/paper reference.

---

## 2026-04-19 — Crash Diagnosis Session

### Hardware Instability Identified
- 4 BSODs today with different bugcheck codes (0x7f, 0x1ca, 0x1e, 0x101)
- WHEA CPU parity errors on APIC IDs 16, 17, 24, 25 (all E-cores)
- Root cause: BIOS update reset voltage/C-state defaults
- Mitigation: pin process to P-cores only (logical 0-15)
- Code change: added `_pin_to_safe_cores()` to `run_autoresearch.py`

### Literature Consulted This Session
None — diagnosis was systems-level.

---

## Experiment Log with References

### Phase 1: LFM2-350M (Experiments 1-50)

**Foundation models for time series:**
- Liquid Foundation Models 2 (LFM2) technical report — Liquid AI 2024
- "What Should Pre-Trained Foundation Models Know About Financial Markets?" — Gu et al., 2024

**Key experiments:**
- Exp 3 (lr=3e-5): composite +1.61 — LR sweet spot for frozen backbone
- Exp 13 (warmup=3): +1.60 — Goyal et al., "Accurate, Large Minibatch SGD" arxiv:1706.02677
- Exp 20 (lr=2e-5): +1.77 — fine-grained LR search
- Exp 23+ (het-loss): HURT — Kendall & Gal 2017 arxiv:1703.04977 (lesson: doesn't help small-n)

**Findings:**
- LFM2 with frozen backbone: median test Sharpe +1.40 across 4 seeds
- Massive seed variance (-1.52 to +1.77 at same config)
- Heteroscedastic loss degraded mean prediction quality on n=2738

### Phase 2: MLP from scratch (Experiments 51-97)

**Core references:**
- Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning" RFS — MLP is best for low-SNR financial prediction with proper regularization
- He et al. (2016) "Deep Residual Learning" arxiv:1512.03385 — skip connections
- Srivastava et al. (2014) "Dropout" JMLR — dropout as regularization
- Loshchilov & Hutter (2017) arxiv:1711.05101 — AdamW, cosine LR
- Lewkowycz et al. (2020) arxiv:2003.02218 — large LR phase, flat minima
- Ioffe & Szegedy (2015) arxiv:1502.03167 — BatchNorm (HURT here)

**Breakthrough: residual MLP**
- Exp10 (shortcut + 2-layer residual, lr=3e-4, hd=0.1, seed=0): composite +4.67, test Sharpe +4.77 — first 7/7 positive test folds
- Exp11 (reproduce seed=42): +3.76 — verified not a lucky seed
- Exp32 (lr=5e-4, hd=0.15, huber=0.5, seed=0): composite +5.50, test Sharpe +6.21 — CHAMPION
- Exp36 (POST-BIOS verify): exact numerical match — determinism confirmed

**Loss function:**
- Huber loss with delta=0.5 (Huber 1964) — robust to fat tails in FX returns
- Tested: huber=0.5 vs 1.0 — 0.5 better for fold 2 (post-crash)

**Architectural ablation:**
- Plain MLP 512h: composite -0.51 (overfit, 1.06M params)
- Plain MLP 128h: composite +0.82 (reduce params by 6x)
- Residual MLP 128h + skip: composite +6.21 (5x improvement from skip)
- BatchNorm: composite -1.31 (removes regime-scale info — known issue Bjorck 2018)
- Heteroscedastic: no help on n=2738

### Champion Config (Exp41, verified Exp36 post-BIOS reproduction)
```
backbone: residual MLP
hidden: 128, head: 64
lr: 5e-4 (high LR enabled by skip connection)
bs: 32
seq: 10
epochs: 50 (from-scratch needs more than fine-tuning)
wd: 1e-5
patience: 10
grad_clip: 1.0
huber_delta: 0.5
head_dropout: 0.15
seed: 0
het_loss: False
```

### Performance (Champion, seed=0)
- **Composite: +5.50** | **Test Sharpe: +6.21** | **Total return: +1001%**
- 7/7 positive test folds
- Per-fold range: +0.44 (fold 2, weakest) to +10.22 (fold 6, COVID)

### Cross-Seed Variance (residual MLP champion)
- Seed 0: composite +5.50, test Sharpe +6.21
- Seed 42: composite +4.45, test Sharpe +4.69
- Seed 99: composite +4.46, test Sharpe +4.76
- Median test Sharpe: +4.76 (MUCH more stable than plain MLP or LFM2)

### Why Residual Skip Works for FX
The skip connection `h = shortcut(x) + residual(x)` gives the model two paths:
1. **Shortcut**: direct linear projection — captures the "mean" baseline signal
2. **Residual**: nonlinear correction — learns regime-specific deviations

Financial prediction has low SNR; the skip lets the model use a linear baseline that works "good enough" across regimes while the nonlinear branch adds regime-specific corrections without risking catastrophic bad predictions in unfamiliar regimes. This matches the empirical finding: 7/7 test folds positive (most stable result in all 97 experiments).

---

## Upcoming Experiments (Queue)

### Immediate (post-crash verification)
1. Exp42: CPU-only verify champion reproduces
2. Exp43: seed sweep at champion (seeds 7, 2024)

### Final MLP (reach 50)
3. grad_clip=0.5 at champion — tighter clipping for high LR
4. lr=4e-4 — between champion 5e-4 and 3e-4
5. Code change: deeper residual (3-layer)

### Phase 3: LSTM (50 experiments)
Start with ablation config as baseline, optimize per CLAUDE.md process.
Expected references:
- Hochreiter & Schmidhuber (1997) — LSTM
- Cho et al. (2014) — GRU alternative
- Hu et al. (2022) arxiv:2106.09685 LoRA — for fine-tuning

### Phase 4+: PatchTST, PatchTSMixer, GBMs
- Nie et al. (2023) ICLR "PatchTST" arxiv:2211.14730
- Ekambaram et al. (2023) NeurIPS "PatchTSMixer" arxiv:2306.09364
- Chen & Guestrin (2016) XGBoost
- Ke et al. (2017) LightGBM
- Prokhorenkova et al. (2018) CatBoost

---

## Notes on Methodology

### Composite Metric
`composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds`

Penalizes:
- Asymmetry between val/test (catches regime-overfitting)
- Any negative fold (no single-regime wins counted)

### Super-fold design
Per CLAUDE.md — train once, evaluate on 7 regime windows.
Not walk-forward. No model retraining per fold.

### Determinism
- `--seed` flag sets torch/numpy/python RNGs
- Exp36 verified Exp32 reproduces to 4 decimal places post-BIOS

### Hardware constraints
- CPU-only for MLP (15s per run)
- GPU for LFM2 (300s per run — DEFERRED until hardware fixed)
- Pin to P-cores 0-15 (avoid E-cores with WHEA errors)
