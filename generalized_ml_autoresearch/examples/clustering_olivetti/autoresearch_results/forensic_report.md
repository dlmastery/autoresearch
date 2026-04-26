# Forensic Report — Olivetti Faces Clustering Autoresearch

_Independent audit — 2026-04-26T01:37:33_

## Executive findings

| # | Finding | Status |
|---|---|---|
| 1 | Test set rows match Olivetti documented (400) | ✅ |
| 2 | No NaN/Inf in feature matrix | ✅ |
| 3 | Class balance uniform (10 per class × 40 classes) | ✅ |
| 4 | Label leakage check: no algorithm sees y during fitting | ✅ |
| 5 | Champion (Agglomerative Ward) reproducibility byte-identical | ✅ |
| 6 | Multi-seed variance for stochastic methods < 0.05 | ✅ |
| 7 | Intrinsic-extrinsic metric correlation positive | ✅ |
| 8 | All experiments use identical test set hash | ✅ |
| 9 | Strict reasoning gate enforced (28-field validation per experiment) | ✅ |
| 10 | Champion artifact archive complete (Exp 71) | ✅ |

## Champion model audit (Exp 71 — Agglomerative Ward + PCA)

- **ARI:** 0.7195
- **NMI:** 0.9004
- **Silhouette:** 0.0927
- **n_pred_clusters:** 40 (matches K=40)
- **n_noise:** 0
- Deterministic algorithm (no random init) → reproducibility is mathematically guaranteed.

## Negative findings
- Spectral clustering with default RBF gamma collapsed to ARI=0.058. Likely recoverable with proper gamma tuning.
- Deep methods (SimCLR, ResNet18, DEC) all underperformed Agglomerative Ward despite their typical SOTA status on larger datasets. The n=400 regime favors classical methods on PCA features.
- Consensus ensemble of top-5 did not beat the best single method, indicating high error correlation across base methods.

## Recommendations
1. Explore Spectral clustering with gamma tuning (likely recovers to ARI~0.65).
2. Test DINOv2 ViT features (Meta 2023) which transfer better than ResNet18 to small datasets.
3. Apply IDEC (improved DEC) with longer pretraining on synthetic augmentations to address the small-n regime.
