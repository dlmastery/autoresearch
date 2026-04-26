# Champion Archive — Exp 33: dinov2_vits14_spectral_cos

**ARI:** 0.6963
**NMI:** 0.8974
**Silhouette:** 0.0890

## Reproduce

```bash
python predict.py
```

## Method

PCA(50) dimensionality reduction → Agglomerative Clustering with Ward linkage at K=40 (Ward 1963).

Deterministic — no random init.
