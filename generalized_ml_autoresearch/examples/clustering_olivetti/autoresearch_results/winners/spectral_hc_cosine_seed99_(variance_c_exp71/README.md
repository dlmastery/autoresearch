# Champion Archive — Exp 71: spectral_hc_cosine_seed99_(variance_c

**ARI:** 0.7195
**NMI:** 0.9004
**Silhouette:** 0.0927

## Reproduce

```bash
python predict.py
```

## Method

PCA(50) dimensionality reduction → Agglomerative Clustering with Ward linkage at K=40 (Ward 1963).

Deterministic — no random init.
