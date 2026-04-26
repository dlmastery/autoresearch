# Champion Archive — Exp 20: dinov2_kmeans

**ARI:** 0.5455
**NMI:** 0.8201
**Silhouette:** 0.0710

## Reproduce

```bash
python predict.py
```

## Method

PCA(50) dimensionality reduction → Agglomerative Clustering with Ward linkage at K=40 (Ward 1963).

Deterministic — no random init.
