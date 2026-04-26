# Champion Archive — Exp 8: agg_ward

**ARI:** 0.5159
**NMI:** 0.8201
**Silhouette:** 0.1608

## Reproduce

```bash
python predict.py
```

## Method

PCA(50) dimensionality reduction → Agglomerative Clustering with Ward linkage at K=40 (Ward 1963).

Deterministic — no random init.
