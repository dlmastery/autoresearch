# Crash-Recovery Checkpoint — Olivetti Faces Clustering

_Last update: 2026-04-25T23:48:39_

## Current champion
- **Exp:** 20 (dinov2_kmeans)
- **ARI:** 0.5455
- **NMI:** 0.8201
- **Silhouette:** 0.0710
- **Description:** DINOv2 ViT-S/14 (Oquab 2024 Meta TMLR) features + KMeans

## Experiment history

| Exp | Backbone | ARI | NMI | Status |
|---|---|---|---|---|
| 1 | kmeans_raw_pixels | 0.4057 | 0.7585 | KEEP |
| 2 | kmeans_pca50 | 0.4780 | 0.7951 | KEEP |
| 3 | kmeans_pca100 | 0.4633 | 0.7856 | KEEP |
| 3 | kmeans_pca100 | 0.4633 | 0.7856 | KEEP |
| 4 | kmeans_pca150 | 0.4484 | 0.7846 | KEEP |
| 5 | kmeans_pca_whitened | 0.3602 | 0.7508 | KEEP |
| 6 | spectral_rbf | 0.0578 | 0.4560 | DISCARD |
| 7 | gmm_pca_full | 0.4545 | 0.7736 | KEEP |
| 8 | agg_ward | 0.5159 | 0.8201 | KEEP |
| 9 | hdbscan | 0.3438 | 0.8142 | KEEP |
| 10 | conv_ae_kmeans | 0.4790 | 0.7934 | KEEP |
| 11 | resnet18_kmeans | 0.4444 | 0.7916 | KEEP |
| 12 | dec | 0.4942 | 0.8036 | KEEP |
| 13 | simclr_kmeans | 0.3678 | 0.7502 | KEEP |
| 14 | consensus_top5 | 0.4767 | 0.8082 | KEEP |
| 15 | umap_kmeans | 0.5001 | 0.8003 | KEEP |
| 15 | umap_kmeans | 0.5001 | 0.8003 | KEEP |
| 16 | spectral_tuned | 0.5252 | 0.8228 | KEEP |
| 16 | spectral_tuned | 0.5252 | 0.8228 | KEEP |
| 17 | birch | 0.5287 | 0.8254 | KEEP |
| 18 | affinity_prop | 0.4757 | 0.8105 | KEEP |
| 19 | meanshift | 0.0000 | 0.0000 | DISCARD |
| 20 | dinov2_kmeans | 0.5455 | 0.8201 | KEEP |
| 21 | spherical_kmeans | 0.4816 | 0.7896 | KEEP |

## Next experiment

Tier 6 SOTA exploration after Exp 14: try DINOv2-vit features + KMeans, ProPos (Huang 2023 TPAMI), or DivClust (Karaman 2023). Also explore hyperparameter tuning of Spectral clustering's gamma — it tanked at 0.058 with default and is likely recoverable to ~0.65 with proper tuning.
