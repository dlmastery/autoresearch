# Crash-Recovery Checkpoint — Olivetti Faces Clustering

_Last update: 2026-04-25T23:56:17_

## Current champion
- **Exp:** 33 (dinov2_vits14_spectral_cos)
- **ARI:** 0.6963
- **NMI:** 0.8974
- **Silhouette:** 0.0890
- **Description:** DINOv2 dinov2_vits14 + Spectral cosine affinity

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
| 22 | dinov2_vits14_minibatch_kmeans | 0.5596 | 0.8393 | KEEP |
| 23 | dinov2_vits14_bisecting_kmeans | 0.4437 | 0.7678 | KEEP |
| 24 | dinov2_vits14_kmeans_random | 0.5000 | 0.8091 | KEEP |
| 25 | dinov2_vits14_kmeans_n50 | 0.5852 | 0.8456 | KEEP |
| 26 | dinov2_vits14_spherical | 0.5602 | 0.8259 | KEEP |
| 27 | dinov2_vits14_agg_ward | 0.6371 | 0.8706 | KEEP |
| 28 | dinov2_vits14_agg_avg | 0.4703 | 0.8158 | KEEP |
| 29 | dinov2_vits14_agg_complete | 0.4805 | 0.8071 | KEEP |
| 30 | dinov2_vits14_agg_cosine_avg | 0.4490 | 0.8174 | KEEP |
| 31 | dinov2_vits14_spectral_g001 | 0.5852 | 0.8533 | KEEP |
| 32 | dinov2_vits14_spectral_g01 | 0.2767 | 0.7672 | DISCARD |
| 33 | dinov2_vits14_spectral_cos | 0.6963 | 0.8974 | KEEP |
| 34 | dinov2_vits14_spectral_knn10 | 0.6389 | 0.8584 | KEEP |
| 35 | dinov2_vits14_birch | 0.6371 | 0.8706 | KEEP |
| 36 | dinov2_vits14_gmm_full | 0.5234 | 0.8133 | KEEP |
| 37 | dinov2_vits14_gmm_diag | 0.5234 | 0.8133 | KEEP |
| 38 | dinov2_vits14_pca50_km | 0.5312 | 0.8184 | KEEP |
| 39 | dinov2_vits14_pca100_km | 0.5473 | 0.8278 | KEEP |
| 40 | dinov2_vits14_umap10_km | 0.5982 | 0.8465 | KEEP |
| 41 | dinov2_vits14_umap2_km | 0.6100 | 0.8455 | KEEP |
| 42 | dinov2_vitb14_vitb_km | 0.5445 | 0.8243 | KEEP |
| 43 | dinov2_vitb14_vitb_spherical | 0.5388 | 0.8119 | KEEP |
| 44 | dinov2_vits14_seed1 | 0.5561 | 0.8301 | KEEP |
| 45 | dinov2_vits14_seed2 | 0.5144 | 0.8110 | KEEP |
| 46 | dinov2_vits14_seed7 | 0.5387 | 0.8175 | KEEP |

## Next experiment

Tier 6 SOTA exploration after Exp 14: try DINOv2-vit features + KMeans, ProPos (Huang 2023 TPAMI), or DivClust (Karaman 2023). Also explore hyperparameter tuning of Spectral clustering's gamma — it tanked at 0.058 with default and is likely recoverable to ~0.65 with proper tuning.
