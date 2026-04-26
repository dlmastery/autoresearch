# Crash-Recovery Checkpoint — Olivetti Faces Clustering

_Last update: 2026-04-26T01:37:33_

## Current champion
- **Exp:** 71 (spectral_hc_cosine_seed99_(variance_c)
- **ARI:** 0.7195
- **NMI:** 0.9004
- **Silhouette:** 0.0927
- **Description:** Spectral hill-climb: cosine seed=99 (variance check) on DINOv2 ViT-S/14

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
| 47 | spectral_hc_cosine_+_assignkmeans_(ch | 0.6963 | 0.8974 | KEEP |
| 48 | spectral_hc_cosine_+_assigncluster_qr | 0.4708 | 0.7628 | KEEP |
| 49 | spectral_hc_cosine_+_L2-normalized_fe | 0.6963 | 0.8974 | KEEP |
| 50 | spectral_hc_nearest_neighbors_k5 | 0.6042 | 0.8577 | KEEP |
| 51 | spectral_hc_nearest_neighbors_k7 | 0.6246 | 0.8538 | KEEP |
| 52 | spectral_hc_nearest_neighbors_k15 | 0.5888 | 0.8358 | KEEP |
| 53 | spectral_hc_nearest_neighbors_k20 | 0.5278 | 0.8059 | KEEP |
| 54 | spectral_hc_nearest_neighbors_k30 | 0.4553 | 0.7806 | KEEP |
| 55 | spectral_hc_RBF_gamma0.0001 | 0.7170 | 0.9102 | KEEP |
| 56 | spectral_hc_RBF_gamma0.0005 | 0.6961 | 0.9001 | KEEP |
| 57 | spectral_hc_RBF_gamma0.005 | 0.2628 | 0.7973 | DISCARD |
| 58 | spectral_hc_RBF_gamma0.05 | 0.0503 | 0.5965 | DISCARD |
| 59 | spectral_hc_RBF_gamma0.5 | 0.0000 | 0.0297 | DISCARD |
| 60 | spectral_hc_ViT-B/14_+_cosine | 0.6552 | 0.8805 | KEEP |
| 61 | spectral_hc_ViT-B/14_+_cluster_qr_+_c | 0.4317 | 0.7495 | KEEP |
| 62 | spectral_hc_ViT-B/14_+_L2-norm_+_cosi | 0.6552 | 0.8805 | KEEP |
| 63 | spectral_hc_ViT-B/14_+_kNN_k10 | 0.5489 | 0.8215 | KEEP |
| 64 | spectral_hc_cosine_+_n_init1 | 0.7064 | 0.9014 | KEEP |
| 65 | spectral_hc_cosine_+_n_init5 | 0.6742 | 0.8829 | KEEP |
| 66 | spectral_hc_cosine_+_n_init25 | 0.6963 | 0.8974 | KEEP |
| 67 | spectral_hc_cosine_+_n_init50 | 0.6666 | 0.8900 | KEEP |
| 68 | spectral_hc_cosine_seed1_(variance_ch | 0.7154 | 0.9051 | KEEP |
| 69 | spectral_hc_cosine_seed7_(variance_ch | 0.6596 | 0.8710 | KEEP |
| 70 | spectral_hc_cosine_seed42_(variance_c | 0.6127 | 0.8609 | KEEP |
| 71 | spectral_hc_cosine_seed99_(variance_c | 0.7195 | 0.9004 | KEEP |
| 72 | ward_hc_linkageward_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 73 | ward_hc_linkageward_on_DINOv2_L2-norm | 0.6366 | 0.8721 | KEEP |
| 74 | ward_hc_linkageaverage_on_DINOv2 | 0.4703 | 0.8158 | KEEP |
| 75 | ward_hc_linkageaverage_on_DINOv2_L2-no | 0.4631 | 0.8234 | KEEP |
| 76 | ward_hc_linkagecomplete_on_DINOv2 | 0.4805 | 0.8071 | KEEP |
| 77 | ward_hc_linkagecomplete_on_DINOv2_L2-n | 0.4926 | 0.8112 | KEEP |
| 78 | ward_hc_linkagesingle_on_DINOv2 | 0.1481 | 0.6689 | DISCARD |
| 79 | ward_hc_linkagesingle_on_DINOv2_L2-nor | 0.1437 | 0.6625 | DISCARD |
| 80 | ward_hc_linkageaverage_+_cosine_distan | 0.4490 | 0.8174 | KEEP |
| 81 | ward_hc_linkagecomplete_+_cosine_dista | 0.4926 | 0.8112 | KEEP |
| 82 | ward_hc_linkagesingle_+_cosine_distanc | 0.1437 | 0.6625 | DISCARD |
| 83 | ward_hc_linkageaverage_+_manhattan_dis | 0.4540 | 0.8206 | KEEP |
| 84 | ward_hc_linkageward_on_PCA(20) | 0.4508 | 0.7910 | KEEP |
| 85 | ward_hc_linkageward_on_PCA(50) | 0.5159 | 0.8201 | KEEP |
| 86 | ward_hc_linkageward_on_PCA(100) | 0.4737 | 0.8081 | KEEP |
| 87 | ward_hc_linkageaverage_+_cosine_on_PCA | 0.3223 | 0.7542 | KEEP |
| 88 | ward_hc_linkageaverage_+_cosine_on_PCA | 0.3229 | 0.7547 | KEEP |
| 89 | ward_hc_linkageaverage_+_cosine_on_PCA | 0.2983 | 0.7444 | DISCARD |
| 90 | ward_hc_Ward_+_connectivity_kNN(k5)_on | 0.6207 | 0.8637 | KEEP |
| 91 | ward_hc_Ward_+_connectivity_kNN(k10)_o | 0.6371 | 0.8706 | KEEP |
| 92 | ward_hc_Ward_+_connectivity_kNN(k20)_o | 0.6371 | 0.8706 | KEEP |
| 93 | ward_hc_Ward_init_+_KMeans_refine_on_D | 0.6308 | 0.8665 | KEEP |
| 94 | ward_hc_Ward_init_+_KMeans_refine_on_D | 0.6303 | 0.8681 | KEEP |
| 95 | ward_hc_Ward_init_+_KMeans_refine_on_P | 0.5013 | 0.8124 | KEEP |
| 96 | ward_hc_Ward_init_+_KMeans_refine_on_P | 0.4591 | 0.7993 | KEEP |
| 97 | birch_hc_threshold0.05_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 98 | birch_hc_threshold0.1_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 99 | birch_hc_threshold0.2_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 100 | birch_hc_threshold0.3_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 101 | birch_hc_threshold0.5_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 102 | birch_hc_threshold0.7_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 103 | birch_hc_threshold1.0_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 104 | birch_hc_threshold1.5_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 105 | birch_hc_branching_factor10_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 106 | birch_hc_branching_factor25_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 107 | birch_hc_branching_factor100_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 108 | birch_hc_branching_factor200_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 109 | birch_hc_default_Birch_on_DINOv2_L2-nor | 0.2306 | 0.6719 | DISCARD |
| 110 | birch_hc_default_Birch_on_PCA(50) | 0.5287 | 0.8254 | KEEP |
| 111 | birch_hc_default_Birch_on_PCA(100) | 0.4737 | 0.8081 | KEEP |
| 112 | birch_hc_default_Birch_on_PCA(20) | 0.4540 | 0.7949 | KEEP |
| 113 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.5461 | 0.8242 | KEEP |
| 114 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.2306 | 0.6719 | DISCARD |
| 115 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.4356 | 0.7685 | KEEP |
| 116 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.4232 | 0.7685 | KEEP |
| 117 | birch_hc_tight_threshold0.01_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 118 | birch_hc_tight_threshold0.02_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 119 | birch_hc_tight_threshold0.03_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 120 | birch_hc_tight_threshold0.04_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 121 | birch_hc_tight_threshold0.05_on_DINOv2 | 0.6371 | 0.8706 | KEEP |
| 122 | umap_hc_n_neighbors=5_on_DINOv2 | 0.6109 | 0.8624 | KEEP |
| 123 | umap_hc_n_neighbors=10_on_DINOv2 | 0.6488 | 0.8693 | KEEP |
| 124 | umap_hc_n_neighbors=30_on_DINOv2 | 0.5680 | 0.8311 | KEEP |
| 125 | umap_hc_n_neighbors=50_on_DINOv2 | 0.5690 | 0.8279 | KEEP |
| 126 | umap_hc_min_dist=0.0_on_DINOv2 | 0.6247 | 0.8606 | KEEP |
| 127 | umap_hc_min_dist=0.3_on_DINOv2 | 0.5949 | 0.8438 | KEEP |
| 128 | umap_hc_min_dist=0.5_on_DINOv2 | 0.6156 | 0.8514 | KEEP |
| 129 | umap_hc_min_dist=0.99_on_DINOv2 | 0.5665 | 0.8258 | KEEP |
| 130 | umap_hc_n_components=3_on_DINOv2 | 0.6177 | 0.8508 | KEEP |
| 131 | umap_hc_n_components=5_on_DINOv2 | 0.5860 | 0.8412 | KEEP |
| 132 | umap_hc_n_components=30_on_DINOv2 | 0.5980 | 0.8495 | KEEP |
| 133 | umap_hc_n_components=50_on_DINOv2 | 0.6107 | 0.8453 | KEEP |
| 134 | umap_hc_metric=cosine | 0.6000 | 0.8421 | KEEP |
| 135 | umap_hc_metric=manhattan | 0.6371 | 0.8579 | KEEP |
| 136 | umap_hc_UMAP(10)_+_Spectral_cosine_dow | 0.1918 | 0.6401 | DISCARD |
| 137 | dec_hc_latent_dim32 | 0.4955 | 0.7982 | KEEP |
| 138 | dec_hc_latent_dim128 | 0.4781 | 0.7994 | KEEP |
| 139 | dec_hc_latent_dim256 | 0.5091 | 0.8162 | KEEP |
| 140 | dec_hc_alpha0.5 | 0.5104 | 0.8120 | KEEP |
| 141 | dec_hc_alpha2.0 | 0.4841 | 0.8060 | KEEP |
| 142 | dec_hc_alpha5.0 | 0.4727 | 0.7933 | KEEP |
| 143 | dec_hc_mse_weight0.0 | 0.4973 | 0.8073 | KEEP |
| 144 | dec_hc_mse_weight0.5 | 0.4435 | 0.7828 | KEEP |
| 145 | dec_hc_mse_weight1.0 | 0.4891 | 0.8118 | KEEP |
| 146 | dec_hc_pretrain_epochs80_(2x_default) | 0.5002 | 0.8081 | KEEP |

## Next experiment

Tier 6 SOTA exploration after Exp 14: try DINOv2-vit features + KMeans, ProPos (Huang 2023 TPAMI), or DivClust (Karaman 2023). Also explore hyperparameter tuning of Spectral clustering's gamma — it tanked at 0.058 with default and is likely recoverable to ~0.65 with proper tuning.
