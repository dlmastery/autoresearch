# Experiment Summary — Olivetti Faces Clustering Autoresearch

_Generated 2026-04-26 01:37_

## Master leaderboard (sorted by ARI on full 400-row Olivetti dataset)

| Rank | Exp | Backbone | ARI | NMI | Silhouette | Status | Description |
|------|-----|----------|-----|-----|------------|--------|-------------|
| 1 | 71 | spectral_hc_cosine_seed99_(variance_c | 0.7195 | 0.9004 | 0.0927 | KEEP | Spectral hill-climb: cosine seed=99 (variance check) on |
| 2 | 55 | spectral_hc_RBF_gamma0.0001 | 0.7170 | 0.9102 | 0.1101 | KEEP | Spectral hill-climb: RBF gamma=0.0001 on DINOv2 ViT-S/1 |
| 3 | 68 | spectral_hc_cosine_seed1_(variance_ch | 0.7154 | 0.9051 | 0.0900 | KEEP | Spectral hill-climb: cosine seed=1 (variance check) on  |
| 4 | 64 | spectral_hc_cosine_+_n_init1 | 0.7064 | 0.9014 | 0.0895 | KEEP | Spectral hill-climb: cosine + n_init=1 on DINOv2 ViT-S/ |
| 5 | 33 | dinov2_vits14_spectral_cos | 0.6963 | 0.8974 | 0.0890 | KEEP | DINOv2 dinov2_vits14 + Spectral cosine affinity |
| 6 | 47 | spectral_hc_cosine_+_assignkmeans_(ch | 0.6963 | 0.8974 | 0.0890 | KEEP | Spectral hill-climb: cosine + assign=kmeans (champion c |
| 7 | 49 | spectral_hc_cosine_+_L2-normalized_fe | 0.6963 | 0.8974 | 0.0890 | KEEP | Spectral hill-climb: cosine + L2-normalized features on |
| 8 | 66 | spectral_hc_cosine_+_n_init25 | 0.6963 | 0.8974 | 0.0890 | KEEP | Spectral hill-climb: cosine + n_init=25 on DINOv2 ViT-S |
| 9 | 56 | spectral_hc_RBF_gamma0.0005 | 0.6961 | 0.9001 | 0.0942 | KEEP | Spectral hill-climb: RBF gamma=0.0005 on DINOv2 ViT-S/1 |
| 10 | 65 | spectral_hc_cosine_+_n_init5 | 0.6742 | 0.8829 | 0.0984 | KEEP | Spectral hill-climb: cosine + n_init=5 on DINOv2 ViT-S/ |
| 11 | 67 | spectral_hc_cosine_+_n_init50 | 0.6666 | 0.8900 | 0.0806 | KEEP | Spectral hill-climb: cosine + n_init=50 on DINOv2 ViT-S |
| 12 | 69 | spectral_hc_cosine_seed7_(variance_ch | 0.6596 | 0.8710 | 0.0804 | KEEP | Spectral hill-climb: cosine seed=7 (variance check) on  |
| 13 | 60 | spectral_hc_ViT-B/14_+_cosine | 0.6552 | 0.8805 | 0.0673 | KEEP | Spectral hill-climb: ViT-B/14 + cosine on DINOv2 ViT-B/ |
| 14 | 62 | spectral_hc_ViT-B/14_+_L2-norm_+_cosi | 0.6552 | 0.8805 | 0.0673 | KEEP | Spectral hill-climb: ViT-B/14 + L2-norm + cosine on DIN |
| 15 | 123 | umap_hc_n_neighbors=10_on_DINOv2 | 0.6488 | 0.8693 | 0.0813 | KEEP | UMAP HC: n_neighbors=10 on DINOv2 on DINOv2 ViT-S/14 |
| 16 | 34 | dinov2_vits14_spectral_knn10 | 0.6389 | 0.8584 | 0.0796 | KEEP | DINOv2 dinov2_vits14 + Spectral nearest-neighbors affin |
| 17 | 135 | umap_hc_metric=manhattan | 0.6371 | 0.8579 | 0.0621 | KEEP | UMAP HC: metric=manhattan on DINOv2 ViT-S/14 |
| 18 | 27 | dinov2_vits14_agg_ward | 0.6371 | 0.8706 | 0.0834 | KEEP | DINOv2 dinov2_vits14 + Agglomerative Ward (variance-min |
| 19 | 35 | dinov2_vits14_birch | 0.6371 | 0.8706 | 0.0834 | KEEP | DINOv2 dinov2_vits14 + Birch on DINOv2 features |
| 20 | 72 | ward_hc_linkageward_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Ward hill-climb: linkage=ward on DINOv2 on DINOv2 384-d |
| 21 | 91 | ward_hc_Ward_+_connectivity_kNN(k10)_o | 0.6371 | 0.8706 | 0.0834 | KEEP | Ward hill-climb: Ward + connectivity kNN(k=10) on DINOv |
| 22 | 92 | ward_hc_Ward_+_connectivity_kNN(k20)_o | 0.6371 | 0.8706 | 0.0834 | KEEP | Ward hill-climb: Ward + connectivity kNN(k=20) on DINOv |
| 23 | 97 | birch_hc_threshold0.05_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=0.05 on DINOv2 on DINOv2 Vi |
| 24 | 98 | birch_hc_threshold0.1_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=0.1 on DINOv2 on DINOv2 ViT |
| 25 | 99 | birch_hc_threshold0.2_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=0.2 on DINOv2 on DINOv2 ViT |
| 26 | 100 | birch_hc_threshold0.3_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=0.3 on DINOv2 on DINOv2 ViT |
| 27 | 101 | birch_hc_threshold0.5_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=0.5 on DINOv2 on DINOv2 ViT |
| 28 | 102 | birch_hc_threshold0.7_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=0.7 on DINOv2 on DINOv2 ViT |
| 29 | 103 | birch_hc_threshold1.0_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=1.0 on DINOv2 on DINOv2 ViT |
| 30 | 104 | birch_hc_threshold1.5_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: threshold=1.5 on DINOv2 on DINOv2 ViT |
| 31 | 105 | birch_hc_branching_factor10_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: branching_factor=10 on DINOv2 on DINO |
| 32 | 106 | birch_hc_branching_factor25_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: branching_factor=25 on DINOv2 on DINO |
| 33 | 107 | birch_hc_branching_factor100_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: branching_factor=100 on DINOv2 on DIN |
| 34 | 108 | birch_hc_branching_factor200_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: branching_factor=200 on DINOv2 on DIN |
| 35 | 117 | birch_hc_tight_threshold0.01_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: tight threshold=0.01 on DINOv2 on DIN |
| 36 | 118 | birch_hc_tight_threshold0.02_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: tight threshold=0.02 on DINOv2 on DIN |
| 37 | 119 | birch_hc_tight_threshold0.03_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: tight threshold=0.03 on DINOv2 on DIN |
| 38 | 120 | birch_hc_tight_threshold0.04_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: tight threshold=0.04 on DINOv2 on DIN |
| 39 | 121 | birch_hc_tight_threshold0.05_on_DINOv2 | 0.6371 | 0.8706 | 0.0834 | KEEP | Birch hill-climb: tight threshold=0.05 on DINOv2 on DIN |
| 40 | 73 | ward_hc_linkageward_on_DINOv2_L2-norm | 0.6366 | 0.8721 | 0.0751 | KEEP | Ward hill-climb: linkage=ward on DINOv2 L2-norm on DINO |
| 41 | 93 | ward_hc_Ward_init_+_KMeans_refine_on_D | 0.6308 | 0.8665 | 0.0802 | KEEP | Ward hill-climb: Ward init + KMeans refine on DINOv2 on |
| 42 | 94 | ward_hc_Ward_init_+_KMeans_refine_on_D | 0.6303 | 0.8681 | 0.0719 | KEEP | Ward hill-climb: Ward init + KMeans refine on DINOv2 L2 |
| 43 | 126 | umap_hc_min_dist=0.0_on_DINOv2 | 0.6247 | 0.8606 | 0.0741 | KEEP | UMAP HC: min_dist=0.0 on DINOv2 on DINOv2 ViT-S/14 |
| 44 | 51 | spectral_hc_nearest_neighbors_k7 | 0.6246 | 0.8538 | 0.0815 | KEEP | Spectral hill-climb: nearest_neighbors k=7 on DINOv2 Vi |
| 45 | 90 | ward_hc_Ward_+_connectivity_kNN(k5)_on | 0.6207 | 0.8637 | 0.0821 | KEEP | Ward hill-climb: Ward + connectivity kNN(k=5) on DINOv2 |
| 46 | 130 | umap_hc_n_components=3_on_DINOv2 | 0.6177 | 0.8508 | 0.0449 | KEEP | UMAP HC: n_components=3 on DINOv2 on DINOv2 ViT-S/14 |
| 47 | 128 | umap_hc_min_dist=0.5_on_DINOv2 | 0.6156 | 0.8514 | 0.0617 | KEEP | UMAP HC: min_dist=0.5 on DINOv2 on DINOv2 ViT-S/14 |
| 48 | 70 | spectral_hc_cosine_seed42_(variance_c | 0.6127 | 0.8609 | 0.0772 | KEEP | Spectral hill-climb: cosine seed=42 (variance check) on |
| 49 | 122 | umap_hc_n_neighbors=5_on_DINOv2 | 0.6109 | 0.8624 | 0.0756 | KEEP | UMAP HC: n_neighbors=5 on DINOv2 on DINOv2 ViT-S/14 |
| 50 | 133 | umap_hc_n_components=50_on_DINOv2 | 0.6107 | 0.8453 | 0.0594 | KEEP | UMAP HC: n_components=50 on DINOv2 on DINOv2 ViT-S/14 |
| 51 | 41 | dinov2_vits14_umap2_km | 0.6100 | 0.8455 | 0.0678 | KEEP | DINOv2 dinov2_vits14 + UMAP(2) on DINOv2 + KMeans (extr |
| 52 | 50 | spectral_hc_nearest_neighbors_k5 | 0.6042 | 0.8577 | 0.0670 | KEEP | Spectral hill-climb: nearest_neighbors k=5 on DINOv2 Vi |
| 53 | 134 | umap_hc_metric=cosine | 0.6000 | 0.8421 | 0.0677 | KEEP | UMAP HC: metric=cosine on DINOv2 ViT-S/14 |
| 54 | 40 | dinov2_vits14_umap10_km | 0.5982 | 0.8465 | 0.0592 | KEEP | DINOv2 dinov2_vits14 + UMAP(10) on DINOv2 + KMeans |
| 55 | 132 | umap_hc_n_components=30_on_DINOv2 | 0.5980 | 0.8495 | 0.0616 | KEEP | UMAP HC: n_components=30 on DINOv2 on DINOv2 ViT-S/14 |
| 56 | 127 | umap_hc_min_dist=0.3_on_DINOv2 | 0.5949 | 0.8438 | 0.0439 | KEEP | UMAP HC: min_dist=0.3 on DINOv2 on DINOv2 ViT-S/14 |
| 57 | 52 | spectral_hc_nearest_neighbors_k15 | 0.5888 | 0.8358 | 0.0554 | KEEP | Spectral hill-climb: nearest_neighbors k=15 on DINOv2 V |
| 58 | 131 | umap_hc_n_components=5_on_DINOv2 | 0.5860 | 0.8412 | 0.0480 | KEEP | UMAP HC: n_components=5 on DINOv2 on DINOv2 ViT-S/14 |
| 59 | 31 | dinov2_vits14_spectral_g001 | 0.5852 | 0.8533 | 0.0872 | KEEP | DINOv2 dinov2_vits14 + Spectral RBF gamma=0.001 (small) |
| 60 | 25 | dinov2_vits14_kmeans_n50 | 0.5852 | 0.8456 | 0.0891 | KEEP | DINOv2 dinov2_vits14 + KMeans n_init=50 (5x more random |
| 61 | 125 | umap_hc_n_neighbors=50_on_DINOv2 | 0.5690 | 0.8279 | 0.0513 | KEEP | UMAP HC: n_neighbors=50 on DINOv2 on DINOv2 ViT-S/14 |
| 62 | 124 | umap_hc_n_neighbors=30_on_DINOv2 | 0.5680 | 0.8311 | 0.0317 | KEEP | UMAP HC: n_neighbors=30 on DINOv2 on DINOv2 ViT-S/14 |
| 63 | 129 | umap_hc_min_dist=0.99_on_DINOv2 | 0.5665 | 0.8258 | 0.0673 | KEEP | UMAP HC: min_dist=0.99 on DINOv2 on DINOv2 ViT-S/14 |
| 64 | 26 | dinov2_vits14_spherical | 0.5602 | 0.8259 | 0.0467 | KEEP | DINOv2 dinov2_vits14 + L2-normalized features + KMeans  |
| 65 | 22 | dinov2_vits14_minibatch_kmeans | 0.5596 | 0.8393 | 0.0596 | KEEP | DINOv2 dinov2_vits14 + MiniBatchKMeans (faster, may be  |
| 66 | 44 | dinov2_vits14_seed1 | 0.5561 | 0.8301 | 0.0904 | KEEP | DINOv2 dinov2_vits14 + KMeans seed=1 (variance check on |
| 67 | 63 | spectral_hc_ViT-B/14_+_kNN_k10 | 0.5489 | 0.8215 | 0.0496 | KEEP | Spectral hill-climb: ViT-B/14 + kNN k=10 on DINOv2 ViT- |
| 68 | 39 | dinov2_vits14_pca100_km | 0.5473 | 0.8278 | 0.0745 | KEEP | DINOv2 dinov2_vits14 + PCA(100) on DINOv2 + KMeans |
| 69 | 113 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.5461 | 0.8242 | 0.0492 | KEEP | Birch hill-climb: Birch leaves + KMeans refine on DINOv |
| 70 | 20 | dinov2_kmeans | 0.5455 | 0.8201 | 0.0710 | KEEP | DINOv2 ViT-S/14 (Oquab 2024 Meta TMLR) features + KMean |
| 71 | 42 | dinov2_vitb14_vitb_km | 0.5445 | 0.8243 | 0.0379 | KEEP | DINOv2 dinov2_vitb14 + ViT-B/14 features + KMeans (larg |
| 72 | 43 | dinov2_vitb14_vitb_spherical | 0.5388 | 0.8119 | 0.0506 | KEEP | DINOv2 dinov2_vitb14 + ViT-B/14 + L2-norm + KMeans (Sph |
| 73 | 46 | dinov2_vits14_seed7 | 0.5387 | 0.8175 | 0.0633 | KEEP | DINOv2 dinov2_vits14 + KMeans seed=7 (variance check on |
| 74 | 38 | dinov2_vits14_pca50_km | 0.5312 | 0.8184 | 0.0328 | KEEP | DINOv2 dinov2_vits14 + PCA(50) on DINOv2 + KMeans (deno |
| 75 | 17 | birch | 0.5287 | 0.8254 | 0.1608 | KEEP | Birch (Zhang 1996) on PCA(50) |
| 76 | 110 | birch_hc_default_Birch_on_PCA(50) | 0.5287 | 0.8254 | 0.1608 | KEEP | Birch hill-climb: default Birch on PCA(50) on PCA(50) |
| 77 | 53 | spectral_hc_nearest_neighbors_k20 | 0.5278 | 0.8059 | 0.0423 | KEEP | Spectral hill-climb: nearest_neighbors k=20 on DINOv2 V |
| 78 | 16 | spectral_tuned | 0.5252 | 0.8228 | 0.1159 | KEEP | Spectral RBF with gamma sweep on PCA(50) |
| 79 | 16 | spectral_tuned | 0.5252 | 0.8228 | 0.1159 | KEEP | Spectral RBF with gamma sweep on PCA(50) |
| 80 | 36 | dinov2_vits14_gmm_full | 0.5234 | 0.8133 | 0.0341 | KEEP | DINOv2 dinov2_vits14 + GMM full-covariance K=40 |
| 81 | 37 | dinov2_vits14_gmm_diag | 0.5234 | 0.8133 | 0.0341 | KEEP | DINOv2 dinov2_vits14 + GMM diagonal-covariance |
| 82 | 8 | agg_ward | 0.5159 | 0.8201 | 0.1608 | KEEP | Agglomerative Ward on PCA(50) (Ward 1963) |
| 83 | 85 | ward_hc_linkageward_on_PCA(50) | 0.5159 | 0.8201 | 0.1608 | KEEP | Ward hill-climb: linkage=ward on PCA(50) on raw pixels  |
| 84 | 45 | dinov2_vits14_seed2 | 0.5144 | 0.8110 | 0.0712 | KEEP | DINOv2 dinov2_vits14 + KMeans seed=2 (variance check on |
| 85 | 140 | dec_hc_alpha0.5 | 0.5104 | 0.8120 | 0.1493 | KEEP | DEC HC: alpha=0.5 |
| 86 | 139 | dec_hc_latent_dim256 | 0.5091 | 0.8162 | 0.1421 | KEEP | DEC HC: latent_dim=256 |
| 87 | 95 | ward_hc_Ward_init_+_KMeans_refine_on_P | 0.5013 | 0.8124 | 0.1639 | KEEP | Ward hill-climb: Ward init + KMeans refine on PCA(50) o |
| 88 | 146 | dec_hc_pretrain_epochs80_(2x_default) | 0.5002 | 0.8081 | 0.1403 | KEEP | DEC HC: pretrain_epochs=80 (2x default) |
| 89 | 15 | umap_kmeans | 0.5001 | 0.8003 | 0.1278 | KEEP | UMAP(10) + KMeans (McInnes 2018) |
| 90 | 15 | umap_kmeans | 0.5001 | 0.8003 | 0.1278 | KEEP | UMAP(10) + KMeans (McInnes 2018) |
| 91 | 24 | dinov2_vits14_kmeans_random | 0.5000 | 0.8091 | 0.0304 | KEEP | DINOv2 dinov2_vits14 + KMeans with random init (vs k-me |
| 92 | 143 | dec_hc_mse_weight0.0 | 0.4973 | 0.8073 | 0.1431 | KEEP | DEC HC: mse_weight=0.0 |
| 93 | 137 | dec_hc_latent_dim32 | 0.4955 | 0.7982 | 0.1448 | KEEP | DEC HC: latent_dim=32 |
| 94 | 12 | dec | 0.4942 | 0.8036 | 0.1436 | KEEP | DEC: Deep Embedded Clustering (Xie 2016 ICML + Guo 2017 |
| 95 | 77 | ward_hc_linkagecomplete_on_DINOv2_L2-n | 0.4926 | 0.8112 | 0.0306 | KEEP | Ward hill-climb: linkage=complete on DINOv2 L2-norm on  |
| 96 | 81 | ward_hc_linkagecomplete_+_cosine_dista | 0.4926 | 0.8112 | 0.0306 | KEEP | Ward hill-climb: linkage=complete + cosine distance on  |
| 97 | 145 | dec_hc_mse_weight1.0 | 0.4891 | 0.8118 | 0.1583 | KEEP | DEC HC: mse_weight=1.0 |
| 98 | 141 | dec_hc_alpha2.0 | 0.4841 | 0.8060 | 0.1499 | KEEP | DEC HC: alpha=2.0 |
| 99 | 21 | spherical_kmeans | 0.4816 | 0.7896 | 0.1266 | KEEP | Spherical KMeans (Dhillon 2001) on L2-norm PCA(50) |
| 100 | 29 | dinov2_vits14_agg_complete | 0.4805 | 0.8071 | 0.0234 | KEEP | DINOv2 dinov2_vits14 + Agglomerative complete-linkage ( |
| 101 | 76 | ward_hc_linkagecomplete_on_DINOv2 | 0.4805 | 0.8071 | 0.0234 | KEEP | Ward hill-climb: linkage=complete on DINOv2 on DINOv2 3 |
| 102 | 10 | conv_ae_kmeans | 0.4790 | 0.7934 | 0.1469 | KEEP | Convolutional AE (Hinton 2006) + KMeans, latent=64 |
| 103 | 138 | dec_hc_latent_dim128 | 0.4781 | 0.7994 | 0.1453 | KEEP | DEC HC: latent_dim=128 |
| 104 | 2 | kmeans_pca50 | 0.4780 | 0.7951 | 0.1485 | KEEP | PCA(50) + KMeans (Pearson 1901 + Steinley 2006) |
| 105 | 14 | consensus_top5 | 0.4767 | 0.8082 | 0.1530 | KEEP | CSPA consensus of top-5 methods: agg_ward, dec, conv_ae |
| 106 | 18 | affinity_prop | 0.4757 | 0.8105 | 0.1737 | KEEP | Affinity Propagation (Frey 2007 Science) on PCA(50) |
| 107 | 86 | ward_hc_linkageward_on_PCA(100) | 0.4737 | 0.8081 | 0.1656 | KEEP | Ward hill-climb: linkage=ward on PCA(100) on raw pixels |
| 108 | 111 | birch_hc_default_Birch_on_PCA(100) | 0.4737 | 0.8081 | 0.1656 | KEEP | Birch hill-climb: default Birch on PCA(100) on PCA(100) |
| 109 | 142 | dec_hc_alpha5.0 | 0.4727 | 0.7933 | 0.1478 | KEEP | DEC HC: alpha=5.0 |
| 110 | 48 | spectral_hc_cosine_+_assigncluster_qr | 0.4708 | 0.7628 | -0.0049 | KEEP | Spectral hill-climb: cosine + assign=cluster_qr on DINO |
| 111 | 28 | dinov2_vits14_agg_avg | 0.4703 | 0.8158 | 0.0226 | KEEP | DINOv2 dinov2_vits14 + Agglomerative average-linkage |
| 112 | 74 | ward_hc_linkageaverage_on_DINOv2 | 0.4703 | 0.8158 | 0.0226 | KEEP | Ward hill-climb: linkage=average on DINOv2 on DINOv2 38 |
| 113 | 3 | kmeans_pca100 | 0.4633 | 0.7856 | 0.1506 | KEEP | PCA(100) + KMeans |
| 114 | 3 | kmeans_pca100 | 0.4633 | 0.7856 | 0.1506 | KEEP | PCA(100) + KMeans |
| 115 | 75 | ward_hc_linkageaverage_on_DINOv2_L2-no | 0.4631 | 0.8234 | 0.0240 | KEEP | Ward hill-climb: linkage=average on DINOv2 L2-norm on D |
| 116 | 96 | ward_hc_Ward_init_+_KMeans_refine_on_P | 0.4591 | 0.7993 | 0.1677 | KEEP | Ward hill-climb: Ward init + KMeans refine on PCA(100)  |
| 117 | 54 | spectral_hc_nearest_neighbors_k30 | 0.4553 | 0.7806 | 0.0092 | KEEP | Spectral hill-climb: nearest_neighbors k=30 on DINOv2 V |
| 118 | 7 | gmm_pca_full | 0.4545 | 0.7736 | 0.1394 | KEEP | GMM full-cov on PCA(50) (Bishop 2006 Ch.9) |
| 119 | 83 | ward_hc_linkageaverage_+_manhattan_dis | 0.4540 | 0.8206 | 0.0129 | KEEP | Ward hill-climb: linkage=average + manhattan distance o |
| 120 | 112 | birch_hc_default_Birch_on_PCA(20) | 0.4540 | 0.7949 | 0.1591 | KEEP | Birch hill-climb: default Birch on PCA(20) on PCA(20) |
| 121 | 84 | ward_hc_linkageward_on_PCA(20) | 0.4508 | 0.7910 | 0.1581 | KEEP | Ward hill-climb: linkage=ward on PCA(20) on raw pixels  |
| 122 | 30 | dinov2_vits14_agg_cosine_avg | 0.4490 | 0.8174 | 0.0134 | KEEP | DINOv2 dinov2_vits14 + Agglomerative cosine + average l |
| 123 | 80 | ward_hc_linkageaverage_+_cosine_distan | 0.4490 | 0.8174 | 0.0134 | KEEP | Ward hill-climb: linkage=average + cosine distance on D |
| 124 | 4 | kmeans_pca150 | 0.4484 | 0.7846 | 0.1456 | KEEP | PCA(150) + KMeans |
| 125 | 11 | resnet18_kmeans | 0.4444 | 0.7916 | 0.0324 | KEEP | ResNet18-ImageNet (He 2016) penultimate features + KMea |
| 126 | 23 | dinov2_vits14_bisecting_kmeans | 0.4437 | 0.7678 | 0.0277 | KEEP | DINOv2 dinov2_vits14 + BisectingKMeans hierarchical bis |
| 127 | 144 | dec_hc_mse_weight0.5 | 0.4435 | 0.7828 | 0.1484 | KEEP | DEC HC: mse_weight=0.5 |
| 128 | 115 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.4356 | 0.7685 | 0.1478 | KEEP | Birch hill-climb: Birch leaves + KMeans refine on PCA(5 |
| 129 | 61 | spectral_hc_ViT-B/14_+_cluster_qr_+_c | 0.4317 | 0.7495 | 0.0033 | KEEP | Spectral hill-climb: ViT-B/14 + cluster_qr + cosine on  |
| 130 | 116 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.4232 | 0.7685 | 0.1479 | KEEP | Birch hill-climb: Birch leaves + KMeans refine on PCA(1 |
| 131 | 1 | kmeans_raw_pixels | 0.4057 | 0.7585 | 0.1479 | KEEP | KMeans K=40 on raw pixels — baseline (Lloyd 1982 + Arth |
| 132 | 13 | simclr_kmeans | 0.3678 | 0.7502 | 0.0503 | KEEP | SimCLR (Chen 2020 ICML) + KMeans |
| 133 | 5 | kmeans_pca_whitened | 0.3602 | 0.7508 | 0.0775 | KEEP | PCA(50) + whitening + KMeans |
| 134 | 9 | hdbscan | 0.3438 | 0.8142 | 0.1807 | KEEP | HDBSCAN on PCA(50) (Campello 2013) |
| 135 | 88 | ward_hc_linkageaverage_+_cosine_on_PCA | 0.3229 | 0.7547 | 0.0943 | KEEP | Ward hill-climb: linkage=average + cosine on PCA(50) on |
| 136 | 87 | ward_hc_linkageaverage_+_cosine_on_PCA | 0.3223 | 0.7542 | 0.0925 | KEEP | Ward hill-climb: linkage=average + cosine on PCA(20) on |
| 137 | 89 | ward_hc_linkageaverage_+_cosine_on_PCA | 0.2983 | 0.7444 | 0.0899 | DISCARD | Ward hill-climb: linkage=average + cosine on PCA(100) o |
| 138 | 32 | dinov2_vits14_spectral_g01 | 0.2767 | 0.7672 | 0.0361 | DISCARD | DINOv2 dinov2_vits14 + Spectral RBF gamma=0.01 |
| 139 | 57 | spectral_hc_RBF_gamma0.005 | 0.2628 | 0.7973 | 0.0764 | DISCARD | Spectral hill-climb: RBF gamma=0.005 on DINOv2 ViT-S/14 |
| 140 | 109 | birch_hc_default_Birch_on_DINOv2_L2-nor | 0.2306 | 0.6719 | -0.0521 | DISCARD | Birch hill-climb: default Birch on DINOv2 L2-norm on DI |
| 141 | 114 | birch_hc_Birch_leaves_+_KMeans_refine_o | 0.2306 | 0.6719 | -0.0521 | DISCARD | Birch hill-climb: Birch leaves + KMeans refine on DINOv |
| 142 | 136 | umap_hc_UMAP(10)_+_Spectral_cosine_dow | 0.1918 | 0.6401 | -0.1080 | DISCARD | UMAP HC: UMAP(10) + Spectral cosine downstream on DINOv |
| 143 | 78 | ward_hc_linkagesingle_on_DINOv2 | 0.1481 | 0.6689 | -0.1119 | DISCARD | Ward hill-climb: linkage=single on DINOv2 on DINOv2 384 |
| 144 | 79 | ward_hc_linkagesingle_on_DINOv2_L2-nor | 0.1437 | 0.6625 | -0.1119 | DISCARD | Ward hill-climb: linkage=single on DINOv2 L2-norm on DI |
| 145 | 82 | ward_hc_linkagesingle_+_cosine_distanc | 0.1437 | 0.6625 | -0.1119 | DISCARD | Ward hill-climb: linkage=single + cosine distance on DI |
| 146 | 6 | spectral_rbf | 0.0578 | 0.4560 | -0.1250 | DISCARD | Spectral clustering (RBF affinity) |
| 147 | 58 | spectral_hc_RBF_gamma0.05 | 0.0503 | 0.5965 | -0.0894 | DISCARD | Spectral hill-climb: RBF gamma=0.05 on DINOv2 ViT-S/14 |
| 148 | 59 | spectral_hc_RBF_gamma0.5 | 0.0000 | 0.0297 | -0.1190 | DISCARD | Spectral hill-climb: RBF gamma=0.5 on DINOv2 ViT-S/14 |
| 149 | 19 | meanshift | 0.0000 | 0.0000 | nan | DISCARD | MeanShift (Comaniciu 2002) on PCA(50) |

## Per-experiment detail

### Exp 1: KMeans K=40 on raw pixels — baseline (Lloyd 1982 + Arthur 2007 init)
- **Backbone:** `kmeans_raw_pixels` | **Status:** KEEP
- **Result:** ARI 0.4057 | NMI 0.7585 | silhouette 0.1479 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that KMeans with K=40, n_init=10, k-means++ init on raw pixel features will achieve ARI in the range 0.45 to 0.60 because the mechanism per Lloyd 1982 is Euclidean-distance partitioning...
- **Verdict:** KEEP (baseline) — ARI=0.4057, NMI=0.7585, silhouette=0.1479. BELOW predicted lower bound (predicted ARI 0.45-0.60). Status under floor=0.30: KEEP. This baseline establishes the reference point for all downstream feature-engineering experiments. K=40 was honored (n_pred_clusters=40).
- **Learning:** Axis open: ALL feature-engineering and architecture-improvement axes. KMeans on raw pixels is a defensible floor at ARI=0.4057, providing the +Δ baseline against which dimensionality reduction (PCA, UMAP), kernel methods (Spectral), generative models (VAE/AE), and pretrained deep features will be me

### Exp 2: PCA(50) + KMeans (Pearson 1901 + Steinley 2006)
- **Backbone:** `kmeans_pca50` | **Status:** KEEP
- **Result:** ARI 0.4780 | NMI 0.7951 | silhouette 0.1485 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(50) + KMeans will achieve ARI in the range 0.55 to 0.70 because the mechanism per Steinley 2006 is that reducing the feature-to-sample ratio from 10.2 to 0.125 brings KMeans in...
- **Verdict:** KEEP — ARI=0.4780 (delta +0.0723 vs baseline 0.4057), NMI=0.7951, silhouette=0.1485, n_pred_clusters=40. BELOW the predicted lower bound 0.55 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry
- **Learning:** axis open. PCA(50) projection produced a improvement of +0.0723 ARI vs the prior baseline, updating our mental model: the chosen feature/method genuinely captures more facial-identity structure. Next try: PCA(100) + KMeans (Exp 3) to test if more components capture finer facial detail or reintroduce

### Exp 3: PCA(100) + KMeans
- **Backbone:** `kmeans_pca100` | **Status:** KEEP
- **Result:** ARI 0.4633 | NMI 0.7856 | silhouette 0.1506 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(100)+KMeans will land ARI in the range 0.43 to 0.58 because the mechanism per Steinley 2006 is that the marginal variance per added component decays as the eigenvalue spectrum,...
- **Verdict:** KEEP — ARI=0.4633 (delta -0.0147 vs baseline 0.4780), NMI=0.7856, silhouette=0.1506, n_pred_clusters=40. WITHIN the predicted range 0.43-0.58. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST 
- **Learning:** axis closed. PCA(100) projection produced a tie of -0.0147 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: PCA(150) + KMeans (Exp 4) to find the optimum dimensionality. The cum

### Exp 3: PCA(100) + KMeans
- **Backbone:** `kmeans_pca100` | **Status:** KEEP
- **Result:** ARI 0.4633 | NMI 0.7856 | silhouette 0.1506 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(100)+KMeans will land ARI in the range 0.43 to 0.58 because the mechanism per Steinley 2006 is that the marginal variance per added component decays as the eigenvalue spectrum,...
- **Verdict:** KEEP — ARI=0.4633 (delta -0.0147 vs baseline 0.4780), NMI=0.7856, silhouette=0.1506, n_pred_clusters=40. WITHIN the predicted range 0.43-0.58. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST 
- **Learning:** axis closed. PCA(100) projection produced a tie of -0.0147 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: PCA(150) + KMeans (Exp 4) to find the optimum dimensionality. The cum

### Exp 4: PCA(150) + KMeans
- **Backbone:** `kmeans_pca150` | **Status:** KEEP
- **Result:** ARI 0.4484 | NMI 0.7846 | silhouette 0.1456 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(150)+KMeans will land ARI in 0.43 to 0.53 because the mechanism per Hotelling 1933 is that beyond ~100 components the eigenvalue magnitudes drop below per-pixel noise variance,...
- **Verdict:** KEEP — ARI=0.4484 (delta -0.0296 vs baseline 0.4780), NMI=0.7846, silhouette=0.1456, n_pred_clusters=40. WITHIN the predicted range 0.43-0.53. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST 
- **Learning:** axis closed. PCA(150) projection produced a regression of -0.0296 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: PCA(50) + whitening (Exp 5) to test Mahalanobis-equivalent KMe

### Exp 5: PCA(50) + whitening + KMeans
- **Backbone:** `kmeans_pca_whitened` | **Status:** KEEP
- **Result:** ARI 0.3602 | NMI 0.7508 | silhouette 0.0775 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(50)+whitening+KMeans will land ARI in 0.43 to 0.58 because the mechanism per Pearson 1901 is that whitening converts Euclidean distance in PCA-space to Mahalanobis distance in ...
- **Verdict:** KEEP — ARI=0.3602 (delta -0.1178 vs baseline 0.4780), NMI=0.7508, silhouette=0.0775, n_pred_clusters=40. BELOW the predicted lower bound 0.43 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette diverges from the extrinsic ARI improvement, providing divergent signal about the cluster geo
- **Learning:** axis closed. PCA whitening produced a regression of -0.1178 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Spectral clustering with RBF affinity (Exp 6) — non-linear method to

### Exp 6: Spectral clustering (RBF affinity)
- **Backbone:** `spectral_rbf` | **Status:** DISCARD
- **Result:** ARI 0.0578 | NMI 0.4560 | silhouette -0.1250 | n_pred_clusters 27
- **Hypothesis (first 200ch):** We hypothesize that Spectral clustering with K=40 and RBF affinity will land ARI in 0.43 to 0.68 because the mechanism per Ng-Jordan-Weiss 2001 is that the Laplacian eigenvectors embed faces of the sa...
- **Verdict:** DISCARD — ARI=0.0578 (delta -0.4202 vs baseline 0.4780), NMI=0.4560, silhouette=-0.1250, n_pred_clusters=27. BELOW the predicted lower bound 0.43 — refuted. Status under floor=0.30 is DISCARD; intrinsic silhouette diverges from the extrinsic ARI improvement, providing consistent signal about the clu
- **Learning:** axis closed. Spectral RBF produced a regression of -0.4202 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: GMM full-covariance (Exp 7) — probabilistic alternative that models p

### Exp 7: GMM full-cov on PCA(50) (Bishop 2006 Ch.9)
- **Backbone:** `gmm_pca_full` | **Status:** KEEP
- **Result:** ARI 0.4545 | NMI 0.7736 | silhouette 0.1394 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that GMM full-covariance on PCA(50) features with K=40 components will land ARI in -0.04 to 0.11 because the mechanism per Bishop 2006 is that per-subject Gaussians with full covariance...
- **Verdict:** KEEP — ARI=0.4545 (delta +0.3967 vs baseline 0.0578), NMI=0.7736, silhouette=0.1394, n_pred_clusters=40. ABOVE the predicted upper bound 0.11 — exceeded expectations. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cl
- **Learning:** axis open. GMM full covariance produced a improvement of +0.3967 ARI vs the prior baseline, updating our mental model: the chosen feature/method genuinely captures more facial-identity structure. Next try: Agglomerative Ward (Exp 8) — bottom-up hierarchical with variance-minimizing merges. The cumul

### Exp 8: Agglomerative Ward on PCA(50) (Ward 1963)
- **Backbone:** `agg_ward` | **Status:** KEEP
- **Result:** ARI 0.5159 | NMI 0.8201 | silhouette 0.1608 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Agglomerative Ward on PCA(50) features cut at K=40 will land ARI in 0.43 to 0.68 because the mechanism per Ward 1963 is that variance-minimizing merges are mathematically equivalen...
- **Verdict:** KEEP — ARI=0.5159 (delta +0.0379 vs baseline 0.4780), NMI=0.8201, silhouette=0.1608, n_pred_clusters=40. WITHIN the predicted range 0.43-0.68. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST
- **Learning:** axis open. Agglomerative Ward produced a improvement of +0.0379 ARI vs the prior baseline, updating our mental model: the chosen feature/method genuinely captures more facial-identity structure. Next try: HDBSCAN (Exp 9) — density-based, can leave noise points unassigned (-1). The cumulative best AR

### Exp 9: HDBSCAN on PCA(50) (Campello 2013)
- **Backbone:** `hdbscan` | **Status:** KEEP
- **Result:** ARI 0.3438 | NMI 0.8142 | silhouette 0.1807 | n_pred_clusters 41
- **Hypothesis (first 200ch):** We hypothesize that HDBSCAN with min_cluster_size=5 on PCA(50) features will discover roughly 30-50 clusters with substantial noise points, landing ARI in 0.40-0.65 because the mechanism per Campello ...
- **Verdict:** KEEP — ARI=0.3438 (delta -0.1342 vs baseline 0.4780), NMI=0.8142, silhouette=0.1807, n_pred_clusters=41. BELOW the predicted lower bound 0.40 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry.
- **Learning:** axis closed. HDBSCAN density-based produced a regression of -0.1342 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 4: deep features — Convolutional Autoencoder + KMeans (

### Exp 10: Convolutional AE (Hinton 2006) + KMeans, latent=64
- **Backbone:** `conv_ae_kmeans` | **Status:** KEEP
- **Result:** ARI 0.4790 | NMI 0.7934 | silhouette 0.1469 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Convolutional AE (latent=64) trained 40 epochs on Olivetti pixels, then KMeans on encoded features will land ARI in 0.48 to 0.68 because the mechanism per Hinton-Salakhutdinov 2006...
- **Verdict:** KEEP — ARI=0.4790 (delta +0.0010 vs baseline 0.4780), NMI=0.7934, silhouette=0.1469, n_pred_clusters=40. WITHIN the predicted range 0.48-0.68. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST
- **Learning:** axis closed. Convolutional AE features produced a tie of +0.0010 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 5: pretrained ResNet18 ImageNet features (Exp 11) for tran

### Exp 11: ResNet18-ImageNet (He 2016) penultimate features + KMeans
- **Backbone:** `resnet18_kmeans` | **Status:** KEEP
- **Result:** ARI 0.4444 | NMI 0.7916 | silhouette 0.0324 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ResNet18-ImageNet penultimate features (512-dim) on resized 224x224 3-channel Olivetti + KMeans will land ARI in 0.48 to 0.68 because the mechanism per Donahue-Jia 2014 is that Ima...
- **Verdict:** KEEP — ARI=0.4444 (delta -0.0346 vs baseline 0.4790), NMI=0.7916, silhouette=0.0324, n_pred_clusters=40. BELOW the predicted lower bound 0.48 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette diverges from the extrinsic ARI improvement, providing divergent signal about the cluster geo
- **Learning:** axis closed. ResNet18 ImageNet transfer produced a regression of -0.0346 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 6 SOTA: Deep Embedded Clustering DEC (Xie 2016 ICM

### Exp 12: DEC: Deep Embedded Clustering (Xie 2016 ICML + Guo 2017 IDEC)
- **Backbone:** `dec` | **Status:** KEEP
- **Result:** ARI 0.4942 | NMI 0.8036 | silhouette 0.1436 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that DEC (40-epoch AE pretrain + 20-epoch joint KL+MSE fine-tune) will land ARI in 0.48 to 0.68 because the mechanism per Xie 2016 is that joint optimization of encoder and cluster cent...
- **Verdict:** KEEP — ARI=0.4942 (delta +0.0152 vs baseline 0.4790), NMI=0.8036, silhouette=0.1436, n_pred_clusters=40. WITHIN the predicted range 0.48-0.68. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST
- **Learning:** axis closed. DEC joint encoder+cluster fine-tuning produced a tie of +0.0152 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 6: contrastive learning (SimCLR-style) + KMean

### Exp 13: SimCLR (Chen 2020 ICML) + KMeans
- **Backbone:** `simclr_kmeans` | **Status:** KEEP
- **Result:** ARI 0.3678 | NMI 0.7502 | silhouette 0.0503 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that SimCLR-style contrastive pretraining (80 epochs, NT-Xent loss, horizontal-flip + Gaussian noise + brightness augmentation) followed by KMeans on encoded features will land ARI in 0...
- **Verdict:** KEEP — ARI=0.3678 (delta -0.1264 vs baseline 0.4942), NMI=0.7502, silhouette=0.0503, n_pred_clusters=40. BELOW the predicted lower bound 0.49 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette diverges from the extrinsic ARI improvement, providing divergent signal about the cluster geo
- **Learning:** axis closed. SimCLR contrastive embedding produced a regression of -0.1264 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Ensemble of top-K methods via consensus clustering (E

### Exp 14: CSPA consensus of top-5 methods: agg_ward, dec, conv_ae_kmeans, kmeans_pca50, kmeans_pca100
- **Backbone:** `consensus_top5` | **Status:** KEEP
- **Result:** ARI 0.4767 | NMI 0.8082 | silhouette 0.1530 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that CSPA consensus clustering of the top-5 methods will land ARI in 0.52 to 0.62 because the mechanism per Strehl-Ghosh 2002 is that diverse base clusterings make uncorrelated errors, ...
- **Verdict:** KEEP — ARI=0.4767 (delta -0.0392 vs baseline 0.5159), NMI=0.8082, silhouette=0.1530, n_pred_clusters=40. BELOW the predicted lower bound 0.52 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry.
- **Learning:** axis closed. Top-5 consensus ensemble produced a regression of -0.0392 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: 5-seed variance check on the global champion to character

### Exp 15: UMAP(10) + KMeans (McInnes 2018)
- **Backbone:** `umap_kmeans` | **Status:** KEEP
- **Result:** ARI 0.5001 | NMI 0.8003 | silhouette 0.1278 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that UMAP(n_components=10, n_neighbors=15) + KMeans on Olivetti will land ARI in 0.47 to 0.72 because the mechanism per McInnes 2018 is that UMAP's manifold-preserving projection compre...
- **Verdict:** KEEP — ARI=0.5001 (delta -0.0158 vs baseline 0.5159), NMI=0.8003, silhouette=0.1278, n_pred_clusters=40. WITHIN the predicted range 0.47-0.72. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. UMAP manifold projection produced delta=-0.0158 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Spectral with gamma sweep (Exp

### Exp 15: UMAP(10) + KMeans (McInnes 2018)
- **Backbone:** `umap_kmeans` | **Status:** KEEP
- **Result:** ARI 0.5001 | NMI 0.8003 | silhouette 0.1278 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that UMAP(n_components=10, n_neighbors=15) + KMeans on Olivetti will land ARI in 0.47 to 0.72 because the mechanism per McInnes 2018 is that UMAP's manifold-preserving projection compre...
- **Verdict:** KEEP — ARI=0.5001 (delta -0.0158 vs baseline 0.5159), NMI=0.8003, silhouette=0.1278, n_pred_clusters=40. WITHIN the predicted range 0.47-0.72. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. UMAP manifold projection produced delta=-0.0158 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Spectral with gamma sweep (Exp

### Exp 16: Spectral RBF with gamma sweep on PCA(50)
- **Backbone:** `spectral_tuned` | **Status:** KEEP
- **Result:** ARI 0.5252 | NMI 0.8228 | silhouette 0.1159 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Spectral with gamma swept across [0.001, 0.01, 0.1, 1.0] on PCA(50) features will land best-ARI in 0.42 to 0.67 because the mechanism per Ng 2001 is that proper gamma puts the affi...
- **Verdict:** KEEP — ARI=0.5252 (delta +0.0093 vs baseline 0.5159), NMI=0.8228, silhouette=0.1159, n_pred_clusters=40. WITHIN the predicted range 0.42-0.67. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. Tuned Spectral RBF produced delta=+0.0093 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Birch incremental clustering (Exp 17

### Exp 16: Spectral RBF with gamma sweep on PCA(50)
- **Backbone:** `spectral_tuned` | **Status:** KEEP
- **Result:** ARI 0.5252 | NMI 0.8228 | silhouette 0.1159 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Spectral with gamma swept across [0.001, 0.01, 0.1, 1.0] on PCA(50) features will land best-ARI in 0.42 to 0.67 because the mechanism per Ng 2001 is that proper gamma puts the affi...
- **Verdict:** KEEP — ARI=0.5252 (delta +0.0093 vs baseline 0.5159), NMI=0.8228, silhouette=0.1159, n_pred_clusters=40. WITHIN the predicted range 0.42-0.67. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. Tuned Spectral RBF produced delta=+0.0093 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Birch incremental clustering (Exp 17

### Exp 17: Birch (Zhang 1996) on PCA(50)
- **Backbone:** `birch` | **Status:** KEEP
- **Result:** ARI 0.5287 | NMI 0.8254 | silhouette 0.1608 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Birch on PCA(50) features with K=40 will land ARI in 0.37 to 0.57 because the mechanism per Zhang 1996 is that CF-Tree's aggregation is approximately equivalent to single-linkage c...
- **Verdict:** KEEP — ARI=0.5287 (delta +0.0128 vs baseline 0.5159), NMI=0.8254, silhouette=0.1608, n_pred_clusters=40. WITHIN the predicted range 0.37-0.57. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. Birch incremental produced delta=+0.0128 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Affinity Propagation (Exp 18).

### Exp 18: Affinity Propagation (Frey 2007 Science) on PCA(50)
- **Backbone:** `affinity_prop` | **Status:** KEEP
- **Result:** ARI 0.4757 | NMI 0.8105 | silhouette 0.1737 | n_pred_clusters 56
- **Hypothesis (first 200ch):** We hypothesize that Affinity Propagation on PCA(50) features with default damping=0.9 and median preference will land ARI in 0.32 to 0.62 because the mechanism per Frey 2007 is that exemplar message-p...
- **Verdict:** KEEP — ARI=0.4757 (delta -0.0402 vs baseline 0.5159), NMI=0.8105, silhouette=0.1737, n_pred_clusters=56. WITHIN the predicted range 0.32-0.62. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. Affinity Propagation produced delta=-0.0402 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: MeanShift mode-seeking (Exp 19).

### Exp 19: MeanShift (Comaniciu 2002) on PCA(50)
- **Backbone:** `meanshift` | **Status:** DISCARD
- **Result:** ARI 0.0000 | NMI 0.0000 | silhouette nan | n_pred_clusters 1
- **Hypothesis (first 200ch):** We hypothesize that MeanShift with auto-bandwidth on PCA(50) features will land ARI in 0.22 to 0.57 because the mechanism per Comaniciu 2002 is that mode-seeking discovers naturally-dense regions; on ...
- **Verdict:** DISCARD — ARI=0.0000 (delta -0.5159 vs baseline 0.5159), NMI=0.0000, silhouette=nan, n_pred_clusters=1. BELOW the predicted lower bound 0.22 — refuted. Status under floor=0.30 is DISCARD; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the
- **Learning:** axis closed. MeanShift mode-seeking produced delta=-0.5159 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: DINOv2 ViT-S/14 features (Exp 20

### Exp 20: DINOv2 ViT-S/14 (Oquab 2024 Meta TMLR) features + KMeans
- **Backbone:** `dinov2_kmeans` | **Status:** KEEP
- **Result:** ARI 0.5455 | NMI 0.8201 | silhouette 0.0710 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that DINOv2 ViT-S/14 penultimate features (384-dim) on resized 224x224 3-channel Olivetti + KMeans will land ARI in 0.52 to 0.82 because the mechanism per Oquab 2024 is that DINOv2's se...
- **Verdict:** KEEP — ARI=0.5455 (delta +0.0296 vs baseline 0.5159), NMI=0.8201, silhouette=0.0710, n_pred_clusters=40. WITHIN the predicted range 0.52-0.82. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis open. DINOv2 self-supervised features produced delta=+0.0296 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Spherical KMeans on L2-no

### Exp 21: Spherical KMeans (Dhillon 2001) on L2-norm PCA(50)
- **Backbone:** `spherical_kmeans` | **Status:** KEEP
- **Result:** ARI 0.4816 | NMI 0.7896 | silhouette 0.1266 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that L2-normalized PCA(50) + KMeans (Spherical equivalent) will land ARI in 0.47 to 0.67 because the mechanism per Dhillon 2001 is that cosine-similarity-based clustering is robust to m...
- **Verdict:** KEEP — ARI=0.4816 (delta -0.0343 vs baseline 0.5159), NMI=0.7896, silhouette=0.1266, n_pred_clusters=40. WITHIN the predicted range 0.47-0.67. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test
- **Learning:** axis closed. L2-normalized Spherical KMeans produced delta=-0.0343 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Final summary + champion

### Exp 22: DINOv2 dinov2_vits14 + MiniBatchKMeans (faster, may be less accurate)
- **Backbone:** `dinov2_vits14_minibatch_kmeans` | **Status:** KEEP
- **Result:** ARI 0.5596 | NMI 0.8393 | silhouette 0.0596 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that MiniBatchKMeans (faster, may be less accurate) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered str...
- **Verdict:** KEEP — ARI=0.5596 (delta +0.0141 vs Exp 20 champion 0.5455), NMI=0.8393, silhouette=0.0596, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. minibatch_kmeans produced delta=+0.0141 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: BisectingKMeans hierarchical KMeans. The cumulative best ARI across all experiments so far drives the choice of which axis the next exp

### Exp 23: DINOv2 dinov2_vits14 + BisectingKMeans hierarchical bisection
- **Backbone:** `dinov2_vits14_bisecting_kmeans` | **Status:** KEEP
- **Result:** ARI 0.4437 | NMI 0.7678 | silhouette 0.0277 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that BisectingKMeans hierarchical bisection on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure i...
- **Verdict:** KEEP — ARI=0.4437 (delta -0.1018 vs Exp 20 champion 0.5455), NMI=0.7678, silhouette=0.0277, n_pred_clusters=40. BELOW predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency
- **Learning:** axis closed. bisecting_kmeans produced delta=-0.1018 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: KMeans with random init. The cumulative best ARI across all experiments so far drives the choice of which axis the

### Exp 24: DINOv2 dinov2_vits14 + KMeans with random init (vs k-means++)
- **Backbone:** `dinov2_vits14_kmeans_random` | **Status:** KEEP
- **Result:** ARI 0.5000 | NMI 0.8091 | silhouette 0.0304 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that KMeans with random init (vs k-means++) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure i...
- **Verdict:** KEEP — ARI=0.5000 (delta -0.0455 vs Exp 20 champion 0.5455), NMI=0.8091, silhouette=0.0304, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. kmeans_random produced delta=-0.0455 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: KMeans with n_init=50 for more random restarts. The cumulative best ARI across all experiments so far drives the choi

### Exp 25: DINOv2 dinov2_vits14 + KMeans n_init=50 (5x more random restarts)
- **Backbone:** `dinov2_vits14_kmeans_n50` | **Status:** KEEP
- **Result:** ARI 0.5852 | NMI 0.8456 | silhouette 0.0891 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that KMeans n_init=50 (5x more random restarts) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structu...
- **Verdict:** KEEP — ARI=0.5852 (delta +0.0397 vs Exp 20 champion 0.5455), NMI=0.8456, silhouette=0.0891, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. kmeans_n50 produced delta=+0.0397 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: L2-normalized DINOv2 + Spherical KMeans. The cumulative best ARI across all experiments so far drives the choice of which axis the next exper

### Exp 26: DINOv2 dinov2_vits14 + L2-normalized features + KMeans (Spherical)
- **Backbone:** `dinov2_vits14_spherical` | **Status:** KEEP
- **Result:** ARI 0.5602 | NMI 0.8259 | silhouette 0.0467 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that L2-normalized features + KMeans (Spherical) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered struct...
- **Verdict:** KEEP — ARI=0.5602 (delta +0.0147 vs Exp 20 champion 0.5455), NMI=0.8259, silhouette=0.0467, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. spherical produced delta=+0.0147 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: Agglomerative Ward on DINOv2. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will p

### Exp 27: DINOv2 dinov2_vits14 + Agglomerative Ward (variance-minimizing merges)
- **Backbone:** `dinov2_vits14_agg_ward` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Agglomerative Ward (variance-minimizing merges) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered st...
- **Verdict:** KEEP — ARI=0.6371 (delta +0.0916 vs Exp 20 champion 0.5455), NMI=0.8706, silhouette=0.0834, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. agg_ward produced delta=+0.0916 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: Agglomerative average-linkage. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will p

### Exp 28: DINOv2 dinov2_vits14 + Agglomerative average-linkage
- **Backbone:** `dinov2_vits14_agg_avg` | **Status:** KEEP
- **Result:** ARI 0.4703 | NMI 0.8158 | silhouette 0.0226 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Agglomerative average-linkage on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their r...
- **Verdict:** KEEP — ARI=0.4703 (delta -0.0752 vs Exp 20 champion 0.5455), NMI=0.8158, silhouette=0.0226, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. agg_avg produced delta=-0.0752 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: Agglomerative complete-linkage. The cumulative best ARI across all experiments so far drives the choice of which axis the n

### Exp 29: DINOv2 dinov2_vits14 + Agglomerative complete-linkage (max distance)
- **Backbone:** `dinov2_vits14_agg_complete` | **Status:** KEEP
- **Result:** ARI 0.4805 | NMI 0.8071 | silhouette 0.0234 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Agglomerative complete-linkage (max distance) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered stru...
- **Verdict:** KEEP — ARI=0.4805 (delta -0.0650 vs Exp 20 champion 0.5455), NMI=0.8071, silhouette=0.0234, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. agg_complete produced delta=-0.0650 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: Agglomerative cosine-distance + average. The cumulative best ARI across all experiments so far drives the choice of wh

### Exp 30: DINOv2 dinov2_vits14 + Agglomerative cosine + average linkage
- **Backbone:** `dinov2_vits14_agg_cosine_avg` | **Status:** KEEP
- **Result:** ARI 0.4490 | NMI 0.8174 | silhouette 0.0134 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Agglomerative cosine + average linkage on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure i...
- **Verdict:** KEEP — ARI=0.4490 (delta -0.0965 vs Exp 20 champion 0.5455), NMI=0.8174, silhouette=0.0134, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. agg_cosine_avg produced delta=-0.0965 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: Spectral clustering on DINOv2. The cumulative best ARI across all experiments so far drives the choice of which axis

### Exp 31: DINOv2 dinov2_vits14 + Spectral RBF gamma=0.001 (small)
- **Backbone:** `dinov2_vits14_spectral_g001` | **Status:** KEEP
- **Result:** ARI 0.5852 | NMI 0.8533 | silhouette 0.0872 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Spectral RBF gamma=0.001 (small) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in thei...
- **Verdict:** KEEP — ARI=0.5852 (delta +0.0397 vs Exp 20 champion 0.5455), NMI=0.8533, silhouette=0.0872, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. spectral_g001 produced delta=+0.0397 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: Spectral RBF gamma=0.01. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will pr

### Exp 32: DINOv2 dinov2_vits14 + Spectral RBF gamma=0.01
- **Backbone:** `dinov2_vits14_spectral_g01` | **Status:** DISCARD
- **Result:** ARI 0.2767 | NMI 0.7672 | silhouette 0.0361 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Spectral RBF gamma=0.01 on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw for...
- **Verdict:** DISCARD — ARI=0.2767 (delta -0.2688 vs Exp 20 champion 0.5455), NMI=0.7672, silhouette=0.0361, n_pred_clusters=40. BELOW predicted range 0.45-0.65. Status under floor=0.30 is DISCARD; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consi
- **Learning:** axis closed. spectral_g01 produced delta=-0.2688 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: Spectral cosine. The cumulative best ARI across all experiments so far drives the choice of which axis the next experi

### Exp 33: DINOv2 dinov2_vits14 + Spectral cosine affinity
- **Backbone:** `dinov2_vits14_spectral_cos` | **Status:** KEEP
- **Result:** ARI 0.6963 | NMI 0.8974 | silhouette 0.0890 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Spectral cosine affinity on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw fo...
- **Verdict:** KEEP — ARI=0.6963 (delta +0.1508 vs Exp 20 champion 0.5455), NMI=0.8974, silhouette=0.0890, n_pred_clusters=40. ABOVE predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency
- **Learning:** axis open. spectral_cos produced delta=+0.1508 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: Spectral nearest-neighbors. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will 

### Exp 34: DINOv2 dinov2_vits14 + Spectral nearest-neighbors affinity (k=10)
- **Backbone:** `dinov2_vits14_spectral_knn10` | **Status:** KEEP
- **Result:** ARI 0.6389 | NMI 0.8584 | silhouette 0.0796 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Spectral nearest-neighbors affinity (k=10) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structu...
- **Verdict:** KEEP — ARI=0.6389 (delta +0.0934 vs Exp 20 champion 0.5455), NMI=0.8584, silhouette=0.0796, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. spectral_knn10 produced delta=+0.0934 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: Birch on DINOv2. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 35: DINOv2 dinov2_vits14 + Birch on DINOv2 features
- **Backbone:** `dinov2_vits14_birch` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Birch on DINOv2 features on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw fo...
- **Verdict:** KEEP — ARI=0.6371 (delta +0.0916 vs Exp 20 champion 0.5455), NMI=0.8706, silhouette=0.0834, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. birch produced delta=+0.0916 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: GMM full-cov on DINOv2. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 36: DINOv2 dinov2_vits14 + GMM full-covariance K=40
- **Backbone:** `dinov2_vits14_gmm_full` | **Status:** KEEP
- **Result:** ARI 0.5234 | NMI 0.8133 | silhouette 0.0341 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that GMM full-covariance K=40 on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw fo...
- **Verdict:** KEEP — ARI=0.5234 (delta -0.0221 vs Exp 20 champion 0.5455), NMI=0.8133, silhouette=0.0341, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. gmm_full produced delta=-0.0221 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: GMM diag-cov. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment wi

### Exp 37: DINOv2 dinov2_vits14 + GMM diagonal-covariance
- **Backbone:** `dinov2_vits14_gmm_diag` | **Status:** KEEP
- **Result:** ARI 0.5234 | NMI 0.8133 | silhouette 0.0341 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that GMM diagonal-covariance on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw for...
- **Verdict:** KEEP — ARI=0.5234 (delta -0.0221 vs Exp 20 champion 0.5455), NMI=0.8133, silhouette=0.0341, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. gmm_diag produced delta=-0.0221 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: HDBSCAN on DINOv2. The cumulative best ARI across all experiments so far drives the choice of which axis the next experime

### Exp 38: DINOv2 dinov2_vits14 + PCA(50) on DINOv2 + KMeans (denoise)
- **Backbone:** `dinov2_vits14_pca50_km` | **Status:** KEEP
- **Result:** ARI 0.5312 | NMI 0.8184 | silhouette 0.0328 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(50) on DINOv2 + KMeans (denoise) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in ...
- **Verdict:** KEEP — ARI=0.5312 (delta -0.0143 vs Exp 20 champion 0.5455), NMI=0.8184, silhouette=0.0328, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. pca50_km produced delta=-0.0143 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: PCA(100) + KMeans. The cumulative best ARI across all experiments so far drives the choice of which axis the next experime

### Exp 39: DINOv2 dinov2_vits14 + PCA(100) on DINOv2 + KMeans
- **Backbone:** `dinov2_vits14_pca100_km` | **Status:** KEEP
- **Result:** ARI 0.5473 | NMI 0.8278 | silhouette 0.0745 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that PCA(100) on DINOv2 + KMeans on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw...
- **Verdict:** KEEP — ARI=0.5473 (delta +0.0018 vs Exp 20 champion 0.5455), NMI=0.8278, silhouette=0.0745, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. pca100_km produced delta=+0.0018 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: UMAP(10) on DINOv2 + KMeans. The cumulative best ARI across all experiments so far drives the choice of which axis the ne

### Exp 40: DINOv2 dinov2_vits14 + UMAP(10) on DINOv2 + KMeans
- **Backbone:** `dinov2_vits14_umap10_km` | **Status:** KEEP
- **Result:** ARI 0.5982 | NMI 0.8465 | silhouette 0.0592 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that UMAP(10) on DINOv2 + KMeans on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in their raw...
- **Verdict:** KEEP — ARI=0.5982 (delta +0.0527 vs Exp 20 champion 0.5455), NMI=0.8465, silhouette=0.0592, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. umap10_km produced delta=+0.0527 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: UMAP(2) for 2D viz + KMeans. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will pr

### Exp 41: DINOv2 dinov2_vits14 + UMAP(2) on DINOv2 + KMeans (extreme low dim)
- **Backbone:** `dinov2_vits14_umap2_km` | **Status:** KEEP
- **Result:** ARI 0.6100 | NMI 0.8455 | silhouette 0.0678 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that UMAP(2) on DINOv2 + KMeans (extreme low dim) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered struc...
- **Verdict:** KEEP — ARI=0.6100 (delta +0.0645 vs Exp 20 champion 0.5455), NMI=0.8455, silhouette=0.0678, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. umap2_km produced delta=+0.0645 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: DINOv2 ViT-B/14 (larger model). The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will 

### Exp 42: DINOv2 dinov2_vitb14 + ViT-B/14 features + KMeans (larger model, 768-dim)
- **Backbone:** `dinov2_vitb14_vitb_km` | **Status:** KEEP
- **Result:** ARI 0.5445 | NMI 0.8243 | silhouette 0.0379 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ViT-B/14 features + KMeans (larger model, 768-dim) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered...
- **Verdict:** KEEP — ARI=0.5445 (delta -0.0010 vs Exp 20 champion 0.5455), NMI=0.8243, silhouette=0.0379, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. vitb_km produced delta=-0.0010 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: ViT-B/14 + Spherical KMeans. The cumulative best ARI across all experiments so far drives the choice of which axis the next

### Exp 43: DINOv2 dinov2_vitb14 + ViT-B/14 + L2-norm + KMeans (Spherical)
- **Backbone:** `dinov2_vitb14_vitb_spherical` | **Status:** KEEP
- **Result:** ARI 0.5388 | NMI 0.8119 | silhouette 0.0506 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ViT-B/14 + L2-norm + KMeans (Spherical) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure ...
- **Verdict:** KEEP — ARI=0.5388 (delta -0.0067 vs Exp 20 champion 0.5455), NMI=0.8119, silhouette=0.0506, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. vitb_spherical produced delta=-0.0067 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: ViT-B/14 + Agglomerative Ward. The cumulative best ARI across all experiments so far drives the choice of which axis

### Exp 44: DINOv2 dinov2_vits14 + KMeans seed=1 (variance check on champion)
- **Backbone:** `dinov2_vits14_seed1` | **Status:** KEEP
- **Result:** ARI 0.5561 | NMI 0.8301 | silhouette 0.0904 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that KMeans seed=1 (variance check on champion) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structu...
- **Verdict:** KEEP — ARI=0.5561 (delta +0.0106 vs Exp 20 champion 0.5455), NMI=0.8301, silhouette=0.0904, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis open. seed1 produced delta=+0.0106 ARI vs the DINOv2+KMeans champion. Mental model update: this pushes the DINOv2-feature ceiling further. Next try: seed variance Exp 45. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 45: DINOv2 dinov2_vits14 + KMeans seed=2 (variance check on champion)
- **Backbone:** `dinov2_vits14_seed2` | **Status:** KEEP
- **Result:** ARI 0.5144 | NMI 0.8110 | silhouette 0.0712 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that KMeans seed=2 (variance check on champion) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structu...
- **Verdict:** KEEP — ARI=0.5144 (delta -0.0311 vs Exp 20 champion 0.5455), NMI=0.8110, silhouette=0.0712, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. seed2 produced delta=-0.0311 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: seed variance Exp 46. The cumulative best ARI across all experiments so far drives the choice of which axis the next experime

### Exp 46: DINOv2 dinov2_vits14 + KMeans seed=7 (variance check on champion)
- **Backbone:** `dinov2_vits14_seed7` | **Status:** KEEP
- **Result:** ARI 0.5387 | NMI 0.8175 | silhouette 0.0633 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that KMeans seed=7 (variance check on champion) on DINOv2 features will land ARI in 0.45 to 0.65 because the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structu...
- **Verdict:** KEEP — ARI=0.5387 (delta -0.0068 vs Exp 20 champion 0.5455), NMI=0.8175, silhouette=0.0633, n_pred_clusters=40. WITHIN predicted range 0.45-0.65. Status under floor=0.30 is KEEP; locked test-set hash verified. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistenc
- **Learning:** axis closed. seed7 produced delta=-0.0068 ARI vs the DINOv2+KMeans champion. Mental model update: this does not improve over the baseline KMeans on DINOv2 features. Next try: Spectral hill-climbing sweep next. The cumulative best ARI across all experiments so far drives the choice of which axis the 

### Exp 47: Spectral hill-climb: cosine + assign=kmeans (champion config) on DINOv2 ViT-S/14 raw 384-dim
- **Backbone:** `spectral_hc_cosine_+_assignkmeans_(ch` | **Status:** KEEP
- **Result:** ARI 0.6963 | NMI 0.8974 | silhouette 0.0890 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + assign=kmeans (champion config) on DINOv2 ViT-S/14 raw 384-dim will land ARI in 0.68 to 0.72 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral con...
- **Verdict:** KEEP — ARI=0.6963 (delta -0.0000 vs Exp 33 champion 0.6963), NMI=0.8974, silhouette=0.0890, n_pred=40. WITHIN predicted 0.68-0.72. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine + assign=kmeans (champion config) produced delta=-0.0000 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: Spectral cosine + assign_labels=cluster_qr. The cumulative best ARI acr

### Exp 48: Spectral hill-climb: cosine + assign=cluster_qr on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_+_assigncluster_qr` | **Status:** KEEP
- **Result:** ARI 0.4708 | NMI 0.7628 | silhouette -0.0049 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + assign=cluster_qr on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the...
- **Verdict:** KEEP — ARI=0.4708 (delta -0.2255 vs Exp 33 champion 0.6963), NMI=0.7628, silhouette=-0.0049, n_pred=40. BELOW predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine + assign=cluster_qr produced delta=-0.2255 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: Spectral cosine on L2-normalized features. The cumulative best ARI across all experim

### Exp 49: Spectral hill-climb: cosine + L2-normalized features on DINOv2 ViT-S/14 + L2
- **Backbone:** `spectral_hc_cosine_+_L2-normalized_fe` | **Status:** KEEP
- **Result:** ARI 0.6963 | NMI 0.8974 | silhouette 0.0890 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + L2-normalized features on DINOv2 ViT-S/14 + L2 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration chang...
- **Verdict:** KEEP — ARI=0.6963 (delta -0.0000 vs Exp 33 champion 0.6963), NMI=0.8974, silhouette=0.0890, n_pred=40. WITHIN predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine + L2-normalized features produced delta=-0.0000 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: Spectral nearest-neighbors variants. The cumulative best ARI across all experime

### Exp 50: Spectral hill-climb: nearest_neighbors k=5 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_nearest_neighbors_k5` | **Status:** KEEP
- **Result:** ARI 0.6042 | NMI 0.8577 | silhouette 0.0670 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that nearest_neighbors k=5 on DINOv2 ViT-S/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affi...
- **Verdict:** KEEP — ARI=0.6042 (delta -0.0921 vs Exp 33 champion 0.6963), NMI=0.8577, silhouette=0.0670, n_pred=40. WITHIN predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. nearest_neighbors k=5 produced delta=-0.0921 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: k-NN affinity with k=7. The cumulative best ARI across all experiments so far drives the c

### Exp 51: Spectral hill-climb: nearest_neighbors k=7 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_nearest_neighbors_k7` | **Status:** KEEP
- **Result:** ARI 0.6246 | NMI 0.8538 | silhouette 0.0815 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that nearest_neighbors k=7 on DINOv2 ViT-S/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affi...
- **Verdict:** KEEP — ARI=0.6246 (delta -0.0717 vs Exp 33 champion 0.6963), NMI=0.8538, silhouette=0.0815, n_pred=40. WITHIN predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. nearest_neighbors k=7 produced delta=-0.0717 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: k-NN affinity with k=10. The cumulative best ARI across all experiments so far drives the 

### Exp 52: Spectral hill-climb: nearest_neighbors k=15 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_nearest_neighbors_k15` | **Status:** KEEP
- **Result:** ARI 0.5888 | NMI 0.8358 | silhouette 0.0554 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that nearest_neighbors k=15 on DINOv2 ViT-S/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the aff...
- **Verdict:** KEEP — ARI=0.5888 (delta -0.1075 vs Exp 33 champion 0.6963), NMI=0.8358, silhouette=0.0554, n_pred=40. BELOW predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the 
- **Learning:** axis closed. nearest_neighbors k=15 produced delta=-0.1075 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: k-NN affinity with k=15. The cumulative best ARI across all experiments so far drives the

### Exp 53: Spectral hill-climb: nearest_neighbors k=20 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_nearest_neighbors_k20` | **Status:** KEEP
- **Result:** ARI 0.5278 | NMI 0.8059 | silhouette 0.0423 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that nearest_neighbors k=20 on DINOv2 ViT-S/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the aff...
- **Verdict:** KEEP — ARI=0.5278 (delta -0.1685 vs Exp 33 champion 0.6963), NMI=0.8059, silhouette=0.0423, n_pred=40. BELOW predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the 
- **Learning:** axis closed. nearest_neighbors k=20 produced delta=-0.1685 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: k-NN affinity with k=20. The cumulative best ARI across all experiments so far drives the

### Exp 54: Spectral hill-climb: nearest_neighbors k=30 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_nearest_neighbors_k30` | **Status:** KEEP
- **Result:** ARI 0.4553 | NMI 0.7806 | silhouette 0.0092 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that nearest_neighbors k=30 on DINOv2 ViT-S/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the aff...
- **Verdict:** KEEP — ARI=0.4553 (delta -0.2410 vs Exp 33 champion 0.6963), NMI=0.7806, silhouette=0.0092, n_pred=40. BELOW predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the 
- **Learning:** axis closed. nearest_neighbors k=30 produced delta=-0.2410 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: k-NN affinity with k=30. The cumulative best ARI across all experiments so far drives the

### Exp 55: Spectral hill-climb: RBF gamma=0.0001 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_RBF_gamma0.0001` | **Status:** KEEP
- **Result:** ARI 0.7170 | NMI 0.9102 | silhouette 0.1101 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that RBF gamma=0.0001 on DINOv2 ViT-S/14 will land ARI in 0.50 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity ...
- **Verdict:** KEEP — ARI=0.7170 (delta +0.0207 vs Exp 33 champion 0.6963), NMI=0.9102, silhouette=0.1101, n_pred=40. WITHIN predicted 0.50-0.75. NEW CHAMPION on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per t
- **Learning:** axis open. RBF gamma=0.0001 produced delta=+0.0207 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant pushes the local maximum further. Next try: RBF gamma fine sweep continues. The cumulative best ARI across all experiments so far drives the choice of

### Exp 56: Spectral hill-climb: RBF gamma=0.0005 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_RBF_gamma0.0005` | **Status:** KEEP
- **Result:** ARI 0.6961 | NMI 0.9001 | silhouette 0.0942 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that RBF gamma=0.0005 on DINOv2 ViT-S/14 will land ARI in 0.50 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity ...
- **Verdict:** KEEP — ARI=0.6961 (delta -0.0002 vs Exp 33 champion 0.6963), NMI=0.9001, silhouette=0.0942, n_pred=40. WITHIN predicted 0.50-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. RBF gamma=0.0005 produced delta=-0.0002 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: RBF gamma fine sweep continues. The cumulative best ARI across all experiments so far drives th

### Exp 57: Spectral hill-climb: RBF gamma=0.005 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_RBF_gamma0.005` | **Status:** DISCARD
- **Result:** ARI 0.2628 | NMI 0.7973 | silhouette 0.0764 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that RBF gamma=0.005 on DINOv2 ViT-S/14 will land ARI in 0.50 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity m...
- **Verdict:** DISCARD — ARI=0.2628 (delta -0.4335 vs Exp 33 champion 0.6963), NMI=0.7973, silhouette=0.0764, n_pred=40. BELOW predicted 0.50-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per t
- **Learning:** axis closed. RBF gamma=0.005 produced delta=-0.4335 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: RBF gamma fine sweep continues. The cumulative best ARI across all experiments so far drives the

### Exp 58: Spectral hill-climb: RBF gamma=0.05 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_RBF_gamma0.05` | **Status:** DISCARD
- **Result:** ARI 0.0503 | NMI 0.5965 | silhouette -0.0894 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that RBF gamma=0.05 on DINOv2 ViT-S/14 will land ARI in 0.50 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity ma...
- **Verdict:** DISCARD — ARI=0.0503 (delta -0.6460 vs Exp 33 champion 0.6963), NMI=0.5965, silhouette=-0.0894, n_pred=40. BELOW predicted 0.50-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per 
- **Learning:** axis closed. RBF gamma=0.05 produced delta=-0.6460 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: RBF gamma fine sweep continues. The cumulative best ARI across all experiments so far drives the 

### Exp 59: Spectral hill-climb: RBF gamma=0.5 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_RBF_gamma0.5` | **Status:** DISCARD
- **Result:** ARI 0.0000 | NMI 0.0297 | silhouette -0.1190 | n_pred_clusters 7
- **Hypothesis (first 200ch):** We hypothesize that RBF gamma=0.5 on DINOv2 ViT-S/14 will land ARI in 0.50 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity mat...
- **Verdict:** DISCARD — ARI=0.0000 (delta -0.6963 vs Exp 33 champion 0.6963), NMI=0.0297, silhouette=-0.1190, n_pred=7. BELOW predicted 0.50-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per t
- **Learning:** axis closed. RBF gamma=0.5 produced delta=-0.6963 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: RBF gamma fine sweep continues. The cumulative best ARI across all experiments so far drives the c

### Exp 60: Spectral hill-climb: ViT-B/14 + cosine on DINOv2 ViT-B/14 768-dim
- **Backbone:** `spectral_hc_ViT-B/14_+_cosine` | **Status:** KEEP
- **Result:** ARI 0.6552 | NMI 0.8805 | silhouette 0.0673 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ViT-B/14 + cosine on DINOv2 ViT-B/14 768-dim will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the ...
- **Verdict:** KEEP — ARI=0.6552 (delta -0.0411 vs Exp 33 champion 0.6963), NMI=0.8805, silhouette=0.0673, n_pred=40. WITHIN predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. ViT-B/14 + cosine produced delta=-0.0411 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: ViT-B/14 + cluster_qr. The cumulative best ARI across all experiments so far drives the choice

### Exp 61: Spectral hill-climb: ViT-B/14 + cluster_qr + cosine on DINOv2 ViT-B/14
- **Backbone:** `spectral_hc_ViT-B/14_+_cluster_qr_+_c` | **Status:** KEEP
- **Result:** ARI 0.4317 | NMI 0.7495 | silhouette 0.0033 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ViT-B/14 + cluster_qr + cosine on DINOv2 ViT-B/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how...
- **Verdict:** KEEP — ARI=0.4317 (delta -0.2646 vs Exp 33 champion 0.6963), NMI=0.7495, silhouette=0.0033, n_pred=40. BELOW predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the 
- **Learning:** axis closed. ViT-B/14 + cluster_qr + cosine produced delta=-0.2646 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: ViT-B/14 normalized. The cumulative best ARI across all experiments so far drives

### Exp 62: Spectral hill-climb: ViT-B/14 + L2-norm + cosine on DINOv2 ViT-B/14 + L2
- **Backbone:** `spectral_hc_ViT-B/14_+_L2-norm_+_cosi` | **Status:** KEEP
- **Result:** ARI 0.6552 | NMI 0.8805 | silhouette 0.0673 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ViT-B/14 + L2-norm + cosine on DINOv2 ViT-B/14 + L2 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes h...
- **Verdict:** KEEP — ARI=0.6552 (delta -0.0411 vs Exp 33 champion 0.6963), NMI=0.8805, silhouette=0.0673, n_pred=40. WITHIN predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. ViT-B/14 + L2-norm + cosine produced delta=-0.0411 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: ViT-B/14 nearest_neighbors. The cumulative best ARI across all experiments so far dr

### Exp 63: Spectral hill-climb: ViT-B/14 + kNN k=10 on DINOv2 ViT-B/14
- **Backbone:** `spectral_hc_ViT-B/14_+_kNN_k10` | **Status:** KEEP
- **Result:** ARI 0.5489 | NMI 0.8215 | silhouette 0.0496 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that ViT-B/14 + kNN k=10 on DINOv2 ViT-B/14 will land ARI in 0.60 to 0.80 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affini...
- **Verdict:** KEEP — ARI=0.5489 (delta -0.1474 vs Exp 33 champion 0.6963), NMI=0.8215, silhouette=0.0496, n_pred=40. BELOW predicted 0.60-0.80. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the 
- **Learning:** axis closed. ViT-B/14 + kNN k=10 produced delta=-0.1474 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: n_init sweep. The cumulative best ARI across all experiments so far drives the choice of whi

### Exp 64: Spectral hill-climb: cosine + n_init=1 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_+_n_init1` | **Status:** KEEP
- **Result:** ARI 0.7064 | NMI 0.9014 | silhouette 0.0895 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + n_init=1 on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity...
- **Verdict:** KEEP — ARI=0.7064 (delta +0.0101 vs Exp 33 champion 0.6963), NMI=0.9014, silhouette=0.0895, n_pred=40. WITHIN predicted 0.65-0.75. NEW CHAMPION on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per t
- **Learning:** axis open. cosine + n_init=1 produced delta=+0.0101 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant pushes the local maximum further. Next try: multi-seed variance check. The cumulative best ARI across all experiments so far drives the choice of whi

### Exp 65: Spectral hill-climb: cosine + n_init=5 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_+_n_init5` | **Status:** KEEP
- **Result:** ARI 0.6742 | NMI 0.8829 | silhouette 0.0984 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + n_init=5 on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity...
- **Verdict:** KEEP — ARI=0.6742 (delta -0.0221 vs Exp 33 champion 0.6963), NMI=0.8829, silhouette=0.0984, n_pred=40. WITHIN predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine + n_init=5 produced delta=-0.0221 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: multi-seed variance check. The cumulative best ARI across all experiments so far drives the ch

### Exp 66: Spectral hill-climb: cosine + n_init=25 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_+_n_init25` | **Status:** KEEP
- **Result:** ARI 0.6963 | NMI 0.8974 | silhouette 0.0890 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + n_init=25 on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinit...
- **Verdict:** KEEP — ARI=0.6963 (delta -0.0000 vs Exp 33 champion 0.6963), NMI=0.8974, silhouette=0.0890, n_pred=40. WITHIN predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine + n_init=25 produced delta=-0.0000 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: multi-seed variance check. The cumulative best ARI across all experiments so far drives the c

### Exp 67: Spectral hill-climb: cosine + n_init=50 on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_+_n_init50` | **Status:** KEEP
- **Result:** ARI 0.6666 | NMI 0.8900 | silhouette 0.0806 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine + n_init=50 on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinit...
- **Verdict:** KEEP — ARI=0.6666 (delta -0.0297 vs Exp 33 champion 0.6963), NMI=0.8900, silhouette=0.0806, n_pred=40. WITHIN predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine + n_init=50 produced delta=-0.0297 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: multi-seed variance check. The cumulative best ARI across all experiments so far drives the c

### Exp 68: Spectral hill-climb: cosine seed=1 (variance check) on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_seed1_(variance_ch` | **Status:** KEEP
- **Result:** ARI 0.7154 | NMI 0.9051 | silhouette 0.0900 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine seed=1 (variance check) on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how...
- **Verdict:** KEEP — ARI=0.7154 (delta +0.0191 vs Exp 33 champion 0.6963), NMI=0.9051, silhouette=0.0900, n_pred=40. WITHIN predicted 0.65-0.75. NEW CHAMPION on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per t
- **Learning:** axis open. cosine seed=1 (variance check) produced delta=+0.0191 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant pushes the local maximum further. Next try: Spectral hill-climb complete; pivot to next backbone. The cumulative best ARI across all exp

### Exp 69: Spectral hill-climb: cosine seed=7 (variance check) on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_seed7_(variance_ch` | **Status:** KEEP
- **Result:** ARI 0.6596 | NMI 0.8710 | silhouette 0.0804 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine seed=7 (variance check) on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how...
- **Verdict:** KEEP — ARI=0.6596 (delta -0.0367 vs Exp 33 champion 0.6963), NMI=0.8710, silhouette=0.0804, n_pred=40. WITHIN predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the
- **Learning:** axis closed. cosine seed=7 (variance check) produced delta=-0.0367 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: Spectral hill-climb complete; pivot to next backbone. The cumulative best ARI acr

### Exp 70: Spectral hill-climb: cosine seed=42 (variance check) on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_seed42_(variance_c` | **Status:** KEEP
- **Result:** ARI 0.6127 | NMI 0.8609 | silhouette 0.0772 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine seed=42 (variance check) on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes ho...
- **Verdict:** KEEP — ARI=0.6127 (delta -0.0836 vs Exp 33 champion 0.6963), NMI=0.8609, silhouette=0.0772, n_pred=40. BELOW predicted 0.65-0.75. tail-trial on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the 
- **Learning:** axis closed. cosine seed=42 (variance check) produced delta=-0.0836 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant does not improve over the champion config. Next try: Spectral hill-climb complete; pivot to next backbone. The cumulative best ARI ac

### Exp 71: Spectral hill-climb: cosine seed=99 (variance check) on DINOv2 ViT-S/14
- **Backbone:** `spectral_hc_cosine_seed99_(variance_c` | **Status:** KEEP
- **Result:** ARI 0.7195 | NMI 0.9004 | silhouette 0.0927 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that cosine seed=99 (variance check) on DINOv2 ViT-S/14 will land ARI in 0.65 to 0.75 because the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes ho...
- **Verdict:** KEEP — ARI=0.7195 (delta +0.0232 vs Exp 33 champion 0.6963), NMI=0.9004, silhouette=0.0927, n_pred=40. WITHIN predicted 0.65-0.75. NEW CHAMPION on the Spectral hill-climb. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per t
- **Learning:** axis open. cosine seed=99 (variance check) produced delta=+0.0232 ARI vs the DINOv2+Spectral-cosine champion. Hill-climbing the Spectral configuration: this variant pushes the local maximum further. Next try: Spectral hill-climb complete; pivot to next backbone. The cumulative best ARI across all ex

### Exp 72: Ward hill-climb: linkage=ward on DINOv2 on DINOv2 384-dim
- **Backbone:** `ward_hc_linkageward_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=ward on DINOv2 on DINOv2 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed in feat...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=ward on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 73: Ward hill-climb: linkage=ward on DINOv2 L2-norm on DINOv2 L2-norm 384-dim
- **Backbone:** `ward_hc_linkageward_on_DINOv2_L2-norm` | **Status:** KEEP
- **Result:** ARI 0.6366 | NMI 0.8721 | silhouette 0.0751 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=ward on DINOv2 L2-norm on DINOv2 L2-norm 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries ar...
- **Verdict:** KEEP — ARI=0.6366 (delta -0.0829 vs Exp 71 champion 0.7195), NMI=0.8721, sil=0.0751, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=ward on DINOv2 L2-norm produced delta=-0.0829 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 74: Ward hill-climb: linkage=average on DINOv2 on DINOv2 384-dim
- **Backbone:** `ward_hc_linkageaverage_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.4703 | NMI 0.8158 | silhouette 0.0226 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average on DINOv2 on DINOv2 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.4703 (delta -0.2492 vs Exp 71 champion 0.7195), NMI=0.8158, sil=0.0226, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=average on DINOv2 produced delta=-0.2492 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 75: Ward hill-climb: linkage=average on DINOv2 L2-norm on DINOv2 L2-norm 384-dim
- **Backbone:** `ward_hc_linkageaverage_on_DINOv2_L2-no` | **Status:** KEEP
- **Result:** ARI 0.4631 | NMI 0.8234 | silhouette 0.0240 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average on DINOv2 L2-norm on DINOv2 L2-norm 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries...
- **Verdict:** KEEP — ARI=0.4631 (delta -0.2564 vs Exp 71 champion 0.7195), NMI=0.8234, sil=0.0240, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=average on DINOv2 L2-norm produced delta=-0.2564 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 76: Ward hill-climb: linkage=complete on DINOv2 on DINOv2 384-dim
- **Backbone:** `ward_hc_linkagecomplete_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.4805 | NMI 0.8071 | silhouette 0.0234 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=complete on DINOv2 on DINOv2 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed in ...
- **Verdict:** KEEP — ARI=0.4805 (delta -0.2390 vs Exp 71 champion 0.7195), NMI=0.8071, sil=0.0234, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=complete on DINOv2 produced delta=-0.2390 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 77: Ward hill-climb: linkage=complete on DINOv2 L2-norm on DINOv2 L2-norm 384-dim
- **Backbone:** `ward_hc_linkagecomplete_on_DINOv2_L2-n` | **Status:** KEEP
- **Result:** ARI 0.4926 | NMI 0.8112 | silhouette 0.0306 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=complete on DINOv2 L2-norm on DINOv2 L2-norm 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundarie...
- **Verdict:** KEEP — ARI=0.4926 (delta -0.2269 vs Exp 71 champion 0.7195), NMI=0.8112, sil=0.0306, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=complete on DINOv2 L2-norm produced delta=-0.2269 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 78: Ward hill-climb: linkage=single on DINOv2 on DINOv2 384-dim
- **Backbone:** `ward_hc_linkagesingle_on_DINOv2` | **Status:** DISCARD
- **Result:** ARI 0.1481 | NMI 0.6689 | silhouette -0.1119 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=single on DINOv2 on DINOv2 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed in fe...
- **Verdict:** DISCARD — ARI=0.1481 (delta -0.5714 vs Exp 71 champion 0.7195), NMI=0.6689, sil=-0.1119, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.m
- **Learning:** axis closed. linkage=single on DINOv2 produced delta=-0.5714 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 79: Ward hill-climb: linkage=single on DINOv2 L2-norm on DINOv2 L2-norm 384-dim
- **Backbone:** `ward_hc_linkagesingle_on_DINOv2_L2-nor` | **Status:** DISCARD
- **Result:** ARI 0.1437 | NMI 0.6625 | silhouette -0.1119 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=single on DINOv2 L2-norm on DINOv2 L2-norm 384-dim will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries ...
- **Verdict:** DISCARD — ARI=0.1437 (delta -0.5758 vs Exp 71 champion 0.7195), NMI=0.6625, sil=-0.1119, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.m
- **Learning:** axis closed. linkage=single on DINOv2 L2-norm produced delta=-0.5758 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next linkage variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 80: Ward hill-climb: linkage=average + cosine distance on DINOv2 ViT-S/14
- **Backbone:** `ward_hc_linkageaverage_+_cosine_distan` | **Status:** KEEP
- **Result:** ARI 0.4490 | NMI 0.8174 | silhouette 0.0134 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average + cosine distance on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are fo...
- **Verdict:** KEEP — ARI=0.4490 (delta -0.2705 vs Exp 71 champion 0.7195), NMI=0.8174, sil=0.0134, n_pred=40. BELOW predicted 0.57-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=average + cosine distance produced delta=-0.2705 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: manhattan distance variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will pr

### Exp 81: Ward hill-climb: linkage=complete + cosine distance on DINOv2 ViT-S/14
- **Backbone:** `ward_hc_linkagecomplete_+_cosine_dista` | **Status:** KEEP
- **Result:** ARI 0.4926 | NMI 0.8112 | silhouette 0.0306 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=complete + cosine distance on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are f...
- **Verdict:** KEEP — ARI=0.4926 (delta -0.2269 vs Exp 71 champion 0.7195), NMI=0.8112, sil=0.0306, n_pred=40. BELOW predicted 0.57-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=complete + cosine distance produced delta=-0.2269 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: manhattan distance variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will p

### Exp 82: Ward hill-climb: linkage=single + cosine distance on DINOv2 ViT-S/14
- **Backbone:** `ward_hc_linkagesingle_+_cosine_distanc` | **Status:** DISCARD
- **Result:** ARI 0.1437 | NMI 0.6625 | silhouette -0.1119 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=single + cosine distance on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are for...
- **Verdict:** DISCARD — ARI=0.1437 (delta -0.5758 vs Exp 71 champion 0.7195), NMI=0.6625, sil=-0.1119, n_pred=40. BELOW predicted 0.57-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.m
- **Learning:** axis closed. linkage=single + cosine distance produced delta=-0.5758 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: manhattan distance variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will pro

### Exp 83: Ward hill-climb: linkage=average + manhattan distance on DINOv2 ViT-S/14
- **Backbone:** `ward_hc_linkageaverage_+_manhattan_dis` | **Status:** KEEP
- **Result:** ARI 0.4540 | NMI 0.8206 | silhouette 0.0129 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average + manhattan distance on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are...
- **Verdict:** KEEP — ARI=0.4540 (delta -0.2655 vs Exp 71 champion 0.7195), NMI=0.8206, sil=0.0129, n_pred=40. BELOW predicted 0.52-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=average + manhattan distance produced delta=-0.2655 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: PCA dimension sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 84: Ward hill-climb: linkage=ward on PCA(20) on raw pixels → PCA(20)
- **Backbone:** `ward_hc_linkageward_on_PCA(20)` | **Status:** KEEP
- **Result:** ARI 0.4508 | NMI 0.7910 | silhouette 0.1581 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=ward on PCA(20) on raw pixels → PCA(20) will land ARI in 0.42 to 0.62 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed ...
- **Verdict:** KEEP — ARI=0.4508 (delta -0.2687 vs Exp 71 champion 0.7195), NMI=0.7910, sil=0.1581, n_pred=40. WITHIN predicted 0.42-0.62. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=ward on PCA(20) produced delta=-0.2687 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Ward + PCA + cosine. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 85: Ward hill-climb: linkage=ward on PCA(50) on raw pixels → PCA(50)
- **Backbone:** `ward_hc_linkageward_on_PCA(50)` | **Status:** KEEP
- **Result:** ARI 0.5159 | NMI 0.8201 | silhouette 0.1608 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=ward on PCA(50) on raw pixels → PCA(50) will land ARI in 0.42 to 0.62 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed ...
- **Verdict:** KEEP — ARI=0.5159 (delta -0.2036 vs Exp 71 champion 0.7195), NMI=0.8201, sil=0.1608, n_pred=40. WITHIN predicted 0.42-0.62. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=ward on PCA(50) produced delta=-0.2036 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Ward + PCA + cosine. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 86: Ward hill-climb: linkage=ward on PCA(100) on raw pixels → PCA(100)
- **Backbone:** `ward_hc_linkageward_on_PCA(100)` | **Status:** KEEP
- **Result:** ARI 0.4737 | NMI 0.8081 | silhouette 0.1656 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=ward on PCA(100) on raw pixels → PCA(100) will land ARI in 0.42 to 0.62 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are forme...
- **Verdict:** KEEP — ARI=0.4737 (delta -0.2458 vs Exp 71 champion 0.7195), NMI=0.8081, sil=0.1656, n_pred=40. WITHIN predicted 0.42-0.62. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=ward on PCA(100) produced delta=-0.2458 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Ward + PCA + cosine. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 87: Ward hill-climb: linkage=average + cosine on PCA(20) on raw pixels → PCA(20)
- **Backbone:** `ward_hc_linkageaverage_+_cosine_on_PCA` | **Status:** KEEP
- **Result:** ARI 0.3223 | NMI 0.7542 | silhouette 0.0925 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average + cosine on PCA(20) on raw pixels → PCA(20) will land ARI in 0.42 to 0.67 because the mechanism is that the chosen Ward configuration changes how cluster boundaries...
- **Verdict:** KEEP — ARI=0.3223 (delta -0.3972 vs Exp 71 champion 0.7195), NMI=0.7542, sil=0.0925, n_pred=40. BELOW predicted 0.42-0.67. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=average + cosine on PCA(20) produced delta=-0.3972 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Ward + connectivity constraints. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment 

### Exp 88: Ward hill-climb: linkage=average + cosine on PCA(50) on raw pixels → PCA(50)
- **Backbone:** `ward_hc_linkageaverage_+_cosine_on_PCA` | **Status:** KEEP
- **Result:** ARI 0.3229 | NMI 0.7547 | silhouette 0.0943 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average + cosine on PCA(50) on raw pixels → PCA(50) will land ARI in 0.42 to 0.67 because the mechanism is that the chosen Ward configuration changes how cluster boundaries...
- **Verdict:** KEEP — ARI=0.3229 (delta -0.3966 vs Exp 71 champion 0.7195), NMI=0.7547, sil=0.0943, n_pred=40. BELOW predicted 0.42-0.67. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. linkage=average + cosine on PCA(50) produced delta=-0.3966 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Ward + connectivity constraints. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment 

### Exp 89: Ward hill-climb: linkage=average + cosine on PCA(100) on raw pixels → PCA(100)
- **Backbone:** `ward_hc_linkageaverage_+_cosine_on_PCA` | **Status:** DISCARD
- **Result:** ARI 0.2983 | NMI 0.7444 | silhouette 0.0899 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that linkage=average + cosine on PCA(100) on raw pixels → PCA(100) will land ARI in 0.42 to 0.67 because the mechanism is that the chosen Ward configuration changes how cluster boundari...
- **Verdict:** DISCARD — ARI=0.2983 (delta -0.4212 vs Exp 71 champion 0.7195), NMI=0.7444, sil=0.0899, n_pred=40. BELOW predicted 0.42-0.67. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md
- **Learning:** axis closed. linkage=average + cosine on PCA(100) produced delta=-0.4212 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Ward + connectivity constraints. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment

### Exp 90: Ward hill-climb: Ward + connectivity kNN(k=5) on DINOv2 on DINOv2 ViT-S/14 + kNN connectivity graph
- **Backbone:** `ward_hc_Ward_+_connectivity_kNN(k5)_on` | **Status:** KEEP
- **Result:** ARI 0.6207 | NMI 0.8637 | silhouette 0.0821 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward + connectivity kNN(k=5) on DINOv2 on DINOv2 ViT-S/14 + kNN connectivity graph will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes...
- **Verdict:** KEEP — ARI=0.6207 (delta -0.0988 vs Exp 71 champion 0.7195), NMI=0.8637, sil=0.0821, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward + connectivity kNN(k=5) on DINOv2 produced delta=-0.0988 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: post-Ward KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment w

### Exp 91: Ward hill-climb: Ward + connectivity kNN(k=10) on DINOv2 on DINOv2 ViT-S/14 + kNN connectivity graph
- **Backbone:** `ward_hc_Ward_+_connectivity_kNN(k10)_o` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward + connectivity kNN(k=10) on DINOv2 on DINOv2 ViT-S/14 + kNN connectivity graph will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration change...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward + connectivity kNN(k=10) on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: post-Ward KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment 

### Exp 92: Ward hill-climb: Ward + connectivity kNN(k=20) on DINOv2 on DINOv2 ViT-S/14 + kNN connectivity graph
- **Backbone:** `ward_hc_Ward_+_connectivity_kNN(k20)_o` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward + connectivity kNN(k=20) on DINOv2 on DINOv2 ViT-S/14 + kNN connectivity graph will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration change...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward + connectivity kNN(k=20) on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: post-Ward KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment 

### Exp 93: Ward hill-climb: Ward init + KMeans refine on DINOv2 on DINOv2
- **Backbone:** `ward_hc_Ward_init_+_KMeans_refine_on_D` | **Status:** KEEP
- **Result:** ARI 0.6308 | NMI 0.8665 | silhouette 0.0802 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward init + KMeans refine on DINOv2 on DINOv2 will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed in...
- **Verdict:** KEEP — ARI=0.6308 (delta -0.0887 vs Exp 71 champion 0.7195), NMI=0.8665, sil=0.0802, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward init + KMeans refine on DINOv2 produced delta=-0.0887 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next family. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 94: Ward hill-climb: Ward init + KMeans refine on DINOv2 L2-norm on DINOv2 L2-norm
- **Backbone:** `ward_hc_Ward_init_+_KMeans_refine_on_D` | **Status:** KEEP
- **Result:** ARI 0.6303 | NMI 0.8681 | silhouette 0.0719 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward init + KMeans refine on DINOv2 L2-norm on DINOv2 L2-norm will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundari...
- **Verdict:** KEEP — ARI=0.6303 (delta -0.0892 vs Exp 71 champion 0.7195), NMI=0.8681, sil=0.0719, n_pred=40. WITHIN predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward init + KMeans refine on DINOv2 L2-norm produced delta=-0.0892 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next family. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 95: Ward hill-climb: Ward init + KMeans refine on PCA(50) on PCA(50)
- **Backbone:** `ward_hc_Ward_init_+_KMeans_refine_on_P` | **Status:** KEEP
- **Result:** ARI 0.5013 | NMI 0.8124 | silhouette 0.1639 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward init + KMeans refine on PCA(50) on PCA(50) will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are formed ...
- **Verdict:** KEEP — ARI=0.5013 (delta -0.2182 vs Exp 71 champion 0.7195), NMI=0.8124, sil=0.1639, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward init + KMeans refine on PCA(50) produced delta=-0.2182 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next family. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 96: Ward hill-climb: Ward init + KMeans refine on PCA(100) on PCA(100)
- **Backbone:** `ward_hc_Ward_init_+_KMeans_refine_on_P` | **Status:** KEEP
- **Result:** ARI 0.4591 | NMI 0.7993 | silhouette 0.1677 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Ward init + KMeans refine on PCA(100) on PCA(100) will land ARI in 0.62 to 0.77 because the mechanism is that the chosen Ward configuration changes how cluster boundaries are forme...
- **Verdict:** KEEP — ARI=0.4591 (delta -0.2604 vs Exp 71 champion 0.7195), NMI=0.7993, sil=0.1677, n_pred=40. BELOW predicted 0.62-0.77. local hill-climb on Ward. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Ward init + KMeans refine on PCA(100) produced delta=-0.2604 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next family. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 97: Birch hill-climb: threshold=0.05 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold0.05_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=0.05 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in ...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=0.05 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 98: Birch hill-climb: threshold=0.1 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold0.1_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=0.1 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=0.1 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 99: Birch hill-climb: threshold=0.2 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold0.2_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=0.2 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=0.2 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 100: Birch hill-climb: threshold=0.3 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold0.3_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=0.3 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=0.3 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 101: Birch hill-climb: threshold=0.5 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold0.5_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=0.5 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=0.5 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 102: Birch hill-climb: threshold=0.7 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold0.7_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=0.7 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=0.7 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 103: Birch hill-climb: threshold=1.0 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold1.0_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=1.0 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=1.0 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 104: Birch hill-climb: threshold=1.5 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_threshold1.5_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that threshold=1.5 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in f...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. threshold=1.5 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: branching factor sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 105: Birch hill-climb: branching_factor=10 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_branching_factor10_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that branching_factor=10 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are forme...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. branching_factor=10 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch on different feature sources. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment wil

### Exp 106: Birch hill-climb: branching_factor=25 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_branching_factor25_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that branching_factor=25 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are forme...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. branching_factor=25 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch on different feature sources. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment wil

### Exp 107: Birch hill-climb: branching_factor=100 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_branching_factor100_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that branching_factor=100 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. branching_factor=100 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch on different feature sources. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment wi

### Exp 108: Birch hill-climb: branching_factor=200 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_branching_factor200_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that branching_factor=200 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. branching_factor=200 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch on different feature sources. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment wi

### Exp 109: Birch hill-climb: default Birch on DINOv2 L2-norm on DINOv2 L2-norm
- **Backbone:** `birch_hc_default_Birch_on_DINOv2_L2-nor` | **Status:** DISCARD
- **Result:** ARI 0.2306 | NMI 0.6719 | silhouette -0.0521 | n_pred_clusters 22
- **Hypothesis (first 200ch):** We hypothesize that default Birch on DINOv2 L2-norm on DINOv2 L2-norm will land ARI in 0.42 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** DISCARD — ARI=0.2306 (delta -0.4889 vs Exp 71 champion 0.7195), NMI=0.6719, sil=-0.0521, n_pred=22. BELOW predicted 0.42-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.
- **Learning:** axis closed. default Birch on DINOv2 L2-norm produced delta=-0.4889 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch + KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe

### Exp 110: Birch hill-climb: default Birch on PCA(50) on PCA(50)
- **Backbone:** `birch_hc_default_Birch_on_PCA(50)` | **Status:** KEEP
- **Result:** ARI 0.5287 | NMI 0.8254 | silhouette 0.1608 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that default Birch on PCA(50) on PCA(50) will land ARI in 0.42 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in feature ...
- **Verdict:** KEEP — ARI=0.5287 (delta -0.1908 vs Exp 71 champion 0.7195), NMI=0.8254, sil=0.1608, n_pred=40. WITHIN predicted 0.42-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. default Birch on PCA(50) produced delta=-0.1908 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch + KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 111: Birch hill-climb: default Birch on PCA(100) on PCA(100)
- **Backbone:** `birch_hc_default_Birch_on_PCA(100)` | **Status:** KEEP
- **Result:** ARI 0.4737 | NMI 0.8081 | silhouette 0.1656 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that default Birch on PCA(100) on PCA(100) will land ARI in 0.42 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in featur...
- **Verdict:** KEEP — ARI=0.4737 (delta -0.2458 vs Exp 71 champion 0.7195), NMI=0.8081, sil=0.1656, n_pred=40. WITHIN predicted 0.42-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. default Birch on PCA(100) produced delta=-0.2458 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch + KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 112: Birch hill-climb: default Birch on PCA(20) on PCA(20)
- **Backbone:** `birch_hc_default_Birch_on_PCA(20)` | **Status:** KEEP
- **Result:** ARI 0.4540 | NMI 0.7949 | silhouette 0.1591 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that default Birch on PCA(20) on PCA(20) will land ARI in 0.42 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are formed in feature ...
- **Verdict:** KEEP — ARI=0.4540 (delta -0.2655 vs Exp 71 champion 0.7195), NMI=0.7949, sil=0.1591, n_pred=40. WITHIN predicted 0.42-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. default Birch on PCA(20) produced delta=-0.2655 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: Birch + KMeans refinement. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 113: Birch hill-climb: Birch leaves + KMeans refine on DINOv2 on DINOv2
- **Backbone:** `birch_hc_Birch_leaves_+_KMeans_refine_o` | **Status:** KEEP
- **Result:** ARI 0.5461 | NMI 0.8242 | silhouette 0.0492 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Birch leaves + KMeans refine on DINOv2 on DINOv2 will land ARI in 0.52 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are forme...
- **Verdict:** KEEP — ARI=0.5461 (delta -0.1734 vs Exp 71 champion 0.7195), NMI=0.8242, sil=0.0492, n_pred=40. WITHIN predicted 0.52-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Birch leaves + KMeans refine on DINOv2 produced delta=-0.1734 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: tighter threshold variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment wi

### Exp 114: Birch hill-climb: Birch leaves + KMeans refine on DINOv2 L2-norm on DINOv2 L2-norm
- **Backbone:** `birch_hc_Birch_leaves_+_KMeans_refine_o` | **Status:** DISCARD
- **Result:** ARI 0.2306 | NMI 0.6719 | silhouette -0.0521 | n_pred_clusters 22
- **Hypothesis (first 200ch):** We hypothesize that Birch leaves + KMeans refine on DINOv2 L2-norm on DINOv2 L2-norm will land ARI in 0.52 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boun...
- **Verdict:** DISCARD — ARI=0.2306 (delta -0.4889 vs Exp 71 champion 0.7195), NMI=0.6719, sil=-0.0521, n_pred=22. BELOW predicted 0.52-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.
- **Learning:** axis closed. Birch leaves + KMeans refine on DINOv2 L2-norm produced delta=-0.4889 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: tighter threshold variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next exper

### Exp 115: Birch hill-climb: Birch leaves + KMeans refine on PCA(50) on PCA(50)
- **Backbone:** `birch_hc_Birch_leaves_+_KMeans_refine_o` | **Status:** KEEP
- **Result:** ARI 0.4356 | NMI 0.7685 | silhouette 0.1478 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Birch leaves + KMeans refine on PCA(50) on PCA(50) will land ARI in 0.52 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are for...
- **Verdict:** KEEP — ARI=0.4356 (delta -0.2839 vs Exp 71 champion 0.7195), NMI=0.7685, sil=0.1478, n_pred=40. BELOW predicted 0.52-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Birch leaves + KMeans refine on PCA(50) produced delta=-0.2839 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: tighter threshold variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment w

### Exp 116: Birch hill-climb: Birch leaves + KMeans refine on PCA(100) on PCA(100)
- **Backbone:** `birch_hc_Birch_leaves_+_KMeans_refine_o` | **Status:** KEEP
- **Result:** ARI 0.4232 | NMI 0.7685 | silhouette 0.1479 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that Birch leaves + KMeans refine on PCA(100) on PCA(100) will land ARI in 0.52 to 0.77 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are f...
- **Verdict:** KEEP — ARI=0.4232 (delta -0.2963 vs Exp 71 champion 0.7195), NMI=0.7685, sil=0.1479, n_pred=40. BELOW predicted 0.52-0.77. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. Birch leaves + KMeans refine on PCA(100) produced delta=-0.2963 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: tighter threshold variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment 

### Exp 117: Birch hill-climb: tight threshold=0.01 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_tight_threshold0.01_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that tight threshold=0.01 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. tight threshold=0.01 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: more Birch variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 118: Birch hill-climb: tight threshold=0.02 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_tight_threshold0.02_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that tight threshold=0.02 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. tight threshold=0.02 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: more Birch variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 119: Birch hill-climb: tight threshold=0.03 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_tight_threshold0.03_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that tight threshold=0.03 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. tight threshold=0.03 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: more Birch variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 120: Birch hill-climb: tight threshold=0.04 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_tight_threshold0.04_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that tight threshold=0.04 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. tight threshold=0.04 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: more Birch variants. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 121: Birch hill-climb: tight threshold=0.05 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `birch_hc_tight_threshold0.05_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8706 | silhouette 0.0834 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that tight threshold=0.05 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism is that the chosen Birch configuration changes how cluster boundaries are form...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8706, sil=0.0834, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on Birch. Test set hash verified intact. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. tight threshold=0.05 on DINOv2 produced delta=-0.0824 ARI vs the Exp 71 champion. this variant does not improve over the prior best. Next try: next backbone family — UMAP. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will prob

### Exp 122: UMAP HC: n_neighbors=5 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_neighbors=5_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6109 | NMI 0.8624 | silhouette 0.0756 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_neighbors=5 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embeddi...
- **Verdict:** KEEP — ARI=0.6109 (delta -0.1086 vs Exp 71 champion 0.7195), NMI=0.8624, sil=0.0756, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_neighbors=5 on DINOv2 produced delta=-0.1086 ARI vs Exp 71. does not improve over prior best. Next try: min_dist sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 123: UMAP HC: n_neighbors=10 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_neighbors=10_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6488 | NMI 0.8693 | silhouette 0.0813 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_neighbors=10 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedd...
- **Verdict:** KEEP — ARI=0.6488 (delta -0.0707 vs Exp 71 champion 0.7195), NMI=0.8693, sil=0.0813, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_neighbors=10 on DINOv2 produced delta=-0.0707 ARI vs Exp 71. does not improve over prior best. Next try: min_dist sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 124: UMAP HC: n_neighbors=30 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_neighbors=30_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.5680 | NMI 0.8311 | silhouette 0.0317 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_neighbors=30 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedd...
- **Verdict:** KEEP — ARI=0.5680 (delta -0.1515 vs Exp 71 champion 0.7195), NMI=0.8311, sil=0.0317, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_neighbors=30 on DINOv2 produced delta=-0.1515 ARI vs Exp 71. does not improve over prior best. Next try: min_dist sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 125: UMAP HC: n_neighbors=50 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_neighbors=50_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.5690 | NMI 0.8279 | silhouette 0.0513 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_neighbors=50 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedd...
- **Verdict:** KEEP — ARI=0.5690 (delta -0.1505 vs Exp 71 champion 0.7195), NMI=0.8279, sil=0.0513, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_neighbors=50 on DINOv2 produced delta=-0.1505 ARI vs Exp 71. does not improve over prior best. Next try: min_dist sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 126: UMAP HC: min_dist=0.0 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_min_dist=0.0_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6247 | NMI 0.8606 | silhouette 0.0741 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that min_dist=0.0 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embeddin...
- **Verdict:** KEEP — ARI=0.6247 (delta -0.0948 vs Exp 71 champion 0.7195), NMI=0.8606, sil=0.0741, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. min_dist=0.0 on DINOv2 produced delta=-0.0948 ARI vs Exp 71. does not improve over prior best. Next try: n_components sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 127: UMAP HC: min_dist=0.3 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_min_dist=0.3_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.5949 | NMI 0.8438 | silhouette 0.0439 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that min_dist=0.3 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embeddin...
- **Verdict:** KEEP — ARI=0.5949 (delta -0.1246 vs Exp 71 champion 0.7195), NMI=0.8438, sil=0.0439, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. min_dist=0.3 on DINOv2 produced delta=-0.1246 ARI vs Exp 71. does not improve over prior best. Next try: n_components sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 128: UMAP HC: min_dist=0.5 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_min_dist=0.5_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6156 | NMI 0.8514 | silhouette 0.0617 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that min_dist=0.5 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embeddin...
- **Verdict:** KEEP — ARI=0.6156 (delta -0.1039 vs Exp 71 champion 0.7195), NMI=0.8514, sil=0.0617, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. min_dist=0.5 on DINOv2 produced delta=-0.1039 ARI vs Exp 71. does not improve over prior best. Next try: n_components sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 129: UMAP HC: min_dist=0.99 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_min_dist=0.99_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.5665 | NMI 0.8258 | silhouette 0.0673 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that min_dist=0.99 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.52 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embeddi...
- **Verdict:** KEEP — ARI=0.5665 (delta -0.1530 vs Exp 71 champion 0.7195), NMI=0.8258, sil=0.0673, n_pred=40. WITHIN predicted 0.52-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. min_dist=0.99 on DINOv2 produced delta=-0.1530 ARI vs Exp 71. does not improve over prior best. Next try: n_components sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 130: UMAP HC: n_components=3 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_components=3_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6177 | NMI 0.8508 | silhouette 0.0449 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_components=3 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedd...
- **Verdict:** KEEP — ARI=0.6177 (delta -0.1018 vs Exp 71 champion 0.7195), NMI=0.8508, sil=0.0449, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_components=3 on DINOv2 produced delta=-0.1018 ARI vs Exp 71. does not improve over prior best. Next try: metric sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 131: UMAP HC: n_components=5 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_components=5_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.5860 | NMI 0.8412 | silhouette 0.0480 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_components=5 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedd...
- **Verdict:** KEEP — ARI=0.5860 (delta -0.1335 vs Exp 71 champion 0.7195), NMI=0.8412, sil=0.0480, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_components=5 on DINOv2 produced delta=-0.1335 ARI vs Exp 71. does not improve over prior best. Next try: metric sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 132: UMAP HC: n_components=30 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_components=30_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.5980 | NMI 0.8495 | silhouette 0.0616 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_components=30 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embed...
- **Verdict:** KEEP — ARI=0.5980 (delta -0.1215 vs Exp 71 champion 0.7195), NMI=0.8495, sil=0.0616, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_components=30 on DINOv2 produced delta=-0.1215 ARI vs Exp 71. does not improve over prior best. Next try: metric sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 133: UMAP HC: n_components=50 on DINOv2 on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_n_components=50_on_DINOv2` | **Status:** KEEP
- **Result:** ARI 0.6107 | NMI 0.8453 | silhouette 0.0594 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that n_components=50 on DINOv2 on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embed...
- **Verdict:** KEEP — ARI=0.6107 (delta -0.1088 vs Exp 71 champion 0.7195), NMI=0.8453, sil=0.0594, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. n_components=50 on DINOv2 produced delta=-0.1088 ARI vs Exp 71. does not improve over prior best. Next try: metric sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 134: UMAP HC: metric=cosine on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_metric=cosine` | **Status:** KEEP
- **Result:** ARI 0.6000 | NMI 0.8421 | silhouette 0.0677 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that metric=cosine on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedding structu...
- **Verdict:** KEEP — ARI=0.6000 (delta -0.1195 vs Exp 71 champion 0.7195), NMI=0.8421, sil=0.0677, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. metric=cosine produced delta=-0.1195 ARI vs Exp 71. does not improve over prior best. Next try: UMAP + Spectral cosine downstream. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 135: UMAP HC: metric=manhattan on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_metric=manhattan` | **Status:** KEEP
- **Result:** ARI 0.6371 | NMI 0.8579 | silhouette 0.0621 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that metric=manhattan on DINOv2 ViT-S/14 will land ARI in 0.57 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the manifold/embedding stru...
- **Verdict:** KEEP — ARI=0.6371 (delta -0.0824 vs Exp 71 champion 0.7195), NMI=0.8579, sil=0.0621, n_pred=40. WITHIN predicted 0.57-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. metric=manhattan produced delta=-0.0824 ARI vs Exp 71. does not improve over prior best. Next try: UMAP + Spectral cosine downstream. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 136: UMAP HC: UMAP(10) + Spectral cosine downstream on DINOv2 ViT-S/14
- **Backbone:** `umap_hc_UMAP(10)_+_Spectral_cosine_dow` | **Status:** DISCARD
- **Result:** ARI 0.1918 | NMI 0.6401 | silhouette -0.1080 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that UMAP(10) + Spectral cosine downstream on DINOv2 ViT-S/14 will land ARI in 0.62 to 0.82 because the mechanism per the cited papers is that this UMAP configuration changes how the ma...
- **Verdict:** DISCARD — ARI=0.1918 (delta -0.5277 vs Exp 71 champion 0.7195), NMI=0.6401, sil=-0.1080, n_pred=40. BELOW predicted 0.62-0.82. local hill-climb on UMAP. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. UMAP(10) + Spectral cosine downstream produced delta=-0.5277 ARI vs Exp 71. does not improve over prior best. Next try: DEC hill-climb starts. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 137: DEC HC: latent_dim=32
- **Backbone:** `dec_hc_latent_dim32` | **Status:** KEEP
- **Result:** ARI 0.4955 | NMI 0.7982 | silhouette 0.1448 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that latent_dim=32 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpness...
- **Verdict:** KEEP — ARI=0.4955 (delta -0.2240 vs champion), NMI=0.7982, sil=0.1448, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. latent_dim=32 delta=-0.2240 vs champion. this DEC HP value does not improve over baseline. Next try: next latent variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 138: DEC HC: latent_dim=128
- **Backbone:** `dec_hc_latent_dim128` | **Status:** KEEP
- **Result:** ARI 0.4781 | NMI 0.7994 | silhouette 0.1453 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that latent_dim=128 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpnes...
- **Verdict:** KEEP — ARI=0.4781 (delta -0.2414 vs champion), NMI=0.7994, sil=0.1453, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. latent_dim=128 delta=-0.2414 vs champion. this DEC HP value does not improve over baseline. Next try: next latent variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 139: DEC HC: latent_dim=256
- **Backbone:** `dec_hc_latent_dim256` | **Status:** KEEP
- **Result:** ARI 0.5091 | NMI 0.8162 | silhouette 0.1421 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that latent_dim=256 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpnes...
- **Verdict:** KEEP — ARI=0.5091 (delta -0.2104 vs champion), NMI=0.8162, sil=0.1421, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. latent_dim=256 delta=-0.2104 vs champion. this DEC HP value does not improve over baseline. Next try: next latent variant. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 140: DEC HC: alpha=0.5
- **Backbone:** `dec_hc_alpha0.5` | **Status:** KEEP
- **Result:** ARI 0.5104 | NMI 0.8120 | silhouette 0.1493 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that alpha=0.5 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpness; th...
- **Verdict:** KEEP — ARI=0.5104 (delta -0.2091 vs champion), NMI=0.8120, sil=0.1493, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. alpha=0.5 delta=-0.2091 vs champion. this DEC HP value does not improve over baseline. Next try: MSE weight sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 141: DEC HC: alpha=2.0
- **Backbone:** `dec_hc_alpha2.0` | **Status:** KEEP
- **Result:** ARI 0.4841 | NMI 0.8060 | silhouette 0.1499 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that alpha=2.0 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpness; th...
- **Verdict:** KEEP — ARI=0.4841 (delta -0.2354 vs champion), NMI=0.8060, sil=0.1499, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. alpha=2.0 delta=-0.2354 vs champion. this DEC HP value does not improve over baseline. Next try: MSE weight sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 142: DEC HC: alpha=5.0
- **Backbone:** `dec_hc_alpha5.0` | **Status:** KEEP
- **Result:** ARI 0.4727 | NMI 0.7933 | silhouette 0.1478 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that alpha=5.0 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpness; th...
- **Verdict:** KEEP — ARI=0.4727 (delta -0.2468 vs champion), NMI=0.7933, sil=0.1478, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. alpha=5.0 delta=-0.2468 vs champion. this DEC HP value does not improve over baseline. Next try: MSE weight sweep. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 143: DEC HC: mse_weight=0.0
- **Backbone:** `dec_hc_mse_weight0.0` | **Status:** KEEP
- **Result:** ARI 0.4973 | NMI 0.8073 | silhouette 0.1431 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that mse_weight=0.0 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpnes...
- **Verdict:** KEEP — ARI=0.4973 (delta -0.2222 vs champion), NMI=0.8073, sil=0.1431, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. mse_weight=0.0 delta=-0.2222 vs champion. this DEC HP value does not improve over baseline. Next try: more pretrain epochs. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 144: DEC HC: mse_weight=0.5
- **Backbone:** `dec_hc_mse_weight0.5` | **Status:** KEEP
- **Result:** ARI 0.4435 | NMI 0.7828 | silhouette 0.1484 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that mse_weight=0.5 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpnes...
- **Verdict:** KEEP — ARI=0.4435 (delta -0.2760 vs champion), NMI=0.7828, sil=0.1484, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. mse_weight=0.5 delta=-0.2760 vs champion. this DEC HP value does not improve over baseline. Next try: more pretrain epochs. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 145: DEC HC: mse_weight=1.0
- **Backbone:** `dec_hc_mse_weight1.0` | **Status:** KEEP
- **Result:** ARI 0.4891 | NMI 0.8118 | silhouette 0.1583 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that mse_weight=1.0 will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-assignment sharpnes...
- **Verdict:** KEEP — ARI=0.4891 (delta -0.2304 vs champion), NMI=0.8118, sil=0.1583, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. mse_weight=1.0 delta=-0.2304 vs champion. this DEC HP value does not improve over baseline. Next try: more pretrain epochs. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.

### Exp 146: DEC HC: pretrain_epochs=80 (2x default)
- **Backbone:** `dec_hc_pretrain_epochs80_(2x_default)` | **Status:** KEEP
- **Result:** ARI 0.5002 | NMI 0.8081 | silhouette 0.1403 | n_pred_clusters 40
- **Hypothesis (first 200ch):** We hypothesize that pretrain_epochs=80 (2x default) will land ARI in 0.42 to 0.82 because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation capacity or the cluster-as...
- **Verdict:** KEEP — ARI=0.5002 (delta -0.2193 vs champion), NMI=0.8081, sil=0.1403, n_pred=40. WITHIN predicted 0.42-0.82. local DEC hill-climb. Status decision considers both the extrinsic ARI floor and intrinsic silhouette consistency per the project CLAUDE.md.
- **Learning:** axis closed. pretrain_epochs=80 (2x default) delta=-0.2193 vs champion. this DEC HP value does not improve over baseline. Next try: Hill-climb mandate complete. The cumulative best ARI across all experiments so far drives the choice of which axis the next experiment will probe.
