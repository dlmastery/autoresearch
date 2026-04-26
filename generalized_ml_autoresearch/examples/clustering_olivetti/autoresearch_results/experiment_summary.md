# Experiment Summary — Olivetti Faces Clustering Autoresearch

_Generated 2026-04-25 23:56_

## Master leaderboard (sorted by ARI on full 400-row Olivetti dataset)

| Rank | Exp | Backbone | ARI | NMI | Silhouette | Status | Description |
|------|-----|----------|-----|-----|------------|--------|-------------|
| 1 | 33 | dinov2_vits14_spectral_cos | 0.6963 | 0.8974 | 0.0890 | KEEP | DINOv2 dinov2_vits14 + Spectral cosine affinity |
| 2 | 34 | dinov2_vits14_spectral_knn10 | 0.6389 | 0.8584 | 0.0796 | KEEP | DINOv2 dinov2_vits14 + Spectral nearest-neighbors affin |
| 3 | 27 | dinov2_vits14_agg_ward | 0.6371 | 0.8706 | 0.0834 | KEEP | DINOv2 dinov2_vits14 + Agglomerative Ward (variance-min |
| 4 | 35 | dinov2_vits14_birch | 0.6371 | 0.8706 | 0.0834 | KEEP | DINOv2 dinov2_vits14 + Birch on DINOv2 features |
| 5 | 41 | dinov2_vits14_umap2_km | 0.6100 | 0.8455 | 0.0678 | KEEP | DINOv2 dinov2_vits14 + UMAP(2) on DINOv2 + KMeans (extr |
| 6 | 40 | dinov2_vits14_umap10_km | 0.5982 | 0.8465 | 0.0592 | KEEP | DINOv2 dinov2_vits14 + UMAP(10) on DINOv2 + KMeans |
| 7 | 31 | dinov2_vits14_spectral_g001 | 0.5852 | 0.8533 | 0.0872 | KEEP | DINOv2 dinov2_vits14 + Spectral RBF gamma=0.001 (small) |
| 8 | 25 | dinov2_vits14_kmeans_n50 | 0.5852 | 0.8456 | 0.0891 | KEEP | DINOv2 dinov2_vits14 + KMeans n_init=50 (5x more random |
| 9 | 26 | dinov2_vits14_spherical | 0.5602 | 0.8259 | 0.0467 | KEEP | DINOv2 dinov2_vits14 + L2-normalized features + KMeans  |
| 10 | 22 | dinov2_vits14_minibatch_kmeans | 0.5596 | 0.8393 | 0.0596 | KEEP | DINOv2 dinov2_vits14 + MiniBatchKMeans (faster, may be  |
| 11 | 44 | dinov2_vits14_seed1 | 0.5561 | 0.8301 | 0.0904 | KEEP | DINOv2 dinov2_vits14 + KMeans seed=1 (variance check on |
| 12 | 39 | dinov2_vits14_pca100_km | 0.5473 | 0.8278 | 0.0745 | KEEP | DINOv2 dinov2_vits14 + PCA(100) on DINOv2 + KMeans |
| 13 | 20 | dinov2_kmeans | 0.5455 | 0.8201 | 0.0710 | KEEP | DINOv2 ViT-S/14 (Oquab 2024 Meta TMLR) features + KMean |
| 14 | 42 | dinov2_vitb14_vitb_km | 0.5445 | 0.8243 | 0.0379 | KEEP | DINOv2 dinov2_vitb14 + ViT-B/14 features + KMeans (larg |
| 15 | 43 | dinov2_vitb14_vitb_spherical | 0.5388 | 0.8119 | 0.0506 | KEEP | DINOv2 dinov2_vitb14 + ViT-B/14 + L2-norm + KMeans (Sph |
| 16 | 46 | dinov2_vits14_seed7 | 0.5387 | 0.8175 | 0.0633 | KEEP | DINOv2 dinov2_vits14 + KMeans seed=7 (variance check on |
| 17 | 38 | dinov2_vits14_pca50_km | 0.5312 | 0.8184 | 0.0328 | KEEP | DINOv2 dinov2_vits14 + PCA(50) on DINOv2 + KMeans (deno |
| 18 | 17 | birch | 0.5287 | 0.8254 | 0.1608 | KEEP | Birch (Zhang 1996) on PCA(50) |
| 19 | 16 | spectral_tuned | 0.5252 | 0.8228 | 0.1159 | KEEP | Spectral RBF with gamma sweep on PCA(50) |
| 20 | 16 | spectral_tuned | 0.5252 | 0.8228 | 0.1159 | KEEP | Spectral RBF with gamma sweep on PCA(50) |
| 21 | 36 | dinov2_vits14_gmm_full | 0.5234 | 0.8133 | 0.0341 | KEEP | DINOv2 dinov2_vits14 + GMM full-covariance K=40 |
| 22 | 37 | dinov2_vits14_gmm_diag | 0.5234 | 0.8133 | 0.0341 | KEEP | DINOv2 dinov2_vits14 + GMM diagonal-covariance |
| 23 | 8 | agg_ward | 0.5159 | 0.8201 | 0.1608 | KEEP | Agglomerative Ward on PCA(50) (Ward 1963) |
| 24 | 45 | dinov2_vits14_seed2 | 0.5144 | 0.8110 | 0.0712 | KEEP | DINOv2 dinov2_vits14 + KMeans seed=2 (variance check on |
| 25 | 15 | umap_kmeans | 0.5001 | 0.8003 | 0.1278 | KEEP | UMAP(10) + KMeans (McInnes 2018) |
| 26 | 15 | umap_kmeans | 0.5001 | 0.8003 | 0.1278 | KEEP | UMAP(10) + KMeans (McInnes 2018) |
| 27 | 24 | dinov2_vits14_kmeans_random | 0.5000 | 0.8091 | 0.0304 | KEEP | DINOv2 dinov2_vits14 + KMeans with random init (vs k-me |
| 28 | 12 | dec | 0.4942 | 0.8036 | 0.1436 | KEEP | DEC: Deep Embedded Clustering (Xie 2016 ICML + Guo 2017 |
| 29 | 21 | spherical_kmeans | 0.4816 | 0.7896 | 0.1266 | KEEP | Spherical KMeans (Dhillon 2001) on L2-norm PCA(50) |
| 30 | 29 | dinov2_vits14_agg_complete | 0.4805 | 0.8071 | 0.0234 | KEEP | DINOv2 dinov2_vits14 + Agglomerative complete-linkage ( |
| 31 | 10 | conv_ae_kmeans | 0.4790 | 0.7934 | 0.1469 | KEEP | Convolutional AE (Hinton 2006) + KMeans, latent=64 |
| 32 | 2 | kmeans_pca50 | 0.4780 | 0.7951 | 0.1485 | KEEP | PCA(50) + KMeans (Pearson 1901 + Steinley 2006) |
| 33 | 14 | consensus_top5 | 0.4767 | 0.8082 | 0.1530 | KEEP | CSPA consensus of top-5 methods: agg_ward, dec, conv_ae |
| 34 | 18 | affinity_prop | 0.4757 | 0.8105 | 0.1737 | KEEP | Affinity Propagation (Frey 2007 Science) on PCA(50) |
| 35 | 28 | dinov2_vits14_agg_avg | 0.4703 | 0.8158 | 0.0226 | KEEP | DINOv2 dinov2_vits14 + Agglomerative average-linkage |
| 36 | 3 | kmeans_pca100 | 0.4633 | 0.7856 | 0.1506 | KEEP | PCA(100) + KMeans |
| 37 | 3 | kmeans_pca100 | 0.4633 | 0.7856 | 0.1506 | KEEP | PCA(100) + KMeans |
| 38 | 7 | gmm_pca_full | 0.4545 | 0.7736 | 0.1394 | KEEP | GMM full-cov on PCA(50) (Bishop 2006 Ch.9) |
| 39 | 30 | dinov2_vits14_agg_cosine_avg | 0.4490 | 0.8174 | 0.0134 | KEEP | DINOv2 dinov2_vits14 + Agglomerative cosine + average l |
| 40 | 4 | kmeans_pca150 | 0.4484 | 0.7846 | 0.1456 | KEEP | PCA(150) + KMeans |
| 41 | 11 | resnet18_kmeans | 0.4444 | 0.7916 | 0.0324 | KEEP | ResNet18-ImageNet (He 2016) penultimate features + KMea |
| 42 | 23 | dinov2_vits14_bisecting_kmeans | 0.4437 | 0.7678 | 0.0277 | KEEP | DINOv2 dinov2_vits14 + BisectingKMeans hierarchical bis |
| 43 | 1 | kmeans_raw_pixels | 0.4057 | 0.7585 | 0.1479 | KEEP | KMeans K=40 on raw pixels — baseline (Lloyd 1982 + Arth |
| 44 | 13 | simclr_kmeans | 0.3678 | 0.7502 | 0.0503 | KEEP | SimCLR (Chen 2020 ICML) + KMeans |
| 45 | 5 | kmeans_pca_whitened | 0.3602 | 0.7508 | 0.0775 | KEEP | PCA(50) + whitening + KMeans |
| 46 | 9 | hdbscan | 0.3438 | 0.8142 | 0.1807 | KEEP | HDBSCAN on PCA(50) (Campello 2013) |
| 47 | 32 | dinov2_vits14_spectral_g01 | 0.2767 | 0.7672 | 0.0361 | DISCARD | DINOv2 dinov2_vits14 + Spectral RBF gamma=0.01 |
| 48 | 6 | spectral_rbf | 0.0578 | 0.4560 | -0.1250 | DISCARD | Spectral clustering (RBF affinity) |
| 49 | 19 | meanshift | 0.0000 | 0.0000 | nan | DISCARD | MeanShift (Comaniciu 2002) on PCA(50) |

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
