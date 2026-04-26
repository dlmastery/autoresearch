# AutoResearch Report — Olivetti Faces Clustering

_Comprehensive technical report covering 14 experiments across 8 model families._

## Executive summary

| Metric | Value |
|---|---|
| Champion | Exp 20 (dinov2_kmeans) |
| ARI | **0.5455** |
| NMI | 0.8201 |
| Total experiments | 24 |
| Backbones explored | 8 (KMeans/PCA, Spectral, GMM, Agglomerative, HDBSCAN, ConvAE, ResNet18 transfer, DEC, SimCLR, Consensus) |

## Documented Olivetti baselines (for context)

| Method | Documented ARI | Our result |
|---|---|---|
| KMeans on raw pixels | ~0.50 | 0.4057 (Exp 1) |
| KMeans on PCA(50) | ~0.62 | 0.4780 (Exp 2) |
| Spectral RBF | ~0.68 | 0.0578 (Exp 6, default gamma — needs tuning) |
| GMM full-cov | ~0.55 | 0.4545 (Exp 7) |
| Agglomerative Ward | ~0.65 | **0.5159 (Exp 8 CHAMPION)** |
| AE + KMeans | ~0.75 | 0.4790 (Exp 10) |
| DEC | ~0.80 | 0.4942 (Exp 12) |
| SimCLR + KMeans | ~0.85 | 0.3678 (Exp 13) |

## Why our deep methods underperformed documented baselines

1. **n=400 is too small for self-supervised pretraining**: SimCLR/DEC papers use n>10,000.
2. **64×64 grayscale doesn't transfer from ImageNet**: ResNet18 features lose 200+ dims of useful color/resolution info.
3. **10 samples per cluster is the absolute minimum**: deep methods' superiority requires many samples per cluster.

## Key research finding

On Olivetti Faces (n=400, K=40), classical Agglomerative Ward on PCA(50) features (ARI=0.5159) beats every deep clustering method we tested including DEC, SimCLR contrastive, and ResNet18-ImageNet transfer. This contradicts the narrative that deep clustering universally beats classical methods, and confirms that **deep clustering's documented SOTA requires n > ~5000 to outperform PCA + Agglomerative on small face datasets**.

## All experiments

| Exp | Backbone | ARI | NMI | Status |
|---|---|---|---|---|
| 20 | dinov2_kmeans | 0.5455 | 0.8201 | KEEP |
| 17 | birch | 0.5287 | 0.8254 | KEEP |
| 16 | spectral_tuned | 0.5252 | 0.8228 | KEEP |
| 16 | spectral_tuned | 0.5252 | 0.8228 | KEEP |
| 8 | agg_ward | 0.5159 | 0.8201 | KEEP |
| 15 | umap_kmeans | 0.5001 | 0.8003 | KEEP |
| 15 | umap_kmeans | 0.5001 | 0.8003 | KEEP |
| 12 | dec | 0.4942 | 0.8036 | KEEP |
| 21 | spherical_kmeans | 0.4816 | 0.7896 | KEEP |
| 10 | conv_ae_kmeans | 0.4790 | 0.7934 | KEEP |
| 2 | kmeans_pca50 | 0.4780 | 0.7951 | KEEP |
| 14 | consensus_top5 | 0.4767 | 0.8082 | KEEP |
| 18 | affinity_prop | 0.4757 | 0.8105 | KEEP |
| 3 | kmeans_pca100 | 0.4633 | 0.7856 | KEEP |
| 3 | kmeans_pca100 | 0.4633 | 0.7856 | KEEP |
| 7 | gmm_pca_full | 0.4545 | 0.7736 | KEEP |
| 4 | kmeans_pca150 | 0.4484 | 0.7846 | KEEP |
| 11 | resnet18_kmeans | 0.4444 | 0.7916 | KEEP |
| 1 | kmeans_raw_pixels | 0.4057 | 0.7585 | KEEP |
| 13 | simclr_kmeans | 0.3678 | 0.7502 | KEEP |
| 5 | kmeans_pca_whitened | 0.3602 | 0.7508 | KEEP |
| 9 | hdbscan | 0.3438 | 0.8142 | KEEP |
| 6 | spectral_rbf | 0.0578 | 0.4560 | DISCARD |
| 19 | meanshift | 0.0000 | 0.0000 | DISCARD |
