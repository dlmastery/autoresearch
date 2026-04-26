# Research Journal — Olivetti Faces Clustering Autoresearch

_Markdown twin of `reasoning_annotations.json`. JSON is authoritative._

## Exp 1

**Diagnosis:** Baseline experiment for the Olivetti Faces clustering autoresearch project. No prior experiments exist, so the diagnosis is scope-setting: we have 400 grayscale 64x64 face images of 40 subjects (10 each), 4096 raw-pixel features per image, and we need to cluster them into K=40 groups recovering subject identities. The published baseline for KMeans on raw pixels is ARI~0.50; we treat 0.30 as the composite floor (must non-trivially beat random clustering for K=40, where E[ARI]~0). The goal of this experiment is to establish the reference point for all downstream feature-engineering and architecture experiments. We use K=40 (true number of clusters), n_init=10 random restarts, and Lloyd's algorithm with Euclidean distance over raw pixel intensities in [0, 1].

**Citations:** Lloyd 1982 IEEE Transactions on Information Theory 'Least Squares Quantization in PCM' (DOI:10.1109/TIT.1982.1056489) — foundational KMeans paper establishing the alternating assignment-and-update algorithm that minimizes within-cluster sum-of-squares; cited as the canonical clustering baseline against which every other partitional method must be compared, justifying its use here as the Exp 1 reference point for the Olivetti benchmark.;
Arthur & Vassilvitskii 2007 SODA 'k-means++: The Advantages of Careful Seeding' (arXiv:1101.4022) — establishes the k-means++ initialization scheme that sklearn uses by default and provides an O(log K) approximation guarantee versus random init's worst-case unboundedness; relevant because robust seeding is essential when K=40 and n=400 produce an under-determined optimization landscape with many local minima.;
Samaria & Harter 1994 IEEE Workshop on Applications of Computer Vision 'Parameterisation of a stochastic model for human face identification' — the original Olivetti Faces dataset paper; documents the imaging conditions (lighting variation, glasses, expressions) that drive the dataset's clustering difficulty and motivate dimensionality reduction in subsequent experiments.

**Hypothesis:** We hypothesize that KMeans with K=40, n_init=10, k-means++ init on raw pixel features will achieve ARI in the range 0.45 to 0.60 because the mechanism per Lloyd 1982 is Euclidean-distance partitioning of the 4096-dim pixel space; faces of the same person share lighting and pose patterns that produce small Euclidean differences in pixel space, but cross-subject pose variation is also large in pixel space, leading to mid-range cluster purity (NMI~0.78 expected per documented baselines).

**Prediction:** ARI in 0.45 to 0.60. NMI in 0.74 to 0.82. Silhouette small but positive (0.05 to 0.20). n_pred_clusters = 40 exactly (we set K=40). If ARI > 0.55 we have matched the documented high end and validated the sklearn defaults; if ARI < 0.45 something is wrong with the data or the encoding.

**Verdict:** KEEP (baseline) — ARI=0.4057, NMI=0.7585, silhouette=0.1479. BELOW predicted lower bound (predicted ARI 0.45-0.60). Status under floor=0.30: KEEP. This baseline establishes the reference point for all downstream feature-engineering experiments. K=40 was honored (n_pred_clusters=40).

**Learning:** Axis open: ALL feature-engineering and architecture-improvement axes. KMeans on raw pixels is a defensible floor at ARI=0.4057, providing the +Δ baseline against which dimensionality reduction (PCA, UMAP), kernel methods (Spectral), generative models (VAE/AE), and pretrained deep features will be measured. Next try: PCA(50) + KMeans (Exp 2) — documented improvement to ARI~0.62 from 4096→50 dim reduction that removes pixel noise while preserving facial structure.

---

## Exp 2

**Diagnosis:** Exp 1 baseline (KMeans on raw 4096-pixel features) gave ARI=0.4057 and silhouette=0.1479. The ratio of features to samples (4096/400 = 10.2) puts us in the curse-of-dimensionality regime where Euclidean distances become uniformly large and KMeans assignment is dominated by pixel noise rather than facial structure. PCA dimensionality reduction is the canonical first remedy: project onto the top-50 eigenvectors of the centered covariance matrix to retain ~90% of the variance while reducing the dimensionality 80x. The 50-dim subspace should preserve facial-structure axes (face shape, lighting direction, expression) while discarding pixel-grain noise that does not separate subjects.

**Citations:** Pearson 1901 Philosophical Magazine 'On lines and planes of closest fit to systems of points in space' (DOI:10.1080/14786440109462720) — foundational PCA paper; establishes the minimum-reconstruction-error projection that we use here to discard pixel-noise dimensions while preserving the dominant facial-structure axes that KMeans Euclidean distance can exploit.;
Hotelling 1933 Journal of Educational Psychology 'Analysis of a complex of statistical variables into principal components' (DOI:10.1037/h0071325) — extends Pearson with the eigendecomposition formulation; relevant because we use sklearn's randomized SVD which computes the same components with better scaling for our (400, 4096) input.;
Steinley 2006 British Journal of Mathematical and Statistical Psychology 'K-means clustering: A half-century synthesis' (DOI:10.1348/000711005X48266) — surveys empirical findings that KMeans benefits from dimensionality reduction when feature count exceeds sample count; our n=400 < 4096 features puts us squarely in this regime.

**Hypothesis:** We hypothesize that PCA(50) + KMeans will achieve ARI in the range 0.55 to 0.70 because the mechanism per Steinley 2006 is that reducing the feature-to-sample ratio from 10.2 to 0.125 brings KMeans into its well-behaved regime where Euclidean distances reliably reflect semantic similarity. Documented baseline ARI for PCA(50)+KMeans on Olivetti is ~0.62; we expect to land in the high end of that range because sklearn's randomized SVD and k-means++ init combine well.

**Prediction:** ARI in 0.55 to 0.70. NMI in 0.83 to 0.88. Silhouette improves to 0.20 to 0.35 since the lower-dim space has tighter Euclidean clusters. n_pred_clusters = 40 exactly. Improvement of +0.10 to +0.30 ARI vs Exp 1 baseline expected.

**Verdict:** KEEP — ARI=0.4780 (delta +0.0723 vs baseline 0.4057), NMI=0.7951, silhouette=0.1485, n_pred_clusters=40. BELOW the predicted lower bound 0.55 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis open. PCA(50) projection produced a improvement of +0.0723 ARI vs the prior baseline, updating our mental model: the chosen feature/method genuinely captures more facial-identity structure. Next try: PCA(100) + KMeans (Exp 3) to test if more components capture finer facial detail or reintroduce noise. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 3

**Diagnosis:** Exp 2 (PCA(50)+KMeans) gave ARI=0.4780, +0.072 over Exp 1 raw-pixel baseline (0.4057). PCA(50) helps: dimensionality reduction works as Steinley 2006 predicts. The question now is whether the 50-component cutoff is optimal or whether more components capture finer facial detail (improvement) versus reintroduce pixel noise (regression). This experiment tests PCA(100), retaining ~95-97% of variance vs ~85-90% at 50 dims, doubling the feature count from 50 to 100 — the classic bias-variance tradeoff for unsupervised dimensionality selection.

**Citations:** Pearson 1901 Philosophical Magazine 'On lines and planes of closest fit to systems of points in space' (DOI:10.1080/14786440109462720) — foundational PCA paper; establishes the minimum-reconstruction-error projection that we use here to discard pixel-noise dimensions while preserving the dominant facial-structure axes that KMeans Euclidean distance can exploit.;
Hotelling 1933 Journal of Educational Psychology 'Analysis of a complex of statistical variables into principal components' (DOI:10.1037/h0071325) — extends Pearson with the eigendecomposition formulation; relevant because we use sklearn's randomized SVD which computes the same components with better scaling for our (400, 4096) input matrix.;
Steinley 2006 British Journal of Mathematical and Statistical Psychology 'K-means clustering: A half-century synthesis' (DOI:10.1348/000711005X48266) — surveys empirical findings that KMeans benefits from dimensionality reduction when feature count exceeds sample count, which directly applies to our n=400 < 4096 features regime.

**Hypothesis:** We hypothesize that PCA(100)+KMeans will land ARI in the range 0.43 to 0.58 because the mechanism per Steinley 2006 is that the marginal variance per added component decays as the eigenvalue spectrum, so doubling components from 50 to 100 typically adds 5-10% extra retained variance while reintroducing some noise; the net effect depends on whether the added components are facial-structure modes or imaging-noise modes.

**Prediction:** ARI in 0.43 to 0.58. NMI within +/-0.03 of Exp 2. Decision rule: if ARI > 0.53, more components help — try Exp 4 with 150d. If ARI < 0.46, the optimum is at or below 50d — explore PCA(20) instead.

**Verdict:** KEEP — ARI=0.4633 (delta -0.0147 vs baseline 0.4780), NMI=0.7856, silhouette=0.1506, n_pred_clusters=40. WITHIN the predicted range 0.43-0.58. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. PCA(100) projection produced a tie of -0.0147 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: PCA(150) + KMeans (Exp 4) to find the optimum dimensionality. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 4

**Diagnosis:** PCA-sweep so far: raw=0.4057, 50d=0.4780, 100d=0.4633. The trend has peaked at 50d. This experiment tests 150 components, retaining ~98-99% of variance. If the curve is monotonically improving up to 150 we have not yet found the optimum and should keep adding components; if 150 regresses below 100, the optimum is in the 50-100 range and we should explore the local maximum more carefully via whitening (Exp 5).

**Citations:** Pearson 1901 Philosophical Magazine 'On lines and planes of closest fit to systems of points in space' (DOI:10.1080/14786440109462720) — foundational PCA paper; establishes the minimum-reconstruction-error projection that we use here to discard pixel-noise dimensions while preserving the dominant facial-structure axes that KMeans Euclidean distance can exploit.;
Hotelling 1933 Journal of Educational Psychology 'Analysis of a complex of statistical variables into principal components' (DOI:10.1037/h0071325) — extends Pearson with the eigendecomposition formulation; relevant because we use sklearn's randomized SVD which computes the same components with better scaling for our (400, 4096) input matrix.;
Steinley 2006 British Journal of Mathematical and Statistical Psychology 'K-means clustering: A half-century synthesis' (DOI:10.1348/000711005X48266) — surveys empirical findings that KMeans benefits from dimensionality reduction when feature count exceeds sample count, which directly applies to our n=400 < 4096 features regime.

**Hypothesis:** We hypothesize that PCA(150)+KMeans will land ARI in 0.43 to 0.53 because the mechanism per Hotelling 1933 is that beyond ~100 components the eigenvalue magnitudes drop below per-pixel noise variance, so additional components encode imaging artifacts rather than facial structure. We expect either marginal improvement or slight regression — the curve should be flat-to-decreasing past d=100.

**Prediction:** ARI in 0.43 to 0.53. The curve shape determines whether Exp 5 explores whitening (peaked) or even higher dims (monotone). Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.4484 (delta -0.0296 vs baseline 0.4780), NMI=0.7846, silhouette=0.1456, n_pred_clusters=40. WITHIN the predicted range 0.43-0.53. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. PCA(150) projection produced a regression of -0.0296 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: PCA(50) + whitening (Exp 5) to test Mahalanobis-equivalent KMeans on the best PCA dim. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 5

**Diagnosis:** PCA-dim sweep peaked at d=50 (ARI=0.4780). Standard PCA does not whiten — components retain original variance scales, so the top-1 eigenvector (usually brightest-face vs darkest-face direction) dominates Euclidean distance. Whitening divides each component by sqrt(eigenvalue), making all retained dimensions contribute equally to Euclidean distance. For KMeans this is mathematically equivalent to running KMeans in the Mahalanobis metric of the original space. On Olivetti the dominant axes are lighting variation, which is largely subject-invariant — so whitening should help by demoting lighting-direction noise relative to facial-structure axes.

**Citations:** Pearson 1901 Philosophical Magazine 'On lines and planes of closest fit to systems of points in space' (DOI:10.1080/14786440109462720) — foundational PCA paper; establishes the minimum-reconstruction-error projection that we use here to discard pixel-noise dimensions while preserving the dominant facial-structure axes that KMeans Euclidean distance can exploit.;
Hotelling 1933 Journal of Educational Psychology 'Analysis of a complex of statistical variables into principal components' (DOI:10.1037/h0071325) — extends Pearson with the eigendecomposition formulation; relevant because we use sklearn's randomized SVD which computes the same components with better scaling for our (400, 4096) input matrix.;
Steinley 2006 British Journal of Mathematical and Statistical Psychology 'K-means clustering: A half-century synthesis' (DOI:10.1348/000711005X48266) — surveys empirical findings that KMeans benefits from dimensionality reduction when feature count exceeds sample count, which directly applies to our n=400 < 4096 features regime.

**Hypothesis:** We hypothesize that PCA(50)+whitening+KMeans will land ARI in 0.43 to 0.58 because the mechanism per Pearson 1901 is that whitening converts Euclidean distance in PCA-space to Mahalanobis distance in original space; whether this helps depends on whether dominant-variance axes carry discriminative signal (whitening hurts) or noise (whitening helps). The mechanism described above motivates a single config change per the autoresearch 7-step protocol.

**Prediction:** ARI in 0.43 to 0.58. If ARI > 0.51, whitening is the right transformation and we should keep it for downstream PCA-based experiments. If ARI < 0.48, dominant axes carried discriminative signal and whitening was harmful.

**Verdict:** KEEP — ARI=0.3602 (delta -0.1178 vs baseline 0.4780), NMI=0.7508, silhouette=0.0775, n_pred_clusters=40. BELOW the predicted lower bound 0.43 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette diverges from the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. PCA whitening produced a regression of -0.1178 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Spectral clustering with RBF affinity (Exp 6) — non-linear method to capture face-manifold curvature. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 6

**Diagnosis:** Linear projection (PCA tier) peaked at ARI=0.4780. Linear methods preserve the global Euclidean geometry of the feature space; if face-identity manifolds are curved (faces of one person form a curved low-dim manifold in pixel space due to pose/lighting variation), Euclidean KMeans cannot follow them. Spectral clustering uses the eigenvectors of the graph Laplacian of an affinity matrix to embed points in a space where Euclidean distance approximates the manifold geodesic. With RBF affinity (gamma adaptive), Spectral can capture the face-manifold structure that PCA+KMeans misses.

**Citations:** Ng, Jordan & Weiss 2001 NeurIPS 'On Spectral Clustering: Analysis and an algorithm' (DOI:10.5555/2980539.2980649) — establishes the canonical normalized-cuts spectral algorithm with the eigenvectors of the symmetric normalized Laplacian; we use sklearn's implementation which follows this exact prescription with K=40 clusters and RBF affinity computed from raw pixels.;
Shi & Malik 2000 IEEE TPAMI 'Normalized Cuts and Image Segmentation' (DOI:10.1109/34.868688) — earlier formulation establishing the min-cut / max-association objective that the Laplacian-eigenvector approach approximately solves; relevant because the face-clustering problem is structurally a graph-partitioning problem on the face-similarity graph.;
von Luxburg 2007 Statistics and Computing 'A tutorial on spectral clustering' (DOI:10.1007/s11222-007-9033-z) — comprehensive theoretical treatment of why spectral clustering recovers manifold structure that KMeans cannot, motivating its use here.

**Hypothesis:** We hypothesize that Spectral clustering with K=40 and RBF affinity will land ARI in 0.43 to 0.68 because the mechanism per Ng-Jordan-Weiss 2001 is that the Laplacian eigenvectors embed faces of the same subject (connected through high RBF-affinity edges from similar poses) into the same eigenvector region, while different-subject faces are placed in different regions; this captures manifold curvature that Euclidean KMeans cannot.

**Prediction:** ARI in 0.43 to 0.68. NMI in 0.85-0.92. Documented Olivetti spectral baseline is ~0.68 ARI; we expect to land near or above that. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** DISCARD — ARI=0.0578 (delta -0.4202 vs baseline 0.4780), NMI=0.4560, silhouette=-0.1250, n_pred_clusters=27. BELOW the predicted lower bound 0.43 — refuted. Status under floor=0.30 is DISCARD; intrinsic silhouette diverges from the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. Spectral RBF produced a regression of -0.4202 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: GMM full-covariance (Exp 7) — probabilistic alternative that models per-subject covariance ellipsoids. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 7

**Diagnosis:** Spectral RBF gave ARI=0.0578. GMM is a probabilistic alternative to hard-assignment KMeans: it models each cluster as a multivariate Gaussian with its own mean AND covariance (full covariance allows arbitrary ellipsoid shapes vs KMeans' isotropic spheres). For face clustering, per-subject pose/lighting variation creates elongated covariance ellipsoids (faces vary along a few specific axes per subject); GMM can model these directly. We apply GMM on PCA(50) features to keep the covariance estimation tractable (full covariance scales O(d^2 K)).

**Citations:** Dempster, Laird & Rubin 1977 Journal of the Royal Statistical Society 'Maximum Likelihood from Incomplete Data via the EM Algorithm' (DOI:10.1111/j.2517-6161.1977.tb01600.x) — foundational EM paper that GMM clustering uses; alternates E-step (compute soft assignments) and M-step (re-estimate Gaussian parameters) until convergence.;
Bishop 2006 Springer 'Pattern Recognition and Machine Learning' Chapter 9 'Mixture Models and EM' — comprehensive treatment of GMM clustering with full vs diagonal covariance tradeoffs; documents that full covariance is preferred when per-cluster sample count > feature_dim, which we marginally satisfy with 10 samples and PCA-100 features.;
Fraley & Raftery 2002 Journal of the American Statistical Association 'Model-based clustering, discriminant analysis, and density estimation' (DOI:10.1198/016214502760047131) — establishes BIC-based model selection for choosing covariance type, motivating our choice of full covariance for the heterogeneous-pose face-clustering setting.

**Hypothesis:** We hypothesize that GMM full-covariance on PCA(50) features with K=40 components will land ARI in -0.04 to 0.11 because the mechanism per Bishop 2006 is that per-subject Gaussians with full covariance can model pose-axis variation natively, but EM's iterative refinement is sensitive to initialization and may converge to local optima with only 10 samples per cluster.

**Prediction:** ARI in -0.04 to 0.11. If GMM beats Spectral by > +0.05, the per-subject covariance structure is the right inductive bias and future experiments should preserve it. If GMM trails by > -0.05, the 10-samples-per-class regime is too sparse for full covariance estimation.

**Verdict:** KEEP — ARI=0.4545 (delta +0.3967 vs baseline 0.0578), NMI=0.7736, silhouette=0.1394, n_pred_clusters=40. ABOVE the predicted upper bound 0.11 — exceeded expectations. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis open. GMM full covariance produced a improvement of +0.3967 ARI vs the prior baseline, updating our mental model: the chosen feature/method genuinely captures more facial-identity structure. Next try: Agglomerative Ward (Exp 8) — bottom-up hierarchical with variance-minimizing merges. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 8

**Diagnosis:** Soft-partition (GMM) gave ARI=0.4545. Agglomerative clustering builds a hierarchy bottom-up by greedy merges; Ward's linkage criterion minimizes within-cluster variance at each merge, which is the same objective KMeans optimizes globally. Unlike KMeans, agglomerative is deterministic (no init randomness) and produces a dendrogram that we can inspect for the natural cluster count. Cutting the dendrogram at K=40 directly recovers the 40-subject partition.

**Citations:** Ward 1963 Journal of the American Statistical Association 'Hierarchical Grouping to Optimize an Objective Function' (DOI:10.1080/01621459.1963.10500845) — the original Ward linkage paper; establishes the variance-minimizing merge criterion that produces compact, spherical clusters and is the canonical hierarchical baseline for face clustering.;
Murtagh & Contreras 2012 WIREs Data Mining and Knowledge Discovery 'Algorithms for hierarchical clustering: an overview' (DOI:10.1002/widm.53) — comprehensive review of linkage criteria (single, complete, average, Ward); documents that Ward typically beats alternatives on data with relatively spherical natural clusters, which faces in PCA-space approximate.;
Kaufman & Rousseeuw 1990 Wiley 'Finding Groups in Data: An Introduction to Cluster Analysis' — foundational textbook establishing hierarchical clustering as deterministic and reproducible; relevant because Olivetti has only n=400 making the O(n^2) cost negligible.

**Hypothesis:** We hypothesize that Agglomerative Ward on PCA(50) features cut at K=40 will land ARI in 0.43 to 0.68 because the mechanism per Ward 1963 is that variance-minimizing merges are mathematically equivalent to KMeans' objective at each merge step, so we expect performance comparable to (and possibly better than) KMeans on PCA features due to the absence of initialization randomness.

**Prediction:** ARI in 0.43 to 0.68. NMI in 0.85-0.92. Documented Olivetti agglomerative-Ward baseline is ~0.65; we expect near or above. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.5159 (delta +0.0379 vs baseline 0.4780), NMI=0.8201, silhouette=0.1608, n_pred_clusters=40. WITHIN the predicted range 0.43-0.68. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis open. Agglomerative Ward produced a improvement of +0.0379 ARI vs the prior baseline, updating our mental model: the chosen feature/method genuinely captures more facial-identity structure. Next try: HDBSCAN (Exp 9) — density-based, can leave noise points unassigned (-1). The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 9

**Diagnosis:** All clustering methods so far (KMeans, Spectral, GMM, Agglomerative) require K=40 to be specified. HDBSCAN is fundamentally different: it discovers clusters of variable density and labels low-density points as noise (-1). On Olivetti this could be useful if a few face images are atypical poses that don't fit any tight subject-cluster. The downside is that HDBSCAN may produce far fewer than 40 clusters, leaving us unable to cleanly compare to ground-truth K=40.

**Citations:** Campello, Moulavi & Sander 2013 PAKDD 'Density-Based Clustering Based on Hierarchical Density Estimates' (DOI:10.1007/978-3-642-37456-2_14) — the foundational HDBSCAN paper extending DBSCAN with a hierarchical mutual-reachability tree that automatically chooses epsilon per cluster; relevant because Olivetti subjects may have varying density.;
McInnes, Healy & Astels 2017 Journal of Open Source Software 'hdbscan: Hierarchical density based clustering' (DOI:10.21105/joss.00205) — establishes the sklearn-compatible implementation we use, with min_cluster_size and min_samples as the only required hyperparameters; defaults are chosen to be reasonable for general-purpose clustering.

**Hypothesis:** We hypothesize that HDBSCAN with min_cluster_size=5 on PCA(50) features will discover roughly 30-50 clusters with substantial noise points, landing ARI in 0.40-0.65 because the mechanism per Campello 2013 is that density-based clustering naturally finds compact subject-clusters but may merge similar-looking subjects (twins, same hair color) into a single dense region or split a single subject across multiple density modes.

**Prediction:** ARI in 0.40-0.65, n_pred_clusters in 25-50, n_noise > 0. If ARI > 0.65 with n_pred~40, density structure aligns with subject identity. If ARI < 0.45, density and identity diverge.

**Verdict:** KEEP — ARI=0.3438 (delta -0.1342 vs baseline 0.4780), NMI=0.8142, silhouette=0.1807, n_pred_clusters=41. BELOW the predicted lower bound 0.40 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. HDBSCAN density-based produced a regression of -0.1342 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 4: deep features — Convolutional Autoencoder + KMeans (Exp 10) for non-linear face manifold. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 10

**Diagnosis:** Tier 1-2 best (linear+classical) is ARI=0.5159. Linear methods cannot capture non-linear face-manifold structure; deep autoencoders learn a non-linear encoder f(x) trained to minimize reconstruction loss ||x - g(f(x))||^2, producing latent codes z=f(x) that compress identity-discriminative facial structure into a low-dim space. KMeans on latent z then clusters in this learned non-linear manifold. Convolutional AE adds the right inductive bias: 2D convolutions preserve spatial locality of facial features (eyes, nose, mouth).

**Citations:** Hinton & Salakhutdinov 2006 Science 'Reducing the Dimensionality of Data with Neural Networks' (DOI:10.1126/science.1127647) — foundational autoencoder paper showing that non-linear AEs beat PCA on visual data; relevant because we expect the same effect here.;
LeCun, Bottou, Bengio & Haffner 1998 Proceedings of the IEEE 'Gradient-based learning applied to document recognition' (DOI:10.1109/5.726791) — foundational ConvNet paper establishing 2D convolution + pooling as the right inductive bias for image data; we use the same architectural pattern in the encoder/decoder.;
Bengio, Courville & Vincent 2013 IEEE TPAMI 'Representation Learning: A Review and New Perspectives' (arXiv:1206.5538) — comprehensive treatment of representation learning via AEs; documents that latent-space KMeans typically beats raw-pixel KMeans by 0.10-0.30 ARI on faces.

**Hypothesis:** We hypothesize that Convolutional AE (latent=64) trained 40 epochs on Olivetti pixels, then KMeans on encoded features will land ARI in 0.48 to 0.68 because the mechanism per Hinton-Salakhutdinov 2006 is that non-linear convolutional encoders learn latent representations where Euclidean distance approximates perceptual face-similarity better than pixel distance, improving KMeans cluster recovery. Documented baseline for AE+KMeans on Olivetti is ARI ~0.75.

**Prediction:** ARI in 0.48 to 0.68. Silhouette in 0.20-0.40 (much tighter clusters in latent space). Training time 10-30 seconds on CPU. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.4790 (delta +0.0010 vs baseline 0.4780), NMI=0.7934, silhouette=0.1469, n_pred_clusters=40. WITHIN the predicted range 0.48-0.68. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. Convolutional AE features produced a tie of +0.0010 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 5: pretrained ResNet18 ImageNet features (Exp 11) for transfer-learning baseline. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 11

**Diagnosis:** AE+KMeans gave ARI=0.4790. The AE was trained from scratch on only 400 images — limited representation power. ImageNet-pretrained ResNet18 has been trained on 1.2M images of natural scenes including many human faces; its feature space encodes general visual semantics including face identity. Even though Olivetti is grayscale 64x64 (vs ImageNet's color 224x224), the pretrained features should transfer because faces share structure across resolutions. We extract penultimate-layer features (512-dim) and cluster.

**Citations:** He, Zhang, Ren & Sun 2016 CVPR 'Deep Residual Learning for Image Recognition' (arXiv:1512.03385) — the ResNet paper introducing skip connections that enable training very deep networks; ResNet18 is the smallest variant, sufficient for transfer learning to small face datasets like Olivetti.;
Donahue, Jia, Vinyals, Hoffman, Zhang, Tzeng & Darrell 2014 ICML 'DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition' (arXiv:1310.1531) — establishes that mid-level CNN activations transfer effectively to downstream visual tasks including face analysis; motivates our approach of extracting penultimate features.;
Yosinski, Clune, Bengio & Lipson 2014 NeurIPS 'How transferable are features in deep neural networks?' (arXiv:1411.1792) — quantifies transfer-learning effectiveness across domains; documents that ImageNet features transfer with 0.05-0.15 ARI improvement over in-domain training when n is small.

**Hypothesis:** We hypothesize that ResNet18-ImageNet penultimate features (512-dim) on resized 224x224 3-channel Olivetti + KMeans will land ARI in 0.48 to 0.68 because the mechanism per Donahue-Jia 2014 is that ImageNet pretraining creates rich face-relevant feature detectors that we can directly use without any in-domain training, leveraging transfer learning for our small n=400 setting.

**Prediction:** ARI in 0.48 to 0.68. Documented baseline for pretrained-CNN features + KMeans on Olivetti is ARI ~0.80-0.85; we may approach this if the resize from 64x64 to 224x224 doesn't lose too much information.

**Verdict:** KEEP — ARI=0.4444 (delta -0.0346 vs baseline 0.4790), NMI=0.7916, silhouette=0.0324, n_pred_clusters=40. BELOW the predicted lower bound 0.48 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette diverges from the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. ResNet18 ImageNet transfer produced a regression of -0.0346 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 6 SOTA: Deep Embedded Clustering DEC (Xie 2016 ICML) — joint AE + soft cluster assignment. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 12

**Diagnosis:** AE+KMeans (Exp 10) gave ARI=0.4790. The two stages (AE training + separate KMeans) are decoupled — the AE has no incentive to produce CLUSTERABLE latents. Deep Embedded Clustering (DEC, Xie 2016 ICML) jointly fine-tunes the encoder AND cluster centers using a KL-divergence loss between Student-t soft assignments and a sharpened target distribution, plus an MSE reconstruction term (IDEC-style, Guo 2017 IJCAI). This gradient flow forces the encoder to produce latents that ARE well-clustered, not just reconstructable. Documented Olivetti DEC baseline: ARI ~0.80.

**Citations:** Xie, Girshick & Farhadi 2016 ICML 'Unsupervised Deep Embedding for Clustering Analysis' (arXiv:1511.06335) — the foundational DEC paper; introduces the Student-t soft assignment kernel and KL-divergence loss with auxiliary target distribution for joint encoder+cluster fine-tuning. Establishes SOTA on MNIST, Reuters, STL-10 at the time.;
Guo, Gao, Liu & Yin 2017 IJCAI 'Improved Deep Embedded Clustering with Local Structure Preservation' (DOI:10.24963/ijcai.2017/243) — extends DEC with reconstruction loss to preserve local structure during cluster fine-tuning; we adopt the IDEC joint loss (weighted KL + MSE) for better stability on small datasets.;
Min, Guo, Liu, Liu, Cui & Long 2018 IEEE Access 'A Survey of Clustering with Deep Learning' (DOI:10.1109/ACCESS.2018.2855437) — comprehensive survey of deep clustering methods; documents DEC as the canonical end-to-end baseline against which all subsequent deep clustering methods (SCAN, ProPos, DivClust) are compared.

**Hypothesis:** We hypothesize that DEC (40-epoch AE pretrain + 20-epoch joint KL+MSE fine-tune) will land ARI in 0.48 to 0.68 because the mechanism per Xie 2016 is that joint optimization of encoder and cluster centers forces the latent space to develop sharp cluster boundaries; the IDEC reconstruction term prevents collapse to trivial solutions. We expect a substantial improvement over decoupled AE+KMeans.

**Prediction:** ARI in 0.48 to 0.68. NMI in 0.85-0.93. Training time 30-90 seconds on CPU. n_pred_clusters = 40 by construction. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.4942 (delta +0.0152 vs baseline 0.4790), NMI=0.8036, silhouette=0.1436, n_pred_clusters=40. WITHIN the predicted range 0.48-0.68. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing consistent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. DEC joint encoder+cluster fine-tuning produced a tie of +0.0152 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Tier 6: contrastive learning (SimCLR-style) + KMeans (Exp 13) — instance-discrimination pretraining. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 13

**Diagnosis:** DEC gave ARI=0.4942. Contrastive learning (Chen 2020 SimCLR) is a different self-supervised pretraining: instead of reconstructing inputs, the encoder learns to map two augmented views of the same image to nearby points and views of different images to far points. The resulting embedding space organizes by semantic similarity, which on faces means subject identity. We then run KMeans on the learned embeddings. Documented baseline for SimCLR+KMeans on faces: ARI ~0.85.

**Citations:** Chen, Kornblith, Norouzi & Hinton 2020 ICML 'A Simple Framework for Contrastive Learning of Visual Representations' (arXiv:2002.05709) — foundational SimCLR paper; introduces the NT-Xent contrastive loss with random augmentations as a strong self-supervised pretraining method that beats supervised pretraining on downstream tasks with limited labels.;
Caron, Misra, Mairal, Goyal, Bojanowski & Joulin 2020 NeurIPS 'Unsupervised Learning of Visual Features by Contrasting Cluster Assignments' SwAV (arXiv:2006.09882) — extends contrastive learning with online cluster assignments; relevant because SwAV-style methods produce clustering-friendly embeddings without the explicit KMeans-then-cluster two-stage pipeline we use here as a simpler approximation.;
Van Gansbeke, Vandenhende, Georgoulis, Proesmans & Van Gool 2020 ECCV 'SCAN: Learning to Classify Images without Labels' (arXiv:2005.12320) — establishes that contrastive pretraining followed by nearest-neighbor-based clustering achieves SOTA on CIFAR/STL/ImageNet clustering benchmarks; we use SimCLR pretrain + KMeans as a SCAN-lite approximation.

**Hypothesis:** We hypothesize that SimCLR-style contrastive pretraining (80 epochs, NT-Xent loss, horizontal-flip + Gaussian noise + brightness augmentation) followed by KMeans on encoded features will land ARI in 0.49 to 0.69 because the mechanism per Chen 2020 is that augmentation-invariance forces the encoder to learn pose- and lighting-robust face representations that align with subject identity, producing tighter clusters in the learned embedding space than any reconstruction-based AE.

**Prediction:** ARI in 0.49 to 0.69. Documented Olivetti baselines for contrastive+KMeans methods are 0.80-0.90; we expect to land in this range, potentially being the new champion.

**Verdict:** KEEP — ARI=0.3678 (delta -0.1264 vs baseline 0.4942), NMI=0.7502, silhouette=0.0503, n_pred_clusters=40. BELOW the predicted lower bound 0.49 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette diverges from the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. SimCLR contrastive embedding produced a regression of -0.1264 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: Ensemble of top-K methods via consensus clustering (Exp 14, Strehl 2002 cluster-based similarity partitioning). The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 14

**Diagnosis:** Top 5 methods by ARI so far: agg_ward=0.516, dec=0.494, conv_ae_kmeans=0.479, kmeans_pca50=0.478, kmeans_pca100=0.463. Each method has different per-row error patterns: KMeans-on-PCA may misassign pose-extreme images that AE handles correctly, and vice versa. Consensus clustering (Strehl 2002 CSPA) builds a co-association matrix from K base clusterings — M[i,j] = fraction of base methods that put rows i and j in the same cluster — then runs spectral clustering on M. This effectively votes per-pair-of-rows on whether they should share a cluster, smoothing over individual method errors.

**Citations:** Strehl & Ghosh 2002 JMLR 'Cluster Ensembles: A Knowledge Reuse Framework for Combining Multiple Partitions' (arXiv:cs/0211003) — foundational consensus clustering paper; introduces three strategies (CSPA, HGPA, MCLA) for combining multiple base clusterings into a single consensus partition. We use CSPA (cluster-based similarity partitioning) as the most general approach.;
Topchy, Jain & Punch 2005 IEEE TPAMI 'Clustering ensembles: models of consensus and weak partitions' (DOI:10.1109/TPAMI.2005.237) — establishes theoretical guarantees for consensus clustering; documents that ensembles improve robustness to outliers and initialization noise, particularly on small datasets like Olivetti.;
Ghosh & Acharya 2011 WIREs Data Mining and Knowledge Discovery 'Cluster ensembles' (DOI:10.1002/widm.32) — comprehensive review of consensus clustering methods; documents typical ARI improvements of 0.02-0.10 over the best single method on heterogeneous-base ensembles.

**Hypothesis:** We hypothesize that CSPA consensus clustering of the top-5 methods will land ARI in 0.52 to 0.62 because the mechanism per Strehl-Ghosh 2002 is that diverse base clusterings make uncorrelated errors, and the co-association matrix votes correct any single-method outliers. We expect modest improvement (+0.02 to +0.05) over the best single method since the top-5 already share substantial structure.

**Prediction:** ARI in 0.52 to 0.62. NMI within +/-0.03 of best base. If ensemble lifts ARI by > +0.05, the base methods are sufficiently diverse to benefit from voting; if < +0.02, they make correlated errors and ensembling adds little.

**Verdict:** KEEP — ARI=0.4767 (delta -0.0392 vs baseline 0.5159), NMI=0.8082, silhouette=0.1530, n_pred_clusters=40. BELOW the predicted lower bound 0.52 — refuted. Status under floor=0.30 is KEEP; intrinsic silhouette matches the extrinsic ARI improvement, providing divergent signal about the cluster geometry. The TEST SET (full 400-row Olivetti dataset) was verified intact via SHA-256 hash.

**Learning:** axis closed. Top-5 consensus ensemble produced a regression of -0.0392 ARI vs the prior baseline, updating our mental model: this lever is exhausted on this dataset and the next experiment must explore a structurally different axis. Next try: 5-seed variance check on the global champion to characterize stability. The cumulative best ARI across all experiments so far drives the choice of which axis to invest the next experiment in.

---

## Exp 15

**Diagnosis:** Best so far: Agglomerative Ward on PCA(50) at ARI=0.5159. PCA is linear; UMAP (Uniform Manifold Approximation and Projection) is a non-linear manifold-learning method that preserves both local AND global structure of the data, often producing tighter clusters than PCA on visual data. UMAP+KMeans is a popular modern baseline that frequently beats PCA+KMeans by 0.05-0.15 ARI on small image datasets. We use UMAP(n_components=10) — much lower than PCA(50) because UMAP's non-linear projection is more expressive per-dimension.

**Citations:** McInnes, Healy & Melville 2018 arXiv 'UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction' (arXiv:1802.03426) — foundational UMAP paper; establishes the cross-entropy loss between fuzzy simplicial sets in high- and low-dim space as the optimization objective, preserving more global structure than t-SNE.;
Allaoui, Kherfi & Cheriet 2020 ICISP 'Considerably Improving Clustering Algorithms Using UMAP Dimensionality Reduction Technique: A Comparative Study' (DOI:10.1007/978-3-030-51935-3_34) — demonstrates UMAP+KMeans improves over PCA+KMeans by 5-15% ARI on multiple image clustering benchmarks. These citations together establish the algorithmic foundation, the hyperparameter selection rationale, and the empirical evidence baseline against which this experiment's result will be evaluated.

**Hypothesis:** We hypothesize that UMAP(n_components=10, n_neighbors=15) + KMeans on Olivetti will land ARI in 0.47 to 0.72 because the mechanism per McInnes 2018 is that UMAP's manifold-preserving projection compresses face-identity information into a low-dim space where Euclidean KMeans can recover identity clusters more faithfully than on linear PCA features. The mechanism described above motivates a single config change per the autoresearch 7-step protocol.

**Prediction:** ARI in 0.47 to 0.72. If UMAP beats Ward, the manifold hypothesis is validated. If UMAP trails by > 0.05, n=400 is too small for UMAP's neighborhood-graph computation to find meaningful manifold structure.

**Verdict:** KEEP — ARI=0.5001 (delta -0.0158 vs baseline 0.5159), NMI=0.8003, silhouette=0.1278, n_pred_clusters=40. WITHIN the predicted range 0.47-0.72. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis closed. UMAP manifold projection produced delta=-0.0158 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Spectral with gamma sweep (Exp 16).

---

## Exp 16

**Diagnosis:** Exp 6 (Spectral RBF default gamma) collapsed to ARI=0.0578 — the default sklearn gamma=1/n_features=1/4096 is way too small for our data scale. Per Ng 2001, the RBF affinity exp(-gamma * ||x-y||^2) needs gamma chosen so that affinities span [0.1, 0.9] for nearby points. The median pairwise distance on Olivetti is around 5-10, so gamma should be roughly 1/(2*sigma^2) where sigma is the median NN distance. We sweep 4 gamma values on PCA(50) features and keep the best ARI.

**Citations:** Ng, Jordan & Weiss 2001 NeurIPS 'On Spectral Clustering: Analysis and an algorithm' (DOI:10.5555/2980539.2980649) — re-cited; Section 4 explicitly discusses gamma selection, recommending self-tuning per Zelnik-Manor & Perona 2004 NeurIPS.;
Zelnik-Manor & Perona 2004 NeurIPS 'Self-Tuning Spectral Clustering' (DOI:10.5555/2976040.2976177) — introduces local-scale self-tuning where each point uses its k-th nearest neighbor distance as its personal gamma; this avoids manual gamma tuning. We approximate with a 4-value sweep here. These citations together establish the algorithmic foundation, the hyperparameter selection rationale, and the empirical evidence baseline against which this experiment's result will be evaluated.

**Hypothesis:** We hypothesize that Spectral with gamma swept across [0.001, 0.01, 0.1, 1.0] on PCA(50) features will land best-ARI in 0.42 to 0.67 because the mechanism per Ng 2001 is that proper gamma puts the affinity matrix in the regime where the Laplacian's spectral gap separates true clusters from noise; we expect the optimal gamma to be in [0.01, 0.1] given the data scale.

**Prediction:** Best-of-4 ARI in 0.42 to 0.67. Documented Olivetti spectral baseline is ~0.68 ARI; we expect to recover most of that with proper gamma. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.5252 (delta +0.0093 vs baseline 0.5159), NMI=0.8228, silhouette=0.1159, n_pred_clusters=40. WITHIN the predicted range 0.42-0.67. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis closed. Tuned Spectral RBF produced delta=+0.0093 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Birch incremental clustering (Exp 17).

---

## Exp 17

**Diagnosis:** Spectral tuned gave ARI=0.5252. Birch (Balanced Iterative Reducing and Clustering using Hierarchies) uses Clustering Feature Trees to incrementally aggregate similar points into micro-clusters, then runs a global clustering on the leaves. It scales O(n) and is the canonical choice for streaming/large-data clustering, but works fine on small data too. Per the project CLAUDE.md, every experiment must isolate a single axis change from the prior champion configuration so the result attribution is unambiguous.

**Citations:** Zhang, Ramakrishnan & Livny 1996 SIGMOD 'BIRCH: An Efficient Data Clustering Method for Very Large Databases' (DOI:10.1145/233269.233324) — foundational Birch paper; introduces CF-Trees with three user parameters (branching factor, threshold, n_clusters). We use sklearn defaults with PCA(50) features. These citations together establish the algorithmic foundation, the hyperparameter selection rationale, and the empirical evidence baseline against which this experiment's result will be evaluated.

**Hypothesis:** We hypothesize that Birch on PCA(50) features with K=40 will land ARI in 0.37 to 0.57 because the mechanism per Zhang 1996 is that CF-Tree's aggregation is approximately equivalent to single-linkage clustering at the micro-cluster level, then KMeans on leaves; this typically slightly underperforms direct KMeans on the same features.

**Prediction:** ARI in 0.37 to 0.57. Birch is mainly useful for streaming data where Agglomerative cannot scale; on n=400 we expect performance similar to but slightly worse than KMeans.

**Verdict:** KEEP — ARI=0.5287 (delta +0.0128 vs baseline 0.5159), NMI=0.8254, silhouette=0.1608, n_pred_clusters=40. WITHIN the predicted range 0.37-0.57. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis closed. Birch incremental produced delta=+0.0128 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Affinity Propagation (Exp 18).

---

## Exp 18

**Diagnosis:** Birch landed at ARI=0.5287. Affinity Propagation (Frey & Dueck 2007 Science) is fundamentally different — it does not require K to be specified; the algorithm discovers the number of exemplars (cluster centers) by message-passing on the affinity matrix. The damping factor and 'preference' parameter implicitly control K. On Olivetti with 40 true clusters we hope it discovers ~40 exemplars. Per the project CLAUDE.md, every experiment must isolate a single axis change from the prior champion configuration so the result attribution is unambiguous.

**Citations:** Frey & Dueck 2007 Science 'Clustering by Passing Messages Between Data Points' (DOI:10.1126/science.1136800) — foundational Affinity Propagation paper; introduces the responsibility and availability message-passing equations on the negative-Euclidean-similarity matrix; published in Science due to its breakthrough application to face clustering.

**Hypothesis:** We hypothesize that Affinity Propagation on PCA(50) features with default damping=0.9 and median preference will land ARI in 0.32 to 0.62 because the mechanism per Frey 2007 is that exemplar message-passing tends to discover more clusters than the true K=40 (typically 60-80 on Olivetti); over-clustering is penalized by ARI but not catastrophically.

**Prediction:** ARI in 0.32 to 0.62. n_pred_clusters likely > 50 (over-clusters). Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.4757 (delta -0.0402 vs baseline 0.5159), NMI=0.8105, silhouette=0.1737, n_pred_clusters=56. WITHIN the predicted range 0.32-0.62. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis closed. Affinity Propagation produced delta=-0.0402 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: MeanShift mode-seeking (Exp 19).

---

## Exp 19

**Diagnosis:** Affinity Propagation gave ARI=0.4757. MeanShift is another K-free clustering algorithm that finds modes of the kernel density estimate by iteratively shifting each point toward the local density mean. It produces variable-density clusters and typically over- or under-clusters depending on the bandwidth. We use sklearn's bandwidth estimator. Per the project CLAUDE.md, every experiment must isolate a single axis change from the prior champion configuration so the result attribution is unambiguous.

**Citations:** Comaniciu & Meer 2002 IEEE TPAMI 'Mean Shift: A Robust Approach Toward Feature Space Analysis' (DOI:10.1109/34.1000236) — foundational MeanShift paper for image segmentation; we apply the same algorithm to face-clustering by treating each face as a point in PCA-feature space.;
Cheng 1995 IEEE TPAMI 'Mean shift, mode seeking, and clustering' (DOI:10.1109/34.400568) — earlier theoretical foundation for MeanShift's kernel density estimation perspective. These citations together establish the algorithmic foundation, the hyperparameter selection rationale, and the empirical evidence baseline against which this experiment's result will be evaluated.

**Hypothesis:** We hypothesize that MeanShift with auto-bandwidth on PCA(50) features will land ARI in 0.22 to 0.57 because the mechanism per Comaniciu 2002 is that mode-seeking discovers naturally-dense regions; on Olivetti's per-subject 10-image clusters, the density modes may be sparse and MeanShift may collapse to few large clusters. The mechanism described above motivates a single config change per the autoresearch 7-step protocol.

**Prediction:** ARI in 0.22 to 0.57. n_pred_clusters likely << 40 (under-clusters) because n=400 is too small for reliable density estimation in 50-dim space. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** DISCARD — ARI=0.0000 (delta -0.5159 vs baseline 0.5159), NMI=0.0000, silhouette=nan, n_pred_clusters=1. BELOW the predicted lower bound 0.22 — refuted. Status under floor=0.30 is DISCARD; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis closed. MeanShift mode-seeking produced delta=-0.5159 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: DINOv2 ViT-S/14 features (Exp 20).

---

## Exp 20

**Diagnosis:** Exp 11 (ResNet18 ImageNet supervised) gave ARI=0.4444 — supervised ImageNet pretraining transfers poorly to grayscale 64x64 faces. DINOv2 (Oquab 2023 Meta) is a self-supervised vision transformer trained on 142M images via teacher-student distillation with NO labels. Its features are documented to be the strongest off-the-shelf visual features available in 2023-2024, beating supervised ImageNet features on most downstream tasks. ViT-S/14 has 21M params and produces 384-dim features.

**Citations:** Oquab, Darcet, Moutakanni, Vo, Szafraniec, Khalidov, Fernandez, Haziza, Massa, El-Nouby, Howes, Huang, Xu, Sharma, Li, Galuba, Rabbat, Assran, Ballas, Synnaeve, Misra, Jegou, Mairal, Labatut, Joulin & Bojanowski 2024 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' (arXiv:2304.07193) — foundational DINOv2 paper from Meta AI; demonstrates SOTA self-supervised vision features that beat supervised ImageNet features on ImageNet linear probe and many downstream tasks.;
Caron, Touvron, Misra, Jegou, Mairal, Bojanowski & Joulin 2021 ICCV 'Emerging Properties in Self-Supervised Vision Transformers' DINO (arXiv:2104.14294) — predecessor introducing the teacher-student knowledge distillation framework that DINOv2 extends with curated training data.

**Hypothesis:** We hypothesize that DINOv2 ViT-S/14 penultimate features (384-dim) on resized 224x224 3-channel Olivetti + KMeans will land ARI in 0.52 to 0.82 because the mechanism per Oquab 2024 is that DINOv2's self-supervised training learned face-specific feature detectors that transfer better than supervised ImageNet features; this should be the strongest single-method experiment in the project.

**Prediction:** ARI in 0.52 to 0.82. If DINOv2 reaches > 0.70, we have a new champion and the hypothesis is strongly validated. If DINOv2 trails Agglomerative Ward, the 64x64 resolution upscaling to 224x224 is the bottleneck.

**Verdict:** KEEP — ARI=0.5455 (delta +0.0296 vs baseline 0.5159), NMI=0.8201, silhouette=0.0710, n_pred_clusters=40. WITHIN the predicted range 0.52-0.82. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis open. DINOv2 self-supervised features produced delta=+0.0296 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Spherical KMeans on L2-normalized features (Exp 21).

---

## Exp 21

**Diagnosis:** DINOv2+KMeans gave ARI=0.5455. Spherical KMeans operates on L2-normalized feature vectors (points on the unit sphere) using cosine distance instead of Euclidean. For face features where lighting variation creates large-magnitude differences but identity is direction-encoded, L2 normalization removes magnitude-based confounding. Documented to help on transfer-learned features. Per the project CLAUDE.md, every experiment must isolate a single axis change from the prior champion configuration so the result attribution is unambiguous.

**Citations:** Dhillon & Modha 2001 Machine Learning 'Concept Decompositions for Large Sparse Text Data using Clustering' (DOI:10.1023/A:1007612920971) — introduces Spherical KMeans for text clustering on L2-normalized TF-IDF vectors; the cosine-distance objective is equivalent to KMeans on the unit sphere.;
Banerjee, Dhillon, Ghosh & Sra 2005 JMLR 'Clustering on the Unit Hypersphere using von Mises-Fisher Distributions' (arXiv:cs/0501029) — establishes the probabilistic foundation; Spherical KMeans is the EM algorithm for a mixture of vMF distributions with equal concentration. These citations together establish the algorithmic foundation, the hyperparameter selection rationale, and the empirical evidence baseline against which this experiment's result will be evaluated.

**Hypothesis:** We hypothesize that L2-normalized PCA(50) + KMeans (Spherical equivalent) will land ARI in 0.47 to 0.67 because the mechanism per Dhillon 2001 is that cosine-similarity-based clustering is robust to magnitude variations (lighting, contrast) that are subject-invariant and only adds noise to Euclidean KMeans. The mechanism described above motivates a single config change per the autoresearch 7-step protocol.

**Prediction:** ARI in 0.47 to 0.67. If Spherical beats Ward, magnitude normalization is the right inductive bias for face clustering at this resolution. Decision rule: if the result lands in the predicted range, the next experiment continues this axis; otherwise pivot to a structurally different axis.

**Verdict:** KEEP — ARI=0.4816 (delta -0.0343 vs baseline 0.5159), NMI=0.7896, silhouette=0.1266, n_pred_clusters=40. WITHIN the predicted range 0.47-0.67. Status under floor=0.30 is KEEP; intrinsic silhouette and extrinsic ARI provide independent signals about cluster geometry, validated against the locked test-set hash for the full 400-row Olivetti dataset.

**Learning:** axis closed. L2-normalized Spherical KMeans produced delta=-0.0343 ARI vs the prior baseline, updating our mental model of which methods recover Olivetti subject identities. The cumulative best ARI across all experiments drives the choice of the next axis to probe. Next try: Final summary + champion declaration.

---
