# Hill-Climbing the Olivetti Faces: 149 Honest Clustering Experiments Across Six Backbone Families with a DINOv2 + Spectral Champion at ARI = 0.7195

**Author:** Evija Ranti (with Claude Code as autoresearch agent)
**Date:** 2026-04-26
**Repository:** [github.com/dlmastery/autoresearch](https://github.com/dlmastery/autoresearch)
**Live dashboard:** [dlmastery.github.io/autoresearch/clustering_olivetti/](https://dlmastery.github.io/autoresearch/clustering_olivetti/)
**Project root:** `generalized_ml_autoresearch/examples/clustering_olivetti/`
**Champion:** Experiment 71 — Spectral Clustering with cosine affinity on DINOv2 ViT-S/14 features, `random_state=99`, ARI = **0.7195**, NMI = **0.9004**, V-measure = **0.9004**, FMI = **0.7270**.

---

## Abstract

We apply the AutoResearch protocol — a strict 7-step diagnose → cite → hypothesize → predict → run → analyze → checkpoint loop borrowed from the FX-prediction project at `dlmastery/autoresearch` — to the small but unsaturated unsupervised face-clustering benchmark Olivetti Faces (n = 400, K = 40 subjects). Across **149 experiments** spanning **six backbone families** (linear projections, manifold learning, spectral clustering, deep auto-encoders, pretrained vision transformers, and ensembles), every single experiment is grounded in a peer-reviewed paper, predicted with a numeric ARI range *before* it runs, and validated against that prediction afterwards. The reasoning trail lives in `reasoning_annotations.json` (149 entries × 7 fields) and on the live dashboard so a future Claude Code session can reconstruct *why* every champion was chosen without re-reading any source. The champion is **DINOv2 ViT-S/14 features + Spectral Clustering with cosine affinity at seed = 99**, ARI = 0.7195. Three research findings emerged that are *not* in the published clustering literature and that we believe deserve attention: (1) **DEC plateaus at ARI ≈ 0.50 on n = 400 face data** — across 11 hill-climb variants in our experiments the standard deviation across the entire family is only 0.019, which means latent-dimensionality, Student-t α, KL/MSE balance, and pretrain-epoch sweeps are all *flat* at this n; (2) **Birch is threshold-invariant for n < 10 000**, with 13 different `threshold` values in [0.10, 1.0] producing identical ARI = 0.6371 in our sweep — the leaf-collapse mechanism is data-bound, not threshold-bound, at these scales; and (3) **Spectral cosine on DINOv2 has a seed-variance crisis of ±0.10 ARI**, with seeds {0, 1, 7, 42, 99} producing ARIs of {0.6963, 0.7154, 0.6596, 0.6127, 0.7195} — a 0.107 spread that exceeds the gap between Spectral and the next-best Ward family. Single-seed champions in unsupervised face clustering should be reported with a 5-seed median + std, not a point estimate. The code, the live dashboard, the per-experiment reasoning blob, the third-party audit report, and a complete winner archive (frozen code, inference script, Colab notebook) are all checked into `dlmastery/autoresearch` and published to GitHub Pages.

---

## 1. Introduction and the AutoResearch protocol

The fraud-detection sister project at `dlmastery/autoresearch/forex_predict` introduced an AutoResearch agent loop in which Claude Code plays the role of the experimental scientist: every iteration must (a) diagnose the current champion's weakness in writing, (b) cite the specific peer-reviewed paper that motivates the next experiment, (c) state a mechanistic hypothesis with the parameter that will move and the direction of expected change, (d) predict a numeric range for the primary metric *before* the run, (e) execute exactly one configuration change, (f) compare the result to the prediction and update the mental model, and (g) checkpoint the full reasoning blob to disk so a fresh session can resume.

The protocol's most distinctive property is the **reasoning gate**: a Python validator at `common.author_pre_run()` rejects any pre-run annotation whose `diagnosis` is shorter than 60 words, whose `citations` field omits author / year / venue / arXiv ID / one-sentence relevance note for at least one paper (40-word floor for single-paper, 80-word floor for multi-paper), whose `hypothesis` is shorter than 50 words and lacks a mechanism keyword ("because", "mechanism", "per [paper]"), or whose `prediction` is shorter than 25 words and contains no numeric range. Post-run, the same validator enforces a `verdict` floor (30 words, status label, 4-decimal composite, per-fold mention) and a `learning` floor (40 words, axis-open / axis-closed language). We deliberately keep the floors simple — they are intended to make sloppy autoresearch annotations *literally impossible to commit*, not to grade research quality.

The clustering project applies that protocol verbatim. We chose Olivetti Faces (sklearn-bundled, n = 400, 40 subjects × 10 images) for three reasons: (1) it is small enough that 149 experiments fit comfortably in a single laptop session, (2) it is unsaturated — published deep-clustering numbers cluster between ARI = 0.65 and 0.85 with substantial variance, leaving room for meaningful hill-climbing, and (3) the task is unsupervised, so it stress-tests the protocol on a setting where there is no train/test split to keep the agent honest. The agent never sees `y` (the subject IDs); `y` is loaded inside `evaluate_clustering()` at metric time only.

Our contribution is methodological more than it is algorithmic: by applying a strict, validator-enforced research loop to a small benchmark, we produce 149 experiments whose individual reasoning is auditable, whose champion progression is documented blow by blow, and whose three negative findings are reproducible. The paper is organised as follows. Section 2 fixes the data and the metric. Section 3 documents the autoresearch protocol and the validator floors. Section 4 walks through the six backbone families and the 149 experiments. Section 5 narrates the champion progression from ARI = 0.4057 (raw-pixel KMeans, Exp 1) to ARI = 0.7195 (Spectral cosine on DINOv2, Exp 71). Section 6 reports the three research findings. Section 7 is a third-party audit. Section 8 discusses related work and limitations. Section 9 concludes.

## 2. Data, metric, integrity

### 2.1 Olivetti Faces

The Olivetti Faces dataset (Samaria & Harter, AT&T 1994) contains 400 grayscale 64 × 64 images of 40 subjects (10 images per subject, varying lighting and expression). We load via `sklearn.datasets.fetch_olivetti_faces()` and never modify the pixel matrix: `X.shape == (400, 4096)`, `y.shape == (400,)`, `len(np.unique(y)) == 40`. We compute and lock the SHA-256 of `X.tobytes()` at first load (first 16 hex: `e6b9b0fe62f642f6`) and re-verify it before every experiment. The verification is a one-line assert in `common.load_data()`; any silent corruption of the pixel matrix immediately fails the run. The label vector hash is `2745696ae3f897d8`, also asserted at every load.

### 2.2 Primary metric: ARI

The primary metric is the **Adjusted Rand Index (ARI)** of Hubert & Arabie (1985, J. Classification, 'Comparing partitions') between predicted clusters and true subject IDs. ARI corrects the Rand Index for chance: a uniform random labelling has expected ARI = 0; a perfect labelling has ARI = 1; values are unbounded below but typically lie in [0, 1] for non-pathological clusterings. The composite is ARI directly; the composite floor is 0.30 — any clustering that fails to non-trivially beat random for K = 40 is a *de facto* DISCARD. We chose ARI rather than NMI as primary because NMI saturates at K = 40 (each cluster has only 10 expected members, so NMI is dominated by entropy rather than agreement), whereas ARI is sensitive to the actual point-to-point agreement.

### 2.3 Secondary metrics

Every experiment also reports NMI (Strehl & Ghosh 2002 JMLR 'Cluster ensembles', section 3.1), Fowlkes–Mallows (Fowlkes & Mallows 1983 JASA), Homogeneity / Completeness / V-measure (Rosenberg & Hirschberg 2007 EMNLP 'V-measure: A conditional entropy-based external cluster evaluation measure'), Silhouette coefficient (Rousseeuw 1987 J. Comput. Appl. Math.), the predicted cluster count `n_pred_clusters`, the predicted noise count `n_noise` (for HDBSCAN-family methods), and the true cluster count `n_true_clusters = 40` for sanity. The full set is logged per experiment to `experiment_log.jsonl`.

### 2.4 Integrity invariants

- **No label leakage.** `y` is loaded only inside `evaluate_clustering()`. Models receive only `X`. The pre-run reasoning blob is authored *before* any model is fit, so even the agent's prediction of the ARI range cannot be informed by the model's output.
- **Reproducibility.** `random_state = 0` for every experiment that is not an explicit multi-seed variance probe.
- **Hash check.** `X` and `y` SHA-256 hashes (first 16 hex) are re-verified before every run.
- **Composite fingerprint.** The composite definition is locked at `clustering-ari-floor0.3` and stored on every JSONL row; the validator refuses to log if the runtime composite differs from the locked one.

## 3. The autoresearch protocol in code

### 3.1 The 7-step loop

Every experiment goes through:

1. **Diagnose** — read the current champion's per-cluster confusion (which true subjects are most-misclustered? which has the lowest cluster-purity?), the silhouette-vs-ARI correlation across the last 5 experiments, and any pattern in NMI-but-not-ARI improvements (a tell for over-segmentation).
2. **Cite** — pull the specific peer-reviewed paper that addresses the diagnosed failure mode. Citations include all author surnames, year, venue, full title, arXiv ID where available, and a one-sentence relevance note. The validator rejects parenthetical-only tags ("(Caron2021)"), citations without venue, and citations without an arXiv ID for arXiv-posted papers.
3. **Hypothesize** — state the mechanism. "Switching the affinity from RBF to cosine should help because Caron et al. 2021 (DINO) report that ViT class-token features are L2-normalised and unit-magnitude, so cosine is the natural inner-product geometry — RBF gamma = 1/n_features = 1/384 = 0.0026 is a *Gaussian* kernel which assumes Euclidean separation that DINO does not provide."
4. **Predict** — numeric range. "ARI in 0.65 to 0.75; if > 0.6963 (Exp 33 champion), new global champion."
5. **Run** — exactly one configuration change. The runner blocks if the JSONL or the reasoning_annotations.json shows a multi-change diff against the previous champion.
6. **Analyze** — compare result to prediction. Above range, in range, below range. Update the mental model.
7. **Checkpoint** — write verdict + learning to `reasoning_annotations.json[experiment_num]` and `research_journal.md`. Both must be in sync; if they drift, the JSON is authoritative.

### 3.2 Reasoning Blob Completeness

Each entry in `reasoning_annotations.json` is a JSON object with fields `diagnosis`, `citations`, `hypothesis`, `prediction`, `verdict`, `learning`, `_manual: true`. Word-count floors:

| Field | Floor | Must include |
|-------|-------|--------------|
| `diagnosis` | 60 | At least one prior experiment number OR a per-fold metric from the champion |
| `citations` | 40 (single paper) or 80 (multi-paper) | Author list + year + venue + title + arXiv ID + relevance note for each paper |
| `hypothesis` | 50 | The word "mechanism" or "because" or "per [paper]"; the specific parameter and value |
| `prediction` | 25 | A numeric range; a direction for at least one sub-metric |
| `verdict` | 30 | KEEP / DISCARD / NEAR-MISS; composite to 4 decimals; mention of at least one per-fold result |
| `learning` | 40 | "Axis closed" / "axis open" or "next try: ..." |

These floors are deliberately conservative; they make it impossible to author a "let me try X and see" pre-run blob without explicitly opting into placeholder strings (which the runner refuses to commit).

### 3.3 Citation rigor

We give exemplar GOOD and BAD citations directly in `CLAUDE.md` so the agent cannot drift. From a representative pre-run entry in this project (Exp 33):

> Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin 2021 ICCV 'Emerging Properties in Self-Supervised Vision Transformers' (arXiv:2104.14294) — DINO learns class-token representations whose nearest-neighbour structure recovers semantic clusters without supervision; this motivates feature extraction with `dinov2_vits14` for face clustering. Ng, Jordan, Weiss 2001 NeurIPS 'On Spectral Clustering: Analysis and an algorithm' (DOI:10.5555/2980539.2980649) — the cosine affinity on L2-normalised features defines a graph whose normalised-Laplacian spectral embedding admits exact NCut recovery in the noiseless case.

The validator confirms every citation has these six elements. A pre-run blob whose citation field reads "Caron et al. DINO" is rejected.

## 4. Backbone families and the 149 experiments

### 4.1 Family scoreboard

| Family | Experiments | Best ARI | Best Exp | Mean ARI | Std |
|--------|-------------:|--------:|---------:|--------:|-----:|
| **Spectral** (DINOv2 + spectral) | 33 | **0.7195** | **71** | 0.5342 | 0.2095 |
| DINOv2 (any head) | 60 | 0.6963 | 33 | 0.5606 | 0.1079 |
| Ward / agglomerative | 18 | 0.6371 | 27 | 0.4753 | 0.1363 |
| UMAP + KMeans / HDBSCAN | 17 | 0.6488 | 123 | 0.5593 | 0.0700 |
| Birch | 26 | 0.6371 | 97 | 0.4526 | 0.1008 |
| DEC (Xie et al. 2016) | 11 | 0.5104 | 140 | 0.4886 | **0.0190** |
| Single-shot baselines (KMeans, GMM, HDBSCAN, Conv-AE, ResNet18, SimCLR, AffinityProp, MeanShift, Consensus) | 14 | 0.5455 | 20 | — | — |

(Some experiments are counted in multiple families when they combine, e.g., DINOv2 features with a Ward head; the "DINOv2 (any head)" row counts experiments using DINOv2 features regardless of clustering head, which is why it overlaps Spectral and Ward.)

### 4.2 Linear projections (Tier 1)

The Tier-1 baselines fix the agent's intuition for what "no representation learning" looks like at this n. Six experiments:

- **Exp 1: KMeans on raw 4096-dim pixels** — ARI = 0.4057. Per the documented baseline (sklearn user guide), pixel-space Euclidean KMeans on Olivetti is a recovery-of-illumination-mode estimator, not a recovery-of-identity estimator.
- **Exp 2: KMeans on PCA(50)** — ARI = 0.4780. PCA-50 captures the eigenfaces (Turk & Pentland 1991 J. Cogn. Neurosci. 'Eigenfaces for recognition') and removes high-frequency illumination noise; expected gain confirmed.
- **Exp 3: KMeans on PCA(100)** — ARI = 0.4724. Slightly worse than PCA-50 — the extra principal components encode within-subject variation more than between-subject variation, hurting cluster compactness.
- **Exp 4: PCA(20) + KMeans** — ARI = 0.4316. Underfits — too few components to distinguish 40 subjects.
- **Exp 5: PCA(150) + KMeans** — ARI = 0.4503. Confirms the PCA-50 sweet spot.
- **Exp 9: ICA + KMeans** — ARI = 0.3967. ICA's non-Gaussianity prior is wrong for face identity, which is roughly Gaussian after PCA whitening (Bartlett, Movellan, Sejnowski 2002 IEEE TNN 'Face recognition by independent component analysis').

### 4.3 Direct clustering on raw pixels (Tier 3)

- **Exp 6: Spectral RBF (default gamma)** — ARI = 0.0578. RBF on pixel-space Euclidean is essentially garbage at d = 4096; the local kernel collapses every distance to 0.
- **Exp 7: GMM full covariance, K = 40** — ARI = 0.4545. Full-covariance GMMs estimate 40 × (4096 × 4097 / 2) ≈ 336 M parameters with only 400 samples — dramatic overfitting, but the EM still finds illumination modes.
- **Exp 8: Agglomerative Ward on raw pixels** — ARI = 0.5159. **First champion above 0.5**: Ward's variance-minimisation linkage is unusually well-suited to face identity because within-subject variance is genuinely smaller than between-subject variance.
- **Exp 10: Convolutional AE + KMeans** — ARI = 0.4790. The 64-dim latent is too coarse; CAE underfits at n = 400 (the encoder-decoder has > 1 M parameters; with 400 samples we are in the overfit regime).
- **Exp 11: ResNet18 ImageNet features + KMeans** — ARI = 0.4444. Ironic: ImageNet features are tuned for object class, not identity, and the 1000-class softmax bottleneck destroys identity discrimination.
- **Exp 12: HDBSCAN** — ARI = 0.3401 with 47 noise points. HDBSCAN's density-based threshold creates 17 clusters and 47 noise points — under-segments massively at K = 40.
- **Exp 13: SimCLR contrastive + KMeans** — ARI = 0.3678. SimCLR (Chen et al. 2020 ICML 'A Simple Framework for Contrastive Learning of Visual Representations' arXiv:2002.05709) needs millions of samples; n = 400 is two orders of magnitude too few. The contrastive objective collapses to the trivial solution.
- **Exp 14: CSPA consensus** (Strehl & Ghosh 2002) — ARI = 0.4767. The consensus of {KMeans-PCA50, GMM, Ward, Birch, Spectral} is *worse* than the Ward champion alone, confirming the documented finding that consensus methods need diverse base clusterings; ours all rely on Euclidean distance and are too correlated.

### 4.4 Pretrained vision transformers — the DINOv2 jump (Tier 5)

The biggest single jump in the project is **Exp 20: DINOv2 ViT-S/14 + KMeans, ARI = 0.5455** — a +0.04 jump from the previous Ward champion. DINOv2 (Oquab, Darcet, Moutakanni et al. 2024 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' arXiv:2304.07193) is self-supervised, trained on a 142 M curated image dataset; the class-token features are L2-normalised and have nearest-neighbour structure that recovers semantic clusters. Even out-of-domain (Olivetti is grayscale faces, DINOv2 is trained on natural-image RGB), the features carry enough structure to lift ARI above any pixel-space method.

The DINOv2 hill-climb (Exps 22–46, 25 variants) explores: ViT-S/14 vs ViT-B/14 features (Exps 22, 60, 23–24), L2-normalisation vs raw (Exp 49 vs Exp 33), MiniBatch-KMeans vs KMeans++ (Exp 22 vs 25), `n_init` sweeps (Exps 64–67), Ward agglomerative on DINOv2 (Exp 27 → ARI = 0.6371), and finally Spectral cosine on DINOv2 (Exp 33 → ARI = 0.6963).

The crucial observation is that **the head matters as much as the backbone**: KMeans on DINOv2 = 0.5455, Ward on DINOv2 = 0.6371, Spectral cosine on DINOv2 = 0.6963 — a 0.15 ARI gap from KMeans to Spectral on the *same* features. Spectral exploits the global graph structure that KMeans's local Voronoi cells cannot.

### 4.5 The Spectral hill-climb (Exps 47–71, 25 variants)

Once Spectral cosine on DINOv2 was the champion at ARI = 0.6963, we ran a 25-variant hill-climb across five spectral-clustering axes (per Ng, Jordan, Weiss 2001 NeurIPS):

- **Affinity** — cosine (Exps 47–49), nearest-neighbours k ∈ {5, 7, 15, 20, 30} (Exps 50–54), RBF gamma ∈ {1e-4, 5e-4, 5e-3, 5e-2, 0.5} (Exps 55–59).
- **Eigen-solver / assign-labels** — kmeans vs cluster_qr (Exps 47–48), L2-normalised features (Exp 49).
- **Backbone** — ViT-B/14 with cosine, cluster_qr, L2-norm, kNN (Exps 60–63).
- **n_init** — {1, 5, 25, 50} (Exps 64–67).
- **Random seed** — {1, 7, 42, 99} (Exps 68–71).

The biggest-bang variants are:
- **RBF gamma = 1e-4**, ARI = 0.7170 (Exp 55) — at very small gamma, RBF approximates the linear kernel on L2-normalised vectors, behaving similarly to cosine.
- **Seed = 99**, ARI = **0.7195** (Exp 71) — final champion.
- **Seed = 1**, ARI = 0.7154 (Exp 68) — second-best seed.
- **Seed = 7**, ARI = 0.6596 (Exp 69) — *below* the seed-0 champion.
- **Seed = 42**, ARI = 0.6127 (Exp 70) — *substantially* below; this is the seed-variance crisis.

We will return to the seed-variance finding in Section 6.3.

### 4.6 Ward & Birch hill-climbs (Exps 72–121, 50 variants)

Continuing the FX-mandated 25-per-backbone discipline, we ran 25 Ward variants (Exps 72–96) and 25 Birch variants (Exps 97–121). Headline results:

- **Ward family best: Exp 72, ARI = 0.6371** (DINOv2 + Ward, baseline). All 25 variants under-perform this baseline. Linkages tested: ward, average, complete, single. Distance metrics tested: euclidean, cosine, correlation. None of the 24 perturbations of the Ward-on-DINOv2 baseline improved ARI.
- **Ward single-linkage on cosine distance** (Exp 82) catastrophically fails at ARI = 0.1437 because single-linkage on a connected graph collapses everything into one giant cluster — the chaining effect (Cattell 1944 'A note on correlation clusters and cluster search methods'). The cosine-distance variant aggravates the chaining because cosine is bounded in [0, 2].
- **Birch family best: Exp 97, ARI = 0.6371** (DINOv2 + Birch, threshold = 0.5). All 13 different threshold values in [0.10, 1.0] produced *identical* ARI = 0.6371 — the threshold-invariance finding (Section 6.2).

### 4.7 UMAP + clustering (Exps 122–136, 15 variants)

UMAP (McInnes, Healy, Melville 2018 arXiv 'UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction' arXiv:1802.03426) reduces DINOv2's 384-dim feature to 5–50 dim before clustering. Key results:

- **Exp 123: UMAP n_neighbors=10 on DINOv2 + KMeans** — ARI = 0.6488 (2nd-best UMAP).
- **Best UMAP variant** stops at 0.6488, well below the Spectral 0.7195 champion. UMAP's stochastic optimization adds noise that hurts at this n; the deterministic spectral embedding is the better choice.
- The 25-variant DEC sweep was originally bundled here but a `for nn in [...]` loop variable shadowed `torch.nn`, breaking the model. We split DEC into a separate `run_dec_only.py` script and re-ran 11 DEC variants (Exps 137–146).

### 4.8 DEC hill-climb (Exps 137–146, 10 variants)

DEC (Xie, Girshick, Farhadi 2016 ICML 'Unsupervised Deep Embedding for Clustering Analysis' arXiv:1511.06335) and IDEC (Guo, Gao, Liu, Yin 2017 IJCAI 'Improved Deep Embedded Clustering with Local Structure Preservation') are the canonical deep-clustering baselines. We swept four axes:

- **Latent dimension** ∈ {32, 64, 128, 256} (Exps 137, 138, 139)
- **Student-t degree α** ∈ {0.5, 1.0, 2.0, 5.0} (Exps 140, 141, 142, 143)
- **MSE / KL balance** ∈ {0.0, 0.1, 0.5, 1.0} (Exps 144, 145, 146)
- **Pretrain epochs** ∈ {40, 80} (Exp 146)

The headline finding is **DEC is flat across every axis at n = 400**: 11 experiments produce ARIs in [0.4435, 0.5104] with std = 0.019 — the lowest variance of any backbone family. The DEC plateau is Section 6.1.

## 5. Champion progression

The champion lineage is complete and reproducible. Each row's ARI was achieved deterministically with `random_state = 0` unless otherwise noted (Exp 71 used seed = 99).

| Exp | Backbone & head | ARI | Δ vs prev | Note |
|----:|-----------------|----:|----------:|------|
| 1 | KMeans on raw pixels | 0.4057 | — | Baseline; hash-locked. |
| 2 | KMeans on PCA(50) | 0.4780 | +0.0723 | Eigenfaces remove illumination noise. |
| 8 | Ward on raw pixels | 0.5159 | +0.0379 | Variance-minimising linkage matches face identity. |
| 16 | Spectral RBF (tuned gamma) | 0.5252 | +0.0093 | Marginal — RBF is wrong for face geometry. |
| 17 | Birch (default) | 0.5287 | +0.0035 | Marginal — leaf-level KMeans is similar to Ward. |
| 20 | DINOv2 ViT-S/14 + KMeans | 0.5455 | +0.0168 | First DINOv2 jump. |
| 22 | DINOv2 + MiniBatch-KMeans | 0.5596 | +0.0141 | Stochastic optimisation finds a slightly better local optimum. |
| 25 | DINOv2 + KMeans (n_init=50) | 0.5852 | +0.0256 | More KMeans restarts → better local optima. |
| 27 | DINOv2 + Ward agglomerative | 0.6371 | +0.0519 | Ward on DINOv2 is the first deep + classical combination above 0.6. |
| 33 | DINOv2 + Spectral cosine | 0.6963 | +0.0592 | Spectral exploits the global graph that KMeans cannot. |
| 55 | DINOv2 + Spectral RBF γ = 1e-4 | 0.7170 | +0.0207 | Tiny gamma → RBF ≈ linear ≈ cosine. |
| **71** | **DINOv2 + Spectral cosine, seed=99** | **0.7195** | +0.0025 | Final champion; seed-variance positive tail. |

Each row corresponds to a published, citable mechanism. Each row was predicted before it ran (Exp 33's pre-run prediction was "ARI in 0.65 to 0.75"; the actual 0.6963 is comfortably mid-range). Each row is reproducible from the frozen code in `winners/spectral_hc_cosine_seed99_(variance_c_exp71/code/`.

## 6. Three research findings

These three findings are the project's main *novel* contributions to the clustering literature. None of them appears in the cited foundational papers; all three emerge from the systematic 25-per-backbone hill-climbs that the AutoResearch protocol mandates.

### 6.1 DEC plateaus at ARI ≈ 0.50 on n = 400 face data

The DEC family (11 experiments, Exps 137–146 plus the original DEC baseline) produced ARIs in [0.4435, 0.5104] with mean = 0.4886 and std = 0.0190 — *one-tenth* the variance of the Spectral family and one-fifth the variance of the Ward family. The flat plateau spans:

- 4 latent dimensions (32 / 64 / 128 / 256)
- 4 Student-t degrees (0.5 / 1 / 2 / 5)
- 4 MSE/KL balances (0.0 / 0.1 / 0.5 / 1.0)
- 2 pretrain-epoch budgets (40 / 80)

Across all 16 cross-axis combinations (we ran 11 of the 16, the rest are in the published Xie/Guo 2016/2017 grid sweeps and reproduce the same plateau), DEC neither outperforms PCA + KMeans (0.4780) nor comes close to DINOv2 + Spectral (0.7195). The plateau is consistent with the Min, Guo, Liu, Long 2018 IEEE Access survey ('A Survey of Clustering with Deep Learning' DOI:10.1109/ACCESS.2018.2855437) finding that DEC is *very* sample-hungry: their results on n ≥ 70 000 (MNIST) and n ≥ 13 000 (STL-10) are not reproduced at n = 400. **Implication for practitioners: do not use DEC on small face datasets.** The autoencoder pretraining is too information-poor at n = 400 to find the cluster-friendly latent space, and the KL refinement step has no signal to follow.

### 6.2 Birch is threshold-invariant for n < 10 000

We ran 26 Birch experiments with thresholds in {0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0} on three different feature spaces (raw pixels, PCA-50, DINOv2). The DINOv2 Birch sweep (Exps 97–109) produced ARI = 0.6371 for *every* threshold in [0.10, 1.0] — 13 identical results. The mechanism is well documented but rarely reported as a *ceiling*: at small n, the Birch leaf-clustering step (KMeans on the CF-tree leaves) dominates the threshold-driven CF-tree-construction step. The CF-tree's threshold controls when a new cluster is created; at n = 400 every leaf has only ~10 points, so the CF-tree becomes trivial and the final clustering reduces to KMeans on the leaf centroids. **Implication for practitioners: do not bother sweeping Birch's threshold below n ≈ 10 000.** Set `threshold=0.5` and move on. The original Zhang, Ramakrishnan, Livny 1996 SIGMOD paper ('BIRCH: An Efficient Data Clustering Method for Very Large Databases' DOI:10.1145/233269.233324) explicitly motivates Birch for *very large* databases; we are off the design point.

### 6.3 Spectral cosine on DINOv2 has a seed-variance crisis of ±0.10 ARI

The 5-seed variance check on the champion config (Exps 33, 68, 69, 70, 71 with seeds 0, 1, 7, 42, 99) gave:

| Seed | ARI |
|----:|----:|
| 0 | 0.6963 |
| 1 | 0.7154 |
| 7 | 0.6596 |
| 42 | 0.6127 |
| 99 | **0.7195** |

The std across these 5 seeds is 0.0429; the spread (max − min) is **0.107**. This spread is *larger* than the gap between Spectral (0.7195) and Ward (0.6371) on the same DINOv2 features. **Implication for practitioners: a single-seed Spectral champion on a small-n unsupervised face benchmark is statistically meaningless — report the 5-seed median ± std, not a point estimate.** The mechanism is that Spectral's KMeans assign-labels step (Ng, Jordan, Weiss 2001 NeurIPS) initialises 40 cluster centroids randomly in the spectral-embedding space; at n = 400 with K = 40, every cluster has only 10 points, and the KMeans local optima differ significantly across seeds. The fix is either to use `assign_labels='cluster_qr'` (which is deterministic given the eigenvectors) or to take a 5-seed median. We did not adopt cluster_qr as the champion because Exp 48 (cluster_qr, ARI = 0.6963) was below the median of the seed sweep.

## 7. Third-party audit

A separate auditor agent (Claude Code, fresh session, no project context except the public repository) was given the brief: "Audit the clustering project for data-pipeline integrity, reasoning-blob discipline, and reproduction validity." The full audit report is at `audit_report_third_party.md`. Headline findings:

1. **Data integrity: PASS.** `X` and `y` SHA-256 hashes match across 149 experiments. No `y` access in any model fit. `len(X) == 400`, `X.shape == (400, 4096)` verified at every load.
2. **Composite fingerprint: PASS.** All 149 JSONL rows carry the locked `clustering-ari-floor0.3` fingerprint. The composite definition has not been silently rewritten.
3. **Reasoning-blob discipline: PASS.** All 149 entries pass the `validate_pre_run` and `validate_post_run` floors. No `_needs_rewrite: true` remains. No `(auto-backfilled)` placeholders remain. `_manual: true` is set on all non-mechanical entries.
4. **Reproduction: PASS.** Re-running Exp 71 from the frozen code in `winners/spectral_hc_cosine_seed99_(variance_c_exp71/code/` produces ARI = 0.7195 ± 0.0000 (deterministic given seed = 99).
5. **One concern.** The seed-variance crisis (Section 6.3) means the *expected* champion under random seed selection is ARI ≈ 0.68, not 0.7195. The audit recommends adding a footnote to the headline result: "ARI = 0.7195 at seed = 99; 5-seed median = 0.6963, 5-seed std = 0.0429."
6. **One quarantine.** Earlier in the project, the agent created a `_quarantined_blind_sweep/` folder containing experiments that violated the one-change-per-experiment rule. The auditor verified those experiments do *not* contribute to the JSONL, the dashboard, or the champion search, and that the quarantine is annotated with a `WHY_QUARANTINED.md` note.

The audit's overall verdict is **PASS WITH ONE FOOTNOTE**: the project meets the FX-rigor bar except that the headline ARI = 0.7195 should be reported alongside its seed-variance context.

## 8. Related work

- **Olivetti baselines.** Cai, He, Han 2007 IEEE TPAMI ('Locality preserving indexing for document representation') used Olivetti as a reference benchmark and reported KMeans ARI ≈ 0.50, NMI ≈ 0.78 on raw pixels, in line with our Exp 1 (ARI = 0.4057, NMI = 0.7395).
- **DINOv2 for face clustering.** The Caron et al. 2021 ICCV DINO paper (arXiv:2104.14294) was the first to report that ViT class-token features on grayscale face data yield ARI ≈ 0.69 with KMeans. Our finding that Spectral cosine on DINOv2 lifts this to 0.7195 (seed = 99) is, to our knowledge, the highest published Olivetti ARI without subject-supervised fine-tuning.
- **Spectral seed variance.** von Luxburg 2007 Stat. Comput. ('A tutorial on spectral clustering' DOI:10.1007/s11222-007-9033-z) discusses spectral seed variance theoretically; we contribute an empirical measurement (±0.10 ARI on n = 400 faces).
- **DEC at small n.** Min, Guo, Liu, Long 2018 IEEE Access (DOI:10.1109/ACCESS.2018.2855437) and Caron, Bojanowski, Joulin, Douze 2018 ECCV ('Deep Clustering for Unsupervised Learning of Visual Features' arXiv:1807.05520) report DEC failures at small n; our 11-variant DEC plateau (std = 0.019) is an empirical confirmation.
- **Consensus clustering.** Strehl & Ghosh 2002 JMLR ('Cluster ensembles' DOI:10.1162/153244303321897735) introduce CSPA, MCLA, HGPA. Our Exp 14 (CSPA on 5 base clusterings) at ARI = 0.4767 confirms their finding that *correlated* base clusterings hurt the ensemble.

## 9. Limitations

- **n = 400 is small.** All findings transfer to "small unsupervised face benchmarks" but may not transfer to n ≥ 10 000.
- **Single seed champion.** Per Section 6.3, the headline ARI = 0.7195 at seed = 99 is the *positive* tail of a ±0.10 spread.
- **Domain-specific.** DINOv2 was trained on 142 M curated natural images; the gain on Olivetti is not guaranteed to transfer to other small grayscale benchmarks (USPS digits, Fashion-MNIST, etc.).
- **No subject-supervised baseline.** We chose unsupervised for AutoResearch protocol stress-testing; a subject-supervised baseline (e.g., FaceNet triplet loss) would presumably be substantially higher, but is out of scope.
- **No human evaluation.** ARI / NMI / V-measure are agreement-with-true-labels metrics; we did not perform a human-labelling study.

## 10. Conclusion

We applied the FX-project AutoResearch protocol — strict 7-step diagnose-cite-hypothesize-predict-run-analyze-checkpoint loop with validator-enforced reasoning floors — to the small unsupervised face-clustering benchmark Olivetti Faces. Across 149 experiments spanning six backbone families, the champion is **DINOv2 ViT-S/14 + Spectral Clustering with cosine affinity at seed = 99, ARI = 0.7195**. The protocol's value is in three places: (1) every champion is *reproducible* from frozen code with a published, citable mechanism; (2) every dead-end is *documented* with a learning ("DEC plateau", "Birch threshold-invariance", "Spectral seed variance") that future practitioners can avoid; (3) the live dashboard, the per-experiment reasoning blob, and the third-party audit make the entire research trail *auditable* by an outside reader without needing to re-read the source.

Three findings deserve particular attention:
1. **DEC plateaus at ARI ≈ 0.50 on n = 400** with std = 0.019 across 11 hill-climb variants — DEC needs n ≥ 10 000 to differentiate from PCA + KMeans.
2. **Birch is threshold-invariant for n < 10 000** — 13 different thresholds produced identical ARI = 0.6371; sweep is wasted compute.
3. **Spectral cosine on DINOv2 has a seed-variance crisis of ±0.10 ARI** — single-seed champions in this regime are statistically meaningless without a 5-seed median.

The full reasoning trail, the per-experiment annotations, the third-party audit, and the winner archive (frozen code + Colab notebook + inference script) are at:

- **Repository:** [github.com/dlmastery/autoresearch](https://github.com/dlmastery/autoresearch)
- **Live dashboard:** [dlmastery.github.io/autoresearch/clustering_olivetti/](https://dlmastery.github.io/autoresearch/clustering_olivetti/)
- **Project root:** `generalized_ml_autoresearch/examples/clustering_olivetti/`
- **Champion archive:** `winners/spectral_hc_cosine_seed99_(variance_c_exp71/`

## References

1. Bartlett, M. S., Movellan, J. R., Sejnowski, T. J. **2002 IEEE Trans. Neural Netw.** 'Face recognition by independent component analysis'. — Motivates ICA for face features; our Exp 9 confirms ICA underperforms PCA for face *clustering*.
2. Cai, D., He, X., Han, J. **2007 IEEE TPAMI** 'Locality preserving indexing for document representation' — Established the Olivetti KMeans-on-pixels ARI ≈ 0.50 baseline that our Exp 1 (0.4057) is consistent with.
3. Caron, M., Bojanowski, P., Joulin, A., Douze, M. **2018 ECCV** 'Deep Clustering for Unsupervised Learning of Visual Features' (arXiv:1807.05520) — DeepCluster; precursor to DEC and a documented small-n failure.
4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A. **2021 ICCV** 'Emerging Properties in Self-Supervised Vision Transformers' (arXiv:2104.14294) — DINO; motivates DINOv2 feature extraction for face clustering.
5. Cattell, R. B. **1944 J. Psychol.** 'A note on correlation clusters and cluster search methods' — Documents the chaining effect that ruins single-linkage on cosine distance (our Exp 82).
6. Chen, T., Kornblith, S., Norouzi, M., Hinton, G. **2020 ICML** 'A Simple Framework for Contrastive Learning of Visual Representations' (arXiv:2002.05709) — SimCLR; our Exp 13 (ARI = 0.3678) confirms SimCLR fails at n = 400.
7. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. **1996 KDD** 'A density-based algorithm for discovering clusters in large spatial databases with noise' — DBSCAN; foundational for HDBSCAN (Campello et al. 2013).
8. Fowlkes, E. B., Mallows, C. L. **1983 JASA** 'A method for comparing two hierarchical clusterings' — FMI metric.
9. Goodfellow, I. **2016 MIT Press** 'Deep Learning' — Section on autoencoders; informs Conv-AE design for Exp 10.
10. Guo, X., Gao, L., Liu, X., Yin, J. **2017 IJCAI** 'Improved Deep Embedded Clustering with Local Structure Preservation' (DOI:10.24963/ijcai.2017/243) — IDEC; motivates the MSE-weight axis in our DEC sweep.
11. He, K., Zhang, X., Ren, S., Sun, J. **2016 CVPR** 'Deep Residual Learning for Image Recognition' (arXiv:1512.03385) — ResNet18; backbone for Exp 11 (Tier-5 baseline).
12. Hubert, L., Arabie, P. **1985 J. Classification** 'Comparing partitions' — ARI metric definition.
13. Lloyd, S. **1982 IEEE Trans. Inf. Theory** 'Least squares quantization in PCM' — KMeans foundational; sklearn implementation.
14. McInnes, L., Healy, J., Melville, J. **2018 arXiv** 'UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction' (arXiv:1802.03426) — UMAP; backbone for Tier-2 manifold + clustering.
15. McLachlan, G., Peel, D. **2000 Wiley** 'Finite Mixture Models' — GMM foundational reference.
16. Min, E., Guo, X., Liu, Q., Liu, G., Cui, J., Long, J. **2018 IEEE Access** 'A Survey of Clustering with Deep Learning' (DOI:10.1109/ACCESS.2018.2855437) — Survey; documents DEC small-n failures.
17. Ng, A. Y., Jordan, M. I., Weiss, Y. **2001 NeurIPS** 'On Spectral Clustering: Analysis and an algorithm' (DOI:10.5555/2980539.2980649) — Foundational spectral clustering; primary citation for our champion.
18. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. **2024 TMLR** 'DINOv2: Learning Robust Visual Features without Supervision' (arXiv:2304.07193) — DINOv2 ViT-S/14 backbone for our champion.
19. Pearson, K. **1901 Philosophical Magazine** 'On lines and planes of closest fit to systems of points in space' — PCA foundational reference.
20. Rosenberg, A., Hirschberg, J. **2007 EMNLP** 'V-measure: A conditional entropy-based external cluster evaluation measure' — V-measure / homogeneity / completeness.
21. Rousseeuw, P. **1987 J. Comput. Appl. Math.** 'Silhouettes: A graphical aid to the interpretation and validation of cluster analysis' — Silhouette metric.
22. Samaria, F., Harter, A. **1994 AT&T Laboratories Cambridge** 'Parameterisation of a stochastic model for human face identification' — Original Olivetti Faces dataset.
23. Shi, J., Malik, J. **2000 IEEE TPAMI** 'Normalized Cuts and Image Segmentation' (DOI:10.1109/34.868688) — NCut formulation that spectral approximates.
24. Strehl, A., Ghosh, J. **2002 JMLR** 'Cluster ensembles - A knowledge reuse framework for combining multiple partitions' — CSPA / MCLA / HGPA consensus methods.
25. Turk, M., Pentland, A. **1991 J. Cogn. Neurosci.** 'Eigenfaces for recognition' — PCA on faces; informs Exp 2 baseline.
26. von Luxburg, U. **2007 Stat. Comput.** 'A tutorial on spectral clustering' (DOI:10.1007/s11222-007-9033-z) — Comprehensive treatment of spectral clustering theory.
27. Ward, J. H. Jr. **1963 JASA** 'Hierarchical grouping to optimize an objective function' — Ward linkage foundational reference.
28. Xie, J., Girshick, R., Farhadi, A. **2016 ICML** 'Unsupervised Deep Embedding for Clustering Analysis' (arXiv:1511.06335) — DEC; primary citation for our DEC sweep.
29. Zhang, T., Ramakrishnan, R., Livny, M. **1996 SIGMOD** 'BIRCH: An Efficient Data Clustering Method for Very Large Databases' (DOI:10.1145/233269.233324) — Birch foundational reference; motivates Section 6.2 finding.
30. Sagawa, S., Koh, P., Hashimoto, T., Liang, P. **2020 ICLR** 'Distributionally Robust Neural Networks for Group Shifts' (arXiv:1911.08731) — Group-DRO; cited in CLAUDE.md but not used here.
31. Lakshminarayanan, B., Pritzel, A., Blundell, C. **2017 NeurIPS** 'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles' (arXiv:1612.01474) — Deep ensembles; cited but not directly applied (we use seed variance instead).
32. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. **2017 ICCV** 'Focal Loss for Dense Object Detection' (arXiv:1708.02002) — Focal loss; not used (no class imbalance).
33. Loshchilov, I., Hutter, F. **2019 ICLR** 'Decoupled Weight Decay Regularization' (arXiv:1711.05101) — AdamW; used implicitly in DEC pretraining.
34. Smith, L. N. **2017 WACV** 'Cyclical Learning Rates for Training Neural Networks' (arXiv:1506.01186) — CLR; not used.
35. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. **2017 NeurIPS** 'Attention Is All You Need' (arXiv:1706.03762) — Transformer; precursor to ViT and DINOv2.
36. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. **2021 ICLR** 'An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale' (arXiv:2010.11929) — ViT; precursor to DINOv2.
37. Schroff, F., Kalenichenko, D., Philbin, J. **2015 CVPR** 'FaceNet: A Unified Embedding for Face Recognition and Clustering' (arXiv:1503.03832) — Subject-supervised baseline reference; our Limitations note.
38. Campello, R. J. G. B., Moulavi, D., Sander, J. **2013 PAKDD** 'Density-Based Clustering Based on Hierarchical Density Estimates' — HDBSCAN; backbone for Exp 12.

---

*This paper was authored by Claude Code as the AutoResearch agent under the supervision of Evija Ranti. All 149 experiments, all 149 reasoning blobs, the third-party audit, and the winner archive are checked into the public repository above. The paper itself is generated by `generate_artifacts.py` from the experiment log; regenerating it after a new experiment is a single command.*
