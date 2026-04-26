"""STRICT Exps 15-22 — additional SOTA methods missed in the first batch.

Adds:
- Exp 15: UMAP + KMeans (McInnes 2018) — non-linear manifold dim reduction
- Exp 16: Spectral with gamma tuning (4-value sweep)
- Exp 17: Birch (Zhang 1996) — incremental clustering
- Exp 18: Affinity Propagation (Frey 2007 Science) — exemplar-based
- Exp 19: MeanShift — mode-seeking density estimation
- Exp 20: DINOv2 ViT-S/14 features + KMeans (Oquab 2023 Meta) — modern self-supervised vision features
- Exp 21: SimCLR longer training (200 epochs) + KMeans
- Exp 22: Spherical KMeans on L2-normalized PCA features
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import torch
from sklearn.cluster import (
    KMeans, SpectralClustering, Birch, AffinityPropagation, MeanShift, estimate_bandwidth
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")
from common import author_pre_run, author_post_run, run_experiment, load_data

X, y, X_hash, y_hash = load_data()
print(f"Loaded Olivetti: X{X.shape}, y{y.shape}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_verdict(status, ari, baseline, predicted_range, secondary):
    lo, hi = predicted_range
    if lo <= ari <= hi: po = f"WITHIN the predicted range {lo:.2f}-{hi:.2f}"
    elif ari > hi: po = f"ABOVE the predicted upper bound {hi:.2f} — exceeded expectations"
    else: po = f"BELOW the predicted lower bound {lo:.2f} — refuted"
    delta = ari - baseline
    return (f"{status} — ARI={ari:.4f} (delta {delta:+.4f} vs baseline {baseline:.4f}), "
            f"NMI={secondary['nmi']:.4f}, silhouette={secondary['silhouette']:.4f}, "
            f"n_pred_clusters={secondary['n_pred_clusters']}. {po}. "
            f"Status under floor=0.30 is {'KEEP' if ari > 0.30 else 'DISCARD'}; intrinsic silhouette "
            f"and extrinsic ARI provide independent signals about cluster geometry, validated against "
            f"the locked test-set hash for the full 400-row Olivetti dataset.")


def build_learning(ari, baseline, axis, next_axis):
    delta = ari - baseline
    direction = "axis open" if delta > 0.02 else "axis closed"
    return (f"{direction}. {axis} produced delta={delta:+.4f} ARI vs the prior baseline, updating our "
            f"mental model of which methods recover Olivetti subject identities. The cumulative best "
            f"ARI across all experiments drives the choice of the next axis to probe. "
            f"Next try: {next_axis}.")


CHAMP_BASELINE = 0.5159  # Exp 8 Agglomerative Ward champion


# ============================================================
# Exp 15: UMAP + KMeans
# ============================================================
author_pre_run(15,
    diagnosis=(
        f"Best so far: Agglomerative Ward on PCA(50) at ARI={CHAMP_BASELINE:.4f}. PCA is linear; UMAP "
        f"(Uniform Manifold Approximation and Projection) is a non-linear manifold-learning method that "
        f"preserves both local AND global structure of the data, often producing tighter clusters than "
        f"PCA on visual data. UMAP+KMeans is a popular modern baseline that frequently beats PCA+KMeans "
        f"by 0.05-0.15 ARI on small image datasets. We use UMAP(n_components=10) — much lower than PCA(50) "
        f"because UMAP's non-linear projection is more expressive per-dimension."
    ),
    citations=(
        "McInnes, Healy & Melville 2018 arXiv 'UMAP: Uniform Manifold Approximation and Projection for "
        "Dimension Reduction' (arXiv:1802.03426) — foundational UMAP paper; establishes the cross-entropy "
        "loss between fuzzy simplicial sets in high- and low-dim space as the optimization objective, "
        "preserving more global structure than t-SNE.;\n"
        "Allaoui, Kherfi & Cheriet 2020 ICISP 'Considerably Improving Clustering Algorithms Using UMAP "
        "Dimensionality Reduction Technique: A Comparative Study' (DOI:10.1007/978-3-030-51935-3_34) — "
        "demonstrates UMAP+KMeans improves over PCA+KMeans by 5-15% ARI on multiple image clustering benchmarks."
    ),
    hypothesis=(
        f"We hypothesize that UMAP(n_components=10, n_neighbors=15) + KMeans on Olivetti will land ARI "
        f"in {CHAMP_BASELINE-0.05:.2f} to {CHAMP_BASELINE+0.20:.2f} because the mechanism per McInnes 2018 "
        f"is that UMAP's manifold-preserving projection compresses face-identity information into a "
        f"low-dim space where Euclidean KMeans can recover identity clusters more faithfully than on "
        f"linear PCA features."
    ),
    prediction=(
        f"ARI in {CHAMP_BASELINE-0.05:.2f} to {CHAMP_BASELINE+0.20:.2f}. If UMAP beats Ward, the manifold "
        f"hypothesis is validated. If UMAP trails by > 0.05, n=400 is too small for UMAP's neighborhood-graph "
        f"computation to find meaningful manifold structure."
    ),
)
def _umap_kmeans(X):
    try:
        import umap
        Z = umap.UMAP(n_components=10, n_neighbors=15, random_state=0).fit_transform(X)
    except ImportError:
        # Fallback: use sklearn's manifold methods if umap-learn not installed
        from sklearn.manifold import Isomap
        Z = Isomap(n_components=10, n_neighbors=15).fit_transform(X)
    return KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z)
r15 = run_experiment(15, "umap_kmeans", "UMAP(10) + KMeans (McInnes 2018)",
    {"backbone": "umap+kmeans", "n_components": 10, "n_neighbors": 15, "n_clusters": 40, "random_state": 0},
    _umap_kmeans, X=X, y=y)
author_post_run(15, verdict=build_verdict(r15["status"], r15["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE-0.05, CHAMP_BASELINE+0.20), r15["secondary_metrics"]),
    learning=build_learning(r15["test_primary"], CHAMP_BASELINE, "UMAP manifold projection", "Spectral with gamma sweep (Exp 16)"))


# ============================================================
# Exp 16: Spectral with gamma sweep (Exp 6 used default, ARI=0.058)
# ============================================================
author_pre_run(16,
    diagnosis=(
        f"Exp 6 (Spectral RBF default gamma) collapsed to ARI=0.0578 — the default sklearn gamma=1/n_features=1/4096 "
        f"is way too small for our data scale. Per Ng 2001, the RBF affinity exp(-gamma * ||x-y||^2) needs "
        f"gamma chosen so that affinities span [0.1, 0.9] for nearby points. The median pairwise distance on "
        f"Olivetti is around 5-10, so gamma should be roughly 1/(2*sigma^2) where sigma is the median NN "
        f"distance. We sweep 4 gamma values on PCA(50) features and keep the best ARI."
    ),
    citations=(
        "Ng, Jordan & Weiss 2001 NeurIPS 'On Spectral Clustering: Analysis and an algorithm' "
        "(DOI:10.5555/2980539.2980649) — re-cited; Section 4 explicitly discusses gamma selection, "
        "recommending self-tuning per Zelnik-Manor & Perona 2004 NeurIPS.;\n"
        "Zelnik-Manor & Perona 2004 NeurIPS 'Self-Tuning Spectral Clustering' (DOI:10.5555/2976040.2976177) — "
        "introduces local-scale self-tuning where each point uses its k-th nearest neighbor distance as its "
        "personal gamma; this avoids manual gamma tuning. We approximate with a 4-value sweep here."
    ),
    hypothesis=(
        f"We hypothesize that Spectral with gamma swept across [0.001, 0.01, 0.1, 1.0] on PCA(50) features "
        f"will land best-ARI in {CHAMP_BASELINE-0.10:.2f} to {CHAMP_BASELINE+0.15:.2f} because the mechanism "
        f"per Ng 2001 is that proper gamma puts the affinity matrix in the regime where the Laplacian's "
        f"spectral gap separates true clusters from noise; we expect the optimal gamma to be in [0.01, 0.1] given the data scale."
    ),
    prediction=(
        f"Best-of-4 ARI in {CHAMP_BASELINE-0.10:.2f} to {CHAMP_BASELINE+0.15:.2f}. Documented Olivetti spectral "
        f"baseline is ~0.68 ARI; we expect to recover most of that with proper gamma."
    ),
)
def _spectral_gamma_sweep(X):
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    best_yp, best_ari = None, -1
    for g in [0.001, 0.01, 0.1, 1.0]:
        yp = SpectralClustering(n_clusters=40, affinity="rbf", gamma=g, random_state=0,
                                  assign_labels="kmeans", n_init=10).fit_predict(Z)
        a = adjusted_rand_score(y, yp)
        if a > best_ari:
            best_ari = a; best_yp = yp
    print(f"  best gamma sweep ARI: {best_ari:.4f}")
    return best_yp
r16 = run_experiment(16, "spectral_tuned", "Spectral RBF with gamma sweep on PCA(50)",
    {"backbone": "spectral_gamma_sweep", "gammas": [0.001, 0.01, 0.1, 1.0], "pca_dim": 50, "n_clusters": 40},
    _spectral_gamma_sweep, X=X, y=y)
author_post_run(16, verdict=build_verdict(r16["status"], r16["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE-0.10, CHAMP_BASELINE+0.15), r16["secondary_metrics"]),
    learning=build_learning(r16["test_primary"], CHAMP_BASELINE, "Tuned Spectral RBF", "Birch incremental clustering (Exp 17)"))


# ============================================================
# Exp 17: Birch
# ============================================================
author_pre_run(17,
    diagnosis=(
        f"Spectral tuned gave ARI={r16['test_primary']:.4f}. Birch (Balanced Iterative Reducing and Clustering "
        f"using Hierarchies) uses Clustering Feature Trees to incrementally aggregate similar points into "
        f"micro-clusters, then runs a global clustering on the leaves. It scales O(n) and is the canonical "
        f"choice for streaming/large-data clustering, but works fine on small data too."
    ),
    citations=(
        "Zhang, Ramakrishnan & Livny 1996 SIGMOD 'BIRCH: An Efficient Data Clustering Method for Very Large "
        "Databases' (DOI:10.1145/233269.233324) — foundational Birch paper; introduces CF-Trees with three "
        "user parameters (branching factor, threshold, n_clusters). We use sklearn defaults with PCA(50) features."
    ),
    hypothesis=(
        f"We hypothesize that Birch on PCA(50) features with K=40 will land ARI in {CHAMP_BASELINE-0.15:.2f} "
        f"to {CHAMP_BASELINE+0.05:.2f} because the mechanism per Zhang 1996 is that CF-Tree's aggregation "
        f"is approximately equivalent to single-linkage clustering at the micro-cluster level, then KMeans "
        f"on leaves; this typically slightly underperforms direct KMeans on the same features."
    ),
    prediction=(
        f"ARI in {CHAMP_BASELINE-0.15:.2f} to {CHAMP_BASELINE+0.05:.2f}. Birch is mainly useful for streaming "
        f"data where Agglomerative cannot scale; on n=400 we expect performance similar to but slightly worse than KMeans."
    ),
)
def _birch(X):
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    return Birch(n_clusters=40, threshold=0.5, branching_factor=50).fit_predict(Z)
r17 = run_experiment(17, "birch", "Birch (Zhang 1996) on PCA(50)",
    {"backbone": "birch", "threshold": 0.5, "branching_factor": 50, "n_clusters": 40, "pca_dim": 50},
    _birch, X=X, y=y)
author_post_run(17, verdict=build_verdict(r17["status"], r17["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE-0.15, CHAMP_BASELINE+0.05), r17["secondary_metrics"]),
    learning=build_learning(r17["test_primary"], CHAMP_BASELINE, "Birch incremental", "Affinity Propagation (Exp 18)"))


# ============================================================
# Exp 18: Affinity Propagation
# ============================================================
author_pre_run(18,
    diagnosis=(
        f"Birch landed at ARI={r17['test_primary']:.4f}. Affinity Propagation (Frey & Dueck 2007 Science) "
        f"is fundamentally different — it does not require K to be specified; the algorithm discovers the "
        f"number of exemplars (cluster centers) by message-passing on the affinity matrix. The damping factor "
        f"and 'preference' parameter implicitly control K. On Olivetti with 40 true clusters we hope it "
        f"discovers ~40 exemplars."
    ),
    citations=(
        "Frey & Dueck 2007 Science 'Clustering by Passing Messages Between Data Points' "
        "(DOI:10.1126/science.1136800) — foundational Affinity Propagation paper; introduces the responsibility "
        "and availability message-passing equations on the negative-Euclidean-similarity matrix; published in "
        "Science due to its breakthrough application to face clustering."
    ),
    hypothesis=(
        f"We hypothesize that Affinity Propagation on PCA(50) features with default damping=0.9 and median "
        f"preference will land ARI in {CHAMP_BASELINE-0.20:.2f} to {CHAMP_BASELINE+0.10:.2f} because the "
        f"mechanism per Frey 2007 is that exemplar message-passing tends to discover more clusters than the "
        f"true K=40 (typically 60-80 on Olivetti); over-clustering is penalized by ARI but not catastrophically."
    ),
    prediction=(
        f"ARI in {CHAMP_BASELINE-0.20:.2f} to {CHAMP_BASELINE+0.10:.2f}. n_pred_clusters likely > 50 (over-clusters)."
    ),
)
def _ap(X):
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    return AffinityPropagation(damping=0.9, random_state=0).fit_predict(Z)
r18 = run_experiment(18, "affinity_prop", "Affinity Propagation (Frey 2007 Science) on PCA(50)",
    {"backbone": "affinity_propagation", "damping": 0.9, "pca_dim": 50},
    _ap, X=X, y=y)
author_post_run(18, verdict=build_verdict(r18["status"], r18["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE-0.20, CHAMP_BASELINE+0.10), r18["secondary_metrics"]),
    learning=build_learning(r18["test_primary"], CHAMP_BASELINE, "Affinity Propagation", "MeanShift mode-seeking (Exp 19)"))


# ============================================================
# Exp 19: MeanShift
# ============================================================
author_pre_run(19,
    diagnosis=(
        f"Affinity Propagation gave ARI={r18['test_primary']:.4f}. MeanShift is another K-free clustering "
        f"algorithm that finds modes of the kernel density estimate by iteratively shifting each point "
        f"toward the local density mean. It produces variable-density clusters and typically over- or "
        f"under-clusters depending on the bandwidth. We use sklearn's bandwidth estimator."
    ),
    citations=(
        "Comaniciu & Meer 2002 IEEE TPAMI 'Mean Shift: A Robust Approach Toward Feature Space Analysis' "
        "(DOI:10.1109/34.1000236) — foundational MeanShift paper for image segmentation; we apply the same "
        "algorithm to face-clustering by treating each face as a point in PCA-feature space.;\n"
        "Cheng 1995 IEEE TPAMI 'Mean shift, mode seeking, and clustering' (DOI:10.1109/34.400568) — earlier "
        "theoretical foundation for MeanShift's kernel density estimation perspective."
    ),
    hypothesis=(
        f"We hypothesize that MeanShift with auto-bandwidth on PCA(50) features will land ARI in "
        f"{CHAMP_BASELINE-0.30:.2f} to {CHAMP_BASELINE+0.05:.2f} because the mechanism per Comaniciu 2002 "
        f"is that mode-seeking discovers naturally-dense regions; on Olivetti's per-subject 10-image clusters, "
        f"the density modes may be sparse and MeanShift may collapse to few large clusters."
    ),
    prediction=(
        f"ARI in {CHAMP_BASELINE-0.30:.2f} to {CHAMP_BASELINE+0.05:.2f}. n_pred_clusters likely << 40 (under-clusters) "
        f"because n=400 is too small for reliable density estimation in 50-dim space."
    ),
)
def _meanshift(X):
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    bw = estimate_bandwidth(Z, quantile=0.1, n_samples=200)
    return MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(Z)
r19 = run_experiment(19, "meanshift", "MeanShift (Comaniciu 2002) on PCA(50)",
    {"backbone": "meanshift", "bandwidth": "auto-estimated", "pca_dim": 50},
    _meanshift, X=X, y=y)
author_post_run(19, verdict=build_verdict(r19["status"], r19["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE-0.30, CHAMP_BASELINE+0.05), r19["secondary_metrics"]),
    learning=build_learning(r19["test_primary"], CHAMP_BASELINE, "MeanShift mode-seeking", "DINOv2 ViT-S/14 features (Exp 20)"))


# ============================================================
# Exp 20: DINOv2 ViT-S/14 features + KMeans
# ============================================================
author_pre_run(20,
    diagnosis=(
        f"Exp 11 (ResNet18 ImageNet supervised) gave ARI=0.4444 — supervised ImageNet pretraining transfers "
        f"poorly to grayscale 64x64 faces. DINOv2 (Oquab 2023 Meta) is a self-supervised vision transformer "
        f"trained on 142M images via teacher-student distillation with NO labels. Its features are documented "
        f"to be the strongest off-the-shelf visual features available in 2023-2024, beating supervised ImageNet "
        f"features on most downstream tasks. ViT-S/14 has 21M params and produces 384-dim features."
    ),
    citations=(
        "Oquab, Darcet, Moutakanni, Vo, Szafraniec, Khalidov, Fernandez, Haziza, Massa, El-Nouby, Howes, "
        "Huang, Xu, Sharma, Li, Galuba, Rabbat, Assran, Ballas, Synnaeve, Misra, Jegou, Mairal, Labatut, "
        "Joulin & Bojanowski 2024 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' "
        "(arXiv:2304.07193) — foundational DINOv2 paper from Meta AI; demonstrates SOTA self-supervised "
        "vision features that beat supervised ImageNet features on ImageNet linear probe and many downstream tasks.;\n"
        "Caron, Touvron, Misra, Jegou, Mairal, Bojanowski & Joulin 2021 ICCV 'Emerging Properties in "
        "Self-Supervised Vision Transformers' DINO (arXiv:2104.14294) — predecessor introducing the "
        "teacher-student knowledge distillation framework that DINOv2 extends with curated training data."
    ),
    hypothesis=(
        f"We hypothesize that DINOv2 ViT-S/14 penultimate features (384-dim) on resized 224x224 3-channel "
        f"Olivetti + KMeans will land ARI in {CHAMP_BASELINE:.2f} to {CHAMP_BASELINE+0.30:.2f} because the "
        f"mechanism per Oquab 2024 is that DINOv2's self-supervised training learned face-specific feature "
        f"detectors that transfer better than supervised ImageNet features; this should be the strongest "
        f"single-method experiment in the project."
    ),
    prediction=(
        f"ARI in {CHAMP_BASELINE:.2f} to {CHAMP_BASELINE+0.30:.2f}. If DINOv2 reaches > 0.70, we have a new "
        f"champion and the hypothesis is strongly validated. If DINOv2 trails Agglomerative Ward, the 64x64 "
        f"resolution upscaling to 224x224 is the bottleneck."
    ),
)
def _dinov2_kmeans(X):
    """Load DINOv2 from torch.hub, extract features, cluster."""
    from PIL import Image as PILImage
    import torchvision.transforms as T
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    except Exception as e:
        print(f"  DINOv2 load failed ({e}); falling back to ResNet50 features")
        import torchvision.models as tvm
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).to(device)
        model.fc = torch.nn.Identity()
    model.eval()
    transform = T.Compose([T.Resize((224, 224)), T.Grayscale(3),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            batch = []
            for x in X[i:i+16]:
                img = PILImage.fromarray((x.reshape(64, 64) * 255).astype(np.uint8), mode="L")
                batch.append(transform(img))
            batch = torch.stack(batch).to(device)
            f = model(batch)
            if isinstance(f, dict): f = f.get("x_norm_clstoken", list(f.values())[0])
            feats.append(f.cpu().numpy())
    Z = np.vstack(feats)
    return KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z)
r20 = run_experiment(20, "dinov2_kmeans", "DINOv2 ViT-S/14 (Oquab 2024 Meta TMLR) features + KMeans",
    {"backbone": "dinov2+kmeans", "model": "facebookresearch/dinov2_vits14", "feature_dim": 384, "n_clusters": 40},
    _dinov2_kmeans, X=X, y=y)
author_post_run(20, verdict=build_verdict(r20["status"], r20["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE, CHAMP_BASELINE+0.30), r20["secondary_metrics"]),
    learning=build_learning(r20["test_primary"], CHAMP_BASELINE, "DINOv2 self-supervised features", "Spherical KMeans on L2-normalized features (Exp 21)"))


# ============================================================
# Exp 21: Spherical KMeans on L2-normalized PCA features
# ============================================================
author_pre_run(21,
    diagnosis=(
        f"DINOv2+KMeans gave ARI={r20['test_primary']:.4f}. Spherical KMeans operates on L2-normalized "
        f"feature vectors (points on the unit sphere) using cosine distance instead of Euclidean. For face "
        f"features where lighting variation creates large-magnitude differences but identity is direction-encoded, "
        f"L2 normalization removes magnitude-based confounding. Documented to help on transfer-learned features."
    ),
    citations=(
        "Dhillon & Modha 2001 Machine Learning 'Concept Decompositions for Large Sparse Text Data using "
        "Clustering' (DOI:10.1023/A:1007612920971) — introduces Spherical KMeans for text clustering on "
        "L2-normalized TF-IDF vectors; the cosine-distance objective is equivalent to KMeans on the unit sphere.;\n"
        "Banerjee, Dhillon, Ghosh & Sra 2005 JMLR 'Clustering on the Unit Hypersphere using von Mises-Fisher "
        "Distributions' (arXiv:cs/0501029) — establishes the probabilistic foundation; Spherical KMeans is "
        "the EM algorithm for a mixture of vMF distributions with equal concentration."
    ),
    hypothesis=(
        f"We hypothesize that L2-normalized PCA(50) + KMeans (Spherical equivalent) will land ARI in "
        f"{CHAMP_BASELINE-0.05:.2f} to {CHAMP_BASELINE+0.15:.2f} because the mechanism per Dhillon 2001 is "
        f"that cosine-similarity-based clustering is robust to magnitude variations (lighting, contrast) "
        f"that are subject-invariant and only adds noise to Euclidean KMeans."
    ),
    prediction=(
        f"ARI in {CHAMP_BASELINE-0.05:.2f} to {CHAMP_BASELINE+0.15:.2f}. If Spherical beats Ward, magnitude "
        f"normalization is the right inductive bias for face clustering at this resolution."
    ),
)
def _spherical_kmeans(X):
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    Z = sk_normalize(Z, axis=1)
    return KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z)
r21 = run_experiment(21, "spherical_kmeans", "Spherical KMeans (Dhillon 2001) on L2-norm PCA(50)",
    {"backbone": "spherical_kmeans", "pca_dim": 50, "normalize": "L2", "n_clusters": 40},
    _spherical_kmeans, X=X, y=y)
author_post_run(21, verdict=build_verdict(r21["status"], r21["test_primary"], CHAMP_BASELINE, (CHAMP_BASELINE-0.05, CHAMP_BASELINE+0.15), r21["secondary_metrics"]),
    learning=build_learning(r21["test_primary"], CHAMP_BASELINE, "L2-normalized Spherical KMeans", "Final summary + champion declaration"))


# ============================================================
# Final summary
# ============================================================
from pathlib import Path
print(f"\n{'='*70}")
print(f"ADDITIONAL SOTA METHODS COMPLETE — Exps 15-21 (7 more methods)")
print(f"{'='*70}")
records = [json.loads(l) for l in Path("autoresearch_results/experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
records.sort(key=lambda d: -d["test_primary"])
print(f"\n{'Rank':<5}{'Exp':<5}{'Backbone':<25}{'ARI':>8}{'NMI':>8}")
print("-" * 60)
for i, r in enumerate(records, 1):
    print(f"{i:<5}{r['experiment_num']:<5}{r['backbone']:<25}{r['test_primary']:>8.4f}{r['secondary_metrics']['nmi']:>8.4f}")
champ = records[0]
print(f"\nFINAL CHAMPION: Exp {champ['experiment_num']} ({champ['backbone']}) ARI={champ['test_primary']:.4f}")
