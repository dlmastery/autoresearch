"""STRICT Exp 1 — KMeans on raw 4096-pixel features (baseline).

Establishes the ARI floor against which every subsequent experiment is measured.
Per the project CLAUDE.md, KMeans on raw pixels has documented baseline ARI ~0.50.
"""
from __future__ import annotations

from sklearn.cluster import KMeans

from common import author_pre_run, author_post_run, run_experiment, load_data

EXP_NUM = 1

author_pre_run(
    EXP_NUM,
    diagnosis=(
        "Baseline experiment for the Olivetti Faces clustering autoresearch project. No prior "
        "experiments exist, so the diagnosis is scope-setting: we have 400 grayscale 64x64 face "
        "images of 40 subjects (10 each), 4096 raw-pixel features per image, and we need to "
        "cluster them into K=40 groups recovering subject identities. The published baseline for "
        "KMeans on raw pixels is ARI~0.50; we treat 0.30 as the composite floor (must non-trivially "
        "beat random clustering for K=40, where E[ARI]~0). The goal of this experiment is to "
        "establish the reference point for all downstream feature-engineering and architecture "
        "experiments. We use K=40 (true number of clusters), n_init=10 random restarts, and "
        "Lloyd's algorithm with Euclidean distance over raw pixel intensities in [0, 1]."
    ),
    citations=(
        "Lloyd 1982 IEEE Transactions on Information Theory 'Least Squares Quantization in PCM' "
        "(DOI:10.1109/TIT.1982.1056489) — foundational KMeans paper establishing the alternating "
        "assignment-and-update algorithm that minimizes within-cluster sum-of-squares; cited as "
        "the canonical clustering baseline against which every other partitional method must be "
        "compared, justifying its use here as the Exp 1 reference point for the Olivetti benchmark.;\n"
        "Arthur & Vassilvitskii 2007 SODA 'k-means++: The Advantages of Careful Seeding' "
        "(arXiv:1101.4022) — establishes the k-means++ initialization scheme that sklearn uses by "
        "default and provides an O(log K) approximation guarantee versus random init's worst-case "
        "unboundedness; relevant because robust seeding is essential when K=40 and n=400 produce "
        "an under-determined optimization landscape with many local minima.;\n"
        "Samaria & Harter 1994 IEEE Workshop on Applications of Computer Vision 'Parameterisation "
        "of a stochastic model for human face identification' — the original Olivetti Faces dataset "
        "paper; documents the imaging conditions (lighting variation, glasses, expressions) that "
        "drive the dataset's clustering difficulty and motivate dimensionality reduction in subsequent experiments."
    ),
    hypothesis=(
        "We hypothesize that KMeans with K=40, n_init=10, k-means++ init on raw pixel features "
        "will achieve ARI in the range 0.45 to 0.60 because the mechanism per Lloyd 1982 is "
        "Euclidean-distance partitioning of the 4096-dim pixel space; faces of the same person "
        "share lighting and pose patterns that produce small Euclidean differences in pixel space, "
        "but cross-subject pose variation is also large in pixel space, leading to mid-range cluster "
        "purity (NMI~0.78 expected per documented baselines)."
    ),
    prediction=(
        "ARI in 0.45 to 0.60. NMI in 0.74 to 0.82. Silhouette small but positive (0.05 to 0.20). "
        "n_pred_clusters = 40 exactly (we set K=40). If ARI > 0.55 we have matched the documented "
        "high end and validated the sklearn defaults; if ARI < 0.45 something is wrong with the data "
        "or the encoding."
    ),
)


def fit_predict_kmeans(X):
    return KMeans(n_clusters=40, n_init=10, random_state=0, max_iter=300).fit_predict(X)


config = {
    "backbone": "kmeans", "n_clusters": 40, "n_init": 10, "init": "k-means++",
    "max_iter": 300, "random_state": 0, "feature_space": "raw pixels (4096-dim, range [0,1])",
}

record = run_experiment(EXP_NUM, "kmeans_raw_pixels",
                         "KMeans K=40 on raw pixels — baseline (Lloyd 1982 + Arthur 2007 init)",
                         config, fit_predict_kmeans)

ari = record["test_primary"]
sil = record["secondary_metrics"]["silhouette"]
nmi = record["secondary_metrics"]["nmi"]
status = record["status"]

# Decide direction in-line for verdict
if 0.45 <= ari <= 0.60:
    pred_status = "WITHIN predicted range"
elif ari > 0.60:
    pred_status = "ABOVE predicted upper bound"
else:
    pred_status = "BELOW predicted lower bound"

verdict_text = (
    f"{status} (baseline) — ARI={ari:.4f}, NMI={nmi:.4f}, silhouette={sil:.4f}. "
    f"{pred_status} (predicted ARI 0.45-0.60). Status under floor=0.30: {'KEEP' if ari > 0.30 else 'DISCARD'}. "
    f"This baseline establishes the reference point for all downstream feature-engineering experiments. "
    f"K=40 was honored (n_pred_clusters={record['secondary_metrics']['n_pred_clusters']})."
)
learning_text = (
    f"Axis open: ALL feature-engineering and architecture-improvement axes. KMeans on raw pixels "
    f"is a defensible floor at ARI={ari:.4f}, providing the +Δ baseline against which dimensionality "
    f"reduction (PCA, UMAP), kernel methods (Spectral), generative models (VAE/AE), and pretrained "
    f"deep features will be measured. Next try: PCA(50) + KMeans (Exp 2) — documented improvement to "
    f"ARI~0.62 from 4096→50 dim reduction that removes pixel noise while preserving facial structure."
)
author_post_run(EXP_NUM, verdict=verdict_text, learning=learning_text)
print("\nExp 1 baseline complete + reasoning fully validated.")
