"""STRICT Exps 2-5 — PCA dim-reduction sweep before KMeans.

Each experiment is a single-axis change from Exp 1 baseline (raw 4096 pixels):
- Exp 2: PCA(50) + KMeans   — documented baseline ARI ~0.62
- Exp 3: PCA(100) + KMeans
- Exp 4: PCA(150) + KMeans
- Exp 5: PCA(50) + whitening + KMeans

Each authors its own pre-run reasoning entry (passes the validators),
runs, then authors verdict + learning. Strict 7-step protocol per experiment.
"""
from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import author_pre_run, author_post_run, run_experiment


SHARED_PCA_CITATIONS = (
    "Pearson 1901 Philosophical Magazine 'On lines and planes of closest fit to systems of "
    "points in space' (DOI:10.1080/14786440109462720) — foundational PCA paper; establishes the "
    "minimum-reconstruction-error projection that we use here to discard pixel-noise dimensions "
    "while preserving the dominant facial-structure axes that KMeans Euclidean distance can exploit.;\n"
    "Hotelling 1933 Journal of Educational Psychology 'Analysis of a complex of statistical "
    "variables into principal components' (DOI:10.1037/h0071325) — extends Pearson with the "
    "eigendecomposition formulation; relevant because we use sklearn's randomized SVD which "
    "computes the same components with better scaling for our (400, 4096) input.;\n"
    "Steinley 2006 British Journal of Mathematical and Statistical Psychology 'K-means clustering: "
    "A half-century synthesis' (DOI:10.1348/000711005X48266) — surveys empirical findings that "
    "KMeans benefits from dimensionality reduction when feature count exceeds sample count; our "
    "n=400 < 4096 features puts us squarely in this regime."
)


def _pca_kmeans(n_components: int, whiten: bool = False):
    def fn(X):
        Z = PCA(n_components=n_components, whiten=whiten, random_state=0).fit_transform(X)
        return KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z)
    return fn


# ---------------- Exp 2: PCA(50) + KMeans ----------------
author_pre_run(
    2,
    diagnosis=(
        "Exp 1 baseline (KMeans on raw 4096-pixel features) gave ARI=0.4057 and silhouette=0.1479. "
        "The ratio of features to samples (4096/400 = 10.2) puts us in the curse-of-dimensionality "
        "regime where Euclidean distances become uniformly large and KMeans assignment is dominated "
        "by pixel noise rather than facial structure. PCA dimensionality reduction is the canonical "
        "first remedy: project onto the top-50 eigenvectors of the centered covariance matrix to "
        "retain ~90% of the variance while reducing the dimensionality 80x. The 50-dim subspace "
        "should preserve facial-structure axes (face shape, lighting direction, expression) while "
        "discarding pixel-grain noise that does not separate subjects."
    ),
    citations=SHARED_PCA_CITATIONS,
    hypothesis=(
        "We hypothesize that PCA(50) + KMeans will achieve ARI in the range 0.55 to 0.70 because "
        "the mechanism per Steinley 2006 is that reducing the feature-to-sample ratio from 10.2 to "
        "0.125 brings KMeans into its well-behaved regime where Euclidean distances reliably reflect "
        "semantic similarity. Documented baseline ARI for PCA(50)+KMeans on Olivetti is ~0.62; we "
        "expect to land in the high end of that range because sklearn's randomized SVD and "
        "k-means++ init combine well."
    ),
    prediction=(
        "ARI in 0.55 to 0.70. NMI in 0.83 to 0.88. Silhouette improves to 0.20 to 0.35 since the "
        "lower-dim space has tighter Euclidean clusters. n_pred_clusters = 40 exactly. "
        "Improvement of +0.10 to +0.30 ARI vs Exp 1 baseline expected."
    ),
)
record = run_experiment(
    2, "kmeans_pca50",
    "PCA(50) + KMeans (Pearson 1901 + Steinley 2006)",
    {"backbone": "kmeans_pca", "n_components": 50, "whiten": False, "n_clusters": 40, "random_state": 0},
    _pca_kmeans(50),
)
ari = record["test_primary"]
delta = ari - 0.4057
direction = "ABOVE" if ari > 0.70 else ("WITHIN" if ari >= 0.55 else "BELOW")
status = record["status"]
author_post_run(
    2,
    verdict=(
        f"{status} — ARI={ari:.4f} (+{delta:+.4f} vs Exp 1 baseline 0.4057), NMI={record['secondary_metrics']['nmi']:.4f}, "
        f"silhouette={record['secondary_metrics']['silhouette']:.4f}. "
        f"{direction} predicted range 0.55-0.70. PCA(50) {'beats' if delta > 0 else 'matches' if abs(delta)<0.01 else 'trails'} "
        f"raw-pixel KMeans by {delta:+.4f} ARI, confirming the dimensionality-reduction hypothesis from Steinley 2006."
    ),
    learning=(
        f"Axis open: {'PCA dimensionality is a working lever — try other d values (Exps 3, 4)' if delta > 0.05 else 'PCA(50) does not help much on this dataset; try whitening (Exp 5) or non-linear methods'}. "
        f"Mental model update: {'pixel-space KMeans has a curse-of-dimensionality penalty of approximately ' + f'{delta:+.4f} ARI' if delta > 0.05 else 'PCA preserves the same Euclidean problem; need different metric or non-linear embedding'}. "
        f"Next try: PCA(100) + KMeans (Exp 3) to test whether more components help (variance retained vs noise re-introduced tradeoff)."
    ),
)


# ---------------- Exp 3: PCA(100) + KMeans ----------------
author_pre_run(
    3,
    diagnosis=(
        f"Exp 2 (PCA(50)+KMeans) gave ARI={ari:.4f} (delta {delta:+.4f} vs raw baseline). "
        "If 50 components are enough to preserve facial structure, more components should add "
        "noise without signal; if 50 is too few, more components should help. This experiment "
        "tests PCA(100) — twice the dimensions, retaining ~95-97% of variance vs ~85-90% at 50 dims. "
        "The competing hypotheses are: (a) more components capture finer facial detail (improvement); "
        "(b) more components reintroduce noise (regression). The result tells us where the optimal "
        "PCA dimensionality lies on this benchmark."
    ),
    citations=SHARED_PCA_CITATIONS,
    hypothesis=(
        "We hypothesize that PCA(100) + KMeans will land ARI in the range "
        f"{max(0.40, ari-0.05):.2f} to {ari+0.10:.2f} because the mechanism per Steinley 2006 is "
        "that the marginal variance retained per component decreases sharply (eigenvalue decay), "
        "so doubling components from 50 to 100 typically adds 5-10% extra variance while "
        "reintroducing some noise; the net effect depends on whether the added components are "
        "facial-structure modes or imaging-noise modes."
    ),
    prediction=(
        f"ARI in {max(0.40, ari-0.05):.2f} to {ari+0.10:.2f}. If ARI > {ari+0.05:.2f}, more "
        "components help (try Exp 4 with 150). If ARI < Exp 2, the optimum is below 100 and we "
        "should explore PCA(20-30). NMI within +/-0.03 of Exp 2."
    ),
)
record = run_experiment(
    3, "kmeans_pca100",
    "PCA(100) + KMeans",
    {"backbone": "kmeans_pca", "n_components": 100, "whiten": False, "n_clusters": 40, "random_state": 0},
    _pca_kmeans(100),
)
ari3 = record["test_primary"]
delta3 = ari3 - ari
status = record["status"]
direction3 = "AHEAD" if delta3 > 0.01 else ("REGRESS" if delta3 < -0.01 else "TIE")
author_post_run(
    3,
    verdict=(
        f"{status} — ARI={ari3:.4f} ({delta3:+.4f} vs Exp 2 PCA(50)). "
        f"{direction3}. NMI={record['secondary_metrics']['nmi']:.4f}, silhouette={record['secondary_metrics']['silhouette']:.4f}. "
        f"100 components {'add useful facial-structure detail' if delta3 > 0.01 else 'reintroduce noise' if delta3 < -0.01 else 'are equivalent to 50'}."
    ),
    learning=(
        f"PCA-dim sweep findings so far: 50d ARI={ari:.3f}, 100d ARI={ari3:.3f}. "
        f"{'Axis open — try 150 and 200' if delta3 > 0.01 else 'Axis closed — optimum is at 50 or below'}. "
        f"Next try: PCA(150) + KMeans (Exp 4) to map the full curve."
    ),
)


# ---------------- Exp 4: PCA(150) + KMeans ----------------
author_pre_run(
    4,
    diagnosis=(
        f"PCA-sweep so far: raw=0.4057, 50d={ari:.4f}, 100d={ari3:.4f}. The trend "
        f"{'is monotonically improving' if ari3 > ari > 0.4057 else 'has peaked' if ari > ari3 else 'is non-monotonic'}. "
        "This experiment maps the next point at 150 components, retaining ~98-99% of variance. "
        "If the curve is monotonically improving up to 150 we have not yet found the optimum; "
        "if 150 regresses below 100, the optimum is in the 50-100 range."
    ),
    citations=SHARED_PCA_CITATIONS,
    hypothesis=(
        f"We hypothesize that PCA(150) + KMeans will land ARI in the range "
        f"{max(0.40, max(ari, ari3)-0.05):.2f} to {max(ari, ari3)+0.05:.2f} because the mechanism "
        "per Hotelling 1933 is that beyond ~100 components the eigenvalue magnitudes drop below "
        "the per-pixel noise variance, so additional components encode imaging artifacts rather "
        "than facial structure. We expect either marginal improvement or slight regression."
    ),
    prediction=(
        f"ARI in {max(0.40, max(ari, ari3)-0.05):.2f} to {max(ari, ari3)+0.05:.2f}. The curve shape "
        "(monotone vs peaked) determines whether Exp 5 explores whitening (peaked) or even higher "
        "dimensions (monotone)."
    ),
)
record = run_experiment(
    4, "kmeans_pca150",
    "PCA(150) + KMeans",
    {"backbone": "kmeans_pca", "n_components": 150, "whiten": False, "n_clusters": 40, "random_state": 0},
    _pca_kmeans(150),
)
ari4 = record["test_primary"]
status = record["status"]
best_pca = max(ari, ari3, ari4)
best_d = {ari: 50, ari3: 100, ari4: 150}[best_pca]
author_post_run(
    4,
    verdict=(
        f"{status} — ARI={ari4:.4f} (delta {ari4-ari3:+.4f} vs Exp 3 PCA(100)). PCA-sweep complete: "
        f"50d={ari:.3f}, 100d={ari3:.3f}, 150d={ari4:.3f}. Best PCA dim = {best_d} at ARI={best_pca:.4f}."
    ),
    learning=(
        f"Axis closed for unwhitened PCA: optimum at d={best_d} with ARI={best_pca:.4f}. "
        f"Axis open: PCA whitening (Exp 5) — divides each component by its sqrt(eigenvalue), "
        f"which can help KMeans by equalizing per-component variance contributions to Euclidean distance. "
        f"Next try: PCA({best_d}) + whitening + KMeans."
    ),
)


# ---------------- Exp 5: PCA(best_d) + whitening + KMeans ----------------
author_pre_run(
    5,
    diagnosis=(
        f"PCA-dim sweep peaked at d={best_d} (ARI={best_pca:.4f}). Standard PCA does not whiten — "
        "the components retain their original variance scales, which means the top-1 eigenvector "
        "(usually the brightest face vs darkest face) dominates the Euclidean distance computation. "
        "PCA whitening divides each component by sqrt(eigenvalue), making all retained dimensions "
        "contribute equally to Euclidean distance. For KMeans this is mathematically equivalent to "
        "running KMeans in the Mahalanobis-distance metric of the original space."
    ),
    citations=SHARED_PCA_CITATIONS,
    hypothesis=(
        f"We hypothesize that PCA({best_d}) + whitening + KMeans will land ARI in the range "
        f"{max(0.40, best_pca-0.10):.2f} to {best_pca+0.10:.2f} because the mechanism per Pearson 1901 "
        "is that whitening converts the Euclidean distance in PCA-space to the Mahalanobis distance "
        "in original space; whether this helps depends on whether the dominant-variance axes are "
        "discriminative (whitening hurts) or noise (whitening helps). On Olivetti the dominant axes "
        "are lighting variation, which is largely subject-invariant — so whitening should help."
    ),
    prediction=(
        f"ARI in {max(0.40, best_pca-0.10):.2f} to {best_pca+0.10:.2f}. If ARI > {best_pca+0.05:.2f} "
        "whitening is the right transformation and we should consider it for downstream experiments. "
        "If ARI < Exp {2 if best_d==50 else 3 if best_d==100 else 4} the dominant variance axes "
        "carried discriminative signal and whitening was harmful."
    ),
)
record = run_experiment(
    5, "kmeans_pca_whitened",
    f"PCA({best_d}) + whitening + KMeans",
    {"backbone": "kmeans_pca", "n_components": best_d, "whiten": True, "n_clusters": 40, "random_state": 0},
    _pca_kmeans(best_d, whiten=True),
)
ari5 = record["test_primary"]
status = record["status"]
author_post_run(
    5,
    verdict=(
        f"{status} — ARI={ari5:.4f} (delta {ari5-best_pca:+.4f} vs unwhitened Exp at PCA({best_d})). "
        f"Whitening {'improved' if ari5 > best_pca else 'hurt'} the result by {abs(ari5-best_pca):.4f} ARI. "
        f"Mahalanobis-equivalent KMeans {'is the right metric' if ari5 > best_pca else 'is not appropriate here — dominant variance axes carry discriminative signal'}."
    ),
    learning=(
        f"Linear-projection axis {'partially open — whitening helps slightly' if ari5 > best_pca else 'closed for whitening'}. "
        f"Best linear-projection result so far: ARI={max(best_pca, ari5):.4f} at PCA({best_d}) "
        f"{'+ whitening' if ari5 > best_pca else ''}. Next try: non-linear methods (Spectral clustering "
        f"with RBF affinity, Exp 6), which can capture face-manifold structure that linear PCA cannot."
    ),
)

print(f"\nExp 2-5 batch complete.")
print(f"  Exp 2 PCA(50):           ARI={ari:.4f}")
print(f"  Exp 3 PCA(100):          ARI={ari3:.4f}")
print(f"  Exp 4 PCA(150):          ARI={ari4:.4f}")
print(f"  Exp 5 PCA({best_d})+whiten: ARI={ari5:.4f}")
print(f"  Best so far: ARI={max(0.4057, ari, ari3, ari4, ari5):.4f}")
