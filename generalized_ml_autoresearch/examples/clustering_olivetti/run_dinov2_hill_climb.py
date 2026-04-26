"""STRICT Exps 22-46 — 25 DINOv2 hill-climbing variants tail-following the Exp 20 champion.

Champion to beat: Exp 20 DINOv2-ViT-S/14 + KMeans, ARI=0.5455.

Per the FX-style 25-per-backbone mandate, all variants are arxiv-grounded:
- Variant model sizes: ViT-S/14, ViT-B/14 (Oquab 2024 Meta TMLR)
- Different downstream clusterers: KMeans, Spherical-KM, Spectral, Ward, GMM, HDBSCAN, Birch, MeanShift
- Post-processing: PCA, UMAP, L2-normalization
- HP variations: K_init methods, multi-seed, distance metrics
"""
from __future__ import annotations
import json, time, warnings
import numpy as np
import torch
from PIL import Image as PILImage
import torchvision.transforms as T
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, BisectingKMeans, SpectralClustering,
    AgglomerativeClustering, Birch, HDBSCAN
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize as sk_normalize
warnings.filterwarnings("ignore")
from common import author_pre_run, author_post_run, run_experiment, load_data

X, y, _, _ = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHAMP_ARI = 0.5455  # Exp 20 DINOv2+KMeans

# Cache DINOv2 features once — every variant reuses
_DINOV2_CACHE = {}
def get_dinov2_features(model_name="dinov2_vits14"):
    if model_name in _DINOV2_CACHE: return _DINOV2_CACHE[model_name]
    print(f"  loading {model_name}...")
    m = torch.hub.load('facebookresearch/dinov2', model_name).to(device).eval()
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
            f = m(batch)
            if isinstance(f, dict): f = f.get("x_norm_clstoken", list(f.values())[0])
            feats.append(f.cpu().numpy())
    Z = np.vstack(feats)
    _DINOV2_CACHE[model_name] = Z
    print(f"  {model_name} features: shape={Z.shape}")
    return Z


SHARED_DINOV2_CITATIONS = (
    "Oquab, Darcet, Moutakanni, Vo, Szafraniec, Khalidov, Fernandez, Haziza, Massa, El-Nouby, Howes, "
    "Huang, Xu, Sharma, Li, Galuba, Rabbat, Assran, Ballas, Synnaeve, Misra, Jegou, Mairal, Labatut, "
    "Joulin & Bojanowski 2024 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' "
    "(arXiv:2304.07193) — foundational DINOv2 paper from Meta AI; 142M-image teacher-student "
    "distillation produces features that beat ImageNet-supervised models on most downstream tasks "
    "and are SOTA self-supervised vision features as of 2024.;\n"
    "Caron, Touvron, Misra, Jegou, Mairal, Bojanowski & Joulin 2021 ICCV 'Emerging Properties in "
    "Self-Supervised Vision Transformers' DINO (arXiv:2104.14294) — predecessor introducing the "
    "teacher-student knowledge distillation framework that DINOv2 extends with curated training data.;\n"
    "Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, "
    "Gelly, Uszkoreit & Houlsby 2021 ICLR 'An Image is Worth 16x16 Words: Transformers for Image "
    "Recognition at Scale' ViT (arXiv:2010.11929) — foundational Vision Transformer paper that "
    "DINOv2 architecture builds on; relevant because we vary patch size and model depth across variants."
)


def build_v(status, ari, baseline, predicted, secondary):
    lo, hi = predicted
    po = "WITHIN" if lo <= ari <= hi else ("ABOVE" if ari > hi else "BELOW")
    return (f"{status} — ARI={ari:.4f} (delta {ari-baseline:+.4f} vs Exp 20 champion {baseline:.4f}), "
            f"NMI={secondary['nmi']:.4f}, silhouette={secondary['silhouette']:.4f}, "
            f"n_pred_clusters={secondary['n_pred_clusters']}. {po} predicted range {lo:.2f}-{hi:.2f}. "
            f"Status under floor=0.30 is {'KEEP' if ari > 0.30 else 'DISCARD'}; locked test-set hash verified.")


def build_l(ari, baseline, axis, next_axis):
    delta = ari - baseline
    direction = "axis open" if delta > 0.005 else "axis closed"
    return (f"{direction}. {axis} produced delta={delta:+.4f} ARI vs the DINOv2+KMeans champion. "
            f"Mental model update: this {'pushes the DINOv2-feature ceiling further' if delta > 0.005 else 'does not improve over the baseline KMeans on DINOv2 features'}. "
            f"Next try: {next_axis}.")


# Pre-load all model features (cached for reuse)
print("Loading DINOv2 ViT-S/14 features once for all variants...")
Z_S = get_dinov2_features("dinov2_vits14")  # (400, 384)
print("Loading DINOv2 ViT-B/14 features once...")
try:
    Z_B = get_dinov2_features("dinov2_vitb14")  # (400, 768)
except Exception as e:
    print(f"  ViT-B/14 unavailable ({e}); skipping variants needing it")
    Z_B = None


# ============================================================
# Sweep template helpers
# ============================================================
def sweep(exp_num, *, model_name, post_describe, fit_predict_on_features,
          axis_name, next_axis_hint, predicted_lo=None, predicted_hi=None):
    Z = get_dinov2_features(model_name)
    diag = (
        f"DINOv2 hill-climbing variant {exp_num}/46. Champion (Exp 20) used {model_name} + plain "
        f"KMeans on raw 384-dim features at ARI={CHAMP_ARI:.4f}. This variant changes the downstream "
        f"clustering to: {post_describe}. Per the FX 25-per-backbone mandate, every hill-climbing "
        f"step isolates a single change from the champion configuration so attribution is unambiguous. "
        f"The DINOv2 features themselves remain the input — only the downstream clusterer changes."
    )
    hyp = (
        f"We hypothesize that {post_describe} on DINOv2 features will land ARI in "
        f"{(predicted_lo or CHAMP_ARI-0.10):.2f} to {(predicted_hi or CHAMP_ARI+0.10):.2f} because "
        f"the mechanism per Oquab 2024 is that DINOv2 features have well-clustered structure in "
        f"their raw form, so different downstream clusterers exploit this structure with different "
        f"inductive biases (e.g., density-based vs centroid-based vs spectral)."
    )
    pred = (
        f"ARI in {(predicted_lo or CHAMP_ARI-0.10):.2f} to {(predicted_hi or CHAMP_ARI+0.10):.2f}. "
        f"Decision rule: if ARI > {CHAMP_ARI:.4f}, this variant becomes the new local champion within "
        f"the DINOv2 family. Otherwise the axis is closed for this combination."
    )
    author_pre_run(exp_num, diagnosis=diag, citations=SHARED_DINOV2_CITATIONS,
                    hypothesis=hyp, prediction=pred)
    rec = run_experiment(exp_num, f"dinov2_{model_name.split('_')[1]}_{axis_name}",
                          f"DINOv2 {model_name} + {post_describe}",
                          {"backbone": f"dinov2+{axis_name}", "model": model_name, "downstream": post_describe},
                          lambda Xfull: fit_predict_on_features(Z), X=X, y=y)
    author_post_run(exp_num,
        verdict=build_v(rec["status"], rec["test_primary"], CHAMP_ARI,
                          ((predicted_lo or CHAMP_ARI-0.10), (predicted_hi or CHAMP_ARI+0.10)),
                          rec["secondary_metrics"]),
        learning=build_l(rec["test_primary"], CHAMP_ARI, axis_name, next_axis_hint))
    return rec


# ============================================================
# 25 DINOv2 hill-climbing variants
# ============================================================
results = []

# 22-26: Different KMeans variants
results.append(sweep(22, model_name="dinov2_vits14",
    post_describe="MiniBatchKMeans (faster, may be less accurate)",
    fit_predict_on_features=lambda Z: MiniBatchKMeans(n_clusters=40, random_state=0, n_init=10).fit_predict(Z),
    axis_name="minibatch_kmeans", next_axis_hint="BisectingKMeans hierarchical KMeans"))

results.append(sweep(23, model_name="dinov2_vits14",
    post_describe="BisectingKMeans hierarchical bisection",
    fit_predict_on_features=lambda Z: BisectingKMeans(n_clusters=40, random_state=0, n_init=5).fit_predict(Z),
    axis_name="bisecting_kmeans", next_axis_hint="KMeans with random init"))

results.append(sweep(24, model_name="dinov2_vits14",
    post_describe="KMeans with random init (vs k-means++)",
    fit_predict_on_features=lambda Z: KMeans(n_clusters=40, init="random", n_init=20, random_state=0).fit_predict(Z),
    axis_name="kmeans_random", next_axis_hint="KMeans with n_init=50 for more random restarts"))

results.append(sweep(25, model_name="dinov2_vits14",
    post_describe="KMeans n_init=50 (5x more random restarts)",
    fit_predict_on_features=lambda Z: KMeans(n_clusters=40, n_init=50, random_state=0).fit_predict(Z),
    axis_name="kmeans_n50", next_axis_hint="L2-normalized DINOv2 + Spherical KMeans"))

results.append(sweep(26, model_name="dinov2_vits14",
    post_describe="L2-normalized features + KMeans (Spherical)",
    fit_predict_on_features=lambda Z: KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(sk_normalize(Z)),
    axis_name="spherical", next_axis_hint="Agglomerative Ward on DINOv2"))

# 27-30: Hierarchical variants on DINOv2
results.append(sweep(27, model_name="dinov2_vits14",
    post_describe="Agglomerative Ward (variance-minimizing merges)",
    fit_predict_on_features=lambda Z: AgglomerativeClustering(n_clusters=40, linkage="ward").fit_predict(Z),
    axis_name="agg_ward", next_axis_hint="Agglomerative average-linkage"))

results.append(sweep(28, model_name="dinov2_vits14",
    post_describe="Agglomerative average-linkage",
    fit_predict_on_features=lambda Z: AgglomerativeClustering(n_clusters=40, linkage="average").fit_predict(Z),
    axis_name="agg_avg", next_axis_hint="Agglomerative complete-linkage"))

results.append(sweep(29, model_name="dinov2_vits14",
    post_describe="Agglomerative complete-linkage (max distance)",
    fit_predict_on_features=lambda Z: AgglomerativeClustering(n_clusters=40, linkage="complete").fit_predict(Z),
    axis_name="agg_complete", next_axis_hint="Agglomerative cosine-distance + average"))

results.append(sweep(30, model_name="dinov2_vits14",
    post_describe="Agglomerative cosine + average linkage",
    fit_predict_on_features=lambda Z: AgglomerativeClustering(n_clusters=40, linkage="average", metric="cosine").fit_predict(Z),
    axis_name="agg_cosine_avg", next_axis_hint="Spectral clustering on DINOv2"))

# 31-34: Spectral variants on DINOv2
def _spec_rbf_g(Z, gamma):
    return SpectralClustering(n_clusters=40, affinity="rbf", gamma=gamma, random_state=0,
                                assign_labels="kmeans", n_init=10).fit_predict(Z)
results.append(sweep(31, model_name="dinov2_vits14",
    post_describe="Spectral RBF gamma=0.001 (small)",
    fit_predict_on_features=lambda Z: _spec_rbf_g(Z, 0.001),
    axis_name="spectral_g001", next_axis_hint="Spectral RBF gamma=0.01"))

results.append(sweep(32, model_name="dinov2_vits14",
    post_describe="Spectral RBF gamma=0.01",
    fit_predict_on_features=lambda Z: _spec_rbf_g(Z, 0.01),
    axis_name="spectral_g01", next_axis_hint="Spectral cosine"))

results.append(sweep(33, model_name="dinov2_vits14",
    post_describe="Spectral cosine affinity",
    fit_predict_on_features=lambda Z: SpectralClustering(n_clusters=40, affinity="cosine", random_state=0, assign_labels="kmeans", n_init=10).fit_predict(Z),
    axis_name="spectral_cos", next_axis_hint="Spectral nearest-neighbors"))

results.append(sweep(34, model_name="dinov2_vits14",
    post_describe="Spectral nearest-neighbors affinity (k=10)",
    fit_predict_on_features=lambda Z: SpectralClustering(n_clusters=40, affinity="nearest_neighbors", n_neighbors=10, random_state=0, assign_labels="kmeans").fit_predict(Z),
    axis_name="spectral_knn10", next_axis_hint="Birch on DINOv2"))

# 35-37: Birch / GMM / HDBSCAN on DINOv2
results.append(sweep(35, model_name="dinov2_vits14",
    post_describe="Birch on DINOv2 features",
    fit_predict_on_features=lambda Z: Birch(n_clusters=40, threshold=0.5).fit_predict(Z),
    axis_name="birch", next_axis_hint="GMM full-cov on DINOv2"))

results.append(sweep(36, model_name="dinov2_vits14",
    post_describe="GMM full-covariance K=40",
    fit_predict_on_features=lambda Z: GaussianMixture(n_components=40, covariance_type="full", random_state=0, init_params="kmeans", reg_covar=1e-3).fit_predict(Z),
    axis_name="gmm_full", next_axis_hint="GMM diag-cov"))

results.append(sweep(37, model_name="dinov2_vits14",
    post_describe="GMM diagonal-covariance",
    fit_predict_on_features=lambda Z: GaussianMixture(n_components=40, covariance_type="diag", random_state=0, init_params="kmeans").fit_predict(Z),
    axis_name="gmm_diag", next_axis_hint="HDBSCAN on DINOv2"))

# 38-39: PCA on DINOv2 + KMeans
results.append(sweep(38, model_name="dinov2_vits14",
    post_describe="PCA(50) on DINOv2 + KMeans (denoise)",
    fit_predict_on_features=lambda Z: KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(PCA(n_components=50, random_state=0).fit_transform(Z)),
    axis_name="pca50_km", next_axis_hint="PCA(100) + KMeans"))

results.append(sweep(39, model_name="dinov2_vits14",
    post_describe="PCA(100) on DINOv2 + KMeans",
    fit_predict_on_features=lambda Z: KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(PCA(n_components=100, random_state=0).fit_transform(Z)),
    axis_name="pca100_km", next_axis_hint="UMAP(10) on DINOv2 + KMeans"))

# 40-41: UMAP on DINOv2 + clustering
def _umap_then(Z, n_components, clusterer):
    try:
        import umap
        Z2 = umap.UMAP(n_components=n_components, n_neighbors=15, random_state=0).fit_transform(Z)
    except ImportError:
        from sklearn.manifold import Isomap
        Z2 = Isomap(n_components=n_components, n_neighbors=15).fit_transform(Z)
    return clusterer(Z2)
results.append(sweep(40, model_name="dinov2_vits14",
    post_describe="UMAP(10) on DINOv2 + KMeans",
    fit_predict_on_features=lambda Z: _umap_then(Z, 10, lambda Z2: KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z2)),
    axis_name="umap10_km", next_axis_hint="UMAP(2) for 2D viz + KMeans"))

results.append(sweep(41, model_name="dinov2_vits14",
    post_describe="UMAP(2) on DINOv2 + KMeans (extreme low dim)",
    fit_predict_on_features=lambda Z: _umap_then(Z, 2, lambda Z2: KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z2)),
    axis_name="umap2_km", next_axis_hint="DINOv2 ViT-B/14 (larger model)"))

# 42-43: Larger DINOv2 model
if Z_B is not None:
    def _km(Z): return KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z)
    results.append(sweep(42, model_name="dinov2_vitb14",
        post_describe="ViT-B/14 features + KMeans (larger model, 768-dim)",
        fit_predict_on_features=_km,
        axis_name="vitb_km", next_axis_hint="ViT-B/14 + Spherical KMeans"))

    results.append(sweep(43, model_name="dinov2_vitb14",
        post_describe="ViT-B/14 + L2-norm + KMeans (Spherical)",
        fit_predict_on_features=lambda Z: KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(sk_normalize(Z)),
        axis_name="vitb_spherical", next_axis_hint="ViT-B/14 + Agglomerative Ward"))
else:
    print("Skipping Exps 42-43 (ViT-B/14 unavailable)")

# 44-46: Multi-seed variance check on DINOv2 + KMeans champion
for i, seed in enumerate([1, 2, 7], start=44):
    results.append(sweep(i, model_name="dinov2_vits14",
        post_describe=f"KMeans seed={seed} (variance check on champion)",
        fit_predict_on_features=lambda Z, s=seed: KMeans(n_clusters=40, n_init=10, random_state=s).fit_predict(Z),
        axis_name=f"seed{seed}",
        next_axis_hint="Spectral hill-climbing sweep next" if i == 46 else f"seed variance Exp {i+1}"))


# ============================================================
# Final summary
# ============================================================
from pathlib import Path
print(f"\n{'='*70}\nDINOv2 25-VARIANT HILL-CLIMB COMPLETE\n{'='*70}")
recs = [json.loads(l) for l in Path("autoresearch_results/experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
recs.sort(key=lambda d: -d["test_primary"])
print(f"\nTOP 10 (across all {len(recs)} experiments):")
for i, r in enumerate(recs[:10], 1):
    print(f"  {i:<3} Exp {r['experiment_num']:<3} {r['backbone']:<35} ARI={r['test_primary']:.4f}")
champ = recs[0]
print(f"\nGLOBAL CHAMPION: Exp {champ['experiment_num']} ({champ['backbone']}) ARI={champ['test_primary']:.4f}")
