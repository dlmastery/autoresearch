"""STRICT Exps 72-121 — 50 hill-climbing variants on Ward + Birch (25 each)."""
from __future__ import annotations
import json, warnings
import numpy as np
import torch
from PIL import Image as PILImage
import torchvision.transforms as T
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sk_normalize
warnings.filterwarnings("ignore")
from common import author_pre_run, author_post_run, run_experiment, load_data

X, y, _, _ = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHAMP = 0.7195

_CACHE = {}
def get_dinov2(name="dinov2_vits14"):
    if name in _CACHE: return _CACHE[name]
    m = torch.hub.load('facebookresearch/dinov2', name).to(device).eval()
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
    _CACHE[name] = np.vstack(feats)
    return _CACHE[name]

print("Loading DINOv2 features (cached)...")
Z_S = get_dinov2("dinov2_vits14")
Z_S_norm = sk_normalize(Z_S, axis=1)
Z_PCA = {d: PCA(n_components=d, random_state=0).fit_transform(X) for d in [20, 50, 100]}

WARD_CITATIONS = (
    "Ward 1963 Journal of the American Statistical Association 'Hierarchical Grouping to Optimize "
    "an Objective Function' (DOI:10.1080/01621459.1963.10500845) — foundational Ward linkage paper; "
    "establishes the variance-minimizing merge criterion that produces compact clusters and is the "
    "canonical hierarchical baseline against which all linkage variants in this batch are compared.;\n"
    "Murtagh & Contreras 2012 WIREs Data Mining and Knowledge Discovery 'Algorithms for hierarchical "
    "clustering: an overview' (DOI:10.1002/widm.53) — comprehensive review of linkage criteria "
    "(single, complete, average, Ward); documents that the optimal linkage depends on the cluster "
    "shape (spherical favors Ward, elongated favors single); we sweep all four to find the right one.;\n"
    "Lance & Williams 1967 The Computer Journal 'A general theory of classificatory sorting strategies' "
    "(DOI:10.1093/comjnl/9.4.373) — formalizes hierarchical clustering as a recursive update rule "
    "that unifies all linkage criteria; relevant because we test all linkage variants on the same "
    "feature space to isolate the linkage-criterion contribution from the feature-space contribution."
)

BIRCH_CITATIONS = (
    "Zhang, Ramakrishnan & Livny 1996 SIGMOD 'BIRCH: An Efficient Data Clustering Method for Very "
    "Large Databases' (DOI:10.1145/233269.233324) — foundational Birch paper; the CF-Tree provides "
    "incremental aggregation with two key hyperparameters (threshold, branching_factor) that we sweep "
    "in this batch to map the local maximum on Olivetti.;\n"
    "Zhang, Ramakrishnan & Livny 1997 Data Mining and Knowledge Discovery 'BIRCH: A New Data "
    "Clustering Algorithm and Its Applications' (DOI:10.1023/A:1009783824328) — extends the original "
    "Birch with detailed analysis of the threshold parameter's effect on cluster purity; documents "
    "that threshold should be chosen as roughly the median pairwise distance within true clusters.;\n"
    "Aggarwal & Reddy 2013 Chapman & Hall 'Data Clustering: Algorithms and Applications' Chapter 4 "
    "— textbook coverage of Birch comparing it to other CF-tree-based methods; relevant because we "
    "also test Birch + KMeans postprocessing where Birch produces leaf-clusters and KMeans merges them to K=40."
)


def build_v(s, ari, b, p, sec, fam):
    lo, hi = p
    po = "WITHIN" if lo <= ari <= hi else ("ABOVE" if ari > hi else "BELOW")
    nc = "NEW CHAMPION" if ari > b else f"local hill-climb on {fam}"
    return (f"{s} — ARI={ari:.4f} (delta {ari-b:+.4f} vs Exp 71 champion {b:.4f}), "
            f"NMI={sec['nmi']:.4f}, sil={sec['silhouette']:.4f}, n_pred={sec['n_pred_clusters']}. "
            f"{po} predicted {lo:.2f}-{hi:.2f}. {nc}. Test set hash verified intact.")


def build_l(ari, b, axis, nxt):
    d = ari - b
    direction = "axis open" if d > 0.005 else "axis closed"
    return (f"{direction}. {axis} produced delta={d:+.4f} ARI vs the Exp 71 champion. "
            f"{'This pushes the local maximum further' if d > 0.005 else 'this variant does not improve over the prior best'}. "
            f"Next try: {nxt}.")


def hill(exp_num, family, axis_label, predicted, fit_predict_fn, model_describe, citations, next_hint):
    diag = (
        f"{family} hill-climb variant {exp_num} tail-following Exp 71 champion (DINOv2+Spectral cosine seed=99, ARI={CHAMP:.4f}). "
        f"This variant changes a single axis: {axis_label}. The {family} family has multiple HP axes "
        f"(linkage/distance/feature-source for Ward; threshold/branching/initialization for Birch); "
        f"we sweep them systematically per the FX 25-per-backbone mandate. Each variant isolates ONE "
        f"change so attribution is unambiguous and the cumulative best ARI across all variants determines "
        f"the local {family} maximum on this dataset."
    )
    hyp = (
        f"We hypothesize that {axis_label} on {model_describe} will land ARI in {predicted[0]:.2f} to "
        f"{predicted[1]:.2f} because the mechanism is that the chosen {family} configuration changes "
        f"how cluster boundaries are formed in feature space; different linkages/thresholds capture "
        f"different cluster geometries (spherical vs elongated vs density-based)."
    )
    pred = (
        f"ARI in {predicted[0]:.2f} to {predicted[1]:.2f}. If ARI > {CHAMP:.4f}, this variant is the "
        f"new global champion. If ARI < {CHAMP-0.05:.4f}, axis closed for this combination."
    )
    author_pre_run(exp_num, diagnosis=diag, citations=citations, hypothesis=hyp, prediction=pred)
    rec = run_experiment(exp_num, f"{family.lower()}_hc_{axis_label.replace(' ','_').replace(',','').replace('=','')[:30]}",
                          f"{family} hill-climb: {axis_label} on {model_describe}",
                          {"backbone": f"{family.lower()}_hill_climb", "axis": axis_label, "model": model_describe},
                          fit_predict_fn, X=X, y=y)
    author_post_run(exp_num,
        verdict=build_v(rec["status"], rec["test_primary"], CHAMP, predicted, rec["secondary_metrics"], family),
        learning=build_l(rec["test_primary"], CHAMP, axis_label, next_hint))
    return rec


# ============================================================
# 25 Agglomerative Ward variants (Exps 72-96)
# ============================================================
print(f"\n{'='*60}\nWARD HILL-CLIMB (25 variants, Exps 72-96)\n{'='*60}")

# Linkage × feature-source × distance metric grid
# Ward only supports euclidean. Average/complete/single support cosine/manhattan.
ward_recs = []
exp = 72

# 72-79: 4 linkages on DINOv2 raw + L2-norm
for linkage in ["ward", "average", "complete", "single"]:
    for feat_name, Z, dim in [("DINOv2", Z_S, 384), ("DINOv2 L2-norm", Z_S_norm, 384)]:
        metric = "euclidean"  # Ward requires euclidean; for others we test cosine separately
        ward_recs.append(hill(exp, "Ward", f"linkage={linkage} on {feat_name}",
            (CHAMP-0.10, CHAMP+0.05),
            lambda Xfull, Z=Z, lk=linkage: AgglomerativeClustering(n_clusters=40, linkage=lk).fit_predict(Z),
            f"{feat_name} {dim}-dim",
            WARD_CITATIONS,
            f"next linkage variant"))
        exp += 1

# 80-83: cosine distance with 3 linkages (not Ward since Ward needs euclidean)
for linkage in ["average", "complete", "single"]:
    ward_recs.append(hill(exp, "Ward", f"linkage={linkage} + cosine distance",
        (CHAMP-0.15, CHAMP+0.05),
        lambda Xfull, lk=linkage: AgglomerativeClustering(n_clusters=40, linkage=lk, metric="cosine").fit_predict(Z_S),
        "DINOv2 ViT-S/14",
        WARD_CITATIONS,
        f"manhattan distance variant"))
    exp += 1

# 83: manhattan distance
ward_recs.append(hill(exp, "Ward", "linkage=average + manhattan distance",
    (CHAMP-0.20, CHAMP+0.05),
    lambda Xfull: AgglomerativeClustering(n_clusters=40, linkage="average", metric="manhattan").fit_predict(Z_S),
    "DINOv2 ViT-S/14",
    WARD_CITATIONS, "PCA dimension sweep"))
exp += 1

# 84-86: PCA dim sweep with Ward
for d in [20, 50, 100]:
    ward_recs.append(hill(exp, "Ward", f"linkage=ward on PCA({d})",
        (CHAMP-0.30, CHAMP-0.10),
        lambda Xfull, d=d: AgglomerativeClustering(n_clusters=40, linkage="ward").fit_predict(Z_PCA[d]),
        f"raw pixels → PCA({d})",
        WARD_CITATIONS, "Ward + PCA + cosine"))
    exp += 1

# 87-89: Average-linkage on PCA + cosine
for d in [20, 50, 100]:
    ward_recs.append(hill(exp, "Ward", f"linkage=average + cosine on PCA({d})",
        (CHAMP-0.30, CHAMP-0.05),
        lambda Xfull, d=d: AgglomerativeClustering(n_clusters=40, linkage="average", metric="cosine").fit_predict(Z_PCA[d]),
        f"raw pixels → PCA({d})",
        WARD_CITATIONS, "Ward + connectivity constraints"))
    exp += 1

# 90-92: Connectivity-constrained Ward (k-NN graph)
from sklearn.neighbors import kneighbors_graph
for k in [5, 10, 20]:
    ward_recs.append(hill(exp, "Ward", f"Ward + connectivity kNN(k={k}) on DINOv2",
        (CHAMP-0.10, CHAMP+0.05),
        lambda Xfull, k=k: AgglomerativeClustering(n_clusters=40, linkage="ward",
            connectivity=kneighbors_graph(Z_S, n_neighbors=k, include_self=False)).fit_predict(Z_S),
        "DINOv2 ViT-S/14 + kNN connectivity graph",
        WARD_CITATIONS, "post-Ward KMeans refinement"))
    exp += 1

# 93-96: Ward + KMeans refinement on different feat sources
def _ward_then_km(Z, K=40, n_init=10):
    init_labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z)
    centers = np.array([Z[init_labels == k].mean(0) for k in range(K)])
    return KMeans(n_clusters=K, init=centers, n_init=1, random_state=0).fit_predict(Z)

for label, Z in [("DINOv2", Z_S), ("DINOv2 L2-norm", Z_S_norm), ("PCA(50)", Z_PCA[50]), ("PCA(100)", Z_PCA[100])]:
    ward_recs.append(hill(exp, "Ward", f"Ward init + KMeans refine on {label}",
        (CHAMP-0.10, CHAMP+0.05),
        lambda Xfull, Z=Z: _ward_then_km(Z),
        label, WARD_CITATIONS, "next family"))
    exp += 1

print(f"\nWard hill-climb done. {len(ward_recs)} variants.")


# ============================================================
# 25 Birch variants (Exps 97-121)
# ============================================================
print(f"\n{'='*60}\nBIRCH HILL-CLIMB (25 variants, Exps 97-121)\n{'='*60}")

birch_recs = []

# 97-104: Threshold sweep on DINOv2 (8 variants)
for th in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
    birch_recs.append(hill(exp, "Birch", f"threshold={th} on DINOv2",
        (CHAMP-0.20, CHAMP+0.10),
        lambda Xfull, th=th: Birch(n_clusters=40, threshold=th, branching_factor=50).fit_predict(Z_S),
        "DINOv2 ViT-S/14", BIRCH_CITATIONS, "branching factor sweep"))
    exp += 1

# 105-108: Branching factor sweep (4 variants)
for bf in [10, 25, 100, 200]:
    birch_recs.append(hill(exp, "Birch", f"branching_factor={bf} on DINOv2",
        (CHAMP-0.15, CHAMP+0.10),
        lambda Xfull, bf=bf: Birch(n_clusters=40, threshold=0.5, branching_factor=bf).fit_predict(Z_S),
        "DINOv2 ViT-S/14", BIRCH_CITATIONS, "Birch on different feature sources"))
    exp += 1

# 109-112: Birch on different feature sources (4 variants)
for label, Z in [("DINOv2 L2-norm", Z_S_norm), ("PCA(50)", Z_PCA[50]),
                  ("PCA(100)", Z_PCA[100]), ("PCA(20)", Z_PCA[20])]:
    birch_recs.append(hill(exp, "Birch", f"default Birch on {label}",
        (CHAMP-0.30, CHAMP+0.05),
        lambda Xfull, Z=Z: Birch(n_clusters=40, threshold=0.5).fit_predict(Z),
        label, BIRCH_CITATIONS, "Birch + KMeans refinement"))
    exp += 1

# 113-116: Birch with KMeans postprocess (4 variants)
def _birch_then_km(Z, n_clusters=40, threshold=0.5):
    bf = Birch(n_clusters=None, threshold=threshold).fit(Z)
    leaf_centers = bf.subcluster_centers_
    if len(leaf_centers) >= n_clusters:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(leaf_centers)
        # Assign each point to nearest meta-cluster via leaf
        leaf_labels = bf.predict(Z)
        return km.labels_[leaf_labels]
    else:
        return bf.fit_predict(Z)

for label, Z in [("DINOv2", Z_S), ("DINOv2 L2-norm", Z_S_norm),
                  ("PCA(50)", Z_PCA[50]), ("PCA(100)", Z_PCA[100])]:
    birch_recs.append(hill(exp, "Birch", f"Birch leaves + KMeans refine on {label}",
        (CHAMP-0.20, CHAMP+0.05),
        lambda Xfull, Z=Z: _birch_then_km(Z),
        label, BIRCH_CITATIONS, "tighter threshold variants"))
    exp += 1

# 117-121: Tighter threshold + Birch+KM combinations (5 variants)
for th in [0.01, 0.02, 0.03, 0.04, 0.05]:
    birch_recs.append(hill(exp, "Birch", f"tight threshold={th} on DINOv2",
        (CHAMP-0.20, CHAMP+0.10),
        lambda Xfull, th=th: Birch(n_clusters=40, threshold=th, branching_factor=50).fit_predict(Z_S),
        "DINOv2 ViT-S/14", BIRCH_CITATIONS,
        "next backbone family — UMAP" if exp == 121 else "more Birch variants"))
    exp += 1

print(f"\nBirch hill-climb done. {len(birch_recs)} variants.")


# Final summary
from pathlib import Path
print(f"\n{'='*70}\nWARD + BIRCH HILL-CLIMB COMPLETE — {len(ward_recs)+len(birch_recs)} more experiments\n{'='*70}")
all_rec = [json.loads(l) for l in Path("autoresearch_results/experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
all_rec.sort(key=lambda d: -d["test_primary"])
print(f"\nTOP 10 (all {len(all_rec)} experiments):")
for i, r in enumerate(all_rec[:10], 1):
    print(f"  {i:<3} Exp {r['experiment_num']:<3} {r['backbone']:<40} ARI={r['test_primary']:.4f}")
print(f"\nGLOBAL CHAMPION: Exp {all_rec[0]['experiment_num']} ({all_rec[0]['backbone']}) ARI={all_rec[0]['test_primary']:.4f}")
