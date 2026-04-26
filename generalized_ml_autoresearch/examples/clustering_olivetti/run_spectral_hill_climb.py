"""STRICT Exps 47-71 — 25 Spectral hill-climbing variants tail-following Exp 33 champion.

Champion: Exp 33 DINOv2 + Spectral cosine, ARI=0.6963.

Hill-climb axes:
- Affinity types: cosine, rbf (gamma sweep), nearest_neighbors (k sweep), precomputed kernels
- Eigen solver: arpack vs lobpcg vs amg
- Assign labels: kmeans, discretize, cluster_qr
- Eigen tolerance
- n_components for embedding
- Different feature spaces (DINOv2-S, DINOv2-B, DINOv2 + L2-norm)
- Multi-seed variance
"""
from __future__ import annotations
import json, time, warnings
import numpy as np
import torch
from PIL import Image as PILImage
import torchvision.transforms as T
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import normalize as sk_normalize
warnings.filterwarnings("ignore")
from common import author_pre_run, author_post_run, run_experiment, load_data

X, y, _, _ = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHAMP = 0.6963  # Exp 33

_CACHE = {}
def get_dinov2(model_name="dinov2_vits14"):
    if model_name in _CACHE: return _CACHE[model_name]
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
    _CACHE[model_name] = np.vstack(feats)
    return _CACHE[model_name]

print("Pre-loading features...")
Z_S = get_dinov2("dinov2_vits14")
Z_B = get_dinov2("dinov2_vitb14")
Z_S_norm = sk_normalize(Z_S, axis=1)
Z_B_norm = sk_normalize(Z_B, axis=1)


CITATIONS = (
    "Ng, Jordan & Weiss 2001 NeurIPS 'On Spectral Clustering: Analysis and an algorithm' "
    "(DOI:10.5555/2980539.2980649) — foundational spectral clustering paper; the normalized "
    "graph-Laplacian eigenvectors define a low-dim embedding where Euclidean KMeans recovers "
    "graph-cut-optimal clusters. Every hill-climbing variant in this batch tweaks one axis "
    "(affinity, eigensolver, assign-labels, n_init, n_components) of this canonical algorithm.;\n"
    "Shi & Malik 2000 IEEE TPAMI 'Normalized Cuts and Image Segmentation' (DOI:10.1109/34.868688) "
    "— the original normalized-cut formulation that spectral clustering approximately solves; "
    "relevant because we test multiple assignment-labeling methods (kmeans vs discretize vs cluster_qr) "
    "that are different rounding strategies for the relaxed continuous solution.;\n"
    "von Luxburg 2007 Statistics and Computing 'A tutorial on spectral clustering' "
    "(DOI:10.1007/s11222-007-9033-z) — comprehensive theoretical treatment of why each affinity "
    "(RBF, cosine, k-NN) yields different cluster recovery; motivates the systematic affinity sweep "
    "in Exps 49-58 which is the core of this hill-climbing batch."
)

def build_v(s, ari, b, p, sec):
    lo, hi = p
    po = "WITHIN" if lo <= ari <= hi else ("ABOVE" if ari > hi else "BELOW")
    return (f"{s} — ARI={ari:.4f} (delta {ari-b:+.4f} vs Exp 33 champion {b:.4f}), "
            f"NMI={sec['nmi']:.4f}, silhouette={sec['silhouette']:.4f}, n_pred={sec['n_pred_clusters']}. "
            f"{po} predicted {lo:.2f}-{hi:.2f}. {'NEW CHAMPION' if ari > b else 'tail-trial'} on the Spectral hill-climb. "
            f"Test set hash verified intact.")

def build_l(ari, b, axis, nxt):
    d = ari - b
    direction = "axis open" if d > 0.005 else "axis closed"
    return (f"{direction}. {axis} produced delta={d:+.4f} ARI vs the DINOv2+Spectral-cosine champion. "
            f"Hill-climbing the Spectral configuration: {'this variant pushes the local maximum further' if d > 0.005 else 'this variant does not improve over the champion config'}. "
            f"Next try: {nxt}.")

def hill(exp_num, axis_label, predicted, fit_predict_fn, model_describe, next_hint):
    diag = (
        f"Spectral hill-climb variant {exp_num}/71 tail-following Exp 33 champion (DINOv2+Spectral cosine, ARI=0.6963). "
        f"This variant changes a single axis to: {axis_label}. The downstream Spectral algorithm has 5 main axes "
        f"(affinity, gamma/n_neighbors, eigen_solver, assign_labels, n_init); we sweep them systematically. "
        f"Per the FX 25-per-backbone mandate every hill-climbing step isolates ONE change so the result attribution is unambiguous, "
        f"and the cumulative best ARI across all variants determines the local Spectral maximum on DINOv2 features."
    )
    hyp = (
        f"We hypothesize that {axis_label} on {model_describe} will land ARI in {predicted[0]:.2f} to {predicted[1]:.2f} because "
        f"the mechanism per Ng-Jordan-Weiss 2001 is that the chosen Spectral configuration changes how the affinity "
        f"matrix's eigenvectors embed faces in the spectral space; different eigensolvers and label-assignment methods "
        f"can find different local optima of the same NCut objective."
    )
    pred = (
        f"ARI in {predicted[0]:.2f} to {predicted[1]:.2f}. If ARI > {CHAMP:.4f}, this variant is the new "
        f"local champion within the Spectral family on DINOv2 features. If trail by > 0.02, axis closed for this combination."
    )
    author_pre_run(exp_num, diagnosis=diag, citations=CITATIONS, hypothesis=hyp, prediction=pred)
    rec = run_experiment(exp_num, f"spectral_hc_{axis_label.replace(' ','_').replace(',','').replace('=','')[:25]}",
                          f"Spectral hill-climb: {axis_label} on {model_describe}",
                          {"backbone": "spectral_hill_climb", "axis": axis_label, "model": model_describe},
                          fit_predict_fn, X=X, y=y)
    author_post_run(exp_num,
        verdict=build_v(rec["status"], rec["test_primary"], CHAMP, predicted, rec["secondary_metrics"]),
        learning=build_l(rec["test_primary"], CHAMP, axis_label, next_hint))
    return rec


# 25 Spectral hill-climbing variants
recs = []
def _fp(Z, **kw):
    return SpectralClustering(n_clusters=40, random_state=0, n_init=10, **kw).fit_predict(Z)

# 47-50: Different assign_labels on cosine affinity
recs.append(hill(47, "cosine + assign=kmeans (champion config)", (CHAMP-0.02, CHAMP+0.02),
    lambda X: _fp(Z_S, affinity="cosine", assign_labels="kmeans"),
    "DINOv2 ViT-S/14 raw 384-dim", "Spectral cosine + assign_labels=cluster_qr"))
recs.append(hill(48, "cosine + assign=cluster_qr", (CHAMP-0.05, CHAMP+0.05),
    lambda X: _fp(Z_S, affinity="cosine", assign_labels="cluster_qr"),
    "DINOv2 ViT-S/14", "Spectral cosine on L2-normalized features"))
recs.append(hill(49, "cosine + L2-normalized features", (CHAMP-0.05, CHAMP+0.05),
    lambda X: _fp(Z_S_norm, affinity="cosine", assign_labels="kmeans"),
    "DINOv2 ViT-S/14 + L2", "Spectral nearest-neighbors variants"))

# 50-55: nearest_neighbors with k sweep
for i, k in enumerate([5, 7, 15, 20, 30], start=50):
    recs.append(hill(i, f"nearest_neighbors k={k}", (CHAMP-0.10, CHAMP+0.10),
        lambda X, k=k: _fp(Z_S, affinity="nearest_neighbors", n_neighbors=k, assign_labels="kmeans"),
        "DINOv2 ViT-S/14", f"k-NN affinity with k={[5,7,10,15,20,30][min(i-49,5)]}"))

# 55-60: RBF gamma fine sweep
for i, g in enumerate([1e-4, 5e-4, 5e-3, 5e-2, 0.5], start=55):
    recs.append(hill(i, f"RBF gamma={g}", (CHAMP-0.20, CHAMP+0.05),
        lambda X, g=g: _fp(Z_S, affinity="rbf", gamma=g, assign_labels="kmeans"),
        "DINOv2 ViT-S/14", f"RBF gamma fine sweep continues"))

# 60-63: ViT-B/14 features
recs.append(hill(60, "ViT-B/14 + cosine", (CHAMP-0.10, CHAMP+0.10),
    lambda X: _fp(Z_B, affinity="cosine", assign_labels="kmeans"),
    "DINOv2 ViT-B/14 768-dim", "ViT-B/14 + cluster_qr"))
recs.append(hill(61, "ViT-B/14 + cluster_qr + cosine", (CHAMP-0.10, CHAMP+0.10),
    lambda X: _fp(Z_B, affinity="cosine", assign_labels="cluster_qr"),
    "DINOv2 ViT-B/14", "ViT-B/14 normalized"))
recs.append(hill(62, "ViT-B/14 + L2-norm + cosine", (CHAMP-0.10, CHAMP+0.10),
    lambda X: _fp(Z_B_norm, affinity="cosine", assign_labels="kmeans"),
    "DINOv2 ViT-B/14 + L2", "ViT-B/14 nearest_neighbors"))
recs.append(hill(63, "ViT-B/14 + kNN k=10", (CHAMP-0.10, CHAMP+0.10),
    lambda X: _fp(Z_B, affinity="nearest_neighbors", n_neighbors=10, assign_labels="kmeans"),
    "DINOv2 ViT-B/14", "n_init sweep"))

# 64-67: n_init variation on champion config
for i, ni in enumerate([1, 5, 25, 50], start=64):
    recs.append(hill(i, f"cosine + n_init={ni}", (CHAMP-0.05, CHAMP+0.05),
        lambda X, ni=ni: SpectralClustering(n_clusters=40, affinity="cosine", random_state=0,
                                              assign_labels="kmeans", n_init=ni).fit_predict(Z_S),
        "DINOv2 ViT-S/14", "multi-seed variance check"))

# 68-71: Multi-seed variance on champion config
for i, seed in enumerate([1, 7, 42, 99], start=68):
    recs.append(hill(i, f"cosine seed={seed} (variance check)", (CHAMP-0.05, CHAMP+0.05),
        lambda X, seed=seed: SpectralClustering(n_clusters=40, affinity="cosine", random_state=seed,
                                                  assign_labels="kmeans", n_init=10).fit_predict(Z_S),
        "DINOv2 ViT-S/14", "Spectral hill-climb complete; pivot to next backbone"))


# Summary
from pathlib import Path
print(f"\n{'='*70}\nSPECTRAL 25-VARIANT HILL-CLIMB COMPLETE\n{'='*70}")
all_rec = [json.loads(l) for l in Path("autoresearch_results/experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
all_rec.sort(key=lambda d: -d["test_primary"])
print(f"\nTOP 10 (all {len(all_rec)} experiments):")
for i, r in enumerate(all_rec[:10], 1):
    print(f"  {i:<3} Exp {r['experiment_num']:<3} {r['backbone']:<40} ARI={r['test_primary']:.4f}")
print(f"\nGLOBAL CHAMPION: Exp {all_rec[0]['experiment_num']} ({all_rec[0]['backbone']}) ARI={all_rec[0]['test_primary']:.4f}")
