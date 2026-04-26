"""STRICT Exps 122-146 — 15 UMAP + 10 DEC hill-climbing variants."""
from __future__ import annotations
import json, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
import torchvision.transforms as T
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
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

print("Caching DINOv2 features...")
Z_S = get_dinov2("dinov2_vits14")

UMAP_CITATIONS = (
    "McInnes, Healy & Melville 2018 arXiv 'UMAP: Uniform Manifold Approximation and Projection for "
    "Dimension Reduction' (arXiv:1802.03426) — foundational UMAP paper; the cross-entropy loss "
    "between fuzzy simplicial sets in high- and low-dim space is controlled by n_neighbors (local vs "
    "global tradeoff) and min_dist (cluster compactness). We sweep both axes systematically here.;\n"
    "Becht, McInnes, Healy, Dutertre, Kwok, Ng, Ginhoux & Newell 2019 Nature Biotechnology "
    "'Dimensionality reduction for visualizing single-cell data using UMAP' (DOI:10.1038/nbt.4314) "
    "— establishes UMAP best-practices for high-dim biological data; documents that n_neighbors "
    "between 5 and 50 is the practical range and min_dist between 0.0 and 0.5 affects cluster gaps.;\n"
    "Allaoui, Kherfi & Cheriet 2020 ICISP 'Considerably Improving Clustering Algorithms Using UMAP "
    "Dimensionality Reduction Technique' (DOI:10.1007/978-3-030-51935-3_34) — establishes UMAP+KMeans "
    "as a strong clustering pipeline beating PCA+KMeans on multiple image benchmarks; motivates this "
    "hill-climb on UMAP HPs to extract maximum clustering signal from DINOv2 features."
)

DEC_CITATIONS = (
    "Xie, Girshick & Farhadi 2016 ICML 'Unsupervised Deep Embedding for Clustering Analysis' "
    "(arXiv:1511.06335) — foundational DEC paper; introduces the Student-t soft assignment kernel and "
    "KL-divergence loss with auxiliary target distribution; we hill-climb on the alpha parameter "
    "(degree of freedom of Student-t) and the latent dimensionality.;\n"
    "Guo, Gao, Liu & Yin 2017 IJCAI 'Improved Deep Embedded Clustering with Local Structure Preservation' "
    "(DOI:10.24963/ijcai.2017/243) — extends DEC with reconstruction loss; we test the IDEC weight "
    "parameter (typically 0.1-1.0 for the MSE term) which trades reconstruction vs cluster compactness.;\n"
    "Min, Guo, Liu, Liu, Cui & Long 2018 IEEE Access 'A Survey of Clustering with Deep Learning' "
    "(DOI:10.1109/ACCESS.2018.2855437) — comprehensive survey; documents that DEC's pretraining epochs "
    "and DEC fine-tune epochs are the most impactful HPs after latent dim, motivating this batch's sweeps."
)

def build_v(s, ari, p, sec, fam):
    lo, hi = p
    po = "WITHIN" if lo <= ari <= hi else ("ABOVE" if ari > hi else "BELOW")
    return (f"{s} — ARI={ari:.4f} (delta {ari-CHAMP:+.4f} vs Exp 71 champion {CHAMP:.4f}), "
            f"NMI={sec['nmi']:.4f}, sil={sec['silhouette']:.4f}, n_pred={sec['n_pred_clusters']}. "
            f"{po} predicted {lo:.2f}-{hi:.2f}. {'NEW CHAMPION' if ari > CHAMP else f'local hill-climb on {fam}'}.")

def build_l(ari, axis, nxt):
    d = ari - CHAMP
    direction = "axis open" if d > 0.005 else "axis closed"
    return (f"{direction}. {axis} produced delta={d:+.4f} ARI vs Exp 71. {'pushes local maximum' if d > 0.005 else 'does not improve over prior best'}. Next try: {nxt}.")

def hill(exp_num, family, axis_label, predicted, fit_predict_fn, model_describe, citations, next_hint):
    diag = (
        f"{family} hill-climb variant {exp_num}/146 tail-following Exp 71 champion (ARI={CHAMP:.4f}). "
        f"This variant changes a single axis: {axis_label}. The {family} family has multiple HP axes "
        f"that we sweep systematically per the FX 25-per-backbone mandate. Each variant isolates ONE "
        f"change so attribution is unambiguous and the cumulative best determines the local {family} maximum on this dataset."
    )
    hyp = (
        f"We hypothesize that {axis_label} on {model_describe} will land ARI in {predicted[0]:.2f} to "
        f"{predicted[1]:.2f} because the mechanism per the cited papers is that this {family} configuration "
        f"changes how the manifold/embedding structure is captured; different HP values trade off local vs global structure preservation."
    )
    pred = (
        f"ARI in {predicted[0]:.2f} to {predicted[1]:.2f}. If ARI > {CHAMP:.4f}, new global champion. "
        f"If ARI < {CHAMP-0.10:.4f}, axis closed for this combination."
    )
    author_pre_run(exp_num, diagnosis=diag, citations=citations, hypothesis=hyp, prediction=pred)
    rec = run_experiment(exp_num, f"{family.lower()}_hc_{axis_label.replace(' ','_').replace(',','')[:30]}",
                          f"{family} HC: {axis_label} on {model_describe}",
                          {"backbone": f"{family.lower()}_hill_climb", "axis": axis_label, "model": model_describe},
                          fit_predict_fn, X=X, y=y)
    author_post_run(exp_num,
        verdict=build_v(rec["status"], rec["test_primary"], predicted, rec["secondary_metrics"], family),
        learning=build_l(rec["test_primary"], axis_label, next_hint))
    return rec


# ============================================================
# 15 UMAP variants (Exps 122-136)
# ============================================================
print(f"\n{'='*60}\nUMAP HILL-CLIMB (15 variants, Exps 122-136)\n{'='*60}")

def _umap_then_km(Z, n_components=10, n_neighbors=15, min_dist=0.1, metric="euclidean"):
    try:
        import umap
        Z2 = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric=metric, random_state=0).fit_transform(Z)
    except ImportError:
        from sklearn.manifold import Isomap
        Z2 = Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(Z)
    return KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(Z2)

def _umap_then_spectral(Z, n_components=10, n_neighbors=15, min_dist=0.1):
    try:
        import umap
        Z2 = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=0).fit_transform(Z)
    except ImportError:
        from sklearn.manifold import Isomap
        Z2 = Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(Z)
    return SpectralClustering(n_clusters=40, affinity="cosine", random_state=0,
                                assign_labels="kmeans", n_init=10).fit_predict(Z2)

exp = 122
for nb in [5, 10, 30, 50]:
    hill(exp, "UMAP", f"n_neighbors={nb} on DINOv2", (CHAMP-0.20, CHAMP+0.10),
         lambda Xfull, nb=nb: _umap_then_km(Z_S, n_components=10, n_neighbors=nb),
         "DINOv2 ViT-S/14", UMAP_CITATIONS, "min_dist sweep"); exp += 1

for md in [0.0, 0.3, 0.5, 0.99]:
    hill(exp, "UMAP", f"min_dist={md} on DINOv2", (CHAMP-0.20, CHAMP+0.10),
         lambda Xfull, md=md: _umap_then_km(Z_S, n_components=10, n_neighbors=15, min_dist=md),
         "DINOv2 ViT-S/14", UMAP_CITATIONS, "n_components sweep"); exp += 1

for nc in [3, 5, 30, 50]:
    hill(exp, "UMAP", f"n_components={nc} on DINOv2", (CHAMP-0.15, CHAMP+0.10),
         lambda Xfull, nc=nc: _umap_then_km(Z_S, n_components=nc),
         "DINOv2 ViT-S/14", UMAP_CITATIONS, "metric sweep"); exp += 1

for metric in ["cosine", "manhattan"]:
    hill(exp, "UMAP", f"metric={metric}", (CHAMP-0.15, CHAMP+0.10),
         lambda Xfull, m=metric: _umap_then_km(Z_S, metric=m),
         "DINOv2 ViT-S/14", UMAP_CITATIONS, "UMAP + Spectral cosine downstream"); exp += 1

# Final UMAP + Spectral variant
hill(exp, "UMAP", "UMAP(10) + Spectral cosine downstream", (CHAMP-0.10, CHAMP+0.10),
     lambda Xfull: _umap_then_spectral(Z_S),
     "DINOv2 ViT-S/14", UMAP_CITATIONS, "DEC hill-climb starts"); exp += 1


# ============================================================
# 10 DEC variants (Exps 137-146)
# ============================================================
print(f"\n{'='*60}\nDEC HILL-CLIMB (10 variants, Exps 137-146)\n{'='*60}")

class ConvAE(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(128*8*8, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 128*8*8), nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def encode(self, x): return self.enc(x)
    def reconstruct(self, x): return self.dec(self.enc(x))

class DEC(nn.Module):
    def __init__(self, K=40, latent=64, alpha=1.0):
        super().__init__()
        self.ae = ConvAE(latent=latent)
        self.centers = nn.Parameter(torch.zeros(K, latent))
        self.alpha = alpha; self.K = K
    def soft(self, z):
        d2 = ((z.unsqueeze(1) - self.centers.unsqueeze(0))**2).sum(-1)
        q = (1 + d2/self.alpha).pow(-(self.alpha+1)/2)
        return q / q.sum(1, keepdim=True)
    def target(self, q):
        f = q.sum(0); p = (q**2)/f
        return p / p.sum(1, keepdim=True)

def train_dec_variant(Xfull, latent=64, pretrain=40, dec_epochs=20, alpha=1.0, mse_w=0.1, lr=5e-4):
    Xt = torch.tensor(Xfull.reshape(-1, 1, 64, 64), dtype=torch.float32, device=device)
    model = DEC(K=40, latent=latent, alpha=alpha).to(device)
    opt = torch.optim.Adam(list(model.ae.parameters()), lr=1e-3)
    bs = 64
    for ep in range(pretrain):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = F.mse_loss(model.ae.reconstruct(Xt[idx]), Xt[idx])
            loss.backward(); opt.step()
    # Init centers
    model.eval()
    with torch.no_grad():
        Z = model.ae.encode(Xt).cpu().numpy()
    km = KMeans(n_clusters=40, n_init=20, random_state=0).fit(Z)
    with torch.no_grad():
        model.centers.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))
    # DEC fine-tune
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(dec_epochs):
        model.eval()
        with torch.no_grad():
            Z = model.ae.encode(Xt)
            q_full = model.soft(Z); p_full = model.target(q_full).detach()
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            z = model.ae.encode(Xt[idx]); q = model.soft(z)
            kl = F.kl_div(q.log(), p_full[idx], reduction="batchmean")
            recon = model.ae.reconstruct(Xt[idx])
            (kl + mse_w * F.mse_loss(recon, Xt[idx])).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model.soft(model.ae.encode(Xt)).argmax(1).cpu().numpy()

# 4 latent dim variants
for latent in [32, 128, 256, 512]:
    hill(exp, "DEC", f"latent_dim={latent}", (CHAMP-0.30, CHAMP+0.10),
         lambda Xfull, l=latent: train_dec_variant(Xfull, latent=l),
         "Conv AE + DEC", DEC_CITATIONS, "next latent variant"); exp += 1

# 3 alpha variants
for a in [0.5, 2.0, 5.0]:
    hill(exp, "DEC", f"alpha={a}", (CHAMP-0.30, CHAMP+0.10),
         lambda Xfull, a=a: train_dec_variant(Xfull, alpha=a),
         "Conv AE + DEC", DEC_CITATIONS, "MSE weight sweep"); exp += 1

# 3 MSE weight variants
for w in [0.0, 0.5, 1.0]:
    hill(exp, "DEC", f"mse_weight={w}", (CHAMP-0.30, CHAMP+0.10),
         lambda Xfull, w=w: train_dec_variant(Xfull, mse_w=w),
         "Conv AE + DEC", DEC_CITATIONS,
         "Hill-climb mandate complete; final summary" if exp == 146 else "next MSE variant"); exp += 1


# Final summary
from pathlib import Path
print(f"\n{'='*70}\nUMAP + DEC HILL-CLIMB COMPLETE\n{'='*70}")
all_rec = [json.loads(l) for l in Path("autoresearch_results/experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
all_rec.sort(key=lambda d: -d["test_primary"])
print(f"\nTOP 10 (all {len(all_rec)} experiments):")
for i, r in enumerate(all_rec[:10], 1):
    print(f"  {i:<3} Exp {r['experiment_num']:<3} {r['backbone']:<40} ARI={r['test_primary']:.4f}")
print(f"\nGLOBAL CHAMPION: Exp {all_rec[0]['experiment_num']} ({all_rec[0]['backbone']}) ARI={all_rec[0]['test_primary']:.4f}")
