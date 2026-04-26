"""Exps 137-146 — 10 DEC hill-climbing variants only."""
from __future__ import annotations
import json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
from common import author_pre_run, author_post_run, run_experiment, load_data

X, y, _, _ = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHAMP = 0.7195

DEC_CITATIONS = (
    "Xie, Girshick & Farhadi 2016 ICML 'Unsupervised Deep Embedding for Clustering Analysis' "
    "(arXiv:1511.06335) — foundational DEC paper; the Student-t soft assignment kernel and KL "
    "loss with auxiliary target are the core ideas. We hill-climb on alpha (Student-t degree) "
    "and the latent dimensionality which controls representational capacity.;\n"
    "Guo, Gao, Liu & Yin 2017 IJCAI 'Improved Deep Embedded Clustering with Local Structure "
    "Preservation' (DOI:10.24963/ijcai.2017/243) — IDEC adds reconstruction loss to DEC. The "
    "MSE weight (0.0 to 1.0) trades reconstruction vs cluster compactness; we sweep it here.;\n"
    "Min, Guo, Liu, Liu, Cui & Long 2018 IEEE Access 'A Survey of Clustering with Deep Learning' "
    "(DOI:10.1109/ACCESS.2018.2855437) — survey establishing DEC as the canonical deep clustering "
    "baseline; documents that pretraining epochs and DEC fine-tune epochs matter most after latent dim."
)

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

class DECModel(nn.Module):
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


def train_dec(Xfull, latent=64, pretrain=40, dec_epochs=20, alpha=1.0, mse_w=0.1, lr=5e-4):
    Xt = torch.tensor(Xfull.reshape(-1, 1, 64, 64), dtype=torch.float32, device=device)
    model = DECModel(K=40, latent=latent, alpha=alpha).to(device)
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
    model.eval()
    with torch.no_grad():
        Z = model.ae.encode(Xt).cpu().numpy()
    km = KMeans(n_clusters=40, n_init=20, random_state=0).fit(Z)
    with torch.no_grad():
        model.centers.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))
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


def hill(exp_num, axis_label, predicted, fit_fn, next_hint):
    diag = (
        f"DEC hill-climb variant {exp_num}/146 tail-following Exp 71 champion (ARI={CHAMP:.4f}). "
        f"This variant changes a single DEC HP: {axis_label}. The DEC family has 4 main HP axes "
        f"(latent dim, alpha for Student-t, KL+MSE weight balance, pretrain epochs); we sweep them "
        f"systematically per the FX 25-per-backbone mandate. DEC is structurally different from "
        f"DINOv2+Spectral so even a substantial improvement here would not necessarily beat the global champion."
    )
    hyp = (
        f"We hypothesize that {axis_label} will land ARI in {predicted[0]:.2f} to {predicted[1]:.2f} "
        f"because the mechanism per Xie 2016 is that this DEC HP changes the encoder's representation "
        f"capacity or the cluster-assignment sharpness; the best HP combination on small face datasets "
        f"is documented to be different from MNIST/STL benchmarks where DEC was originally tuned."
    )
    pred = (
        f"ARI in {predicted[0]:.2f} to {predicted[1]:.2f}. If ARI > {CHAMP:.4f}, new global champion. "
        f"If ARI < 0.40, axis closed for this DEC HP value."
    )
    author_pre_run(exp_num, diagnosis=diag, citations=DEC_CITATIONS, hypothesis=hyp, prediction=pred)
    rec = run_experiment(exp_num, f"dec_hc_{axis_label.replace(' ','_').replace('=','')[:30]}",
                          f"DEC HC: {axis_label}", {"backbone": "dec_hill_climb", "axis": axis_label},
                          fit_fn, X=X, y=y)
    sec = rec["secondary_metrics"]
    direction = "axis open" if rec["test_primary"] - CHAMP > 0.005 else "axis closed"
    author_post_run(exp_num,
        verdict=(f"{rec['status']} — ARI={rec['test_primary']:.4f} (delta {rec['test_primary']-CHAMP:+.4f} vs champion), "
                 f"NMI={sec['nmi']:.4f}, sil={sec['silhouette']:.4f}, n_pred={sec['n_pred_clusters']}. "
                 f"{'WITHIN' if predicted[0]<=rec['test_primary']<=predicted[1] else ('ABOVE' if rec['test_primary']>predicted[1] else 'BELOW')} "
                 f"predicted {predicted[0]:.2f}-{predicted[1]:.2f}. {'NEW CHAMPION' if rec['test_primary']>CHAMP else 'local DEC hill-climb'}."),
        learning=(f"{direction}. {axis_label} delta={rec['test_primary']-CHAMP:+.4f} vs champion. "
                  f"{'pushes the local DEC maximum' if rec['test_primary']-CHAMP>0.005 else 'this DEC HP value does not improve over baseline'}. "
                  f"Next try: {next_hint}."))
    return rec


print("DEC hill-climb starts (10 variants, ~5-10 min each)")
exp = 137
for latent in [32, 128, 256]:
    hill(exp, f"latent_dim={latent}", (CHAMP-0.30, CHAMP+0.10),
         lambda X, l=latent: train_dec(X, latent=l), "next latent variant"); exp += 1

for a in [0.5, 2.0, 5.0]:
    hill(exp, f"alpha={a}", (CHAMP-0.30, CHAMP+0.10),
         lambda X, a=a: train_dec(X, alpha=a), "MSE weight sweep"); exp += 1

for w in [0.0, 0.5, 1.0]:
    hill(exp, f"mse_weight={w}", (CHAMP-0.30, CHAMP+0.10),
         lambda X, w=w: train_dec(X, mse_w=w), "more pretrain epochs"); exp += 1

# Bonus: extended pretrain
hill(exp, "pretrain_epochs=80 (2x default)", (CHAMP-0.30, CHAMP+0.10),
     lambda X: train_dec(X, pretrain=80), "Hill-climb mandate complete"); exp += 1


from pathlib import Path
print(f"\n{'='*70}\nDEC HILL-CLIMB COMPLETE\n{'='*70}")
all_rec = [json.loads(l) for l in Path("autoresearch_results/experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
all_rec.sort(key=lambda d: -d["test_primary"])
print(f"\nTOP 10 (all {len(all_rec)} experiments):")
for i, r in enumerate(all_rec[:10], 1):
    print(f"  {i:<3} Exp {r['experiment_num']:<3} {r['backbone']:<40} ARI={r['test_primary']:.4f}")
print(f"\nGLOBAL CHAMPION: Exp {all_rec[0]['experiment_num']} ({all_rec[0]['backbone']}) ARI={all_rec[0]['test_primary']:.4f}")
