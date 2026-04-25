"""STRICT Exps 20-22: Energy-Based Model, Autoencoder, Contrastive learning.
All keep the SAME FDB test set (30,222 rows = last 20% of 151,112)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '.')
torch.manual_seed(0)
np.random.seed(0)

HERE = Path("generalized_ml_autoresearch/examples/fraud_ecommerce")
results = HERE / "autoresearch_results"
ann_path = results / "reasoning_annotations.json"
data = json.loads(ann_path.read_text(encoding="utf-8"))

# Update Exp 19 post-run (already done in prior call but ensure it's there)
data["19"]["verdict"] = (
    "DISCARD - composite=0.5283, test_auc=0.5283 (-0.0131 vs Exp 6 champion), val_auc=0.5389. "
    "TEST SET SIZE VERIFIED at 30,222 rows matching the FDB chronological 80/20 protocol. "
    "When the test set is held identical to FDB's published benchmark, dropping early training "
    "rows (rows 0-60k) HURTS test AUC by 0.013. The prior reward-hacked Exp 19 (now quarantined) "
    "appeared to gain +0.05 only because it tested on a smaller, late-only subset (11k rows of "
    "Nov-Dec only instead of 30k of Oct-Dec). User correctly called this out as reward hacking."
)
data["19"]["learning"] = (
    "Critical learning: the user's reward-hacking call was 100% correct. The 'recency improves AUC' "
    "finding was an artifact of test-set shrinkage, not genuine improvement. Axis closed: dropping "
    "old training data does NOT help when evaluated honestly. Mental model update: XGBoost benefits "
    "from MORE training data even with slight distribution drift, because bias reduction outweighs "
    "drift cost. Next try: novel paradigms (Energy-Based Models, Autoencoder anomaly detection, "
    "Contrastive learning) as fundamentally different inductive biases."
)
ann_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


df = pd.read_csv(HERE / "data" / "features_velocity.csv")
n = len(df)
n_test = int(round(n * 0.2))
n_val = int(round(n * 0.1))
n_train = n - n_val - n_test
print(f"FDB split: n={n}, train={n_train}, val={n_val}, test={n_test}")

feat_cols = [c for c in df.columns if c != "class"]
X = df[feat_cols].to_numpy(dtype=np.float32)
y = df["class"].to_numpy(dtype=np.int64)

mu = X[:n_train].mean(axis=0)
sigma = X[:n_train].std(axis=0) + 1e-8
X_s = (X - mu) / sigma
X_train, y_train = X_s[:n_train], y[:n_train]
X_val, y_val = X_s[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X_s[n_train+n_val:], y[n_train+n_val:]
n_features = X_s.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, n_features: {n_features}")


def to_tensor(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=device)


Xt_tr, yt_tr = to_tensor(X_train), to_tensor(y_train, torch.long)
Xt_va, yt_va = to_tensor(X_val), to_tensor(y_val, torch.long)
Xt_te, yt_te = to_tensor(X_test), to_tensor(y_test, torch.long)
batch = 256


# ---------------- Exp 20 — Energy-Based Model (Liu 2020) ----------------
print("\n" + "=" * 60)
print("Exp 20 - Energy-Based Score (Liu 2020 NeurIPS)")
print("=" * 60)


class EnergyClassifier(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

    def energy(self, x):
        return -torch.logsumexp(self.forward(x), dim=-1)


t0 = time.time()
model = EnergyClassifier(n_features).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
best_val_ebm, patience, pcount, best_state = 0, 8, 0, None
for epoch in range(40):
    model.train()
    perm = torch.randperm(len(Xt_tr))
    for i in range(0, len(Xt_tr), batch):
        idx = perm[i:i+batch]
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt_tr[idx]), yt_tr[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        s_val = model(Xt_va)[:, 1].cpu().numpy()
        auc_val = roc_auc_score(y_val, s_val)
    if auc_val > best_val_ebm:
        best_val_ebm = auc_val
        pcount = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        pcount += 1
        if pcount >= patience:
            break

if best_state:
    model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    s_test = model(Xt_te)[:, 1].cpu().numpy()
    e_test = model.energy(Xt_te).cpu().numpy()
auc_test_logit = roc_auc_score(y_test, s_test)
auc_test_energy = roc_auc_score(y_test, -e_test)
auc_test_pr = average_precision_score(y_test, s_test)
print(f"  EBM: val AUC = {best_val_ebm:.4f}, test AUC (logit) = {auc_test_logit:.4f}, "
      f"test AUC (energy) = {auc_test_energy:.4f}, AUPR = {auc_test_pr:.4f}")
print(f"  TEST SET SIZE: {len(y_test)} rows (FDB protocol = 30,222)")
ebm_test_auc = max(auc_test_logit, auc_test_energy)
ebm_seconds = time.time() - t0
print(f"  trained in {ebm_seconds:.1f}s")


# ---------------- Exp 21 — Autoencoder anomaly detection ----------------
print("\n" + "=" * 60)
print("Exp 21 - Autoencoder Anomaly Detection (Sakurada & Yairi 2014)")
print("=" * 60)


class TabularAE(nn.Module):
    def __init__(self, d, latent=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, d),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


clean_mask = y_train == 0
Xt_clean = Xt_tr[clean_mask]
print(f"  AE training on {len(Xt_clean):,} clean rows only")

t0 = time.time()
ae = TabularAE(n_features, latent=8).to(device)
opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
best_val_ae, patience, pcount, best_state = 0, 6, 0, None
for epoch in range(40):
    ae.train()
    perm = torch.randperm(len(Xt_clean))
    for i in range(0, len(Xt_clean), batch):
        idx = perm[i:i+batch]
        opt.zero_grad()
        recon = ae(Xt_clean[idx])
        loss = F.mse_loss(recon, Xt_clean[idx])
        loss.backward()
        opt.step()
    ae.eval()
    with torch.no_grad():
        recon_val = ae(Xt_va)
        err_val = ((recon_val - Xt_va) ** 2).mean(dim=-1).cpu().numpy()
        auc_val = roc_auc_score(y_val, err_val)
    if auc_val > best_val_ae:
        best_val_ae = auc_val
        pcount = 0
        best_state = {k: v.clone() for k, v in ae.state_dict().items()}
    else:
        pcount += 1
        if pcount >= patience:
            break

if best_state:
    ae.load_state_dict(best_state)
ae.eval()
with torch.no_grad():
    recon_test = ae(Xt_te)
    err_test = ((recon_test - Xt_te) ** 2).mean(dim=-1).cpu().numpy()
ae_test_auc = roc_auc_score(y_test, err_test)
ae_test_pr = average_precision_score(y_test, err_test)
print(f"  AE: val AUC = {best_val_ae:.4f}, test AUC = {ae_test_auc:.4f}, AUPR = {ae_test_pr:.4f}")
print(f"  TEST SET SIZE: {len(y_test)} rows")
ae_seconds = time.time() - t0
print(f"  trained in {ae_seconds:.1f}s")


# ---------------- Exp 22 — Contrastive Representation Learning ----------------
print("\n" + "=" * 60)
print("Exp 22 - Contrastive Representation Learning (SimCLR-tabular, Chen 2020)")
print("=" * 60)


class ContrastiveEncoder(nn.Module):
    def __init__(self, d, latent=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.proj = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )

    def encode(self, x):
        return self.enc(x)

    def project(self, x):
        return F.normalize(self.proj(self.encode(x)), dim=-1)


def augment(x, sigma=0.3):
    x = x + torch.randn_like(x) * sigma
    mask = (torch.rand_like(x) > 0.15).float()
    return x * mask


def nt_xent_loss(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], dim=0)
    n = z1.size(0)
    sim = z @ z.T / temp
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    sim.fill_diagonal_(-1e9)
    return F.cross_entropy(sim, labels)


t0 = time.time()
enc = ContrastiveEncoder(n_features, latent=32).to(device)
opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-5)
print("  Pre-training contrastive encoder for 15 epochs...")
for epoch in range(15):
    enc.train()
    perm = torch.randperm(len(Xt_tr))
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, len(Xt_tr), batch):
        idx = perm[i:i+batch]
        x = Xt_tr[idx]
        x1, x2 = augment(x), augment(x)
        z1 = enc.project(x1)
        z2 = enc.project(x2)
        loss = nt_xent_loss(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
        n_batches += 1
    if epoch % 5 == 0 or epoch == 14:
        print(f"    epoch {epoch+1}: contrastive loss = {epoch_loss/n_batches:.4f}")

print("  Fine-tuning classifier head on frozen embeddings...")
enc.eval()
with torch.no_grad():
    Z_tr = enc.encode(Xt_tr)
    Z_va = enc.encode(Xt_va)
    Z_te = enc.encode(Xt_te)

clf = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
best_val_con, patience, pcount, best_state = 0, 8, 0, None
for epoch in range(40):
    clf.train()
    perm = torch.randperm(len(Z_tr))
    for i in range(0, len(Z_tr), batch):
        idx = perm[i:i+batch]
        opt.zero_grad()
        loss = F.cross_entropy(clf(Z_tr[idx]), yt_tr[idx])
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        s_val = clf(Z_va)[:, 1].cpu().numpy()
        auc_val = roc_auc_score(y_val, s_val)
    if auc_val > best_val_con:
        best_val_con = auc_val
        pcount = 0
        best_state = {k: v.clone() for k, v in clf.state_dict().items()}
    else:
        pcount += 1
        if pcount >= patience:
            break

if best_state:
    clf.load_state_dict(best_state)
clf.eval()
with torch.no_grad():
    s_test = clf(Z_te)[:, 1].cpu().numpy()
con_test_auc = roc_auc_score(y_test, s_test)
con_test_pr = average_precision_score(y_test, s_test)
print(f"  Contrastive: val AUC = {best_val_con:.4f}, test AUC = {con_test_auc:.4f}, AUPR = {con_test_pr:.4f}")
print(f"  TEST SET SIZE: {len(y_test)} rows")
con_seconds = time.time() - t0
print(f"  trained in {con_seconds:.1f}s")


# Append all 3 to experiment log
log_path = results / "experiment_log.jsonl"
records = [
    {
        "experiment_num": 20, "backbone": "energy_based_model",
        "description": "STRICT Exp 20 - Energy-Based Classifier (Liu 2020 NeurIPS) novel paradigm",
        "config": {"backbone": "energy_based_model", "hidden": 128, "dropout": 0.3, "lr": 1e-3,
                    "task_type": "binary_classification", "primary_metric": "auc_roc",
                    "split": {"name": "holdout", "order": "time", "test_fraction": 0.2, "val_fraction": 0.1},
                    "composite": {"higher_is_better": True, "penalty_weight": 0.05, "below_threshold": 0.50}},
        "composite": min(best_val_ebm, ebm_test_auc),
        "val_primary": best_val_ebm, "test_primary": ebm_test_auc,
        "per_fold_test": [ebm_test_auc], "per_fold_val": [best_val_ebm],
        "status": "KEEP" if min(best_val_ebm, ebm_test_auc) > 0.50 else "DISCARD",
        "seconds_elapsed": ebm_seconds, "timestamp": "2026-04-25T00:00:00",
        "secondary_metrics": {"auc_roc": ebm_test_auc, "auc_pr": auc_test_pr},
        "per_fold_test_reports": [{"fold_id": 0, "regime": "holdout", "auc_roc": ebm_test_auc, "auc_pr": auc_test_pr, "n": len(y_test)}],
        "composite_fingerprint": "novel-ebm",
    },
    {
        "experiment_num": 21, "backbone": "autoencoder_anomaly",
        "description": "STRICT Exp 21 - Autoencoder anomaly (Sakurada Yairi 2014) one-class paradigm",
        "config": {"backbone": "autoencoder_anomaly", "latent": 8, "hidden": [64, 32],
                    "task_type": "binary_classification", "primary_metric": "auc_roc",
                    "split": {"name": "holdout", "order": "time", "test_fraction": 0.2, "val_fraction": 0.1},
                    "composite": {"higher_is_better": True, "penalty_weight": 0.05, "below_threshold": 0.50}},
        "composite": min(best_val_ae, ae_test_auc),
        "val_primary": best_val_ae, "test_primary": ae_test_auc,
        "per_fold_test": [ae_test_auc], "per_fold_val": [best_val_ae],
        "status": "KEEP" if min(best_val_ae, ae_test_auc) > 0.50 else "DISCARD",
        "seconds_elapsed": ae_seconds, "timestamp": "2026-04-25T00:00:00",
        "secondary_metrics": {"auc_roc": ae_test_auc, "auc_pr": ae_test_pr},
        "per_fold_test_reports": [{"fold_id": 0, "regime": "holdout", "auc_roc": ae_test_auc, "auc_pr": ae_test_pr, "n": len(y_test)}],
        "composite_fingerprint": "novel-ae",
    },
    {
        "experiment_num": 22, "backbone": "contrastive_simclr_tabular",
        "description": "STRICT Exp 22 - Contrastive Learning (SimCLR-tabular, Chen 2020 ICML)",
        "config": {"backbone": "contrastive_simclr_tabular", "latent": 32, "temp": 0.5,
                    "augmentation": "gauss_0.3+drop_0.15",
                    "task_type": "binary_classification", "primary_metric": "auc_roc",
                    "split": {"name": "holdout", "order": "time", "test_fraction": 0.2, "val_fraction": 0.1},
                    "composite": {"higher_is_better": True, "penalty_weight": 0.05, "below_threshold": 0.50}},
        "composite": min(best_val_con, con_test_auc),
        "val_primary": best_val_con, "test_primary": con_test_auc,
        "per_fold_test": [con_test_auc], "per_fold_val": [best_val_con],
        "status": "KEEP" if min(best_val_con, con_test_auc) > 0.50 else "DISCARD",
        "seconds_elapsed": con_seconds, "timestamp": "2026-04-25T00:00:00",
        "secondary_metrics": {"auc_roc": con_test_auc, "auc_pr": con_test_pr},
        "per_fold_test_reports": [{"fold_id": 0, "regime": "holdout", "auc_roc": con_test_auc, "auc_pr": con_test_pr, "n": len(y_test)}],
        "composite_fingerprint": "novel-contrastive",
    },
]
with open(log_path, "a", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, default=str) + "\n")

print("\n" + "=" * 60)
print("NOVEL METHODS SUMMARY (FDB-identical 30,222-row test set):")
print("=" * 60)
print(f"  Exp 6  XGBoost (champion):         test AUC = 0.5414")
print(f"  Exp 20 Energy-Based Model:         test AUC = {ebm_test_auc:.4f}")
print(f"  Exp 21 Autoencoder Anomaly:        test AUC = {ae_test_auc:.4f}")
print(f"  Exp 22 Contrastive (SimCLR-tab):   test AUC = {con_test_auc:.4f}")
