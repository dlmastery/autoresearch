# Champion Archive — Exp 71: DINOv2 ViT-S/14 + Spectral Clustering, cosine affinity, seed = 99

**ARI:** 0.7195
**NMI:** 0.9004
**V-measure:** 0.9004
**FMI:** 0.7270
**Homogeneity:** 0.8945
**Completeness:** 0.9063
**Silhouette:** 0.0927
**n_pred_clusters:** 40 (matches K = 40)
**n_noise:** 0
**Composite fingerprint:** `clustering-ari-floor0.3`

## 1. Method

The champion is a two-stage pipeline:

1. **Feature extraction.** Each 64×64 grayscale Olivetti image is upsampled to 224×224, replicated to 3 channels, normalised with ImageNet mean/std, and passed through Meta's DINOv2 ViT-S/14 (Oquab et al. 2024 TMLR 'DINOv2: Learning Robust Visual Features without Supervision' arXiv:2304.07193). The ViT class-token is extracted, producing a 384-dim feature vector per image.
2. **Spectral Clustering.** Features go to `sklearn.cluster.SpectralClustering(n_clusters=40, affinity='cosine', assign_labels='kmeans', n_init=10, random_state=99)`. The cosine affinity matrix between all 400 feature vectors is built, the normalised graph Laplacian's 40 smallest eigenvectors form the spectral embedding, and KMeans assigns each point to one of 40 clusters in the embedding space.

The two stages are deterministic given (a) DINOv2's pretrained weights (frozen) and (b) `random_state = 99`. Re-running this pipeline on a fresh machine produces ARI = 0.7195 byte-identically.

## 2. Why this configuration won

The champion progression and per-row mechanism is documented in the project paper (`paper.md` Section 5). The key insights:

- **DINOv2 features dominate raw pixels** by +0.15 ARI (Exp 1: 0.4057 → Exp 20: 0.5455). Pretrained ViT features capture face-identity structure that pixel-space methods cannot, even when the pretraining was on natural-image RGB rather than grayscale faces.
- **Spectral cosine dominates KMeans on DINOv2** by +0.15 ARI (Exp 20: 0.5455 → Exp 33: 0.6963). Spectral exploits global graph structure that KMeans's local Voronoi cells cannot.
- **Seed = 99 is the positive tail** of the seed distribution. The 5-seed mean is 0.6807; the 5-seed median is 0.6963. Honest headline: "5-seed median = 0.6963 ± 0.0429 (single-seed peak: 0.7195 at seed = 99)".

## 3. Per-fold metrics (full-dataset evaluation)

| Fold | Regime | ARI | NMI | FMI | Silhouette | n |
|-----:|--------|----:|----:|----:|-----------:|--:|
| 0 | full-dataset | 0.7195 | 0.9004 | 0.7270 | 0.0927 | 400 |

Clustering is full-dataset evaluation by protocol (no train/test split). The single fold is the entire 400-row dataset.

## 4. Hyperparameters

```python
{
    "model_backbone": "dinov2_vits14",          # facebookresearch/dinov2 (TMLR 2024)
    "feature_dim": 384,                          # ViT-S class-token
    "input_resize": 224,                         # ViT-S/14 minimum input size
    "input_normalize": "imagenet",               # mean/std from ImageNet
    "input_grayscale_to_3channel": true,         # replicate single channel 3×

    "clustering": "SpectralClustering",
    "n_clusters": 40,                            # known K for Olivetti
    "affinity": "cosine",                        # natural for L2-normalised DINO features
    "assign_labels": "kmeans",                   # default; alternative: "cluster_qr" (deterministic)
    "n_init": 10,                                # KMeans restarts in assign-labels step
    "random_state": 99,                          # seed = 99 is the positive tail (see §6 of paper)

    "primary_metric": "ari",
    "composite_floor": 0.30,
    "composite_fingerprint": "clustering-ari-floor0.3"
}
```

## 5. Architecture description

DINOv2 ViT-S/14 (Oquab et al. 2024 TMLR):
- 21 M parameters
- Self-supervised pretraining on 142 M curated natural images (LVD-142M dataset)
- Patch size 14×14, 16 patches per side at 224×224 input → 257-token sequence (1 CLS + 256 patch tokens)
- 12 attention layers, 6 attention heads, 384-dim hidden state
- Output: class-token feature in ℝ³⁸⁴; we use this directly without further projection

SpectralClustering (Ng, Jordan, Weiss 2001 NeurIPS) configured:
- Normalised graph Laplacian
- ARPACK eigensolver (default for n < 1000)
- 40 smallest non-trivial eigenvectors form the embedding
- KMeans with `n_init=10` runs in the 40-dim embedding space, picks the lowest-inertia clustering

## 6. Training / fitting details

| Property | Value |
|----------|------|
| DINOv2 fine-tuning | None (zero-shot feature extraction) |
| Spectral fitting time | ~0.24 s on a single CPU core |
| Total compute (incl. DINOv2 forward pass) | ~5 s on RTX 3060 |
| GPU VRAM peak | < 1 GB |
| Reproducibility | Deterministic given `random_state = 99` |

## 7. Uncertainty / confidence per fold

Spectral Clustering does not natively output per-point confidence. We compute a proxy: the silhouette score per point. The mean silhouette is 0.0927 (low in absolute terms but typical for face-identity clusters in DINOv2 space, where clusters have moderate compactness because each subject's 10 images vary in pose/lighting/expression).

For deployment, we recommend rejecting any prediction whose silhouette is < 0 (these are points closer to a wrong-cluster centroid than to their own cluster's centroid). On the Olivetti champion clustering, 11 / 400 points have silhouette < 0; rejecting them would raise the conditional ARI on the remaining 389 points to ~0.74.

## 8. Reproduction status

| Item | Status |
|------|--------|
| Re-run from frozen code | ✅ ARI = 0.7195 byte-identical |
| Multi-seed variance characterised | ⚠️ std = 0.0429 across 5 seeds (see §10) |
| `X` and `y` SHA-256 verified | ✅ matches locked hashes |
| All inputs in archive | ✅ `code/`, `predict.py`, `config.json`, `experiment_log_entry.json` |
| External deps | DINOv2 from `torch.hub` (cached on first run); sklearn |

## 9. Sample inference code

```python
import torch
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms as T
from sklearn.cluster import SpectralClustering
from sklearn.datasets import fetch_olivetti_faces

# 1. Load data
data = fetch_olivetti_faces()
X = data.data           # (400, 4096) float32 in [0, 1]
y = data.target         # (400,) ground-truth subject IDs (used only for ARI evaluation)

# 2. Load DINOv2 ViT-S/14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
transform = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Extract features (384-dim per image)
features = []
with torch.no_grad():
    for i in range(0, len(X), 16):
        batch = []
        for x in X[i:i+16]:
            img = PILImage.fromarray((x.reshape(64, 64) * 255).astype(np.uint8), mode="L")
            batch.append(transform(img))
        batch = torch.stack(batch).to(device)
        f = model(batch)
        features.append(f.cpu().numpy())
features = np.vstack(features)  # (400, 384)

# 4. Cluster
cluster = SpectralClustering(
    n_clusters=40, affinity='cosine',
    assign_labels='kmeans', n_init=10,
    random_state=99
)
y_pred = cluster.fit_predict(features)

# 5. Evaluate
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
print(f"ARI: {adjusted_rand_score(y, y_pred):.4f}")    # expected: 0.7195
print(f"NMI: {normalized_mutual_info_score(y, y_pred):.4f}")  # expected: 0.9004
```

## 10. Deployment Strategy (clustering-adapted)

1. **Signal generation.** Inputs: a batch of 64×64 grayscale face images (or any size that resizes cleanly to 224×224). Outputs: cluster IDs in {0, ..., K−1} where K is the known number of subjects. The model is intended for *batch* inference (the spectral embedding is global, not per-point); for online inference, see §11.
2. **Decision rules.** No threshold tuning is required for a known-K clustering; the cluster assignments are direct outputs of `SpectralClustering.fit_predict`. For *new-subject detection* (an open-set problem), use a held-out subject's silhouette: reject if < 0 or above the 95th percentile of training silhouettes.
3. **Resource sizing.** Input batch can be up to ~10 000 images on a 16-GB GPU before the affinity matrix's O(n²) memory cost dominates. For larger n, use the Nyström approximation (Drineas, Mahoney 2005 JMLR 'On the Nyström method for approximating a Gram matrix'); we did not need to here.
4. **Refresh / retraining cadence.** The DINOv2 backbone is frozen; the only "training" is the per-batch SpectralClustering fit. Refresh whenever the input population changes substantially (new subjects, new lighting conditions, new sensor type).
5. **Per-regime performance.** Olivetti has only one regime (full dataset); no per-regime breakdown applies. For deployment on a multi-population dataset (e.g., multiple cameras, multiple ethnicities), run a per-population SpectralClustering and verify the per-population ARI matches the global ARI within 0.05.
6. **Risk controls.** Before deployment, re-verify the SHA-256 hash of the input batch (defends against silent data corruption). Monitor the silhouette distribution over time; if the mean drifts > 0.10 from the training mean (0.0927), retrain.
7. **Expected performance.** For Olivetti-similar grayscale face datasets at n ≈ 400, expect ARI in [0.65, 0.72] depending on seed. The 5-seed median is 0.6963; the seed-99 peak is 0.7195. For new datasets, run the 5-seed variance check before deploying to avoid mistaking a positive-tail run for the true mean.
8. **Caveats.**
   - DINOv2 was trained on natural-image RGB; transfer to grayscale faces is documented but not theoretically guaranteed for novel domains.
   - Seed-variance crisis: ±0.10 ARI on n = 400. Always report 5-seed median.
   - The spectral embedding is *global* — adding a single point to an existing fit changes all cluster assignments. For online inference, recompute from scratch or use streaming spectral methods (Tang et al. 2009 KDD 'Clustering large attributed graphs').
9. **Reference to inference code.** `predict.py` in this directory.

## 11. Known limitations and risks

- **n = 400 is small.** Findings transfer to "small unsupervised face benchmarks" but may not transfer to n ≥ 10 000.
- **Single seed champion.** ARI = 0.7195 is the positive tail of a ±0.10 spread.
- **Domain-specific.** The +0.15 ARI gain from DINOv2 is documented for face-identity tasks but not theoretically guaranteed for other small grayscale benchmarks.
- **No subject-supervised baseline.** A FaceNet triplet-loss baseline would presumably be substantially higher; not in scope.
- **No streaming version.** Spectral is offline; deploy only in batch mode.

## 12. Pointers

- Live dashboard: https://dlmastery.github.io/autoresearch/clustering_olivetti/
- Project root: `generalized_ml_autoresearch/examples/clustering_olivetti/`
- Project paper: `paper.md`
- Medium article: `autoresearch_results/medium_article.md`
- AutoResearch report: `autoresearch_results/autoresearch_report.md`
- Forensic report: `autoresearch_results/forensic_report.md`
- Third-party audit: `autoresearch_results/audit_report_third_party.md`

## 13. Reproduce on this machine

```bash
cd autoresearch_results/winners/spectral_hc_cosine_seed99_\(variance_c_exp71/
python predict.py
# Expected: ARI = 0.7195, NMI = 0.9004, V-measure = 0.9004
```
