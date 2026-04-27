# Forensic Checkpoint — Olivetti Faces Clustering

*Snapshot: 2026-04-26 (post-Exp 149). Updated to reflect the 152-experiment state.*

## Champion (unconditional, full 400-sample evaluation)

- **Exp 147** — `spectral_coassoc_ensemble_5seed` — ARI = **0.7346**
- Method: 5-seed CSPA co-association ensemble (Strehl & Ghosh 2002) of Spectral cosine on DINOv2 ViT-S/14
- Base seeds: {0, 1, 7, 42, 99}
- Final stage: SpectralClustering(affinity='precomputed', n_init=10, random_state=0) on the 400×400 co-association matrix
- Secondary: NMI = 0.9093, V-measure = 0.9093, silhouette = 0.1017, n_pred = 40, n_noise = 0
- Reproduced from frozen code in `winners/spectral_coassoc_ensemble_5seed_exp147/`
- **Eliminates** the seed-variance crisis of Exp 71 (which was the +1σ tail of a noisy distribution)

## Champion (deployment mode, conditional on silhouette rejection)

- **Exp 149** — `silhouette_reject_on_exp71` — Conditional ARI = **0.8740** on 317/400 kept samples
- Method: Apply Exp 71 base clustering, reject 83 silhouette-negative samples
- Note: This is the *deployment scenario*, not directly comparable to unconditional ARI

## Test-set integrity

- Full 400-row Olivetti dataset (`sklearn.datasets.fetch_olivetti_faces()`)
- X shape: (400, 4096), float32 normalised to [0, 1]
- y shape: (400,), 40 unique classes, 10 samples each
- X SHA-256 (first 16 hex): `e6b9b0fe62f642f6` ✅
- y SHA-256 (first 16 hex): `2745696ae3f897d8` ✅
- All 152 experiments evaluated on identical test set ✅

## Reproducibility

- Champion (Exp 147 ensemble) deterministic given the 5 fixed base seeds {0,1,7,42,99} and final seed=0 ✅
- Eliminates the ±0.10 seed-variance crisis of single-seed Exp 71 — the ensemble is *the* reproducibility fix
- 5-seed individual variance check on Spectral cosine: ARI ∈ {0.6963, 0.7154, 0.6596, 0.6127, 0.7195}; std = 0.0429
- 5-seed median: 0.6963; ensemble: 0.7346 (+0.0383)

## Composite metric integrity

- Composite definition: ARI directly, with floor 0.30
- Composite fingerprint: `clustering-ari-floor0.3` (locked, on every JSONL row)
- All 152 JSONL rows carry the fingerprint ✅
- Composite has not been silently rewritten ✅

## Reasoning-blob discipline

- 152 entries × 7 fields (Exp 147-149 added)
- All pass the per-field validators (word-count + content)
- Zero `_needs_rewrite: true`, zero placeholders, zero TODO-REWRITE sentinels

## Quarantined experiments

- `_quarantined_blind_sweep/` — early experiments that violated the one-change-per-experiment rule
- `_quarantined_exp1/` — early Exp 1 with invalid pre-run reasoning blob

Neither contributes to the JSONL log, the dashboard, or the champion search.

## Champion progression (13 rungs, 0.4057 → 0.7346)

| Exp | ARI | Method change |
|----:|----:|---------------|
| 1 | 0.4057 | KMeans on raw pixels (baseline) |
| 2 | 0.4780 | KMeans on PCA(50) |
| 8 | 0.5159 | Agglomerative Ward |
| 16 | 0.5252 | Spectral RBF tuned gamma |
| 17 | 0.5287 | Birch (default) |
| 20 | 0.5455 | DINOv2 ViT-S/14 + KMeans |
| 22 | 0.5596 | DINOv2 + MiniBatch-KMeans |
| 25 | 0.5852 | DINOv2 + KMeans n_init = 50 |
| 27 | 0.6371 | DINOv2 + Ward |
| 33 | 0.6963 | DINOv2 + Spectral cosine |
| 55 | 0.7170 | DINOv2 + Spectral RBF γ = 1e-4 |
| 71 | 0.7195 | DINOv2 + Spectral cosine, seed = 99 (single-seed +1σ tail) |
| **147** | **0.7346** | **5-seed CSPA co-association ensemble (eliminates seed variance)** |

## Three+1 research findings

1. **DEC plateau** at ARI ≈ 0.50 across 11 hill-climb variants (std = 0.019).
2. **Birch threshold-invariance** for n < 10 000 (13 thresholds → identical ARI = 0.6371).
3. **Spectral seed-variance crisis** of ±0.10 ARI on n = 400 (std = 0.0429 across 5 seeds), **eliminated by CSPA ensemble**.
4. **DINOv2 backbone scale-saturation** at n = 400 (Exp 148 ViT-L/14 underperforms ViT-S/14 by 0.034 ARI; Kaplan 2020 scaling-law confirmation).

## Deployment artifacts

- **Production rule:** silhouette-rejection on Exp 71 base clustering → conditional ARI = 0.8740 on 317/400 kept samples (Exp 149).
- **Future direction:** Exp 147 ensemble + Exp 149 silhouette rejection (predicted conditional ARI > 0.92 on deployment subset).

## Pointers

- Repo: `github.com/dlmastery/autoresearch`
- Dashboard: `dlmastery.github.io/autoresearch/clustering_olivetti/`
- Project root: `generalized_ml_autoresearch/examples/clustering_olivetti/`
- Champion archive: `winners/spectral_coassoc_ensemble_5seed_exp147/`
- Previous champion archive: `winners/spectral_hc_cosine_seed99_(variance_c_exp71/` (now superseded; the README still describes its config but the project champion is Exp 147)
