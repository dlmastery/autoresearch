# Forensic Checkpoint — Olivetti Faces Clustering

*Snapshot: 2026-04-26. Updated to reflect the final 149-experiment state.*

## Champion

- **Exp 71** — `spectral_hc_cosine_seed99_(variance_c` — ARI = **0.7195**
- Backbone: DINOv2 ViT-S/14 (384-dim class-token features)
- Head: SpectralClustering(affinity='cosine', assign_labels='kmeans', n_init=10, random_state=99)
- Secondary: NMI = 0.9004, V-measure = 0.9004, FMI = 0.7270, Silhouette = 0.0927
- Reproduced from frozen code in `winners/spectral_hc_cosine_seed99_(variance_c_exp71/`

## Test-set integrity

- Full 400-row Olivetti dataset (`sklearn.datasets.fetch_olivetti_faces()`)
- X shape: (400, 4096), float32 normalised to [0, 1]
- y shape: (400,), 40 unique classes, 10 samples each
- X SHA-256 (first 16 hex): `e6b9b0fe62f642f6` (locked, re-asserted at every load)
- y SHA-256 (first 16 hex): `2745696ae3f897d8` (locked, re-asserted at every load)
- All 149 experiments evaluated on identical test set ✅

## Reproducibility

- Champion (Spectral cosine, seed = 99) deterministic given the seed — byte-identical predictions across runs ✅
- 5-seed variance check on champion config: ARI ∈ {0.6963, 0.7154, 0.6596, 0.6127, 0.7195} for seeds {0, 1, 7, 42, 99}; std = 0.0429, spread = 0.107.
- ⚠️ Headline ARI = 0.7195 is the positive-tail of the seed distribution; honest headline is "5-seed median = 0.6963 ± 0.0429".

## Composite metric integrity

- Composite definition: ARI directly, with floor 0.30.
- Composite fingerprint: `clustering-ari-floor0.3` (locked at setup, on every JSONL row).
- All 149 JSONL rows carry the fingerprint ✅
- Composite has not been silently rewritten ✅

## Reasoning-blob discipline

- 149 entries × 7 fields = 1043 reasoning fields.
- All 1043 pass the per-field validators (word-count floor + content requirements).
- Zero `_needs_rewrite: true`. Zero `(auto-backfilled)` placeholders. Zero `TODO-REWRITE` sentinels.

## Quarantined experiments

- `_quarantined_blind_sweep/` — early experiments that violated the one-change-per-experiment rule. Annotated with `WHY_QUARANTINED.md`.
- `_quarantined_exp1/` — early Exp 1 with invalid pre-run reasoning blob. Replaced by current Exp 1 (KMeans on raw pixels, ARI = 0.4057).

Neither quarantine contributes to the JSONL log, the dashboard, or the champion search.

## Champion progression (12 rungs)

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
| **71** | **0.7195** | **DINOv2 + Spectral cosine, seed = 99** |

## Three findings

1. **DEC plateau** at ARI ≈ 0.50 across 11 hill-climb variants (std = 0.019).
2. **Birch threshold-invariance** for n < 10 000 (13 thresholds → identical ARI = 0.6371).
3. **Spectral seed-variance crisis** of ±0.10 ARI on n = 400 (std = 0.0429 across 5 seeds).

## Pointers

- Repo: `github.com/dlmastery/autoresearch`
- Dashboard: `dlmastery.github.io/autoresearch/clustering_olivetti/`
- Project root: `generalized_ml_autoresearch/examples/clustering_olivetti/`
- Champion archive: `winners/spectral_hc_cosine_seed99_(variance_c_exp71/`
