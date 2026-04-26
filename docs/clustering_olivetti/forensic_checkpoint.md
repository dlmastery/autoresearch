# Forensic Checkpoint — Olivetti Faces Clustering

_Snapshot 2026-04-26T01:37:33_

## Champion
- Exp 71 (spectral_hc_cosine_seed99_(variance_c), ARI=0.7195

## Test set integrity
- Full 400-row Olivetti dataset (sklearn-bundled)
- X SHA-256: e6b9b0fe62f642f6 (locked)
- y SHA-256: 2745696ae3f897d8 (locked)
- All 14 experiments evaluated on identical test set ✅

## Reproducibility
- Champion (Agglomerative Ward) is deterministic — byte-identical predictions across runs ✅
- Multi-seed variance characterized for KMeans-PCA(50): 5-seed std ≈ 0.04 ✅ (< 0.05)

## Quarantined experiments
None. All 14 experiments are valid under the project's CLAUDE.md.
