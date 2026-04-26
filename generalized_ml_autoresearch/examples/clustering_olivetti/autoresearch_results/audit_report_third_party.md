# Third-Party Audit Report — Olivetti Faces Clustering Autoresearch

_Audit run: 2026-04-25T23:38:03_

This audit applies the standard data-science compliance checklist to the clustering autoresearch project. Each section returns 🟢 PASS / 🟡 WARN / 🔴 FAIL.

## 1. Data Integrity

- **n_samples:** 400 (🟢 PASS matches Olivetti documented 400)
- **n_features:** 4096 (🟢 PASS matches 64x64=4096)
- **n_classes:** 40 (🟢 PASS matches 40 subjects)
- **samples per class:** uniform 10 (🟢 PASS)
- **X SHA-256 (first 16):** `e6b9b0fe62f642f6` (locked)
- **y SHA-256 (first 16):** `2745696ae3f897d8` (locked)
- **X range:** [0.0000, 1.0000] (🟢 PASS normalized to [0,1])
- **NaN/Inf:** 0 / 0 (🟢 PASS)

## 2. Label Leakage Check

Verify that no clustering algorithm received `y` during fitting; `y` is only used at metric-evaluation time. Inspected by reading every `run_*.py` file:

- `run_exp01_kmeans_raw.py`: model receives X only, y consumed in `evaluate_clustering(y, y_pred, X)` ✅
- `run_full_pipeline.py`: same pattern for all 13 subsequent experiments ✅
- 🟢 PASS no label leakage.

## 3. Reproducibility — Re-run Champion (Exp 8 Agglomerative Ward)

- Two consecutive runs produce identical labels: True (🟢 PASS)
- ARI run 1: 0.515932
- ARI run 2: 0.515932
- Agglomerative Ward is deterministic (no random init), so byte-identical predictions are expected.

## 4. Multi-Seed Variance — KMeans on PCA(50) (5 seeds)

- 5-seed ARIs: [0.478, 0.4496, 0.4584, 0.4721, 0.4367]
- Mean: 0.4590, Std: 0.0149, Range: [0.4367, 0.4780]
- 🟢 PASS std=0.0149 (< 0.05 stability threshold)

## 5. Class Balance Per Cluster

- All 40 classes have exactly 10 samples — perfect uniform balance (🟢 PASS)

## 6. Intrinsic (Silhouette) vs Extrinsic (ARI) Correlation

- Pearson correlation across 15 experiments: 0.8351
- 🟢 PASS silhouette is a useful proxy for ARI on this dataset

## 7. Test Set Integrity (full-dataset evaluation)

- Clustering uses the full 400-row dataset for evaluation (no train/test split per standard clustering protocol). Dataset hash locked at `{X_hash}`.
- All 14 experiments evaluate on the SAME 400 rows (🟢 PASS).

## 8. Compliance Summary

| Check | Status |
|---|---|
| Data integrity | 🟢 PASS |
| No label leakage | 🟢 PASS |
| Reproducibility (champion deterministic) | 🟢 PASS |
| Multi-seed variance < 0.05 | 🟢 PASS |
| Class balance uniform | 🟢 PASS |
| Intrinsic-extrinsic correlation | 🟢 PASS |
| Test set hash locked | 🟢 PASS |
| All 14 experiments use identical evaluation dataset | 🟢 PASS |
