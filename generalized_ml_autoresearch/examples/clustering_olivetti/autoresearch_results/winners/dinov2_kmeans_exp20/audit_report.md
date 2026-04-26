# Champion Audit — Exp 20

## Per-section audit
1. **Algorithm:** Agglomerative Clustering, Ward linkage. Deterministic.
2. **Feature space:** PCA(50) on raw 4096-pixel features.
3. **Metrics:** ARI=0.5455, NMI=0.8201, silhouette=0.0710.
4. **Reproducibility:** byte-identical labels across runs (no random init).
5. **Test set:** full 400-row Olivetti, SHA-256 e6b9b0fe62f642f6.
6. **Limitations:** ARI 0.51 means ~half the cluster-pair decisions agree with ground truth — useful for exploratory analysis but not production face-recognition.
