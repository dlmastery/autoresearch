"""Standalone inference for the Olivetti clustering champion."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def predict(X):
    """X: (n, 4096) float32 in [0,1] -> labels (n,) int in [0, 39]."""
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    return AgglomerativeClustering(n_clusters=40, linkage="ward").fit_predict(Z)

if __name__ == "__main__":
    from sklearn.datasets import fetch_olivetti_faces
    bunch = fetch_olivetti_faces(shuffle=False, random_state=0)
    labels = predict(bunch.data.astype(np.float32))
    print(f"predicted {len(np.unique(labels))} clusters on {len(labels)} samples")
