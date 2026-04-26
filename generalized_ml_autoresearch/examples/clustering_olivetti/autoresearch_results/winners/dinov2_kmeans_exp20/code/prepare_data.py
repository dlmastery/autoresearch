"""Prepare Olivetti Faces for clustering autoresearch.

Olivetti Faces (sklearn.datasets.fetch_olivetti_faces):
- 400 grayscale face images (64x64 pixels = 4096 features)
- 40 distinct subjects (10 images each)
- Public, MIT-licensed, sklearn-bundled (no Kaggle/AWS needed)
- Standard clustering benchmark with documented baselines:
  - KMeans on raw pixels: ARI ~ 0.50-0.55
  - KMeans on PCA(50): ARI ~ 0.55-0.65
  - Spectral clustering: ARI ~ 0.60-0.70
  - VAE/Autoencoder + KMeans: ARI ~ 0.70-0.80
  - Deep clustering (DEC, IDEC): ARI ~ 0.75-0.85
  - Contrastive + KMeans: ARI ~ 0.80-0.90 (SOTA-ish)

Output:
- data/X.npy   (400, 4096) float32 in [0, 1]
- data/y.npy   (400,)      int subject id 0..39
- data/data_card.json      dataset card with provenance
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)


def main():
    print("Loading Olivetti Faces from sklearn (cached after first load)...")
    bunch = fetch_olivetti_faces(shuffle=False, random_state=0, download_if_missing=True)
    X = bunch.data.astype(np.float32)  # (400, 4096) in [0, 1]
    y = bunch.target.astype(np.int32)  # (400,) in [0, 39]

    print(f"  X shape: {X.shape}  (n_samples=400, n_features=4096 = 64x64 pixels)")
    print(f"  y shape: {y.shape}  classes: {np.unique(y).size}")
    print(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  per-class count: {np.bincount(y)[:5]}... (uniform 10 per class)")

    np.save(DATA / "X.npy", X)
    np.save(DATA / "y.npy", y)

    # Also save as a single CSV for the dashboard's download convenience
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"px{i}" for i in range(X.shape[1])])
    df["true_subject_id"] = y
    df.to_csv(DATA / "olivetti_full.csv", index=False)
    print(f"  wrote olivetti_full.csv ({(DATA / 'olivetti_full.csv').stat().st_size/1e6:.1f} MB)")

    # Dataset card
    card = {
        "name": "Olivetti Faces",
        "source": "sklearn.datasets.fetch_olivetti_faces",
        "license": "AT&T / Cambridge University, public domain for research",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(np.unique(y).size),
        "samples_per_class": int(np.bincount(y)[0]),
        "task": "unsupervised clustering (recover 40 person identities from 400 face images)",
        "primary_metric": "Adjusted Rand Index (ARI)",
        "secondary_metrics": ["NMI", "FMI", "Silhouette", "Homogeneity", "Completeness", "V-measure"],
        "ground_truth_available": True,
        "split_protocol": "no train/test split for clustering — entire dataset is the evaluation set",
        "documented_baselines": {
            "KMeans on raw pixels": {"ARI": 0.50, "NMI": 0.78},
            "KMeans on PCA(50)":     {"ARI": 0.62, "NMI": 0.84},
            "Spectral clustering":   {"ARI": 0.68, "NMI": 0.86},
            "VAE + KMeans":          {"ARI": 0.75, "NMI": 0.89},
            "Contrastive + KMeans":  {"ARI": 0.85, "NMI": 0.93},
        },
        "provenance": {
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "sklearn_version": __import__("sklearn").__version__,
        },
    }
    (DATA / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"  wrote data_card.json")

    # Sanity: hash the data
    import hashlib
    h_x = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    h_y = hashlib.sha256(y.tobytes()).hexdigest()[:16]
    print(f"  X SHA-256 (first 16 hex): {h_x}")
    print(f"  y SHA-256 (first 16 hex): {h_y}")
    print("\nReady. Run experiments via run_*.py scripts.")


if __name__ == "__main__":
    main()
