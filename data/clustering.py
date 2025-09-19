from typing import Tuple, List

import numpy as np
import pandas as pd


def generate_clustering_data(
    n_samples: int = 200_000,
    seed: int = 2024,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Dense Gaussian blobs
    blob_centers = np.array([
        [5.0, 5.0, 5.0],
        [-5.0, 0.0, 3.0],
        [0.0, -6.0, -4.0],
    ])
    blob_spreads = np.array([0.8, 1.4, 0.6])
    samples_per_blob = n_samples // 4

    blobs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for idx, (center, spread) in enumerate(zip(blob_centers, blob_spreads)):
        blob = rng.normal(loc=center, scale=spread, size=(samples_per_blob, 3))
        blobs.append(blob)
        labels.append(np.full(samples_per_blob, idx))

    # Anisotropic elongated cluster (good for HDBSCAN)
    elongation = np.array([[3.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.2]])
    base_points = rng.normal(size=(samples_per_blob, 3))
    elongated = base_points @ elongation + np.array([0.0, 6.0, -2.0])
    blobs.append(elongated)
    labels.append(np.full(samples_per_blob, len(blob_centers)))

    # Noise/outliers to make clustering harder
    noise_count = n_samples - samples_per_blob * 4
    noise = rng.uniform(-12, 12, size=(noise_count, 3))
    blobs.append(noise)
    labels.append(np.full(noise_count, -1))  # mark noise separately

    X = np.vstack(blobs)
    y = np.concatenate(labels)

    feature_names = ["cluster_x", "cluster_y", "cluster_z"]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y