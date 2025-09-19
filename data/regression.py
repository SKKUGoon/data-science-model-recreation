from typing import Tuple

import numpy as np
import pandas as pd


def generate_regression_data(
    n_samples: int = 150_000,
    n_features: int = 25,
    noise_std: float = 0.75,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    # Structured coefficients: decaying weights + sparse spikes for variety
    base_coefs = np.linspace(2.0, 0.1, n_features)
    sparse_mask = rng.random(n_features) < 0.2
    spike_coefs = rng.normal(0, 3.0, size=n_features) * sparse_mask
    coefs = base_coefs + spike_coefs

    nonlinear_term = 0.75 * np.sin(X[:, :3].sum(axis=1))
    interaction_term = 0.5 * (X[:, 3] * X[:, 4])

    y = X @ coefs + nonlinear_term + interaction_term
    y += rng.normal(scale=noise_std, size=n_samples)

    feature_names = [f"reg_feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target_regression")
    return X_df, y_series


