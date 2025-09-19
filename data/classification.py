from typing import Tuple

import numpy as np
import pandas as pd


def generate_classification_data(
    n_samples: int = 180_000,
    n_features: int = 18,
    n_classes: int = 4,
    seed: int = 314,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    # Class-specific linear weights + offsets
    weights = rng.normal(size=(n_classes, n_features))
    offsets = rng.uniform(-2.0, 2.0, size=n_classes)

    logits = X @ weights.T + offsets
    logits += 0.5 * np.tanh(X[:, :4] @ rng.normal(size=(4, n_classes)))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    y = np.array([rng.choice(n_classes, p=p_row) for p_row in probs])

    feature_names = [f"clf_feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target_classification")
    return X_df, y_series