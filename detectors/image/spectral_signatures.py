"""
DataSentinel - Spectral Signatures Detector (Detector 3)
Detects Trojan backdoor representations using Singular Value Decomposition (SVD)
to capture dominant directional projections within each class representation.
"""

from typing import Union
import numpy as np


def spectral_signature_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: Union[np.ndarray, list]
) -> np.ndarray:
    """
    Computes spectral signature scores via top-singular vector projection.
    Poisoned samples that force specific backdoor features create distinct spectral outliers.

    Parameters:
        embeddings: Latent feature matrix (N, D).
        labels: Class labels (N,).
        train_indices: Reference training indices for SVD centering and baseline calibration.

    Returns:
        Continuous projection anomaly percentiles in [0, 1].
    """
    scores = np.zeros(len(embeddings), dtype=float)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        class_all = np.where(labels == cls)[0]
        class_train = np.array([i for i in class_all if i in train_set])
        if len(class_train) < 10:
            continue

        X_train = embeddings[class_train]
        mean_vector = X_train.mean(axis=0)
        X_centered = X_train - mean_vector

        # SVD on centered representations
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        top_direction = Vt[0]

        X_all = embeddings[class_all]
        projections = (X_all - mean_vector) @ top_direction
        raw_scores = projections ** 2

        train_projections = (X_train - mean_vector) @ top_direction
        train_raw = train_projections ** 2
        sorted_scores = np.sort(train_raw)

        percentile = np.searchsorted(sorted_scores, raw_scores, side="right") / len(sorted_scores)
        scores[class_all] = np.clip(percentile, 0.0, 1.0)

    return scores
