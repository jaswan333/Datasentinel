"""
DataSentinel - Embedding Anomaly Detector (Detector 1)
Evaluates local neighborhood distance in the deep latent representation space.
"""

from typing import Union
import numpy as np
from sklearn.neighbors import NearestNeighbors


def embedding_anomaly_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: Union[np.ndarray, list],
    k: int = 5
) -> np.ndarray:
    """
    Computes per-sample embedding anomaly percentiles within each class.
    Measures the mean Euclidean distance to k-nearest neighbors in latent space.

    Parameters:
        embeddings: Latent feature matrix (N, D).
        labels: Class labels (N,).
        train_indices: Reference indices used to calibrate the nearest-neighbor model.
        k: Number of nearest neighbors.

    Returns:
        Continuous anomaly risk scores in [0, 1].
    """
    scores = np.zeros(len(embeddings), dtype=float)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        class_all = np.where(labels == cls)[0]
        class_train = np.array([i for i in class_all if i in train_set])
        if len(class_train) <= k:
            continue

        X_train = embeddings[class_train]
        nn_model = NearestNeighbors(n_neighbors=k + 1)
        nn_model.fit(X_train)

        raw_train = nn_model.kneighbors(X_train, return_distance=True)[0][:, 1:].mean(axis=1)
        class_query = class_all
        X_query = embeddings[class_query]
        distances = nn_model.kneighbors(X_query, n_neighbors=k, return_distance=True)[0]
        raw_query = distances.mean(axis=1)

        sorted_scores = np.sort(raw_train)
        percentile = np.searchsorted(sorted_scores, raw_query, side="right") / len(sorted_scores)
        scores[class_query] = np.clip(percentile, 0.0, 1.0)

    return scores
