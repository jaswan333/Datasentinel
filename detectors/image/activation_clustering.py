"""
DataSentinel - Activation Clustering Detector (Detector 2)
Isolates anomalous representations by clustering latent activations per class
using PCA dimensionality reduction and 2-cluster KMeans.
"""

from typing import Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def activation_clustering_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: Union[np.ndarray, list],
    n_components: int = 10,
    random_state: int = 42
) -> np.ndarray:
    """
    Computes activation clustering risk scores per sample.
    Backdoored samples typically form a compact sub-cluster divergent from the authentic distribution.

    Parameters:
        embeddings: Latent feature matrix (N, D).
        labels: Class labels (N,).
        train_indices: Reference indices for clustering calibration.
        n_components: Number of PCA components for representation compression.
        random_state: Random state for deterministic clustering.

    Returns:
        Continuous sigmoid risk scores in [0, 1].
    """
    scores = np.zeros(len(embeddings), dtype=float)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        class_all = np.where(labels == cls)[0]
        class_train = np.array([i for i in class_all if i in train_set])
        if len(class_train) < 20:
            continue

        X_train = embeddings[class_train]
        n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=random_state)
        Z_train = pca.fit_transform(X_train)

        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(Z_train)
        counts = np.bincount(clusters, minlength=2)
        suspicious_cluster = np.argmin(counts)

        Z_all = pca.transform(embeddings[class_all])
        cluster_distances = kmeans.transform(Z_all)
        suspicious_dist = cluster_distances[:, suspicious_cluster]
        other_dist = cluster_distances[:, 1 - suspicious_cluster]

        margin = other_dist - suspicious_dist
        scale = np.std(margin) + 1e-6
        risk = 1.0 / (1.0 + np.exp(-margin / scale))
        scores[class_all] = np.clip(risk, 0.0, 1.0)

    return scores
