"""
DataSentinel - 14-Signal Image Evidence Suite
Implements 14 complementary evidence signals across 4 distinct groups:
Group A — Label / Model Consistency (1-5)
Group B — Deep Representation Signals (6-10)
Group C — Image Integrity Signals (11-13)
Group D — Model-Behaviour Signal (14)
"""

from typing import Dict, List, Optional, Tuple, Union
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


SIGNAL_COLUMNS = [
    "training_loss_risk",
    "prediction_entropy_risk",
    "prediction_margin_risk",
    "assigned_confidence_risk",
    "oof_label_disagreement",
    "knn_embedding_risk",
    "lof_embedding_risk",
    "centroid_distance_risk",
    "activation_cluster_risk",
    "spectral_signature_risk",
    "exact_duplicate_risk",
    "perceptual_duplicate_risk",
    "image_statistics_risk",
    "local_patch_sensitivity_risk"
]


# =====================================================================
# GROUP A — LABEL / MODEL CONSISTENCY SIGNALS
# =====================================================================

def compute_loss_and_probabilities(
    logits: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-sample cross entropy loss and softmax probabilities.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits_t, dim=1).numpy()

    # Numerical stability
    eps = 1e-12
    n = len(labels)
    losses = np.zeros(n, dtype=np.float32)
    for i in range(n):
        y = int(labels[i])
        losses[i] = -np.log(np.clip(probs[i, y], eps, 1.0))

    return losses, probs


def signal_training_loss_risk(losses: np.ndarray) -> np.ndarray:
    """Signal 1: Training loss risk mapped to [0, 1] via empirical percentiles."""
    sorted_losses = np.sort(losses)
    ranks = np.searchsorted(sorted_losses, losses, side="right") / len(sorted_losses)
    return np.clip(ranks, 0.0, 1.0)


def signal_prediction_entropy_risk(probs: np.ndarray) -> np.ndarray:
    """Signal 2: Normalized prediction entropy in [0, 1]. High entropy = ambiguity."""
    eps = 1e-12
    probs_clamped = np.clip(probs, eps, 1.0)
    k = probs.shape[1]
    entropy = -np.sum(probs_clamped * np.log2(probs_clamped), axis=1)
    max_entropy = np.log2(k)
    return np.clip(entropy / max_entropy, 0.0, 1.0)


def signal_prediction_margin_risk(probs: np.ndarray) -> np.ndarray:
    """Signal 3: Prediction margin risk. 1 - (top1 - top2). Low margin = uncertainty."""
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return np.clip(1.0 - margin, 0.0, 1.0)


def signal_assigned_confidence_risk(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Signal 4: Assigned-label confidence risk. 1 - P(assigned_label)."""
    n = len(labels)
    conf_risk = np.zeros(n, dtype=np.float32)
    for i in range(n):
        conf_risk[i] = 1.0 - probs[i, int(labels[i])]
    return np.clip(conf_risk, 0.0, 1.0)


def signal_oof_label_disagreement(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Signal 5: Split-aware latent-space linear probe disagreement: 1 - P_probe(assigned).
    Fitted strictly on train_indices to prevent data leakage onto val/test partitions.
    """
    n = len(labels)
    scores = np.zeros(n, dtype=np.float32)

    train_X = embeddings[train_indices]
    train_y = labels[train_indices]

    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X)
    all_X_s = scaler.transform(embeddings)

    probe = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
        C=1.0
    )
    probe.fit(train_X_s, train_y)
    probe_probs = probe.predict_proba(all_X_s)

    classes = list(probe.classes_)
    for i in range(n):
        y = int(labels[i])
        if y in classes:
            idx = classes.index(y)
            scores[i] = 1.0 - probe_probs[i, idx]
        else:
            scores[i] = 1.0

    return np.clip(scores, 0.0, 1.0)


# =====================================================================
# GROUP B — DEEP REPRESENTATION SIGNALS
# =====================================================================

def signal_knn_embedding_risk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """Signal 6: KNN embedding anomaly percentile in 512-D space within class."""
    n = len(embeddings)
    scores = np.zeros(n, dtype=np.float32)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        cls_all = np.where(labels == cls)[0]
        cls_train = np.array([idx for idx in cls_all if idx in train_set])
        if len(cls_train) <= k:
            continue

        nn_model = NearestNeighbors(n_neighbors=k + 1)
        nn_model.fit(embeddings[cls_train])

        train_dist = nn_model.kneighbors(embeddings[cls_train], return_distance=True)[0][:, 1:].mean(axis=1)
        query_dist = nn_model.kneighbors(embeddings[cls_all], n_neighbors=k, return_distance=True)[0].mean(axis=1)

        sorted_d = np.sort(train_dist)
        percentile = np.searchsorted(sorted_d, query_dist, side="right") / len(sorted_d)
        scores[cls_all] = np.clip(percentile, 0.0, 1.0)

    return scores


def signal_lof_embedding_risk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """Signal 7: Local Outlier Factor in 512-D space per class."""
    n = len(embeddings)
    scores = np.zeros(n, dtype=np.float32)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        cls_all = np.where(labels == cls)[0]
        cls_train = np.array([idx for idx in cls_all if idx in train_set])
        if len(cls_train) <= k + 2:
            continue

        lof = LocalOutlierFactor(n_neighbors=min(k, len(cls_train) - 1), novelty=True)
        lof.fit(embeddings[cls_train])

        # decision_function: large negative means outlier
        dec = lof.decision_function(embeddings[cls_all])
        # scale to [0, 1] via sigmoid on inverted score
        margin = -dec
        scale = np.std(margin) + 1e-6
        risk = 1.0 / (1.0 + np.exp(-margin / scale))
        scores[cls_all] = np.clip(risk, 0.0, 1.0)

    return scores


def signal_centroid_distance_risk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray
) -> np.ndarray:
    """
    Signal 8: Ratio of distance to assigned class centroid vs opposite classes:
    d(z, centroid_assigned) / (d(z, centroid_assigned) + min_{c != y} d(z, centroid_c))
    """
    n = len(embeddings)
    scores = np.zeros(n, dtype=np.float32)
    classes = np.unique(labels)
    train_set = set(train_indices)

    centroids: Dict[int, np.ndarray] = {}
    for cls in classes:
        cls_train = [idx for idx in np.where(labels == cls)[0] if idx in train_set]
        if cls_train:
            centroids[cls] = embeddings[cls_train].mean(axis=0)
        else:
            centroids[cls] = embeddings[labels == cls].mean(axis=0)

    for i in range(n):
        y = int(labels[i])
        z = embeddings[i]
        d_assigned = np.linalg.norm(z - centroids[y])

        d_others = [
            np.linalg.norm(z - centroids[c]) for c in classes if c != y and c in centroids
        ]
        d_min_other = min(d_others) if d_others else (d_assigned + 1e-6)

        denom = d_assigned + d_min_other + 1e-12
        scores[i] = np.clip(d_assigned / denom, 0.0, 1.0)

    return scores


def signal_activation_cluster_risk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """Signal 9: Activation Clustering via PCA (10-D) + 2-cluster KMeans."""
    n = len(embeddings)
    scores = np.zeros(n, dtype=np.float32)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        cls_all = np.where(labels == cls)[0]
        cls_train = np.array([idx for idx in cls_all if idx in train_set])
        if len(cls_train) < 20:
            continue

        n_comp = min(10, embeddings.shape[1], len(cls_train) - 1)
        pca = PCA(n_components=n_comp, random_state=seed)
        Z_train = pca.fit_transform(embeddings[cls_train])

        kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
        clusters = kmeans.fit_predict(Z_train)
        counts = np.bincount(clusters, minlength=2)
        suspicious_cluster = np.argmin(counts)

        Z_all = pca.transform(embeddings[cls_all])
        dists = kmeans.transform(Z_all)
        susp_dist = dists[:, suspicious_cluster]
        auth_dist = dists[:, 1 - suspicious_cluster]

        margin = auth_dist - susp_dist
        scale = np.std(margin) + 1e-6
        risk = 1.0 / (1.0 + np.exp(-margin / scale))
        scores[cls_all] = np.clip(risk, 0.0, 1.0)

    return scores


def signal_spectral_signature_risk(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray
) -> np.ndarray:
    """Signal 10: Spectral Signatures via top-singular vector SVD projection anomaly."""
    n = len(embeddings)
    scores = np.zeros(n, dtype=np.float32)
    train_set = set(train_indices)

    for cls in np.unique(labels):
        cls_all = np.where(labels == cls)[0]
        cls_train = np.array([idx for idx in cls_all if idx in train_set])
        if len(cls_train) < 10:
            continue

        X_train = embeddings[cls_train]
        mean_vec = X_train.mean(axis=0)
        X_centered = X_train - mean_vec

        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        top_vec = Vt[0]

        train_proj = ((X_train - mean_vec) @ top_vec) ** 2
        all_proj = ((embeddings[cls_all] - mean_vec) @ top_vec) ** 2

        sorted_proj = np.sort(train_proj)
        percentile = np.searchsorted(sorted_proj, all_proj, side="right") / len(sorted_proj)
        scores[cls_all] = np.clip(percentile, 0.0, 1.0)

    return scores


# =====================================================================
# GROUP C — IMAGE INTEGRITY SIGNALS
# =====================================================================

def signal_exact_duplicate_risk(images: np.ndarray) -> np.ndarray:
    """Signal 11: Exact byte hash collision detection. Duplicates flagged with 1.0."""
    hashes: Dict[str, List[int]] = {}
    for i, img in enumerate(images):
        h = hashlib.md5(img.tobytes()).hexdigest()
        hashes.setdefault(h, []).append(i)

    n = len(images)
    scores = np.zeros(n, dtype=np.float32)
    for idx_list in hashes.values():
        if len(idx_list) > 1:
            for idx in idx_list:
                scores[idx] = 1.0

    return scores


def compute_dhash(img: np.ndarray) -> int:
    """Computes a 64-bit difference hash (dHash) for an image array."""
    pil_img = Image.fromarray(img).convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.asarray(pil_img, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def signal_perceptual_duplicate_risk(images: np.ndarray) -> np.ndarray:
    """
    Signal 12: Perceptual near-duplicate detection via 64-bit dHash Hamming distance.
    Hamming distance <= 4 bits flagged as high near-duplicate risk.
    """
    n = len(images)
    dhashes = np.array([compute_dhash(img) for img in images], dtype=np.uint64)
    scores = np.zeros(n, dtype=np.float32)

    # For fast comparison on moderate datasets (e.g. 5,000 samples)
    for i in range(n):
        h_i = dhashes[i]
        # XOR to get differing bits
        xor_diffs = np.bitwise_xor(dhashes, h_i)
        # Exclude self by setting index i to 64
        xor_diffs[i] = 0xFFFFFFFFFFFFFFFF
        # Compute popcounts
        # Python bin() is vectorized over small arrays or loop
        dists = np.array([bin(int(x)).count("1") for x in xor_diffs[:min(n, 1000)]])
        min_dist = dists.min() if len(dists) > 0 else 64

        if min_dist <= 2:
            scores[i] = 1.0
        elif min_dist <= 6:
            scores[i] = float(np.clip(1.0 - (min_dist - 2) / 5.0, 0.0, 1.0))

    return scores


def signal_image_statistics_risk(
    images: np.ndarray,
    train_indices: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Signal 13: Image-level statistical anomalies (RGB channel means, stds, high-freq energy)
    evaluated via Isolation Forest calibrated on train_indices.
    """
    n = len(images)
    feats = np.zeros((n, 7), dtype=np.float32)

    for i, img in enumerate(images):
        img_f = img.astype(np.float32)
        means = img_f.mean(axis=(0, 1))
        stds = img_f.std(axis=(0, 1))
        # High frequency gradient energy
        diff_y = np.diff(img_f, axis=0)
        diff_x = np.diff(img_f, axis=1)
        hf_energy = float(np.mean(diff_y ** 2) + np.mean(diff_x ** 2))
        feats[i] = [*means, *stds, hf_energy]

    train_feats = feats[train_indices]
    iso = IsolationForest(n_estimators=100, random_state=seed, contamination="auto")
    iso.fit(train_feats)

    dec_func = iso.decision_function(feats)
    margin = -dec_func
    scale = np.std(margin) + 1e-6
    scores = 1.0 / (1.0 + np.exp(-margin / scale))

    return np.clip(scores, 0.0, 1.0)


# =====================================================================
# GROUP D — MODEL-BEHAVIOUR SIGNAL
# =====================================================================

def signal_local_patch_sensitivity_risk(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    patch_size: int = 8,
    batch_size: int = 128
) -> np.ndarray:
    """
    Signal 14: Model behavioral sensitivity to candidate localized masks.
    Tests 5 regions: top-left, top-right, bottom-left, bottom-right, center.
    Evaluates maximum prediction divergence |P(y) - P_masked(y)|.
    A high shift indicates local patch reliance (Trojan backdoor behavior).
    """
    model.eval()
    n = len(images)
    h, w, c = images.shape[1:]

    regions = [
        (0, patch_size, 0, patch_size),                          # top-left
        (0, patch_size, w - patch_size, w),                      # top-right
        (h - patch_size, h, 0, patch_size),                      # bottom-left
        (h - patch_size, h, w - patch_size, w),                  # bottom-right
        (h // 2 - patch_size // 2, h // 2 + patch_size // 2,     # center
         w // 2 - patch_size // 2, w // 2 + patch_size // 2)
    ]

    mean_color = np.array([128, 128, 128], dtype=np.uint8)

    # Base predictions
    base_probs_list = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_t = torch.tensor(batch_imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            batch_t = batch_t.to(device)
            out = model(batch_t)
            base_probs_list.append(F.softmax(out, dim=1).cpu().numpy())

    base_probs = np.vstack(base_probs_list)

    max_divergences = np.zeros(n, dtype=np.float32)

    for y1, y2, x1, x2 in regions:
        masked_imgs = images.copy()
        masked_imgs[:, y1:y2, x1:x2, :] = mean_color

        masked_probs_list = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_imgs = masked_imgs[i:i + batch_size]
                batch_t = torch.tensor(batch_imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                batch_t = batch_t.to(device)
                out = model(batch_t)
                masked_probs_list.append(F.softmax(out, dim=1).cpu().numpy())

        masked_probs = np.vstack(masked_probs_list)

        # Measure drop in assigned label probability or TV distance
        for i in range(n):
            y = int(labels[i])
            shift = abs(base_probs[i, y] - masked_probs[i, y])
            if shift > max_divergences[i]:
                max_divergences[i] = shift

    return np.clip(max_divergences, 0.0, 1.0)


# =====================================================================
# MASTER SIGNAL EXTRACTION PIPELINE
# =====================================================================

def extract_all_14_signals(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    logits: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    compute_behavior: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Extracts all 14 evidence signals in a strict split-aware, leakage-free pipeline.
    Returns a structured DataFrame ready for multi-signal evidence fusion.
    """
    print("    [Signals] Computing Group A (Label Consistency: Loss, Entropy, Margin, Confidence, OOF)...")
    losses, probs = compute_loss_and_probabilities(logits, labels)
    sig1_loss = signal_training_loss_risk(losses)
    sig2_entropy = signal_prediction_entropy_risk(probs)
    sig3_margin = signal_prediction_margin_risk(probs)
    sig4_conf = signal_assigned_confidence_risk(probs, labels)
    sig5_oof = signal_oof_label_disagreement(embeddings, labels, train_indices, seed=seed)

    print("    [Signals] Computing Group B (Deep Representations: KNN, LOF, Centroid, Cluster, Spectral)...")
    sig6_knn = signal_knn_embedding_risk(embeddings, labels, train_indices)
    sig7_lof = signal_lof_embedding_risk(embeddings, labels, train_indices)
    sig8_centroid = signal_centroid_distance_risk(embeddings, labels, train_indices)
    sig9_cluster = signal_activation_cluster_risk(embeddings, labels, train_indices, seed=seed)
    sig10_spectral = signal_spectral_signature_risk(embeddings, labels, train_indices)

    print("    [Signals] Computing Group C (Image Integrity: Exact Dups, Perceptual Dups, Image Stats)...")
    sig11_exact = signal_exact_duplicate_risk(images)
    sig12_perceptual = signal_perceptual_duplicate_risk(images)
    sig13_stats = signal_image_statistics_risk(images, train_indices, seed=seed)

    if compute_behavior:
        print("    [Signals] Computing Group D (Model Behaviour: Local Patch Sensitivity)...")
        sig14_behavior = signal_local_patch_sensitivity_risk(model, images, labels, device)
    else:
        print("    [Signals] Skipping Group D (Model Behaviour flag disabled)...")
        sig14_behavior = np.zeros(len(images), dtype=np.float32)

    df_signals = pd.DataFrame({
        "training_loss_risk": np.round(sig1_loss, 4),
        "prediction_entropy_risk": np.round(sig2_entropy, 4),
        "prediction_margin_risk": np.round(sig3_margin, 4),
        "assigned_confidence_risk": np.round(sig4_conf, 4),
        "oof_label_disagreement": np.round(sig5_oof, 4),
        "knn_embedding_risk": np.round(sig6_knn, 4),
        "lof_embedding_risk": np.round(sig7_lof, 4),
        "centroid_distance_risk": np.round(sig8_centroid, 4),
        "activation_cluster_risk": np.round(sig9_cluster, 4),
        "spectral_signature_risk": np.round(sig10_spectral, 4),
        "exact_duplicate_risk": np.round(sig11_exact, 4),
        "perceptual_duplicate_risk": np.round(sig12_perceptual, 4),
        "image_statistics_risk": np.round(sig13_stats, 4),
        "local_patch_sensitivity_risk": np.round(sig14_behavior, 4)
    })

    return df_signals
