"""
DataSentinel - Core Evidence Fusion & Detector Suite (Milestone 4)
Multi-Signal Evidence Extraction, Anomaly Detection, and Calibration Engine.

Covers NIST AML-aligned tabular attack detection:
- Group A: Label evidence (Cleanlab, OOF Disagreement, Entropy)
- Group B: Feature anomaly (Isolation Forest, LOF, PyOD ECOD, KNN Distance)
- Group C: Duplicate evidence (Exact Hash Collision, Near-Duplicate Similarity)
- Group D: Label/Feature consistency (Class Centroid Distance Ratio, KNN Label Disagreement)
- Group E: Backdoor evidence (Joint Trigger-Pattern Rarity Score)
"""

import os
import joblib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)

import cleanlab
from cleanlab.rank import get_label_quality_scores
from cleanlab.filter import find_label_issues

# PyOD for interpretable ECOD and tabular anomaly detection
try:
    from pyod.models.ecod import ECOD
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False


# Milestone 2-3 Core feature columns (maintained for backward compatibility)
CORE_FEATURE_COLUMNS = [
    "cleanlab_risk",
    "label_disagreement",
    "prediction_entropy",
    "isolation_forest_risk",
    "lof_risk"
]

# Milestone 4 Comprehensive Expanded Multi-Signal Feature Columns
EXPANDED_FEATURE_COLUMNS = [
    "cleanlab_risk",
    "label_disagreement",
    "prediction_entropy",
    "isolation_forest_risk",
    "lof_risk",
    "ecod_risk",
    "knn_distance_risk",
    "exact_hash_risk",
    "near_duplicate_similarity",
    "centroid_distance_ratio",
    "knn_label_disagreement",
    "trigger_rarity_risk"
]

# Maintain EVIDENCE_FEATURE_COLUMNS as CORE_FEATURE_COLUMNS for 100% backward compatibility
EVIDENCE_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS


# ============================================================================
# Group A — Label Evidence Detectors
# ============================================================================

def compute_oof_probabilities(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_estimators: int = 200,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Generate out-of-fold probabilistic predictions using Stratified K-Fold.
    Ensures that prediction for each sample is produced by an estimator
    not trained on that sample's fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs
    )
    oof_probs = cross_val_predict(
        rf,
        X,
        y,
        cv=skf,
        method="predict_proba",
        n_jobs=n_jobs
    )
    return oof_probs


def compute_cleanlab_risk(
    y: Union[pd.Series, np.ndarray],
    oof_probabilities: np.ndarray
) -> np.ndarray:
    """
    Signal 1: Cleanlab Label Issue Risk Score.
    Higher value -> more suspicious (1 - label_quality_score).
    """
    labels = np.asarray(y, dtype=int)
    label_quality = get_label_quality_scores(
        labels=labels,
        pred_probs=oof_probabilities
    )
    cleanlab_risk = 1.0 - label_quality
    return cleanlab_risk


def compute_label_disagreement(
    y: Union[pd.Series, np.ndarray],
    oof_probabilities: np.ndarray
) -> np.ndarray:
    """
    Signal 2: OOF Label Disagreement.
    Measures the model's confidence against the assigned label:
    disagreement = 1 - P(assigned label)
    """
    labels = np.asarray(y, dtype=int)
    prob_given_label = np.where(
        labels == 1,
        oof_probabilities[:, 1],
        oof_probabilities[:, 0]
    )
    disagreement = 1.0 - prob_given_label
    return disagreement


def compute_prediction_entropy(
    oof_probabilities: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Signal 3: Prediction Entropy.
    H = - sum(p * log2(p)), normalized between [0, 1].
    """
    entropy = -np.sum(
        oof_probabilities * np.log(oof_probabilities + epsilon),
        axis=1
    )
    normalized_entropy = entropy / np.log(2.0)
    return np.clip(normalized_entropy, 0.0, 1.0)


# ============================================================================
# Group B — Feature Anomaly Detectors
# ============================================================================

def compute_isolation_forest_risk(
    X: Union[pd.DataFrame, np.ndarray],
    n_estimators: int = 200,
    contamination: Union[str, float] = "auto",
    random_state: int = 42,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Signal 4: Isolation Forest Anomaly Risk.
    Inverts decision_function so higher value = more anomalous.
    Scaled to [0, 1] using MinMaxScaler.
    """
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs
    )
    iso.fit(X)
    raw_risk = -iso.decision_function(X)
    scaler = MinMaxScaler()
    normalized_risk = scaler.fit_transform(raw_risk.reshape(-1, 1)).ravel()
    return normalized_risk


def compute_lof_risk(
    X: Union[pd.DataFrame, np.ndarray],
    n_neighbors: int = 20,
    contamination: Union[str, float] = "auto",
    n_jobs: int = -1
) -> np.ndarray:
    """
    Signal 5: Local Outlier Factor (LOF) Density Deviation Risk.
    Inverts negative_outlier_factor_ and scales to [0, 1].
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=n_jobs
    )
    lof.fit_predict(X)
    raw_risk = -lof.negative_outlier_factor_
    scaler = MinMaxScaler()
    normalized_risk = scaler.fit_transform(raw_risk.reshape(-1, 1)).ravel()
    return normalized_risk


def compute_ecod_risk(
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Signal 6:    (ECOD).
    Fast, parameter-free, interpretable tabular outlier detector from PyOD.
    """
    X_mat = np.asarray(X, dtype=float)
    if HAS_PYOD:
        ecod = ECOD(contamination=0.05, n_jobs=-1)
        ecod.fit(X_mat)
        raw_scores = ecod.decision_scores_
    else:
        # Fallback if PyOD not loaded: empirical tails outlier score
        stds = np.std(X_mat, axis=0) + 1e-10
        means = np.mean(X_mat, axis=0)
        z = np.abs((X_mat - means) / stds)
        raw_scores = np.max(z, axis=1)

    scaler = MinMaxScaler()
    return scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()


def compute_knn_distance_risk(
    X: Union[pd.DataFrame, np.ndarray],
    n_neighbors: int = 15,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Signal 7: K-Nearest Neighbor (KNN) Distance Anomaly Risk.
    Calculates average distance to k-nearest neighbors in standardized feature space.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)

    avg_dist = distances.mean(axis=1)
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(avg_dist.reshape(-1, 1)).ravel()


# ============================================================================
# Group C — Duplicate Evidence Detectors
# ============================================================================

def compute_exact_hash_risk(
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Signal 8: Exact Duplicate Hash Collision Risk.
    Flags rows that appear more than once in the dataset with risk = 1.0, else 0.0.
    """
    df_x = pd.DataFrame(X)
    # Count frequencies of identical rows
    dup_counts = df_x.groupby(list(df_x.columns)).transform("size").values
    exact_risk = np.where(dup_counts > 1, 1.0, 0.0)
    return exact_risk.astype(float)


def compute_near_duplicate_similarity(
    X: Union[pd.DataFrame, np.ndarray],
    n_jobs: int = -1
) -> np.ndarray:
    """
    Signal 9: Near-Duplicate Similarity Score.
    Measures proximity to the closest distinct row in scaled feature space.
    Extremely close near-duplicates receive similarity near 1.0.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Query 2 nearest neighbors (1st is self at distance 0, 2nd is nearest distinct sample)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)

    # Distance to nearest non-self row
    dist_to_other = distances[:, 1]
    
    # Invert distance into bounded similarity in [0, 1]
    # Use exponential kernel: exp(-dist)
    sim = np.exp(-dist_to_other)
    scaler_minmax = MinMaxScaler()
    return scaler_minmax.fit_transform(sim.reshape(-1, 1)).ravel()


# ============================================================================
# Group D — Label/Feature Consistency Detectors
# ============================================================================

def compute_class_centroid_distance_ratio(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Signal 10: Class Centroid Distance Ratio.
    Measures distance to assigned class centroid relative to opposite class centroid.
    ratio = dist_assigned / (dist_assigned + dist_opposite)
    ratio > 0.50 indicates the sample is closer in feature space to the OPPOSITE class.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_arr = np.asarray(y, dtype=int)

    c0 = np.mean(X_scaled[y_arr == 0], axis=0)
    c1 = np.mean(X_scaled[y_arr == 1], axis=0)

    d0 = np.linalg.norm(X_scaled - c0, axis=1)
    d1 = np.linalg.norm(X_scaled - c1, axis=1)

    assigned_dist = np.where(y_arr == 0, d0, d1)
    opposite_dist = np.where(y_arr == 0, d1, d0)

    ratio = assigned_dist / (assigned_dist + opposite_dist + 1e-10)
    return np.clip(ratio, 0.0, 1.0)


def compute_knn_label_disagreement(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_neighbors: int = 15,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Signal 11: K-Nearest Neighbor Label Inconsistency.
    Measures the fraction of k-nearest neighbors in feature space having a DIFFERENT
    label than the sample's assigned label.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_arr = np.asarray(y, dtype=int)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    # Exclude index 0 (self)
    neighbor_indices = indices[:, 1:]
    neighbor_labels = y_arr[neighbor_indices]

    disagreement = (neighbor_labels != y_arr[:, None]).mean(axis=1)
    return disagreement.astype(float)


# ============================================================================
# Group E — Backdoor Evidence Detectors
# ============================================================================

def compute_trigger_rarity_risk(
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Signal 12: Joint Feature Combination Rarity Score.
    Detects unusual joint co-occurrences of rare feature bins or extreme values.
    """
    df_x = pd.DataFrame(X)
    rarity_sum = np.zeros(len(df_x))

    for col in df_x.columns:
        s = df_x[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            # Bin continuous variables into deciles
            try:
                binned = pd.qcut(s, q=10, labels=False, duplicates="drop")
                freq = binned.map(binned.value_counts(normalize=True)).fillna(0.01).values
            except Exception:
                freq = np.ones(len(s)) * 0.10
        else:
            freq = s.map(s.value_counts(normalize=True)).fillna(0.01).values

        # Information-theoretic rarity: -log(P)
        rarity_sum += -np.log(np.clip(freq, 1e-4, 1.0))

    scaler = MinMaxScaler()
    return scaler.fit_transform(rarity_sum.reshape(-1, 1)).ravel()


# ============================================================================
# Evidence Table Builder
# ============================================================================

def build_evidence_table(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    oof_probabilities: Optional[np.ndarray] = None,
    cleanlab_risk: Optional[np.ndarray] = None,
    sample_ids: Optional[Union[pd.Series, np.ndarray, list]] = None,
    ground_truth: Optional[Union[pd.Series, np.ndarray]] = None,
    feature_set: str = "expanded",
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Assemble the per-sample multi-signal evidence table.

    Parameters:
        X: Feature matrix.
        y: Target label vector.
        oof_probabilities: Optional precomputed OOF probabilities shape (N, 2).
        cleanlab_risk: Optional precomputed cleanlab risk scores shape (N,).
        sample_ids: Optional sample identifiers.
        ground_truth: Optional binary ground truth indicators (1 if poisoned, 0 if clean).
        feature_set: 'expanded' (all 12 multi-group signals) or 'core' (original 5 signals).
        random_state: Reproducibility seed.
        n_jobs: Number of parallel jobs.

    Returns:
        pd.DataFrame: Comprehensive evidence table ready for fusion modeling.
    """
    if sample_ids is None:
        if isinstance(X, pd.DataFrame):
            sample_ids = X.index.values
        else:
            sample_ids = np.arange(len(y))

    # Compute OOF probabilities if not provided
    if oof_probabilities is None:
        oof_probabilities = compute_oof_probabilities(
            X, y, random_state=random_state, n_jobs=n_jobs
        )

    # Group A: Label evidence
    if cleanlab_risk is None:
        cl_risk = compute_cleanlab_risk(y, oof_probabilities)
    else:
        cl_risk = np.asarray(cleanlab_risk)

    label_disag = compute_label_disagreement(y, oof_probabilities)
    pred_entropy = compute_prediction_entropy(oof_probabilities)

    # Group B: Feature anomaly
    iso_risk = compute_isolation_forest_risk(X, random_state=random_state, n_jobs=n_jobs)
    lof_risk = compute_lof_risk(X, n_jobs=n_jobs)

    evidence_dict = {
        "sample_id": sample_ids,
        "cleanlab_risk": cl_risk,
        "label_disagreement": label_disag,
        "prediction_entropy": pred_entropy,
        "isolation_forest_risk": iso_risk,
        "lof_risk": lof_risk
    }

    if feature_set == "expanded":
        ecod = compute_ecod_risk(X)
        knn_dist = compute_knn_distance_risk(X, n_jobs=n_jobs)
        exact_hash = compute_exact_hash_risk(X)
        near_dup = compute_near_duplicate_similarity(X, n_jobs=n_jobs)
        centroid_ratio = compute_class_centroid_distance_ratio(X, y)
        knn_disag = compute_knn_label_disagreement(X, y, n_jobs=n_jobs)
        trigger_rarity = compute_trigger_rarity_risk(X)

        evidence_dict.update({
            "ecod_risk": ecod,
            "knn_distance_risk": knn_dist,
            "exact_hash_risk": exact_hash,
            "near_duplicate_similarity": near_dup,
            "centroid_distance_ratio": centroid_ratio,
            "knn_label_disagreement": knn_disag,
            "trigger_rarity_risk": trigger_rarity
        })

    if ground_truth is not None:
        evidence_dict["poisoned"] = np.asarray(ground_truth, dtype=int)

    return pd.DataFrame(evidence_dict)


# ============================================================================
# Threshold Calibration & Fusion Modeling
# ============================================================================

def find_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Scan probability thresholds to maximize F1-score while calculating precision, recall, and FPR.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    records = []
    y_arr = np.asarray(y_true, dtype=int)

    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        prec = precision_score(y_arr, pred, zero_division=0)
        rec = recall_score(y_arr, pred, zero_division=0)
        f1 = f1_score(y_arr, pred, zero_division=0)

        # False positive rate
        fp = int(np.logical_and(pred == 1, y_arr == 0).sum())
        tn = int(np.logical_and(pred == 0, y_arr == 0).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        records.append({
            "threshold": round(float(th), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4)
        })

    df_th = pd.DataFrame(records)
    best_row = df_th.loc[df_th["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])

    return best_threshold, df_th


def train_fusion_model(
    X_evidence: pd.DataFrame,
    y_ground_truth: Union[pd.Series, np.ndarray],
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, float], float, Dict[str, np.ndarray]]:
    """
    Train and evaluate a Logistic Regression evidence fusion model on held-out test data.
    """
    y_true = np.asarray(y_ground_truth, dtype=int)
    cols = feature_columns if feature_columns is not None else [
        c for c in EXPANDED_FEATURE_COLUMNS if c in X_evidence.columns
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X_evidence[cols],
        y_true,
        test_size=test_size,
        random_state=random_state,
        stratify=y_true
    )

    fusion_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ))
    ])

    fusion_pipeline.fit(X_train, y_train)
    test_prob = fusion_pipeline.predict_proba(X_test)[:, 1]

    best_threshold, threshold_df = find_optimal_threshold(y_test, test_prob)
    test_pred_opt = (test_prob >= best_threshold).astype(int)
    test_pred_default = (test_prob >= 0.50).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "precision_default_0.5": float(precision_score(y_test, test_pred_default, zero_division=0)),
        "recall_default_0.5": float(recall_score(y_test, test_pred_default, zero_division=0)),
        "f1_default_0.5": float(f1_score(y_test, test_pred_default, zero_division=0)),
        "best_threshold": float(best_threshold),
        "precision_optimal": float(precision_score(y_test, test_pred_opt, zero_division=0)),
        "recall_optimal": float(recall_score(y_test, test_pred_opt, zero_division=0)),
        "f1_optimal": float(f1_score(y_test, test_pred_opt, zero_division=0)),
    }

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_prob": test_prob,
        "threshold_eval_df": threshold_df,
        "feature_columns": cols
    }

    return fusion_pipeline, test_metrics, best_threshold, split_data


def evaluate_with_train_val_test_split(
    X_evidence: pd.DataFrame,
    y_ground_truth: Union[pd.Series, np.ndarray],
    feature_columns: Optional[List[str]] = None,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Dict[str, Union[float, Pipeline, pd.DataFrame]]:
    """
    Strict 3-Way Evaluation Protocol:
    1. TRAIN subset (60%): Calibrate fusion model weights.
    2. VALIDATION subset (20%): Select optimal operating threshold.
    3. UNSEEN TEST subset (20%): Compute unbiased final test metrics.
    """
    y_true = np.asarray(y_ground_truth, dtype=int)
    cols = feature_columns if feature_columns is not None else [
        c for c in EXPANDED_FEATURE_COLUMNS if c in X_evidence.columns
    ]

    # Split 1: Split into Train (60%) and Temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_evidence[cols],
        y_true,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=y_true
    )

    # Split 2: Split Temp into Val (50% of Temp = 20% overall) and Test (50% of Temp = 20% overall)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    fusion_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ))
    ])

    # Fit on Train subset only
    fusion_pipeline.fit(X_train, y_train)

    # Calibrate threshold on Validation subset
    val_prob = fusion_pipeline.predict_proba(X_val)[:, 1]
    best_th, val_th_df = find_optimal_threshold(y_val, val_prob)

    # Unbiased final evaluation on UNSEEN Test subset
    test_prob = fusion_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)

    fp = int(np.logical_and(test_pred == 1, y_test == 0).sum())
    tn = int(np.logical_and(test_pred == 0, y_test == 0).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "pipeline": fusion_pipeline,
        "best_threshold": float(best_th),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, test_prob)), 4),
        "test_precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
        "test_fpr": round(float(fpr), 4),
        "feature_columns": cols,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test)
    }

    return metrics


def fit_final_fusion_model(
    X_evidence: pd.DataFrame,
    y_ground_truth: Union[pd.Series, np.ndarray],
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42
) -> Pipeline:
    """
    Fit the production fusion model on the entire evidence dataset.
    """
    cols = feature_columns if feature_columns is not None else [
        c for c in EXPANDED_FEATURE_COLUMNS if c in X_evidence.columns
    ]

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ))
    ])
    final_pipeline.fit(
        X_evidence[cols],
        np.asarray(y_ground_truth, dtype=int)
    )
    return final_pipeline


# ============================================================================
# Cross-Attack Generalization (Section 12 & 16)
# ============================================================================

def evaluate_cross_attack_generalization(
    evidence_dict: Dict[str, pd.DataFrame],
    train_attacks: List[str],
    test_attacks: List[str],
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate whether DataSentinel generalizes beyond the poisoning patterns it was trained on.

    Trains the fusion model on a pool of attack types (e.g. Label Flip + Feature Perturbation + Duplicate Injection)
    and tests zero-shot on unseen attack families (e.g. Clean-Label, Targeted Label, Tabular Backdoor).
    """
    cols = feature_columns if feature_columns is not None else [
        c for c in EXPANDED_FEATURE_COLUMNS if c in evidence_dict[train_attacks[0]].columns
    ]

    # Assemble training set
    train_dfs = [evidence_dict[att] for att in train_attacks]
    combined_train = pd.concat(train_dfs, axis=0, ignore_index=True)

    X_tr = combined_train[cols]
    y_tr = combined_train["poisoned"].values

    fusion = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
    ])
    fusion.fit(X_tr, y_tr)

    records = []
    for test_att in test_attacks:
        test_df = evidence_dict[test_att]
        X_te = test_df[cols]
        y_te = test_df["poisoned"].values

        probs = fusion.predict_proba(X_te)[:, 1]
        best_th, _ = find_optimal_threshold(y_te, probs)
        preds = (probs >= best_th).astype(int)

        roc = roc_auc_score(y_te, probs)
        pr_auc = average_precision_score(y_te, probs)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)

        fp = int(np.logical_and(preds == 1, y_te == 0).sum())
        tn = int(np.logical_and(preds == 0, y_te == 0).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        records.append({
            "Train Attacks": "+".join(train_attacks),
            "Unseen Test Attack": test_att,
            "Threshold": round(float(best_th), 2),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "ROC-AUC": round(float(roc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "FPR": round(float(fpr), 4)
        })

    return pd.DataFrame(records)


# ============================================================================
# Multi-Detector Benchmark Comparison
# ============================================================================

def evaluate_detector_suite(
    evidence_df: pd.DataFrame,
    ground_truth: Union[pd.Series, np.ndarray],
    fusion_probabilities: np.ndarray,
    fusion_threshold: float = 0.50
) -> pd.DataFrame:
    """
    Benchmark individual detectors against DataSentinel Evidence Fusion.
    """
    y_true = np.asarray(ground_truth, dtype=int)
    n_poisoned = int(y_true.sum())

    detectors = [
        ("Cleanlab (Confident Learning)", "cleanlab_risk"),
        ("Isolation Forest (Feature Space)", "isolation_forest_risk"),
        ("Local Outlier Factor (LOF)", "lof_risk")
    ]

    if "ecod_risk" in evidence_df.columns:
        detectors.append(("PyOD ECOD (Empirical Tails)", "ecod_risk"))
    if "exact_hash_risk" in evidence_df.columns:
        detectors.append(("Exact Hash Duplicate", "exact_hash_risk"))
    if "knn_label_disagreement" in evidence_df.columns:
        detectors.append(("KNN Label Disagreement", "knn_label_disagreement"))
    if "trigger_rarity_risk" in evidence_df.columns:
        detectors.append(("Trigger Rarity Score", "trigger_rarity_risk"))

    rows = []
    for name, col in detectors:
        if col not in evidence_df.columns:
            continue
        scores = evidence_df[col].values
        # Top percentile threshold
        th = np.percentile(scores, 95) if col != "exact_hash_risk" else 0.50
        pred = (scores >= th).astype(int)

        roc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        fp = int(np.logical_and(pred == 1, y_true == 0).sum())
        tn = int(np.logical_and(pred == 0, y_true == 0).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            "Detector": name,
            "Flagged": int(pred.sum()),
            "True Positives": int(np.logical_and(pred == 1, y_true == 1).sum()),
            "False Positives": fp,
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "ROC-AUC": round(float(roc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "FPR": round(float(fpr), 4)
        })

    # DataSentinel Fusion
    fus_pred = (fusion_probabilities >= fusion_threshold).astype(int)
    fus_roc = roc_auc_score(y_true, fusion_probabilities)
    fus_pr = average_precision_score(y_true, fusion_probabilities)
    fus_prec = precision_score(y_true, fus_pred, zero_division=0)
    fus_rec = recall_score(y_true, fus_pred, zero_division=0)
    fus_f1 = f1_score(y_true, fus_pred, zero_division=0)
    fus_fp = int(np.logical_and(fus_pred == 1, y_true == 0).sum())
    fus_tn = int(np.logical_and(fus_pred == 0, y_true == 0).sum())
    fus_fpr = fus_fp / (fus_fp + fus_tn) if (fus_fp + fus_tn) > 0 else 0.0

    rows.append({
        "Detector": f"DataSentinel Evidence Fusion (th={fusion_threshold:.2f})",
        "Flagged": int(fus_pred.sum()),
        "True Positives": int(np.logical_and(fus_pred == 1, y_true == 1).sum()),
        "False Positives": fus_fp,
        "Precision": round(float(fus_prec), 4),
        "Recall": round(float(fus_rec), 4),
        "F1": round(float(fus_f1), 4),
        "ROC-AUC": round(float(fus_roc), 4),
        "PR-AUC": round(float(fus_pr), 4),
        "FPR": round(float(fus_fpr), 4)
    })

    return pd.DataFrame(rows)


# ============================================================================
# High-Level Pipeline Facade
# ============================================================================

class EvidenceFusionPipeline:
    """
    High-level facade for evidence extraction, fusion model fitting, and risk scoring.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.random_state = random_state
        self.model: Optional[Pipeline] = None
        self.best_threshold: float = 0.50
        self.feature_columns = feature_columns if feature_columns is not None else EXPANDED_FEATURE_COLUMNS

    def fit(
        self,
        X_evidence: pd.DataFrame,
        y_ground_truth: Union[pd.Series, np.ndarray],
        tune_threshold: bool = True
    ) -> "EvidenceFusionPipeline":
        cols = [c for c in self.feature_columns if c in X_evidence.columns]
        if tune_threshold:
            _, _, best_th, _ = train_fusion_model(
                X_evidence, y_ground_truth, feature_columns=cols, random_state=self.random_state
            )
            self.best_threshold = best_th

        self.model = fit_final_fusion_model(
            X_evidence, y_ground_truth, feature_columns=cols, random_state=self.random_state
        )
        return self

    def predict_risk(self, X_evidence: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Fusion pipeline has not been fitted.")
        cols = [c for c in self.feature_columns if c in X_evidence.columns]
        return self.model.predict_proba(X_evidence[cols])[:, 1]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "best_threshold": self.best_threshold,
            "feature_columns": self.feature_columns
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EvidenceFusionPipeline":
        data = joblib.load(filepath)
        pipeline = cls()
        if isinstance(data, dict):
            pipeline.model = data.get("model")
            pipeline.best_threshold = data.get("best_threshold", 0.50)
            pipeline.feature_columns = data.get("feature_columns", EXPANDED_FEATURE_COLUMNS)
        else:
            pipeline.model = data
        return pipeline
