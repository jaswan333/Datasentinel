"""
DataSentinel - Unified Tabular Auditor (Milestone 4.5)
End-to-End Tabular Security Auditing Engine:
1. Schema inspection (numeric, categorical, target, IDs)
2. Mixed-type preprocessing (imputation, scaling, one-hot encoding)
3. Grouped multi-signal evidence extraction (supervised vs unsupervised)
4. Adaptive risk scoring with evidence coverage & renormalization
5. Comprehensive dataset trust report & executive terminal UI card
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .tabular_schema import inspect_tabular_schema
from .tabular_preprocessing import fit_transform_tabular
from .adaptive_risk_score import (
    calculate_sample_risk,
    calculate_dataset_summary,
    format_dataset_audit_card,
    DEFAULT_SUPERVISED_WEIGHTS,
    DEFAULT_UNSUPERVISED_WEIGHTS
)
from .evidence_fusion import (
    compute_oof_probabilities,
    compute_cleanlab_risk,
    compute_label_disagreement,
    compute_prediction_entropy,
    compute_isolation_forest_risk,
    compute_lof_risk,
    compute_exact_hash_risk,
    compute_near_duplicate_similarity,
    compute_class_centroid_distance_ratio,
    compute_knn_label_disagreement,
    compute_trigger_rarity_risk
)

logger = logging.getLogger("datasentinel.auditor")


class TabularAuditResult:
    """Container for tabular audit outputs and reporting utilities."""

    def __init__(
        self,
        schema: Dict[str, Any],
        summary: Dict[str, Any],
        evidence_df: pd.DataFrame,
        audit_card: str,
        transformed_features: pd.DataFrame
    ):
        self.schema = schema
        self.summary = summary
        self.evidence_df = evidence_df
        self.audit_card = audit_card
        self.transformed_features = transformed_features

    def __repr__(self) -> str:
        mode = self.summary.get("mode", "UNKNOWN")
        risk = self.summary.get("dataset_risk_score", "N/A")
        trust = self.summary.get("dataset_trust_score")
        if trust is not None:
            return f"<TabularAuditResult mode='{mode}' trust_score={trust}/100 risk={risk}/100>"
        return f"<TabularAuditResult mode='{mode}' anomaly_risk={risk}/100>"

    def print_card(self) -> None:
        """Print executive ASCII audit card to console."""
        print(self.audit_card)

    def get_suspicious_samples(
        self,
        target_levels: Union[str, List[str]] = "HIGH",
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """Extract filtered suspicious samples sorted by risk score descending."""
        if isinstance(target_levels, str):
            target_levels = [target_levels]

        subset = self.evidence_df[self.evidence_df["risk_level"].isin(target_levels)].copy()
        if "risk_score" in subset.columns:
            subset = subset.sort_values(by="risk_score", ascending=False)
        if top_n is not None:
            subset = subset.head(top_n)
        return subset


def audit_tabular_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    minimum_coverage: float = 0.75,
    disabled_signals: Optional[List[str]] = None,
    random_state: int = 42
) -> TabularAuditResult:
    """
    Perform a complete DataSentinel security audit on any structured tabular dataset.

    Parameters:
        df: Input pandas DataFrame.
        target_column: Optional label column name. If present, runs SUPERVISED audit.
                       If None, runs UNSUPERVISED audit.
        categorical_columns: Optional list of explicit categorical column names.
        sample_size: Optional subsampling limit for ultra-large datasets (for fast interactive audits).
        weights: Custom signal weights. If None, uses adaptive defaults.
        minimum_coverage: Minimum coverage ratio required (default 0.75 / 75%).
        disabled_signals: Optional list of signals to simulate as missing (e.g. ['duplicate_risk']).
        random_state: Seed for reproducibility.

    Returns:
        TabularAuditResult object containing schema, summary metrics, sample-level evidence, and ASCII card.
    """
    df_audit = df.copy()
    if sample_size is not None and len(df_audit) > sample_size:
        df_audit = df_audit.sample(n=sample_size, random_state=random_state).copy()

    disabled = set(disabled_signals or [])

    # 1. Schema Inspection
    schema = inspect_tabular_schema(
        df_audit,
        target_column=target_column,
        categorical_columns=categorical_columns
    )
    mode = schema["mode"]

    numeric_cols = schema["numeric"]
    categorical_cols = schema["categorical"]
    id_cols = schema["id"]

    # 2. Extract feature columns (dropping IDs, text, and target)
    drop_from_features = set(id_cols + schema["text"] + schema["unsupported"])
    if target_column is not None:
        drop_from_features.add(target_column)

    feature_col_candidates = [c for c in df_audit.columns if c not in drop_from_features]
    if not feature_col_candidates:
        raise ValueError("No usable features found after dropping IDs, text, and target column.")

    features_df = df_audit[feature_col_candidates].copy()

    # Determine which features are numeric vs categorical
    active_numeric = [c for c in numeric_cols if c in features_df.columns]
    active_categorical = [c for c in categorical_cols if c in features_df.columns]

    # Preprocessing
    X_transformed, _ = fit_transform_tabular(
        features_df,
        numeric_columns=active_numeric,
        categorical_columns=active_categorical
    )

    evidence_dict: Dict[str, np.ndarray] = {
        "sample_id": np.asarray(df_audit.index)
    }

    # 3. Evidence Extraction
    if mode == "supervised":
        y_raw = df_audit[target_column]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)

        # Signal A: Label Evidence (Cleanlab Confident Learning)
        if "label_risk" not in disabled:
            try:
                oof_probs = compute_oof_probabilities(
                    X_transformed.values,
                    y_encoded,
                    n_estimators=100,
                    n_splits=5,
                    random_state=random_state
                )
                evidence_dict["label_risk"] = compute_cleanlab_risk(y_encoded, oof_probs)
            except Exception as e:
                logger.warning(f"Error computing label_risk: {e}")
                oof_probs = None

        # Signal B: Model Uncertainty (OOF disagreement & Entropy)
        if "uncertainty_risk" not in disabled:
            if "label_risk" not in disabled and oof_probs is not None:
                disag = compute_label_disagreement(y_encoded, oof_probs)
                entropy = compute_prediction_entropy(oof_probs)
                evidence_dict["uncertainty_risk"] = np.clip(0.50 * disag + 0.50 * entropy, 0.0, 1.0)
            else:
                try:
                    oof_probs = compute_oof_probabilities(
                        X_transformed.values,
                        y_encoded,
                        n_estimators=100,
                        n_splits=5,
                        random_state=random_state
                    )
                    disag = compute_label_disagreement(y_encoded, oof_probs)
                    entropy = compute_prediction_entropy(oof_probs)
                    evidence_dict["uncertainty_risk"] = np.clip(0.50 * disag + 0.50 * entropy, 0.0, 1.0)
                except Exception as e:
                    logger.warning(f"Error computing uncertainty_risk: {e}")

        # Signal C: Feature Anomalies (Isolation Forest + LOF)
        if "anomaly_risk" not in disabled:
            try:
                iso_risk = compute_isolation_forest_risk(X_transformed.values, random_state=random_state)
                lof_risk = compute_lof_risk(X_transformed.values)
                evidence_dict["anomaly_risk"] = np.clip(0.50 * iso_risk + 0.50 * lof_risk, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Error computing anomaly_risk: {e}")

        # Signal D: Duplicate Evidence (Exact hash + Near duplicate)
        if "duplicate_risk" not in disabled:
            try:
                exact_risk = compute_exact_hash_risk(features_df)
                near_risk = compute_near_duplicate_similarity(X_transformed.values)
                evidence_dict["duplicate_risk"] = np.clip(0.50 * exact_risk + 0.50 * near_risk, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Error computing duplicate_risk: {e}")

        # Signal E: Consistency Evidence (Centroid distance + KNN label disagreement)
        if "consistency_risk" not in disabled:
            try:
                centroid_risk = compute_class_centroid_distance_ratio(X_transformed.values, y_encoded)
                knn_disag = compute_knn_label_disagreement(X_transformed.values, y_encoded)
                evidence_dict["consistency_risk"] = np.clip(0.50 * centroid_risk + 0.50 * knn_disag, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Error computing consistency_risk: {e}")

    else:
        # UNSUPERVISED MODE
        # Signal A: Feature Anomalies (Isolation Forest + LOF)
        if "anomaly_risk" not in disabled:
            try:
                iso_risk = compute_isolation_forest_risk(X_transformed.values, random_state=random_state)
                lof_risk = compute_lof_risk(X_transformed.values)
                evidence_dict["anomaly_risk"] = np.clip(0.50 * iso_risk + 0.50 * lof_risk, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Error computing unsupervised anomaly_risk: {e}")

        # Signal B: Duplicate Evidence (Exact hash + Near duplicate)
        if "duplicate_risk" not in disabled:
            try:
                exact_risk = compute_exact_hash_risk(features_df)
                near_risk = compute_near_duplicate_similarity(X_transformed.values)
                evidence_dict["duplicate_risk"] = np.clip(0.50 * exact_risk + 0.50 * near_risk, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Error computing unsupervised duplicate_risk: {e}")

        # Signal C: Rarity Evidence (Copula / Rare joint combination)
        if "rarity_risk" not in disabled:
            try:
                evidence_dict["rarity_risk"] = compute_trigger_rarity_risk(features_df)
            except Exception as e:
                logger.warning(f"Error computing unsupervised rarity_risk: {e}")

    # Build raw evidence DataFrame
    evidence_df = pd.DataFrame(evidence_dict, index=df_audit.index)

    # 4. Adaptive Risk Scoring
    scored_df = calculate_sample_risk(
        evidence_df=evidence_df,
        mode=mode,
        weights=weights,
        minimum_coverage=minimum_coverage
    )

    # 5. Dataset-level Summary & ASCII Audit Card
    summary = calculate_dataset_summary(
        risk_df=scored_df,
        mode=mode,
        minimum_coverage=minimum_coverage * 100.0
    )
    summary["dataset_shape"] = {"rows": len(df_audit), "columns": len(df_audit.columns)}
    summary["feature_counts"] = {
        "numeric": len(active_numeric),
        "categorical": len(active_categorical),
        "encoded_dimensions": X_transformed.shape[1]
    }

    card = format_dataset_audit_card(summary)

    return TabularAuditResult(
        schema=schema,
        summary=summary,
        evidence_df=scored_df,
        audit_card=card,
        transformed_features=X_transformed
    )
