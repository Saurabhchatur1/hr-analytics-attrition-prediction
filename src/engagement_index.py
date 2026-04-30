"""
Engagement Index Module
Standalone engagement scoring with analytics and cohort reporting.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

ENGAGEMENT_COMPONENTS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
]

DEFAULT_WEIGHTS = {
    "JobInvolvement": 0.30,
    "JobSatisfaction": 0.30,
    "EnvironmentSatisfaction": 0.20,
    "RelationshipSatisfaction": 0.20,
}


def compute_engagement_index(
    df: pd.DataFrame,
    method: str = "weighted",
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Computes EngagementIndex [0,1] and EngagementBand.
    method: 'weighted' | 'pca' | 'equal'
    """
    df = df.copy()
    available = [c for c in ENGAGEMENT_COMPONENTS if c in df.columns]

    if not available:
        logger.error("No engagement component columns found!")
        df["EngagementIndex"] = np.nan
        df["EngagementBand"] = np.nan
        return df

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[available]),
        columns=available,
        index=df.index,
    )

    if method == "pca":
        pca = PCA(n_components=1)
        raw = pca.fit_transform(scaled)[:, 0]
        idx = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        logger.info(f"PCA variance explained (PC1): {pca.explained_variance_ratio_[0]:.2%}")

    elif method == "equal":
        idx = scaled[available].mean(axis=1).values

    else:  # weighted
        w = weights or DEFAULT_WEIGHTS
        w_arr = np.array([w.get(c, 1 / len(available)) for c in available])
        w_arr = w_arr / w_arr.sum()
        idx = (scaled.values @ w_arr)

    df["EngagementIndex"] = idx
    df["EngagementBand"] = pd.cut(
        df["EngagementIndex"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["Low", "Medium", "High"],
    )
    return df


def cohort_engagement_report(
    df: pd.DataFrame,
    group_by: List[str],
) -> pd.DataFrame:
    """
    Aggregated engagement statistics by cohort (e.g., Department, JobRole).
    """
    if "EngagementIndex" not in df.columns:
        df = compute_engagement_index(df)

    agg = (
        df.groupby(group_by)["EngagementIndex"]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Std="std",
            P25=lambda x: x.quantile(0.25),
            P75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    agg["Mean"] = agg["Mean"].round(3)
    agg["Median"] = agg["Median"].round(3)
    agg["Std"] = agg["Std"].round(3)
    return agg.sort_values("Mean")


def engagement_attrition_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compares average engagement between attrition Yes/No groups."""
    if "EngagementIndex" not in df.columns:
        df = compute_engagement_index(df)

    target_col = "Attrition" if "Attrition" in df.columns else None
    if target_col is None:
        return pd.DataFrame()

    attrition_col = df[target_col]
    if attrition_col.dtype == object:
        attrition_col = attrition_col.map({"Yes": 1, "No": 0})

    result = (
        df.assign(Attrition_Bin=attrition_col)
        .groupby("Attrition_Bin")["EngagementIndex"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "AvgEngagement", "median": "MedianEngagement", "count": "Count"})
        .reset_index()
    )
    result["Attrition_Label"] = result["Attrition_Bin"].map({1: "Yes", 0: "No"})
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_and_preprocess

    _, df, _ = load_and_preprocess()
    df = compute_engagement_index(df)

    print("\nEngagement Band Distribution:")
    print(df["EngagementBand"].value_counts())

    print("\nDepartment Cohort Report:")
    report = cohort_engagement_report(df, group_by=["Department"])
    print(report.to_string(index=False))

    print("\nAttrition vs Engagement:")
    print(engagement_attrition_correlation(df).to_string(index=False))
