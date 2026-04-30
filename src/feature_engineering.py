"""
Feature Engineering Module
Constructs derived features: Engagement Index, Burnout Risk Score,
Workload Stress Index, Satisfaction Stability Score, and more.
"""

import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EngagementIndexBuilder:
    """
    Computes a composite Engagement Index using weighted average or PCA.
    Components: JobInvolvement, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction.
    """

    COMPONENTS = [
        "JobInvolvement",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config["engagement_index"]
        self.method = self.config.get("method", "weighted")
        self.weights = {c["name"]: c["weight"] for c in self.config["components"]}
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=1)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.COMPONENTS if c in df.columns]
        if not available:
            raise ValueError("No engagement components found in dataframe")

        component_data = df[available].copy()

        # Normalize each component to [0, 1]
        scaled = pd.DataFrame(
            self.scaler.fit_transform(component_data),
            columns=available,
            index=df.index,
        )

        if self.method == "pca":
            logger.info("Computing Engagement Index via PCA")
            scores = self.pca.fit_transform(scaled)
            raw_index = scores[:, 0]
            # Re-scale to [0, 1]
            min_v, max_v = raw_index.min(), raw_index.max()
            df["EngagementIndex"] = (raw_index - min_v) / (max_v - min_v + 1e-9)
            variance_explained = self.pca.explained_variance_ratio_[0]
            logger.info(f"PCA variance explained by PC1: {variance_explained:.2%}")
        else:
            logger.info("Computing Engagement Index via weighted average")
            weights = np.array([self.weights.get(c, 0.25) for c in available])
            weights = weights / weights.sum()  # normalize weights
            df["EngagementIndex"] = scaled.values @ weights

        df["EngagementBand"] = pd.cut(
            df["EngagementIndex"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["Low", "Medium", "High"],
        )
        logger.info(f"Engagement bands:\n{df['EngagementBand'].value_counts().to_string()}")
        return df


class BurnoutRiskScorer:
    """
    Rule-based burnout risk scoring with continuous score and categorical band.
    High risk: OverTime=Yes AND WorkLifeBalance <= 2
    Additional factors: frequent travel, low satisfaction
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["burnout"]["rules"]
        self.thresholds = config["burnout"]["risk_thresholds"]
        self.travel_weights = self.cfg["travel_weights"]
        self.wlb_threshold = self.cfg["wlb_threshold"]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        score = pd.Series(0.0, index=df.index)

        # OverTime component (0-1 normalized)
        if "OverTime" in df.columns:
            ot_val = df["OverTime"]
            if ot_val.dtype == object:
                ot_val = ot_val.map({"Yes": 1, "No": 0})
            score += ot_val.fillna(0) * 0.35

        # Work-life balance (inverted: lower WLB = higher burnout)
        if "WorkLifeBalance" in df.columns:
            wlb_norm = (4 - df["WorkLifeBalance"].clip(1, 4)) / 3.0
            score += wlb_norm * 0.30

        # Business travel
        if "BusinessTravel" in df.columns and df["BusinessTravel"].dtype == object:
            travel_score = df["BusinessTravel"].map(self.travel_weights).fillna(0) / 2.0
            score += travel_score * 0.15
        elif "BusinessTravel_Enc" in df.columns:
            score += df["BusinessTravel_Enc"].fillna(0) / 2.0 * 0.15

        # Low satisfaction (avg of satisfaction cols)
        sat_cols = [c for c in ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"] if c in df.columns]
        if sat_cols:
            avg_sat = df[sat_cols].mean(axis=1)
            sat_burnout = (4 - avg_sat.clip(1, 4)) / 3.0
            score += sat_burnout * 0.20

        # Normalize to [0,1]
        score = score.clip(0, 1)
        df["BurnoutScore"] = score

        df["BurnoutRisk"] = pd.cut(
            score,
            bins=[-0.001, self.thresholds["low"], self.thresholds["medium"], 1.001],
            labels=["Low", "Medium", "High"],
        )
        logger.info(f"Burnout risk distribution:\n{df['BurnoutRisk'].value_counts().to_string()}")
        return df


class WorkloadStressIndexBuilder:
    """
    Composite Workload Stress Index from:
    - OverTime (Yes/No)
    - BusinessTravel frequency
    - DistanceFromHome
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        components = {}

        if "OverTime" in df.columns:
            ot = df["OverTime"]
            if ot.dtype == object:
                ot = ot.map({"Yes": 1, "No": 0})
            components["OT_norm"] = ot.fillna(0)

        if "BusinessTravel_Enc" in df.columns:
            bt = df["BusinessTravel_Enc"].fillna(0)
            components["BT_norm"] = bt / bt.max()
        elif "BusinessTravel" in df.columns and df["BusinessTravel"].dtype == object:
            bt_map = {"Non-Travel": 0, "Travel_Rarely": 0.5, "Travel_Frequently": 1.0}
            components["BT_norm"] = df["BusinessTravel"].map(bt_map).fillna(0)

        if "DistanceFromHome" in df.columns:
            dist_norm = (df["DistanceFromHome"] - df["DistanceFromHome"].min()) / (
                df["DistanceFromHome"].max() - df["DistanceFromHome"].min() + 1e-9
            )
            components["Dist_norm"] = dist_norm

        if components:
            comp_df = pd.DataFrame(components, index=df.index)
            weights = np.array([0.50, 0.30, 0.20][: len(components)])
            weights = weights / weights.sum()
            df["WorkloadStressIndex"] = comp_df.values @ weights
        else:
            df["WorkloadStressIndex"] = 0.0

        logger.info("WorkloadStressIndex computed.")
        return df


class SatisfactionStabilityScorer:
    """
    Computes variance across satisfaction dimensions.
    Lower stability = higher variance = potential flight risk indicator.
    """

    SAT_COLS = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
        "JobInvolvement",
    ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.SAT_COLS if c in df.columns]
        if len(available) < 2:
            df["SatisfactionVariance"] = 0.0
            df["SatisfactionStabilityScore"] = 1.0
            return df

        df["SatisfactionVariance"] = df[available].var(axis=1)
        max_var = df["SatisfactionVariance"].max()
        df["SatisfactionStabilityScore"] = 1 - (df["SatisfactionVariance"] / (max_var + 1e-9))
        logger.info("SatisfactionStabilityScore computed.")
        return df


class TenureRiskFeatureBuilder:
    """Derives tenure-based risk signals."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        if "YearsAtCompany" in df.columns:
            df["TenureBand"] = pd.cut(
                df["YearsAtCompany"],
                bins=[-1, 2, 5, 10, 100],
                labels=["0-2yr", "3-5yr", "6-10yr", "10yr+"],
            )
        if all(c in df.columns for c in ["YearsAtCompany", "YearsSinceLastPromotion"]):
            df["StagnationIndex"] = (
                df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
            ).clip(0, 1)
        if all(c in df.columns for c in ["TotalWorkingYears", "YearsAtCompany"]):
            df["CompanyLoyaltyRatio"] = (
                df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
            ).clip(0, 1)
        return df


def run_feature_engineering(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Master function: runs all feature engineering steps in sequence."""
    config = load_config(config_path)
    logger.info(f"Starting feature engineering. Input shape: {df.shape}")

    df = EngagementIndexBuilder(config).build(df)
    df = BurnoutRiskScorer(config).score(df)
    df = WorkloadStressIndexBuilder().build(df)
    df = SatisfactionStabilityScorer().build(df)
    df = TenureRiskFeatureBuilder().build(df)

    logger.info(f"Feature engineering complete. Output shape: {df.shape}")
    logger.info(f"New features: {[c for c in df.columns if c not in ['Attrition']][-10:]}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_and_preprocess

    _, processed, _ = load_and_preprocess()
    enriched = run_feature_engineering(processed)
    print(f"\nFinal shape: {enriched.shape}")
    print(enriched[["EngagementIndex", "BurnoutScore", "BurnoutRisk", "WorkloadStressIndex", "SatisfactionStabilityScore"]].head(10))
