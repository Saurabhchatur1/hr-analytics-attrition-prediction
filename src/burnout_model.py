"""
Burnout Risk Prediction Model
Multi-class classification: Low / Medium / High burnout risk.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

BURNOUT_FEATURES = [
    "OverTime",
    "WorkLifeBalance",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "JobInvolvement",
    "DistanceFromHome",
    "Age",
    "NumCompaniesWorked",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "MonthlyIncome",
    "StockOptionLevel",
    "WorkloadStressIndex",
    "EngagementIndex",
    "SatisfactionStabilityScore",
    "StagnationIndex",
    "CompanyLoyaltyRatio",
]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_burnout_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list, LabelEncoder]:
    """Extract features and encode BurnoutRisk label."""
    from typing import List

    available = [c for c in BURNOUT_FEATURES if c in df.columns]
    X = df[available].copy()

    # Handle OverTime if still string
    if "OverTime" in X.columns and X["OverTime"].dtype == object:
        X["OverTime"] = X["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    # Handle BusinessTravel if present
    if "BusinessTravel" in X.columns and X["BusinessTravel"].dtype == object:
        bt_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
        X["BusinessTravel"] = X["BusinessTravel"].map(bt_map).fillna(0)

    X = X.fillna(X.median(numeric_only=True))

    le = LabelEncoder()
    y = le.fit_transform(df["BurnoutRisk"].astype(str))

    return X.values, y, available, le


class BurnoutRiskModel:
    """
    Burnout risk multi-class classifier with CV and hyperparameter tuning.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.model: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: list = []
        self.cv_results: Dict[str, Any] = {}

    def _get_param_grid(self) -> Tuple[Any, Dict]:
        estimator = RandomForestClassifier(
            random_state=self.config["models"]["random_state"],
            class_weight="balanced",
        )
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
        }
        return estimator, param_grid

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        from typing import List

        logger.info("Preparing burnout training data...")
        X, y, feature_names, le = prepare_burnout_data(df)
        self.feature_names = feature_names
        self.label_encoder = le

        logger.info(f"Feature matrix shape: {X.shape}, Classes: {le.classes_}")

        # SMOTE for class balancing
        smote = SMOTE(random_state=self.config["models"]["random_state"])
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"After SMOTE: {X_res.shape}")

        estimator, param_grid = self._get_param_grid()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])

        cv = StratifiedKFold(n_splits=self.config["models"]["cv_folds"], shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0
        )
        logger.info("Running GridSearchCV for burnout model...")
        grid_search.fit(X_res, y_res)

        self.model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")

        # Cross-validation on original data
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_macro")
        self.cv_results = {
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "best_params": grid_search.best_params_,
        }
        logger.info(f"CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self.cv_results

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X, _, _, _ = prepare_burnout_data(df)
        pred_encoded = self.model.predict(X)
        return pd.Series(self.label_encoder.inverse_transform(pred_encoded), index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _, _, _ = prepare_burnout_data(df)
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.label_encoder.classes_, index=df.index)

    def get_feature_importances(self) -> pd.DataFrame:
        rf = self.model.named_steps["classifier"]
        importances = rf.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )

    def save(self, path: str = "models/burnout_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "le": self.label_encoder, "features": self.feature_names}, f)
        logger.info(f"Burnout model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/burnout_model.pkl") -> "BurnoutRiskModel":
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        instance.model = bundle["model"]
        instance.label_encoder = bundle["le"]
        instance.feature_names = bundle["features"]
        instance.cv_results = {}
        logger.info(f"Burnout model loaded from {path}")
        return instance


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_and_preprocess
    from src.feature_engineering import run_feature_engineering

    _, processed, _ = load_and_preprocess()
    enriched = run_feature_engineering(processed)

    model = BurnoutRiskModel()
    results = model.train(enriched)
    print("\nCV Results:", results)

    importances = model.get_feature_importances()
    print("\nTop 10 Features:")
    print(importances.head(10).to_string(index=False))

    model.save()
