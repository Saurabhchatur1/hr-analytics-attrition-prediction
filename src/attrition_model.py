import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_attrition_data(df: pd.DataFrame, target_col: str = "Attrition"):
    df = df.copy()

    # Encode target
    y = df[target_col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    X = df.drop(columns=[target_col])

    # Encode categoricals safely
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.Categorical(X[col]).codes

    # Fill missing
    X = X.fillna(X.median(numeric_only=True))

    return X.values, y.values, list(X.columns)


class AttritionModel:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ))
        ])
        self.feature_names = []
        self.metrics = {}

    def train(self, df: pd.DataFrame):
        X, y, features = prepare_attrition_data(df)
        self.feature_names = features

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba)
        }

        logger.info(f"Model trained: {self.metrics}")
        return self.metrics

    def predict(self, df: pd.DataFrame):
        X, _, _ = prepare_attrition_data(df)
        return self.model.predict(X)

    def save(self, path="models/attrition_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "features": self.feature_names,
                "metrics": self.metrics
            }, f)

    def load(self, path="models/attrition_model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.metrics = data["metrics"]