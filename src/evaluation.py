import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }

        if y_proba is not None:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)

        return results

    @staticmethod
    def confusion(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def feature_importance(model, feature_names):
        clf = model.named_steps["classifier"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)