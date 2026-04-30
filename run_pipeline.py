"""
Master Pipeline Runner
Run this script to execute the full training pipeline end-to-end.

Usage:
    python run_pipeline.py
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("pipeline")

sys.path.insert(0, ".")


def run():
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Step 1: Data Preprocessing
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preprocessing & Validation")
    logger.info("=" * 60)
    from src.data_preprocessing import load_and_preprocess
    raw_df, processed_df, val_report = load_and_preprocess()
    logger.info(f"Raw shape: {raw_df.shape} | Processed shape: {processed_df.shape}")
    logger.info(f"Validation issues: {sum(len(v) for v in val_report.values() if isinstance(v, dict))}")

    # Step 2: Feature Engineering
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 60)
    from src.feature_engineering import run_feature_engineering
    enriched_df = run_feature_engineering(processed_df)
    enriched_df.to_csv("data/processed/hr_enriched.csv", index=False)
    logger.info(f"Enriched shape: {enriched_df.shape}")
    logger.info(f"Burnout distribution: {enriched_df['BurnoutRisk'].value_counts().to_dict()}")
    logger.info(f"Engagement distribution: {enriched_df['EngagementBand'].value_counts().to_dict()}")

    # Step 3: Attrition Models
    logger.info("=" * 60)
    logger.info("STEP 3: Training Attrition Models (LR + RF + XGBoost)")
    logger.info("=" * 60)
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier

    FEATS = [
        'Age','DistanceFromHome','Education','JobLevel','MonthlyIncome',
        'NumCompaniesWorked','PercentSalaryHike','StockOptionLevel','TotalWorkingYears',
        'TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole',
        'YearsSinceLastPromotion','YearsWithCurrManager','JobInvolvement',
        'JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction',
        'WorkLifeBalance','PerformanceRating','OverTime','EngagementIndex',
        'BurnoutScore','WorkloadStressIndex','SatisfactionStabilityScore',
        'StagnationIndex','CompanyLoyaltyRatio',
    ]
    avail = [c for c in FEATS if c in enriched_df.columns]
    X = enriched_df[avail].fillna(0).values
    y_raw = enriched_df['Attrition']
    y = (y_raw == 1).astype(int) if y_raw.dtype != object else y_raw.map({'Yes':1,'No':0}).fillna(0).astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_tr_r, y_tr_r = sm.fit_resample(X_tr, y_tr)

    attrition_models = {
        'LogisticRegression': Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(C=1, max_iter=1000, random_state=42))]),
        'RandomForest': Pipeline([('sc', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'))]),
        'XGBoost': Pipeline([('sc', StandardScaler()), ('clf', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss'))]),
    }

    results = {}
    best_name, best_auc = '', 0.0
    for name, model in attrition_models.items():
        model.fit(X_tr_r, y_tr_r)
        yp = model.predict(X_te)
        ypr = model.predict_proba(X_te)[:, 1]
        metrics = {
            'accuracy': round(float(accuracy_score(y_te, yp)), 4),
            'f1_score': round(float(f1_score(y_te, yp)), 4),
            'roc_auc': round(float(roc_auc_score(y_te, ypr)), 4),
            'avg_precision': round(float(average_precision_score(y_te, ypr)), 4),
        }
        results[name] = metrics
        logger.info(f"  {name}: Acc={metrics['accuracy']} | F1={metrics['f1_score']} | AUC={metrics['roc_auc']}")
        with open(f'models/attrition_{name.lower()}.pkl', 'wb') as f:
            pickle.dump({'model': model, 'features': avail, 'results': metrics}, f)
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_name = name

    with open('models/best_attrition_model.pkl', 'wb') as f:
        pickle.dump({'model': attrition_models[best_name], 'features': avail, 'model_name': best_name, 'results': results}, f)
    logger.info(f"Best attrition model: {best_name} (AUC={best_auc:.4f})")

    # Step 4: Burnout Model
    logger.info("=" * 60)
    logger.info("STEP 4: Training Burnout Risk Model")
    logger.info("=" * 60)
    from sklearn.preprocessing import LabelEncoder
    BFEATS = ['OverTime','WorkLifeBalance','JobSatisfaction','EnvironmentSatisfaction',
              'RelationshipSatisfaction','JobInvolvement','DistanceFromHome','Age',
              'WorkloadStressIndex','EngagementIndex','SatisfactionStabilityScore']
    bavail = [c for c in BFEATS if c in enriched_df.columns]
    Xb = enriched_df[bavail].fillna(0).values
    le = LabelEncoder()
    yb = le.fit_transform(enriched_df['BurnoutRisk'].astype(str))
    Xb_r, yb_r = sm.fit_resample(Xb, yb)
    bf = Pipeline([('sc', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])
    bf.fit(Xb_r, yb_r)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(bf, Xb, yb, cv=cv, scoring='f1_macro')
    logger.info(f"Burnout CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    with open('models/burnout_model.pkl', 'wb') as f:
        pickle.dump({'model': bf, 'le': le, 'features': bavail}, f)

    # Step 5: Generate Reports
    logger.info("=" * 60)
    logger.info("STEP 5: Generating Reports")
    logger.info("=" * 60)
    _generate_model_report(results, best_name, enriched_df, cv_scores.mean())

    logger.info("=" * 60)
    logger.info("✅ PIPELINE COMPLETE — All models trained and saved.")
    logger.info("Run: streamlit run app/streamlit_app.py")
    logger.info("=" * 60)


def _generate_model_report(results, best_name, df, burnout_f1):
    lines = [
        "# HR Analytics: Model Performance Report",
        f"\n**Generated by:** `run_pipeline.py`\n",
        "## Attrition Model Comparison",
        "| Model | Accuracy | F1 Score | ROC-AUC | Avg Precision |",
        "|-------|----------|----------|---------|---------------|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['accuracy']} | {m['f1_score']} | {m['roc_auc']} | {m['avg_precision']} |")
    lines += [
        f"\n**Best Model:** {best_name}",
        f"\n## Burnout Risk Model\n- CV F1-Macro: **{burnout_f1:.4f}**",
        "\n## Feature Distribution Summary",
    ]
    if 'EngagementBand' in df.columns:
        for band, cnt in df['EngagementBand'].value_counts().items():
            lines.append(f"- Engagement {band}: {cnt} ({cnt/len(df)*100:.1f}%)")
    if 'BurnoutRisk' in df.columns:
        for risk, cnt in df['BurnoutRisk'].value_counts().items():
            lines.append(f"- Burnout {risk}: {cnt} ({cnt/len(df)*100:.1f}%)")
    with open("reports/insights.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
