# 🧠 HR Analytics Platform
### Employee Engagement, Satisfaction & Burnout Diagnostic System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

---

## Overview

A production-grade HR analytics system that predicts, diagnoses, and visualizes:
- **Employee Engagement** (weighted composite index)
- **Burnout Risk** (Low / Medium / High multi-class)
- **Attrition Signals** (binary classification, 3 models)
- **Actionable Manager Alerts** (priority-ranked intervention list)

---

## Project Structure

```
hr_analytics_project/
├── data/
│   ├── raw/                    # Raw HR dataset (CSV)
│   └── processed/              # Cleaned & feature-engineered data
├── src/
│   ├── data_preprocessing.py   # Validation, cleaning, encoding
│   ├── feature_engineering.py  # Engagement Index, Burnout Score, etc.
│   ├── engagement_index.py     # Standalone engagement module + cohort analytics
│   ├── attrition_model.py      # LR + RF + XGBoost with CV & GridSearch
│   ├── burnout_model.py        # RF multi-class burnout classifier
│   └── evaluation.py           # Metrics, SHAP, bias check, report generation
├── app/
│   └── streamlit_app.py        # 5-tab professional dashboard
├── models/                     # Trained model artifacts (.pkl)
├── reports/
│   ├── insights.md             # Research report
│   └── executive_summary.md   # Business summary & ROI analysis
├── config/
│   └── config.yaml             # Centralized configuration
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/hr-analytics-platform.git
cd hr-analytics-platform

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
# Step 1: Preprocess data
python src/data_preprocessing.py

# Step 2: Feature engineering
python src/feature_engineering.py

# Step 3: Train attrition models (LR, RF, XGBoost)
python src/attrition_model.py

# Step 4: Train burnout model
python src/burnout_model.py
```

### 3. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Docker Deployment

### Local Docker Run

```bash
# Build image
docker build -t hr-analytics:latest .

# Run container
docker run -p 8501:8501 hr-analytics:latest

# With volume mount (persist models)
docker run -p 8501:8501 -v $(pwd)/models:/app/models hr-analytics:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  hr-analytics:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

```bash
docker-compose up -d
```

---

## Cloud Deployment

### Streamlit Community Cloud (Free)

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app/streamlit_app.py` as main file
4. Deploy

### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize & deploy
eb init hr-analytics --platform docker
eb create hr-analytics-prod
eb deploy
```

### AWS ECS / Fargate

```bash
# Build & push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t hr-analytics .
docker tag hr-analytics:latest <ecr-uri>:latest
docker push <ecr-uri>:latest

# Deploy via ECS task definition (see infra/ecs-task.json)
```

---

## Dashboard Modules

| Tab | Description |
|-----|-------------|
| 📊 Engagement Overview | KPI cards, distribution plots, satisfaction heatmap |
| 🔥 Burnout Risk | Risk scores, department breakdowns, high-risk employee list |
| 👥 Role & Career | Engagement by role, tenure trends, stagnation analysis |
| 🚨 Manager Action Panel | Priority intervention list, department health scorecard |
| 🤖 Model Performance | Model comparison, feature importance, bias audit |

---

## Configuration

All parameters are centralized in `config/config.yaml`:

```yaml
engagement_index:
  method: "weighted"     # Options: weighted | pca | equal
  components:
    - name: "JobInvolvement"
      weight: 0.30
    ...

burnout:
  risk_thresholds:
    low: 0.33
    medium: 0.66

models:
  cv_folds: 5
  test_size: 0.2
```

---

## Feature Engineering Summary

| Feature | Description |
|---------|-------------|
| `EngagementIndex` | Weighted composite of 4 satisfaction dimensions, scaled [0,1] |
| `BurnoutScore` | Rule-based composite: overtime, WLB, travel, satisfaction |
| `BurnoutRisk` | Categorical: Low / Medium / High |
| `WorkloadStressIndex` | Overtime + travel + distance composite |
| `SatisfactionStabilityScore` | 1 - normalized variance across satisfaction columns |
| `StagnationIndex` | Promotion recency ratio |
| `CompanyLoyaltyRatio` | Years at company / total working years |
| `TenureBand` | 0-2yr / 3-5yr / 6-10yr / 10yr+ |

---

## Model Architecture

```
Attrition Prediction:
├── Logistic Regression   (baseline)
├── Random Forest         (ensemble)
└── XGBoost               (gradient boosting)
    └── Best model saved → models/best_attrition_model.pkl

Burnout Risk:
└── Random Forest (multi-class, SMOTE balanced)
    └── Saved → models/burnout_model.pkl

Validation:
├── 5-fold Stratified Cross-Validation
├── GridSearchCV hyperparameter tuning
└── SMOTE class balancing (training set only)
```

---

## Requirements

- Python 3.11+
- pandas, numpy, scikit-learn, xgboost, imbalanced-learn
- shap (explainability)
- streamlit, plotly (dashboard)
- pyyaml, joblib, scipy, loguru

See `requirements.txt` for pinned versions.

---

## License

MIT License © 2025 HR Analytics Platform

---
*Built for enterprise HR teams. Production-ready. Scalable.*
