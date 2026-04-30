from setuptools import setup, find_packages

setup(
    name="hr_analytics_platform",
    version="1.0.0",
    description="Employee Engagement, Burnout & Attrition Diagnostic System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.2",
        "xgboost>=2.0.3",
        "imbalanced-learn>=0.12.2",
        "shap>=0.44.1",
        "streamlit>=1.33.0",
        "plotly>=5.20.0",
        "matplotlib>=3.8.4",
        "seaborn>=0.13.2",
        "pyyaml>=6.0.1",
        "joblib>=1.4.0",
        "scipy>=1.13.0",
        "loguru>=0.7.2",
    ],
)
