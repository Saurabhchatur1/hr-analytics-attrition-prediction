"""
HR Analytics Platform — Streamlit Dashboard
Employee Engagement, Satisfaction, and Burnout Diagnostic System
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Analytics Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Page title ─────────────────────────────────────────────────────── */
.main-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-title { font-size: 1rem; color: #6b7280; margin-top: 4px; margin-bottom: 1.5rem; }

/* ── KPI Cards ───────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    color: #ffffff !important; margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(102,126,234,0.3);
}
.kpi-card.green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 24px rgba(17,153,142,0.3); }
.kpi-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 4px 24px rgba(245,87,108,0.3); }
.kpi-card.blue   { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 4px 24px rgba(79,172,254,0.3); }
.kpi-card.red    { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); box-shadow: 0 4px 24px rgba(250,112,154,0.3); }
.kpi-card .kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85; color: #ffffff !important; }
.kpi-card .kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.2; margin: 4px 0; color: #ffffff !important; }
.kpi-card .kpi-delta { font-size: 0.8rem; opacity: 0.9; color: #ffffff !important; }

/* ── Alert Boxes — explicit dark text, always visible on light bg ────── */
.alert-box {
    background: #fef9c3 !important;
    border-left: 5px solid #f59e0b !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.2rem !important;
    margin: 0.6rem 0 !important;
    font-size: 0.93rem !important;
    color: #713f12 !important;
    font-weight: 600 !important;
    line-height: 1.5 !important;
}
.alert-box strong { color: #713f12 !important; font-weight: 700 !important; }

.alert-box.danger {
    background: #fff1f2 !important;
    border-left-color: #dc2626 !important;
    color: #7f1d1d !important;
}
.alert-box.danger strong { color: #7f1d1d !important; }

.alert-box.success {
    background: #f0fdf4 !important;
    border-left-color: #16a34a !important;
    color: #14532d !important;
}
.alert-box.success strong { color: #14532d !important; }

/* ── Section headers ─────────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #1e293b !important;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 1rem;
}

/* ── Sidebar — scoped dark theme, no bleed into main content ─────────── */
[data-testid="stSidebar"] { background: #0f172a !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] caption { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.6px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading  —  reconstructs OHE columns back to readable strings
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/hr_enriched.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/raw/HR_Employee_Attrition.csv")

    # ── Reconstruct categorical columns from One-Hot Encoded columns ──────────
    def _rebuild_from_ohe(df, prefix, new_col):
        """Finds all columns starting with `prefix_`, picks argmax per row."""
        ohe_cols = [c for c in df.columns if c.startswith(prefix + "_")]
        if ohe_cols and new_col not in df.columns:
            df[new_col] = (
                df[ohe_cols]
                .idxmax(axis=1)
                .str.replace(prefix + "_", "", regex=False)
            )
        return df

    df = _rebuild_from_ohe(df, "Department",    "Department")
    df = _rebuild_from_ohe(df, "JobRole",        "JobRole")
    df = _rebuild_from_ohe(df, "MaritalStatus",  "MaritalStatus")
    df = _rebuild_from_ohe(df, "EducationField", "EducationField")

    # OverTime: numeric → string label for display
    if "OverTime" in df.columns:
        if df["OverTime"].dtype != object:
            df["OverTime_Label"] = df["OverTime"].map({1: "Yes", 0: "No"}).fillna("No")
        else:
            df["OverTime_Label"] = df["OverTime"]
    else:
        df["OverTime_Label"] = "No"

    # Gender: numeric → string label
    if "Gender" in df.columns:
        if df["Gender"].dtype != object:
            df["Gender_Label"] = df["Gender"].map({1: "Male", 0: "Female"}).fillna("Unknown")
        else:
            df["Gender_Label"] = df["Gender"]
    else:
        df["Gender_Label"] = "Unknown"

    # ── Ensure EngagementIndex exists ─────────────────────────────────────────
    if "EngagementIndex" not in df.columns:
        from sklearn.preprocessing import MinMaxScaler
        sat_cols = ["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
        avail = [c for c in sat_cols if c in df.columns]
        if avail:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[avail])
            weights = np.array([0.30, 0.30, 0.20, 0.20][: len(avail)])
            weights /= weights.sum()
            df["EngagementIndex"] = scaled @ weights

    if "EngagementIndex" in df.columns and "EngagementBand" not in df.columns:
        df["EngagementBand"] = pd.cut(
            df["EngagementIndex"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["Low", "Medium", "High"],
        )

    # ── Ensure BurnoutScore / BurnoutRisk exist ───────────────────────────────
    if "BurnoutScore" not in df.columns:
        score = pd.Series(0.0, index=df.index)
        if "OverTime" in df.columns:
            ot = df["OverTime"] if df["OverTime"].dtype in [int, float] else df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)
            score += ot.fillna(0) * 0.35
        if "WorkLifeBalance" in df.columns:
            score += (4 - df["WorkLifeBalance"].clip(1, 4)) / 3.0 * 0.30
        df["BurnoutScore"] = score.clip(0, 1)

    if "BurnoutRisk" not in df.columns:
        df["BurnoutRisk"] = pd.cut(
            df["BurnoutScore"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["Low", "Medium", "High"],
        )

    # ── Attrition binary ─────────────────────────────────────────────────────
    if "Attrition" in df.columns:
        attrition_clean = df["Attrition"].astype(str).str.strip().str.capitalize()

    # Map values safely
    df["Attrition_Binary"] = attrition_clean.map({"Yes": 1, "No": 0})

    # Handle unexpected values
    if df["Attrition_Binary"].isnull().any():
        df["Attrition_Binary"] = df["Attrition_Binary"].fillna(0)

    df["Attrition_Binary"] = df["Attrition_Binary"].astype(int)

    return df


@st.cache_resource
def load_models():
    models = {}
    for name, path in [("attrition", "models/best_attrition_model.pkl"), ("burnout", "models/burnout_model.pkl")]:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                models[name] = pickle.load(f)
    return models


df_full = load_data()
models  = load_models()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Filters  —  uses reconstructed string columns
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 HR Analytics")
    st.markdown("**Employee Diagnostic System**")
    st.markdown("---")

    # Department dropdown
    dept_options = ["All"] + sorted(df_full["Department"].dropna().unique().tolist()) \
        if "Department" in df_full.columns else ["All"]
    dept_filter = st.selectbox("🏢 Department", dept_options)

    # Job Role dropdown — filtered by selected dept
    if dept_filter != "All" and "Department" in df_full.columns:
        roles_src = df_full[df_full["Department"] == dept_filter]
    else:
        roles_src = df_full
    role_options = ["All"] + sorted(roles_src["JobRole"].dropna().unique().tolist()) \
        if "JobRole" in roles_src.columns else ["All"]
    role_filter = st.selectbox("👤 Job Role", role_options)

    # OverTime filter
    overtime_filter = st.selectbox("⏰ OverTime Status", ["All", "Yes", "No"])

    # Engagement threshold slider
    if "EngagementIndex" in df_full.columns:
        min_eng = float(df_full["EngagementIndex"].min())
        max_eng = float(df_full["EngagementIndex"].max())
        eng_threshold = st.slider(
            "📊 Engagement Threshold",
            min_value=min_eng, max_value=max_eng,
            value=min_eng, step=0.01, format="%.2f",
        )
    else:
        eng_threshold = 0.0

    st.markdown("---")
    st.caption("v1.0 · HR Analytics Platform")


# ──────────────────────────────────────────────────────────────────────────────
# Apply Filters
# ──────────────────────────────────────────────────────────────────────────────
df = df_full.copy()

if dept_filter != "All" and "Department" in df.columns:
    df = df[df["Department"] == dept_filter]

if role_filter != "All" and "JobRole" in df.columns:
    df = df[df["JobRole"] == role_filter]

if overtime_filter != "All" and "OverTime_Label" in df.columns:
    df = df[df["OverTime_Label"] == overtime_filter]

if "EngagementIndex" in df.columns:
    df = df[df["EngagementIndex"] >= eng_threshold]

total = max(len(df), 1)  # avoid division by zero


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧠 HR Analytics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Employee Engagement · Burnout Risk · Attrition Diagnostics</p>', unsafe_allow_html=True)

# Active filter pills
active = []
if dept_filter != "All":    active.append(f"🏢 {dept_filter}")
if role_filter != "All":    active.append(f"👤 {role_filter}")
if overtime_filter != "All": active.append(f"⏰ OT={overtime_filter}")
if active:
    st.markdown(f"**Active filters:** {' &nbsp;|&nbsp; '.join(active)} &nbsp; — &nbsp; **{len(df):,} employees**", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Engagement Overview",
    "🔥 Burnout Risk",
    "👥 Role & Career Analysis",
    "🚨 Manager Action Panel",
    "🤖 Model Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Engagement Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    attrition_rate  = df["Attrition_Binary"].mean() * 100 if "Attrition_Binary" in df.columns else 0
    avg_engagement  = df["EngagementIndex"].mean()         if "EngagementIndex"  in df.columns else 0
    high_risk_pct   = (df["BurnoutRisk"] == "High").mean() * 100 if "BurnoutRisk" in df.columns else 0
    low_eng_pct     = (df["EngagementBand"] == "Low").mean() * 100 if "EngagementBand" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Employees</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Avg Engagement</div><div class="kpi-value">{avg_engagement:.2f}</div><div class="kpi-delta">0–1 scale</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Attrition Rate</div><div class="kpi-value">{attrition_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card red"><div class="kpi-label">High Burnout</div><div class="kpi-value">{high_risk_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Low Engagement</div><div class="kpi-value">{low_eng_pct:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Engagement Index Distribution")
        if "EngagementIndex" in df.columns and len(df) > 0:
            fig = px.histogram(df, x="EngagementIndex", nbins=40,
                               color_discrete_sequence=["#667eea"], template="plotly_white")
            fig.add_vline(x=avg_engagement, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"Mean: {avg_engagement:.2f}")
            fig.update_layout(height=320, margin=dict(t=20, b=10), showlegend=False,
                               xaxis_title="Engagement Index", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("##### Engagement Band Breakdown")
        if "EngagementBand" in df.columns and len(df) > 0:
            band_counts = df["EngagementBand"].astype(str).value_counts().reset_index()
            band_counts.columns = ["Band", "Count"]
            fig = px.pie(band_counts, values="Count", names="Band", hole=0.45,
                         color="Band",
                         color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"},
                         template="plotly_white")
            fig.update_layout(height=320, margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown("##### Engagement vs Attrition")
        if "EngagementIndex" in df.columns and "Attrition_Binary" in df.columns and len(df) > 0:
            fig = px.box(df, x="Attrition_Binary", y="EngagementIndex",
                         color="Attrition_Binary",
                         color_discrete_map={0: "#10b981", 1: "#ef4444"},
                         labels={"Attrition_Binary": "Attrition (0=No, 1=Yes)", "EngagementIndex": "Engagement Index"},
                         template="plotly_white")
            fig.update_layout(height=320, margin=dict(t=20, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown("##### Satisfaction Heatmap by Department")
        sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
                    "WorkLifeBalance", "JobInvolvement"]
        avail_sat = [c for c in sat_cols if c in df.columns]
        if avail_sat and "Department" in df.columns and len(df) > 0:
            heat_data = df.groupby("Department")[avail_sat].mean().round(2)
            fig = px.imshow(heat_data.T, text_auto=True, color_continuous_scale="RdYlGn",
                            aspect="auto", template="plotly_white", labels={"color": "Avg Score"})
            fig.update_layout(height=320, margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select 'All' departments to see the heatmap.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Burnout Risk Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Burnout Risk Analysis</div>', unsafe_allow_html=True)

    if "BurnoutRisk" in df.columns and len(df) > 0:
        _br    = df["BurnoutRisk"].astype(str)
        r_low  = (_br == "Low").sum()
        r_med  = (_br == "Medium").sum()
        r_high = (_br == "High").sum()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Low Risk</div><div class="kpi-value">{r_low}</div><div class="kpi-delta">{r_low/total*100:.1f}%</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Medium Risk</div><div class="kpi-value">{r_med}</div><div class="kpi-delta">{r_med/total*100:.1f}%</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-card red"><div class="kpi-label">High Risk</div><div class="kpi-value">{r_high}</div><div class="kpi-delta">{r_high/total*100:.1f}%</div></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Burnout Score Distribution")
        if "BurnoutScore" in df.columns and len(df) > 0:
            _df_plot = df.copy()
            _df_plot["BurnoutRisk"] = _df_plot["BurnoutRisk"].astype(str)
            fig = px.histogram(_df_plot, x="BurnoutScore", color="BurnoutRisk", nbins=30,
                               color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                               template="plotly_white",
                               labels={"BurnoutScore": "Burnout Score (0–1)", "BurnoutRisk": "Risk"})
            fig.update_layout(height=350, margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("##### Burnout Risk by Department")
        if "BurnoutRisk" in df.columns and "Department" in df.columns and len(df) > 0:
            grp = df.groupby(["Department", "BurnoutRisk"]).size().reset_index(name="Count")
            fig = px.bar(grp, x="Department", y="Count", color="BurnoutRisk", barmode="stack",
                         color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                         template="plotly_white")
            fig.update_layout(height=350, margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### High-Risk Employee Segments")
    if "BurnoutRisk" in df.columns and len(df) > 0:
        high_risk = df[df["BurnoutRisk"].astype(str) == "High"].copy()
        show_cols = [c for c in ["Department", "JobRole", "Age", "OverTime_Label",
                                  "WorkLifeBalance", "BurnoutScore", "EngagementIndex"] if c in high_risk.columns]
        if not high_risk.empty:
            st.markdown(
                f'<div class="alert-box danger">' +
                f'⚠️ <strong>{len(high_risk)} employees</strong> at HIGH burnout risk ' +
                f'— immediate intervention required.' +
                f'</div>',
                unsafe_allow_html=True
            )
            show_df = high_risk[show_cols].rename(columns={"OverTime_Label": "OverTime"}).head(20).reset_index(drop=True)
            for col in ["BurnoutScore", "EngagementIndex"]:
                if col in show_df.columns:
                    show_df[col] = show_df[col].round(3)
            st.dataframe(show_df, use_container_width=True, height=300)
        else:
            st.markdown(
                '<div class="alert-box success">✅ No high-risk employees in current selection.</div>',
                unsafe_allow_html=True
            )

    st.markdown("##### OverTime Impact on Burnout Score")
    if "BurnoutScore" in df.columns and "OverTime_Label" in df.columns and len(df) > 0:
        fig = px.box(df, x="OverTime_Label", y="BurnoutScore", color="OverTime_Label",
                     color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                     template="plotly_white",
                     labels={"OverTime_Label": "OverTime", "BurnoutScore": "Burnout Score"})
        fig.update_layout(height=320, margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Role & Career Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Role & Career Analysis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Avg Engagement by Job Role")
        if "JobRole" in df.columns and "EngagementIndex" in df.columns and len(df) > 0:
            role_eng = df.groupby("JobRole")["EngagementIndex"].mean().sort_values().reset_index()
            fig = px.bar(role_eng, x="EngagementIndex", y="JobRole", orientation="h",
                         color="EngagementIndex", color_continuous_scale="RdYlGn",
                         template="plotly_white", labels={"EngagementIndex": "Avg Engagement"})
            fig.update_layout(height=380, margin=dict(t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("##### Attrition Rate by Job Role")
        if "JobRole" in df.columns and "Attrition_Binary" in df.columns and len(df) > 0:
            role_attr = (df.groupby("JobRole")["Attrition_Binary"].mean()
                          .mul(100).sort_values(ascending=False).reset_index())
            role_attr.columns = ["JobRole", "Attrition_Rate"]
            fig = px.bar(role_attr, x="Attrition_Rate", y="JobRole", orientation="h",
                         color="Attrition_Rate", color_continuous_scale="Reds",
                         template="plotly_white", labels={"Attrition_Rate": "Attrition %"})
            fig.update_layout(height=380, margin=dict(t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Tenure vs Engagement")
    if "YearsAtCompany" in df.columns and "EngagementIndex" in df.columns and len(df) > 0:
        color_col = "Department" if "Department" in df.columns else None
        fig = px.scatter(df, x="YearsAtCompany", y="EngagementIndex", color=color_col,
                         opacity=0.65, trendline="lowess", template="plotly_white",
                         labels={"YearsAtCompany": "Years at Company", "EngagementIndex": "Engagement Index"})
        fig.update_layout(height=380, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Job Level vs Monthly Income")
        if "JobLevel" in df.columns and "MonthlyIncome" in df.columns and len(df) > 0:
            fig = px.box(df, x="JobLevel", y="MonthlyIncome", color="JobLevel",
                         color_discrete_sequence=px.colors.qualitative.Set2, template="plotly_white")
            fig.update_layout(height=320, margin=dict(t=20, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Tenure Band Distribution")
        if "TenureBand" in df.columns and len(df) > 0:
            tb = df["TenureBand"].value_counts().reset_index()
            tb.columns = ["Band", "Count"]
            fig = px.bar(tb, x="Band", y="Count", color="Band",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         template="plotly_white")
            fig.update_layout(height=320, margin=dict(t=20, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Stagnation Index by Department")
    if "StagnationIndex" in df.columns and "Department" in df.columns and len(df) > 0:
        stag = df.groupby("Department")["StagnationIndex"].agg(["mean", "std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stag["Department"], y=stag["mean"],
                             error_y=dict(type="data", array=stag["std"].fillna(0)),
                             marker_color=["#667eea", "#f5576c", "#11998e"],
                             name="Avg Stagnation Index"))
        fig.update_layout(template="plotly_white", height=300, margin=dict(t=20, b=10),
                          yaxis_title="Stagnation Index (avg)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Manager Action Panel
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Manager Action Panel</div>', unsafe_allow_html=True)

    if "BurnoutRisk" in df.columns and "EngagementBand" in df.columns and len(df) > 0:
        # Cast to str to safely compare pandas Categorical columns
        burnout_str   = df["BurnoutRisk"].astype(str)
        engagement_str = df["EngagementBand"].astype(str)
        critical = df[(burnout_str == "High") & (engagement_str == "Low")]
        at_risk  = df[(burnout_str == "Medium") & (engagement_str == "Low")]
        st.markdown(
            f'<div class="alert-box danger">' +
            f'🔴 <strong>CRITICAL:</strong> {len(critical)} employees have ' +
            f'HIGH burnout + LOW engagement — immediate action required.' +
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="alert-box">' +
            f'🟡 <strong>WATCH:</strong> {len(at_risk)} employees are ' +
            f'medium burnout risk with low engagement — monitor closely.' +
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("##### 📋 Priority Action List — Top 30 Employees")

    priority_cols = [c for c in ["Department", "JobRole", "Age", "OverTime_Label", "YearsAtCompany",
                                  "BurnoutRisk", "EngagementBand", "BurnoutScore", "EngagementIndex",
                                  "Attrition_Binary"] if c in df.columns]
    if "BurnoutScore" in df.columns and "EngagementIndex" in df.columns and len(df) > 0:
        prio = df[priority_cols].copy()
        prio["Priority_Score"] = prio["BurnoutScore"] * 0.6 + (1 - prio["EngagementIndex"]) * 0.4
        prio = prio.sort_values("Priority_Score", ascending=False).head(30).reset_index(drop=True)
        prio.index = prio.index + 1
        prio = prio.rename(columns={"OverTime_Label": "OverTime"})
        for col in ["BurnoutScore", "EngagementIndex", "Priority_Score"]:
            if col in prio.columns:
                prio[col] = prio[col].round(3)
        st.dataframe(prio, use_container_width=True, height=420)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔴 High Burnout Employees:**
        - Redistribute workload immediately
        - Mandatory flexible working schedule
        - Wellness program enrollment
        - 1-on-1 manager check-in this week
        - Encourage pending leave utilization

        **🟡 Low Engagement Employees:**
        - Career growth conversation
        - Learning & development plan
        - Recognition and rewards review
        - Team bonding activities
        """)
    with col2:
        st.markdown("""
        **📈 Stagnation Risk:**
        - Promotion pathway discussion
        - Internal mobility options
        - Mentorship program pairing
        - Compensation benchmarking review

        **🚨 High Attrition Risk:**
        - Retention bonus evaluation
        - Stay interview within 30 days
        - Special project assignment
        - Remote/hybrid work arrangement
        """)

    st.markdown("---")
    st.markdown("##### 📊 Department Health Scorecard")
    if "Department" in df.columns and len(df) > 0:
        scorecard = {}
        for dept in sorted(df["Department"].dropna().unique()):
            ddf = df[df["Department"] == dept]
            scorecard[dept] = {
                "Employees": len(ddf),
                "Avg Engagement": round(ddf["EngagementIndex"].mean(), 3) if "EngagementIndex" in ddf.columns else "—",
                "Attrition %":    round(ddf["Attrition_Binary"].mean() * 100, 1) if "Attrition_Binary" in ddf.columns else "—",
                "High Burnout %": round((ddf["BurnoutRisk"].astype(str) == "High").mean() * 100, 1) if "BurnoutRisk" in ddf.columns else "—",
                "Low Engage %":   round((ddf["EngagementBand"].astype(str) == "Low").mean() * 100, 1) if "EngagementBand" in ddf.columns else "—",
            }
        st.dataframe(pd.DataFrame(scorecard).T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Model Performance & Explainability</div>', unsafe_allow_html=True)

    if models.get("attrition"):
        bundle    = models["attrition"]
        results   = bundle.get("results", {})
        best_name = bundle.get("model_name", "Best Model")

        st.markdown(f"**Best Model:** `{best_name}`")
        st.markdown("##### Model Comparison")

        rows = []
        for mname, metrics in results.items():
            if isinstance(metrics, dict):
                rows.append({
                    "Model": mname,
                    "Accuracy":      metrics.get("accuracy", "—"),
                    "F1 Score":      metrics.get("f1_score", "—"),
                    "ROC-AUC":       metrics.get("roc_auc", "—"),
                    "Avg Precision": metrics.get("avg_precision", "—"),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        st.markdown("##### Feature Importance")
        model_obj    = bundle["model"]
        feature_names = bundle["features"]
        clf = model_obj.steps[-1][1]
        if hasattr(clf, "feature_importances_"):
            fi = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            fi = np.abs(clf.coef_[0])
        else:
            fi = None

        if fi is not None:
            fi_df = (pd.DataFrame({"Feature": feature_names, "Importance": fi})
                      .sort_values("Importance", ascending=False).head(15))
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues", template="plotly_white")
            fig.update_layout(height=450, margin=dict(t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trained models found. Run `python run_pipeline.py` first.")

    st.markdown("---")
    st.markdown("##### Bias / Fairness Audit")
    if "Attrition_Binary" in df.columns and len(df) > 0:
        # Use reconstructed string columns
        bias_checks = [
            ("Gender",        "Gender_Label"),
            ("Department",    "Department"),
            ("MaritalStatus", "MaritalStatus"),
        ]
        for label, col in bias_checks:
            if col in df.columns and df[col].nunique() > 1:
                grp = (df.groupby(col)["Attrition_Binary"]
                         .agg(["mean", "count"]).reset_index())
                grp.columns = [col, "Attrition_Rate", "Count"]
                grp["Attrition_%"] = (grp["Attrition_Rate"] * 100).round(1)
                fig = px.bar(grp, x=col, y="Attrition_%", text="Attrition_%",
                             color="Attrition_%", color_continuous_scale="RdYlGn_r",
                             template="plotly_white",
                             title=f"Attrition Rate by {label}",
                             labels={"Attrition_%": "Attrition %"})
                fig.update_layout(height=280, margin=dict(t=40, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)