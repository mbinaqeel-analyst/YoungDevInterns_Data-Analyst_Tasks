"""
app.py
──────
Premium multi-page Streamlit application.
Pages: Home · Predict · Analytics · Download

Run:  streamlit run app.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from pathlib import Path

from predict  import predict_one, load_pipeline, predict_batch, classify_band
from insights import (
    generate_insights, load_feature_importance,
    salary_by_group, salary_by_experience,
    salary_by_education, salary_by_remote,
)
from utils import (
    TARGET_COL, ALL_FEATURES,
    PIPELINE_PATH, METRICS_PATH, COMPARISON_PATH,
    FEAT_IMP_PATH, PREDICTIONS_PATH, OUTPUTS_DIR,
    fmt_currency,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Intelligence Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography ── */
h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 100%);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    color: white;
    margin-bottom: 0.5rem;
}
.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.85;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.8rem;
    opacity: 0.75;
    margin-top: 0.3rem;
}

/* ── Prediction result card ── */
.predict-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.predict-card.high   { background: linear-gradient(135deg, #064e3b, #10b981); color: white; }
.predict-card.mid    { background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; }
.predict-card.low    { background: linear-gradient(135deg, #78350f, #f59e0b); color: white; }
.predict-card.entry  { background: linear-gradient(135deg, #7f1d1d, #ef4444); color: white; }
.salary-amount { font-size: 3.2rem; font-weight: 800; line-height: 1; }
.salary-range  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }
.salary-band   { font-size: 1.15rem; margin-top: 0.8rem; opacity: 0.9; }

/* ── Insight cards ── */
.insight-card {
    background: #f8fafc;
    border-left: 4px solid #2563EB;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.insight-title   { font-weight: 700; font-size: 1rem; color: #1e3a5f; }
.insight-value   { font-size: 1.6rem; font-weight: 800; color: #2563EB; float: right; }
.insight-finding { font-size: 0.87rem; color: #475569; margin-top: 0.3rem; }
.insight-action  {
    font-size: 0.82rem; color: #0f766e; margin-top: 0.4rem;
    border-top: 1px solid #e2e8f0; padding-top: 0.4rem;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* ── Sidebar nav ── */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 1rem; padding: 0.4rem 0; }

/* ── Tables ── */
.dataframe { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML pipeline…")
def get_pipeline():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading dataset…")
def get_dataset(path: str = "data/salary_data.csv") -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@st.cache_data
def get_comparison() -> pd.DataFrame | None:
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return None


@st.cache_data
def get_feature_importance() -> pd.DataFrame | None:
    return load_feature_importance(top_n=20)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Bold
COLOR_PRIMARY = "#2563EB"

def apply_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>
    """


def model_ready() -> bool:
    return PIPELINE_PATH.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💼 Salary Intelligence")
    st.markdown("*Portfolio Edition*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Home Dashboard", "🔍  Salary Prediction", "📊  Analytics & Insights", "⬇️  Download Reports"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Model status badge
    if model_ready():
        st.success("✅ Model ready")
    else:
        st.error("⚠️ Model not trained")
        st.code("python src/train.py", language="bash")

    st.markdown("---")
    st.caption("Built with scikit-learn · Streamlit · Plotly")
    st.caption("250,000 salary records · 4 ML models")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — HOME DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠  Home Dashboard":
    st.title("Salary Intelligence Platform")
    st.markdown("Machine learning-powered salary analysis trained on **250,000 real salary records**.")

    df      = get_dataset()
    compare = get_comparison()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("### 📌 Dataset Overview")
    k1, k2, k3, k4, k5 = st.columns(5)

    if df is not None:
        k1.markdown(kpi_card("Total Records", f"{len(df):,}", "salary records"), unsafe_allow_html=True)
        k2.markdown(kpi_card("Avg Salary", fmt_currency(df[TARGET_COL].mean()), "annual USD"), unsafe_allow_html=True)
        k3.markdown(kpi_card("Median Salary", fmt_currency(df[TARGET_COL].median()), "50th percentile"), unsafe_allow_html=True)
        k4.markdown(kpi_card("Unique Job Titles", str(df["job_title"].nunique()), "distinct roles"), unsafe_allow_html=True)
        k5.markdown(kpi_card("Industries", str(df["industry"].nunique()), "sectors covered"), unsafe_allow_html=True)
    else:
        st.warning("Dataset not found at `data/salary_data.csv`. Add your file to see stats.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Best Model Metrics ────────────────────────────────────────────────────
    st.markdown("### 🏆 Best Model Performance")
    if compare is not None:
        best = compare.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(kpi_card("Best Model", str(best["Model"]), "auto-selected"), unsafe_allow_html=True)
        m2.markdown(kpi_card("R² Score", f"{best['Test R²']:.4f}", "variance explained"), unsafe_allow_html=True)
        m3.markdown(kpi_card("MAE", fmt_currency(best["Test MAE"]), "avg dollar error"), unsafe_allow_html=True)
        m4.markdown(kpi_card("RMSE", fmt_currency(best["Test RMSE"]), "weighted error"), unsafe_allow_html=True)

        # Model comparison table
        st.markdown("**All Models Comparison**")
        display_cols = ["Model", "Test MAE", "Test RMSE", "Test R²", "CV R² (mean)", "CV R² (std)"]
        available = [c for c in display_cols if c in compare.columns]
        styled = compare[available].style.background_gradient(
            subset=["Test R²"], cmap="Blues"
        ).format({
            "Test MAE": "${:,.0f}", "Test RMSE": "${:,.0f}",
            "Test R²": "{:.4f}", "CV R² (mean)": "{:.4f}", "CV R² (std)": "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Train the model first to see performance metrics.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Salary Distribution Chart ─────────────────────────────────────────────
    if df is not None:
        st.markdown("### 📊 Salary Distribution")
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.histogram(
                df, x=TARGET_COL, nbins=80,
                title="Salary Distribution",
                color_discrete_sequence=[COLOR_PRIMARY],
            )
            fig.update_layout(bargap=0.05)
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            top_jobs = salary_by_group(df, "job_title", top_n=10)
            fig2 = px.bar(
                top_jobs, x="avg_salary", y="job_title",
                orientation="h",
                title="Top 10 Roles by Avg Salary",
                color="avg_salary",
                color_continuous_scale="Blues",
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — SALARY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍  Salary Prediction":
    st.title("🔍 Salary Prediction")
    st.markdown("Enter candidate or job details to get an ML-powered salary estimate.")

    if not model_ready():
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    pipeline = get_pipeline()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.markdown("### Candidate Details")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            job_title = st.selectbox("Job Title", [
                "Data Scientist", "Software Engineer", "ML Engineer",
                "Data Analyst", "Product Manager", "HR Manager",
                "Marketing Manager", "Financial Analyst", "DevOps Engineer",
                "Business Analyst", "UX Designer", "Project Manager",
            ])
        with r1c2:
            industry = st.selectbox("Industry", [
                "Technology", "Finance", "Healthcare", "Education",
                "Retail", "Manufacturing", "Consulting", "Media",
                "Government", "Non-profit", "Real Estate", "Energy",
            ])
        with r1c3:
            location = st.selectbox("Location", [
                "New York", "San Francisco", "Seattle", "Austin",
                "Chicago", "Boston", "Los Angeles", "Denver",
                "Atlanta", "Miami", "Remote", "Other",
            ])

        st.markdown("---")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            experience_years = st.slider("Years Experience", 0, 30, 5)
        with r2c2:
            education_level = st.selectbox(
                "Education", ["High School", "Associate", "Bachelor's", "Master's", "PhD"],
                index=2,
            )
        with r2c3:
            company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"], index=2)
        with r2c4:
            skills_count = st.slider("Skills Count", 1, 25, 8)

        r3c1, r3c2, _ = st.columns([1, 1, 2])
        with r3c1:
            remote_work = st.selectbox("Remote Work", ["Yes", "No"])
        with r3c2:
            certifications = st.selectbox("Certifications", ["Yes", "No"])

        st.markdown("---")
        submitted = st.form_submit_button("🔍  Predict Salary", type="primary", use_container_width=True)

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        raw = dict(
            job_title=job_title, experience_years=experience_years,
            education_level=education_level, skills_count=skills_count,
            industry=industry, company_size=company_size,
            location=location, remote_work=remote_work,
            certifications=certifications,
        )

        with st.spinner("Running ML model…"):
            result = predict_one(raw, pipeline)

        salary = result["predicted_salary"]

        # Card colour
        if salary >= 130_000:   card_cls = "high"
        elif salary >= 85_000:  card_cls = "mid"
        elif salary >= 50_000:  card_cls = "low"
        else:                   card_cls = "entry"

        st.markdown(f"""
        <div class="predict-card {card_cls}">
            <div style="font-size:1rem; opacity:0.85; margin-bottom:0.5rem;">Predicted Annual Salary</div>
            <div class="salary-amount">{result['formatted']}</div>
            <div class="salary-range">95% confidence range: {result['range_formatted']}</div>
            <div class="salary-band">{result['band_emoji']} {result['salary_band']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual (predicted)", result["formatted"])
        c2.metric("Monthly estimate", fmt_currency(salary / 12))
        c3.metric("Salary band", result["salary_band"])

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Personalised Recommendations ─────────────────────────────────────
        st.markdown("### 💡 Personalised Recommendations")

        # Fetch real data for comparison
        df = get_dataset()
        tips = []

        # Experience comparison
        if df is not None:
            avg_exp = df[df["experience_years"] == experience_years][TARGET_COL].mean()
            if not np.isnan(avg_exp) and avg_exp > 0:
                diff = salary - avg_exp
                direction = "above" if diff >= 0 else "below"
                tips.append(f"📊 **Market position**: Your predicted salary is ${abs(diff):,.0f} "
                             f"{direction} the dataset average for {experience_years} years experience.")

        # Education advice
        edu_order = ["High School", "Associate", "Bachelor's", "Master's", "PhD"]
        edu_idx = edu_order.index(education_level)
        if edu_idx < 3:
            tips.append(f"🎓 **Education upgrade**: Advancing to Master's typically provides a significant salary lift based on dataset trends.")
        else:
            tips.append(f"✅ **Strong education profile**: {education_level} is among the highest-paying education tiers in this dataset.")

        if certifications == "No":
            if df is not None:
                with_c = df[df["certifications"].str.lower() == "yes"][TARGET_COL].mean()
                without_c = df[df["certifications"].str.lower() == "no"][TARGET_COL].mean()
                if with_c > without_c:
                    tips.append(f"📜 **Certification value**: Certified professionals earn ${with_c - without_c:,.0f} more on average in this dataset.")
        else:
            tips.append("✅ **Active certifications**: Positively contributing to your salary estimate.")

        if company_size == "Small":
            if df is not None:
                large_avg = df[df["company_size"] == "Large"][TARGET_COL].mean()
                small_avg = df[df["company_size"] == "Small"][TARGET_COL].mean()
                if large_avg > small_avg:
                    tips.append(f"🏢 **Company size effect**: Large companies pay ${large_avg - small_avg:,.0f} more on average. Consider targeting larger organisations.")

        if skills_count < 8:
            tips.append(f"🛠️ **Skills gap**: You have {skills_count} skills. Profiles with 8+ skills tend to command higher salaries in this dataset.")

        for tip in tips:
            st.markdown(f'<div class="insight-card"><div class="insight-finding">{tip}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊  Analytics & Insights":
    st.title("📊 Analytics & Insights")

    df    = get_dataset()
    fi_df = get_feature_importance()

    if df is None:
        st.warning("Dataset not found. Add `data/salary_data.csv` to see charts.")
        st.stop()

    # ── Business Insights Cards ───────────────────────────────────────────────
    st.markdown("### 💡 Data-Driven Business Insights")
    st.caption("All findings are computed from actual dataset statistics — no invented numbers.")

    with st.spinner("Computing insights…"):
        insights = generate_insights(df)

    ins_cols = st.columns(2)
    for i, ins in enumerate(insights):
        with ins_cols[i % 2]:
            st.markdown(f"""
            <div class="insight-card">
                <span class="insight-value">{ins['value']}</span>
                <div class="insight-title">{ins['icon']} {ins['title']}</div>
                <div class="insight-finding">{ins['finding']}</div>
                <div class="insight-action">→ {ins['action']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    if fi_df is not None:
        st.markdown("### 🔬 Feature Importance  (What Drives Salary?)")
        fig = px.bar(
            fi_df.head(15), x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            title="Top 15 Salary Predictors",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = apply_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    st.markdown("### 📈 Salary Trends")
    ca, cb = st.columns(2)

    with ca:
        exp_df = salary_by_experience(df)
        fig = px.line(
            exp_df, x="experience_years", y="avg_salary",
            title="Avg Salary by Years of Experience",
            markers=True,
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        edu_df = salary_by_education(df)
        fig = px.bar(
            edu_df, x="education_level", y="avg_salary",
            title="Avg Salary by Education Level",
            color="avg_salary",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    cc, cd = st.columns(2)

    with cc:
        ind_df = salary_by_group(df, "industry", top_n=10)
        fig = px.bar(
            ind_df, x="avg_salary", y="industry",
            orientation="h",
            title="Avg Salary by Industry",
            color="avg_salary",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig = apply_theme(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with cd:
        co_df = salary_by_group(df, "company_size", top_n=3)
        fig = px.bar(
            co_df, x="company_size", y="avg_salary",
            title="Avg Salary by Company Size",
            color="avg_salary",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Actual vs Predicted Scatter ───────────────────────────────────────────
    test_pred_path = OUTPUTS_DIR / "test_predictions.csv"
    if test_pred_path.exists():
        st.markdown("### 🎯 Actual vs Predicted Salary")
        test_df = pd.read_csv(test_pred_path).sample(min(5000, len(pd.read_csv(test_pred_path))), random_state=42)
        fig = px.scatter(
            test_df, x="actual_salary", y="predicted_salary",
            opacity=0.4,
            color_discrete_sequence=[COLOR_PRIMARY],
            title="Actual vs Predicted Salary  (test set sample — 5,000 pts)",
        )
        # Perfect prediction line
        min_v = min(test_df["actual_salary"].min(), test_df["predicted_salary"].min())
        max_v = max(test_df["actual_salary"].max(), test_df["predicted_salary"].max())
        fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                      line=dict(color="#ef4444", width=2, dash="dash"))
        fig.add_annotation(x=max_v * 0.7, y=max_v * 0.78, text="Perfect prediction line",
                           showarrow=False, font=dict(color="#ef4444", size=11))
        fig = apply_theme(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Actual vs predicted chart will appear after training. Run `python src/train.py`.")

    # ── Salary Band Donut ─────────────────────────────────────────────────────
    st.markdown("### 🎨 Salary Band Distribution")
    bins   = [0, 50_000, 85_000, 130_000, float("inf")]
    labels = ["Entry Level", "Mid Level", "Senior Level", "Executive"]
    df["_band"] = pd.cut(df[TARGET_COL], bins=bins, labels=labels, right=False)
    band_counts = df["_band"].value_counts().reset_index()
    band_counts.columns = ["band", "count"]
    fig = px.pie(
        band_counts, values="count", names="band",
        hole=0.55,
        title="Salary Band Distribution",
        color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6", "#10b981"],
    )
    fig = apply_theme(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — DOWNLOAD REPORTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⬇️  Download Reports":
    st.title("⬇️ Download Reports")
    st.markdown("Export data for Power BI, Excel, or internal reporting.")

    pipeline = get_pipeline()
    df       = get_dataset()

    if not model_ready():
        st.error("Train the model first (`python src/train.py`) before exporting.")
        st.stop()

    st.markdown("---")

    # ── Batch Prediction Export ───────────────────────────────────────────────
    st.markdown("### 📦 Full Predictions Export  (Power BI ready)")
    st.markdown(
        "Generates predictions for every row in the dataset using **vectorized batch prediction** "
        "(not row-by-row — handles 250k rows in seconds)."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        sample_opt = st.selectbox(
            "Export size",
            ["Sample (10,000 rows — fast)", "Full dataset (250,000 rows)"],
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("⚡ Generate Predictions CSV", type="primary", use_container_width=True)

    if gen_btn and df is not None and pipeline is not None:
        n = 10_000 if "Sample" in sample_opt else len(df)
        export_df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

        with st.spinner(f"Running vectorized predictions on {len(export_df):,} rows…"):
            result_df = predict_batch(export_df, pipeline, save=True)

        st.success(f"✅ Exported {len(result_df):,} rows")

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥  Download predictions.csv",
            data=csv,
            file_name="salary_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.dataframe(result_df.head(10), use_container_width=True)

    st.markdown("---")

    # ── Individual file downloads ─────────────────────────────────────────────
    st.markdown("### 📂 Individual Report Files")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown("**Model Comparison**")
        if COMPARISON_PATH.exists():
            st.download_button(
                "⬇️  model_comparison.csv",
                data=open(COMPARISON_PATH, "rb").read(),
                file_name="model_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Not yet generated.")

    with dl2:
        st.markdown("**Feature Importance**")
        if FEAT_IMP_PATH.exists():
            st.download_button(
                "⬇️  feature_importance.csv",
                data=open(FEAT_IMP_PATH, "rb").read(),
                file_name="feature_importance.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Not yet generated.")

    with dl3:
        st.markdown("**Test Set Predictions**")
        test_pred = OUTPUTS_DIR / "test_predictions.csv"
        if test_pred.exists():
            st.download_button(
                "⬇️  test_predictions.csv",
                data=open(test_pred, "rb").read(),
                file_name="test_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Not yet generated.")

    st.markdown("---")

    # ── Power BI Guide ────────────────────────────────────────────────────────
    st.markdown("### 📊 Power BI Integration Guide")
    with st.expander("Click to view recommended dashboard layout"):
        st.markdown("""
**Files to import:**
- `salary_predictions.csv` — full enriched predictions
- `model_comparison.csv` — model performance table
- `feature_importance.csv` — driver analysis

---
**KPI Cards (top row):**
1. Average Actual Salary
2. Average Predicted Salary
3. Average Salary Gap
4. Count by Salary Band

**Charts:**
| Chart | Visual Type | Axes |
|---|---|---|
| Salary by job title | Horizontal bar | job_title → avg salary |
| Salary by industry | Horizontal bar | industry → avg salary |
| Experience trend | Line chart | experience_years → avg salary |
| Actual vs predicted | Scatter plot | predicted ↔ actual |
| Salary bands | Donut chart | salary_band → count |
| Geographic heatmap | Map visual | location → avg salary |

**Slicers (filters):** industry · education_level · company_size · remote_work · salary_band

**Recommended layout:**
- Row 1: 4 KPI cards
- Row 2: Bar (roles) + Line (experience) + Donut (bands)
- Row 3: Scatter (actual vs predicted) + Bar (industry)
- Row 4: Full-width sortable table with salary_gap column
        """)
