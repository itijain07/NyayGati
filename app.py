import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

from model import load_or_train_model, predict_case
from simulation import run_simulation

st.set_page_config(
    page_title="NyayGati",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main { background: #f7f9fc; }
    .block-container { padding-top: 1.5rem; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #102a43 0%, #173f5f 65%, #20639b 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.5rem; }
    .hero p { margin: .35rem 0 0; opacity: .9; }
    .metric-card {
        background: white; padding: 1rem 1.1rem; border-radius: 14px;
        border: 1px solid #e6ebf1; box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }
    .metric-label { color: #64748b; font-size: .86rem; }
    .metric-value { color: #102a43; font-size: 1.55rem; font-weight: 700; }
    .note {
        padding: .8rem 1rem; border-radius: 10px; background: #fff8e8;
        border: 1px solid #f2d28a; color: #664d03;
    }
    .success {
        padding: .8rem 1rem; border-radius: 10px; background: #edf9f1;
        border: 1px solid #b9e5c6; color: #1d5c2e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_or_train_model()

@st.cache_data
def get_data():
    return pd.read_csv("data/cases.csv", parse_dates=["filing_date", "disposal_date"])

model_bundle = get_model()
model = model_bundle[0]
feature_info = model_bundle[1]
data = get_data()

st.sidebar.markdown("## ⚖️ NYAYGATI")
st.sidebar.caption("Predict. Identify. Improve.")
role = st.sidebar.radio("Choose portal", ["Citizen Portal", "Court Analytics", "Admin Sandbox"])

st.markdown("""
<div class="hero">
    <h1>NYAYGATI</h1>
    <p>Judicial Backlog Intelligence & Decision-Support Platform</p>
</div>
""", unsafe_allow_html=True)

if role == "Citizen Portal":
    st.subheader("Citizen Case Prediction")
    st.caption("Estimate a disposal-time range from historical patterns. This is a statistical estimate, not a guaranteed completion date.")

    with st.form("case_form"):
        c1, c2, c3 = st.columns(3)
        court = c1.selectbox("Court", sorted(data["court"].unique()))
        case_type = c2.selectbox("Case Type", sorted(data["case_type"].unique()))
        status = c3.selectbox("Current Status", ["Pending"])
        filing_date = c1.date_input("Filing Date", date(2024, 1, 15))
        hearings = c2.number_input("Hearings so far", min_value=0, max_value=60, value=7)
        adjournments = c3.number_input("Adjournments so far", min_value=0, max_value=40, value=2)
        submitted = st.form_submit_button("Predict disposal horizon", type="primary")

    if submitted:
        result = predict_case(
            model=model,
            court=court,
            case_type=case_type,
            filing_date=pd.Timestamp(filing_date),
            hearings=hearings,
            adjournments=adjournments,
        )

        st.divider()
        st.subheader("Estimated Disposal Horizon")

        m1, m2, m3 = st.columns(3)
        for col, label, value in [
            (m1, "50% horizon", result["p50"]),
            (m2, "75% horizon", result["p75"]),
            (m3, "90% horizon", result["p90"]),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value:.0f} months</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        left, right = st.columns([1.35, 1])

        with left:
            st.markdown("### Probability curve")
            fig = px.line(
                result["curve"],
                x="months",
                y="probability_disposed",
                labels={"months": "Months from filing", "probability_disposed": "Estimated probability of disposal"},
            )
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### Main case factors")
            st.write("The model uses the case characteristics available in the prototype dataset.")
            factors = result["factors"]
            for item in factors:
                st.markdown(f"- **{item}**")
            st.markdown(
                '<div class="note">⚠️ Prototype estimates use synthetic demonstration data. '
                'They must not be interpreted as a real court date or legal advice.</div>',
                unsafe_allow_html=True,
            )

elif role == "Court Analytics":
    st.subheader("Court Administration Analytics")
    st.caption("Aggregate view of pending cases, age patterns and potential delay clusters.")

    pending = data[data["event"] == 0].copy()
    total = len(data)
    pending_count = len(pending)
    avg_age = pending["current_age_months"].mean()
    old_cases = (pending["current_age_months"] >= 60).sum()

    a, b, c, d = st.columns(4)
    metrics = [
        (a, "Total cases", f"{total:,}"),
        (b, "Pending cases", f"{pending_count:,}"),
        (c, "Avg pending age", f"{avg_age:.1f} mo"),
        (d, "Pending ≥ 5 years", f"{old_cases:,}"),
    ]
    for col, label, value in metrics:
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        by_type = pending.groupby("case_type").size().reset_index(name="pending_cases").sort_values("pending_cases", ascending=False)
        fig = px.bar(by_type, x="case_type", y="pending_cases", title="Pending cases by category")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            pending, x="current_age_months", nbins=30,
            title="Age distribution of pending cases",
            labels={"current_age_months": "Current age (months)"}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Potential bottleneck clusters")
    summary = (
        pending.groupby("case_type")
        .agg(
            pending_cases=("case_id", "count"),
            avg_age_months=("current_age_months", "mean"),
            avg_adjournments=("adjournments", "mean"),
            avg_hearings=("hearings", "mean"),
        )
        .reset_index()
    )
    summary["risk_score"] = (
        summary["pending_cases"].rank(pct=True) * 0.4
        + summary["avg_age_months"].rank(pct=True) * 0.35
        + summary["avg_adjournments"].rank(pct=True) * 0.25
    )
    summary = summary.sort_values("risk_score", ascending=False)
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    st.info("These are potential delay clusters for administrative review; the dashboard does not automatically claim a root cause.")

else:
    st.subheader("Admin Sandbox — What-If Backlog Simulation")
    st.caption("Test hypothetical capacity/prioritisation changes without changing any real court schedule.")

    pending = data[data["event"] == 0]
    current_backlog = len(pending)
    current_monthly_capacity = max(1, int(data[data["event"] == 1].groupby(data["disposal_date"].dt.to_period("M")).size().mean()))

    left, right = st.columns([1, 1])
    with left:
        extra_capacity = st.slider("Additional disposals per month", 0, 500, 100, 10)
        priority_share = st.slider("Share of capacity directed to selected priority category", 0, 100, 30, 5)
        priority_category = st.selectbox("Priority category", sorted(data["case_type"].unique()))

    with right:
        st.markdown("### Baseline")
        st.metric("Current pending backlog", f"{current_backlog:,}")
        st.metric("Estimated monthly disposal capacity", f"{current_monthly_capacity:,}")

    if st.button("Run what-if simulation", type="primary"):
        result = run_simulation(
            backlog=current_backlog,
            monthly_capacity=current_monthly_capacity,
            extra_capacity=extra_capacity,
            priority_share=priority_share / 100,
            priority_category=priority_category,
        )

        st.divider()
        x, y, z = st.columns(3)
        x.metric("Baseline clearance", f"{result['baseline_months']:.1f} months")
        y.metric("Simulated clearance", f"{result['simulated_months']:.1f} months")
        z.metric("Projected improvement", f"{result['improvement_pct']:.1f}%")

        chart = pd.DataFrame({
            "Scenario": ["Baseline", "Simulated"],
            "Clearance months": [result["baseline_months"], result["simulated_months"]],
        })
        fig = px.bar(chart, x="Scenario", y="Clearance months", title="Projected clearance time")
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="note">⚠️ Simulation result is a projection based on prototype assumptions. '
            'It is not a guaranteed real-world outcome and does not automatically schedule cases.</div>',
            unsafe_allow_html=True,
        )
