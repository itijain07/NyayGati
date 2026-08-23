import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from model import load_or_train_model, predict_case
from simulation import run_simulation

st.set_page_config(
    page_title="NyayGati | Judicial Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- THEME --------------------
st.markdown(
    """
    <style>
    :root {
        --navy:#0B1F33;
        --blue:#155E8A;
        --teal:#0F766E;
        --ink:#172033;
        --muted:#64748B;
        --line:#E2E8F0;
        --soft:#F8FAFC;
        --white:#FFFFFF;
    }
    .stApp { background: #F4F7FB; }
    /* Force readable dark text in the main content area across Streamlit themes. */
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"],
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] [data-testid="stMetricLabel"],
    section[data-testid="stMain"] [data-testid="stMetricValue"],
    section[data-testid="stMain"] [data-testid="stMetricDelta"] {
        color: #172033 !important;
    }
    section[data-testid="stMain"] .stSelectbox label,
    section[data-testid="stMain"] .stNumberInput label,
    section[data-testid="stMain"] .stDateInput label,
    section[data-testid="stMain"] .stSlider label {
        color: #172033 !important;
    }
    .block-container { max-width: 1400px; padding-top: 1.1rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background: #0B1F33; }
    [data-testid="stSidebar"] * { color: #EAF2F8 !important; }
    [data-testid="stSidebar"] .stRadio label { padding: .45rem .2rem; }
    .brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.3rem; }
    .brand-icon { font-size:1.55rem; }
    .brand-name { font-size:1.25rem; font-weight:800; letter-spacing:.08em; }
    .brand-sub { color:#9FB3C8 !important; font-size:.78rem; margin-bottom:1.2rem; }
    .side-note { border:1px solid #29445E; background:#102A43; border-radius:12px; padding:.75rem; font-size:.78rem; color:#BFD0DF !important; }
    .hero { background:linear-gradient(115deg,#0B1F33 0%,#123C5A 62%,#155E8A 100%); border-radius:20px; padding:1.5rem 1.7rem; color:white; margin-bottom:1rem; box-shadow:0 10px 30px rgba(11,31,51,.13); }
    .hero-row { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
    .hero-kicker { font-size:.76rem; text-transform:uppercase; letter-spacing:.13em; opacity:.72; margin-bottom:.35rem; }
    .hero h1 { margin:0; font-size:2.35rem; letter-spacing:.02em; }
    .hero p { margin:.35rem 0 0; opacity:.86; font-size:1rem; }
    .badge { display:inline-block; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.09); padding:.35rem .65rem; border-radius:999px; font-size:.75rem; white-space:nowrap; }
    .section-title { color:var(--ink); font-size:1.35rem; font-weight:750; margin:.5rem 0 .15rem; }
    .section-sub { color:var(--muted); font-size:.88rem; margin-bottom:1rem; }
    .card { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; box-shadow:0 4px 18px rgba(15,23,42,.04); }
    .metric { background:white; border:1px solid var(--line); border-radius:15px; padding:1rem 1.05rem; min-height:92px; }
    .metric-label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
    .metric-value { color:var(--ink); font-size:1.55rem; font-weight:800; margin-top:.25rem; }
    .metric-delta { color:var(--teal); font-size:.78rem; margin-top:.15rem; }
    .result-card { background:linear-gradient(180deg,#FFFFFF 0%,#F8FBFD 100%); border:1px solid #D9E5EE; border-radius:16px; padding:1.05rem; text-align:center; }
    .result-card .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }
    .result-card .value { color:var(--navy); font-size:1.7rem; font-weight:850; margin:.25rem 0; }
    .result-card .hint { color:var(--muted); font-size:.72rem; }
    .callout { border-radius:12px; padding:.8rem 1rem; border:1px solid #F1D28A; background:#FFF9E8; color:#6B5200; font-size:.82rem; }
    .ok-callout { border-radius:12px; padding:.8rem 1rem; border:1px solid #B9E3D5; background:#ECFDF5; color:#14532D; font-size:.82rem; }
    .mini-title { font-weight:750; color:var(--ink); margin-bottom:.45rem; }
    .footer-note { color:#7A8798; font-size:.72rem; text-align:center; margin-top:1.5rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:.8rem; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_or_train_model()


@st.cache_data
def get_data():
    return pd.read_csv("data/cases.csv", parse_dates=["filing_date", "disposal_date"])


model, feature_info, data = get_model()

# -------------------- SIDEBAR --------------------
st.sidebar.markdown(
    """
    <div class="brand"><span class="brand-icon">⚖️</span><span class="brand-name">NYAYGATI</span></div>
    <div class="brand-sub">Judicial Backlog Intelligence</div>
    """,
    unsafe_allow_html=True,
)

role = st.sidebar.radio(
    "NAVIGATION",
    ["Citizen Portal", "Court Analytics", "Admin Sandbox"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="side-note">
    <b>Prototype mode</b><br>
    Results use synthetic demonstration data and statistical/simulation methods.
    They are not legal advice and do not guarantee a disposal date.
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Model: Cox Proportional Hazards • v1 prototype")

# -------------------- HERO --------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-row">
        <div>
          <div class="hero-kicker">Judicial intelligence & decision support</div>
          <h1>NYAYGATI</h1>
          <p>Predict. Identify. Improve.</p>
        </div>
        <div class="badge">● LIVE PROTOTYPE · SYNTHETIC DATA</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- CITIZEN --------------------
if role == "Citizen Portal":
    st.markdown('<div class="section-title">Citizen Case Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate a statistical disposal horizon from historical case patterns.</div>', unsafe_allow_html=True)

    with st.form("case_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        court = c1.selectbox("Court", sorted(data["court"].unique()))
        case_type = c2.selectbox("Case type", sorted(data["case_type"].unique()))
        status = c3.selectbox("Current status", ["Pending"])
        filing_date = c1.date_input("Filing date", date(2024, 1, 15))
        hearings = c2.number_input("Hearings so far", min_value=0, max_value=60, value=7)
        adjournments = c3.number_input("Adjournments so far", min_value=0, max_value=40, value=2)
        submitted = st.form_submit_button("⚡ Generate disposal estimate", type="primary", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        result = predict_case(
            model=model,
            court=court,
            case_type=case_type,
            filing_date=pd.Timestamp(filing_date),
            hearings=hearings,
            adjournments=adjournments,
        )

        st.markdown("### Estimated disposal horizon")
        r1, r2, r3 = st.columns(3)
        for col, label, value, hint in [
            (r1, "50% horizon", result["p50"], "median-style estimate"),
            (r2, "75% horizon", result["p75"], "higher confidence horizon"),
            (r3, "90% horizon", result["p90"], "long-tail horizon"),
        ]:
            col.markdown(
                f'<div class="result-card"><div class="label">{label}</div><div class="value">{value:.0f} mo</div><div class="hint">{hint}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        left, right = st.columns([1.45, 1])
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="mini-title">Probability of disposal over time</div>', unsafe_allow_html=True)
            fig = px.line(
                result["curve"],
                x="months",
                y="probability_disposed",
                labels={"months": "Months from filing", "probability_disposed": "Estimated probability"},
            )
            fig.add_hline(y=.5, line_dash="dot", annotation_text="50%")
            fig.add_hline(y=.75, line_dash="dot", annotation_text="75%")
            fig.add_hline(y=.9, line_dash="dot", annotation_text="90%")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="mini-title">What influenced this estimate?</div>', unsafe_allow_html=True)
            for item in result["factors"]:
                st.markdown(f"• {item}")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="callout">⚠️ This is a statistical estimate based on synthetic demonstration data. It is not a guaranteed completion date or legal advice.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### How NyayGati works")
        a, b, c = st.columns(3)
        a.markdown('<div class="card"><b>01 · Observe</b><br><span style="color:#64748B">Case attributes, hearings and adjournments become model inputs.</span></div>', unsafe_allow_html=True)
        b.markdown('<div class="card"><b>02 · Predict</b><br><span style="color:#64748B">Survival analysis estimates a distribution instead of one exact date.</span></div>', unsafe_allow_html=True)
        c.markdown('<div class="card"><b>03 · Explain</b><br><span style="color:#64748B">The interface shows horizons and contributing case characteristics.</span></div>', unsafe_allow_html=True)

# -------------------- ANALYTICS --------------------
elif role == "Court Analytics":
    st.markdown('<div class="section-title">Court Administration Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A decision-support view of backlog volume, case age and potential delay clusters.</div>', unsafe_allow_html=True)

    pending = data[data["event"] == 0].copy()
    total = len(data)
    pending_count = len(pending)
    avg_age = pending["current_age_months"].mean()
    old_cases = (pending["current_age_months"] >= 60).sum()

    a, b, c, d = st.columns(4)
    for col, label, value, sub in [
        (a, "Total cases", f"{total:,}", "prototype records"),
        (b, "Pending backlog", f"{pending_count:,}", f"{pending_count/total:.0%} of records"),
        (c, "Average pending age", f"{avg_age:.1f} mo", "among pending cases"),
        (d, "Pending ≥ 5 years", f"{old_cases:,}", "requires attention"),
    ]:
        col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        by_type = pending.groupby("case_type").size().reset_index(name="pending_cases").sort_values("pending_cases", ascending=False)
        fig = px.bar(by_type, x="pending_cases", y="case_type", orientation="h", title="Pending cases by category", labels={"pending_cases":"Pending cases", "case_type":""})
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=55,b=10), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = px.histogram(pending, x="current_age_months", nbins=30, title="Age distribution of pending cases", labels={"current_age_months":"Current age (months)"})
        fig.add_vline(x=60, line_dash="dot", annotation_text="5 years")
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=55,b=10), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
    st.markdown('<div class="callout">ℹ️ These are potential delay clusters for administrative review. The dashboard does not automatically claim a root cause.</div>', unsafe_allow_html=True)

# -------------------- SANDBOX --------------------
else:
    st.markdown('<div class="section-title">Admin Sandbox</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore hypothetical capacity changes without changing any real court schedule.</div>', unsafe_allow_html=True)

    pending = data[data["event"] == 0]
    current_backlog = len(pending)
    current_monthly_capacity = max(1, int(data.loc[data["event"] == 1, "disposal_date"].pipe(pd.to_datetime).dt.to_period("M").value_counts().mean()))

    left, right = st.columns([1.05, .95])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-title">Scenario controls</div>', unsafe_allow_html=True)
        extra_capacity = st.slider("Additional disposals per month", 0, 500, 100, 10)
        priority_share = st.slider("Capacity directed to priority category", 0, 100, 30, 5)
        priority_category = st.selectbox("Priority category", sorted(data["case_type"].unique()))
        run = st.button("▶ Run what-if simulation", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-title">Current baseline</div>', unsafe_allow_html=True)
        x, y = st.columns(2)
        x.metric("Pending backlog", f"{current_backlog:,}")
        y.metric("Monthly disposal capacity", f"{current_monthly_capacity:,}")
        st.markdown("<br><span style='color:#64748B'>Use the controls to test a hypothetical capacity intervention.</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if run:
        result = run_simulation(
            backlog=current_backlog,
            monthly_capacity=current_monthly_capacity,
            extra_capacity=extra_capacity,
            priority_share=priority_share / 100,
            priority_category=priority_category,
        )
        st.markdown("### Scenario result")
        x, y, z = st.columns(3)
        x.metric("Baseline clearance", f"{result['baseline_months']:.1f} months")
        y.metric("Simulated clearance", f"{result['simulated_months']:.1f} months")
        z.metric("Projected improvement", f"{result['improvement_pct']:.1f}%")

        chart = pd.DataFrame({"Scenario": ["Baseline", "Simulated"], "Clearance months": [result["baseline_months"], result["simulated_months"]]})
        fig = px.bar(chart, x="Scenario", y="Clearance months", text="Clearance months", title="Projected clearance time")
        fig.update_traces(texttemplate="%{text:.1f} mo", textposition="outside")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=55,b=10), paper_bgcolor="white", plot_bgcolor="white", yaxis_title="Months", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f'<div class="ok-callout">✓ Scenario tested: +{extra_capacity} disposals/month, with {priority_share}% of capacity directed toward <b>{priority_category}</b>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="callout">⚠️ This is a transparent prototype projection. It does not automatically schedule, rank or reorder real cases.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">NyayGati • SIH prototype • Synthetic demonstration data • Human oversight required</div>', unsafe_allow_html=True)
