import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM STYLING (Professional Dashboard Theme)
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%); }
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        color: white;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.35rem; }
    .hero-subtitle { font-size: 1rem; opacity: 0.92; line-height: 1.55; }
    .section-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .mini-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        text-align: center;
    }
    .metric-label { color: #64748b; font-size: 0.9rem; margin-bottom: 0.15rem; }
    .metric-value { color: #0f172a; font-size: 1.45rem; font-weight: 700; }
    .status-box {
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        color: white;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
    }
    .status-value { font-size: 2rem; font-weight: 800; margin-bottom: 0.15rem; }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem 1rem 0.5rem 1rem;
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS & ASSETS
# =========================================================
LABELS = {
    "smv": "Standard Minute Value (SMV)",
    "wip": "Work in Progress (WIP)",
    "no_of_workers": "Number of Workers",
    "idle_time": "Idle Time (Minutes)",
    "idle_men": "Idle Workers",
    "incentive": "Incentive Amount",
    "over_time": "Overtime (Minutes)",
    "no_of_style_change": "Number of Style Changes",
}

AVERAGES = {
    "High": {"smv": 13.7, "wip": 770.5, "incentive": 50.0, "workers": 33.1},
    "Moderate": {"smv": 16.7, "wip": 682.5, "incentive": 34.1, "workers": 37.8},
    "Low": {"smv": 15.5, "wip": 478.0, "incentive": 15.1, "workers": 32.5},
}

CLASS_ORDER = ["High", "Low", "Moderate"]
DISPLAY_ORDER = ["Low", "Moderate", "High"]
STATUS_COLORS = {
    "High": "linear-gradient(135deg, #15803d 0%, #16a34a 100%)",
    "Moderate": "linear-gradient(135deg, #c2410c 0%, #f97316 100%)",
    "Low": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)",
}

@st.cache_resource
def load_assets():
    m_path, c_path = "rf_model.pkl", "rf_columns.pkl"
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# =========================================================
# HELPERS
# =========================================================
def build_input_dataframe(department, quarter, day, team_num, smv, wip, incentive, idle_time, idle_men, workers, overtime_raw, style):
    # Initialize all columns to 0.0
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # 1. Numeric Mapping (We use raw overtime_raw as the Pipeline has its own scaler)
    numeric_map = {
        "team": float(team_num), "smv": smv, "wip": wip, "incentive": incentive,
        "idle_time": idle_time, "idle_men": idle_men, "no_of_workers": workers,
        "over_time_scaled": float(overtime_raw), 
    }
    for key, value in numeric_map.items():
        if key in model_columns:
            input_df[key] = float(value)

    # 2. Dummy Mapping (Prevents overlapping)
    def set_dummy(category, value):
        col_name = f"{category}_{value}"
        if col_name in model_columns:
            input_df[col_name] = 1.0

    set_dummy("department", department.lower())
    set_dummy("quarter", quarter)
    set_dummy("day", day)
    if style > 0:
        set_dummy("no_of_style_change", str(style))

    return input_df

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🏭 Garment Productivity Prediction Dashboard</div>
        <div class="hero-subtitle">
            A professional decision-support prototype for evaluating production conditions and predicting operational efficiency tiers.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# INPUT FORM
# =========================================================
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Enter Production Parameters")
    with st.form("prediction_form"):
        f1, f2 = st.columns(2)
        with f1:
            department = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
            quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
            day = st.selectbox("Production Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
            team_num = st.selectbox("Team Number", list(range(1, 13)))
            smv = st.number_input(LABELS["smv"], 2.9, 60.0, 22.0, step=0.1)
            workers = st.number_input(LABELS["no_of_workers"], 2.0, 90.0, 30.0, step=0.5)

        with f2:
            if department == "Finished":
                wip = 0.0
                st.text_input(LABELS["wip"], value="0 (Locked for Finished)", disabled=True)
            else:
                wip = st.number_input(LABELS["wip"], 0.0, 23500.0, 500.0, step=10.0)
            
            incentive = st.number_input(LABELS["incentive"], 0, 3600, 0)
            overtime_raw = st.number_input(LABELS["over_time"], 0, 10000, 0, step=10)
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

        submit = st.form_submit_button("Generate Prediction", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Reference High-Performance Metrics")
    st.info("The averages below represent lines that consistently achieve 'High' productivity status.")
    c1, c2 = st.columns(2)
    c1.metric("Avg SMV", "13.7", "-3.0 vs Avg")
    c1.metric("Avg Incentive", "50.0", "+15.0 vs Avg")
    c2.metric("Avg Workers", "33.1", "-4.0 vs Avg")
    c2.metric("Avg WIP", "770.5", "+80.0 vs Avg")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RESULTS
# =========================================================
if submit:
    input_df = build_input_dataframe(department, quarter, day, team_num, smv, wip, incentive, idle_time, idle_men, workers, overtime_raw, style)
    
    # Run prediction
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    status = CLASS_ORDER[pred_idx]
    top_prob = float(max(probs))

    st.markdown("---")
    res1, res2 = st.columns([1, 1.3])

    with res1:
        st.markdown(f"""
            <div class="status-box" style="background:{STATUS_COLORS[status]};">
                <div class="status-title">Predicted Tier</div>
                <div class="status-value">{status.upper()}</div>
                <div class="status-note">Confidence: {top_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
    
    with res2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model Probability Distribution")
        for cls in DISPLAY_ORDER:
            p = probs[CLASS_ORDER.index(cls)]
            st.progress(float(p), text=f"{cls}: {p*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabs for detail
    tab1, tab2 = st.tabs(["Strategic Insights", "Benchmark Comparison"])
    
    with tab1:
        if status == "High":
            st.success(f"Team {team_num} is optimized. Maintain current flow and document this shift for training.")
        elif status == "Moderate":
            st.warning(f"Incentive gap detected. Increasing incentives closer to 50.0 may bridge the gap to High status.")
        else:
            st.error(f"Action Required: High idle time ({idle_time}m) or worker shortage detected for this complexity (SMV {smv}).")

    with tab2:
        st.markdown("#### Variance Analysis (vs High Benchmark)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("SMV", f"{smv:.1f}", f"{smv-13.7:.1f}", delta_color="inverse")
        b2.metric("Incentive", f"{incentive}", f"{incentive-50.0:.1f}")
        b3.metric("WIP", f"{wip:.1f}", f"{wip-770.5:.1f}")
        b4.metric("Workers", f"{workers}", f"{workers-33.1:.1f}")

else:
    st.markdown('<div class="mini-card" style="margin-top:2rem;">Configure parameters and click "Generate Prediction" to start analysis.</div>', unsafe_allow_html=True)
