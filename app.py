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
# CUSTOM STYLING (Kept exactly as you designed)
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        color: white;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
        line-height: 1.55;
    }

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

    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.15rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 700;
    }

    .status-box {
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        color: white;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
    }

    .status-title {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }

    .status-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .status-note {
        font-size: 0.95rem;
        opacity: 0.92;
        line-height: 1.4;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem 1rem 0.5rem 1rem;
        border-radius: 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
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

CLASS_ORDER = ["Low", "Moderate", "High"] # FIXED: Matches Colab mapping (0,1,2)
DISPLAY_ORDER = ["Low", "Moderate", "High"]
STATUS_COLORS = {
    "High": "linear-gradient(135deg, #15803d 0%, #16a34a 100%)",
    "Moderate": "linear-gradient(135deg, #c2410c 0%, #f97316 100%)",
    "Low": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)",
}
STATUS_SUMMARY = {
    "High": "The line is aligned with strong productivity patterns and is likely operating efficiently.",
    "Moderate": "The line is stable, but several production settings can still be improved for better output.",
    "Low": "The line shows a higher risk of inefficiency and may require immediate operational adjustment.",
}

# =========================================================
# LOAD MODEL FILES
# =========================================================
@st.cache_resource
def load_assets():
    model_path = "rf_model.pkl"
    columns_path = "rf_columns.pkl"

    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        st.error("Model files are missing. Please keep 'rf_model.pkl' and 'rf_columns.pkl' in the same folder as app.py.")
        st.stop()

    return joblib.load(model_path), joblib.load(columns_path)

pipeline, model_columns = load_assets()

# =========================================================
# HELPERS (Fixed to match Colab logic)
# =========================================================
def build_input_dataframe(
    department,
    quarter,
    team_num,
    smv,
    wip,
    incentive,
    idle_time,
    idle_men,
    workers,
    overtime_raw,
    style,
):
    # Create empty row with model features
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # Fill Numeric Columns - Send raw values to let the Pipeline Scaler handle it
    numeric_map = {
        "team": float(team_num),
        "smv": float(smv),
        "wip": float(wip),
        "incentive": float(incentive),
        "idle_time": float(idle_time),
        "idle_men": float(idle_men),
        "no_of_workers": float(workers),
        "over_time_scaled": float(overtime_raw), # Pipeline scaler expects raw value
    }

    for key, value in numeric_map.items():
        if key in model_columns:
            input_df[key] = value

    # Map Dummies
    if department.lower() == "sewing" and "department_sewing" in model_columns:
        input_df["department_sewing"] = 1.0
    
    q_col = f"quarter_{quarter}"
    if q_col in model_columns:
        input_df[q_col] = 1.0
        
    if style > 0:
        s_col = f"no_of_style_change_{style}"
        if s_col in model_columns:
            input_df[s_col] = 1.0

    return input_df


def get_priority_actions(status, incentive, wip, workers, smv, idle_time, overtime_raw, team_num):
    if status == "High":
        return [
            f"Maintain current operating conditions for Team {team_num} and use this line as a benchmark for other teams.",
            "Keep overtime controlled to avoid fatigue and to preserve quality consistency.",
            "Document this line's workflow, staffing pattern, and incentive structure for future replication.",
        ]
    if status == "Moderate":
        return [
            f"Review whether the current incentive level ({incentive}) is strong enough to motivate higher line output.",
            f"Check whether WIP accumulation ({wip}) is causing bottlenecks at specific stations.",
            f"Consider small staffing or line-balancing adjustments because the line currently has {workers} workers handling SMV {smv:.1f}.",
        ]
    return [
        f"Investigate the main source of idle time immediately because the current idle time is {idle_time:.1f} minutes.",
        f"Reassess workforce allocation because {workers} workers may be insufficient for an SMV of {smv:.1f} under current conditions.",
        f"Review overtime usage ({overtime_raw} minutes). If overtime is high but output stays weak, fatigue may be reducing efficiency.",
    ]


def comparison_delta(label, value, benchmark):
    diff = value - benchmark
    if label == "SMV":
        if diff <= 0:
            return f"{abs(diff):.1f} below high benchmark"
        return f"{diff:.1f} above high benchmark"

    if diff >= 0:
        return f"{diff:.1f} above high benchmark"
    return f"{abs(diff):.1f} below high benchmark"


def confidence_message(top_probability):
    if top_probability >= 0.80:
        return "High model confidence"
    if top_probability >= 0.60:
        return "Moderate model confidence"
    return "Prediction should be interpreted carefully"


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🏭 Garment Productivity Prediction Dashboard</div>
        <div class="hero-subtitle">
            A professional decision-support prototype for evaluating production conditions and predicting whether a garment line is likely to perform at a <b>Low</b>, <b>Moderate</b>, or <b>High</b> productivity level.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="mini-card"><div class="metric-label">Model</div><div class="metric-value">Random Forest</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="mini-card"><div class="metric-label">Prediction Output</div><div class="metric-value">3 Productivity Tiers</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="mini-card"><div class="metric-label">Prototype Purpose</div><div class="metric-value">Operational Decision Support</div></div>', unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Prototype Overview")
st.sidebar.info("This prototype helps managers evaluate line conditions, estimate productivity tier, and review operational recommendations.")
st.sidebar.markdown("**Predicted classes:** Low, Moderate, High")

# =========================================================
# INPUT FORM
# =========================================================
left_col, right_col = st.columns([1.25, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Enter Production Parameters")

    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            department = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
            quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
            team_num = st.selectbox("Team Number", list(range(1, 13)))
            smv = st.number_input(LABELS["smv"], min_value=2.0, max_value=60.0, value=22.0, step=0.1)
            workers = st.number_input(LABELS["no_of_workers"], min_value=1.0, max_value=100.0, value=30.0, step=0.5)

        with form_col2:
            if department == "Finished":
                wip = 0.0
                st.text_input(LABELS["wip"], value="0 (Locked for Finished department)", disabled=True)
            else:
                wip = st.number_input(LABELS["wip"], min_value=0.0, max_value=25000.0, value=500.0, step=10.0)

            incentive = st.number_input(LABELS["incentive"], min_value=0, max_value=1000, value=0)
            overtime_raw = st.number_input(LABELS["over_time"], min_value=0, max_value=10000, value=0)
            idle_time = st.number_input(LABELS["idle_time"], min_value=0.0, max_value=300.0, value=0.0)
            idle_men = st.number_input(LABELS["idle_men"], min_value=0, max_value=50, value=0)

        style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])
        submit = st.form_submit_button("Generate Productivity Prediction", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How to Read This Prototype")
    st.markdown("""
        **Step 1:** Enter the production conditions for a garment line.  
        **Step 2:** Generate the model prediction.  
        **Step 3:** Review the predicted productivity level.  
        **Step 4:** Use the recommended actions to improve performance.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION SECTION
# =========================================================
if submit:
    input_df = build_input_dataframe(
        department=department, quarter=quarter, team_num=team_num,
        smv=smv, wip=wip, incentive=incentive, idle_time=idle_time,
        idle_men=idle_men, workers=workers, overtime_raw=overtime_raw, style=style,
    )

    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    status = CLASS_ORDER[pred_idx]
    top_probability = float(max(probs))

    st.markdown("### Prediction Results")
    res1, res2 = st.columns([1.05, 1.25])

    with res1:
        st.markdown(f"""
            <div class="status-box" style="background:{STATUS_COLORS[status]};">
                <div class="status-title">Predicted Productivity Level</div>
                <div class="status-value">{status}</div>
                <div class="status-note">{STATUS_SUMMARY[status]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")
        st.metric("Top Prediction Confidence", f"{top_probability * 100:.1f}%", confidence_message(top_probability))

    with res2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Class Probability Distribution")
        for cls in DISPLAY_ORDER:
            probability = probs[CLASS_ORDER.index(cls)]
            st.progress(float(probability), text=f"{cls}: {probability * 100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Benchmark Comparison", "Managerial Recommendations"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if status == "High": st.success("Strong efficiency range detected.")
        elif status == "Moderate": st.warning("Stable, but optimization required.")
        else: st.error("High risk of inefficiency detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("Compare against **High-productivity** benchmarks.")
        b1, b2, b3, b4 = st.columns(4)
        benchmark_map = [("SMV", smv, 13.7), ("WIP", wip, 770.5), ("Incentive", incentive, 50.0), ("Workers", workers, 33.1)]
        for col, (label, value, benchmark) in zip([b1, b2, b3, b4], benchmark_map):
            col.metric(label, f"{value:.1f}", comparison_delta(label, value, benchmark))

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Priority Actions")
        for i, action in enumerate(get_priority_actions(status, incentive, wip, workers, smv, idle_time, overtime_raw, team_num), 1):
            st.markdown(f"**{i}.** {action}")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Enter parameters and click predict to begin.")
