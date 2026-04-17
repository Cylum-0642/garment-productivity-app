import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment AI Consultant",
    page_icon="🧵",
    layout="wide",
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
}
.result-card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    text-align: center;
}
.status-badge {
    font-size: 2.5rem;
    font-weight: 800;
    padding: 0.5rem 2rem;
    border-radius: 9999px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Missing model files.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# =========================================================
# LABEL HANDLING (CRITICAL FIX)
# =========================================================
def decode_label(label):
    """
    Ensure model output is always converted to business label.
    """
    label_map = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }
    
    if isinstance(label, str):
        return label
    return label_map.get(label, str(label))

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">
    <h1>🧵 Intelligent Production Consultant</h1>
    <p>Operational Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
col_input, col_output = st.columns([1, 1.2])

with col_input:
    with st.form("input_form"):
        dept = st.radio("Department", ["Sewing", "Finished"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        team = st.selectbox("Team", list(range(1, 13)))

        smv = st.number_input("SMV", 2.0, 60.0, 22.0)
        workers = st.number_input("Workers", 1.0, 100.0, 30.0)
        wip = st.number_input("WIP", 0.0, 3000.0, 500.0)

        incentive = st.number_input("Incentive", 0, 3600, 0)
        overtime = st.number_input("Overtime", 0, 10000, 0)

        idle_time = st.number_input("Idle Time", 0.0, 300.0, 0.0)
        idle_men = st.number_input("Idle Workers", 0, 50, 0)
        style = st.selectbox("Style Changes", [0, 1, 2])

        submit = st.form_submit_button("Analyze")

with col_output:
    if submit:

        # VALIDATION
        if dept == "Finished" and wip > 0:
            st.error("Finished department must have WIP = 0")
            st.stop()

        # BUILD INPUT
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

        numeric = {
            'team': team,
            'smv': smv,
            'wip': wip,
            'incentive': incentive,
            'idle_time': idle_time,
            'idle_men': idle_men,
            'no_of_workers': workers,
            'over_time': overtime
        }

        for k, v in numeric.items():
            if k in model_columns:
                input_df[k] = float(v)

        def set_dummy(prefix, val):
            col = f"{prefix}_{val}"
            if col in model_columns:
                input_df[col] = 1.0

        set_dummy('department', dept.lower())
        set_dummy('quarter', quarter)
        set_dummy('day', day)

        if style > 0:
            set_dummy('no_of_style_change', str(style))

        # PREDICTION
        probs = pipeline.predict_proba(input_df)[0]
        labels = list(pipeline.classes_)

        pred_idx = probs.argmax()
        raw_status = labels[pred_idx]

        # FIXED LABEL
        status = decode_label(raw_status)
        conf = probs[pred_idx]

        # COLOR LOGIC (NOW SAFE)
        if status == "High":
            color = "#16a34a"
        elif status == "Moderate":
            color = "#ea580c"
        else:
            color = "#dc2626"

        # DISPLAY
        st.markdown(f"""
        <div class="result-card">
            <p>Prediction</p>
            <div class="status-badge" style="background:{color}">
                {status.upper()}
            </div>
            <p>Confidence: {conf*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # PROBABILITIES
        st.subheader("Probability Breakdown")
        for i, lab in enumerate(labels):
            decoded = decode_label(lab)
            st.progress(float(probs[i]), text=f"{decoded}: {probs[i]*100:.1f}%")

        # BUSINESS LOGIC
        st.subheader("Recommendation")

        if status == "High":
            st.success("Maintain current setup.")
        elif status == "Moderate":
            st.warning("Improve incentives or reduce SMV.")
        else:
            st.error("High risk. Review operations immediately.")

    else:
        st.info("Enter inputs and click Analyze.")
