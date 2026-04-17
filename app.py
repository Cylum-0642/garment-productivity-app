import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Garment AI Consultant", layout="wide")

LABELS = {
    "smv": "Task Complexity (SMV)",
    "wip": "Current Workload (WIP)",
    "no_of_workers": "Number of Workers",
    "idle_time": "Idle Time (Minutes)",
    "idle_men": "Idle Workers",
    "incentive": "Incentive Amount (Bonus)",
    "over_time": "Overtime (Minutes)",
    "no_of_style_change": "Number of Style Changes"
}

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- TITLE ---
st.title("🧵 Intelligent Production Consultant")
st.caption("Decision Support System for Garment Factory Productivity")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Production Parameters")

    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0, step=0.1)

        if dept == "Finished":
            wip = 0.0
            st.info("WIP locked at 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0, step=10.0)

    with col2:
        team_num = st.selectbox("Team Number", list(range(1, 13)))
        workers = st.number_input(LABELS["no_of_workers"], 1.0, 100.0, 30.0, step=0.5)
        incentive = st.number_input(LABELS["incentive"], 0, 1000, 0)

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)

        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 10000, 0, step=10)
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)

        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Analyze Production Status", type="primary")

# --- PREDICTION LOGIC ---
if submit:

    # FIX 1: Proper overtime scaling (stable transformation)
    ot_scaled = np.log1p(overtime_raw)

    # INIT INPUT DF
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # NUMERIC FEATURES
    numeric_map = {
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time": overtime_raw,
        "over_time_scaled": ot_scaled
    }

    for k, v in numeric_map.items():
        if k in model_columns:
            input_df[k] = float(v)

    # OPTIONAL TEAM FEATURE (SAFE)
    if "team" in model_columns:
        input_df["team"] = float(team_num)

    # CATEGORICAL ENCODING
    def set_dummy(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns:
            input_df[col] = 1.0

    set_dummy("department", dept.lower())
    set_dummy("quarter", quarter)
    set_dummy("no_of_style_change", str(style))

    # ALIGN FEATURES SAFELY
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # --- PREDICTION ---
    probs = pipeline.predict_proba(input_df)[0]
    pred_idx = int(np.argmax(probs))

    # FIX 2: correct label mapping (CRITICAL)
    labels = list(pipeline.classes_)
    status = labels[pred_idx]
    conf = float(np.max(probs))

    # --- SIDEBAR RESULT ---
    st.sidebar.title("📊 Final Result")

    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"

    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <h2 style="margin:0;">{status.upper()}</h2>
            <p style="margin:0;">Productivity Level</p>
            <p style="margin:0;">Confidence: {conf*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN TABS ---
    t1, t2 = st.tabs(["Analysis & Recommendations", "Operational Benchmarks"])

    with t1:
        st.subheader("📊 Probability Breakdown")

        # FIX 3: correct probability alignment
        for i, lab in enumerate(labels):
            st.progress(float(probs[i]), text=f"{lab}: {probs[i]*100:.1f}%")

        st.divider()
        st.subheader("💡 Strategic Recommendations")

        if status == "High":
            st.success("Optimized production detected.")
            st.write(f"Team {team_num} is operating at benchmark efficiency.")

        elif status == "Moderate":
            st.warning("Stable but improvable performance.")
            st.write(f"Incentive level: {incentive} vs expected ~50 benchmark.")

        else:
            st.error("Low productivity risk detected.")
            st.write(f"Idle time: {idle_time} mins is a key driver of inefficiency.")

    with t2:
        st.subheader("📈 Benchmark Comparison")

        benchmarks = {
            "SMV": (smv, 13.7),
            "WIP": (wip, 770.5),
            "Incentive": (incentive, 50.0),
            "Workers": (workers, 33.1)
        }

        cols = st.columns(4)

        for i, (name, (val, avg)) in enumerate(benchmarks.items()):
            diff = val - avg
            cols[i].metric(name, val, f"{diff:.1f} vs High Avg")
