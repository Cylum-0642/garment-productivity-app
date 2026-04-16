import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment AI: Consultant Mode", layout="wide")

# Hardcoded Averages from your Cleaned Dataset for Comparison
AVERAGES = {
    'High':     {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1},
    'Moderate': {'smv': 16.7, 'wip': 682.5, 'incentive': 34.1, 'workers': 37.8},
    'Low':      {'smv': 15.5, 'wip': 478.0, 'incentive': 15.1, 'workers': 32.5}
}

@st.cache_resource
def load_assets():
    # Using your updated filenames
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error(f"Missing files: {m_path} or {c_path}")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- 2. MAIN UI ---
st.title("🧵 Intelligent Production Consultant")
st.markdown("Enter factory data to receive a productivity forecast and feature analysis.")

with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📅 Context")
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finished"])
    
    with c2:
        st.subheader("⚙️ Complexity")
        smv = st.number_input("SMV (Complexity)", 2.0, 60.0, 22.0)
        wip = st.number_input("WIP (Workload)", 0.0, 25000.0, 500.0)
        workers = st.number_input("No. of Workers", 1.0, 100.0, 30.0)
        style = st.selectbox("Style Changes", [0, 1, 2])

    with c3:
        st.subheader("💰 Performance")
        incentive = st.number_input("Incentive Amount", 0, 1000, 0)
        overtime_raw = st.number_input("Overtime (Minutes)", 0, 5000, 0)
        idle_time = st.number_input("Idle Time", 0.0, 300.0, 0.0)
        idle_men = st.number_input("Idle Men", 0, 50, 0)

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True, type="primary")

# --- 3. LOGIC & PREDICTION ---
if submit:
    # A. Modified Z-Score Scaling for Overtime
    # Median = 0, MAD = 2520 based on your cleaned dataset
    ot_scaled = (overtime_raw - 0.0) / (2520.0 * 1.4826) if overtime_raw > 0 else -0.5

    # B. Build Feature Dataframe
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # Map Numerics
    numeric_map = {
        'team': 1.0, 'smv': smv, 'wip': wip, 'incentive': incentive,
        'idle_time': idle_time, 'idle_men': idle_men, 'no_of_workers': workers,
        'over_time_scaled': ot_scaled
    }
    for k, v in numeric_map.items():
        if k in model_columns: input_df[k] = float(v)

    # Map One-Hot
    def set_dummy(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns: input_df[col] = 1.0

    set_dummy('quarter', quarter)
    set_dummy('department', dept.lower())
    set_dummy('day', day)
    set_dummy('no_of_style_change', str(style))

    # C. Prediction
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    
    # Alphabetical Order: High (0), Low (1), Moderate (2)
    labels = ['High', 'Low', 'Moderate']
    status = labels[pred_idx]

    # --- 4. SIDEBAR STATUS (FIXED HTML) ---
    st.sidebar.title("Final Status")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    
    # FIXED: Changed unsafe_allow_all_html to unsafe_allow_html
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:white; margin:0;">{status.upper()}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.divider()
    st.sidebar.metric("Scaled Overtime", f"{ot_scaled:.3f}")
    st.sidebar.write(f"Raw OT: {overtime_raw} mins")

    # --- 5. FACTOR ANALYSIS ---
    st.subheader("📊 Comparative Feature Analysis")
    st.info("The table below compares your current inputs against typical levels found in our dataset.")
    
    analysis_data = {
        "Key Metric": ["Complexity (SMV)", "Work-in-Progress", "Incentives", "Labor (Workers)"],
        "Your Current Input": [smv, wip, incentive, workers],
        "Typical Moderate Level": [AVERAGES['Moderate']['smv'], AVERAGES['Moderate']['wip'], AVERAGES['Moderate']['incentive'], AVERAGES['Moderate']['workers']],
        "Typical Low Level": [AVERAGES['Low']['smv'], AVERAGES['Low']['wip'], AVERAGES['Low']['incentive'], AVERAGES['Low']['workers']]
    }
    
    st.table(pd.DataFrame(analysis_data))

    # Automated Consultant Feedback
    st.subheader("💡 Consultant Observations")
    if status != "High":
        if incentive < AVERAGES['High']['incentive']:
            st.warning(f"Note: Your incentive ({incentive}) is significantly lower than the average for 'High' productivity teams ({AVERAGES['High']['incentive']}).")
        if idle_time > 0:
            st.error(f"Note: Idle time of {idle_time} mins detected. Top-performing teams maintain zero downtime.")
    else:
        st.success("Configuration matches 'High Productivity' patterns. Maintain this balance!")
        st.balloons()
