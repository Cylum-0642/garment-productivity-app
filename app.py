import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment AI: Explainable Mode", layout="wide")

# Dataset Averages (Calculated from your cleaned dataset)
# Used to show the user how their inputs "lean" towards different levels
AVERAGES = {
    'High':     {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1, 'idle': 0.0},
    'Moderate': {'smv': 16.7, 'wip': 682.5, 'incentive': 34.1, 'workers': 37.8, 'idle': 0.9},
    'Low':      {'smv': 15.5, 'wip': 478.0, 'incentive': 15.1, 'workers': 32.5, 'idle': 2.3}
}

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Missing model files.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- 2. INPUT SECTION ---
st.title("🧵 Intelligent Production Analysis")
st.markdown("This prototype explains **why** the model chooses a specific productivity tier.")

with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Shift Context")
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finished"])
    
    with c2:
        st.subheader("Complexity & Load")
        smv = st.number_input("SMV (Complexity)", 2.0, 60.0, 22.0)
        wip = st.number_input("WIP (Workload)", 0.0, 25000.0, 500.0)
        workers = st.number_input("Workers", 1.0, 100.0, 30.0)
        style = st.selectbox("Style Changes", [0, 1, 2])

    with c3:
        st.subheader("Performance")
        incentive = st.number_input("Incentive Amount", 0, 1000, 0)
        overtime_raw = st.number_input("Overtime (Mins)", 0, 5000, 0)
        idle_time = st.number_input("Idle Time (Mins)", 0.0, 300.0, 0.0)
        idle_men = st.number_input("Idle Men", 0, 50, 0)

    submit = st.form_submit_button("Analyze Productivity", use_container_width=True)

# --- 3. LOGIC & PREDICTION ---
if submit:
    # A. Custom Scaling for Overtime (Modified Z-Score)
    # Using Median and MAD from your cleaned dataset
    ot_scaled = (overtime_raw - 0.0) / (2520.0 * 1.4826) if overtime_raw > 0 else -0.5

    # B. Build Feature Vector
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    input_df['team'] = 1.0 # Default/Placeholder
    input_df['smv'] = float(smv)
    input_df['wip'] = float(wip)
    input_df['incentive'] = float(incentive)
    input_df['idle_time'] = float(idle_time)
    input_df['idle_men'] = float(idle_men)
    input_df['no_of_workers'] = float(workers)
    input_df['over_time_scaled'] = float(ot_scaled)

    def set_cat(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns: input_df[col] = 1.0
    set_cat('quarter', quarter); set_cat('department', dept.lower()); set_cat('day', day); set_cat('no_of_style_change', str(style))

    # C. Run Model
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    labels = ['High', 'Low', 'Moderate']
    status = labels[pred_idx]

    # --- 4. DISPLAY RESULTS ---
    
    # ONE WORD STATUS IN SIDEBAR
    st.sidebar.title("Status")
    color = "green" if status == "High" else "orange" if status == "Moderate" else "red"
    st.sidebar.markdown(f"<h1 style='color:{color};'>{status.upper()}</h1>", unsafe_allow_all_html=True)
    st.sidebar.metric("Scaled Overtime", round(ot_scaled, 3))
    
    # FACTOR ANALYSIS CHART
    st.subheader("📊 Feature Influence Analysis")
    st.write("How your inputs compare to typical **Moderate** and **Low** production environments:")

    # Simple comparison logic: Percent of the Moderate Average
    comparison_data = {
        "Metric": ["Complexity (SMV)", "Incentives", "Workforce Size", "WIP Level"],
        "Your Value": [smv, incentive, workers, wip],
        "Moderate Avg": [AVERAGES['Moderate']['smv'], AVERAGES['Moderate']['incentive'], AVERAGES['Moderate']['workers'], AVERAGES['Moderate']['wip']],
        "Low Avg": [AVERAGES['Low']['smv'], AVERAGES['Low']['incentive'], AVERAGES['Low']['workers'], AVERAGES['Low']['wip']]
    }
    comp_df = pd.DataFrame(comparison_data)
    st.table(comp_df)

    # Explanation text
    if status == 'Moderate' and incentive < AVERAGES['High']['incentive']:
        st.warning(f"💡 **Improvement Tip:** Your incentive ({incentive}) is below the 'High Productivity' average of {AVERAGES['High']['incentive']}. Increasing this might shift the status.")
    
    if idle_time > 0:
        st.error(f"🚨 **Downtime Alert:** Idle time detected. In 'High' productivity teams, idle time is usually 0.0.")
