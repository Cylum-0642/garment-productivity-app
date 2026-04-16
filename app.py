import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Using your NEW filenames
    model_file = 'model.pkl'
    cols_file = 'columns.pkl'
    
    if not os.path.exists(model_file) or not os.path.exists(cols_file):
        st.error("❌ Required files (model.pkl or columns.pkl) are missing from GitHub!")
        st.stop()
        
    pipeline = joblib.load(model_file)
    model_columns = joblib.load(cols_file)
    return pipeline, model_columns

pipeline, model_columns = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Factory Productivity Predictor")
st.markdown("This tool uses a Random Forest Pipeline to forecast production tiers.")

# --- 4. INPUT SECTIONS ---
with st.expander("📅 1. Operational Context", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
    with c2:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        team = st.number_input("Team Number", 1, 12, 1)

with st.expander("⚙️ 2. Labor & Complexity", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        workers = st.number_input("Number of Workers", value=30.0)
        smv = st.number_input("SMV (Complexity Content)", value=22.0)
    with c2:
        wip = st.number_input("Work in Progress (WIP)", value=500.0)
        style_change = st.selectbox("Style Changes", [0, 1, 2])

with st.expander("💰 3. Performance & Overtime", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        incentive = st.number_input("Incentive Amount", value=0)
        # We take the pre-scaled value as per your cleaned dataset strategy
        ot_scaled = st.number_input("Overtime (Already Scaled)", value=0.0, format="%.4f")
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0.0)
        idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True, type="primary"):
    try:
        # Create a DataFrame with all zeros matching the model's expected 20 columns
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # 1. Map Numeric Columns (The pipeline handles scaling for these)
        input_df['team'] = float(team)
        input_df['smv'] = float(smv)
        input_df['wip'] = float(wip)
        input_df['incentive'] = float(incentive)
        input_df['idle_time'] = float(idle_time)
        input_df['idle_men'] = float(idle_men)
        input_df['no_of_workers'] = float(workers)
        input_df['over_time_scaled'] = float(ot_scaled)

        # 2. Map Categorical Columns (One-Hot Encoding matching your columns.pkl)
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower()) # Maps to 'department_sewing' or baseline
        set_dummy('day', day)
        set_dummy('no_of_style_change', str(style_change)) # Matches 'no_of_style_change_1', etc.

        # 3. Ensure column order is EXACTLY as trained
        input_df = input_df[model_columns]
        
        # 4. Predict using the Pipeline (Pipeline.predict handles internal scaling)
        prediction = pipeline.predict(input_df)[0]
        probs = pipeline.predict_proba(input_df)[0]
        
        # Mapping Result (Alphabetical Order: High=0, Low=1, Moderate=2)
        labels = ['High', 'Low', 'Moderate']
        result = labels[prediction]
        
        # 5. Result Display
        st.subheader(f"Prediction: {result}")
        st.write(f"Model Confidence: {probs[prediction]:.1%}")
        
        if result == 'High':
            st.success("Target likely to be exceeded. Productivity is optimized.")
            st.balloons()
        elif result == 'Moderate':
            st.info("Target likely to be met with standard efficiency.")
        else:
            st.error("Warning: High probability of productivity shortfall.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure that 'columns.pkl' and 'model.pkl' were created using the same dataset structure.")
