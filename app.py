import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment AI Prototype", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # FILENAMES must match your GitHub exactly
    model_file = 'rf_garment_model.pkl'
    cols_file = 'rf_model_columns.pkl'
    
    if not os.path.exists(model_file) or not os.path.exists(cols_file):
        st.error("Missing .pkl files in the repository!")
        st.stop()
        
    model = joblib.load(model_file)
    model_columns = joblib.load(cols_file)
    return model, model_columns

model, model_columns = load_assets()

# --- 3. UI ---
st.title("🧵 Productivity Forecast Tool")
st.markdown("This prototype uses **Pre-Scaled Overtime** values as per the cleaned dataset.")

with st.expander("📝 Input Production Data", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.selectbox("Department", ["Sewing", "Finished"])
        team = st.number_input("Team Number", 1, 12, 1)
        smv = st.number_input("SMV", value=22.0)
        workers = st.number_input("No. of Workers", value=30.0)

    with col2:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        wip = st.number_input("WIP", value=500.0)
        incentive = st.number_input("Incentive", value=0)
        # USER ENTERS THE SCALED VALUE DIRECTLY AS PER YOUR PLAN
        ot_scaled = st.number_input("Overtime (Scaled Value)", value=0.0, format="%.4f")
        style_change = st.selectbox("Style Changes", [0, 1, 2])

# --- 4. PREDICTION LOGIC ---
if st.button("Generate Prediction", use_container_width=True, type="primary"):
    # Initialize 20-column dataframe with zeros
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # 1. Map Numeric Values
    input_df['team'] = float(team)
    input_df['smv'] = float(smv)
    input_df['wip'] = float(wip)
    input_df['incentive'] = float(incentive)
    input_df['no_of_workers'] = float(workers)
    input_df['over_time_scaled'] = float(ot_scaled)
    # The columns idle_time and idle_men are in your model_columns.pkl, 
    # so we keep them as 0.0 unless you add inputs for them.

    # 2. Map Categoricals (One-Hot Encoding)
    # Note: drop_first=True means Monday, Quarter1, and Finished are '0' for all other columns
    def set_dummy(category, value):
        col_name = f"{category}_{value}"
        if col_name in model_columns:
            input_df[col_name] = 1.0

    set_dummy('quarter', quarter)
    set_dummy('department', dept.lower()) # Matches 'department_sewing'
    set_dummy('day', day)
    set_dummy('no_of_style_change', style_change)

    # 3. Final Prediction
    # Ensure column order is identical to training
    input_df = input_df[model_columns]
    
    try:
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        # Mapping: 0:High, 1:Low, 2:Moderate (Standard Alphabetical)
        labels = ['High', 'Low', 'Moderate']
        result = labels[prediction]

        st.divider()
        st.subheader(f"Result: {result}")
        st.progress(float(probs[prediction]))
        st.write(f"Confidence: {probs[prediction]:.1%}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
