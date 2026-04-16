import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- 2. ASSET LOADING WITH ERROR CHECKING ---
@st.cache_resource
def load_assets():
    files = {
        "model": 'rf_garment_model.pkl',
        "cols": 'rf_model_columns.pkl',
        "scaler": 'garment_scaler.pkl'
    }
    
    # Check if files exist to prevent the silent error
    for key, name in files.items():
        if not os.path.exists(name):
            st.error(f"❌ File missing from GitHub: **{name}**")
            st.stop()
            
    model = joblib.load(files["model"])
    model_columns = joblib.load(files["cols"])
    scaler = joblib.load(files["scaler"])
    return model, model_columns, scaler

model, model_columns, scaler = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Factory Productivity Predictor")
st.info("Industrial Prototype: Random Forest Model (Tuned)")

# --- 4. INPUT SECTIONS ---
with st.expander("📅 1. Shift & Timeline Info", expanded=True):
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
        smv = st.number_input("SMV (Complexity)", value=22.0)
    with c2:
        wip = st.number_input("Work in Progress (WIP)", value=500.0)
        style_change = st.selectbox("Style Changes", [0, 1, 2])

with st.expander("💰 3. Metrics & Downtime", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        incentive = st.number_input("Incentive Amount", value=0)
        # Note: We will use 0 for overtime_scaled as a baseline if not provided
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0.0)
        idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("Run Forecast", use_container_width=True, type="primary"):
    try:
        # Step A: Scale the 7 numeric features exactly as the scaler expects
        # Order: team, smv, wip, incentive, idle_time, idle_men, no_of_workers
        numeric_data = pd.DataFrame([[
            float(team), float(smv), float(wip), float(incentive), 
            float(idle_time), float(idle_men), float(workers)
        ]], columns=['team', 'smv', 'wip', 'incentive', 'idle_time', 'idle_men', 'no_of_workers'])
        
        scaled_features = scaler.transform(numeric_data)
        
        # Step B: Create the full feature set (20 columns)
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # Mapping scaled values
        input_df['team'] = scaled_features[0][0]
        input_df['smv'] = scaled_features[0][1]
        input_df['wip'] = scaled_features[0][2]
        input_df['incentive'] = scaled_features[0][3]
        input_df['idle_time'] = scaled_features[0][4]
        input_df['idle_men'] = scaled_features[0][5]
        input_df['no_of_workers'] = scaled_features[0][6]
        
        # Set baseline for overtime (0 is the mean for scaled data)
        if 'over_time_scaled' in input_df.columns:
            input_df['over_time_scaled'] = 0.0

        # Step C: One-Hot Encoding
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', str(style_change)) # Convert to string for matching

        # Step D: Final Prediction
        input_df = input_df[model_columns]
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # Results Display
        labels = ['High', 'Low', 'Moderate'] # Alphabetical order 0,1,2
        result = labels[prediction]
        
        st.subheader(f"Forecast: {result}")
        if result == 'High':
            st.success(f"Confidence: {probs[0]:.1%}")
            st.balloons()
        elif result == 'Moderate':
            st.info(f"Confidence: {probs[2]:.1%}")
        else:
            st.error(f"Confidence: {probs[1]:.1%}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
