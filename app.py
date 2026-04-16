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
    # FILENAMES: Ensure these match your GitHub exactly
    # Based on your latest update:
    model_file = 'rf_garment_model.pkl'
    cols_file = 'rf_model_columns.pkl'
    scaler_file = 'garment_scaler.pkl' 
    
    # Safety check for file existence
    for f in [model_file, cols_file, scaler_file]:
        if not os.path.exists(f):
            st.error(f"❌ File not found in GitHub: {f}")
            st.stop()
            
    model = joblib.load(model_file)
    model_columns = joblib.load(cols_file)
    scaler = joblib.load(scaler_file)
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
        # Using the scaled value as per your cleaned dataset plan
        ot_scaled = st.slider("Overtime (Scaled Value)", -2.0, 2.0, 0.0)
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0.0)
        idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("Run Forecast", use_container_width=True, type="primary"):
    try:
        # STEP A: Scale the 7 features the scaler expects
        # The order must match: team, smv, wip, incentive, idle_time, idle_men, no_of_workers
        numeric_data = pd.DataFrame([[
            float(team), float(smv), float(wip), float(incentive), 
            float(idle_time), float(idle_men), float(workers)
        ]], columns=['team', 'smv', 'wip', 'incentive', 'idle_time', 'idle_men', 'no_of_workers'])
        
        scaled_numeric = scaler.transform(numeric_data)
        
        # STEP B: Prepare the full 20-column DataFrame for the model
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # Mapping scaled values to the model input
        input_df['team'] = scaled_numeric[0][0]
        input_df['smv'] = scaled_numeric[0][1]
        input_df['wip'] = scaled_numeric[0][2]
        input_df['incentive'] = scaled_numeric[0][3]
        input_df['idle_time'] = scaled_numeric[0][4]
        input_df['idle_men'] = scaled_numeric[0][5]
        input_df['no_of_workers'] = scaled_numeric[0][6]
        
        # Handle overtime_scaled (entered directly by user)
        input_df['over_time_scaled'] = float(ot_scaled)

        # STEP C: Categorical Encoding (One-Hot)
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', str(style_change))

        # STEP D: Final Prediction
        input_df = input_df[model_columns]
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # Results Mapping (Alphabetical: High=0, Low=1, Moderate=2)
        labels = ['High', 'Low', 'Moderate']
        result = labels[prediction]
        
        st.subheader(f"Forecasted Result: {result}")
        st.write(f"Confidence: {probs[prediction]:.1%}")
        
        if result == 'High':
            st.success("Target likely to be exceeded.")
            st.balloons()
        elif result == 'Moderate':
            st.info("Target likely to be met.")
        else:
            st.error("High risk of productivity shortfall.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
