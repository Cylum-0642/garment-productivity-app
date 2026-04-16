import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Using your NEW filenames
    model = joblib.load('rf_garment_model.pkl')
    model_columns = joblib.load('rf_model_columns.pkl')
    scaler = joblib.load('garment_scaler.pkl') 
    return model, model_columns, scaler

model, model_columns, scaler = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Factory Productivity Predictor")
st.info("Industrial Prototype: Random Forest Model (Tuned)")

form_errors = []

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
        overtime_raw = st.number_input("Overtime (Minutes)", value=0)
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0.0)
        idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("Run Forecast", use_container_width=True, type="primary"):
    # 1. Create Input DataFrame for the Scaler
    # Your scaler expects these 7 columns in this exact order:
    scaler_input = pd.DataFrame([[
        team, smv, wip, incentive, idle_time, idle_men, workers
    ]], columns=['team', 'smv', 'wip', 'incentive', 'idle_time', 'idle_men', 'no_of_workers'])
    
    # Scale the numeric features
    scaled_values = scaler.transform(scaler_input)
    
    # 2. Create Final DataFrame for the Model
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # Fill scaled numeric values into the model input
    input_df['team'] = scaled_values[0][0]
    input_df['smv'] = scaled_values[0][1]
    input_df['wip'] = scaled_values[0][2]
    input_df['incentive'] = scaled_values[0][3]
    input_df['idle_time'] = scaled_values[0][4]
    input_df['idle_men'] = scaled_values[0][5]
    input_df['no_of_workers'] = scaled_values[0][6]
    
    # Manual Scaling for Overtime (Since your scaler didn't include it in this pkl)
    # If your over_time_scaled in the dataset was (x - mean)/std, we use a simple placeholder or 0
    input_df['over_time_scaled'] = 0.0 

    # 3. Handle Categoricals
    def set_dummy(category, value):
        col_name = f"{category}_{value}"
        if col_name in model_columns:
            input_df[col_name] = 1.0

    set_dummy('quarter', quarter)
    set_dummy('department', dept.lower())
    set_dummy('day', day)
    set_dummy('no_of_style_change', style_change)

    # 4. Predict
    input_df = input_df[model_columns] 
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    # Display Results
    labels = ['High', 'Low', 'Moderate'] 
    result = labels[pred]
    
    st.subheader(f"Forecast: {result}")
    if result == 'High':
        st.success(f"Confidence: {probs[0]:.1%}")
    elif result == 'Moderate':
        st.info(f"Confidence: {probs[2]:.1%}")
    else:
        st.error(f"Confidence: {probs[1]:.1%}")
