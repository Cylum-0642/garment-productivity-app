import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Note: Ensure these filenames match what you uploaded to GitHub!
    model = joblib.load('garment_rf_model.pkl')
    model_columns = joblib.load('rf_model_columns.pkl')
    # If using scaler: scaler = joblib.load('garment_scaler.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- HEADER ---
st.title("🧵 Garment Productivity Predictor")
st.markdown("---")

# Global validation flag
form_errors = []

# --- SECTION 1: CORE OPERATIONAL DATA ---
with st.expander("📅 1. Shift & Timeline Info", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
    with c2:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        team = st.slider("Team Number", 1, 12, 1)

# --- SECTION 2: LABOR & COMPLEXITY ---
with st.expander("⚙️ 2. Labor & Complexity Details", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        workers = st.number_input("Number of Workers", value=30, help="Total workforce assigned to the line.")
        if workers < 2 or workers > 90:
            st.error("⚠️ Range: 2 to 90 workers")
            form_errors.append("Workers")
            
        smv = st.number_input("SMV (Complexity)", value=22.0, help="Standard Minute Value: Higher means more complex garment.")
        if smv < 2.0 or smv > 60.0:
            st.error("⚠️ Range: 2.0 to 60.0")
            form_errors.append("SMV")
            
    with c2:
        wip = st.number_input("Work in Progress (WIP)", value=500)
        if wip < 0 or wip > 25000:
            st.error("⚠️ Range: 0 to 25,000")
            form_errors.append("WIP")
            
        style_change = st.selectbox("Style Changes", ["0", "1", "2"])

# --- SECTION 3: INCENTIVES & DOWNTIME ---
with st.expander("💰 3. Incentives & Downtime", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        incentive = st.number_input("Incentive Amount ($)", value=0)
        if incentive < 0 or incentive > 4000:
            st.error("⚠️ Max allowed: $4,000")
            form_errors.append("Incentive")
            
        overtime = st.slider("Overtime Level (Scaled)", -2.0, 2.0, 0.0)
        
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0)
        if idle_time > 300:
            st.error("⚠️ Max allowed: 300 mins")
            form_errors.append("Idle Time")
            
        idle_men = st.number_input("Idle Workers Count", value=0)
        if idle_men > 50:
            st.error("⚠️ Max allowed: 50 workers")
            form_errors.append("Idle Men")

# --- PREDICTION ENGINE ---
st.markdown("---")

if form_errors:
    st.warning(f"Please fix the inputs for: {', '.join(form_errors)}")
    st.button("Run Productivity Forecast", disabled=True)
else:
    if st.button("Run Productivity Forecast", use_container_width=True, type="primary"):
        with st.spinner('Analyzing floor data...'):
            # 1. Prepare Data
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # 2. Map Numeric Inputs
            input_df['team'] = team
            input_df['smv'] = smv
            input_df['wip'] = wip
            input_df['incentive'] = incentive
            input_df['idle_time'] = idle_time
            input_df['idle_men'] = idle_men
            input_df['no_of_workers'] = workers
            input_df['over_time_scaled'] = overtime 
            
            # 3. Map Categorical (One-Hot Encoding)
            def set_dummy(category, value):
                col_name = f"{category}_{value}"
                if col_name in model_columns:
                    input_df[col_name] = 1

            set_dummy('quarter', quarter)
            set_dummy('department', dept.lower())
            set_dummy('day', day)
            set_dummy('no_of_style_change', style_change)

            # 4. Predict
            input_df = input_df[model_columns]
            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            
            # 5. Display Results
            labels = ['Low', 'Moderate', 'High']
            result = labels[pred]
            
            st.subheader(f"Forecasted Result: {result}")
            
            if result == 'High':
                st.success(f"Confidence: {probs[2]:.1%} - Production is optimized.")
                st.balloons()
            elif result == 'Moderate':
                st.info(f"Confidence: {probs[1]:.1%} - Target met with standard efficiency.")
            else:
                st.error(f"Confidence: {probs[0]:.1%} - Warning: Significant productivity shortfall expected.")
