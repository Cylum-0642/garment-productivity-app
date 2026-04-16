import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Loading the specific files you generated
    model = joblib.load('rf_garment_model.pkl')
    model_columns = joblib.load('rf_model_columns.pkl')
    # If your model needs the scaler for 'over_time', load it here:
    # scaler = joblib.load('garment_scaler.pkl')
    return model, model_columns

model, rf_model_columns = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Factory Productivity Predictor")
st.markdown("Enter floor data below to forecast if the team will meet their productivity tier.")

# Tracking validation errors
form_errors = []

# --- 4. INPUT SECTIONS (OPTION C: EXPANDERS) ---

# Section A: Operational Context
with st.expander("📅 1. Shift & Timeline Info", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
    with c2:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        team = st.number_input("Team Number", min_value=1, max_value=12, value=1)

# Section B: Workforce & SMV
with st.expander("⚙️ 2. Labor & Complexity", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        # Range based on your dataset summary
        workers = st.number_input("Number of Workers", value=30.0, step=1.0)
        if workers < 2 or workers > 90:
            st.error("⚠️ Industry Range: 2 to 90")
            form_errors.append("Workers")
            
        smv = st.number_input("SMV (Complexity)", value=22.0, help="Standard Minute Value: Work content of the task.")
        if smv < 2.0 or smv > 60.0:
            st.error("⚠️ Industry Range: 2.0 to 60.0")
            form_errors.append("SMV")
            
    with c2:
        wip = st.number_input("Work in Progress (WIP)", value=500.0)
        if wip < 0 or wip > 25000:
            st.error("⚠️ Max Limit: 25,000")
            form_errors.append("WIP")
            
        style_change = st.selectbox("Style Changes", [0, 1, 2])

# Section C: Financials & Downtime
with st.expander("💰 3. Metrics & Downtime", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        incentive = st.number_input("Incentive Amount", value=0)
        if incentive < 0 or incentive > 4000:
            st.error("⚠️ Max Incentive: 4,000")
            form_errors.append("Incentive")
            
        # Your data used 'over_time_scaled'
        overtime = st.slider("Overtime (Scaled Value)", -2.0, 2.0, 0.0, help="Scaled value from your preprocessing.")
        
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0.0)
        if idle_time > 300:
            st.error("⚠️ Max Idle Time: 300")
            form_errors.append("Idle Time")
            
        idle_men = st.number_input("Idle Workers Count", value=0)
        if idle_men > 50:
            st.error("⚠️ Max Idle Workers: 50")
            form_errors.append("Idle Men")

# --- 5. PREDICTION LOGIC ---
st.divider()

if form_errors:
    st.warning("Please correct the validation errors to enable forecasting.")
    st.button("Calculate Productivity Tier", disabled=True)
else:
    if st.button("Calculate Productivity Tier", use_container_width=True, type="primary"):
        # Create empty DataFrame with identical columns to X_train
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # 1. Fill Numeric Values (Names match your model_columns.pkl exactly)
        input_df['team'] = float(team)
        input_df['smv'] = float(smv)
        input_df['wip'] = float(wip)
        input_df['incentive'] = float(incentive)
        input_df['idle_time'] = float(idle_time)
        input_df['idle_men'] = float(idle_men)
        input_df['no_of_workers'] = float(workers)
        input_df['over_time_scaled'] = float(overtime) 

        # 2. Fill Categorical (One-Hot Encoding matching drop_first=True)
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower()) # 'sewing' vs 'finished'
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # 3. Predict
        # Ensure column order is identical to training
        input_df = input_df[model_columns]
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # Mapping back to your 'productivity_level' labels
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        # 4. Industrial Results Dashboard
        st.subheader(f"Forecasted Status: {result}")
        
        if result == 'High':
            st.success(f"Confidence: {probs[2]:.1%} - Team is highly likely to exceed targets.")
            st.balloons()
        elif result == 'Moderate':
            st.info(f"Confidence: {probs[1]:.1%} - Team is on track for standard targets.")
        else:
            st.error(f"Confidence: {probs[0]:.1%} - High risk of failing to meet production goals.")
