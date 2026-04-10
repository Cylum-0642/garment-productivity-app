import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('garment_xgb_model.pkl')
    model_columns = joblib.load('xgb_model_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# --- UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.info("""**Model Info:** Currently using a **Tuned XGBoost Classifier** (Optimized via RandomizedSearchCV). 
This model evaluates non-linear relationships between floor operations and labor efficiency.""")

# We use this to track if ANY input is currently invalid
form_is_invalid = False

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finished"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resource Allocation")
    
    # WIP Validation
    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error(f"⚠️ Limit Exceeded: Max WIP is 23,122")
        form_is_invalid = True
        
    # Workers Validation
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error(f"⚠️ Limit Exceeded: Range is 2 to 90")
        form_is_invalid = True

    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])
    
    # SMV Validation
    smv = st.number_input("SMV (Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error(f"⚠️ Limit Exceeded: Range is 2.9 to 54.6")
        form_is_invalid = True

with col3:
    st.subheader("💰 Incentives & Metrics")
    
    # Incentive Validation
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error(f"⚠️ Limit Exceeded: Max Incentive is 3,600")
        form_is_invalid = True
        
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)
    
    # Idle Time Validation
    idle_time = st.number_input("Idle Time (Mins)", value=0)
    if idle_time > 300:
        st.error(f"⚠️ Limit Exceeded: Max Idle Time is 300")
        form_is_invalid = True
        
    # Idle Men Validation
    idle_men = st.number_input("Idle Workers Count", value=0)
    if idle_men > 45:
        st.error(f"⚠️ Limit Exceeded: Max Idle Workers is 45")
        form_is_invalid = True

# --- PREDICTION LOGIC ---
st.divider()

# Disable the button if the form is invalid
if form_is_invalid:
    st.warning("Please correct the errors above to enable the prediction.")
    st.button("Generate Productivity Forecast", disabled=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # Initialize DataFrame
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Mapping Values
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime 
        
        # Encoding
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # Align & Predict
        input_df = input_df[model_columns]
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        st.markdown(f"## Predicted Tier: **{result}**")
        if result == 'High':
            st.success(f"Confidence: {probs[2]:.2%} - Optimized production detected.")
            st.balloons()
        elif result == 'Moderate':
            st.warning(f"Confidence: {probs[1]:.2%} - Operating within standard range.")
        else:
            st.error(f"Confidence: {probs[0]:.2%} - High risk of target shortfall.")
