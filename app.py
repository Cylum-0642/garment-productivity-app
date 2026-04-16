import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # FILENAMES: Ensure these match exactly what you upload to GitHub!
    model = joblib.load('rf_garment_model.pkl')
    model_columns = joblib.load('rf_model_columns.pkl')
    scaler = joblib.load('garment_scaler.pkl') # For scaling over_time
    return model, model_columns, scaler

model, model_columns, scaler = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Factory Productivity Predictor")
st.info("Industrial Prototype: Random Forest Model (Tuned)")

# Validation flag
form_errors = []

# --- 4. INPUT SECTIONS ---

# Section A: Operational Context
with st.expander("📅 1. Shift & Timeline Info", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
    with c2:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        team = st.number_input("Team Number", 1, 12, 1)

# Section B: Labor & Complexity
with st.expander("⚙️ 2. Labor & Complexity", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        workers = st.number_input("Number of Workers", value=30.0)
        if workers < 2 or workers > 90:
            st.error("⚠️ Range: 2 to 90")
            form_errors.append("Workers")
            
        smv = st.number_input("SMV (Complexity)", value=22.0)
        if smv < 2.0 or smv > 60.0:
            st.error("⚠️ Range: 2.0 to 60.0")
            form_errors.append("SMV")
            
    with c2:
        wip = st.number_input("Work in Progress (WIP)", value=500.0)
        style_change = st.selectbox("Style Changes", [0, 1, 2])

# Section C: Financials & Downtime
with st.expander("💰 3. Metrics & Downtime", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        incentive = st.number_input("Incentive Amount", value=0)
        # Taking RAW minutes now, consistent with industry use
        overtime_raw = st.number_input("Overtime (Minutes)", value=0)
        
    with c2:
        idle_time = st.number_input("Idle Time (Mins)", value=0.0)
        idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if form_errors:
    st.warning("Please fix inputs to enable prediction.")
else:
    if st.button("Run Forecast", use_container_width=True, type="primary"):
        # 1. Create DataFrame matching training columns
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # 2. Assign Numeric Values
        input_df['team'] = float(team)
        input_df['smv'] = float(smv)
        input_df['wip'] = float(wip)
        input_df['incentive'] = float(incentive)
        input_df['idle_time'] = float(idle_time)
        input_df['idle_men'] = float(idle_men)
        input_df['no_of_workers'] = float(workers)
        
        # 3. Scale Overtime using your saved scaler
        # We wrap it in a dataframe because that's what the scaler expects
        ot_scaled = scaler.transform(pd.DataFrame({'over_time': [overtime_raw]}))
        input_df['over_time_scaled'] = ot_scaled[0][0]

        # 4. Handle Categoricals (One-Hot Encoding)
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # 5. Predict
        input_df = input_df[model_columns] # Final alignment
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # NOTE: Verify your LabelEncoder order! 
        # Standard alphabetical for (High, Low, Moderate) is [0, 1, 2]
        labels = ['High', 'Low', 'Moderate'] 
        result = labels[pred]
        
        # 6. Results UI
        st.subheader(f"Forecast: {result}")
        if result == 'High':
            st.success(f"Confidence: {probs[0]:.1%}") # Index 0 for High
            st.balloons()
        elif result == 'Moderate':
            st.info(f"Confidence: {probs[2]:.1%}") # Index 2 for Moderate
        else:
            st.error(f"Confidence: {probs[1]:.1%}") # Index 1 for Low
