import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIG ---
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")   # PIPELINE (IMPORTANT)
    model_columns = joblib.load("rf_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

# --- 3. UI ---
st.title("🧵 Factory Productivity Predictor")
st.info("ML Pipeline-Based Prediction System")

# --- INPUTS ---
with st.expander("Shift Info", expanded=True):
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])
    dept = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
    quarter = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])
    team = st.number_input("Team", 1, 12, 1)

with st.expander("Workload"):
    workers = st.number_input("Workers", 30)
    smv = st.number_input("SMV", 22.0)
    wip = st.number_input("WIP", 500.0)
    style_change = st.selectbox("Style Change", [0,1,2])

with st.expander("Performance Metrics"):
    incentive = st.number_input("Incentive", 0.0)
    overtime_raw = st.number_input("Overtime (Minutes)", 0)
    idle_time = st.number_input("Idle Time", 0.0)
    idle_men = st.number_input("Idle Men", 0)

# --- PREDICTION ---
if st.button("Run Forecast", use_container_width=True):

    # 1. Build raw input (NO SCALING HERE!)
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    numeric_map = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime_raw  # pipeline handles scaling internally
    }

    for k, v in numeric_map.items():
        if k in input_df.columns:
            input_df[k] = v

    # 2. Categorical encoding
    def set_dummy(prefix, value):
        col = f"{prefix}_{value}"
        if col in input_df.columns:
            input_df[col] = 1

    set_dummy("day", day)
    set_dummy("department", dept.lower())
    set_dummy("quarter", quarter)
    set_dummy("no_of_style_change", str(style_change))

    # 3. Predict
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    labels = ["High", "Low", "Moderate"]
    result = labels[pred]

    # 4. Output
    st.subheader(f"Forecast: {result}")

    confidence = np.max(probs)

    if result == "High":
        st.success(f"Confidence: {confidence:.1%}")
    elif result == "Moderate":
        st.info(f"Confidence: {confidence:.1%}")
    else:
        st.error(f"Confidence: {confidence:.1%}")
