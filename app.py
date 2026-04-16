import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Garment Productivity AI", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    model_columns = joblib.load("rf_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

# =========================
# UI
# =========================
st.title("🧵 Factory Productivity Predictor")

with st.form("input_form"):
    day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Saturday","Sunday"])
    dept = st.selectbox("Department", ["sewing", "finished"])
    quarter = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4","Quarter5"])
    style_change = st.selectbox("Style Change", [0,1,2])

    team = st.number_input("Team", 1, 12, 1)
    smv = st.number_input("SMV", 1.0)
    wip = st.number_input("WIP", 0.0)
    incentive = st.number_input("Incentive", 0.0)
    idle_time = st.number_input("Idle Time", 0.0)
    idle_men = st.number_input("Idle Men", 0)
    workers = st.number_input("No of Workers", 1)
    overtime_scaled = st.number_input("Overtime (Scaled)", 0.0)

    submit = st.form_submit_button("Predict")

# =========================
# PREDICTION
# =========================
if submit:

    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # numeric features
    numeric_map = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime_scaled
    }

    for k, v in numeric_map.items():
        if k in input_df.columns:
            input_df[k] = v

    # categorical encoding
    def set_dummy(prefix, value):
        col = f"{prefix}_{value}"
        if col in input_df.columns:
            input_df[col] = 1

    set_dummy("day", day)
    set_dummy("department", dept)
    set_dummy("quarter", quarter)
    set_dummy("no_of_style_change", str(style_change))

    # ensure exact alignment
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # prediction
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    labels = ["High", "Low", "Moderate"]
    result = labels[pred]

    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {np.max(probs):.2%}")
