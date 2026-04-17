import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

MODEL_PATH = "rf_model.pkl"
COLUMNS_PATH = "rf_columns.pkl"

CLASS_MAP = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
QUARTERS = ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
DEPARTMENTS = ["finished", "sewing"]
STYLE_CHANGES = [0, 1, 2]

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        st.error("Missing rf_model.pkl or rf_columns.pkl in the app folder.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

pipeline, model_columns = load_assets()


def build_input_row(day, quarter, department, team, smv, wip, incentive,
                    idle_time, idle_men, style_change, workers, over_time_scaled):
    row = pd.DataFrame(0.0, index=[0], columns=model_columns)

    numeric_values = {
        "team": float(team),
        "smv": float(smv),
        "wip": float(wip),
        "incentive": float(incentive),
        "idle_time": float(idle_time),
        "idle_men": float(idle_men),
        "no_of_workers": float(workers),
        "over_time_scaled": float(over_time_scaled),
    }
    for key, value in numeric_values.items():
        if key in row.columns:
            row.at[0, key] = value

    dummy_candidates = [
        f"quarter_{quarter}",
        f"department_{department}",
        f"day_{day}",
        f"no_of_style_change_{style_change}",
    ]
    for col in dummy_candidates:
        if col in row.columns:
            row.at[0, col] = 1.0

    return row[model_columns]


st.title("Garment Productivity Predictor")
st.caption("Prediction app aligned to the cleaned dataset structure used in your Random Forest model.")

st.warning(
    "Important: To match a record from the cleaned dataset exactly, enter the same values from the cleaned file, "
    "including Day and over_time_scaled."
)

with st.form("prediction_form"):
    st.subheader("Enter the same fields used in the cleaned dataset")
    col1, col2, col3 = st.columns(3)

    with col1:
        day = st.selectbox("Day", DAYS)
        quarter = st.selectbox("Quarter", QUARTERS)
        department = st.selectbox("Department", DEPARTMENTS)
        team = st.number_input("Team", min_value=1, max_value=12, value=1, step=1)

    with col2:
        smv = st.number_input("SMV", min_value=0.0, value=22.0, step=0.1)
        wip = st.number_input("WIP", min_value=0.0, value=0.0, step=1.0)
        incentive = st.number_input("Incentive", min_value=0.0, value=0.0, step=1.0)
        no_of_workers = st.number_input("Number of Workers", min_value=1.0, value=30.0, step=1.0)

    with col3:
        idle_time = st.number_input("Idle Time", min_value=0.0, value=0.0, step=1.0)
        idle_men = st.number_input("Idle Men", min_value=0.0, value=0.0, step=1.0)
        no_of_style_change = st.selectbox("Number of Style Change", STYLE_CHANGES)
        over_time_scaled = st.number_input(
            "over_time_scaled",
            value=0.0,
            step=0.01,
            help="Use the exact scaled value from the cleaned dataset if you are validating against an existing record.",
        )

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

if submitted:
    input_df = build_input_row(
        day=day,
        quarter=quarter,
        department=department,
        team=team,
        smv=smv,
        wip=wip,
        incentive=incentive,
        idle_time=idle_time,
        idle_men=idle_men,
        style_change=no_of_style_change,
        workers=no_of_workers,
        over_time_scaled=over_time_scaled,
    )

    pred_code = int(pipeline.predict(input_df)[0])
    class_names = list(getattr(pipeline, "classes_", getattr(pipeline.named_steps.get("model"), "classes_", [0, 1, 2])))
    probs = pipeline.predict_proba(input_df)[0]

    predicted_label = CLASS_MAP.get(pred_code, str(pred_code))

    st.subheader("Prediction Result")
    if predicted_label == "High":
        st.success(f"Predicted Productivity Level: {predicted_label}")
    elif predicted_label == "Moderate":
        st.info(f"Predicted Productivity Level: {predicted_label}")
    else:
        st.error(f"Predicted Productivity Level: {predicted_label}")

    st.subheader("Class Probabilities")
    prob_table = pd.DataFrame({
        "class_code": class_names,
        "productivity_level": [CLASS_MAP.get(int(c), str(c)) for c in class_names],
        "probability": probs,
    }).sort_values("class_code")
    st.dataframe(prob_table, use_container_width=True, hide_index=True)

    st.subheader("Encoded input used by the model")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.markdown(
        "### Why your old app could show the wrong level\n"
        "1. The old app decoded the model output incorrectly using `['High', 'Low', 'Moderate']`. "
        "If your model uses `Low=0, Moderate=1, High=2`, that mapping is wrong.\n"
        "2. The old app did not include the `day` field, even though `day` is part of the trained model features.\n"
        "3. The old app asked for raw overtime and converted it with a custom formula, but your cleaned dataset uses `over_time_scaled` directly."
    )
