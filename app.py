import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(
    page_title="Garment Productivity AI", 
    page_icon="🧵", 
    layout="centered"
)

@st.cache_resource
def load_production_assets():
    # Ensuring we use your new renamed files
    model_path = 'model.pkl'
    cols_path = 'columns.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        st.error("⚠️ Critical Error: Model or Column files not found on GitHub.")
        st.stop()
        
    pipeline = joblib.load(model_path)
    model_columns = joblib.load(cols_path)
    return pipeline, model_columns

pipeline, model_columns = load_production_assets()

# --- 2. HEADER ---
st.title("🧵 Factory Productivity Predictor")
st.markdown("""
    This AI tool predicts the **Productivity Level** of a garment production team 
    based on operational metrics and labor complexity.
""")
st.divider()

# --- 3. INPUT FORM ---
with st.form("factory_data_form"):
    st.subheader("📋 Operational Context")
    c1, c2 = st.columns(2)
    
    with c1:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        department = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
    
    with c2:
        team = st.number_input("Team Number", min_value=1, max_value=12, value=1)
        style_change = st.selectbox("Number of Style Changes", [0, 1, 2])
        workers = st.number_input("Number of Workers", min_value=1.0, value=30.0)

    st.subheader("⚙️ Production Metrics")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        smv = st.number_input("SMV (Complexity)", value=20.0, help="Standard Minute Value")
        wip = st.number_input("WIP", value=500.0, help="Work In Progress")
    
    with m2:
        incentive = st.number_input("Incentive", value=0)
        ot_scaled = st.number_input("Overtime (Scaled)", value=0.0, format="%.4f")
        
    with m3:
        idle_time = st.number_input("Idle Time", value=0.0)
        idle_men = st.number_input("Idle Men", value=0)

    # Submission button
    submit_btn = st.form_submit_button("Generate Forecast", use_container_width=True, type="primary")

# --- 4. PREDICTION LOGIC ---
if submit_btn:
    try:
        # Create a single-row DataFrame with zeros matching training columns
        input_row = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # A. Fill Numeric Values
        input_row['team'] = float(team)
        input_row['smv'] = float(smv)
        input_row['wip'] = float(wip)
        input_row['incentive'] = float(incentive)
        input_row['idle_time'] = float(idle_time)
        input_row['idle_men'] = float(idle_men)
        input_row['no_of_workers'] = float(workers)
        input_row['over_time_scaled'] = float(ot_scaled)

        # B. Map Categorical Dummies (One-Hot Encoding)
        def apply_one_hot(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_row[col_name] = 1.0

        apply_one_hot('quarter', quarter)
        apply_one_hot('department', department.lower())
        apply_one_hot('day', day)
        apply_one_hot('no_of_style_change', str(style_change))

        # C. Predict using Pipeline (Handles internal scaling if included)
        # Ensure column order matches exactly
        final_input = input_row[model_columns]
        
        prediction = pipeline.predict(final_input)[0]
        probabilities = pipeline.predict_proba(final_input)[0]

        # Results Mapping
        # Note: Scikit-learn orders labels alphabetically [High, Low, Moderate]
        labels = ['High', 'Low', 'Moderate']
        result_label = labels[prediction]
        confidence = probabilities[prediction]

        # --- 5. RESULT DISPLAY ---
        st.divider()
        st.subheader(f"Predicted Productivity: {result_label}")
        
        # Color-coded metric
        if result_label == "High":
            st.success(f"The team is performing optimally with {confidence:.1%} confidence.")
            st.balloons()
        elif result_label == "Moderate":
            st.info(f"The team is meeting standard targets with {confidence:.1%} confidence.")
        else:
            st.error(f"High risk of productivity shortfall. Confidence: {confidence:.1%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure your input features match the model training requirements.")
