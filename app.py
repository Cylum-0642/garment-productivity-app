import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment AI Predictor", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # FILENAMES: Ensure these match your new names on GitHub
    model_file = 'model.pkl'
    cols_file = 'columns.pkl'
    
    if not os.path.exists(model_file) or not os.path.exists(cols_file):
        st.error(f"❌ Missing files on GitHub! Need {model_file} and {cols_file}")
        st.stop()
        
    pipeline = joblib.load(model_file)
    model_columns = joblib.load(cols_file)
    return pipeline, model_columns

pipeline, model_columns = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Factory Productivity Forecast")
st.markdown("Enter production metrics to predict the productivity tier.")

# --- 4. INPUT SECTIONS ---
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Context")
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        department = st.radio("Department", ["Sewing", "Finished"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        team = st.number_input("Team Number", 1, 12, 1)
        
    with c2:
        st.subheader("Metrics")
        smv = st.number_input("SMV", value=20.0)
        wip = st.number_input("WIP", value=500.0)
        incentive = st.number_input("Incentive", value=0)
        ot_scaled = st.number_input("Overtime (Scaled)", value=0.0, format="%.4f")

    st.subheader("Workforce")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        workers = st.number_input("No. of Workers", value=30.0)
    with cc2:
        idle_time = st.number_input("Idle Time", value=0.0)
    with cc3:
        idle_men = st.number_input("Idle Men", value=0)
        
    style_change = st.selectbox("Style Changes", [0, 1, 2])
    
    submit = st.form_submit_button("Predict Productivity Tier", use_container_width=True)

# --- 5. PREDICTION LOGIC ---
if submit:
    try:
        # 1. Create a blank DataFrame with all 20 columns from your columns.pkl
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # 2. Map Numerics
        input_df['team'] = float(team)
        input_df['smv'] = float(smv)
        input_df['wip'] = float(wip)
        input_df['incentive'] = float(incentive)
        input_df['idle_time'] = float(idle_time)
        input_df['idle_men'] = float(idle_men)
        input_df['no_of_workers'] = float(workers)
        input_df['over_time_scaled'] = float(ot_scaled)

        # 3. Map One-Hot Categoricals
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns:
                input_df[col_name] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', department.lower()) # 'sewing' vs 'finished'
        set_dummy('day', day)
        set_dummy('no_of_style_change', str(style_change))

        # 4. Final Alignment & Prediction
        input_df = input_df[model_columns]
        prediction = pipeline.predict(input_df)[0]
        probs = pipeline.predict_proba(input_df)[0]

        # 5. Result Display
        # Labels mapping (Alphabetical: High=0, Low=1, Moderate=2)
        labels = ['High', 'Low', 'Moderate']
        result = labels[prediction]
        
        st.divider()
        st.header(f"Forecast: {result}")
        st.write(f"Confidence Level: {probs[prediction]:.1%}")
        
        if result == "High":
            st.success("The team is expected to exceed productivity targets.")
            st.balloons()
        elif result == "Moderate":
            st.info("The team is on track for standard targets.")
        else:
            st.error("Warning: Team is likely to underperform.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
