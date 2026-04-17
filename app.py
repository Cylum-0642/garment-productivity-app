import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Garment AI Consultant", layout="wide")

LABELS = {
    "smv": "Task Complexity (SMV)",
    "wip": "Current Workload (WIP)",
    "no_of_workers": "Number of Workers",
    "idle_time": "Idle Time (Minutes)",
    "idle_men": "Idle Workers",
    "incentive": "Incentive Amount (Bonus)",
    "over_time": "Overtime (Minutes)",
    "no_of_style_change": "Number of Style Changes"
}

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- TITLE ---
st.title("🧵 Intelligent Production Consultant")
st.caption("🚀 All features from the cleaned dataset are included to ensure prediction accuracy.")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Production Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        # ADDED: Day selection because it exists in your cleaned dataset/model columns
        day = st.selectbox("Day of the Week", ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"])
        team_num = st.selectbox("Team Number", list(range(1, 13)))

    with col2:
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0)
        workers = st.number_input(LABELS["no_of_workers"], 1.0, 100.0, 30.0)
        incentive = st.number_input(LABELS["incentive"], 0, 3600, 0)
        
    with col3:
        if dept == "Finished":
            wip = 0.0
            st.info("WIP locked at 0 for Finished.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0)
        
        overtime_raw = st.number_input(LABELS["over_time"], 0, 10000, 0)
        style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    with st.expander("⚙️ Idle Settings"):
        idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)
        idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True, type="primary")

# --- LOGIC ---
if submit:
    # 1. Initialize DataFrame with all 0.0
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # 2. Map Numerical Features
    # Note: We use 'over_time_scaled' as the column name because your rf_columns.pkl uses that name.
    # We pass the raw minutes because your Pipeline includes a scaler.
    numeric_map = {
        'team': float(team_num), 'smv': float(smv), 'wip': float(wip), 
        'incentive': float(incentive), 'idle_time': float(idle_time), 
        'idle_men': float(idle_men), 'no_of_workers': float(workers),
        'over_time_scaled': float(overtime_raw)
    }
    for k, v in numeric_map.items():
        if k in model_columns:
            input_df[k] = v

    # 3. Map Categorical Dummies
    # Function to set dummy columns to 1.0 if they exist in the model
    def set_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in model_columns:
            input_df[col_name] = 1.0

    set_dummy('department', dept.lower())
    set_dummy('quarter', quarter)
    set_dummy('day', day) # Now properly mapping the Day feature
    if style > 0:
        set_dummy('no_of_style_change', str(style))

    # 4. Prediction
    # Use only the columns the model was trained on, in the correct order
    features = input_df[model_columns]
    pred_idx = pipeline.predict(features)[0]
    probs = pipeline.predict_proba(features)[0]
    
    # Sklearn classes are usually sorted: [High, Low, Moderate] 
    # Check your Colab training to confirm if 0=High, 1=Low, 2=Moderate
    labels = ['High', 'Low', 'Moderate']
    status = labels[pred_idx]

    # --- RESULTS ---
    st.sidebar.title("📊 Results")
    st.sidebar.metric("Prediction", status)
    
    st.subheader(f"Predicted Status: {status}")
    for i, lab in enumerate(labels):
        st.write(f"{lab}: {probs[i]*100:.1f}%")
        st.progress(float(probs[i]))

    if status == "High":
        st.balloons()
