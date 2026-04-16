import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment AI Predictor", layout="wide")

@st.cache_resource
def load_assets():
    # UPDATED FILENAMES
    model_path = 'rf_model.pkl'
    cols_path = 'rf_columns.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        st.error(f"❌ Files not found: {model_path} or {cols_path}")
        st.stop()
        
    return joblib.load(model_path), joblib.load(cols_path)

pipeline, model_columns = load_assets()

# --- 2. SIDEBAR FOR RESULTS (No Scrolling Needed) ---
st.sidebar.title("🚀 Prediction Result")
st.sidebar.info("Adjust inputs and click 'Run Forecast' to update.")

# --- 3. MAIN UI LAYOUT ---
st.title("🧵 Factory Production Control")
st.markdown("Enter the operational metrics for the current shift below.")

# --- 4. INPUT VALIDATION & SELECTION ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Context")
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        dept = st.radio("Department", ["Sewing", "Finished"], horizontal=True)
        day = st.selectbox("Shift Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        team = st.number_input("Team ID", 1, 12, 1)
        
    with col2:
        st.subheader("⚙️ Complexity & Labor")
        # CONSTRAINT: We limit these based on realistic industry ranges
        smv = st.number_input("SMV (2.0 - 60.0)", 2.0, 60.0, 22.0)
        workers = st.number_input("Number of Workers (1 - 100)", 1.0, 100.0, 30.0)
        wip = st.number_input("WIP (0 - 25000)", 0.0, 25000.0, 500.0)
        style_change = st.selectbox("Style Changes", [0, 1, 2])

    st.subheader("💰 Performance Metrics")
    m1, m2, m3 = st.columns(3)
    with m1:
        incentive = st.number_input("Incentive (0 - 1000)", 0, 1000, 0)
    with m2:
        # USER INPUTS RAW MINUTES
        overtime_mins = st.number_input("Overtime (Minutes)", 0, 10000, 0)
    with m3:
        idle_time = st.number_input("Idle Time (Mins)", 0.0, 300.0, 0.0)
        # Note: idle_men is also part of your model
        idle_men = st.number_input("Idle Men", 0, 50, 0)

    submit = st.form_submit_button("Run Productivity Forecast", use_container_width=True)

# --- 5. LOGIC: SCALING & PREDICTION ---
if submit:
    try:
        # A. IMPLEMENT MODIFIED Z-SCORE SCALING
        # Values based on training set statistics
        MEDIAN_OT = 0.0
        MAD_OT = 2520.0  # Median Absolute Deviation
        
        # Modified Z-Score Formula
        if overtime_mins == 0:
            ot_scaled = -0.5 # Baseline for zero OT in many garment datasets
        else:
            ot_scaled = (overtime_mins - MEDIAN_OT) / (MAD_OT * 1.4826)

        # B. BUILD FEATURE DATAFRAME
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        
        # Fill Numeric
        input_df['team'] = float(team)
        input_df['smv'] = float(smv)
        input_df['wip'] = float(wip)
        input_df['incentive'] = float(incentive)
        input_df['idle_time'] = float(idle_time)
        input_df['idle_men'] = float(idle_men)
        input_df['no_of_workers'] = float(workers)
        input_df['over_time_scaled'] = float(ot_scaled)

        # Fill One-Hot
        def set_dummy(cat, val):
            col = f"{cat}_{val}"
            if col in model_columns: input_df[col] = 1.0

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', str(style_change))

        # C. PREDICT
        prediction = pipeline.predict(input_df[model_columns])[0]
        probs = pipeline.predict_proba(input_df[model_columns])[0]
        labels = ['High', 'Low', 'Moderate']
        
        # --- 6. OUTPUT TO SIDEBAR ---
        st.sidebar.divider()
        st.sidebar.header(f"Level: {labels[prediction]}")
        st.sidebar.metric("Confidence", f"{probs[prediction]:.1%}")
        
        if labels[prediction] == 'High':
            st.sidebar.success("Optimal Performance")
            st.balloons()
        elif labels[prediction] == 'Moderate':
            st.sidebar.info("Stable Performance")
        else:
            st.sidebar.error("Efficiency Warning")
            
        st.sidebar.write(f"Scaled Overtime used: {ot_scaled:.4f}")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
