import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Garment AI Consultant", layout="wide", page_icon="🧵")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stNumberInput, .stRadio { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

LABELS = {
    "smv": "Task Complexity (SMV)",
    "wip": "Work in Progress (WIP)",
    "no_of_workers": "Team Size (Workers)",
    "idle_time": "Idle Time (Minutes)",
    "idle_men": "Idle Workers Count",
    "incentive": "Incentive (Bonus)",
    "over_time": "Overtime (Minutes)",
    "no_of_style_change": "Style Changes"
}

# Values based on 95th percentiles of your cleaned dataset for better realism
DATA_BOUNDS = {
    "smv": {"min": 2.0, "max": 60.0, "default": 22.0},
    "wip": {"min": 0.0, "max": 5000.0, "default": 500.0}, # Capped for stability
    "workers": {"min": 2.0, "max": 100.0, "default": 30.0},
    "incentive": {"min": 0, "max": 200, "default": 0}, # Realistic business bonus
    "overtime": {"min": 0, "max": 10000, "default": 0}
}

AVERAGES = {
    'High':     {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1},
    'Moderate': {'smv': 16.7, 'wip': 682.5, 'incentive': 34.1, 'workers': 37.8},
    'Low':      {'smv': 15.5, 'wip': 478.0, 'incentive': 15.1, 'workers': 32.5}
}

@st.cache_resource
def load_assets():
    # Update paths to match your deployment environment
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error(f"Missing model files in {os.getcwd()}")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- TITLE ---
st.title("🧵 Intelligent Production Consultant")
st.markdown("Optimize factory floor decisions with data-driven productivity predictions.")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Core Production Variables")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"], help="Note: WIP is automatically set to 0 for Finished department.")
        smv = st.number_input(LABELS["smv"], DATA_BOUNDS["smv"]["min"], DATA_BOUNDS["smv"]["max"], DATA_BOUNDS["smv"]["default"])
        
        # --- DATA VALIDATION LOGIC ---
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP is locked to 0 for Finished department (Production complete).")
        else:
            wip = st.number_input(LABELS["wip"], DATA_BOUNDS["wip"]["min"], DATA_BOUNDS["wip"]["max"], DATA_BOUNDS["wip"]["default"])

    with col2:
        workers = st.number_input(LABELS["no_of_workers"], DATA_BOUNDS["workers"]["min"], DATA_BOUNDS["workers"]["max"], DATA_BOUNDS["workers"]["default"])
        incentive = st.number_input(LABELS["incentive"], DATA_BOUNDS["incentive"]["min"], DATA_BOUNDS["incentive"]["max"], DATA_BOUNDS["incentive"]["default"])

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)
        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], DATA_BOUNDS["overtime"]["min"], DATA_BOUNDS["overtime"]["max"], DATA_BOUNDS["overtime"]["default"])
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)
        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Run AI Analysis", use_container_width=True)

# --- LOGIC ---
if submit:
    # Consistency Scaling for Overtime (Matching the Training Data scale)
    ot_scaled = (overtime_raw - 0.0) / (2520.0 * 1.4826) if overtime_raw > 0 else -0.5

    # Prepare DataFrame
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # Map Numeric Inputs
    numeric_map = {
        'smv': smv, 'wip': wip, 'incentive': incentive,
        'idle_time': idle_time, 'idle_men': idle_men,
        'no_of_workers': workers, 'over_time_scaled': ot_scaled,
        'team': 1.0 # Defaulting to team 1 if not provided
    }
    
    for k, v in numeric_map.items():
        if k in model_columns:
            input_df[k] = float(v)

    # Map Dummies
    def set_dummy(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns:
            input_df[col] = 1.0

    set_dummy('department', dept.lower())
    set_dummy('no_of_style_change', str(style))

    # Prediction
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    
    # In some models, classes are alphabetized (High, Low, Moderate)
    # Ensure this matches your pipeline.classes_ order!
    labels = list(pipeline.classes_) 
    status = labels[pred_idx]

    # --- SIDEBAR RESULT ---
    st.sidebar.title("📊 Analysis Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:white; margin:0;">{status}</h2>
            <p style="color:white; opacity:0.8;">Productivity Level</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN DASHBOARD ---
    t1, t2 = st.tabs(["Analysis", "Operational Benchmarks"])

    with t1:
        st.subheader("🔍 Model Confidence")
        ordered_display = ['Low', 'Moderate', 'High']
        for lab in ordered_display:
            if lab in labels:
                val = probs[labels.index(lab)]
                st.progress(val, text=f"**{lab}**: {val*100:.1f}%")

        st.divider()
        st.subheader("💡 Strategic Recommendations")
        if status != "High":
            if incentive < 40:
                st.warning("⚠️ **Boost Incentive:** Current bonus is below the 'High' productivity benchmark (50.0). Consider increasing it.")
            if idle_time > 0:
                st.error("❌ **Reduce Idle Time:** Any machine downtime significantly drops probability of High output.")
        else:
            st.success("✅ **Balanced Setup:** This configuration is likely to meet or exceed targets.")

    with t2:
        st.subheader("📈 How you compare to 'High' Performers")
        # Visualizing distance from the successful average
        cols = st.columns(4)
        met_list = [
            ("SMV", smv, 13.7),
            ("WIP", wip, 770.5),
            ("Incentive", incentive, 50.0),
            ("Workers", workers, 33.1)
        ]
        for i, (name, val, avg) in enumerate(met_list):
            cols[i].metric(name, val, f"{val-avg:.1f} vs Avg")
