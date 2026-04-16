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

# Benchmarks derived from High-Productivity shifts in your cleaned dataset
AVERAGES = {
    'High':     {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1},
    'Moderate': {'smv': 16.7, 'wip': 682.5, 'incentive': 34.1, 'workers': 37.8},
    'Low':      {'smv': 15.5, 'wip': 478.0, 'incentive': 15.1, 'workers': 32.5}
}

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing. Please ensure 'rf_model.pkl' and 'rf_columns.pkl' are in the repository.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- TITLE & PURPOSE ---
st.title("🧵 Intelligent Production Consultant")
st.markdown("""
**Purpose:** This tool serves as a **Decision Support System** for Factory Managers. 
1. **Review:** Evaluate the productivity tier of past shifts.
2. **Predict:** Forecast the success of upcoming shifts by entering 'Target Values'.
3. **Optimize:** Identify which production levers (Incentives, WIP, Staffing) need adjustment to reach 'High' status.
""")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Production Parameters")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"], help="Note: Finished department usually has 0 WIP.")
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0, step=0.1, help="Range: 2 - 60. Higher = More complex garment.")
        
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP is locked at 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0, step=10.0, help="Range: 0 - 25,000. Volume of items currently on the line.")

    with col2:
        team_num = st.selectbox("Team Number", list(range(1, 13)))
        workers = st.number_input(LABELS["no_of_workers"], 1.0, 100.0, 30.0, step=0.5, help="Range: 1 - 100 workers.")
        incentive = st.number_input(LABELS["incentive"], 0, 1000, 0, help="Range: 0 - 1000. Performance bonus.")

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)
        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 10000, 0, step=10, help="Total minutes. Max 10k.")
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0, help="Non-productive time in minutes.")
        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0, help="Number of workers waiting for work/repairs.")
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True)

# --- LOGIC ---
if submit:
    # Overtime scaling (Modified Z-Score)
    ot_scaled = (overtime_raw - 0.0) / (2520.0 * 1.4826) if overtime_raw > 0 else -0.5

    # Build dataframe for model
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    numeric_map = {
        'team': float(team_num), 'smv': smv, 'wip': wip, 'incentive': incentive,
        'idle_time': idle_time, 'idle_men': idle_men, 'no_of_workers': workers,
        'over_time_scaled': ot_scaled
    }
    for k, v in numeric_map.items():
        if k in model_columns: input_df[k] = float(v)

    def set_dummy(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns: input_df[col] = 1.0
    set_dummy('department', dept.lower()); set_dummy('quarter', quarter); set_dummy('no_of_style_change', str(style))

    # Prediction
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    labels = ['High', 'Low', 'Moderate'] # Note: Scikit-learn alphabetical order
    status = labels[pred_idx]

    # --- SIDEBAR RESULT ---
    st.sidebar.title("📊 Final Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <h2 style="margin:0;">{status}</h2>
            <p style="margin:0; opacity:0.8;">Productivity Level</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN OUTPUT ---
    st.subheader("🔍 Model Confidence Levels")
    ordered_labels = ['Low', 'Moderate', 'High']
    for label in ordered_labels:
        idx = labels.index(label)
        st.progress(probs[idx], text=f"{label}: {probs[idx]*100:.1f}%")

    # --- KEY INSIGHTS (CORRECTED LOGIC) ---
    st.subheader("💡 Actionable Insights")
    c_low, c_high = st.columns(2)
    
    with c_low:
        if status == "Low":
            st.error("🚨 Critical: The model predicts a Low Productivity outcome.")
            if probs[labels.index("High")] < 0.2:
                st.warning("⚠️ High productivity is statistically unlikely with current settings.")
    
    with c_high:
        if incentive < AVERAGES['High']['incentive']:
            st.info(f"💰 **Opportunity:** Benchmark for 'High' teams is {AVERAGES['High']['incentive']}. Try increasing incentives.")
        if idle_time > 0:
            st.warning(f"⏳ **Efficiency Loss:** {idle_time} mins of idle time detected. Minimize machine downtime.")

    # --- BENCHMARK COMPARISON ---
    with st.expander("📈 Benchmark vs. Target Analysis", expanded=True):
        st.write("Comparing your current inputs against 'High Productivity' targets:")
        
        metrics_meta = {
            "Incentive Target": {"val": incentive, "ref": AVERAGES['High']['incentive'], "icon": "💰", "color": "#ffc107"},
            "Labor Capacity": {"val": workers, "ref": AVERAGES['High']['workers'], "icon": "👥", "color": "#20c997"},
            "Complexity Match": {"val": smv, "ref": AVERAGES['High']['smv'], "icon": "🧩", "color": "#17a2b8"}
        }

        for name, data in metrics_meta.items():
            # Calculate Percentage of Target
            percent_of_target = (data["val"] / data["ref"]) * 100 if data["ref"] != 0 else 0
            display_percent = min(percent_of_target, 150.0) # Cap display at 150% for visual clarity
            
            st.markdown(f"""
                <div style="margin-top:15px;">
                    <strong>{data['icon']} {name}</strong>: {percent_of_target:.1f}% of High-Prod Benchmark
                </div>
            """, unsafe_allow_html=True)
            st.progress(min(display_percent/100, 1.0))
