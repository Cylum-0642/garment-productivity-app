import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Garment AI Consultant", layout="wide")

LABELS = {
    "smv": "Task Complexity (Standard Minute Value)",
    "wip": "Current Workload (Work in Progress)",
    "no_of_workers": "Number of Workers",
    "idle_time": "Idle Time (Minutes)",
    "idle_men": "Idle Workers",
    "incentive": "Incentive Amount (Bonus)",
    "over_time": "Overtime (Minutes)",
    "no_of_style_change": "Number of Style Changes"
}

AVERAGES = {
    'High':     {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1},
    'Moderate': {'smv': 16.7, 'wip': 682.5, 'incentive': 34.1, 'workers': 37.8},
    'Low':      {'smv': 15.5, 'wip': 478.0, 'incentive': 15.1, 'workers': 32.5}
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
st.markdown("Enter key production details to get a prediction and actionable insights.")

# --- INPUT FORM ---
with st.form("input_form"):

    st.subheader("🔹 Basic Inputs")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"], help="Select the factory department. Note: Finished department usually has 0 WIP.")
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"], help="The specific period of the year for this production run.")
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0, help="Standard Minute Value: The time allocated for a specific task.")
        
        # Lock WIP for Finished department
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP is set to 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0, help="Work in Progress: Number of unfinished items currently in the line.")

    with col2:
        team_num = st.selectbox("Team Number", list(range(1, 13)), help="Select the specific production team (1-12).")
        workers = st.number_input(LABELS["no_of_workers"], 1.0, 100.0, 30.0, help="The total number of workers allocated to this team.")
        incentive = st.number_input(LABELS["incentive"], 0, 1000, 0, help="Financial bonus/incentive offered to the workers.")

    with st.expander("⚙️ Advanced Settings"):
        col3, col4 = st.columns(2)

        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 5000, 0, help="Total overtime minutes worked by the team.")
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0, help="Duration in minutes where the production was halted.")

        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0, help="Number of workers who were idle during production.")
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2], help="The number of changes in product style during this session.")

    submit = st.form_submit_button("Analyze Production", use_container_width=True)

# --- LOGIC ---
if submit:
    # Overtime scaling
    ot_scaled = (overtime_raw - 0.0) / (2520.0 * 1.4826) if overtime_raw > 0 else -0.5

    # Build dataframe
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    numeric_map = {
        'team': float(team_num),
        'smv': smv,
        'wip': wip,
        'incentive': incentive,
        'idle_time': idle_time,
        'idle_men': idle_men,
        'no_of_workers': workers,
        'over_time_scaled': ot_scaled
    }

    for k, v in numeric_map.items():
        if k in model_columns:
            input_df[k] = float(v)

    def set_dummy(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns:
            input_df[col] = 1.0

    set_dummy('department', dept.lower())
    set_dummy('quarter', quarter)
    set_dummy('no_of_style_change', str(style))

    # Prediction
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]

    labels = ['High', 'Low', 'Moderate']
    status = labels[pred_idx]

    # --- SIDEBAR ---
    st.sidebar.title("📊 Final Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:white; margin:0;">{status}</h2>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN OUTPUT ---
    st.subheader("🔍 Model Confidence")
    ordered_labels = ['Low', 'Moderate', 'High']
    for label in ordered_labels:
        idx = labels.index(label)
        st.progress(probs[idx], text=f"{label}: {probs[idx]*100:.1f}%")

    st.subheader("💡 Key Insights")
    if probs[labels.index("High")] < 0.4:
        st.warning("Low probability of achieving High productivity.")
    if incentive < AVERAGES['High']['incentive']:
        st.info(f"Increasing incentives closer to benchmark ({AVERAGES['High']['incentive']}) may improve performance.")
    if idle_time > 0:
        st.error("Idle time detected — reduces efficiency.")

    # --- COLORFUL DETAILED COMPARISON ---
    with st.expander("📈 View Detailed Performance Insights", expanded=True):
        st.markdown("### Benchmarking vs Targets")
        
        def normalize(value, benchmark):
            return min(value / benchmark, 1.0) if benchmark != 0 else 0

        # Define display icons and colors
        metrics_meta = {
            "Task Complexity (SMV)": {"val": smv, "ref": AVERAGES['Moderate']['smv'], "icon": "🧩", "color": "#17a2b8"},
            "Workload (WIP)": {"val": wip, "ref": AVERAGES['Moderate']['wip'], "icon": "📦", "color": "#6f42c1"},
            "Incentive": {"val": incentive, "ref": AVERAGES['High']['incentive'], "icon": "💰", "color": "#ffc107"},
            "Workers": {"val": workers, "ref": AVERAGES['Moderate']['workers'], "icon": "👥", "color": "#20c997"}
        }

        for name, data in metrics_meta.items():
            ratio = normalize(data["val"], data["ref"])
            
            # Create a colorful header for each progress bar
            st.markdown(f"""
                <div style="border-left: 5px solid {data['color']}; padding-left: 10px; margin-top: 20px;">
                    <span style="font-size: 18px; font-weight: bold;">{data['icon']} {name}</span><br>
                    <span style="color: gray;">Current: <b>{data['val']}</b> | Benchmark: <b>{data['ref']}</b></span>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(ratio)
