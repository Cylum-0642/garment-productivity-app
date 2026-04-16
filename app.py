import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Garment AI Consultant", layout="wide", page_icon="🧵")

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

AVERAGES = {
    'High':     {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1},
    'Moderate': {'smv': 16.7, 'wip': 682.5, 'incentive': 34.1, 'workers': 37.8},
    'Low':      {'smv': 15.5, 'wip': 478.0, 'incentive': 15.1, 'workers': 32.5}
}

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing. Please ensure 'rf_model.pkl' and 'rf_columns.pkl' are in the directory.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- TITLE ---
st.title("🧵 Intelligent Production Consultant")
st.markdown("Enter production parameters to analyze efficiency and receive AI-driven strategic recommendations.")

# --- INPUT FORM ---
with st.form("input_form"):
    
    st.subheader("🔹 Primary Production Parameters")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"], help="WIP is locked to 0 for Finished department.")
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0)
        
        # Dynamic WIP Logic
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP automatically set to 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0)
            
        workers = st.number_input(LABELS["no_of_workers"], 2.0, 100.0, 30.0)

    with col2:
        # Added missing categorical variables: Quarter, Day, and Team
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        team = st.slider("Team Number", 1, 12, 1)
        incentive = st.number_input(LABELS["incentive"], 0, 200, 0)

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)
        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 5000, 0)
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)
        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Run Analysis", use_container_width=True)

# --- LOGIC & OUTPUT ---
if submit:
    # 1. Scaling & Data Prep
    ot_scaled = (overtime_raw - 0.0) / (2520.0 * 1.4826) if overtime_raw > 0 else -0.5
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    numeric_map = {
        'team': team,
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
    set_dummy('day', day)
    set_dummy('no_of_style_change', str(style))

    # 2. Prediction
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    
    # Matching labels to model classes
    labels = list(pipeline.classes_)
    status = labels[pred_idx]

    # --- SIDEBAR (FINAL RESULT) ---
    st.sidebar.title("📊 Prediction Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <p style="margin:0; font-size:14px; opacity:0.8;">CURRENT STATUS</p>
            <h2 style="margin:0; font-size:32px;">{status}</h2>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN DASHBOARD TABS ---
    tab1, tab2 = st.tabs(["AI Analysis", "Benchmarks"])

    with tab1:
        st.subheader("🔍 Model Confidence")
        ordered_labels = ['Low', 'Moderate', 'High']
        for label in ordered_labels:
            if label in labels:
                idx = labels.index(label)
                st.progress(probs[idx], text=f"**{label}**: {probs[idx]*100:.1f}%")

        st.divider()
        st.subheader("💡 Strategic Recommendations")
        if status != "High":
            if incentive < AVERAGES['High']['incentive']:
                st.warning(f"**Incentive Gap:** Current incentive is below the high-productivity benchmark ({AVERAGES['High']['incentive']}). Consider a bonus increase.")
            if idle_time > 0:
                st.error("**Efficiency Leak:** Idle time detected. Investigate machine maintenance or material flow bottlenecks.")
        else:
            st.success("**Optimal Setup:** Current parameters align with high-productivity outcomes.")

    with tab2:
        st.subheader("📈 Performance Benchmarks")
        cols = st.columns(4)
        
        metrics = [
            ("SMV", smv, AVERAGES['Moderate']['smv']),
            ("WIP", wip, AVERAGES['Moderate']['wip']),
            ("Incentive", incentive, AVERAGES['High']['incentive']),
            ("Workers", workers, AVERAGES['Moderate']['workers'])
        ]
        
        for i, (name, val, ref) in enumerate(metrics):
            cols[i].metric(name, val, f"{val-ref:.1f} vs Avg")
            
        st.divider()
        st.write("📊 **Comparison to Production Tiers (Progress Bars)**")
        for name, (val, ref) in zip(["Task Complexity", "Workload", "Bonus Level", "Staffing"], metrics):
            ratio = min(val / ref, 1.5) if ref != 0 else 0
            st.write(f"{name} vs Benchmark")
            st.progress(min(ratio / 1.5, 1.0))
