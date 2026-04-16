import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Garment AI Consultant", layout="wide", page_icon="🧵")

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
        st.error("Model files missing. Ensure 'rf_model.pkl' and 'rf_columns.pkl' are in the same folder.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# --- TITLE ---
st.title("🧵 Intelligent Production Consultant")
st.markdown("Enter production details to get a probability-based prediction and strategic insights.")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Production Parameters")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"], help="WIP is locked to 0 for Finished department.")
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0)
        
        # --- WIP VALIDATION LOGIC ---
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP is automatically set to 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0)
        
        workers = st.number_input(LABELS["no_of_workers"], 2.0, 100.0, 30.0)

    with col2:
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        team = st.slider("Team Number", 1, 12, 1)
        incentive = st.number_input(LABELS["incentive"], 0, 200, 0)
        day_default = "Wednesday" # Hidden background default for model stability

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)
        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 5000, 0)
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)
        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Analyze Production", use_container_width=True)

# --- LOGIC & OUTPUT ---
if submit:
    # 1. Data Prep
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
    set_dummy('day', day_default)
    set_dummy('no_of_style_change', str(style))

    # 2. Prediction
    model_classes = list(pipeline.classes_) 
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    status = pipeline.predict(input_df[model_columns])[0]

    # --- SIDEBAR RESULT ---
    st.sidebar.title("📊 Analysis Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <p style="margin:0; font-size:12px; opacity:0.8;">PREDICTED TIER</p>
            <h2 style="margin:0; font-size:28px;">{status}</h2>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN CONTENT TABS ---
    tab1, tab2 = st.tabs(["AI Confidence & Insights", "Benchmarking"])

    with tab1:
        st.subheader("🔍 Model Confidence")
        display_order = ['Low', 'Moderate', 'High']
        
        for label in display_order:
            if label in model_classes:
                idx = model_classes.index(label)
                conf_val = probs[idx]
                st.write(f"**{label} Productivity**")
                st.progress(conf_val, text=f"{conf_val*100:.1f}% confidence")

        st.divider()
        st.subheader("💡 Strategic Insights")
        
        # Fixed the string literal error here:
        if "High" in model_classes:
            high_idx = model_classes.index("High")
            if probs[high_idx] < 0.5:
                st.warning("Current setup has a low probability of 'High' output. Review the Benchmarking tab to identify constraints.")
        
        if incentive < AVERAGES['High']['incentive']:
            st.info(f"Financial Insight: Your incentive is below the high-tier average ({AVERAGES['High']['incentive']}). Performance may increase with a bonus adjustment.")

    with tab2:
        st.subheader("📈 How you compare to the Industry Average")
        
        def normalize(value, benchmark):
            return min(value / benchmark, 1.5) if benchmark != 0 else 0

        metrics = {
            "Task Complexity (SMV)": (smv, AVERAGES['Moderate']['smv']),
            "Workload (WIP)": (wip, AVERAGES['Moderate']['wip']),
            "Incentive Level": (incentive, AVERAGES['High']['incentive']),
            "Staffing (Workers)": (workers, AVERAGES['Moderate']['workers'])
        }

        for name, (val, ref) in metrics.items():
            ratio = normalize(val, ref)
            st.write(f"**{name}**: {val} (Benchmark: {ref})")
            st.progress(min(ratio/1.5, 1.0))
