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
st.caption("🚀 Powered by a **Random Forest Classification Model** trained on historical garment factory performance data.")

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
        # Validation range adjusted to CSV (min 2.9)
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0, step=0.1, help="Range: 2 - 60. Higher = More complex garment.")
        
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP is locked at 0 for Finished department.")
        else:
            # Validation range adjusted to CSV (max ~23k)
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0, step=10.0, help="Range: 0 - 25,000. Volume of items currently on the line.")

    with col2:
        team_num = st.selectbox("Team Number", list(range(1, 13)))
        # Validation range adjusted to CSV (max 90 workers)
        workers = st.number_input(LABELS["no_of_workers"], 1.0, 100.0, 30.0, step=0.5, help="Range: 1 - 100 workers.")
        # Validation range adjusted to CSV (max 3600 incentive)
        incentive = st.number_input(LABELS["incentive"], 0, 3600, 0, help="Range: 0 - 3600. Performance bonus.")

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)
        with col3:
            # We take RAW overtime here. The Pipeline's StandardScaler will scale it internally.
            overtime_raw = st.number_input(LABELS["over_time"], 0, 10000, 0, step=10, help="Total minutes. Max 10k.")
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0, help="Non-productive time in minutes.")
        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0, help="Number of workers waiting for work/repairs.")
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True, type="primary")

# --- LOGIC ---
if submit:
    # 1. Build blank dataframe with correct column structure
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # 2. Map Numerical Features (Directly mapping to expected column names)
    numeric_map = {
        'team': float(team_num), 
        'smv': float(smv), 
        'wip': float(wip), 
        'incentive': float(incentive),
        'idle_time': float(idle_time), 
        'idle_men': float(idle_men), 
        'no_of_workers': float(workers),
        'over_time_scaled': float(overtime_raw) # We send raw value; Pipeline Scaler fixes the scale
    }
    for k, v in numeric_map.items():
        if k in model_columns: 
            input_df[k] = v

    # 3. Handle Dummy/Categorical Variables
    # Department (Based on rf_columns.pkl, only 'department_sewing' exists)
    if dept.lower() == "sewing" and "department_sewing" in model_columns:
        input_df["department_sewing"] = 1.0
        
    # Quarter
    q_col = f"quarter_{quarter}"
    if q_col in model_columns:
        input_df[q_col] = 1.0
        
    # Style Changes
    if style > 0:
        s_col = f"no_of_style_change_{style}"
        if s_col in model_columns:
            input_df[s_col] = 1.0

    # 4. Prediction
    # We pass input_df[model_columns] to ensure feature order is identical to training
    pred_idx = pipeline.predict(input_df[model_columns])[0]
    probs = pipeline.predict_proba(input_df[model_columns])[0]
    
    # Mapping based on Sklearn's alphabetical/LabelEncoder order (High, Low, Moderate)
    labels = ['High', 'Low', 'Moderate']
    status = labels[pred_idx]

    # --- SIDEBAR RESULT ---
    st.sidebar.title("📊 Final Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <h2 style="margin:0;">{status.upper()}</h2>
            <p style="margin:0; opacity:0.8;">Productivity Level</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN DASHBOARD ---
    t1, t2 = st.tabs(["Analysis & Recommendations", "Operational Benchmarks"])

    with t1:
        st.subheader("🔍 Model Confidence")
        ordered_display = ['Low', 'Moderate', 'High']
        for lab in ordered_display:
            if lab in labels:
                val = probs[labels.index(lab)]
                st.progress(val, text=f"**{lab}**: {val*100:.1f}%")

        st.divider()
        st.subheader("💡 Strategic Recommendations")
        
        if status == "High":
            st.success("### 🌟 Target Met: Optimized Production")
            st.write("""
            **Observation:** Your current configuration aligns with peak efficiency patterns.
            
            **Recommendations:**
            - **Sustainability:** Avoid increasing 'Overtime' beyond current levels to prevent worker burnout.
            - **Quality Assurance:** Since volume is high, increase frequency of spot checks to ensure 'High' productivity doesn't compromise seam quality.
            - **Knowledge Sharing:** This team (Team {}) is a benchmark. Document their workflow for underperforming lines.
            """.format(team_num))
            st.balloons()

        elif status == "Moderate":
            st.info("### ⚖️ Target Partial: Stability Mode")
            st.write("""
            **Observation:** The line is steady but underperforming compared to potential capacity.
            
            **Recommendations:**
            - **Incentive Gap:** Your incentive is currently {}. High-performing teams average 50.0. A small increase could bridge the productivity gap.
            - **Bottleneck Analysis:** Check if 'WIP' ({}) is accumulating at a specific station. Moderate levels often suggest imbalanced line loading.
            - **Skill Matrix:** Consider moving 1-2 cross-trained workers to this team to handle the current SMV complexity.
            """.format(incentive, wip))

        else:
            st.error("### ⚠️ Target Missed: Efficiency Warning")
            st.write("""
            **Observation:** Critical inefficiencies detected. High probability of failing to meet production quotas.
            
            **Recommendations:**
            - **Eliminate Idle Time:** You have {} mins of idle time. This is the primary driver of 'Low' status. Investigate machine breakdowns or material delays immediately.
            - **Resource Reallocation:** The worker count ({}) may be insufficient for an SMV of {}. 
            - **Overtime Review:** If overtime is high but productivity is low, workers are likely fatigued. Consider an extra shift instead of extended overtime.
            """.format(idle_time, workers, smv))

    with t2:
        st.subheader("📈 How you compare to 'High' Performers")
        st.markdown("This section shows the variance between your input and the **ideal averages** for High productivity.")
        cols = st.columns(4)
        met_list = [
            ("SMV", smv, 13.7),
            ("WIP", wip, 770.5),
            ("Incentive", incentive, 50.0),
            ("Workers", workers, 33.1)
        ]
        for i, (name, val, avg) in enumerate(met_list):
            diff = val - avg
            cols[i].metric(name, val, f"{diff:.1f} vs High-Avg", delta_color="inverse" if name == "SMV" else "normal")

        st.divider()
        st.write("**Industrial Logic:**")
        st.write("- **SMV:** Lower SMV (simpler styles) typically results in higher volume/productivity.")
        st.write("- **WIP:** High-performance teams maintain a steady flow (~770 units) to avoid line starvation.")
