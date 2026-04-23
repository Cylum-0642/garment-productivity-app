import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# =========================================================
# PAGE CONFIG & STYLING
# =========================================================
st.set_page_config(page_title="Garment AI Consultant", page_icon="🧵", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    div[data-testid="stForm"] {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# DATA & ASSETS
# =========================================================
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

# =========================================================
# DATA & MODEL ASSETS LOADING
# =========================================================

@st.cache_resource
def load_assets():
    """Load the trained Random Forest model and the required column sequence."""
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error(f"❌ Critical Error: '{m_path}' or '{c_path}' not found.")
        st.stop()
        
    model = joblib.load(m_path)
    model_columns = joblib.load(c_path)
    return model, model_columns

@st.cache_data
def load_dataset():
    """Load the cleaned dataset for benchmarks."""
    d_path = "final_classification_dataset.csv"
    if os.path.exists(d_path):
        return pd.read_csv(d_path)
    return None

# Initialize the assets
pipeline, model_columns = load_assets()
df = load_dataset()

# =========================================================
# HEADER
# =========================================================
st.title("🧵 Intelligent Production Consultant")
st.caption("🚀 Decision Support System | Random Forest Classification Model")

st.markdown("""
**Purpose:** Optimize production tiers by adjusting Incentives, WIP, and Staffing.
1. **Predict:** Forecast if a shift will be High, Moderate, or Low productivity.
2. **Optimize:** Identify levers to reach **'High'** status.
""")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Production Parameters")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        smv = st.number_input(LABELS["smv"], 2.9, 60.0, 22.0, step=0.1)
        
        if dept == "Finished":
            wip = 0.0
            st.info("ℹ️ WIP is locked at 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 2700.0, 500.0, step=10.0)

    with col2:
        workers = st.number_input(LABELS["no_of_workers"], 2.0, 90.0, 30.0, step=1.0)
        incentive = st.number_input(LABELS["incentive"], 0, 3600, 0, step=10)
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)
        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 26000, 0, step=100)
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)
        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 45, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True, type="primary")

# =========================================================
# PREDICTION & RESULTS
# =========================================================
if submit:
    # 1. Create a dictionary from user inputs (Matching Dataset columns)
    input_data = {
        'smv': smv,
        'wip': wip,
        'over_time': overtime_raw,
        'incentive': incentive,
        'idle_time': idle_time,
        'idle_men': idle_men,
        'no_of_style_change': style,
        'no_of_workers': np.ceil(workers)
    }

    # 2. Initialize blank DataFrame using EXACT column order from pickle
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # 3. Fill numeric values
    for col, val in input_data.items():
        if col in model_columns:
            input_df.at[0, col] = val

    # 4. Fill Categorical (Dummies)
    def set_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in model_columns:
            input_df.at[0, col_name] = 1.0

    set_dummy('department', dept.lower())
    set_dummy('quarter', quarter)
    set_dummy('day', day)

    # 5. Predict (Using dynamic labels from the model)
    probs = pipeline.predict_proba(input_df)[0]
    labels = pipeline.classes_  # Dynamically fetch classes: e.g., ['High', 'Low', 'Moderate']
    status = str(labels[np.argmax(probs)])

    # 6. SIDEBAR RESULT
    st.sidebar.title("📊 Final Result")
    color = "#28a745" if status == "High" else "#fd7e14" if status == "Moderate" else "#dc3545"
    
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <h2 style="margin:0;">{status.upper()}</h2>
            <p style="margin:0; opacity:0.8;">Productivity Level</p>
        </div>
    """, unsafe_allow_html=True)

    # 7. DASHBOARD TABS
    t1, t2 = st.tabs(["Analysis & Recommendations", "Operational Benchmarks"])

    with t1:
        st.subheader("🔍 Model Confidence")
        # Create a mapping so we can display results in a logical order (Low -> High)
        prob_map = dict(zip(labels, probs))
        ordered_display = ['Low', 'Moderate', 'High']
        
        for lab in ordered_display:
            val = prob_map.get(lab, 0.0)
            st.progress(val, text=f"**{lab}**: {val*100:.1f}%")

        st.divider()
        st.subheader("💡 Strategic Recommendations")
        
        if status == "High":
            st.success("### 🌟 Target Met: Optimized Production")
            st.write("""
            - **Sustainability:** Avoid increasing 'Overtime' beyond current levels to prevent burnout.
            - **Quality:** High volume detected; increase frequency of spot checks.
            """)
            st.balloons()
        elif status == "Moderate":
            st.warning("### ⚖️ Target Partial: Stability Mode")
            st.write(f"""
            - **Incentive Gap:** Current: {incentive}. High-performing teams average 50.0.
            - **Bottleneck:** Check if WIP ({wip}) is causing station starvation or overloading.
            """)
        else:
            st.error("### ⚠️ Target Missed: Efficiency Warning")
            st.write(f"""
            - **Idle Time:** {idle_time} mins detected. Investigate machine breakdowns or supply chain delays immediately.
            - **Staffing:** {workers} workers may be insufficient for the current Task Complexity (SMV {smv}).
            """)

    with t2:
        st.subheader("📈 How you compare to 'High' Performers")
        cols = st.columns(4)
        # Static benchmarks based on dataset averages for "High" level
        met_list = [("SMV", smv, 13.7), ("WIP", wip, 770.5), ("Incentive", incentive, 50.0), ("Workers", workers, 33.0)]
        
        for i, (name, val, avg) in enumerate(met_list):
            diff = val - avg
            cols[i].metric(name, val, f"{diff:.1f} vs High-Avg", delta_color="inverse" if name == "SMV" else "normal")

        st.divider()
        st.write("- **Industrial Logic:** Lower SMV (simpler styles) and steady WIP (~770) drive High productivity.")
