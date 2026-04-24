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
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-badge {
        font-size: 2rem;
        font-weight: 800;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        color: white;
        display: inline-block;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# ASSETS & LOGIC
# =========================================================
AVERAGES = {'High': {'smv': 13.7, 'wip': 771, 'incentive': 50, 'workers': 33}}

@st.cache_resource
def load_assets():
    return joblib.load('rf_model.pkl'), joblib.load('rf_columns.pkl')

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_assets()
df_raw = load_dataset()

def safe_one_hot(df_input, prefix, value):
    col_name = f"{prefix}_{str(value).strip()}"
    if col_name in df_input.columns:
        df_input[col_name] = 1.0

def normalize_label(pred):
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    try: return label_map[int(pred)]
    except: return str(pred)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
    <div class="main-header">
        <h1>🧵 Intelligent Production Consultant</h1>
        <p>🚀 Decision Support System | Reorganized for User Workflow</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT FORM
# =========================================================
with st.form("input_form"):
    st.subheader("🗓️ 1. Shift Context")
    c1, c2, c3 = st.columns(3)
    with c1:
        quarter = st.selectbox("Quarter", sorted(df_raw["quarter"].unique()))
    with c2:
        # We select the department here
        dept = st.selectbox("Department", sorted(df_raw["department"].unique()))
    with c3:
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

    st.divider()
    
    st.subheader("⚙️ 2. Primary Production Levers")
    # This is where the model order matters for the user flow
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        smv = st.number_input("Task Complexity (SMV)", 2.0, 60.0, 22.0, step=0.1, help="Standard Minute Value")
    with n2:
        # --- CORRECTED WIP DISABLE LOGIC ---
        # We check the string value of the selected department
        is_finished = "finished" in dept.lower()
        # If finished, we force value to 0 and disable input
        wip = st.number_input("Current Workload (WIP)", 0, 25000, 0 if is_finished else 500, disabled=is_finished)
    with n3:
        workers = st.number_input("Number of Workers", 1, 100, 30, step=1)
    with n4:
        incentive = st.number_input("Incentive (Bonus)", 0, 3600, 0, step=1)

    with st.expander("🛠️ 3. Operational Stability & Exceptions", expanded=False):
        st.caption("Adjust these only if there were style changes or production delays.")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            overtime = st.number_input("Overtime (Mins)", 0, 10000, 0, step=10)
        with s2:
            idle_time = st.number_input("Idle Time (Mins)", 0.0, 300.0, 0.0, step=0.5)
        with s3:
            idle_men = st.number_input("Idle Workers", 0, 50, 0)
        with s4:
            style = st.selectbox("Style Changes", sorted(df_raw["no_of_style_change"].unique()))

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True, type="primary")

# =========================================================
# PREDICTION & ANALYSIS
# =========================================================
if submit:
    # 1. Align features with model sequence
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # 2. Map Numerical Values
    # Sequence: smv, wip, overtime, incentive, workers, others
    numeric_map = {
        "smv": smv, 
        "wip": wip, 
        "over_time": overtime, 
        "incentive": incentive, 
        "no_of_workers": workers,
        "idle_time": idle_time, 
        "idle_men": idle_men
    }
    for k, v in numeric_map.items():
        if k in input_df.columns:
            input_df[k] = float(v)

    # 3. Map Categorical Values
    safe_one_hot(input_df, "quarter", quarter)
    safe_one_hot(input_df, "department", dept)
    safe_one_hot(input_df, "day", day)
    safe_one_hot(input_df, "no_of_style_change", style)

    # 4. Run Model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        status = normalize_label(np.argmax(probs))
        conf = float(np.max(probs))
    else:
        status = normalize_label(model.predict(input_df)[0])
        probs, conf = None, 1.0

    # 5. UI Result Section
    res_col, advice_col = st.columns([1, 2])

    with res_col:
        color = "#16a34a" if status == "High" else "#ea580c" if status == "Moderate" else "#dc2626"
        st.markdown(f"""
            <div class="result-card">
                <p style="margin-bottom:0; color:#64748b;">PREDICTED STATUS</p>
                <div class="status-badge" style="background-color:{color}">
                    {status.upper()}
                </div>
                <p>Model Confidence: <b>{conf*100:.1f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        if probs is not None:
            labels = ["Low", "Moderate", "High"]
            for i, l in enumerate(labels):
                st.progress(float(probs[i]), text=f"{l}")

    with advice_col:
        t1, t2 = st.tabs(["💡 Strategic Recommendations", "📈 Comparison vs. High Performers"])
        
        with t1:
            if status == "High":
                st.success(f"### Target Met\nYour balance of {int(workers)} workers for an SMV of {smv} is optimal.")
                st.balloons()
            elif status == "Moderate":
                st.warning(f"### Efficiency Gap\nAdjust worker distribution or review WIP ({int(wip)}) flow.")
            else:
                st.error(f"### Critical Warning\nThe current setup indicates a major structural bottleneck.")

        with t2:
            st.write("Variance against High-Productivity benchmarks:")
            m_cols = st.columns(2)
            comp_data = [
                ("SMV", smv, AVERAGES['High']['smv'], True),
                ("WIP", wip, AVERAGES['High']['wip'], False),
                ("Incentive", incentive, AVERAGES['High']['incentive'], False),
                ("Workers", workers, AVERAGES['High']['workers'], False)
            ]
            for i, (name, val, avg, dec) in enumerate(comp_data):
                diff = val - avg
                fmt = ".1f" if dec else ".0f"
                target_col = m_cols[0] if i < 2 else m_cols[1]
                target_col.metric(name, f"{val:{fmt}}", f"{diff:{fmt}} vs High-Avg", 
                                 delta_color="inverse" if name == "SMV" else "normal")
