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
# DATA & ASSETS
# =========================================================
# Benchmarks for the Comparison Tab
AVERAGES = {
    'High': {'smv': 13.7, 'wip': 770.5, 'incentive': 50.0, 'workers': 33.1}
}

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

@st.cache_data
def load_dataset():
    # Used only for population of selectboxes to ensure data alignment
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_assets()
df_raw = load_dataset()

# Helper for encoding
def safe_one_hot(df_input, prefix, value):
    col_name = f"{prefix}_{str(value).strip()}"
    if col_name in df_input.columns:
        df_input[col_name] = 1.0

def normalize_label(pred):
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    try: return label_map[int(pred)]
    except: return str(pred)

# =========================================================
# SIDEBAR / HEADER
# =========================================================
st.markdown("""
    <div class="main-header">
        <h1>🧵 Intelligent Production Consultant</h1>
        <p>🚀 Decision Support System | Raondom Forest Model</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT FORM (Reordered based on Dataset Sequence)
# =========================================================
with st.form("input_form"):
    st.subheader("📋 Production Parameters")
    
    # Row 1: Categorical Foundation
    c1, c2, c3 = st.columns(3)
    with c1:
        quarter = st.selectbox("Quarter", sorted(df_raw["quarter"].unique()))
    with c2:
        dept = st.selectbox("Department", sorted(df_raw["department"].unique()))
    with c3:
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

    # Row 2: Core Numeric Metrics
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        smv = st.number_input("Task Complexity (SMV)", 2.0, 60.0, 22.0)
    with n2:
        is_finished = dept.strip().lower() == "finished"
        wip = st.number_input("Current Workload (WIP)", 0.0, 25000.0, 500.0, disabled=is_finished)
        if is_finished: wip = 0.0
    with n3:
        overtime = st.number_input("Overtime (Minutes)", 0, 10000, 0)
    with n4:
        incentive = st.number_input("Incentive (Bonus)", 0, 3600, 0)

    # Row 3: Workforce & Secondary factors
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        workers = st.number_input("Number of Workers", 1.0, 100.0, 30.0)
    with s2:
        idle_time = st.number_input("Idle Time (Mins)", 0.0, 300.0, 0.0)
    with s3:
        idle_men = st.number_input("Idle Workers", 0, 50, 0)
    with s4:
        style = st.selectbox("Style Changes", sorted(df_raw["no_of_style_change"].unique()))

    submit = st.form_submit_button("Analyze Production Status", use_container_width=True, type="primary")

# =========================================================
# PREDICTION & RESULTS
# =========================================================
if submit:
    # 1. Build Feature Vector
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # Map Numerics
    numeric_map = {
        "smv": smv, "wip": wip, "incentive": incentive,
        "idle_time": idle_time, "idle_men": idle_men,
        "no_of_workers": workers, "over_time": overtime
    }
    for k, v in numeric_map.items():
        if k in input_df.columns: input_df[k] = float(v)

    # Map Categoricals
    safe_one_hot(input_df, "department", dept)
    safe_one_hot(input_df, "quarter", quarter)
    safe_one_hot(input_df, "day", day)
    safe_one_hot(input_df, "no_of_style_change", style)

    # 2. Prediction
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        pred_idx = int(np.argmax(probs))
        status = normalize_label(pred_idx)
        conf = float(np.max(probs))
    else:
        status = normalize_label(model.predict(input_df)[0])
        probs, conf = None, 1.0

    # 3. Main Dashboard Display
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
            st.write("Confidence per Class:")
            labels = ["Low", "Moderate", "High"]
            for i, l in enumerate(labels):
                st.progress(float(probs[i]), text=f"{l}")

    with advice_col:
        t1, t2 = st.tabs(["💡 Strategic Advice", "📈 Comparison to 'High' Performers"])
        
        with t1:
            if status == "High":
                st.success("### 🌟 Target Met: Optimized Production")
                st.write(f"**Insight:** Workforce ({workers}) and WIP ({wip}) are in balance. Avoid increasing overtime to prevent burnout.")
                st.balloons()
            elif status == "Moderate":
                st.warning("### ⚖️ Target Partial: Efficiency Gap")
                st.write(f"**Insight:** Efficiency is limited by allocation. Check if SMV {smv} is too complex for current worker skill levels.")
            else:
                st.error("### ⚠️ Efficiency Warning: Structural Mismatch")
                st.write(f"**Insight:** Critical planning failure. High WIP ({wip}) relative to workers ({workers}) is creating a bottleneck.")

        with t2:
            st.write("Comparison against High-Productivity benchmarks:")
            m_cols = st.columns(2)
            comp_data = [
                ("SMV", smv, AVERAGES['High']['smv']),
                ("WIP", wip, AVERAGES['High']['wip']),
                ("Incentive", incentive, AVERAGES['High']['incentive']),
                ("Workers", workers, AVERAGES['High']['workers'])
            ]
            for i, (name, val, avg) in enumerate(comp_data):
                diff = val - avg
                target_col = m_cols[0] if i < 2 else m_cols[1]
                target_col.metric(name, val, f"{diff:.1f} vs High-Avg", delta_color="inverse" if name == "SMV" else "normal")

    st.divider()
    st.info("**Industrial Logic:** High productivity in this system is usually driven by **structural flow** (WIP vs Worker count) rather than just increasing incentives.")
