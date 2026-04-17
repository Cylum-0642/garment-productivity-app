import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment AI Consultant",
    page_icon="🧵",
    layout="wide",
)

# =========================================================
# PROFESSIONAL STYLING
# =========================================================
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .status-badge {
        font-size: 2.5rem;
        font-weight: 800;
        padding: 0.5rem 2rem;
        border-radius: 9999px;
        color: white;
        margin: 1rem 0;
        display: inline-block;
    }
    div[data-testid="stForm"] {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ASSETS
# =========================================================
@st.cache_resource
def load_model_assets():
    model = joblib.load("rf_model.pkl")
    model_columns = joblib.load("rf_columns.pkl")
    return model, model_columns

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_model_assets()
df = load_dataset()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
    <div class="main-header">
        <h1>🧵 Intelligent Production Consultant</h1>
        <p>Operational Decision Support System for Garment Factory Managers</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT
# =========================================================
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 Shift Parameters")
    with st.form("input_form"):
        # Section 1: Categories
        c1, c2 = st.columns(2)
        with c1:
            dept = st.radio("Department", ["Sewing", "Finished"])
            day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        with c2:
            quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])

        st.divider()

        # Section 2: Metrics
        smv = st.number_input("Task Complexity (SMV)", 2.0, 60.0, 22.0)
        workers = st.number_input("Number of Workers", 1.0, 100.0, 30.0)

        if dept == "Finished":
            wip = 0.0
            st.info("WIP locked at 0 for Finished Dept.")
        else:
            wip = st.number_input("Current WIP", 0.0, 25000.0, 500.0)

        incentive = st.number_input("Incentive Bonus", 0, 3600, 0)
        overtime = st.number_input("Overtime (Minutes)", 0, 10000, 0)

        with st.expander("⚙️ Advanced Operational Settings"):
            idle_time = st.number_input("Idle Time (Mins)", 0.0, 300.0, 0.0)
            idle_men = st.number_input("Idle Workers", 0, 50, 0)
            style = st.selectbox("Style Changes", [0, 1, 2])

        submit = st.form_submit_button("Analyze Production", use_container_width=True, type="primary")

with col_output:
    if submit:
        # =========================================================
        # PREDICTION LOGIC (FIXED)
        # =========================================================

        # 1. Initialize DataFrame
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

        # 2. Correct Numeric Mapping (ALIGNED WITH DATASET)
        numeric_map = {
            'smv': float(smv),
            'wip': float(wip),
            'incentive': float(incentive),
            'idle_time': float(idle_time),
            'idle_men': float(idle_men),
            'no_of_workers': float(workers),
            'over_time': float(overtime)
        }

        for k, v in numeric_map.items():
            if k in model_columns:
                input_df[k] = v

        # 3. Categorical Mapping
        def set_dummy(prefix, val):
            col = f"{prefix}_{val}"
            if col in model_columns:
                input_df[col] = 1.0

        set_dummy('department', dept.lower())
        set_dummy('quarter', quarter)
        set_dummy('day', day)

        if style > 0:
            set_dummy('no_of_style_change', str(style))

        # 4. Prediction
        probs = pipeline.predict_proba(input_df[model_columns])[0]
        labels = ['High', 'Low', 'Moderate']
        pred_idx = probs.argmax()
        status = labels[pred_idx]
        conf = probs[pred_idx]

        # =========================================================
        # DISPLAY RESULTS
        # =========================================================
        color = "#16a34a" if status == "High" else "#ea580c" if status == "Moderate" else "#dc2626"

        st.markdown(f"""
            <div class="result-card">
                <p style="color: #64748b; font-weight: 600; margin-bottom: 0;">PREDICTED PRODUCTIVITY</p>
                <div class="status-badge" style="background-color: {color};">
                    {status.upper()}
                </div>
                <p style="color: #64748b;">Model Confidence: {conf*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Probability Breakdown
        st.subheader("📊 Probability Breakdown")
        for i, lab in enumerate(labels):
            st.progress(float(probs[i]), text=f"**{lab}**: {probs[i]*100:.1f}%")

        # Benchmark
        st.subheader("📈 Variance vs. High-Performance Benchmark")
        b1, b2 = st.columns(2)
        b1.metric("SMV Gap", f"{smv:.1f}", f"{smv - 13.7:.1f}", delta_color="inverse")
        b2.metric("Incentive Gap", f"{incentive}", f"{incentive - 50.0:.1f}")

        # Recommendation
        st.subheader("💡 Strategic Advice")
        if status == "High":
            st.success("Configuration is optimal. High productivity confirmed.")
            st.balloons()
        elif status == "Moderate":
            st.warning("Line is stable. Look into reducing bottlenecks or increasing incentives to reach 'High' status.")
        else:
            st.error("Operational Risk. High idle time or workforce mismatch detected.")

    else:
        st.markdown("""
            <div style="text-align: center; padding: 5rem; color: #94a3b8; border: 2px dashed #e2e8f0; border-radius: 20px;">
                <h3>Ready for Analysis</h3>
                <p>Adjust the parameters on the left and click 'Analyze Production' to see results here.</p>
            </div>
        """, unsafe_allow_html=True)
