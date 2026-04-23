import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Factory Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

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
# DATA OPTIONS
# =========================================================
quarter_options = sorted(df["quarter"].dropna().unique().tolist())
department_options = sorted(df["department"].dropna().unique().tolist())
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_options = [d for d in day_order if d in df["day"].unique()]
style_change_options = sorted(df["no_of_style_change"].dropna().unique().tolist())

# =========================================================
# SAFE ENCODING HELPERS (FIXED)
# =========================================================
def safe_one_hot(df_input, prefix, value):
    col_name = f"{prefix}_{str(value).strip()}"
    if col_name in df_input.columns:
        df_input[col_name] = 1
        return True
    return False

def normalize_label(pred):
    # robust handling (FIXED)
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    try:
        return label_map[int(pred)]
    except:
        return str(pred)

# =========================================================
# STYLING (UNCHANGED)
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
    }
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .status-badge {
        font-size: 2.5rem;
        font-weight: 800;
        padding: 0.5rem 2rem;
        border-radius: 9999px;
        color: white;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER (UNCHANGED)
# =========================================================
st.markdown("""
    <div class="main-header">
        <h1>🧵 Intelligent Production Consultant</h1>
        <p>🚀 Decision Support System | Random Forest Classification Model</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT (UNCHANGED)
# =========================================================
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 Shift Parameters")

    dept = st.selectbox("Department", department_options)
    day = st.selectbox("Day", day_options)
    quarter = st.selectbox("Quarter", quarter_options)

    smv = st.number_input("SMV", 2.0, 60.0, 22.0)
    workers = st.number_input("Workers", 1.0, 100.0, 30.0)

    is_finished = dept.strip().lower() == "finished"

    wip = st.number_input(
        "WIP",
        0.0,
        25000.0,
        0.0,
        disabled=is_finished
    )

    if is_finished:
        wip = 0.0

    incentive = st.number_input("Incentive", 0, 3600, 0)
    overtime = st.number_input("Overtime", 0, 10000, 0)

    idle_time = st.number_input("Idle Time", 0.0, 300.0, 0.0)
    idle_men = st.number_input("Idle Workers", 0, 50, 0)
    style = st.selectbox("Style Changes", style_change_options)

    submit = st.button("Analyze Production", use_container_width=True)

# =========================================================
# OUTPUT
# =========================================================
with col_output:
    if submit:

        # INIT INPUT (IMPORTANT FIX: float type stability)
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

        # NUMERIC SAFE MAP
        numeric_map = {
            "smv": smv,
            "wip": wip,
            "incentive": incentive,
            "idle_time": idle_time,
            "idle_men": idle_men,
            "no_of_workers": workers,
            "over_time": overtime
        }

        for k, v in numeric_map.items():
            if k in input_df.columns:
                input_df[k] = float(v)

        # CATEGORICAL FIX (more stable than your version)
        safe_one_hot(input_df, "department", dept)
        safe_one_hot(input_df, "quarter", quarter)
        safe_one_hot(input_df, "day", day)

        safe_one_hot(input_df, "no_of_style_change", style)

        # ALIGN FEATURES (CRITICAL FIX)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # =========================
        # PREDICTION (FIXED)
        # =========================
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            pred_idx = int(np.argmax(probs))
            status = normalize_label(pred_idx)
            conf = float(np.max(probs))
        else:
            pred = model.predict(input_df)[0]
            status = normalize_label(pred)
            probs = None
            conf = 1.0

        # =========================
        # DISPLAY (UNCHANGED STYLE LOGIC)
        # =========================
        color = "#16a34a" if status == "High" else "#ea580c" if status == "Moderate" else "#dc2626"

        st.markdown(f"""
            <div class="result-card">
                <h3>PREDICTED PRODUCTIVITY</h3>
                <div class="status-badge" style="background-color:{color}">
                    {status.upper()}
                </div>
                <p>Confidence: {conf*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # PROBABILITY SAFE DISPLAY
        if probs is not None:
            st.subheader("📊 Probability Breakdown")
            labels = ["Low", "Moderate", "High"]
            for i, l in enumerate(labels):
                st.progress(float(probs[i]), text=f"{l}: {probs[i]*100:.1f}%")

        # BASIC INSIGHT (kept minimal like your design)
# STRATEGIC ADVICE
        st.subheader("💡 Strategic Advice")

        if status == "High":
            st.success(f"""
            **The current setup reflects an efficient production balance.**

            **Key strengths:**
            * **Capacity Alignment:** Workforce ({workers}) is well-matched with workload (WIP: {wip}).
            * **Complexity Handling:** SMV ({smv}) is being handled effectively at current capacity.
            * **Operational Flow:** Minimal idle time indicates smooth workflow execution.

            **Data Insight:**
            High productivity in this system is driven by **workload distribution**—not necessarily higher incentives.

            **Management Recommendation:**
            Maintain this balance. Focus on consistency in planning and avoid unnecessary shifts in workforce or workload allocation.
            """)
            st.balloons()

        elif status == "Moderate":
            st.warning(f"""
            **The system shows acceptable performance but lacks optimal balance.**

            **Observed Gaps:**
            * **Resource Mismatch:** Potential imbalance between workers ({workers}) and workload (WIP: {wip}).
            * **Capacity Strain:** Task complexity (SMV: {smv}) may not align perfectly with current workforce capacity.
            * **Flow Inefficiency:** Productivity is limited by **allocation efficiency** rather than idle time.

            **Data Insight:**
            Moderate performance in this dataset usually stems from **inefficient distribution** rather than a lack of worker motivation.

            **Management Recommendation:**
            * **Rebalance:** Reallocate workload across the existing worker pool.
            * **Match SMV:** Adjust worker placement to better match the specific task complexity (SMV).
            * **Process over Incentives:** Prioritize flow management instead of relying on incentives as a primary solution.
            """)

        else:
            st.error(f"""
            **The configuration indicates poor production alignment.**

            **Critical Issues:**
            * **Structural Imbalance:** Significant mismatch between workforce ({workers}) and workload (WIP: {wip}).
            * **Handling Capacity:** SMV ({smv}) likely exceeds the current line's effective handling capacity.
            * **Planning Failure:** Inefficiency is driven by **poor structural planning**, not worker effort or idle time.

            **Data Insight:**
            Low productivity in this system is typically caused by a **structural mismatch**—worker motivation is rarely the root cause.

            **Management Recommendation:**
            * **Reallocate:** Move workers based strictly on real-time workload demand.
            * **Simplify:** Break down high SMV tasks or redistribute them to more capable lines.
            * **System Overhaul:** Reassess production planning entirely; increasing incentives will not fix a structural bottleneck.
            """)
