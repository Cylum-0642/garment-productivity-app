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
st.caption("Decision Support System for Garment Factory Productivity")

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("🔹 Production Parameters")

    col1, col2 = st.columns(2)

    with col1:
        dept = st.radio("Department", ["Sewing", "Finished"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        smv = st.number_input(LABELS["smv"], 2.0, 60.0, 22.0, step=0.1)

        if dept == "Finished":
            wip = 0.0
            st.info("WIP locked at 0 for Finished department.")
        else:
            wip = st.number_input(LABELS["wip"], 0.0, 25000.0, 500.0, step=10.0)

    with col2:
        workers = st.number_input(LABELS["no_of_workers"], 1.0, 100.0, 30.0, step=0.5)
        incentive = st.number_input(LABELS["incentive"], 0, 1000, 0)

    with st.expander("⚙️ Advanced Operational Settings"):
        col3, col4 = st.columns(2)

        with col3:
            overtime_raw = st.number_input(LABELS["over_time"], 0, 10000, 0, step=10)
            idle_time = st.number_input(LABELS["idle_time"], 0.0, 300.0, 0.0)

        with col4:
            idle_men = st.number_input(LABELS["idle_men"], 0, 50, 0)
            style = st.selectbox(LABELS["no_of_style_change"], [0, 1, 2])

    submit = st.form_submit_button("Analyze Production Status", type="primary")

# --- LOGIC ---
if submit:

    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)

    numeric_map = {
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time": overtime_raw
    }

    for k, v in numeric_map.items():
        if k in model_columns:
            input_df[k] = float(v)

    def set_dummy(cat, val):
        col = f"{cat}_{val}"
        if col in model_columns:
            input_df[col] = 1.0

    set_dummy("department", dept.lower())
    set_dummy("quarter", quarter)
    set_dummy("no_of_style_change", str(style))

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # =========================
    # 🔴 FIX 1: correct label mapping
    # =========================
    raw_classes = list(pipeline.classes_)
    probs = pipeline.predict_proba(input_df)[0]

    pred_idx = int(probs.argmax())
    status = str(raw_classes[pred_idx])

    # =========================
    # FIX 2: color must depend on TRUE label
    # =========================
    if status == "High":
        color = "#28a745"
    elif status == "Moderate":
        color = "#fd7e14"
    else:
        color = "#dc3545"

    # --- SIDEBAR RESULT ---
    st.sidebar.title("📊 Final Result")
    st.sidebar.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <h2 style="margin:0;">{status.upper()}</h2>
            <p style="margin:0; opacity:0.8;">Productivity Level</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MAIN TABS ---
    t1, t2 = st.tabs(["Analysis & Recommendations", "Operational Benchmarks"])

    # =========================
    # FIX 3: recommendations fully aligned with corrected status
    # =========================
    with t1:
        st.subheader("🔍 Model Confidence")

        for i, lab in enumerate(raw_classes):
            st.progress(float(probs[i]), text=f"{lab}: {probs[i]*100:.1f}%")

        st.divider()
        st.subheader("💡 Strategic Recommendations")

        if status == "High":
            st.success("### 🌟 Target Met: Optimized Production")
            st.write("""
            **Observation:** Your current configuration aligns with peak efficiency patterns.

            **Recommendations:**
            - Avoid increasing overtime unnecessarily.
            - Maintain current workflow stability.
            """)

        elif status == "Moderate":
            st.info("### ⚖️ Target Partial: Stability Mode")
            st.write("""
            **Observation:** Stable but not optimal performance.

            **Recommendations:**
            - Improve incentive efficiency.
            - Balance workload distribution (WIP).
            """)

        else:
            st.error("### ⚠️ Target Missed: Efficiency Warning")
            st.write("""
            **Observation:** Low productivity detected.

            **Recommendations:**
            - Reduce idle time immediately.
            - Check workforce allocation vs SMV complexity.
            - Investigate bottlenecks in production flow.
            """)

    # =========================
    # YOU REQUESTED: KEEP WORDING EXACT (UNCHANGED SECTION)
    # =========================
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
            cols[i].metric(name, val, f"{diff:.1f} vs High-Avg",
                           delta_color="inverse" if name == "SMV" else "normal")

        st.divider()
        st.write("**Industrial Logic:**")
        st.write("- **SMV:** Lower SMV (simpler styles) typically results in higher volume/productivity.")
        st.write("- **WIP:** High-performance teams maintain a steady flow (~770 units) to avoid line starvation.")
