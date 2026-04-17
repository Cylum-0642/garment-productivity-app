import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG & ASSETS
# =========================================================
st.set_page_config(page_title="Garment AI Consultant", page_icon="🧵", layout="wide")

@st.cache_resource
def load_assets():
    m_path, c_path = 'rf_model.pkl', 'rf_columns.pkl'
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        st.error("Model files missing.")
        st.stop()
    return joblib.load(m_path), joblib.load(c_path)

pipeline, model_columns = load_assets()

# =========================================================
# STYLING
# =========================================================
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
    }
    .result-card {
        background: white; padding: 2rem; border-radius: 20px;
        border: 1px solid #e2e8f0; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .status-badge {
        font-size: 2.5rem; font-weight: 800; padding: 0.5rem 2rem;
        border-radius: 9999px; color: white; margin: 1rem 0; display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🧵 Intelligent Production Consultant</h1><p>Operational Decision Support System</p></div>', unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 Shift Parameters")
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            dept = st.radio("Department", ["Sewing", "Finished"])
            day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        with c2:
            quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
            team_num = st.selectbox("Team Number", list(range(1, 13)))
        
        smv = st.number_input("Task Complexity (SMV)", 2.0, 60.0, 22.0)
        workers = st.number_input("Number of Workers", 1.0, 100.0, 30.0)
        wip = st.number_input("Current WIP", 0.0, 3000.0, 500.0)
        incentive = st.number_input("Incentive Bonus", 0, 3600, 0)
        overtime = st.number_input("Overtime (Minutes)", 0, 10000, 0)
        
        with st.expander("⚙️ Advanced Settings"):
            idle_time = st.number_input("Idle Time (Mins)", 0.0, 300.0, 0.0)
            idle_men = st.number_input("Idle Workers", 0, 50, 0)
            style = st.selectbox("Style Changes", [0, 1, 2])

        submit = st.form_submit_button("Analyze Production", use_container_width=True, type="primary")

with col_output:
    if submit:
        # VALIDATION
        if dept == "Finished" and wip > 0:
            st.error("🚨 Finished department must have 0 WIP.")
            st.stop()

        # DATA PREP
        input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
        numeric_map = {
            'team': float(team_num), 'smv': float(smv), 'wip': float(wip), 
            'incentive': float(incentive), 'idle_time': float(idle_time), 
            'idle_men': float(idle_men), 'no_of_workers': float(workers),
            'over_time': float(overtime) 
        }
        for k, v in numeric_map.items():
            if k in model_columns: input_df[k] = v

        def set_dummy(prefix, val):
            col = f"{prefix}_{val}"
            if col in model_columns: input_df[col] = 1.0

        set_dummy('department', dept.lower())
        set_dummy('quarter', quarter)
        set_dummy('day', day)
        if style > 0: set_dummy('no_of_style_change', str(style))

        # PREDICTION
        # Using .predict()[0] directly gets the string label ("High", "Moderate", etc.)
        status = pipeline.predict(input_df[model_columns])[0]
        probs = pipeline.predict_proba(input_df[model_columns])[0]
        classes = list(pipeline.classes_)
        conf = probs[classes.index(status)]

        # RESULTS DISPLAY (Now safely inside the 'if submit' block)
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
        st.subheader("📊 Probability Breakdown")
        for i, lab in enumerate(classes):
            st.progress(float(probs[i]), text=f"**{lab}**: {probs[i]*100:.1f}%")

        # Metric Variance
        st.subheader("📈 Variance Analysis")
        b1, b2 = st.columns(2)
        b1.metric("SMV Gap", f"{smv:.1f}", f"{smv - 13.7:.1f}", delta_color="inverse")
        b2.metric("Incentive Gap", f"{incentive}", f"{incentive - 50.0:.1f}")
        
    else:
        # Default view when app loads
        st.markdown("""
            <div style="text-align: center; padding: 5rem; color: #94a3b8; border: 2px dashed #e2e8f0; border-radius: 20px; margin-top: 2rem;">
                <h3>Ready for Analysis</h3>
                <p>Adjust parameters and click 'Analyze Production' to generate results.</p>
            </div>
        """, unsafe_allow_html=True)
