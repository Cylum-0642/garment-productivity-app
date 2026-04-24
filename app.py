import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# =========================================================
# PAGE CONFIG & STYLING
# =========================================================
st.set_page_config(page_title="Production Advisor", page_icon="🧵", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    div[data-testid="stForm"] {
        background: white; border-radius: 20px; padding: 2rem; border: 1px solid #e2e8f0;
    }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
    }
    .result-card {
        background: white; padding: 1.5rem; border-radius: 15px;
        border: 1px solid #e2e8f0; text-align: center; margin-bottom: 1rem;
    }
    .status-badge {
        font-size: 2rem; font-weight: 800; padding: 0.5rem 2rem;
        border-radius: 10px; color: white; display: inline-block; margin: 10px 0;
    }
    .action-box {
        background-color: #ffffff; padding: 15px; border-left: 5px solid #3b82f6;
        margin-bottom: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# ASSETS & BENCHMARKS
# =========================================================
HIGH_BENCH = {'smv': 13.7, 'wip': 771, 'incentive': 50, 'workers': 33}

@st.cache_resource
def load_assets():
    return joblib.load('rf_model.pkl'), joblib.load('rf_columns.pkl')

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

model, model_columns = load_assets()
df_raw = load_dataset()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
    <div class="main-header">
        <h1>🧵 Smart Factory Advisor</h1>
        <p>Simple production guidance for Floor Managers and Supervisors</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT FORM (Simplified Terms)
# =========================================================
with st.form("input_form"):
    st.subheader("📅 Section 1: When and Where?")
    c1, c2, c3 = st.columns(3)
    with c1: 
        quarter = st.selectbox("Week of the Month", sorted(df_raw["quarter"].unique()), help="Which part of the month is this?")
    with c2: 
        dept = st.selectbox("Department", sorted(df_raw["department"].unique()), help="Sewing or Finished goods?")
    with c3: 
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

    st.divider()
    st.subheader("🚀 Section 2: Main Planning (How to hit your target)")
    n1, n2, n3, n4 = st.columns(4)
    with n1: 
        smv = st.number_input("Difficulty of Style", 2.0, 60.0, 22.0, step=0.1, help="How many minutes is allocated to sew one piece? (Technical term: SMV)")
    with n2:
        is_finished = "finished" in dept.lower()
        wip = st.number_input("Unfinished Pieces in Line", 0, 25000, 0 if is_finished else 500, disabled=is_finished, help="Number of garments currently being worked on. (Technical term: WIP)")
    with n3: 
        workers = st.number_input("Number of Workers", 1, 100, 30, step=1, help="Total people working in this team.")
    with n4: 
        incentive = st.number_input("Cash Bonus (Total)", 0, 3600, 0, step=1, help="Extra money offered to motivate the team.")

    with st.expander("⚠️ Section 3: Interruptions & Overtime (If any)", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1: overtime = st.number_input("Total Overtime (Mins)", 0, 10000, 0, step=10)
        with s2: idle_time = st.number_input("Production Stoppage (Mins)", 0.0, 300.0, 0.0, step=0.5, help="Time lost due to machine breakdown or material delay.")
        with s3: idle_men = st.number_input("Waiting Workers", 0, 50, 0, help="Number of workers sitting idle during a stoppage.")
        with s4: style = st.selectbox("Style Changes Today", sorted(df_raw["no_of_style_change"].unique()), help="How many times did the team switch to a different product design?")

    submit = st.form_submit_button("Check Production Health", use_container_width=True, type="primary")

# =========================================================
# RESULTS & SIMPLE ADVICE
# =========================================================
if submit:
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    # Map to model features
    features = {"smv": smv, "wip": wip, "over_time": overtime, "incentive": incentive, 
                "no_of_workers": workers, "idle_time": idle_time, "idle_men": idle_men}
    for k, v in features.items():
        if k in input_df.columns: input_df[k] = float(v)

    # One-hot encoding
    for cat, val in [("quarter", quarter), ("department", dept), ("day", day), ("no_of_style_change", style)]:
        col = f"{cat}_{str(val).strip()}"
        if col in input_df.columns: input_df[col] = 1.0

    probs = model.predict_proba(input_df)[0]
    labels = ["Low", "Moderate", "High"]
    status = labels[np.argmax(probs)]
    conf = np.max(probs)

    res_col, advice_col = st.columns([1, 2])

    with res_col:
        color = "#16a34a" if status == "High" else "#ea580c" if status == "Moderate" else "#dc2626"
        st.markdown(f"""
            <div class="result-card">
                <p style="margin:0; color:#64748b;">PREDICTED PERFORMANCE</p>
                <div class="status-badge" style="background-color:{color}">{status.upper()}</div>
                <p>Accuracy of this guess: <b>{conf*100:.0f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        for i, l in enumerate(labels):
            st.progress(float(probs[i]), text=f"Chance of being {l}")

    with advice_col:
        st.subheader("📋 Manager's Action Plan")
        
        # Calculations for advice
        wip_needed = int(HIGH_BENCH['wip'] - wip)
        worker_needed = int(HIGH_BENCH['workers'] - workers)

        if status == "High":
            st.success("### Excellent! The line is healthy.")
            st.write("Everything is balanced. Just make sure to check for quality mistakes, as the team is working fast.")
        else:
            # 1. Check for "Starving" (Empty Line)
            if wip < (HIGH_BENCH['wip'] * 0.7) and not is_finished:
                st.markdown(f"""<div class="action-box"><b>📦 THE LINE IS RUNNING EMPTY (Starving)</b><br>
                You only have {int(wip)} pieces on the line. Workers are waiting for work. <br>
                <b>Action:</b> Feed <b>{wip_needed} more pieces</b> into the line immediately.</div>""", unsafe_allow_html=True)
            
            # 2. Check for "Traffic Jam" (Congestion)
            elif wip > (HIGH_BENCH['wip'] * 1.3):
                st.markdown(f"""<div class="action-box"><b>📦 PRODUCTION TRAFFIC JAM (Bottleneck)</b><br>
                Too many pieces ({int(wip)}) are piled up. This causes confusion and delays.<br>
                <b>Action:</b> Stop adding new work for an hour. Move extra workers to help clear the pile-up.</div>""", unsafe_allow_html=True)

            # 3. Check Manpower
            if workers < (HIGH_BENCH['workers'] - 5):
                st.markdown(f"""<div class="action-box"><b>👥 TEAM IS TOO SMALL</b><br>
                This product is difficult. Your team of {int(workers)} is too small to hit the target.<br>
                <b>Action:</b> Try to add <b>{abs(worker_needed)} more workers</b> to this team.</div>""", unsafe_allow_html=True)

            # 4. Check Interruptions
            if idle_time > 0:
                st.markdown(f"""<div class="action-box" style="border-left-color: #dc2626;"><b>🛠️ STOPPAGE ALERT</b><br>
                You lost {idle_time} minutes. <br>
                <b>Action:</b> Check if a machine is broken or if fabric is missing. Fix this first before worrying about speed!</div>""", unsafe_allow_html=True)

        with st.expander("📊 Show Simple Goal Comparison"):
            st.write("How your current numbers compare to a 'Top Performing' team:")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pieces on Line", int(wip), f"{wip_needed} to go" if wip_needed > 0 else "Good")
            c2.metric("Team Size", int(workers), f"{worker_needed} needed" if worker_needed > 0 else "Good")
            c3.metric("Bonus Level", f"{int(incentive)}", f"{int(HIGH_BENCH['incentive'] - incentive)} gap")
