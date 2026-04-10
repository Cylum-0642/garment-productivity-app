import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie

# --- CONFIGURATION ---
st.set_page_config(page_title="EcoMetric | Productivity Predictor", layout="wide", page_icon="🧵")

# --- CUSTOM CSS (The "Glow Up") ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Card-style containers for columns */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e6e9ef;
    }

    /* Gradient Title */
    .main-title {
        font-weight: 800;
        background: -webkit-linear-gradient(#0e1117, #2e3192);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0;
    }
    
    /* Style the buttons */
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        background-color: #2e3192;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1bffff;
        color: #0e1117;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

@st.cache_resource
def load_assets():
    model = joblib.load('garment_xgb_model.pkl')
    model_columns = joblib.load('xgb_model_columns.pkl')
    return model, model_columns

# --- LOAD DATA ---
model, model_columns = load_assets()
lottie_factory = load_lottieurl("https://lottie.host/81776993-2747-49f3-8025-06a19f40c6c5/H3Kj6m3VfM.json")

# --- HEADER SECTION ---
with st.container():
    head_col1, head_col2 = st.columns()
    with head_col1:
        st.markdown('<h1 class="main-title">Garment Productivity Predictor</h1>', unsafe_allow_html=True)
        st.write("### EcoMetric Solutions | AI-Driven Floor Optimization")
        st.info("""**Model Engine:** Tuned XGBoost Classifier. Optimized to evaluate non-linear floor operation variables and labor efficiency metrics.""")
    with head_col2:
        if lottie_factory:
            st_lottie(lottie_factory, height=200, key="factory_anim")

# --- INPUT UI ---
form_is_invalid = False
st.divider()

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 📅 Time & Place")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finished"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.markdown("### ⚙️ Resource Allocation")
    wip = st.number_input("Work in Progress (WIP)", value=500)
    if wip > 23122:
        st.error("⚠️ Max WIP: 23,122")
        form_is_invalid = True
        
    workers = st.number_input("Number of Workers", value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Range: 2 to 90")
        form_is_invalid = True

    style_change = st.selectbox("Style Changes", ["0", "1", "2"])
    smv = st.number_input("SMV (Complexity)", value=22.0)
    if smv > 55 or smv < 2.9:
        st.error("⚠️ Range: 2.9 to 54.6")
        form_is_invalid = True

with col3:
    st.markdown("### 💰 Incentives & Metrics")
    incentive = st.number_input("Incentive Amount", value=100)
    if incentive > 3600:
        st.error("⚠️ Max: 3,600")
        form_is_invalid = True
        
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)
    idle_time = st.number_input("Idle Time (Mins)", value=0)
    if idle_time > 300:
        st.error("⚠️ Max: 300")
        form_is_invalid = True
        
    idle_men = st.number_input("Idle Workers Count", value=0)
    if idle_men > 45:
        st.error("⚠️ Max: 45")
        form_is_invalid = True

# --- PREDICTION LOGIC ---
st.write("") # Spacer
if form_is_invalid:
    st.warning("Please correct the validation errors above.")
    st.button("Generate Productivity Forecast", disabled=True, use_container_width=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # Data Preparation
        input_df = pd.DataFrame(0, index=, columns=model_columns)
        input_df['team'] = team
        input_df['smv'] = smv
        input_df['wip'] = wip
        input_df['incentive'] = incentive
        input_df['idle_time'] = idle_time
        input_df['idle_men'] = idle_men
        input_df['no_of_workers'] = workers
        input_df['over_time_scaled'] = overtime 

        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in model_columns: input_df[col_name] = 1

        set_dummy('quarter', quarter)
        set_dummy('department', dept.lower())
        set_dummy('day', day)
        set_dummy('no_of_style_change', style_change)

        # Predict
        input_df = input_df[model_columns]
        prediction_idx = model.predict(input_df)
        probs = model.predict_proba(input_df)
        
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        confidence = probs[prediction_idx]

        # --- FANCY RESULTS DISPLAY ---
        st.markdown("---")
        res_col1, res_col2 = st.columns()
        
        with res_col1:
            st.write("### Prediction Result")
            if result == 'High':
                st.success(f"## {result} Productivity")
                st.balloons()
            elif result == 'Moderate':
                st.warning(f"## {result} Productivity")
            else:
                st.error(f"## {result} Productivity")
            
            st.metric("Model Confidence", f"{confidence:.2%}")

        with res_col2:
            st.write("### Probability Distribution")
            # Create a simple horizontal bar chart for probabilities
            prob_data = pd.DataFrame({'Tier': labels, 'Probability': probs})
            st.bar_chart(prob_data.set_index('Tier'))

        st.caption("Note: High productivity indicates optimized floor operations and target achievement.")
