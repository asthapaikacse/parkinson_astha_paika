import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Parkinson's Disease Prediction App")
st.write("This app predicts **Parkinson's Disease** and **UPDRS Severity** using machine learning models.")

# --------------------------------------------------------------------------------
# LOAD MODELS SAFELY
# --------------------------------------------------------------------------------

MODEL_PATH = "models"

try:
    diagnosis_model = joblib.load(os.path.join(MODEL_PATH, "best_model.pkl"))
    selector = joblib.load(os.path.join(MODEL_PATH, "selector.pkl"))
    updrs_model = joblib.load(os.path.join(MODEL_PATH, "updrs_regressor.pkl"))
    scaler = joblib.load("scaler.pkl")  # this one is in root
    model_loaded = True
except Exception as e:
    st.error("❌ Failed to load models. Make sure the following files exist:")
    st.code("""
models/best_model.pkl
models/selector.pkl
models/updrs_regressor.pkl
scaler.pkl (root folder)
    """)
    model_loaded = False

if not model_loaded:
    st.stop()

# --------------------------------------------------------------------------------
# USER INPUT FORM
# --------------------------------------------------------------------------------

st.subheader("📋 Input Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 90, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    jitter = st.slider("Jitter (%)", 0.0, 1.0, 0.12)

with col2:
    shimmer = st.slider("Shimmer (%)", 0.0, 1.0, 0.18)
    HNR = st.slider("HNR", 0.0, 40.0, 21.5)
    RPDE = st.slider("RPDE", 0.0, 1.0, 0.45)

with col3:
    DFA = st.slider("DFA", 0.4, 1.0, 0.65)
    PPE = st.slider("PPE", 0.0, 2.0, 0.23)
    spread1 = st.slider("Spread1", -7.0, -1.0, -4.5)

# Convert gender to numeric
gender_num = 1 if gender == "Male" else 0

# Feature vector
input_data = pd.DataFrame([[
    age, gender_num, jitter, shimmer, HNR,
    RPDE, DFA, PPE, spread1
]], columns=[
    "age", "gender", "jitter", "shimmer", "HNR",
    "RPDE", "DFA", "PPE", "spread1"
])

# --------------------------------------------------------------------------------
# PROCESS FEATURES
# --------------------------------------------------------------------------------

scaled_features = scaler.transform(input_data)
selected_features = selector.transform(scaled_features)

# --------------------------------------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------------------------------------

if st.button("🔍 Predict Parkinson's & Severity"):
    diagnosis = diagnosis_model.predict(selected_features)[0]
    probability = diagnosis_model.predict_proba(selected_features)[0][diagnosis]

    updrs = updrs_model.predict(selected_features)[0]

    # Severity mapping
    if updrs <= 32:
        severity = "Minimal"
        color = "green"
    elif updrs <= 58:
        severity = "Mild"
        color = "yellow"
    elif updrs <= 95:
        severity = "Moderate"
        color = "orange"
    else:
        severity = "Severe"
        color = "red"

    # --------------------------------------------------------------------------------
    # OUTPUT BLOCK
    # --------------------------------------------------------------------------------

    st.subheader("📊 Prediction Results")

    # Parkinson Detection
    if diagnosis == 1:
        st.error(f"🧬 Parkinson's Detected (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"✔ No Parkinson's Detected (Confidence: {probability*100:.2f}%)")

    # UPDRS Severity
    st.markdown(f"""
    ### 🩺 UPDRS Severity Prediction  
    - **UPDRS Score:** {updrs:.2f}  
    - **Severity Level:** <span style='color:{color}; font-weight:700;'>{severity}</span>
    """, unsafe_allow_html=True)
