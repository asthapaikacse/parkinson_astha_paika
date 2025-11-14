import streamlit as st
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier

# ============================
# Load Models
# ============================

@st.cache_resource
def load_models():
    try:
        parkinson_model = CatBoostClassifier()
        severity_model = CatBoostClassifier()

        parkinson_model.load_model("models/catboost_parkinson.pkl")
        severity_model.load_model("models/catboost_severity.pkl")

        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        features = pickle.load(open("models/selected_features.pkl", "rb"))
        return parkinson_model, severity_model, scaler, features

    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None, None, None


parkinson_model, severity_model, scaler, features = load_models()

if parkinson_model is None:
    st.stop()

# ============================
# UI
# ============================

st.title("🧠 Parkinson's Disease Prediction (ML-Powered)")
st.write("Provide patient details below to get prediction & severity classification.")

# User Inputs
def user_inputs():
    Age = st.number_input("Age", 30, 90, 55)
    BMI = st.number_input("BMI", 10.0, 50.0, 22.5)
    AlcoholConsumption = st.slider("Alcohol Consumption (per week)", 0, 20, 2)
    PhysicalActivity = st.slider("Physical Activity (Hr/week)", 0, 20, 5)
    DietQuality = st.slider("Diet Quality Score", 0, 10, 6)
    SleepQuality = st.slider("Sleep Quality Score", 0, 10, 6)
    SystolicBP = st.number_input("Systolic BP", 90, 200, 120)
    DiastolicBP = st.number_input("Diastolic BP", 50, 120, 80)
    CholesterolTotal = st.number_input("Total Cholesterol", 100, 400, 180)
    CholesterolLDL = st.number_input("LDL Cholesterol", 20, 250, 90)
    CholesterolHDL = st.number_input("HDL Cholesterol", 10, 100, 45)
    CholesterolTriglycerides = st.number_input("Triglycerides", 50, 400, 150)
    MoCA = st.slider("MoCA Cognitive Score", 0, 30, 26)
    FunctionalAssessment = st.slider("Functional Assessment", 0, 100, 70)

    # Hidden fields
    Ethnicity = 0
    EducationLevel = 12

    data = pd.DataFrame([[
        Age, Ethnicity, EducationLevel, BMI, AlcoholConsumption, PhysicalActivity,
        DietQuality, SleepQuality, SystolicBP, DiastolicBP, CholesterolTotal,
        CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, MoCA,
        FunctionalAssessment
    ]], columns=[
        'Age','Ethnicity','EducationLevel','BMI','AlcoholConsumption',
        'PhysicalActivity','DietQuality','SleepQuality','SystolicBP',
        'DiastolicBP','CholesterolTotal','CholesterolLDL','CholesterolHDL',
        'CholesterolTriglycerides','MoCA','FunctionalAssessment'
    ])

    # Drop removed columns
    data = data[features]

    return data


input_data = user_inputs()

# ============================
# Prediction
# ============================

if st.button("Predict"):
    scaled = scaler.transform(input_data)

    pred_prob = parkinson_model.predict_proba(scaled)[0][1]
    pred = int(pred_prob > 0.5)

    if pred == 1:
        st.error(f"⚠️ PARKINSON DETECTED\nConfidence: {pred_prob*100:.1f}%")
    else:
        st.success(f"✅ NO PARKINSON\nConfidence: {(1-pred_prob)*100:.1f}%")

    # Severity model prediction
    sev = int(severity_model.predict(scaled)[0])

    severity_names = ["Minimal", "Mild", "Moderate", "Severe"]
    st.info(f"🩺 **Predicted Severity:** {severity_names[sev]}")
