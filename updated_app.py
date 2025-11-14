import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Parkinson's Disease Prediction System")
st.write("Upload patient details to check Parkinson's disease probability and severity.")

# -----------------------------
# Load Models Safely
# -----------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except:
        return None

models = {
    "diagnosis": safe_load("models/voting_ensemble_model.pkl"),
    "severity": safe_load("models/severity_classifier.pkl"),
    "scaler": safe_load("models/scaler.pkl"),
    "features": safe_load("models/selected_features.pkl"),
}

# Error if anything missing
if None in models.values():
    st.error("❌ Failed to load models. Please ensure all model files exist in /models folder.")
    st.stop()

diagnosis_model = models["diagnosis"]
severity_model = models["severity"]
scaler = models["scaler"]
selected_features = models["features"]

# -----------------------------
# UI Input Fields (Clean UI)
# Ethnicity + EducationLevel removed from UI
# -----------------------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=120, value=45)
    BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
    Alcohol = st.slider("Alcohol Consumption (units/week)", 0, 50, 2)
    PhysicalActivity = st.slider("Physical Activity (hrs/week)", 0, 20, 3)

with col2:
    DietQuality = st.slider("Diet Quality (1-10)", 1, 10, 7)
    SleepQuality = st.slider("Sleep Quality (1-10)", 1, 10, 6)
    SBP = st.number_input("Systolic BP", 80, 200, 120)
    DBP = st.number_input("Diastolic BP", 50, 120, 80)

col3, col4 = st.columns(2)

with col3:
    CholTotal = st.number_input("Total Cholesterol", 100, 400, 180)
    CholLDL = st.number_input("LDL Cholesterol", 50, 250, 100)
    CholHDL = st.number_input("HDL Cholesterol", 20, 100, 45)

with col4:
    Triglycerides = st.number_input("Triglycerides", 50, 600, 150)
    MoCA = st.slider("MoCA Score (0-30)", 0, 30, 26)
    FunctionalAssessment = st.slider("Functional Assessment (0-100)", 0, 100, 85)

# -----------------------------
# Internally Fill Missing UI Fields
# to match training feature order
# -----------------------------
Ethnicity = 0              # neutral category
EducationLevel = 12        # approximate school+college (neutral)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_dict = {
    "Age": Age,
    "Ethnicity": Ethnicity,
    "EducationLevel": EducationLevel,
    "BMI": BMI,
    "AlcoholConsumption": Alcohol,
    "PhysicalActivity": PhysicalActivity,
    "DietQuality": DietQuality,
    "SleepQuality": SleepQuality,
    "SystolicBP": SBP,
    "DiastolicBP": DBP,
    "CholesterolTotal": CholTotal,
    "CholesterolLDL": CholLDL,
    "CholesterolHDL": CholHDL,
    "CholesterolTriglycerides": Triglycerides,
    "MoCA": MoCA,
    "FunctionalAssessment": FunctionalAssessment
}

input_df = pd.DataFrame([input_dict])

# Select required features
input_df = input_df[selected_features]

# Scale input
scaled_input = scaler.transform(input_df)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Run Parkinson's Prediction"):
    
    # --- Diagnosis Prediction ---
    prob = diagnosis_model.predict_proba(scaled_input)[0][1]
    diagnosis = 1 if prob >= 0.5 else 0

    # --- Severity Prediction (Only if Disease = YES) ---
    if diagnosis == 1:
        severity_class = severity_model.predict(scaled_input)[0]
        severity_map = {
            0: "Mild",
            1: "Moderate",
            2: "Severe",
            3: "Extreme"
        }
        severity_label = severity_map.get(severity_class, "Unknown")
    else:
        severity_label = "Not Applicable"

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("🩺 Prediction Result")

    if diagnosis == 1:
        st.error(f"⚠️ PARKINSON'S DISEASE DETECTED")
        st.write(f"**Confidence:** {prob*100:.2f}%")
        st.write(f"**Predicted Severity (ML-based):** {severity_label}")
    else:
        st.success("✅ No Parkinson's Detected")
        st.write(f"**Confidence:** {(1 - prob)*100:.2f}%")

