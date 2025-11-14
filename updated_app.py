import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Parkinson's Disease Prediction App")
st.write("Upload patient features to predict: **Parkinson’s YES/NO**, **Confidence**, **UPDRS**, and **Severity Level**.")

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource
def load_all_models():
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    best_model = joblib.load("best_model.pkl")
    updrs_model = joblib.load("UPDRS_regressor.pkl")
    return scaler, selector, best_model, updrs_model

try:
    scaler, selector, best_model, updrs_model = load_all_models()
except Exception as e:
    st.error("❌ Failed to load models. Make sure all model files are uploaded.")
    st.stop()

# ------------------------------
# SEVERITY LOGIC (ML-based)
# ------------------------------
def updrs_to_severity(updrs):
    if updrs <= 32:
        return "Minimal (0–32)"
    elif updrs <= 58:
        return "Mild (33–58)"
    elif updrs <= 95:
        return "Moderate (59–95)"
    else:
        return "Severe (96–199)"

# ------------------------------
# USER INPUT FORM
# ------------------------------
st.subheader("📥 Enter Patient Feature Values")

with st.form("prediction_form"):
    values = st.text_area(
        "Enter comma-separated 750 feature values (same order as training dataset):",
        placeholder="0.128, 0.334, 2.556, ..."
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Convert to numpy array
        raw_vals = np.array([float(x.strip()) for x in values.split(",")])

        if raw_vals.shape[0] != 750:
            st.error(f"❌ Expected **750 features**, but got {raw_vals.shape[0]}.")
            st.stop()

        user_input = raw_vals.reshape(1, -1)

        # ------------------------------
        # APPLY SCALER + FEATURE SELECTOR
        # ------------------------------
        scaled = scaler.transform(user_input)
        selected = selector.transform(scaled)

        # ------------------------------
        # PARKINSON YES/NO PREDICTION
        # ------------------------------
        pred = best_model.predict(selected)[0]
        prob = best_model.predict_proba(selected)[0]

        confidence = round(max(prob) * 100, 2)

        # ------------------------------
        # UPDRS REGRESSION (ML)
        # ------------------------------
        updrs_score = float(updrs_model.predict(selected)[0])
        updrs_score = round(updrs_score, 2)

        severity = updrs_to_severity(updrs_score)

        # ------------------------------
        # SHOW RESULTS
        # ------------------------------
        st.subheader("🔍 Prediction Results")

        if pred == 1:
            st.error(f"🧠 Parkinson’s Detected")
        else:
            st.success("🟢 No Parkinson’s Detected")

        st.write(f"### 🔹 Confidence: **{confidence}%**")
        st.write(f"### 🔹 UPDRS Score (ML Predicted): **{updrs_score}**")
        st.write(f"### 🔹 Severity Level: **{severity}**")

    except Exception as e:
        st.error(f"❌ Error processing input: {str(e)}")
