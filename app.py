import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

import pickle
import numpy as np
import pandas as pd
import os

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    try:
        with open("best_xgb_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎓 Student Predictor")
    st.markdown("Created by: *Olalekan Ayinde*")
    st.markdown("---")
    st.write("Enter student details to predict academic performance.")

# ---------------- Main UI ----------------
st.title("📊 Student Performance Prediction App")
st.markdown(
    "This app predicts a student's academic performance using a trained machine learning model."
)

# ---------------- Input Section ----------------
st.subheader("📝 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    study_time = st.number_input("Study Time (hours per week)", min_value=0.0, max_value=50.0, value=10.0)
    absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=2)
    failures = st.number_input("Past Class Failures", min_value=0, max_value=5, value=0)

with col2:
    g1 = st.number_input("First Term Score (G1)", min_value=0, max_value=100, value=60)
    g2 = st.number_input("Second Term Score (G2)", min_value=0, max_value=100, value=65)

# ---------------- Prediction ----------------
if st.button("🚀 Predict Performance"):
    try:
        input_data = np.array([[study_time, absences, failures, g1, g2]])
        prediction = model.predict(input_data)

        st.success("✅ Prediction Successful")
        st.subheader("📈 Predicted Final Score")
        st.metric(label="Expected Performance", value=round(float(prediction[0]), 2))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("⚙️ Powered by DeepTech | DSN | Streamlit App")
