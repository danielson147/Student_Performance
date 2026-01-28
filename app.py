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
        # ✅ Best practice: ensure xgboost is available
        import xgboost  # required to unpickle the model

        with open("best_xgb_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model

    except ModuleNotFoundError:
        st.error(
            "❌ XGBoost is not installed.\n\n"
            "Please add **xgboost** to your requirements.txt file "
            "before deploying this app."
        )
        return None

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


model = load_model()

if model is None:
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    # ✅ DeepTech logo
    if os.path.exists("deeptech_logo.png"):
        st.image("deeptech_logo.png", width=150)

    st.title("🎓 Student Performance Predictor")
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

# ---------------- Input Section ----------------
st.subheader("📝 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=93,
        value=2
    )

    guardian_label = st.selectbox(
        "Student Guardian",
        options=["mother", "father", "other"]
    )

    goout = st.slider(
        "Going Out With Friends",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    g1 = st.slider(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    g2 = st.slider(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=11
    )

# ---------------- Prediction ----------------
if st.button("🚀 Predict Performance"):
    try:
        input_data = np.array([[study_time, absences, failures, g1, g2]])
        prediction = model.predict(input_data)

        st.success("✅ Prediction Successful")
        st.subheader("📈 Predicted Final Score")
        st.metric(
            label="Expected Performance",
            value=round(float(prediction[0]), 2)
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("⚙️ Powered by DeepTech | DSN | Streamlit App")


