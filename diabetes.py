import os
import json
import time
import uuid
import pickle
import logging
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)


# ---------------- LOGGING SETUP ----------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("diabetes_app")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def write_log(event_type: str, payload: dict) -> None:
    """Write structured JSON logs for monitoring and troubleshooting."""
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload
    }

    logger.info(json.dumps(log_record))


# ---------------- SESSION SETUP ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    with open("diabetes.pkl", "rb") as model_file:
        return pickle.load(model_file)


try:
    model = load_model()

except Exception as error:
    write_log(
        "model_load_failed",
        {
            "session_id": st.session_state.session_id,
            "error": str(error)
        }
    )

    st.error("Model could not be loaded. Please contact the administrator.")
    st.stop()


# ---------------- ENCODING MAPS ----------------
# Important:
# These values must be the same values used during model training.
gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_history_map = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}


# ---------------- APPLICATION UI ----------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below to predict diabetes risk.")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", list(gender_map.keys()))

    age = st.number_input(
        "Age",
        min_value=1.0,
        max_value=120.0,
        value=30.0,
        step=1.0
    )

    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No"
    )

    smoking_history = st.selectbox(
        "Smoking History",
        list(smoking_history_map.keys())
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5,
        step=0.1
    )

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=400,
        value=100,
        step=1
    )

    submit_button = st.form_submit_button("Predict")


# ---------------- PREDICTION LOGIC ----------------
if submit_button:
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        input_data = pd.DataFrame(
            [
                {
                    "gender": gender_map[gender],
                    "age": age,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "smoking_history": smoking_history_map[smoking_history],
                    "bmi": bmi,
                    "HbA1c_level": hba1c,
                    "blood_glucose_level": blood_glucose
                }
            ]
        )

        prediction = int(model.predict(input_data)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Do not log individual medical input values in production.
        write_log(
            "prediction_success",
            {
                "session_id": st.session_state.session_id,
                "request_id": request_id,
                "prediction": prediction,
                "probability": probability,
                "latency_ms": latency_ms
            }
        )

        st.divider()

        if prediction == 1:
            st.error("Prediction: Diabetic")
        else:
            st.success("Prediction: Not Diabetic")

        if probability is not None:
            st.metric(
                label="Probability of Diabetes",
                value=f"{probability:.2%}"
            )

        st.write("### Encoded Input Data Used for Prediction")
        st.dataframe(input_data, use_container_width=True)

        st.caption(f"Prediction completed in {latency_ms} ms")

    except Exception as error:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        write_log(
            "prediction_failed",
            {
                "session_id": st.session_state.session_id,
                "request_id": request_id,
                "error": str(error),
                "latency_ms": latency_ms
            }
        )

        st.error("Prediction failed. Please verify the input data and model configuration.")
