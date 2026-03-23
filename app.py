import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

st.set_page_config(page_title="PulseGuard", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("models/hypertension_model.pkl")
    scaler = joblib.load("models/hypertension_scaler.pkl")
    return model, scaler

model, scaler = load_model()

if "page" not in st.session_state:
    st.session_state.page = "main"

# ------------------ RESULT PAGE ------------------ #
if st.session_state.page == "result":

    risk = st.session_state.risk_level

    if risk == "Low":
        with open("pages/low.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=700)

    elif risk == "Moderate":
        with open("pages/moderate.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=700)

    else:
        with open("pages/high.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=700)

    if st.button("⬅ Back"):
        st.session_state.page = "main"
        st.rerun()

# ------------------ MAIN PAGE ------------------ #
else:

    st.title("❤️ PulseGuard - Hypertension Risk Predictor")
    st.header("Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])

        salt_habit = st.selectbox(
            "Salt Consumption",
            ["Low", "Moderate", "High"]
        )

        stress_level = st.selectbox(
            "Stress Level",
            ["Low", "Moderate", "High"]
        )

        sleep_duration = st.number_input("Sleep Duration (hours)", 3.0, 12.0, 7.0)

    with col2:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)

        height_unit = st.selectbox("Height Unit", ["Centimeters", "Meters", "Feet"])

        if height_unit == "Centimeters":
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
            height_m = height / 100
        elif height_unit == "Meters":
            height = st.number_input("Height (m)", 1.0, 2.5, 1.7)
            height_m = height
        else:
            height = st.number_input("Height (feet)", 3.0, 8.0, 5.5)
            height_m = height * 0.3048

        exercise = st.selectbox("Exercise Level", ["Never", "Rarely", "Daily"])
        bp_history = st.selectbox("BP History", ["No", "Yes"])
        medication = st.selectbox("Medication", ["No", "Yes"])
        family_history = st.selectbox("Family History", ["No", "Yes"])
        smoking_status = st.selectbox("Smoking", ["No", "Yes"])

    bmi = weight / (height_m ** 2)
    st.subheader(f"Calculated BMI: {round(bmi, 2)}")

    if st.button("Predict Risk"):

        salt_intake = 5 if salt_habit == "Low" else 8 if salt_habit == "Moderate" else 12
        stress_score = 3 if stress_level == "Low" else 6 if stress_level == "Moderate" else 9
        exercise_level = 0 if exercise == "Never" else 1 if exercise == "Rarely" else 2

        bp_history = 1 if bp_history == "Yes" else 0
        medication = 1 if medication == "Yes" else 0
        family_history = 1 if family_history == "Yes" else 0
        smoking_status = 1 if smoking_status == "Yes" else 0

        input_data = pd.DataFrame([[
            age,
            salt_intake,
            stress_score,
            bp_history,
            sleep_duration,
            bmi,
            medication,
            family_history,
            exercise_level,
            smoking_status
        ]], columns=[
            "age",
            "salt_intake",
            "stress_score",
            "bp_history",
            "sleep_duration",
            "bmi",
            "medication",
            "family_history",
            "exercise_level",
            "smoking_status"
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if probability < 40:
            st.session_state.risk_level = "Low"
        elif probability < 70:
            st.session_state.risk_level = "Moderate"
        else:
            st.session_state.risk_level = "High"

        st.session_state.page = "result"
        st.rerun()