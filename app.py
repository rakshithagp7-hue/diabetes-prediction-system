import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

saved = load_model()
model, scaler, medians = saved["model"], saved["scaler"], saved["medians"]
features = saved["features"]

st.title("🩺 AI-Based Diabetes Prediction System")
st.write("Enter patient details to estimate the likelihood of diabetes.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 21, 100, 35)
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose level (mg/dL)", 0, 250, 120)
    blood_pressure = st.number_input("Blood pressure (diastolic, mm Hg)", 0, 150, 70)
with col2:
    bmi = st.number_input("BMI", 0.0, 70.0, 27.0, step=0.1)
    insulin = st.number_input("Insulin (mu U/ml), 0 if unknown", 0, 900, 0)
    skin = st.number_input("Skin thickness (mm), 0 if unknown", 0, 100, 0)
    dpf = st.number_input("Diabetes pedigree function", 0.0, 2.5, 0.4, step=0.01)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])[features]

    # Same preprocessing as training: 0 means missing, fill with median
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    row[zero_cols] = row[zero_cols].replace(0, np.nan)
    row = row.fillna(medians)

    scaled = scaler.transform(row)
    prob = model.predict_proba(scaled)[0, 1]
    label = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

    st.subheader("Result")
    if label == "Diabetic":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

    st.metric("Diabetes risk", f"{prob * 100:.1f}%")
    st.progress(float(prob))

    st.caption("This tool is for educational purposes only and is not a "
               "medical diagnosis. Please consult a healthcare professional.")

with st.expander("About the model"):
    st.write("Random Forest classifier trained on the Pima Indians Diabetes "
             "dataset. Test ROC-AUC: 0.827, recall: 0.74.")