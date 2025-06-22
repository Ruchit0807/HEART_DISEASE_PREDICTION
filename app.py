import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("💓 Heart Disease Prediction App by - Ruchit")
st.markdown("This app predicts the chances of getting Heart Attack on the basis of information provided by you.")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    
    raw_input = {
        "Age": age,
        "RESTING_BP": resting_bp,
        "cholesterol": cholesterol,
        "max_heart_rate": max_heart_rate,
        f"ST_SLOPE_{slope}": 1,
        f"Fasting_Blood_Sugar_{fasting_blood_sugar}": 1,
        f"Sex_{sex}": 1,
        f"Chest_Pain_Type_{chest_pain}": 1,
        f"Resting_ECG_{resting_ecg}": 1,
        f"Exercise_Angina_{exercise_angina}": 1,
    }

    
    input_data = pd.DataFrame([raw_input])

   
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    
    input_data = input_data[expected_columns]

    
    scaled_input = scaler.transform(input_data)

    
    prediction = model.predict(scaled_input)[0]

    
    if prediction == 1:
        st.error("🔴 The model predicts that you **may have heart disease**. Please consult a healthcare professional.")
    else:
        st.success("🟢 The model predicts that you **are unlikely to have heart disease**. Still, consider regular check-ups.")
