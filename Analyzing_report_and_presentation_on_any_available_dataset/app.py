import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("LogReg_heart.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction")
st.markdown("Provide the following details to check your heart stroke risk:")

# Collect user input
age = st.slider("Age", 15, 100, 50)
gender = st.selectbox("Gender", ["M", "F"])

if gender == "M":
    val_gender = 1
elif gender == "F":
    val_gender = 0
    
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])

if chest_pain == "TA":
    val_CP = 0
elif chest_pain == "ATA":
    val_CP = 1
elif chest_pain == "NAP":
    val_CP = 2
elif chest_pain == "ASY":
    val_CP = 3
    
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 300, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 800, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

if resting_ecg == "Normal":
    val_ecg = 0
elif resting_ecg == "ST":
    val_ecg = 1
elif resting_ecg == "LVH":
    val_ecg = 2

max_hr = st.slider("Max Heart Rate", 60, 300, 120)
exercise_angina = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 7.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

if st_slope == "Upsloping":
    val_slope = 1
elif st_slope == "Flat":
    val_slope = 2
elif st_slope == "Downsloping":
    val_slope = 3
    
vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    
    # When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'age': age,
        'gender':val_gender,
        'chestpain':val_CP,
        'restingBP': resting_bp,
        'serumcholestrol': cholesterol,
        'fastingbloodsugar': fasting_bs,
        'restingrelectro': val_ecg,
        'maxheartrate': max_hr,
        'exerciseangia': exercise_angina,
        'oldpeak': oldpeak,
        'slope': val_slope,
        'noofmajorvessels': vessels,
    }
    
    # Create input dataframe
    input_df = pd.DataFrame([raw_input])
    
     # Make prediction
    prediction = model.predict(input_df)[0]

    # Show result
    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")