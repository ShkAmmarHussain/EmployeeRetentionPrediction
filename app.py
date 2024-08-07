import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

# Load the model
model = xgb.Booster()
model.load_model('xgboost_model.json')

# Define the prediction function
def predict(features):
    feature_names = ['satisfaction_level', 'last_evaluation', 'number_project', 
                     'average_montly_hours', 'time_spend_company', 'Work_accident', 
                     'promotion_last_5years', 'low', 'medium']
    df = pd.DataFrame([features], columns=feature_names)
    dmatrix = xgb.DMatrix(df)
    prediction = model.predict(dmatrix)
    return prediction[0]

# Streamlit app
st.title("Employee Attrition Prediction")

# Input features
st.header("Input Features")
satisfaction_level = st.number_input("Satisfaction Level", value=0.38)
last_evaluation = st.number_input("Last Evaluation", value=0.53)
number_project = st.number_input("Number of Projects", value=2)
average_montly_hours = st.number_input("Average Monthly Hours", value=157)
time_spend_company = st.number_input("Time Spent in Company (years)", value=3)
work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion Last 5 Years", [0, 1])
low = st.checkbox("Low Salary", value=True)
medium = st.checkbox("Medium Salary", value=False)

# Convert salary feature to categorical value
salary = 0  # default to 'high'
if low:
    salary = 1
elif medium:
    salary = 2

# Prepare the features
features = [satisfaction_level, last_evaluation, number_project, average_montly_hours,
            time_spend_company, work_accident, promotion_last_5years, low, medium]

# Make prediction
if st.button("Predict"):
    prediction = predict(features)
    st.write(f"Prediction Score: {prediction:.2f}")

# Run the app using the command: streamlit run app.py
