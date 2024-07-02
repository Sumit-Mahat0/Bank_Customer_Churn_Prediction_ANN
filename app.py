import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load the scaler and the trained model
scaler = joblib.load("bank_scaler.pkl")
nn_model = tf.keras.models.load_model("bank_churn.h5")

def preprocess_input(data):
    # Convert data to DataFrame for consistency
    df = pd.DataFrame([data])
    # Scaling the data
    scaled_data = scaler.transform(df)
    return scaled_data

st.title("Bank Customer Churn Prediction")

# Collect user inputs
CreditScore = st.number_input("Credit Score", value=622.0)
Age = st.number_input("Age", value=45)
Tenure = st.selectbox("Tenure", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Balance = st.number_input("Balance", value=107073.60)
NumOfProducts = st.selectbox("Number of Products", [0, 1, 2, 3, 4])
HasCrCard = st.radio("Has Credit Card", [0, 1])
IsActiveMember = st.radio("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", value=30984.40)
Geography_Germany = st.radio("Geography (Germany)", [0, 1])
Geography_Spain = st.radio("Geography (Spain)", [0, 1])
Gender_Male = st.radio("Gender (Male)", [0, 1])

# Make prediction
if st.button("Predict"):
    # Create dictionary from user inputs
    user_data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography_Germany': Geography_Germany,
        'Geography_Spain': Geography_Spain,
        'Gender_Male': Gender_Male
    }
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = nn_model.predict(processed_data)
    prediction = prediction.flatten()  # Flatten the array to get a single value
    
    # Display prediction result
    if prediction[0] > 0.5:  # Check the first element of the prediction
        st.write("The customer is likely to churn")
    else:
        st.write("The customer is likely to stay")
