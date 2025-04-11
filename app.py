import streamlit as st
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("ML Model Predictor")

# Example: Two input fields
feature1 = st.number_input("Enter Feature 1:")
feature2 = st.number_input("Enter Feature 2:")

# Predict on button click
if st.button("Predict"):
    result = model.predict([[feature1, feature2]])
    st.success(f"Prediction: {result[0]}")
