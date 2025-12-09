import streamlit as st
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction App")

tenure = st.number_input("Tenure (Months)")
contract = st.selectbox("Contract Type", [0,1,2])
payment = st.selectbox("Payment Method", [0,1,2,3])

if st.button("Predict"):
    input_data = np.array([[tenure, contract, payment]])
    pred = model.predict(input_data)
    st.write("Churn" if pred == 1 else "Not Churn")
