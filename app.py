import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import datetime

# ✅ Load trained model
model_path = os.path.join(os.path.dirname(__file__), "car_price_model.joblib")
model = joblib.load(model_path)

st.title("🚗 Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price")

# --- Inputs ---
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=2000000, value=50000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# --- Derived feature ---
car_age = 2025 - year

# --- Predict ---
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "car_age": car_age,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }])

    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)

    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:,.0f}")
