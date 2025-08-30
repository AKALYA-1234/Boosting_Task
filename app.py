import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime

# Load model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🚗 Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details below to predict its selling price.")

# User inputs
year = st.number_input("Year of Purchase", min_value=1990, max_value=datetime.datetime.now().year, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# ✅ Compute car_age immediately after year is selected
car_age = datetime.datetime.now().year - year

# Predict button
if st.button("Predict Price"):
    # Create dataframe with same columns as training
    input_df = pd.DataFrame([{
        'car_age': car_age,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }])

    # Make prediction
    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)   # reverse log transform
    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:,.0f}")
