import streamlit as st
import pandas as pd
import joblib
import datetime

# Load trained model
model_path = "car_price_model.pkl"
pipeline = joblib.load(model_path)

st.title("🚗 Car Price Prediction App")

# User inputs
year = st.number_input("Car Manufacturing Year", min_value=1990, max_value=datetime.datetime.now().year, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# Compute car age
current_year = datetime.datetime.now().year
car_age = current_year - year

# Build input data
input_data = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "car_age": car_age
}])

st.write("### Input Data Preview", input_data)

# Prediction
if st.button("Predict Price"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {prediction:,.2f}")
