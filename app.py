import streamlit as st
import pickle
import pandas as pd

import pickle
import sklearn
print(sklearn.__version__)

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully")

with open("car_price_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("🚗 Car Price Prediction App")

# Inputs
name = st.text_input("Car Name (e.g. Maruti Swift Dzire)")
year = st.number_input("Year of Purchase", min_value=1990, max_value=2024, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# Convert year to car_age
car_age = 2024 - year

# Prepare input data as DataFrame
input_dict = {
    "name": [name],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "car_age": [car_age]
}

input_df = pd.DataFrame(input_dict)

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {prediction:,.2f}")
