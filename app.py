# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("car_price_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🚗 Car Price Prediction App")

year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=500)
fuel = st.selectbox("Fuel Type", encoders["fuel"].classes_)
seller_type = st.selectbox("Seller Type", encoders["seller_type"].classes_)
transmission = st.selectbox("Transmission", encoders["transmission"].classes_)
owner = st.selectbox("Owner", encoders["owner"].classes_)
name = st.selectbox("Car Name", encoders["name"].classes_)

if st.button("🔮 Predict Car Price"):
    # Encode categorical values using saved encoders
    input_df = pd.DataFrame([{
        "name": encoders["name"].transform([name])[0],
        "year": year,
        "km_driven": km_driven,
        "fuel": encoders["fuel"].transform([fuel])[0],
        "seller_type": encoders["seller_type"].transform([seller_type])[0],
        "transmission": encoders["transmission"].transform([transmission])[0],
        "owner": encoders["owner"].transform([owner])[0]
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Selling Price: **₹{prediction:,.0f}**")
