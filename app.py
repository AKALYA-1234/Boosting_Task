import pickle
import streamlit as st

import sklearn
st.write("scikit-learn version:", sklearn.__version__)
# Load trained model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names if needed
with open("car_price_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("Car Price Prediction App")

# Example input form
input_data = [st.number_input(f"Enter {feature}") for feature in feature_names]

if st.button("Predict Price"):
    prediction = model.predict([input_data])
    st.write(f"Predicted Price: {prediction[0]}")
