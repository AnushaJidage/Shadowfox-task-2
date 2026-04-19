import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Car Selling Price Prediction")

# Inputs
present_price = st.number_input("Showroom Price (in lakhs)")
kms_driven = st.number_input("Kilometers Driven")
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
year = st.number_input("Year of Purchase", 1990, 2026)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Feature engineering
car_age = 2026 - year

# Encoding (must match training)
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0

seller_individual = 1 if seller_type == "Individual" else 0

trans_manual = 1 if transmission == "Manual" else 0

# Prediction
if st.button("Predict Price"):
    features = np.array([[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        trans_manual
    ]])

    prediction = model.predict(features)

    st.success(f"Estimated Selling Price: ₹ {round(prediction[0], 2)} Lakhs")
