# ============================================
# CAR PRICE PREDICTION - NEXT LEVEL DASHBOARD
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.express as px

# ============================================
# PAGE CONFIG (DARK THEME)
# ============================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "model/car_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Run train_model.py first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# ============================================
# CUSTOM DARK STYLE
# ============================================

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown("""
# 🚗 Car Price Prediction
### AI-powered resale value estimator
""")

st.markdown("---")

# ============================================
# SIDEBAR INPUT
# ============================================

st.sidebar.header("Car Details")

present_price = st.sidebar.number_input("Showroom Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 30000)
owners = st.sidebar.selectbox("Owners", [0, 1, 2, 3])
car_age = st.sidebar.slider("Car Age", 0, 20, 5)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# ============================================
# ENCODING
# ============================================

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

fuel = fuel_map[fuel]
seller = seller_map[seller]
transmission = trans_map[transmission]

# ============================================
# PREDICTION + CONFIDENCE
# ============================================

st.subheader("Prediction")

if st.button("Predict Price"):

    input_data = np.array([[present_price, kms_driven, owners,
                            fuel, seller, transmission, car_age]])

    prediction = model.predict(input_data)[0]

    # Confidence (approx using tree variance)
    preds = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    confidence = 100 - (np.std(preds) * 10)

    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")
    st.info(f"Model Confidence: {confidence:.2f}%")

# ============================================
# DATA VISUALIZATION (PLOTLY)
# ============================================

st.markdown("---")
st.subheader("Market Insights")

DATA_PATH = "car_data.csv"

if os.path.exists(DATA_PATH):

    df = pd.read_csv(DATA_PATH)

    col1, col2 = st.columns(2)

    # Interactive histogram
    with col1:
        fig = px.histogram(df, x="Selling_Price", title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Interactive scatter
    with col2:
        fig = px.scatter(df, x="Kms_Driven", y="Selling_Price",
                         title="Price vs Distance")
        st.plotly_chart(fig, use_container_width=True)

    # Fuel analysis
    st.subheader("Fuel Type Analysis")

    fuel_avg = df.groupby("Fuel_Type")["Selling_Price"].mean().reset_index()

    fig = px.bar(fuel_avg, x="Fuel_Type", y="Selling_Price",
                 title="Average Price by Fuel Type")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Dataset not found")

# ============================================
# PORTFOLIO FOOTER
# ============================================

st.markdown("---")
st.markdown("""
### About This Project
- Machine Learning model using Random Forest
- Hyperparameter tuning applied
- Interactive dashboard with Plotly
- Built for real-world resale price prediction

**Created by:** Your Name  
""")