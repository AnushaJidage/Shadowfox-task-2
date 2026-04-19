import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ML Trainer App", layout="wide")

st.title("📊 Machine Learning Trainer Dashboard")


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

st.success("✅ Data loaded successfully!")
st.write("### Preview of Data")
st.dataframe(df.head())


# -----------------------------
# EDA SECTION
# -----------------------------
st.write("## 📊 Exploratory Data Analysis")

# Numeric columns only (prevents crash)
numeric_df = df.select_dtypes(include=['number'])

if not numeric_df.empty:
    st.write("### Correlation Heatmap")
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Histograms")
    for col in numeric_df.columns:
        fig = px.histogram(df, x=col, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric columns available for EDA.")


# -----------------------------
# TARGET COLUMN SELECTION
# -----------------------------
st.write("## 🎯 Select Target Column")

target_column = st.selectbox(
    "Choose the column you want to predict",
    df.columns
)

if target_column is None:
    st.stop()


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_data(df, target_column):
    df = df.copy()

    # Drop missing target
    df = df.dropna(subset=[target_column])

    # Fill numeric NaN
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# -----------------------------
# TRAIN BUTTON
# -----------------------------
if st.button("🚀 Train Model"):

    df_processed, encoders = preprocess_data(df, target_column)

    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    if X.shape[1] == 0:
        st.error("❌ No features available for training.")
        st.stop()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success("✅ Model trained successfully!")

    st.write("### 📈 Model Performance")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(model, "outputs/model.pkl")
    joblib.dump(encoders, "outputs/encoders.pkl")

    st.info("💾 Model saved to outputs/model.pkl")

    # -----------------------------
    # DOWNLOAD BUTTONS
    # -----------------------------
    with open("outputs/model.pkl", "rb") as f:
        st.download_button(
            label="⬇️ Download Model",
            data=f,
            file_name="model.pkl"
        )

    with open("outputs/encoders.pkl", "rb") as f:
        st.download_button(
            label="⬇️ Download Encoders",
            data=f,
            file_name="encoders.pkl"
        )
