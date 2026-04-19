import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# EDA FUNCTION
# -----------------------------
def perform_eda(df):
    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    os.makedirs("outputs", exist_ok=True)

    # Safe correlation
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("outputs/heatmap.png")
        plt.close()
        print("Saved: outputs/heatmap.png")

    # Histograms
    if not numeric_df.empty:
        numeric_df.hist(figsize=(10, 8), bins=20)
        plt.tight_layout()
        plt.savefig("outputs/histograms.png")
        plt.close()
        print("Saved: outputs/histograms.png")


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_data(df, target_column):
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Fill numeric NaN
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== MODEL PERFORMANCE =====")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    return model


# -----------------------------
# SAFE FILE LOADER
# -----------------------------
def load_dataset(file_path):
    if os.path.exists(file_path):
        print(f"Loading file: {file_path}")
        return pd.read_csv(file_path)

    print(f"\n❌ File not found: {file_path}")

    # Show available CSV files
    files = [f for f in os.listdir() if f.endswith(".csv")]

    if not files:
        print("No CSV files found in this folder.")
        exit()

    print("\nAvailable CSV files:")
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")

    # Auto-pick first file
    print("\n👉 Using first available file:", files[0])
    return pd.read_csv(files[0])


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Current working directory:", os.getcwd())

    # 🔴 CHANGE THIS if needed
    file_path = "car_data.csv"

    df = load_dataset(file_path)

    print("\nData loaded successfully!")
    print("Columns:", df.columns.tolist())

    # 🔴 AUTO target selection (last column)
    target_column = df.columns[-1]
    print("Using target column:", target_column)

    # EDA
    perform_eda(df)

    # Preprocess
    df_processed, encoders = preprocess_data(df, target_column)

    # Train
    model = train_model(df_processed, target_column)

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(model, "outputs/model.pkl")
    joblib.dump(encoders, "outputs/encoders.pkl")

    print("\n✅ Saved:")
    print("outputs/model.pkl")
    print("outputs/encoders.pkl")


# -----------------------------
if __name__ == "__main__":
    main()