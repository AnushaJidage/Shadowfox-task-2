import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =========================
# Load Dataset
# =========================

file_path = "car_data.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found!")

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully ✅")

# =========================
# Data Preprocessing
# =========================

# Create new feature
df['Car_Age'] = 2026 - df['Year']

# Drop unwanted columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model
# =========================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully ✅")

# =========================
# Save Model
# =========================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl ✅")