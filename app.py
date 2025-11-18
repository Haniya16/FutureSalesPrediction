import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load models and encoders
MODEL_FOLDER = "ModelFiles"
xgb_model = joblib.load(os.path.join(MODEL_FOLDER, "xgb_sales_model.pkl"))
family_encoder = joblib.load(os.path.join(MODEL_FOLDER, "family_encoder.pkl"))
type_encoder = joblib.load(os.path.join(MODEL_FOLDER, "type_encoder.pkl"))
state_encoder = joblib.load(os.path.join(MODEL_FOLDER, "state_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))

st.title("📊 Future Sales Prediction Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File (optional for demo)", type=["csv"])

# --- If no file uploaded, use small sample automatically
if uploaded_file is None:
    st.info("No CSV uploaded. Using built-in demo data for presentation.")
    data = pd.DataFrame({
        "Family": ["TECH", "FOODS", "HOUSEHOLD"],
        "ProductType": ["A", "B", "C"],
        "State": ["CA", "TX", "NY"],
        "Sales": [1000, 1500, 1200],
        "StoreSize": [1500, 2000, 1800],
        "OilPrice": [70, 72, 69]
    })
else:
    data = pd.read_csv(uploaded_file)

# Encode categorical columns
for col, encoder in zip(["Family", "ProductType", "State"],
                        [family_encoder, type_encoder, state_encoder]):
    if col in data.columns:
        data[col] = encoder.transform(data[col])

# Scale numeric columns
numeric_cols = ["Sales", "StoreSize", "OilPrice"]  # change as per your training
for col in numeric_cols:
    if col not in data.columns:
        data[col] = 0
data[numeric_cols] = data[numeric_cols].astype(float)
data[numeric_cols] = scaler.transform(data[numeric_cols])

# Predict
data["Predicted_Sales"] = xgb_model.predict(data)

# Show results
st.subheader("Prediction Results")
st.dataframe(data)

st.subheader("Predicted Sales Trend")
st.line_chart(data["Predicted_Sales"])

st.subheader("Summary Statistics")
st.write(data["Predicted_Sales"].describe())

st.subheader("Total Predicted Sales")
st.success(data["Predicted_Sales"].sum())
