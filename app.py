import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Load Models and Encoders
# ------------------------------
MODEL_FOLDER = "ModelFiles"

xgb_model = joblib.load(os.path.join(MODEL_FOLDER, "xgb_sales_model.pkl"))
family_encoder = joblib.load(os.path.join(MODEL_FOLDER, "family_encoder.pkl"))
type_encoder = joblib.load(os.path.join(MODEL_FOLDER, "type_encoder.pkl"))
state_encoder = joblib.load(os.path.join(MODEL_FOLDER, "state_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Future Sales Prediction", layout="wide")
st.title("📊 Future Sales Prediction Dashboard")

st.info("This dashboard uses built-in demo data. No CSV upload required for demo purposes.")

# ------------------------------
# Built-in demo dataset
# ------------------------------
data = pd.DataFrame({
    "Family": ["FOODS", "HOBBIES", "HOUSEHOLD", "FOODS", "HOBBIES", "HOUSEHOLD"],
    "ProductType": ["A", "B", "C", "A", "B", "C"],
    "State": ["CA", "TX", "NY", "CA", "TX", "NY"],
    "Sales": [1000, 1500, 1200, 1100, 1400, 1300],
    "StoreSize": [1500, 2000, 1800, 1600, 2100, 1900],
    "OilPrice": [70, 72, 69, 71, 73, 68]
})

st.subheader("Data Preview")
st.dataframe(data)

# ------------------------------
# Safe categorical encoding
# ------------------------------
def safe_transform(encoder, series):
    result = []
    for val in series:
        if val in encoder.classes_:
            result.append(int(encoder.transform([val])[0]))
        else:
            result.append(0)
    return np.array(result)

categorical_columns = {
    "Family": family_encoder,
    "ProductType": type_encoder,
    "State": state_encoder
}

for col, encoder in categorical_columns.items():
    if col in data.columns:
        data[col] = safe_transform(encoder, data[col])

# ------------------------------
# Numeric columns scaling
# ------------------------------
numeric_cols = ["Sales", "StoreSize", "OilPrice"]
for col in numeric_cols:
    if col not in data.columns:
        data[col] = 0
data[numeric_cols] = data[numeric_cols].astype(float)
data[numeric_cols] = scaler.transform(data[numeric_cols])

# ------------------------------
# Predictions
# ------------------------------
data["Predicted_Sales"] = xgb_model.predict(data)

# ------------------------------
# Show Results Table
# ------------------------------
st.subheader("Predicted Sales Results")
st.dataframe(data)

# ------------------------------
# Line Chart: Predicted Sales Trend
# ------------------------------
st.subheader("Predicted Sales Trend")
st.line_chart(data["Predicted_Sales"])

# ------------------------------
# Bar Chart: Actual vs Predicted Sales
# ------------------------------
st.subheader("Actual vs Predicted Sales")
st.bar_chart(data[["Sales","Predicted_Sales"]])

# ------------------------------
# Summary Statistics
# ------------------------------
st.subheader("Summary Statistics of Predicted Sales")
st.write(data["Predicted_Sales"].describe())

# ------------------------------
# Total Predicted Sales
# ------------------------------
st.subheader("Total Predicted Sales")
st.success(data["Predicted_Sales"].sum())
