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
# Streamlit Dashboard UI
# ------------------------------
st.set_page_config(page_title="Future Sales Prediction", layout="wide")
st.title("📊 Future Sales Prediction Dashboard")

# ------------------------------
# File uploader
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File (Task 2 dataset, e.g., train.csv) or leave empty for demo",
    type=["csv"]
)

# ------------------------------
# Built-in demo data (if no file uploaded)
# ------------------------------
if uploaded_file is None:
    st.info("No CSV uploaded. Using built-in demo data for presentation.")
    data = pd.DataFrame({
        "Family": ["FOODS", "HOBBIES", "HOUSEHOLD"],
        "ProductType": ["A", "B", "C"],
        "State": ["CA", "TX", "NY"],
        "Sales": [1000, 1500, 1200],
        "StoreSize": [1500, 2000, 1800],
        "OilPrice": [70, 72, 69]
    })
else:
    data = pd.read_csv(uploaded_file)

st.subheader("Data Preview")
st.dataframe(data.head())

# ------------------------------
# Safe categorical encoding
# ------------------------------
def safe_transform(encoder, series):
    """Encodes known categories, unseen categories mapped to 0"""
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
numeric_cols = ["Sales", "StoreSize", "OilPrice"]  # Update as per your training

for col in numeric_cols:
    if col not in data.columns:
        data[col] = 0

data[numeric_cols] = data[numeric_cols].astype(float)
data[numeric_cols] = scaler.transform(data[numeric_cols])

# ------------------------------
# Make predictions
# ------------------------------
data["Predicted_Sales"] = xgb_model.predict(data)

# ------------------------------
# Show results
# ------------------------------
st.subheader("Predicted Sales Results")
st.dataframe(data)

st.subheader("Predicted Sales Trend")
st.line_chart(data["Predicted_Sales"])

st.subheader("Summary Statistics of Predicted Sales")
st.write(data["Predicted_Sales"].describe())

st.subheader("Total Predicted Sales")
st.success(data["Predicted_Sales"].sum())
