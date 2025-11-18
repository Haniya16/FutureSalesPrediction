import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Load Model and Encoders
# ------------------------------

model_folder = "ModelFiles"   # folder inside GitHub repo

xgb_model = joblib.load(os.path.join(model_folder, "xgb_sales_model.pkl"))
family_encoder = joblib.load(os.path.join(model_folder, "family_encoder.pkl"))
type_encoder = joblib.load(os.path.join(model_folder, "type_encoder.pkl"))
state_encoder = joblib.load(os.path.join(model_folder, "state_encoder.pkl"))
scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))

# ------------------------------
# UI
# ------------------------------

st.title("📊 Future Sales Prediction Dashboard")

uploaded_file = st.file_uploader("Upload CSV File (max 100MB)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    if "Family" in df.columns:
        df["Family"] = family_encoder.transform(df["Family"])

    if "ProductType" in df.columns:
        df["ProductType"] = type_encoder.transform(df["ProductType"])

    if "State" in df.columns:
        df["State"] = state_encoder.transform(df["State"])

    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predict
    df["Predicted_Sales"] = xgb_model.predict(df)

    st.subheader("Prediction Results")
    st.dataframe(df)

    st.subheader("Predicted Sales Trend")
    st.line_chart(df["Predicted_Sales"])

    st.subheader("Summary Statistics")
    st.write(df["Predicted_Sales"].describe())

    st.subheader("Total Predicted Sales")
    st.success(df["Predicted_Sales"].sum())
