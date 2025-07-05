
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("aircraft_failure_predictor.pkl")

# Page title
st.title("✈️ Aircraft Maintenance Risk Predictor")
st.write("Upload your commercial aircraft log data to predict component failure risks in the next 30 days.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("📄 Uploaded Data Preview")
    st.write(data.head())

    # Check required columns
    required_columns = [
        "Flight_Hours", "Flight_Cycles", "Engine_Vibration",
        "Brake_Temperature", "Hydraulic_Pressure",
        "Days_Since_Last_Maintenance", "Previous_Failure"
    ]

    if all(col in data.columns for col in required_columns):
        # Run predictions
        predictions = model.predict(data[required_columns])
        data["Failure_Risk_Next_30_Days"] = ["YES" if pred == 1 else "NO" for pred in predictions]

        # Display results
        st.subheader("🔍 Prediction Results")
        st.write(data)

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "prediction_results.csv", "text/csv")
    else:
        st.error("❌ Missing required columns in your CSV file.")
