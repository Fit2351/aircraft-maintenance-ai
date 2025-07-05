
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("âœˆï¸ Aircraft Maintenance Risk Predictor (Live Training)")

st.write("Upload your commercial aircraft log data. This app will train a model live and predict component failure risks.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("ðŸ“„ Uploaded Data Preview")
    st.write(data.head())

    required_columns = [
        "Flight_Hours", "Flight_Cycles", "Engine_Vibration",
        "Brake_Temperature", "Hydraulic_Pressure",
        "Days_Since_Last_Maintenance", "Previous_Failure",
        "Component_Failure_Next_30_Days"
    ]

    if all(col in data.columns for col in required_columns):
        # Prepare data
        X = data[required_columns[:-1]]
        y = data["Component_Failure_Next_30_Days"]

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on full data
        predictions = model.predict(X)

        data["Predicted_Failure_Risk"] = ["YES" if p == 1 else "NO" for p in predictions]

        st.subheader("ðŸ” Prediction Results")
        st.write(data)

        # Download option
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("ðŸ“¥ Download Results as CSV", csv, "predictions.csv", "text/csv")
    else:
        st.error("âŒ Your CSV must include all required columns, including the target: 'Component_Failure_Next_30_Days'.")
