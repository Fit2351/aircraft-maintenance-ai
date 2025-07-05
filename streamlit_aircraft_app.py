
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("✈️ Aircraft Maintenance Risk Predictor")

st.write("Upload your aircraft log data. The app will train a model live and predict component failure risk.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = [
        "Flight_Hours", "Flight_Cycles", "Engine_Vibration",
        "Brake_Temperature", "Hydraulic_Pressure",
        "Days_Since_Last_Maintenance", "Previous_Failure",
        "Component_Failure_Next_30_Days"
    ]

    if all(col in df.columns for col in required_cols):
        X = df[required_cols[:-1]]
        y = df["Component_Failure_Next_30_Days"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X)
        df["Predicted_Failure_Risk"] = ["YES" if p == 1 else "NO" for p in predictions]

        st.subheader("📊 Prediction Results")
        st.write(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
    else:
        st.error("❌ Missing required columns in uploaded file.")
