
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import openai
import os

# 🔐 Set your OpenAI API key here
openai.api_key = "sk-..."  # Replace with your actual OpenAI key

# ✈️ Streamlit UI
st.title("✈️ Aircraft Maintenance Risk Predictor + GPT Reports")
st.write("Upload your aircraft log data. The app will train a model live, predict failure risk, and generate GPT-based maintenance reports.")

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# 🧠 GPT Report Generator
def generate_gpt_report(row):
    prompt = f"""
    Aircraft Maintenance Report:
    - Engine vibration: {row['Engine_Vibration']}
    - Brake temperature: {row['Brake_Temperature']}
    - Days since last maintenance: {row['Days_Since_Last_Maintenance']}
    - Previous failure: {'Yes' if row['Previous_Failure'] else 'No'}
    - Risk prediction: {row['Predicted_Failure_Risk']}

    Based on the data above, provide a short maintenance recommendation.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error generating report: {e}"

# 🔍 Main app logic
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

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        df["Predicted_Failure_Risk"] = ["YES" if p == 1 else "NO" for p in predictions]

        with st.spinner("Generating AI maintenance reports..."):
            df["GPT_Report"] = df.apply(generate_gpt_report, axis=1)

        st.subheader("📊 Prediction Results with GPT Maintenance Reports")
        st.write(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Report as CSV", csv, "aircraft_predictions_with_reports.csv", "text/csv")
    else:
        st.error("❌ Missing required columns in uploaded file.")
