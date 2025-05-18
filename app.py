import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('model.pkl')
scaler = joblib.load('minmaxscaler.pkl')

st.set_page_config(page_title="NVDA Stock Forecast", layout="wide")
st.title("📈 NVIDIA Stock Price Forecasting App")

st.markdown("""
This app uses a **Tuned LSTM model** trained on multiseries data to forecast NVIDIA's stock price.
Upload your CSV file containing 60 time steps with 37 features (already normalized), and receive a prediction.
""")

# Sidebar for file upload
st.sidebar.header("📂 Upload Data")
file = st.sidebar.file_uploader("Upload 60x37 CSV", type=['csv'])

if file:
    try:
        df = pd.read_csv(file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        if df.shape == (60, 37):
            input_data = np.array(df).reshape(1, 60, 37)
            prediction = model.predict(input_data)[0][0]

            # Inverse scale only the NVDA_Close (assumed last feature)
            dummy_input = np.zeros((1, 37))
            dummy_input[0, -1] = prediction
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            st.success(f"📊 Predicted NVDA Closing Price: **${inv_pred:.2f}**")
        else:
            st.error(f"Uploaded file must have shape (60, 37). Found: {df.shape}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV to begin prediction.")

st.markdown("""
---
*Developed using TensorFlow, Streamlit, and Optuna-tuned LSTM.*
""")
