import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === Page Configuration ===
st.set_page_config(
    page_title="NVDA Stock Forecast",
    page_icon="📈",
    layout="centered"
)

# === Load Model and Scaler ===
try:
    model = load_model("best_tuned_lstm_optuna.keras")
    scaler = joblib.load("minmaxscaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# === App Title ===
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>NVDA Stock Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a time-series dataset and get the predicted closing price of NVIDIA stock.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Sidebar ===
with st.sidebar:
    st.image("https://companiesmarketcap.com/img/company-logos/256/NVDA.png", use_column_width=True)
    st.header("Upload Input Data")
    st.caption("Ensure your file is a **60x37 normalized CSV**. The last column must be NVDA_Close.")
    file = st.file_uploader("Upload 60x37 CSV", type=["csv"])

# === Main Prediction Logic ===
if file:
    try:
        df = pd.read_csv(file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        if df.shape != (60, 37):
            st.error(f"Incorrect shape: expected (60, 37), got {df.shape}")
        else:
            # Reshape to 3D tensor for LSTM
            input_data = np.array(df).reshape(1, 60, 37)

            # Make prediction
            raw_pred = model.predict(input_data)[0][0]

            # Inverse scale only the NVDA_Close value (assumed to be the last column)
            dummy_input = np.zeros((1, 37))
            dummy_input[0, -1] = raw_pred
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            st.markdown("---")
            st.subheader("Prediction Result")
            st.success(f"Predicted NVIDIA Closing Price: **${inv_pred:.2f}**")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to begin prediction.")

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9em;'>"
    "Powered by a Tuned LSTM Model • Developed with Streamlit & TensorFlow • © 2025"
    "</div>",
    unsafe_allow_html=True
)
