import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Set page config as the first Streamlit command
st.set_page_config(page_title="NVIDIA Stock Forecast", layout="wide")

# === Custom Styling ===
st.markdown("""
<style>
    .main {
        background-color: #0d47a1; /* Dark blue */
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #1976d2;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stRadio > div {
        background-color: #1565c0;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .stDataFrame {
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# === Load Model and Scaler ===
model = load_model("tuned_cnn_lstm_a_nvda_only0.9395.keras")
scaler = joblib.load("minmaxscaler.pkl")

# === App Title & Description ===
st.title("NVIDIA Stock Price Forecasting App")
st.markdown("""
This application uses a tuned CNN-LSTM model trained exclusively on NVIDIA data to forecast the next-day closing price.  
You can upload either 60x35 normalized model-ready input or raw historical data.
""")

# === Input Mode Selection ===
mode = st.radio("Select Input Mode", ["Upload Normalized Data (60x35)", "Upload Raw Historical Data"])

# === Mode 1: Upload Normalized Data ===
if mode == "Upload Normalized Data (60x35)":
    st.subheader("Upload 60x35 Normalized CSV File")
    file = st.file_uploader("Choose a 60x35 normalized CSV file", type=["csv"])

    with st.expander("Download Example Format"):
        st.download_button(
            label="Download Sample CSV",
            data=pd.DataFrame(np.zeros((60, 35))).to_csv(index=False).encode('utf-8'),
            file_name="sample_input_60x35.csv",
            mime="text/csv"
        )

    if file:
        try:
            df = pd.read_csv(file)
            st.success("File uploaded successfully.")
            st.dataframe(df.head())

            if df.shape == (60, 35):
                input_data = np.array(df).reshape(1, 60, 35)
                prediction = model.predict(input_data)[0][0]

                dummy_input = np.zeros((1, 35))
                dummy_input[0, -1] = prediction
                inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

                st.success(f"Predicted NVIDIA Closing Price: ${inv_pred:.2f}")
            else:
                st.error(f"Incorrect input shape. Expected (60, 35), but received: {df.shape}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a normalized 60x35 input file.")

# === Mode 2: Upload Raw Historical Data ===
else:
    st.subheader("Upload Raw Historical Stock Data")
    raw_file = st.file_uploader("Upload a CSV with historical stock prices", type=["csv"], key="raw_upload")

    if raw_file:
        try:
            raw_df = pd.read_csv(raw_file)
            st.success("File uploaded successfully.")
            st.dataframe(raw_df.tail())

            if raw_df.shape[0] < 60:
                st.error("Insufficient data. At least 60 rows required.")
            else:
                raw_df = raw_df.select_dtypes(include=[np.number])
                scaled = scaler.transform(raw_df)
                scaled_df = pd.DataFrame(scaled, columns=raw_df.columns)

                last_sequence = scaled_df.tail(60).to_numpy().reshape(1, 60, -1)
                prediction = model.predict(last_sequence)[0][0]

                dummy = np.zeros((1, scaled_df.shape[1]))
                dummy[0, -1] = prediction
                inv_pred = scaler.inverse_transform(dummy)[0][-1]

                st.success(f"Predicted Next NVIDIA Closing Price: ${inv_pred:.2f}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload at least 60 rows of stock data.")

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: white;'>
        NVIDIA Forecasting App | Built using CNN-LSTM and Streamlit | © 2025  
        <br><a style='color: #90caf9;' href="https://github.com/yourusername/nvda-forecast-app" target="_blank">View Source Code</a>
    </div>
    """, unsafe_allow_html=True
)
