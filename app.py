import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Set page config as the first Streamlit command
st.set_page_config(page_title="NVIDIA Stock Forecast", layout="wide")

# Custom Styling for full blue background and white text
st.markdown("""
<style>
    body {
        background-color: #0d47a1 !important;
        color: white !important;
    }
    .main, .block-container {
        background-color: #0d47a1 !important;
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6, label, .stRadio label {
        color: white !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #1976d2;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stRadio > div {
        background-color: #1976d2;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .stDataFrame, .stTable {
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

                # Inverse transform for final price
                dummy_input = np.zeros((1, 35))
                dummy_input[0, -1] = prediction
                inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

                # Plot the trend line
                import matplotlib.pyplot as plt

                past_close = df.iloc[:, -1].values  # assuming last col is normalized Close
                full_series = np.append(past_close, prediction)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(60), past_close, label="Last 60 Days (Normalized)", color="white")
                ax.plot(60, prediction, 'ro', label="Next Day Prediction", markersize=6)
                ax.set_title("Normalized Close Price Trend", color="white")
                ax.set_xlabel("Time Step", color="white")
                ax.set_ylabel("Normalized Price", color="white")
                ax.tick_params(colors='white')
                ax.legend()

                st.pyplot(fig)
                st.success(f"Predicted NVIDIA Closing Price: ${inv_pred:.2f}")
            else:
                st.error(f"Incorrect input shape. Expected (60, 35), but received: {df.shape}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a normalized 60x35 input file.")
