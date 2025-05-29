import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Set page config ===
st.set_page_config(page_title="NVIDIA Stock Forecast", layout="wide")

# === Custom Styling: Blue background, white text ===
st.markdown("""
<style>
    body {
        background-color: #061f46 !important;
        color: white !important;
    }
    .main, .block-container {
        background-color: #061f46 !important;
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6, label, .stRadio label, .css-17eq0hr, .css-1v0mbdj {
        color: white !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #ffffff;
        color: black;
        font-weight: bold;
        border-radius: 8px;
    }
    .stRadio > div {
        background-color: #ffffff;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .stFileUploader, .stDownloadButton, .stExpander {
        background-color: #061f46 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
    }
    .stDataFrame, .stTable {
        background-color: white !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# === Load Model and Scaler ===
model = load_model("tuned_cnn_lstm_a_nvda_only0.9395.keras")
scaler = joblib.load("minmaxscaler.pkl")

# === App Title & Description ===
st.title("NVIDIA Stock Price Forecasting App")
st.markdown("""
This application uses a tuned CNN-LSTM model trained exclusively on NVIDIA stock data (60 time steps × 4 features)  
to forecast the next-day closing price. Upload normalized model-ready input to begin.
""")

# === Upload 60x4 Normalized Input ===
st.subheader("Upload 60x4 Normalized CSV File")
file = st.file_uploader("Choose a 60x4 normalized CSV file", type=["csv"])

with st.expander("Download Example Format"):
    st.download_button(
        label="Download Sample CSV",
        data=pd.DataFrame(np.zeros((60, 4))).to_csv(index=False).encode('utf-8'),
        file_name="sample_input_60x4.csv",
        mime="text/csv"
    )

if file:
    try:
        df = pd.read_csv(file)
        st.success("File uploaded successfully.")
        st.dataframe(df.head())

        if df.shape == (60, 4):
            input_data = np.array(df).reshape(1, 60, 4)
            prediction = model.predict(input_data)[0][0]

            # Inverse transform to actual closing price
            dummy_input = np.zeros((1, 4))
            dummy_input[0, -1] = prediction
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            # Plot Normalized Trend (last column assumed to be close price)
            past_close = df.iloc[:, -1].values
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
            st.error(f"Incorrect input shape. Expected (60, 4), but got {df.shape}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a normalized 60x4 input file.")
