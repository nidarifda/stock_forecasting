import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(page_title="NVIDIA Stock Forecast", layout="wide")

# Load model and scaler
model = load_model("tuned_cnn_lstm_a_nvda_only0.9395.keras")
scaler = joblib.load("minmaxscaler.pkl")

# App title and description
st.title("NVIDIA Stock Price Forecasting App")
st.markdown("""
This application utilizes a tuned CNN-LSTM model trained on NVIDIA stock data to forecast the next-day closing price. 
Users may upload either normalized model-ready input or raw historical data.
""")

# Input mode selection
mode = st.radio("Select Input Mode", ["Upload Normalized Data (60x37)", "Upload Raw Historical Data"])

# Input mode 1: Normalized 60x37 data
if mode == "Upload Normalized Data (60x37)":
    st.subheader("Upload 60x37 Normalized CSV File")
    file = st.file_uploader("Choose a 60x37 normalized CSV file", type=["csv"])

    with st.expander("Download Example Input Format"):
        st.download_button(
            label="Download Sample CSV",
            data=pd.DataFrame(np.zeros((60, 37))).to_csv(index=False).encode('utf-8'),
            file_name="sample_input_60x37.csv",
            mime="text/csv"
        )

    if file:
        try:
            df = pd.read_csv(file)
            st.success("File uploaded successfully.")
            st.dataframe(df.head())

            if df.shape == (60, 37):
                input_data = np.array(df).reshape(1, 60, 37)
                prediction = model.predict(input_data)[0][0]

                dummy_input = np.zeros((1, df.shape[1]))
                dummy_input[0, -1] = prediction
                inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

                st.success(f"Predicted NVIDIA Closing Price: ${inv_pred:.2f}")
            else:
                st.error(f"Incorrect input shape. Expected (60, 37), but received: {df.shape}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a normalized 60x37 input file.")

# Input mode 2: Raw historical stock data
else:
    st.subheader("Upload Raw Historical Stock Data")
    raw_file = st.file_uploader("Upload a CSV file containing historical stock prices", type=["csv"], key="raw_upload")

    if raw_file:
        try:
            raw_df = pd.read_csv(raw_file)
            st.success("File uploaded successfully.")
            st.dataframe(raw_df.tail())

            if raw_df.shape[0] < 60:
                st.error("Insufficient data. At least 60 rows are required.")
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
            st.error(f"An error occurred while processing the data: {e}")
    else:
        st.info("Please upload a valid CSV with historical stock data (minimum 60 rows).")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em;'>
        NVIDIA Forecasting App | Built using CNN-LSTM and Streamlit | © 2025  
        <br><a href="https://github.com/yourusername/nvda-forecast-app" target="_blank">View Source Code</a>
    </div>
    """, unsafe_allow_html=True
)
