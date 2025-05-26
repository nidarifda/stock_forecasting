import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === Page Configuration ===
st.set_page_config(page_title="NVIDIA Stock Forecast", page_icon="📈", layout="wide")

# === Load Model and Scaler ===
model = load_model("tuned_cnn_lstm_a_nvda_only0.9395.keras")
scaler = joblib.load("minmaxscaler.pkl")

# === App Title ===
st.title("NVIDIA Stock Price Forecasting App")

st.markdown("""
This app uses a **Tuned LSTM model** trained on multiseries data to forecast NVIDIA's next-day closing price.  
Choose your input method below and upload the relevant CSV:
""")

# === Select Input Mode ===
mode = st.radio("Select Input Mode", ["Upload 60x37 Normalized Data", "Upload Raw Historical Data"])

# === Input Mode 1: 60x37 Normalized Data ===
if mode == "Upload 60x37 Normalized Data":
    st.subheader("Upload Normalized 60x37 CSV")
    file = st.file_uploader("Upload a 60x37 normalized CSV", type=["csv"])

    with st.expander("Download Sample Input Format"):
        st.download_button(
            label="⬇Download Sample CSV",
            data=pd.DataFrame(np.zeros((60, 37))).to_csv(index=False).encode('utf-8'),
            file_name="sample_input_60x37.csv",
            mime="text/csv"
        )

    if file:
        try:
            df = pd.read_csv(file)
            st.success("✅ File uploaded!")
            st.dataframe(df.head())

            if df.shape == (60, 37):
                input_data = np.array(df).reshape(1, 60, 37)
                prediction = model.predict(input_data)[0][0]

                dummy_input = np.zeros((1, df.shape[1]))
                dummy_input[0, -1] = prediction
                inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

                st.success(f"📈 Predicted NVDA Closing Price: **${inv_pred:.2f}**")
            else:
                st.error(f"❌ File must be shape (60, 37). Found: {df.shape}")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
    else:
        st.info("Please upload a normalized 60x37 CSV to proceed.")

# === Input Mode 2: Raw Historical Data ===
else:
    st.subheader("Upload Raw NVDA Stock CSV (Any Shape)")
    raw_file = st.file_uploader("Upload a raw historical stock CSV", type=["csv"], key="raw_upload")

    if raw_file:
        try:
            raw_df = pd.read_csv(raw_file)
            st.success("✅ File uploaded!")
            st.dataframe(raw_df.tail())

            if raw_df.shape[0] < 60:
                st.error("❌ Not enough rows. Please upload at least 60 rows of data.")
            else:
                # Optional: drop date or irrelevant columns
                # raw_df.drop(columns=['Date'], errors='ignore', inplace=True)

                raw_df = raw_df.select_dtypes(include=[np.number])
                scaled = scaler.transform(raw_df)
                scaled_df = pd.DataFrame(scaled, columns=raw_df.columns)

                last_sequence = scaled_df.tail(60).to_numpy().reshape(1, 60, -1)

                prediction = model.predict(last_sequence)[0][0]
                dummy = np.zeros((1, scaled_df.shape[1]))
                dummy[0, -1] = prediction
                inv_pred = scaler.inverse_transform(dummy)[0][-1]

                st.success(f"Predicted Next NVDA Close: **${inv_pred:.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error processing raw data: {e}")
    else:
        st.info("Upload a CSV containing historical NVDA stock data (min 60 rows).")

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em;'>
        📊 NVDA Forecast App • Powered by Optuna-Tuned LSTM • © 2025 All rights reserved  
        <br>🔗 <a href="https://github.com/yourusername/nvda-forecast-app" target="_blank">View Code on GitHub</a>
    </div>
    """, unsafe_allow_html=True
)
