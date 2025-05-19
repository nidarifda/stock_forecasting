import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === Page Configuration ===
st.set_page_config(
    page_title="NVDA Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# === Load model and scaler ===
model = load_model("best_tuned_lstm_optuna.keras")
scaler = joblib.load("minmaxscaler.pkl")

# === App Title ===
st.title("NVIDIA Stock Price Forecasting App")

st.markdown("""
This app uses a **Tuned LSTM model** trained on multiseries data to forecast NVIDIA's stock price.
Upload your CSV file containing 60 time steps with 37 features (already normalized), and receive a prediction.
""")

# === Sidebar for File Upload ===
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload 60x37 CSV", type=['csv'])

# === Downloadable Template ===
with st.expander("Download Example Input Format"):
    st.download_button(
        label="Download Sample CSV (60x37)",
        data=pd.DataFrame(np.zeros((60, 37))).to_csv(index=False).encode('utf-8'),
        file_name="sample_input_60x37.csv",
        mime="text/csv"
    )

# === Main Prediction Logic ===
if file:
    try:
        df = pd.read_csv(file)
        st.success("✅ File uploaded successfully!")
        st.markdown("### First 5 Rows of Input Data")
        st.dataframe(df.head())

        if df.shape == (60, 37):
            input_data = np.array(df).reshape(1, 60, 37)
            prediction = model.predict(input_data)[0][0]

            # Get column names (if needed for inverse transform)
            scaler_columns = df.columns.tolist()
            dummy_input = np.zeros((1, len(scaler_columns)))
            dummy_input[0, -1] = prediction
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            st.success(f" **Predicted NVDA Closing Price: ${inv_pred:.2f}**")
        else:
            st.error(f"❌ Uploaded file must have shape (60, 37). Found: {df.shape}")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("Upload a CSV to begin prediction.")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em;'>
📡 TelcoChurn AI • Powered by Optuna-Tuned LSTM • © 2025 All rights reserved  
<br>
🔗 <a href="https://github.com/yourusername/yourrepo" target="_blank">View Code on GitHub</a>
</div>
""", unsafe_allow_html=True)
