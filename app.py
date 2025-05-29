import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

# === Set Page Configuration ===
st.set_page_config(page_title="NVIDIA Stock Forecast", layout="wide")

# === Custom Styling: Clean UI ===
st.markdown("""
<style>
body {
    background-color: #f5f7fa !important;
    color: #2b2f42 !important;
}
.main, .block-container {
    background-color: #f5f7fa !important;
    padding: 2rem;
}
h1, h2, h3, h4 {
    color: #2b2f42;
    font-family: 'Segoe UI', sans-serif;
}
.stButton>button, .stDownloadButton>button {
    background-color: #1976d2;
    color: white;
    border-radius: 6px;
    font-weight: 600;
}
.stRadio > div {
    background-color: #ffffff;
    color: black;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1);
}
.stFileUploader, .stExpander {
    background-color: #ffffff !important;
    padding: 1rem;
    border-radius: 8px;
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
st.title("Stock Price Forecasting App")
st.markdown("""
This application leverages a tuned CNN-LSTM model trained on 60 time steps of stock data, each with 4 technical indicators, to forecast the next-day closing price.
Upload a 60×4 normalized CSV (60 rows × 4 features) to generate a prediction.
""")

# === Upload 60x4 Normalized CSV ===
st.subheader("Upload 60x4 Normalized CSV File")
file = st.file_uploader("Choose a 60x4 normalized CSV file", type=["csv"])

# === Download Sample File ===
with st.expander("Download Example Format"):
    sample_df = pd.read_csv("sample_nvda_input_60x4.csv")
    st.download_button(
        label="Download Sample CSV",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_nvda_input_60x4.csv",
        mime="text/csv"
    )

# === Prediction & Visualization ===
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

            # === Plot with Plotly ===
            past_close = df.iloc[:, -1].values  # assumes last column = normalized close
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(60)),
                y=past_close,
                mode='lines+markers',
                name='Last 60 Days (Normalized)',
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=[60],
                y=[prediction],
                mode='markers+text',
                name='Next Day Prediction',
                marker=dict(color='blue', size=10),
                text=[f"{prediction:.4f}"],
                textposition="top center"
            ))
            fig.update_layout(
                title="Normalized Close Price Forecast",
                xaxis_title="Time Step",
                yaxis_title="Normalized Price",
                plot_bgcolor='white',
                paper_bgcolor='#f5f7fa',
                font=dict(color="#2b2f42"),
                height=400
            )
            st.plotly_chart(fig)

            st.success(f"Predicted Stock Closing Price: ${inv_pred:.2f}")
        else:
            st.error(f"Incorrect input shape. Expected (60, 4), but got {df.shape}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a normalized 60x4 input file.")
