import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# === Page Configuration ===
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# === Custom Styling ===
st.markdown("""
<style>
body {
    background-color: #f4ecf9 !important;
    color: #2b2f42 !important;
}
.main, .block-container {
    background-color: #f4ecf9 !important;
    padding: 2rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #7e3ff2 !important;
    text-align: center;
}
p {
    text-align: center;
    font-size: 1.1rem;
}
.stButton>button, .stDownloadButton>button {
    background-color: #7e3ff2;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.stFileUploader, .stExpander {
    background-color: #ffffff !important;
    padding: 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# === Load Model and Scaler ===
model = load_model("tuned_cnn_lstm_a_nvda_only0.9395.keras")
scaler = joblib.load("minmaxscaler.pkl")

# === Header ===
st.title("Stock Price Forecasting App")
st.markdown("""
<div style='text-align: center; max-width: 720px; margin: auto; font-size: 0.95rem; color: #3a3a3a;'>
This application uses a tuned CNN-LSTM model trained on 60 time steps of stock data, each with 4 normalized technical indicators, to forecast the next-day closing price.  
Upload a 60×5 CSV (including a date column) to generate a prediction.
</div>
""", unsafe_allow_html=True)

# === File Upload ===
st.subheader("Upload Normalized CSV (60 rows × 5 columns incl. Date)")
file = st.file_uploader("Choose a CSV file", type=["csv"])

# === Sample Download ===
with st.expander("Download Sample CSV Format"):
    try:
        sample_df = pd.read_csv("Simulated_NVDA_Sample_Input__60x4_.csv")
        st.download_button(
            label="Download Sample CSV",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="Simulated_NVDA_Sample_Input__60x4_.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning("Sample file not found in repository.")

# === Prediction & Plot ===
if file:
    try:
        df = pd.read_csv(file)
        st.success("File uploaded successfully.")
        st.dataframe(df.head())

        if df.shape == (60, 5):
            dates = pd.to_datetime(df.iloc[:, 0])       # First column as Date
            features = df.iloc[:, 1:].astype(float)     # Next 4 columns = features
            input_data = np.array(features).reshape(1, 60, 4)

            # Prediction
            prediction = model.predict(input_data)[0][0]

            # Inverse transform
            dummy_input = np.zeros((1, 4))
            dummy_input[0, -1] = prediction
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            # === Plotly Line Chart ===
            past_close = features.iloc[:, -1].values  # Assumes last col = close
            next_day = dates.iloc[-1] + pd.Timedelta(days=1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=past_close,
                mode='lines+markers',
                name='Last 60 Days (Normalized)',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=6),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: %{y:.4f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[next_day],
                y=[prediction],
                mode='markers+text',
                name='Next Day Prediction',
                marker=dict(color='black', size=10),
                text=[f"{prediction:.4f}"],
                textposition="top center",
                hovertemplate='Next Day Prediction<br>Price: %{y:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title="Normalized Close Price Forecast",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                font=dict(family="Segoe UI", size=14, color="#2b2f42"),
                plot_bgcolor='white',
                paper_bgcolor='#f4ecf9',
                xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                height=450,
                margin=dict(l=40, r=40, t=80, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Predicted Next Closing Price: ${inv_pred:.2f}")
        else:
            st.error(f"Expected shape (60, 5) but got {df.shape}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a normalized 60x5 CSV file with [Date, Feature1, Feature2, Feature3, Close].")
