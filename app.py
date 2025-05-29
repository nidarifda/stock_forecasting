import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

# === Page Configuration ===
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# === Custom Styling (Lilac Theme) ===
st.markdown("""
<style>
body {
    background-color: #f3e8f5 !important;
    color: #2b2f42 !important;
}
.main, .block-container {
    background-color: #f3e8f5 !important;
    padding: 2rem;
}
h1, h2, h3, h4 {
    color: #7c3aed !important;
    font-family: 'Segoe UI', sans-serif;
}
.stButton>button, .stDownloadButton>button {
    background-color: #7c3aed;
    color: white;
    border-radius: 6px;
    font-weight: 600;
}
.stRadio > div {
    background-color: #ffffff;
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

# === Title and Description ===
st.title("Stock Price Forecasting App")
st.markdown("""
This application uses a tuned CNN-LSTM model trained on 60 time steps of stock data,  
each with 4 normalized technical indicators, to forecast the next-day closing price.  
Upload a 60×5 CSV (including a date column) to generate a prediction.
""")

# === Upload File ===
st.subheader("Upload Normalized CSV (60 rows × 5 columns incl. Date)")
file = st.file_uploader("Upload your 60x5 normalized CSV (Date + 4 features)", type=["csv"])

# === Sample CSV Download ===
with st.expander("Download Example Format"):
    sample_data = pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.today(), periods=60).strftime('%Y-%m-%d'),
        'MA_10': np.random.rand(60),
        'RSI': np.random.rand(60),
        'Volume': np.random.rand(60),
        'Normalized_Close': np.random.rand(60)
    })
    st.download_button(
        label="Download Sample CSV",
        data=sample_data.to_csv(index=False).encode('utf-8'),
        file_name="sample_stock_input_60x5.csv",
        mime="text/csv"
    )

# === Prediction ===
if file:
    try:
        df = pd.read_csv(file)
        st.success("File uploaded successfully.")
        st.dataframe(df.head())

        if df.shape == (60, 5):
            # Drop date, keep numeric
            input_data = np.array(df.iloc[:, 1:]).reshape(1, 60, 4)

            prediction = model.predict(input_data)[0][0]

            # Inverse transform to actual closing price
            dummy_input = np.zeros((1, 4))
            dummy_input[0, -1] = prediction
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            # Plot with Plotly
            dates = df['Date'].values
            past_close = df.iloc[:, -1].values
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=past_close,
                mode='lines+markers',
                name='Last 60 Days (Normalized)',
                line=dict(color='#a855f7')
            ))
            fig.add_trace(go.Scatter(
                x=[dates[-1]],  # Last date
                y=[prediction],
                mode='markers+text',
                name='Next Day Prediction',
                marker=dict(color='black', size=10),
                text=[f"{prediction:.4f}"],
                textposition="top center"
            ))
            fig.update_layout(
                title="Normalized Close Price Forecast",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                plot_bgcolor='white',
                paper_bgcolor='#f3e8f5',
                font=dict(color="#2b2f42"),
                height=400
            )
            st.plotly_chart(fig)

            st.success(f"Predicted Closing Price: **${inv_pred:.2f}**")
        else:
            st.error(f"Incorrect input shape. Expected 60 rows × 5 columns, got {df.shape}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a normalized CSV file in the correct format.")
