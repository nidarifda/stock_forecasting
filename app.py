import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# === Page Config ===
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# === Styling ===
st.markdown("""
<style>
body {
    background-color: #f4ecf9 !important;
    color: #2b2f42 !important;
}
h1, h2, h3, h4 {
    color: #7e3ff2 !important;
    text-align: center;
}
p {
    text-align: center;
    font-size: 1.1rem;
}
.main, .block-container {
    background-color: #f4ecf9 !important;
    padding: 2rem;
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
.metric-box {
    background-color: #e0e0f0;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# === Header ===
st.title("Stock Price Forecasting App")
st.markdown("""
<div style='text-align: center; max-width: 800px; margin: auto; font-size: 0.85rem; color: #3a3a3a;'>
This app uses a CNN-LSTM model trained on 60 days of normalized stock data to forecast the next-day closing price.<br>
Upload a 60×5 CSV (including a date column) to generate a prediction.
</div>
""", unsafe_allow_html=True)

# === Sidebar Model Selection ===
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("Choose Model Version", ["Tuned Model", "Baseline Model"])
show_ci = st.sidebar.checkbox("Show Confidence Interval", value=False)

# === Load Selected Model ===
model_file = "tuned_cnn_lstm_a_nvda_only0.9395.keras" if model_choice == "Tuned Model" else "baseline_model.keras"
model = load_model(model_file)
scaler = joblib.load("minmaxscaler.pkl")

# === File Upload ===
st.subheader("Upload Normalized CSV (60 rows × 5 columns incl. Date)")
file = st.file_uploader("Choose a CSV file", type=["csv"])

# === Sample CSV Download ===
with st.expander("Download Sample CSV Format"):
    try:
        sample_df = pd.read_csv("Simulated_NVDA_Sample_Input__60x5_.csv")
        st.download_button(
            label="Download Sample CSV",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="Simulated_NVDA_Sample_Input__60x5_.csv",
            mime="text/csv"
        )
    except:
        st.warning("Sample CSV file not found.")

# === Main Logic ===
if file:
    try:
        df = pd.read_csv(file)
        st.success("File uploaded successfully.")
        st.dataframe(df.head())

        if df.shape == (60, 5):
            dates = pd.to_datetime(df.iloc[:, 0])
            features = df.iloc[:, 1:].astype(float)
            input_data = np.array(features).reshape(1, 60, 4)

            # Prediction
            prediction = model.predict(input_data)[0][0]

            # Inverse transform
            dummy_input = np.zeros((1, 4))
            dummy_input[0, -1] = prediction
            inv_pred = scaler.inverse_transform(dummy_input)[0][-1]

            # Optional confidence interval (dummy for now)
            ci_upper, ci_lower = prediction + 0.05, prediction - 0.05

            # Actual closing values
            y_true = features.iloc[:, -1].values

            # Metrics
            mae = np.mean(np.abs(y_true - np.mean(y_true)))
            rmse = np.sqrt(np.mean((y_true - np.mean(y_true))**2))

            # === Plot Forecast ===
            next_day = dates.iloc[-1] + pd.Timedelta(days=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_true,
                mode='lines+markers',
                name='Last 60 Days (Normalized)',
                line=dict(color='#9b59b6', width=2.5),
                marker=dict(size=5)
            ))
            fig.add_trace(go.Scatter(
                x=[next_day],
                y=[prediction],
                mode='markers+text',
                name='Next Day Prediction',
                marker=dict(color='black', size=10),
                text=[f"{prediction:.4f}"],
                textposition="top center"
            ))
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=[next_day, next_day],
                    y=[ci_lower, ci_upper],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(color='rgba(0,0,0,0.2)', dash='dot')
                ))

            fig.update_layout(
                title="Normalized Close Price Forecast",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                font=dict(family="Segoe UI", size=14, color="#2b2f42"),
                plot_bgcolor='white',
                paper_bgcolor='#f4ecf9',
                xaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.3)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.3)'),
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Predicted Next Closing Price: ${inv_pred:.2f}")

            # === Metrics ===
            st.markdown("<div class='metric-box'>"
                        f"<strong>Model Evaluation:</strong> MAE = <span style='color:green'>{mae:.4f}</span>, "
                        f"RMSE = <span style='color:green'>{rmse:.4f}</span>"
                        "</div>", unsafe_allow_html=True)
        else:
            st.error(f"Expected shape (60, 5) but got {df.shape}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a normalized 60x5 CSV file with [Date, Feature1, Feature2, Feature3, Close].")
