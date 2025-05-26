# NVIDIA Stock Price Forecasting App

This Streamlit app predicts NVIDIA’s next closing stock price using a **Tuned CNN-LSTM Model A**, trained exclusively on NVIDIA stock data with engineered features and optimized hyperparameters via Optuna.

## 🚀 Features
- Upload preprocessed and normalized 60×35 CSV sequence
- Predict next-day NVIDIA closing price
- Tuned CNN-LSTM architecture (Conv1D → LSTM → Dense)
- Uses TensorFlow (.keras) model and MinMaxScaler for output inversion
- Interactive Streamlit UI for fast deployment and testing

## 🛠 How to Run
1. Clone this repository
2. Install dependencies:
3. Start the app:

4. Upload your 60×35 normalized CSV to receive prediction

## 📁 Project Structure

| File                        | Description                                          |
|-----------------------------|------------------------------------------------------|
| `app.py`                   | Streamlit web interface                              |
| `best_tuned_lstm_optuna.keras` | Tuned CNN-LSTM A (NVIDIA-only) model              |
| `minmaxscaler.pkl`         | MinMaxScaler for reversing normalization             |
| `requirements.txt`         | Python dependencies                                  |
| `.devcontainer/`           | (Optional) VS Code Dev Container setup               |

## ⚠️ Input Requirements
- CSV shape must be **(60, 35)** — 60 time steps with 35 engineered NVIDIA-only features.
- Ensure the input is normalized using the same `MinMaxScaler` used during training.

## 👤 Contact
Developed by **Nida Rifda Chairuli**  
MSc Data Science | Asia Pacific University (APU)  
[(https://www.linkedin.com/in/nida-rifda-chairuli/)]

---

*Note: This app is intended for educational and research purposes. It assumes financial domain knowledge and correct data preprocessing steps have been applied.*
