# NVIDIA Stock Price Forecasting App

This Streamlit app predicts NVIDIA’s next closing stock price using a Tuned LSTM model trained on multiseries inputs from affiliated companies (TSMC, ASML, Cadence, Synopsys).

## Features
- Upload 60x37 CSV data (preprocessed & normalized)
- Predict next-day NVDA closing price
- Uses TensorFlow LSTM with Optuna-tuned hyperparameters
- Built with Streamlit for deployment

## How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Upload your 60x37 CSV to predict

## Files
- `app.py`: Streamlit interface
- `model.pkl`: Trained LSTM model
- `minmaxscaler.pkl`: MinMaxScaler for inverse transformation
- `requirements.txt`: Dependencies

## Contact
Developed by [Nida Rifda Chairuli] | MSc Data Science (APU)

---

*Note: This app assumes the input data is already normalized and contains exactly 60 time steps with 37 engineered features.*
