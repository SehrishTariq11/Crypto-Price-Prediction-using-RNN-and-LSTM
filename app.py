import os
import gdown
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------------------------------------
# 🗂️ 1. Setup: Download and unzip models if missing
# ------------------------------------------------
def setup_models():
    models_dir = "models_output"
    zip_file = "models_output.zip"

    if os.path.exists(models_dir):
        print("✅ Models folder already exists — skipping download.")
        return

    FILE_ID = "1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ"  # 👈 only this part
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("⬇️ Downloading models_output.zip from Google Drive...")
    gdown.download(url, zip_file, quiet=False)

    print("📦 Extracting models_output.zip ...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("✅ Extraction complete! Models ready to use.")

setup_models()

# ------------------------------------------------
# ⚙️ 2. Helper functions
# ------------------------------------------------
def load_coin_models(coin_name):
    base_path = os.path.join("models_output", coin_name)
    if not os.path.exists(base_path):
        st.error(f"No model found for {coin_name}")
        return None, None, None

    lstm_path = os.path.join(base_path, f"{coin_name}_lstm_best.h5")
    rnn_path = os.path.join(base_path, f"{coin_name}_rnn_best.h5")
    scaler_path = os.path.join(base_path, f"{coin_name}_scaler.pkl")

    lstm_model = load_model(lstm_path) if os.path.exists(lstm_path) else None
    rnn_model = load_model(rnn_path) if os.path.exists(rnn_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return lstm_model, rnn_model, scaler


def plot_predictions(df, coin_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df["actual"], label="Actual", color="black")
    plt.plot(df["rnn_pred"], label="RNN Prediction", color="orange")
    plt.plot(df["lstm_pred"], label="LSTM Prediction", color="blue")
    plt.title(f"{coin_name} - Actual vs Predicted")
    plt.legend()
    st.pyplot(plt)


def predict_next_days(model, data, scaler, window_size=60, days=15):
    data_scaled = scaler.transform(np.array(data).reshape(-1, 1)).flatten()
    seq = data_scaled[-window_size:]
    preds = []

    for _ in range(days):
        X = np.array(seq[-window_size:]).reshape(1, window_size, 1)
        pred_scaled = model.predict(X, verbose=0)[0][0]
        preds.append(pred_scaled)
        seq = np.append(seq, pred_scaled)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


# ------------------------------------------------
# 🖥️ 3. Streamlit UI
# ------------------------------------------------
st.set_page_config(page_title="Crypto Prediction App", page_icon="📈", layout="centered")
st.title("📈 Crypto Price Prediction using RNN & LSTM")

# Find all coins available in models_output/
coin_folders = [f for f in os.listdir("models_output") if os.path.isdir(os.path.join("models_output", f))]
coin_folders.sort()

selected_coin = st.selectbox("Select a coin:", coin_folders)

if selected_coin:
    st.subheader(f"🔍 Results for {selected_coin}")

    pred_path = os.path.join("models_output", selected_coin, f"{selected_coin}_predictions.csv")
    if not os.path.exists(pred_path):
        st.error(f"Prediction file not found for {selected_coin}")
    else:
        df_preds = pd.read_csv(pred_path)
        plot_predictions(df_preds, selected_coin)

        lstm_model, rnn_model, scaler = load_coin_models(selected_coin)
        if lstm_model and scaler:
            # Use LSTM for future forecast
            last_close = df_preds["actual"].values
            future_preds = predict_next_days(lstm_model, last_close, scaler, window_size=60, days=15)

            st.subheader("📅 15-Day Future Price Prediction (LSTM)")
            st.line_chart(pd.DataFrame(future_preds, columns=["Predicted Price"]))
        else:
            st.warning("LSTM model or scaler not available for this coin.")
