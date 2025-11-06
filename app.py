import streamlit as st
import os
import zipfile
import gdown
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
ZIP_URL = "https://drive.google.com/uc?id=1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ"
ZIP_PATH = "models_output.zip"
MODELS_DIR = "models_output"

# -----------------------------
# STEP 1: Download models if not exist
# -----------------------------
if not os.path.exists(MODELS_DIR):
    st.write("⬇️ Downloading trained models from Google Drive... (~67MB)")
    gdown.download(ZIP_URL, ZIP_PATH, quiet=False)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

    # --- Auto-detect models folder ---
    if not os.path.exists(MODELS_DIR):
        for f in os.listdir("."):
            path = os.path.join(".", f)
            if os.path.isdir(path) and any(x in f.lower() for x in ["model", "output"]):
                os.rename(path, MODELS_DIR)
                st.info(f"📁 Found and renamed extracted folder '{f}' → '{MODELS_DIR}'")
                break

# Final check
if not os.path.exists(MODELS_DIR):
    st.error("❌ 'models_output' folder not found after extraction. Please check your Google Drive link or folder name inside zip.")
    st.stop()

st.success("✅ Models extracted successfully!")

# -----------------------------
# STEP 2: Streamlit UI
# -----------------------------
st.title("📈 Crypto Price Prediction using RNN & LSTM")

coin_folders = [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]

if not coin_folders:
    st.error("No coin model folders found inside models_output/")
    st.stop()

selected_coin = st.selectbox("Select a cryptocurrency:", coin_folders)

coin_dir = os.path.join(MODELS_DIR, selected_coin)
preds_path = os.path.join(coin_dir, f"{selected_coin}_predictions.csv")

if os.path.exists(preds_path):
    preds = pd.read_csv(preds_path)

    # --- Plot actual vs predicted ---
    st.subheader(f"📊 Actual vs Predicted Prices for {selected_coin}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(preds['actual'], label="Actual", color="black")
    ax.plot(preds['rnn_pred'], label="RNN Prediction", linestyle="--")
    ax.plot(preds['lstm_pred'], label="LSTM Prediction", linestyle=":")
    ax.legend()
    st.pyplot(fig)

    # --- Load best model and predict next 15 days ---
    meta = joblib.load(os.path.join(coin_dir, f"{selected_coin}_meta.pkl"))
    scaler = joblib.load(meta['scaler_path'])

    model_path = os.path.join(coin_dir, f"{selected_coin}_lstm_best.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(coin_dir, f"{selected_coin}_rnn_best.h5")

    model = tf.keras.models.load_model(model_path)

    # Predict next 15 days
    last_window = preds['actual'].values[-meta['window_size']:]
    scaled_window = scaler.transform(last_window.reshape(-1, 1)).reshape(1, meta['window_size'], 1)

    preds_future = []
    for _ in range(15):
        pred = model.predict(scaled_window)[0][0]
        preds_future.append(pred)
        scaled_window = np.append(scaled_window[:, 1:, :], [[[pred]]], axis=1)

    preds_future = scaler.inverse_transform(np.array(preds_future).reshape(-1, 1)).flatten()
    st.subheader("🔮 Next 15 Days Price Prediction")
    st.line_chart(preds_future)

else:
    st.warning(f"No prediction file found for {selected_coin}.")
