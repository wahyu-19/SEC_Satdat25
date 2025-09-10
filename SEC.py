import streamlit as st
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# === Fungsi utama proses peramalan ===
def proses_peramalan(uploaded_file):
    # === 1. Baca dataset ===
    df = pd.read_csv(uploaded_file)

    # Pastikan ada kolom TANGGAL & CURAH_HUJAN
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL'])
    df = df.sort_values('TANGGAL')

    # Handle sentinel values (8888, 9999)
    df['CURAH_HUJAN'] = pd.to_numeric(df['CURAH_HUJAN'], errors='coerce')
    df['CURAH_HUJAN'] = df['CURAH_HUJAN'].replace([8888, 9999], np.nan)
    median_val = df['CURAH_HUJAN'].median()
    df['CURAH_HUJAN'] = df['CURAH_HUJAN'].fillna(median_val)

    dataset = df[['CURAH_HUJAN']].values

    # === 2. Scaling ===
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_scaled = scaler.fit_transform(dataset)

    # === 3. Load model LSTM (pretrained) ===
    try:
        model = tf.keras.models.load_model("model_lstm.h5")
    except Exception as e:
        st.error("Model LSTM belum tersedia. Upload dulu file model_lstm.h5 ke folder yang sama.")
        return df, pd.DataFrame()

    # === 4. Forecasting autoregressive ===
    look_back = 30  # pastikan sama dengan waktu training
    predictions = []

    if len(dataset_scaled) < look_back:
        st.error(f"Dataset terlalu pendek! Minimal {look_back+1} data diperlukan, sekarang hanya {len(dataset_scaled)}.")
        return df, pd.DataFrame()

    last_data = dataset_scaled[-look_back:]      # (look_back, 1)
    current_input = last_data.reshape(1, look_back, 1)

    for _ in range(365):  # prediksi setahun
        next_pred = model.predict(current_input, verbose=0)  # (1,1)
        predictions.append(next_pred[0][0])

        # update input window pakai concatenate
        current_input = np.concatenate(
            (current_input[:, 1:, :], next_pred.reshape(1,1,1)),
            axis=1
        )

    # === 5. Balikkan ke skala asli ===
    forecast = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    ).flatten()

    # === 6. Susun DataFrame hasil forecast ===
    last_date = df['TANGGAL'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)

    df_forecast = pd.DataFrame({
        'TANGGAL': forecast_dates,
        'FORECAST_CURAH_HUJAN': forecast
    })

    return df, df_forecast


# === Streamlit UI ===
st.title("📈 Aplikasi Peramalan Curah Hujan dengan LSTM")

st.write("Upload dataset iklim (format CSV dengan kolom **TANGGAL** & **CURAH_HUJAN**).")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    if st.button("🔮 Jalankan Peramalan"):
        df, df_forecast = proses_peramalan(uploaded_file)

        if not df_forecast.empty:
            st.subheader("Data Asli")
            st.dataframe(df.tail())

            st.subheader("Hasil Peramalan 365 Hari ke Depan")
            st.dataframe(df_forecast.head())

            # Plot hasil
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df['TANGGAL'], df['CURAH_HUJAN'], label="Data Aktual")
            ax.plot(df_forecast['TANGGAL'], df_forecast['FORECAST_CURAH_HUJAN'], label="Forecast", color='red')
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Curah Hujan")
            ax.legend()
            st.pyplot(fig)

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
