import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ===== Konfigurasi Streamlit =====
st.set_page_config(page_title="AgroForecast", layout="wide")
st.markdown("<h1 style='text-align:center;'>AGROFORECAST</h1>", unsafe_allow_html=True)

# ===== Upload Data =====
st.subheader("Upload Data Curah Hujan")
uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, usecols=["TANGGAL", "RR"])
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")

    # --- Preprocessing ---
    df = df.replace([8888, 9999, "-"], np.nan)
    df["RR"] = pd.to_numeric(df["RR"], errors="coerce")
    rr_median = df["RR"].median()
    df["RR"] = df["RR"].fillna(rr_median)

    st.write("📊 Data Curah Hujan (sample):")
    st.dataframe(df.head())

    # ===== Normalisasi =====
    dataset = df["RR"].values.astype("float32").reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_norm = scaler.fit_transform(dataset)

    # fungsi buat dataset
    def create_dataset(dataset, look_back=7):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i+look_back), 0])
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    look_back = 7
    X, y = create_dataset(dataset_norm, look_back)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # ===== Modeling LSTM =====
    if st.button("🚀 Jalankan Forecasting"):
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        history = model.fit(
            X_train, y_train, epochs=20, batch_size=32,
            validation_data=(X_test, y_test), shuffle=False, verbose=0
        )

        # Prediksi
        testPredict = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        testPredict_inv = scaler.inverse_transform(testPredict)

        # Evaluasi
        rmse = np.sqrt(mean_squared_error(y_test_inv, testPredict_inv))
        mae = mean_absolute_error(y_test_inv, testPredict_inv)

        st.success(f"✅ Model selesai! Test RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # Plot hasil prediksi
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test_inv, label="Actual")
        ax.plot(testPredict_inv, label="Prediksi", color="red")
        ax.set_title("Prediksi Curah Hujan (Test Set)")
        ax.set_ylabel("Curah Hujan (mm)")
        ax.legend()
        st.pyplot(fig)

        # Forecast ke depan (contoh 90 hari)
        forecast_horizon = 90
        last_window = dataset_norm[-look_back:].reshape(1, look_back, 1)
        forecast_scaled = []
        for _ in range(forecast_horizon):
            next_pred = model.predict(last_window, verbose=0)
            forecast_scaled.append(next_pred[0, 0])
            last_window = np.append(last_window[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))

        # Buat dataframe hasil forecast
        last_date = df["TANGGAL"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
        df_forecast = pd.DataFrame({"Tanggal": future_dates, "Forecast_RR": forecast.flatten()})

        st.subheader("📅 Hasil Forecasting")
        st.dataframe(df_forecast.head(30))

        # Plot forecast
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df["TANGGAL"], df["RR"], label="Data Aktual")
        ax2.plot(df_forecast["Tanggal"], df_forecast["Forecast_RR"], label="Forecast", color="red")
        ax2.set_title("Forecast Curah Hujan")
        ax2.set_ylabel("Curah Hujan (mm)")
        ax2.legend()
        st.pyplot(fig2)

        # Klasifikasi Oldeman
        df_forecast["Bulan"] = df_forecast["Tanggal"].dt.month
        df_bulanan = df_forecast.groupby("Bulan")["Forecast_RR"].sum().reset_index()

        def klasifikasi(rr):
            if rr > 200: return "Basah"
            elif rr >= 100: return "Lembab"
            else: return "Kering"

        df_bulanan["Klasifikasi"] = df_bulanan["Forecast_RR"].apply(klasifikasi)
        st.subheader("🌦️ Klasifikasi Bulanan (Oldeman)")
        st.dataframe(df_bulanan)

        # Rekomendasi sederhana
        st.subheader("🌱 Rekomendasi Tanam")
        rekom = "Padi dua kali setahun + palawija" if (df_bulanan["Klasifikasi"]=="Basah").sum() >= 7 else "Padi sekali + palawija"
        st.write(rekom)
