import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# === Atur tampilan ===
st.set_page_config(page_title="AgroForecast", layout="wide")
st.markdown("<h1 style='text-align:center;'>AGROFORECAST</h1>", unsafe_allow_html=True)

# === Kalender ===
st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")
months = ["Januari","Februari","Maret","April","Mei","Juni",
          "Juli","Agustus","September","Oktober","November","Desember"]
colors = ["#42A5F5","#42A5F5","#42A5F5","#81C784","#81C784",
          "#FFD54F","#FFD54F","#FFD54F","#FFD54F","#81C784",
          "#42A5F5","#42A5F5"]
cols = st.columns(12, gap="small")
for i, month in enumerate(months):
    with cols[i]:
        st.markdown(f"""
            <div style='background-color:{colors[i]};
                        padding:20px;text-align:center;
                        border-radius:8px;color:black;
                        font-weight:bold;font-size:14px;
                        min-height:60px;'>
                {month}
            </div>""", unsafe_allow_html=True)

st.markdown("---")

# === Input data ===
col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Curah Hujan")
    uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx", "xls"])
with col2:
    st.subheader("Luas Lahan")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

st.markdown("---")

# Fungsi olah data
def proses_peramalan(file):
    df = pd.read_excel(file, usecols=["TANGGAL","RR"])
    df = df.replace([8888, 9999, "-"], np.nan)
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
    df['RR'] = df['RR'].fillna(df['RR'].median())
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')

    # normalisasi
    dataset = df['RR'].values.astype('float32').reshape(-1,1)
    scaler = MinMaxScaler()
    dataset_norm = scaler.fit_transform(dataset)

    # dataset window
    def create_dataset(dataset, look_back=7):
        X, y = [], []
        for i in range(len(dataset)-look_back):
            X.append(dataset[i:(i+look_back),0])
            y.append(dataset[i+look_back,0])
        return np.array(X), np.array(y)

    look_back = 7
    X, y = create_dataset(dataset_norm, look_back)
    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # model LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back,1)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # forecast 365 hari
    last_window = dataset_norm[-look_back:].reshape(1,look_back,1)
    forecast_scaled = []
    for _ in range(365):
        next_pred = model.predict(last_window, verbose=0)
        forecast_scaled.append(next_pred[0,0])
        last_window = np.append(last_window[:,1:,:], next_pred.reshape(1,1,1), axis=1)
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

    last_date = df['TANGGAL'].max()
    future_dates = pd.date_range(start=last_date+timedelta(days=1), periods=365, freq="D")
    df_forecast = pd.DataFrame({"Tanggal": future_dates, "Forecast_RR": forecast.flatten()})

    return df, df_forecast

# === Tombol proses ===
col3, col4 = st.columns(2)
with col3:
    st.subheader("Padi")
    if st.button("Hasil periode tanam - Padi", use_container_width=True) and uploaded_file:
        df, df_forecast = proses_peramalan(uploaded_file)
        st.success("Periode tanam padi berhasil dihitung!")
        st.line_chart(df_forecast.set_index("Tanggal")["Forecast_RR"])
        st.dataframe(df_forecast.head())
with col4:
    st.subheader("Palawija")
    if st.button("Hasil periode tanam - Palawija", use_container_width=True) and uploaded_file:
        df, df_forecast = proses_peramalan(uploaded_file)
        st.success("Periode tanam palawija berhasil dihitung!")
        st.line_chart(df_forecast.set_index("Tanggal")["Forecast_RR"])
        st.dataframe(df_forecast.head())

st.markdown("---")

# === Rekomendasi Subsidi ===
st.markdown("<h3 style='text-align:center;'>Rekomendasi subsidi bibit yang dapat diberikan : 12 ton</h3>", unsafe_allow_html=True)
