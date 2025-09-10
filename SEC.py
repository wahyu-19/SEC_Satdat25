import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import io
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ============================================
# Fungsi peramalan
# ============================================
def proses_peramalan(file):
    try:
        # ✅ baca file excel dengan BytesIO agar aman
        df = pd.read_excel(io.BytesIO(file.read()), engine="openpyxl")
        if not set(["TANGGAL", "RR"]).issubset(df.columns):
            st.error("File Excel harus ada kolom 'TANGGAL' dan 'RR'")
            st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # Pastikan TANGGAL dalam datetime
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL'])
    df = df.sort_values("TANGGAL")

    # Data asli
    st.subheader("📊 Data Asli")
    st.write(df.head())

    # Missing values
    st.subheader("🔍 Cek Missing Values")
    st.write(df.isnull().sum())

    fig1 = plt.figure()
    msno.bar(df)
    st.pyplot(fig1)
    
    fig2 = plt.figure()
    msno.matrix(df)
    st.pyplot(fig2)

    # Bersihkan data (drop NA di RR)
    df_nan = df.dropna(subset=['RR']).copy()
    rr_clean = df_nan['RR']

    # Distribusi curah hujan
    st.subheader("🌧 Distribusi Curah Hujan")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(rr_clean, bins=30, kde=True, color="skyblue", edgecolor="black", ax=ax)
    ax.set_xlabel("Curah Hujan (mm)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Curah Hujan (RR)")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Statistik sederhana
    rr_median = rr_clean.median()
    st.write("Median RR:", rr_median)

    # Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_nan['RR_norm'] = scaler.fit_transform(df_nan[['RR']])

    # Buat dataset untuk LSTM
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    dataset = df_nan['RR_norm'].values.reshape(-1, 1)
    look_back = 14
    trainX, trainY = create_dataset(dataset, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # Bangun model LSTM sederhana
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Training
    with st.spinner("Training model LSTM..."):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # Peramalan 30 hari ke depan
    forecast = []
    last_data = dataset[-look_back:].reshape(1, look_back, 1)

    for _ in range(30):
        next_pred = model.predict(last_data, verbose=0)
        forecast.append(next_pred[0][0])
        last_data = np.append(last_data[:,1:,:], [[next_pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))

    # Buat dataframe hasil peramalan
    future_dates = pd.date_range(start=df_nan['TANGGAL'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    df_forecast = pd.DataFrame({"TANGGAL": future_dates, "RR_Prediksi": forecast.flatten()})

    # Plot hasil
    st.subheader("📈 Hasil Peramalan")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_nan['TANGGAL'], df_nan['RR'], label="Data Aktual")
    ax.plot(df_forecast['TANGGAL'], df_forecast['RR_Prediksi'], label="Peramalan", color='red')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.set_title("Peramalan Curah Hujan dengan LSTM")
    ax.legend()
    st.pyplot(fig)

    return df, df_forecast

# ============================================
# Streamlit UI
# ============================================
st.title("🌾 Aplikasi Peramalan Curah Hujan (RR) dengan LSTM")

uploaded_file = st.file_uploader("📂 Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    st.subheader("📋 Hasil Peramalan 30 Hari")
    st.dataframe(df_forecast)

    csv = df_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Peramalan", csv, "hasil_peramalan.csv", "text/csv")
