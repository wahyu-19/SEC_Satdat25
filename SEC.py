import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ============================================
# Fungsi peramalan
# ============================================
def proses_peramalan(file):
    try:
        df = pd.read_csv(file)  # <-- pakai CSV
        if not set(["TANGGAL", "RR"]).issubset(df.columns):
            st.error("File CSV harus ada kolom 'TANGGAL' dan 'RR'")
            st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # pastikan TANGGAL benar
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL']).sort_values("TANGGAL")

    # bersihkan data
    df_nan = df.dropna(subset=['RR'])
    rr_clean = df_nan['RR']

    # normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_nan['RR_norm'] = scaler.fit_transform(df_nan[['RR']])

    # fungsi dataset
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

    # model LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    with st.spinner("Training model LSTM..."):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # forecast 30 hari
    forecast = []
    last_data = dataset[-look_back:].reshape(1, look_back, 1)

    for _ in range(30):
        next_pred = model.predict(last_data, verbose=0)  # (1,1)
        forecast.append(next_pred[0][0])
        # ubah ke (1,1,1) supaya bisa digabung
        next_pred = next_pred.reshape(1, 1, 1)
        last_data = np.concatenate((last_data[:, 1:, :], next_pred), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # dataframe hasil prediksi
    future_dates = pd.date_range(start=df_nan['TANGGAL'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    df_forecast = pd.DataFrame({
        "TANGGAL": future_dates,
        "RR_Prediksi": forecast.flatten()
    })

    return df, df_forecast


# ============================================
# Layout UI
# ============================================
st.set_page_config(page_title="AgroForecast", layout="wide")

st.markdown("<h1 style='text-align: center;'>AGROFORECAST</h1>", unsafe_allow_html=True)
st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")

# tombol bulan (statis dengan warna)
cols = st.columns(12)
bulan = ["Januari","Februari","Maret","April","Mei","Juni",
         "Juli","Agustus","September","Oktober","November","Desember"]
warna = ["#4CAF50","#4CAF50","#4CAF50",  # hijau = basah
         "#2196F3","#2196F3",            # biru = lembab
         "#FFC107","#FFC107","#FFC107","#FFC107",  # kuning = kering
         "#2196F3","#4CAF50","#4CAF50"]

for i, b in enumerate(bulan):
    with cols[i]:
        st.markdown(
            f"<div style='background-color:{warna[i]};color:white;text-align:center;"
            f"padding:8px;border-radius:8px;font-weight:bold;'>{b}</div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# dua kolom: upload + luas lahan
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Data Curah Hujan")
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

with col2:
    st.subheader("Luas Lahan")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

# proses peramalan
if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    # median prediksi
    median_pred = df_forecast["RR_Prediksi"].median()

    # rekomendasi padi/palawija
    if median_pred >= 100:
        hasil_padi = "Cocok untuk tanam Padi 🌾"
        hasil_palawija = "Kurang cocok untuk Palawija"
    elif 50 <= median_pred < 100:
        hasil_padi = "Kurang optimal untuk Padi"
        hasil_palawija = "Cocok untuk tanam Palawija 🌽"
    else:
        hasil_padi = "Tidak disarankan tanam Padi"
        hasil_palawija = "Tidak disarankan tanam Palawija"

    col3, col4 = st.columns(2)
    with col3:
        st.text_input("Padi", value=hasil_padi)
    with col4:
        st.text_input("Palawija", value=hasil_palawija)

    # rekomendasi hasil (contoh sederhana: luas_lahan * median_pred / 100)
    if luas_lahan > 0:
        rekomendasi = int(luas_lahan * median_pred / 100)
        st.markdown(
            f"<h3 style='text-align:center;'>Rekomendasi subsidi bibit : "
            f"{rekomendasi} ton 🌾</h3>", unsafe_allow_html=True
        )

    st.subheader("📈 Hasil Peramalan 30 Hari")
    st.dataframe(df_forecast)

    csv = df_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Peramalan", csv, "hasil_peramalan.csv", "text/csv")
