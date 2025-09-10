import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ============================================
# Fungsi peramalan
# ============================================
def proses_peramalan(file):
    try:
        df = pd.read_csv(file)
        if not set(["TANGGAL", "RR"]).issubset(df.columns):
            st.error("File CSV harus ada kolom 'TANGGAL' dan 'RR'")
            st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # ================== Preprocessing ==================
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL']).sort_values("TANGGAL")

    # Ganti nilai sentinel 8888, 9999, dan "-" jadi NaN
    df['RR'] = df['RR'].replace([8888, 9999, "-"], np.nan)

    # Pastikan numerik
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')

    # Isi NaN dengan median
    rr_median = df['RR'].median()
    df['RR'] = df['RR'].fillna(rr_median)

    # ================== Normalisasi ==================
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['RR_norm'] = scaler.fit_transform(df[['RR']])

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    dataset = df['RR_norm'].values.reshape(-1, 1)
    look_back = 30
    trainX, trainY = create_dataset(dataset, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # ================== Model LSTM ==================
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    with st.spinner("Training model LSTM..."):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # ================== Forecast 365 Hari ==================
    forecast = []
    last_data = dataset[-look_back:].reshape(1, look_back, 1)
    for _ in range(365):
        next_pred = model.predict(last_data, verbose=0)
        forecast.append(next_pred[0][0])
        next_pred = next_pred.reshape(1, 1, 1)
        last_data = np.concatenate((last_data[:, 1:, :], next_pred), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Mulai dari 2025
    start_date = pd.Timestamp(year=2025, month=1, day=1)
    future_dates = pd.date_range(start=start_date, periods=365)

    df_forecast = pd.DataFrame({
        "TANGGAL": future_dates,
        "RR_Prediksi": forecast.flatten()
    })
    return df, df_forecast


# ============================================
# Layout UI
# ============================================
st.set_page_config(page_title="AgroForecast", layout="wide")

st.markdown("<h1 style='text-align:center;'>🌱 AGROFORECAST</h1>", unsafe_allow_html=True)

# =================== Kotak bulan ===================
bulan_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

cols = st.columns(12)
for i, b in enumerate(bulan_labels):
    with cols[i]:
        st.markdown(
            f"<div style='background-color:#95a5a6; padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
            unsafe_allow_html=True
        )

st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")

# =================== Upload + luas lahan ===================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data Curah Hujan")
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
with col2:
    st.subheader("Luas Lahan")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

# =================== Proses jika ada file ===================
if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    # ========== Rekomendasi tanam ==========
    median_pred = df_forecast["RR_Prediksi"].median()
    hasil_padi = "✅ Cocok untuk tanam Padi 🌾" if median_pred >= 100 else "❌ Tidak disarankan tanam Padi"
    hasil_palawija = "✅ Cocok untuk tanam Palawija 🌽" if 50 <= median_pred < 200 else "❌ Tidak disarankan tanam Palawija"

    col3, col4 = st.columns(2)
    with col3:
        st.success(hasil_padi)
    with col4:
        st.info(hasil_palawija)

    if luas_lahan > 0:
        rekomendasi = int(luas_lahan * median_pred / 100)
        st.markdown(
            f"<h3 style='text-align:center;'>Rekomendasi subsidi bibit : {rekomendasi} ton 🌱</h3>",
            unsafe_allow_html=True
        )

    # ========== Update warna bulan ==========
    cols = st.columns(12)
    for i, b in enumerate(bulan_labels):
        month_data = df_forecast[df_forecast["TANGGAL"].dt.month == (i+1)]
        mean_rr = month_data["RR_Prediksi"].mean()

        if mean_rr > 200:
            color = "#3498db"  # basah
        elif mean_rr >= 100:
            color = "#2ecc71"  # lembab
        else:
            color = "#e74c3c"  # kering

        with cols[i]:
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
                unsafe_allow_html=True
            )

    # ========== Tabel forecast ==========
    st.subheader("📈 Hasil Peramalan 365 Hari (2025)")
    st.dataframe(df_forecast)

    csv = df_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Peramalan", csv, "hasil_peramalan.csv", "text/csv")
