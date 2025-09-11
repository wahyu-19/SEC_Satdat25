import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

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

    # ======== Preprocessing Tanggal ========
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL']).sort_values("TANGGAL")

    # ======== Preprocessing RR ========
    df['RR'] = df['RR'].replace([8888, 9999, "-"], np.nan)
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
    df['RR'] = df['RR'].fillna(df['RR'].median())

    # ======== Normalisasi untuk LSTM ========
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['RR_norm'] = scaler.fit_transform(df[['RR']])

    # ======== Dataset untuk LSTM ========
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

    # ======== Model LSTM ========
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    with st.spinner("Training model LSTM..."):
        model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0)

    # ======== Forecast 365 Hari ========
    forecast = []
    last_data = dataset[-look_back:].reshape(1, look_back, 1)
    for _ in range(365):
        next_pred = model.predict(last_data, verbose=0)
        forecast.append(next_pred[0][0])
        next_pred = next_pred.reshape(1, 1, 1)
        last_data = np.concatenate((last_data[:, 1:, :], next_pred), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # ======== Set tanggal mulai 2025 ========
    start_date = pd.Timestamp(year=2025, month=1, day=1)
    future_dates = pd.date_range(start=start_date, periods=365)

    df_forecast = pd.DataFrame({
        "TANGGAL": future_dates,
        "RR_Prediksi": forecast.flatten()
    })

    # ======== Agregasi Bulanan ========
    df_forecast['Tahun'] = df_forecast['TANGGAL'].dt.year
    df_forecast['Bulan'] = df_forecast['TANGGAL'].dt.month

    df_bulanan = (
        df_forecast.groupby(['Tahun', 'Bulan'])['RR_Prediksi']
        .sum()
        .reset_index()
    )

    df_bulanan['Nama_Bulan'] = df_bulanan['Bulan'].map({
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
        7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"
    })

    return df_bulanan

# ============================================
# Layout UI
# ============================================
st.set_page_config(page_title="AgroForecast", layout="wide")

st.markdown("<h1 style='text-align:center;'>🌱 AGROFORECAST</h1>", unsafe_allow_html=True)
st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")

# =================== Kotak-kotak bulan ===================
bulan_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

cols_bulan = st.columns(12)
placeholders_bulan = []
for i, b in enumerate(bulan_labels):
    with cols_bulan[i]:
        ph = st.empty()
        ph.markdown(
            f"<div style='background-color:#95a5a6; "
            f"padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
            unsafe_allow_html=True
        )
        placeholders_bulan.append(ph)

# Garis pemisah dengan jarak kecil
st.markdown(
    "<hr style='margin:20px 0;'>",  # ubah angka 20px jadi lebih kecil kalau mau lebih rapat
    unsafe_allow_html=True
)

# =================== Upload data + luas lahan ===================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data Curah Hujan")
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
with col2:
    st.subheader("Luas Lahan")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

# =================== Jika ada file ===================
if uploaded_file is not None:
    df_bulanan = proses_peramalan(uploaded_file)

    # ========== HASIL REKOMENDASI ==========
    median_pred = df_bulanan["RR_Prediksi"].median()
    hasil_padi = "✅ Cocok untuk tanam Padi 🌾" if median_pred >= 300 else "❌ Tidak disarankan tanam Padi"
    hasil_palawija = "✅ Cocok untuk tanam Palawija 🌽" if 150 <= median_pred < 300 else "❌ Tidak disarankan tanam Palawija"

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

    # ========== UPDATE WARNA BULAN ==========
    for i, b in enumerate(bulan_labels):
        month_data = df_bulanan[df_bulanan["Bulan"] == (i + 1)]
        if not month_data.empty:
            total_rr = month_data["RR_Prediksi"].values[0]
        else:
            total_rr = 0

        if total_rr > 300:   # total bulanan > 300 mm = basah
            color = "#3498db"  # biru (basah)
        elif total_rr >= 150:
            color = "#2ecc71"  # hijau (lembab)
        else:
            color = "#e74c3c"  # merah (kering)

        placeholders_bulan[i].markdown(
            f"<div style='background-color:{color}; "
            f"padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
            unsafe_allow_html=True
        )

    # ========== TABEL BULANAN ==========
    st.subheader("📊 Akumulasi Curah Hujan Bulanan (2025)")
    st.dataframe(df_bulanan)

    # ========== DOWNLOAD ==========
    csv_bulanan = df_bulanan.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Peramalan Bulanan", csv_bulanan, "hasil_peramalan_bulanan.csv", "text/csv")



