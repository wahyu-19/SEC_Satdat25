import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# ---------------------------
# Preprocessing (termasuk count 8888/9999 dan imputasi)
# ---------------------------
def preprocess_data(df):
    # Hitung jumlah 8888 / 9999 per kolom (tampilkan di UI)
    count_8888 = (df == 8888).sum()
    count_9999 = (df == 9999).sum()
    st.write("✅ Jumlah nilai 8888 per kolom:", dict(count_8888))
    st.write("✅ Jumlah nilai 9999 per kolom:", dict(count_9999))

    # Ganti sentinel dan "-" menjadi NaN di seluruh dataframe
    df_nan = df.replace([8888, 9999, "-"], np.nan).copy()

    # Pastikan kolom RR numeric
    df_nan['RR'] = pd.to_numeric(df_nan['RR'], errors='coerce')

    # Ambil median dari nilai valid (untuk imputasi)
    rr_median = df_nan['RR'].median()
    st.write(f"✅ Median RR sebelum imputasi: {rr_median}")

    # Isi NaN dengan median
    df_nan['RR'] = df_nan['RR'].fillna(rr_median)

    # Tampilkan statistik sederhana setelah imputasi
    st.write("✅ Statistik RR setelah imputasi:")
    st.write(f"Min: {df_nan['RR'].min():.3f}  |  Max: {df_nan['RR'].max():.3f}  |  Median: {df_nan['RR'].median():.3f}")

    return df_nan

# ---------------------------
# Utility: buat dataset supervised untuk LSTM
# ---------------------------
def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    # gunakan range(len - look_back) agar valid dan menghasilkan sample bila len>look_back
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    if len(dataX) == 0:
        return np.array([]), np.array([])
    return np.array(dataX), np.array(dataY)

# ---------------------------
# Fungsi utama proses peramalan
# ---------------------------
def proses_peramalan(file):
    # Baca CSV
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        return None, None

    # Validasi kolom
    if not set(["TANGGAL", "RR"]).issubset(df.columns):
        st.error("File CSV harus ada kolom 'TANGGAL' dan 'RR'")
        return None, None

    # Konversi tanggal dan sort
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL']).sort_values("TANGGAL").reset_index(drop=True)
    if df.empty:
        st.error("Kolom TANGGAL semua kosong / tidak valid.")
        return None, None

    # Preprocessing (imputasi sentinel, konversi)
    df_nan = preprocess_data(df)

    # Pastikan masih ada data RR setelah imputasi
    if df_nan['RR'].isna().all():
        st.error("Semua nilai RR kosong setelah preprocessing. Tidak dapat melanjutkan.")
        return df_nan, None

    # Normalisasi
    dataset = df_nan[['RR']].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    look_back = 30

    # Cek cukup panjang data
    if len(dataset_scaled) <= look_back:
        st.warning(
            f"Dataset terlalu pendek ({len(dataset_scaled)} baris). Dibutuhkan > {look_back} baris untuk LSTM. "
            "Menggunakan nilai median sebagai fallback untuk semua hari forecast."
        )
        avg_value = float(df_nan['RR'].median())
        start_date = df_nan['TANGGAL'].iloc[-1] + pd.Timedelta(days=1)
        dates = [start_date + pd.Timedelta(days=i) for i in range(365)]
        df_forecast = pd.DataFrame({"TANGGAL": dates, "RR_Prediksi": [avg_value] * 365})
        return df_nan, df_forecast

    # Siapkan data training
    trainX, trainY = create_dataset(dataset_scaled, look_back)
    if trainX.size == 0 or trainY.size == 0:
        st.warning("Set pelatihan kosong setelah pembuatan dataset. Menggunakan fallback median.")
        avg_value = float(df_nan['RR'].median())
        start_date = df_nan['TANGGAL'].iloc[-1] + pd.Timedelta(days=1)
        dates = [start_date + pd.Timedelta(days=i) for i in range(365)]
        df_forecast = pd.DataFrame({"TANGGAL": dates, "RR_Prediksi": [avg_value] * 365})
        return df_nan, df_forecast

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # Build model LSTM sederhana
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Training (simpan progress di UI)
    with st.spinner("🔄 Training model LSTM (sedang berlangsung)..."):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # Forecast 365 hari (autoregressive)
    predictions = []
    last_data = dataset_scaled[-look_back:]  # shape (look_back, 1)
    current_input = np.reshape(last_data, (1, look_back, 1))

    for _ in range(365):
        next_pred = model.predict(current_input, verbose=0)   # shape (1,1)
        val = float(next_pred[0][0])
        predictions.append(val)
        # update current_input dengan nilai prediksi (autoregressive)
        next_pred_reshaped = np.reshape([[val]], (1, 1, 1))
        current_input = np.append(current_input[:, 1:, :], next_pred_reshaped, axis=1)

    # Inverse transform prediksi ke skala asli
    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Tanggal mulai = hari setelah tanggal terakhir input
    start_date = df_nan['TANGGAL'].iloc[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=365)
    df_forecast = pd.DataFrame({"TANGGAL": future_dates, "RR_Prediksi": forecast})

    return df_nan, df_forecast

# ---------------------------
# Fungsi warna kotak bulan
# ---------------------------
def get_color(value):
    if value > 200:
        return "#3498db"  # biru (basah)
    elif value >= 100:
        return "#2ecc71"  # hijau (lembab)
    else:
        return "#e74c3c"  # merah (kering)

# ---------------------------
# Tampilan Streamlit
# ---------------------------
st.set_page_config(page_title="AgroForecast", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌱 AGROFORECAST</h1>", unsafe_allow_html=True)
st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")

bulan_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

cols_bulan = st.columns(12)
for i, b in enumerate(bulan_labels):
    with cols_bulan[i]:
        st.markdown(
            f"<div style='background-color:#7f8c8d; padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data Curah Hujan")
    uploaded_file = st.file_uploader("Upload File CSV (.csv) dengan kolom TANGGAL dan RR", type=["csv"])
with col2:
    st.subheader("Luas Lahan")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

# Proses peramalan
if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    if df is None:
        st.error("Gagal memproses file.")
        st.stop()

    if df_forecast is None or df_forecast.empty:
        st.warning("Tidak ada hasil forecast (df_forecast kosong).")
        st.stop()

    # Update warna kotak bulan berdasarkan hasil forecast
    # (gunakan kolom 'RR_Prediksi')
    monthly_avg = df_forecast.groupby(df_forecast['TANGGAL'].dt.month)['RR_Prediksi'].mean()

    month_cols = st.columns(12)
    for i, b in enumerate(bulan_labels):
        if (i+1) in monthly_avg.index:
            mean_rr = monthly_avg.loc[i+1]
            color = get_color(mean_rr)
        else:
            color = "#7f8c8d"
        with month_cols[i]:
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
                unsafe_allow_html=True
            )

    # Rekomendasi tanaman berdasarkan rata-rata tahunan forecast
    avg_annual = df_forecast['RR_Prediksi'].mean()
    st.write(f"Rata-rata RR forecast (1 tahun): {avg_annual:.2f} mm")

    col3, col4 = st.columns(2)
    with col3:
        if avg_annual > 150:
            st.success("Cocok untuk tanam Padi 🌾")
        else:
            st.info("Tidak disarankan tanam Padi berdasarkan rata-rata RR forecast")
    with col4:
        if 100 <= avg_annual <= 200:
            st.success("Cocok untuk tanam Palawija 🌽")
        else:
            st.info("Tidak disarankan tanam Palawija berdasarkan rata-rata RR forecast")

    # Rekomendasi subsidi bibit (contoh sederhana)
    if luas_lahan > 0:
        rekomendasi = int(luas_lahan * (avg_annual / 100))  # contoh formula sederhana
        st.markdown(f"<h3 style='text-align:center;'>Rekomendasi subsidi bibit : {rekomendasi} ton 🌱</h3>", unsafe_allow_html=True)

    # Tampilkan hasil forecast dan plot
    st.subheader("📈 Hasil Peramalan 365 Hari (mulai hari setelah data terakhir)")
    st.dataframe(df_forecast)

    

    # Download hasil
    csv = df_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Peramalan", csv, "hasil_peramalan.csv", "text/csv")
