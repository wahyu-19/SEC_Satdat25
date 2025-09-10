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
        df = pd.read_csv(file)  # <-- pakai CSV
        if not set(["TANGGAL", "RR"]).issubset(df.columns):
            st.error("File CSV harus ada kolom 'TANGGAL' dan 'RR'")
            st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL']).sort_values("TANGGAL")

    df_nan = df.dropna(subset=['RR'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_nan['RR_norm'] = scaler.fit_transform(df_nan[['RR']])

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

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    with st.spinner("Training model LSTM..."):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # ========== FORECAST 365 HARI ==========
    forecast = []
    last_data = dataset[-look_back:].reshape(1, look_back, 1)
    for _ in range(365):
        next_pred = model.predict(last_data, verbose=0)
        forecast.append(next_pred[0][0])
        next_pred = next_pred.reshape(1, 1, 1)
        last_data = np.concatenate((last_data[:, 1:, :], next_pred), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # ====== SET MULAI TANGGAL 2025 ======
    last_date = df_nan['TANGGAL'].iloc[-1]
    start_date = pd.Timestamp(year=2025, month=1, day=1)
    if last_date.year >= 2025:
        start_date = last_date + pd.Timedelta(days=1)

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
st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")

# Upload data + luas lahan
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data Curah Hujan")
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
with col2:
    st.subheader("Luas Lahan")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

# Proses peramalan
if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    # ========== HASIL REKOMENDASI ==========
    median_pred = df_forecast["RR_Prediksi"].median()
    if median_pred >= 100:
        hasil_padi = "✅ Cocok untuk tanam Padi 🌾"
    else:
        hasil_padi = "❌ Tidak disarankan tanam Padi"

    if 50 <= median_pred < 200:
        hasil_palawija = "✅ Cocok untuk tanam Palawija 🌽"
    else:
        hasil_palawija = "❌ Tidak disarankan tanam Palawija"

    col3, col4 = st.columns(2)
    with col3:
        st.success(hasil_padi)
    with col4:
        st.info(hasil_palawija)

    # ========== REKOMENDASI SUBSIDI ==========
    if luas_lahan > 0:
        rekomendasi = int(luas_lahan * median_pred / 100)
        st.markdown(
            f"<h3 style='text-align:center;'>Rekomendasi subsidi bibit : {rekomendasi} ton 🌱</h3>",
            unsafe_allow_html=True
        )

    # ========== TAMPILKAN BULAN DENGAN WARNA ==========
    st.markdown("---")
    st.subheader("📅 Kalender Bulanan 2025")

    bulan_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    cols = st.columns(12)

    for i, b in enumerate(bulan_labels):
        month_data = df_forecast[df_forecast["TANGGAL"].dt.month == (i+1)]
        mean_rr = month_data["RR_Prediksi"].mean()

        if mean_rr > 200:
            color = "#3498db"  # biru
        elif mean_rr >= 100:
            color = "#2ecc71"  # hijau
        else:
            color = "#e74c3c"  # merah

        with cols[i]:
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:8px; text-align:center; color:white;'>{b}</div>",
                unsafe_allow_html=True
            )

    # ========== TABEL FORECAST ==========
    st.subheader("📈 Hasil Peramalan 365 Hari (2025)")
    st.dataframe(df_forecast)

    csv = df_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Peramalan", csv, "hasil_peramalan.csv", "text/csv")
