import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===========================
# Judul Aplikasi
# ===========================
st.set_page_config(page_title="AgroForecast", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌱 AGROFORECAST</h1>", unsafe_allow_html=True)
st.write("Kalender Musim Tanam (Basah - Lembab - Kering)")

# ===========================
# Upload Data
# ===========================
uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.1, value=10.0, step=0.1)

# ===========================
# Fungsi Peramalan LSTM
# ===========================
def proses_peramalan(file):
    df = pd.read_csv(file)
    if not set(["TANGGAL", "RR"]).issubset(df.columns):
        st.error("CSV harus memiliki kolom 'TANGGAL' dan 'RR'")
        return None

    # Convert tanggal
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"])
    df = df.set_index("TANGGAL")

    # Normalisasi
    data = df["RR"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Buat dataset untuk LSTM
    def create_dataset(dataset, look_back=12):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            X.append(dataset[i:(i+look_back), 0])
            Y.append(dataset[i+look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 12
    X, y = create_dataset(data_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Model LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)

    # Prediksi 12 bulan ke depan
    last_data = data_scaled[-look_back:]
    predictions = []
    input_seq = last_data.reshape(1, look_back, 1)
    for i in range(12):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[:,1:,:], [[pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

    # Dataframe hasil prediksi
    bulan_list = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
    tahun_pred = 2025
    df_pred = pd.DataFrame({
        "Tahun": [tahun_pred]*12,
        "Bulan": bulan_list,
        "Prediksi Curah Hujan": forecast.flatten()
    })

    return df_pred

# ===========================
# Jalankan Forecast
# ===========================
if uploaded_file is not None:
    df_bulanan = proses_peramalan(uploaded_file)

    if df_bulanan is not None:
        st.subheader("📊 Hasil Peramalan Bulanan")
        st.dataframe(df_bulanan.head(10))  # tampilkan 10 bulan pertama

        # ===========================
        # Rekomendasi dalam kalimat
        # ===========================
        rata_hujan = df_bulanan["Prediksi Curah Hujan"].mean()

        if rata_hujan > 300:
            st.success(f"✅ Cocok untuk tanam **Padi** 🌾\n\nRekomendasi subsidi bibit: **{int(luas_lahan*3.2)} ton**")
        elif 200 <= rata_hujan <= 300:
            st.warning(f"⚠️ Cocok untuk tanam **Palawija** 🌽\n\nRekomendasi subsidi bibit: **{int(luas_lahan*2)} ton**")
        else:
            st.error("❌ Tidak disarankan menanam saat ini 🚫")

        # Tombol download hasil
        csv = df_bulanan.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Peramalan Bulanan",
            data=csv,
            file_name="hasil_peramalan.csv",
            mime="text/csv"
        )
