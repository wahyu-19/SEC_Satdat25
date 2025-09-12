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
    df_forecast['Bulan_Angka'] = df_forecast['TANGGAL'].dt.month

    df_bulanan = (
        df_forecast.groupby(['Tahun', 'Bulan_Angka'])['RR_Prediksi']
        .sum()
        .reset_index()
    )

    # Ganti angka bulan jadi nama bulan
    df_bulanan['Bulan'] = pd.to_datetime(df_bulanan['Bulan_Angka'], format='%m').dt.month_name()

    # Rename kolom prediksi
    df_bulanan = df_bulanan.rename(columns={"RR_Prediksi": "Prediksi Curah Hujan"})

    # Pastikan Tahun tidak ada koma ribuan
    df_bulanan["Tahun"] = df_bulanan["Tahun"].astype(str)

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

st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

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

    # ========== Tambah klasifikasi bulanan ==========
    def klasifikasi_bulanan(rr):
        if rr > 200:
            return "Bulan Basah", "#27ae60"  # hijau
        elif rr >= 100:
            return "Bulan Lembab", "#f1c40f"  # kuning
        else:
            return "Bulan Kering", "#e74c3c"  # merah

    df_bulanan[["Klasifikasi", "Warna"]] = df_bulanan["Prediksi Curah Hujan"].apply(
        lambda x: pd.Series(klasifikasi_bulanan(x))
    )

    # Update warna kalender sesuai forecast 2025
    for i in range(12):
        month_data = df_bulanan[(df_bulanan["Tahun"] == "2025") & (df_bulanan["Bulan_Angka"] == (i + 1))]
        if not month_data.empty:
            warna = month_data["Warna"].values[0]
            klasifikasi = month_data["Klasifikasi"].values[0]
            placeholders_bulan[i].markdown(
                f"<div style='background-color:{warna}; "
                f"padding:10px; border-radius:8px; text-align:center; color:white;'>{bulan_labels[i]}<br><small>{klasifikasi}</small></div>",
                unsafe_allow_html=True
            )

    # ========== Fungsi klasifikasi tahunan ==========
    def klasifikasi_tahunan(group):
        bulan_basah = (group["Klasifikasi"] == "Bulan Basah").sum()
        bulan_kering = (group["Klasifikasi"] == "Bulan Kering").sum()

        # huruf
        if bulan_basah > 9:
            huruf = "A"
        elif 7 <= bulan_basah <= 9:
            huruf = "B"
        elif 5 <= bulan_basah <= 6:
            huruf = "C"
        elif 3 <= bulan_basah <= 4:
            huruf = "D"
        else:
            huruf = "E"

        # angka
        if bulan_kering < 2:
            angka = "1"
        elif 2 <= bulan_kering <= 3:
            angka = "2"
        elif 4 <= bulan_kering <= 6:
            angka = "3"
        elif 7 <= bulan_kering <= 9:
            angka = "4"
        else:
            angka = "5"

        tipe = f"{huruf}{angka}"

        # mapping rekomendasi
        rekom_dict = {
            "A1": "Sesuai untuk padi terus menerus tetapi produksi kurang karena radiasi surya rendah sepanjang tahun",
            "A2": "Sesuai untuk padi terus menerus tetapi produksi kurang karena radiasi surya rendah sepanjang tahun",
            "B1": "Sesuai untuk padi terus menerus dengan perencanaan awal musim tanam yang baik; produksi tinggi bila panen musim kemarau",
            "B2": "Sesuai untuk tanam padi dua kali setahun dengan varietas umur pendek dan musim kering pendek cukup untuk palawija",
            "B3": "Sesuai untuk tanam padi dua kali setahun dengan varietas umur pendek dan musim kering pendek cukup untuk palawija",
            "C1": "Sesuai untuk tanam padi sekali dan dua kali palawija dalam setahun",
            "C2": "Sesuai untuk tanam padi sekali dan dua kali palawija; palawija kedua tidak boleh musim kering",
            "C3": "Sesuai untuk tanam padi sekali dan dua kali palawija; palawija kedua tidak boleh musim kering",
            "C4": "Sesuai untuk tanam padi sekali dan dua kali palawija; palawija kedua tidak boleh musim kering",
            "D1": "Sesuai untuk tanam padi umur pendek sekali dengan produksi tinggi + sekali palawija",
            "D2": "Sesuai untuk sekali tanam padi atau sekali palawija, tergantung irigasi",
            "D3": "Sesuai untuk sekali tanam padi atau sekali palawija, tergantung irigasi",
            "D4": "Sesuai untuk sekali tanam padi atau sekali palawija, tergantung irigasi",
            "E1": "Sesuai untuk sekali tanam palawija, tergantung adanya hujan",
            "E2": "Sesuai untuk sekali tanam palawija, tergantung adanya hujan",
            "E3": "Sesuai untuk sekali tanam palawija, tergantung adanya hujan",
            "E4": "Sesuai untuk sekali tanam palawija, tergantung adanya hujan",
            "E5": "Sesuai untuk sekali tanam palawija, tergantung adanya hujan",
        }

        rekom = rekom_dict.get(tipe, "Tidak ada rekomendasi")
        return pd.Series({"Tipe_Iklim": tipe, "Rekomendasi": rekom})

    # Terapkan per tahun
    hasil_klasifikasi = df_bulanan.groupby("Tahun").apply(klasifikasi_tahunan).reset_index()

    # ========== Tampilkan hasil ==========
    st.subheader("📊 Hasil Peramalan Bulanan")
    st.dataframe(df_bulanan[["Tahun", "Bulan", "Prediksi Curah Hujan", "Klasifikasi"]])

    st.subheader("🌦️ Rekomendasi Tanam")
    for _, row in hasil_klasifikasi.iterrows():
        st.markdown(
            f"<p style='background-color:#ecf0f1; padding:10px; border-radius:8px;'>"
            f"➡️ Tahun <b>{row['Tahun']}</b> termasuk tipe iklim <b>{row['Tipe_Iklim']}</b>. "
            f"Rekomendasi: <i>{row['Rekomendasi']}</i></p>",
            unsafe_allow_html=True
        )

    # Tombol download hanya untuk hasil peramalan bulanan
    csv_bulanan = df_bulanan.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Hasil Peramalan Bulanan",
        csv_bulanan,
        "hasil_peramalan_bulanan.csv",
        "text/csv"
    )
