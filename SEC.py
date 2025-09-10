import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

# ---------------------------
# Fungsi preprocessing
# ---------------------------
def preprocess_data(df):
    # Ganti 8888, 9999, "-" jadi NaN
    df = df.replace([8888, 9999, "-"], np.nan)

    # Pastikan RR numerik
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')

    # Isi NaN dengan median
    rr_median = df['RR'].median()
    df['RR'] = df['RR'].fillna(rr_median)

    return df

# ---------------------------
# Dataset ke supervised learning
# ---------------------------
def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# ---------------------------
# Proses peramalan
# ---------------------------
def proses_peramalan(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Preprocessing data
    df = preprocess_data(df)

    # Ambil kolom RR
    dataset = df[['RR']].values.astype('float32')

    # Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    # Buat dataset supervised
    look_back = 30
    trainX, trainY = create_dataset(dataset_scaled, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # Model LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # Forecast 365 hari (2025)
    predictions = []
    last_data = dataset_scaled[-look_back:]
    current_input = np.reshape(last_data, (1, look_back, 1))

    for _ in range(365):
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[next_pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Buat DataFrame hasil forecast
    start_date = datetime.date(2025, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
    df_forecast = pd.DataFrame({"Tanggal": dates, "Curah Hujan": forecast.flatten()})

    return df, df_forecast

# ---------------------------
# Warna berdasarkan curah hujan
# ---------------------------
def get_color(value):
    if value > 200:
        return "background-color: blue; color: white; font-weight: bold;"
    elif 100 <= value <= 200:
        return "background-color: green; color: white; font-weight: bold;"
    else:
        return "background-color: orange; color: white; font-weight: bold;"

# ---------------------------
# Tampilan Streamlit
# ---------------------------
st.set_page_config(page_title="AgroForecast", layout="wide")

st.markdown("<h1 style='text-align: center;'>🌱 AGROFORECAST</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Kalender Musim Tanam (Basah - Lembab - Kering)</h3>", unsafe_allow_html=True)

# Kotak bulan default (abu-abu)
bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
         "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
cols = st.columns(12)
for i, b in enumerate(bulan):
    cols[i].button(b, key=f"default_{i}", disabled=True)

st.write("")

uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    # Ambil rata-rata forecast per bulan
    df_forecast['Bulan'] = df_forecast['Tanggal'].dt.month
    monthly_avg = df_forecast.groupby('Bulan')['Curah Hujan'].mean()

    st.subheader("Rekomendasi Subsidi Bibit")
    rekomendasi = luas_lahan * 5.5  # contoh hitungan ton bibit
    st.success(f"Rekomendasi subsidi bibit: {rekomendasi:.1f} ton 🌱")

    # Kotak bulan dengan warna sesuai hasil forecast
    cols = st.columns(12)
    for i, b in enumerate(bulan):
        if (i+1) in monthly_avg.index:
            value = monthly_avg[i+1]
            style = get_color(value)
            cols[i].markdown(
                f"<div style='padding:10px; border-radius:8px; text-align:center; {style}'>{b}</div>",
                unsafe_allow_html=True
            )
        else:
            cols[i].markdown(
                f"<div style='padding:10px; border-radius:8px; text-align:center; background-color:gray; color:white'>{b}</div>",
                unsafe_allow_html=True
            )
