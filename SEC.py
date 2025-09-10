import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_data(df):
    df = df.replace([8888, 9999, "-"], np.nan)
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
    rr_median = df['RR'].median()
    df['RR'] = df['RR'].fillna(rr_median)
    return df

def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# ---------------------------
# Forecasting
# ---------------------------
def proses_peramalan(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)

    dataset = df[['RR']].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    look_back = 30
    trainX, trainY = create_dataset(dataset_scaled, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # Forecast 365 hari
    predictions = []
    last_data = dataset_scaled[-look_back:]
    current_input = np.reshape(last_data, (1, look_back, 1))

    for _ in range(365):
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[next_pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    start_date = datetime.date(2025, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
    df_forecast = pd.DataFrame({"Tanggal": dates, "Curah Hujan": forecast.flatten()})

    return df, df_forecast

# ---------------------------
# Warna kotak bulan
# ---------------------------
def get_color(value):
    if value > 200:
        return "#1E90FF"  # biru
    elif 100 <= value <= 200:
        return "#2E8B57"  # hijau
    else:
        return "#FF8C00"  # oranye

# ---------------------------
# Tampilan Streamlit
# ---------------------------
st.set_page_config(page_title="AgroForecast", layout="wide")

st.markdown("<h1 style='text-align: center;'>🌱 AGROFORECAST</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Kalender Musim Tanam (Basah - Lembab - Kering)</h3>", unsafe_allow_html=True)

bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
         "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# Kotak bulan default abu-abu
month_cols = st.columns(12)
for i, b in enumerate(bulan):
    month_cols[i].markdown(
        f"<div style='padding:10px; border-radius:8px; text-align:center; background-color:gray; color:white'>{b}</div>",
        unsafe_allow_html=True
    )

# Input data
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
with col2:
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

# Jika ada file
if uploaded_file is not None:
    df, df_forecast = proses_peramalan(uploaded_file)

    # Hitung rata-rata bulanan
    df_forecast['Bulan'] = df_forecast['Tanggal'].dt.month
    monthly_avg = df_forecast.groupby('Bulan')['Curah Hujan'].mean()

    st.success(f"Rekomendasi subsidi bibit: {luas_lahan*5.5:.1f} ton 🌱")

    # Update kotak bulan dengan warna forecast
    month_cols = st.columns(12)
    for i, b in enumerate(bulan):
        if (i+1) in monthly_avg.index:
            value = monthly_avg[i+1]
            color = get_color(value)
            month_cols[i].markdown(
                f"<div style='padding:10px; border-radius:8px; text-align:center; background-color:{color}; color:white'>{b}</div>",
                unsafe_allow_html=True
            )
        else:
            month_cols[i].markdown(
                f"<div style='padding:10px; border-radius:8px; text-align:center; background-color:gray; color:white'>{b}</div>",
                unsafe_allow_html=True
            )
