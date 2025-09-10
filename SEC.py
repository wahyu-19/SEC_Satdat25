def proses_peramalan(file):
    try:
        df = pd.read_csv(file)
        if not set(["TANGGAL", "RR"]).issubset(df.columns):
            st.error("File CSV harus ada kolom 'TANGGAL' dan 'RR'")
            st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL']).sort_values("TANGGAL")

    # ==================================================
    # PREPROCESSING DATA
    # ==================================================
    # Ganti kode error (8888, 9999, "-") jadi NaN
    df['RR'] = df['RR'].replace([8888, 9999, "-"], np.nan)

    # Pastikan tipe datanya numerik
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')

    # Hitung jumlah error code
    count_8888 = (df == 8888).sum()
    count_9999 = (df == 9999).sum()

    st.write("Jumlah 8888 per kolom:\n", count_8888)
    st.write("Jumlah 9999 per kolom:\n", count_9999)

    # Ambil median dari nilai valid
    rr_median = df['RR'].median()

    # Imputasi NaN dengan median
    df['RR'] = df['RR'].fillna(rr_median)

    st.write("Median RR:", rr_median)
    st.write("Min RR setelah imputasi:", df['RR'].min())
    st.write("Max RR setelah imputasi:", df['RR'].max())

    # ==================================================
    # Normalisasi untuk training
    # ==================================================
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['RR_norm'] = scaler.fit_transform(df[['RR']])

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    dataset = df['RR_norm'].values.reshape(-1, 1)
    look_back = 30
    trainX, trainY = create_dataset(dataset, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # ==================================================
    # Model LSTM
    # ==================================================
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    with st.spinner("Training model LSTM..."):
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

    # Forecast 365 hari
    forecast = []
    last_data = dataset[-look_back:].reshape(1, look_back, 1)
    for _ in range(365):
        next_pred = model.predict(last_data, verbose=0)
        forecast.append(next_pred[0][0])
        last_data = np.append(last_data[:, 1:, :], [[next_pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Buat tanggal 2025
    future_dates = pd.date_range(start="2025-01-01", periods=365)
    df_forecast = pd.DataFrame({"TANGGAL": future_dates, "RR_Prediksi": forecast.flatten()})
    return df, df_forecast
