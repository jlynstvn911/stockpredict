import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

# Konfigurasi halaman
st.set_page_config(page_title="Stock Price Forecasting", layout="centered")

st.title("📈 Stock Price Forecasting with ARIMA")
st.write("Masukkan ticker saham dan rentang tanggal untuk melakukan forecasting harga saham.")

# Input dari pengguna
ticker = st.text_input("Masukkan ticker saham (misal: AAPL, GOTO.JK):")
start_date = st.date_input("Tanggal mulai", datetime(2020, 1, 1))
end_date = st.date_input("Tanggal selesai", datetime.today())
interval = st.selectbox("Pilih Interval Data", ("Harus pilih", "Daily", "Weekly", "Monthly"))

# Ubah interval jadi format yfinance
if interval == "Daily":
    data_interval = "1d"
elif interval == "Weekly":
    data_interval = "1wk"
elif interval == "Monthly":
    data_interval = "1mo"
else:
    data_interval = None

# Jika semua input valid
if ticker and start_date and end_date and data_interval:
    try:
        # Ambil data dari Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, interval=data_interval)

        if df.empty:
            st.error(f"Data untuk {ticker} tidak ditemukan.")
        else:
            st.subheader(f"Data Harga Saham {ticker} ({interval})")
            st.line_chart(df['Close'])

            st.subheader("ARIMA Forecasting")
            periods = st.slider("Forecast Period (months)", 1, 24, 6)

            try:
                # Menentukan order ARIMA secara otomatis
                st.text("Menentukan parameter ARIMA secara otomatis...")
                auto_model = auto_arima(
                    df['Close'],
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                order = auto_model.order
                st.success(f"Order ARIMA yang dipilih: {order}")

                # Fit ARIMA model dengan order yang ditemukan
                model = ARIMA(df['Close'], order=order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=periods)

                # Buat tanggal prediksi
                last_date = df.index[-1]
                forecast_dates = [last_date + timedelta(days=30 * i) for i in range(1, periods + 1)]

                # Buat DataFrame untuk forecast
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
                forecast_df.set_index('Date', inplace=True)

                # Gabungkan dengan data asli
                combined_df = pd.concat([df['Close'], forecast_df['Forecast']])
                st.line_chart(combined_df)

                st.write("Forecast Table:")
                st.dataframe(forecast_df)

            except Exception as e:
                st.error(f"Model Error: {e}")

    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")