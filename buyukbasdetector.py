import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

st.set_page_config(page_title="Anomali Tespiti (Case 2)", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>📊 Anomali Tespiti (Case 2) - Binance Verisi</h1>
    <p style='text-align:center;'>Binance API üzerinden alınan mum kapanışlarına göre anomali tespiti</p>
""", unsafe_allow_html=True)

# --- Binance public client (API anahtarı gerekmez)
client = Client()

symbol = st.text_input("📈 Parite giriniz (örn: BTCUSDT, ETHUSDT):", "BTCUSDT")
interval = st.selectbox("⏱️ Zaman Aralığı Seçiniz:", ["15m", "30m", "1h", "4h", "1d"], index=0)
days = st.slider("📅 Son kaç günlük veri alınsın?", 10, 90, 60)

if st.button("🚀 Veriyi Al ve Anomali Tespit Et"):
    try:
        # Binance'tan mum verilerini al
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")

        if not klines or len(klines) == 0:
            st.error("❌ Veri alınamadı. Lütfen pariteyi veya aralığı değiştirin.")
        else:
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(klines, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df['Close'] = df['Close'].astype(float)

            st.success(f"✅ {symbol} için {len(df)} satır veri alındı ({interval} - {days} gün)")

            # --- Anomali Tespiti ---
            clf = IsolationForest(contamination=0.03, random_state=42)
            df['Anomaly'] = clf.fit_predict(df[['Close']])
            df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
            anomalies = df[df['Anomaly'] == 1]

            # --- Grafik ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Open time'], y=df['Close'],
                mode='lines', name='Kapanış Fiyatı', line=dict(color='cyan')
            ))
            fig.add_trace(go.Scatter(
                x=anomalies['Open time'], y=anomalies['Close'],
                mode='markers', name='Anomaliler',
                marker=dict(color='red', size=8, line=dict(color='white', width=1))
            ))
            fig.update_layout(
                title=f"{symbol} Fiyat Grafiği ve Anomaliler",
                xaxis_title="Zaman",
                yaxis_title="Kapanış Fiyatı (USDT)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Tablo ---
            st.subheader("📍 Anomali Tespit Edilen Zamanlar:")
            st.dataframe(anomalies[['Open time', 'Close']].tail(20))

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
