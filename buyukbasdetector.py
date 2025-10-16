import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------
# Streamlit Sayfa Ayarları
# -----------------------------
st.set_page_config(page_title="Anomali Tespiti (Case 2)", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>📊 Anomali Tespiti (Case 2) - Binance Verisi</h1>
    <p style='text-align:center;'>Binance API üzerinden alınan mum kapanışlarına göre anomali tespiti</p>
""", unsafe_allow_html=True)

# -----------------------------
# Kullanıcı Girdileri
# -----------------------------
symbol = st.text_input("📈 Parite giriniz (örn: BTCUSDT, ETHUSDT):", "BTCUSDT")
interval = st.selectbox("⏱️ Zaman Aralığı Seçiniz:", ["15m", "30m", "1h", "4h", "1d"], index=0)
days = st.slider("📅 Son kaç günlük veri alınsın?", 10, 90, 60)

# -----------------------------
# Ana İşlem
# -----------------------------
if st.button("🚀 Veriyi Al ve Anomali Tespit Et"):
    try:
        with st.spinner("📡 Binance verileri alınıyor, lütfen bekleyin..."):
            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

            # Binance Public API (anahtarsız)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}"
            response = requests.get(url)

            if response.status_code != 200:
                st.error(f"❌ Binance API yanıt vermedi (Status Code: {response.status_code})")
            else:
                data = response.json()
                if not data:
                    st.error("❌ Veri bulunamadı. Lütfen farklı bir parite veya zaman aralığı seçin.")
                else:
                    df = pd.DataFrame(data, columns=[
                        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Close time', 'Quote asset volume', 'Number of trades',
                        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                    ])

                    # Zaman & veri formatlama
                    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
                    df['Close'] = df['Close'].astype(float)

                    st.success(f"✅ {symbol} için {len(df)} satır veri alındı ({interval} - {days} gün).")

                    # -----------------------------
                    # Anomali Tespiti (IsolationForest)
                    # -----------------------------
                    clf = IsolationForest(contamination=0.03, random_state=42)
                    df['Anomaly'] = clf.fit_predict(df[['Close']])
                    df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
                    anomalies = df[df['Anomaly'] == 1]

                    # -----------------------------
                    # Görselleştirme
                    # -----------------------------
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['Open time'], y=df['Close'],
                        mode='lines', name='Kapanış Fiyatı',
                        line=dict(color='cyan')
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

                    # -----------------------------
                    # Tablo Gösterimi
                    # -----------------------------
                    st.subheader("📍 Anomali Tespit Edilen Zamanlar:")
                    st.dataframe(anomalies[['Open time', 'Close']].tail(20))

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
