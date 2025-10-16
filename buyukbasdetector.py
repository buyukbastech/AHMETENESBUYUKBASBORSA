import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# -------------------------------------
# Streamlit ayarları
# -------------------------------------
st.set_page_config(page_title="Anomali Tespiti (CoinDesk API)", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>📊 Anomali Tespiti (Case 2) - CoinDesk API</h1>
    <p style='text-align:center;'>CoinDesk Data API üzerinden alınan kripto fiyat verileriyle otomatik anomali tespiti</p>
""", unsafe_allow_html=True)

# -------------------------------------
# Kullanıcı girdi alanları
# -------------------------------------
symbol = st.text_input("📈 Kripto sembolü (örn: BTC, ETH, SOL):", "BTC").strip().upper()
vs_currency = st.text_input("💵 Karşı para birimi:", "USD").strip().upper()
interval = st.selectbox("⏱️ Zaman Aralığı:", ["15m", "30m", "1h", "4h", "1d"], index=0)
days = st.slider("📅 Kaç günlük veri alınsın?", 10, 90, 60)
contamination = st.slider("🔴 Anomali oranı:", 0.001, 0.2, 0.03, step=0.001)

# -------------------------------------
# CoinDesk API (senin anahtarın ile)
# -------------------------------------
API_KEY = "9b81bbe2f519e5e147866732d3ba26940035de9ecc90cefd9c7fb320adfc527a"
BASE_URL = "https://data-api.coindesk.com/api/v1"

# -------------------------------------
# Veri çekme fonksiyonu
# -------------------------------------
def get_coindesk_data(symbol: str, vs_currency: str, days: int, interval: str):
    # CoinDesk API OHLC endpoint (örnek: /prices/ohlc)
    url = f"{BASE_URL}/price/ohlc"
    params = {
        "symbol": f"{symbol}-{vs_currency}",
        "period": interval,
        "days": days,
        "api_key": API_KEY
    }
    headers = {"Content-Type": "application/json; charset=UTF-8"}

    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        return {"error": True, "status": r.status_code, "text": r.text}

    data = r.json()
    if "data" not in data or not data["data"]:
        return {"error": True, "status": 204, "text": "no data"}

    df = pd.DataFrame(data["data"], columns=["timestamp", "Open", "High", "Low", "Close"])
    df["Open time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["Close"] = df["Close"].astype(float)
    df = df.sort_values("Open time").reset_index(drop=True)
    return {"error": False, "df": df}

# -------------------------------------
# Ana işlem
# -------------------------------------
if st.button("🚀 Veriyi Al ve Anomali Tespit Et"):
    with st.spinner("📡 CoinDesk verileri alınıyor..."):
        res = get_coindesk_data(symbol, vs_currency, days, interval)

    if res.get("error"):
        st.error(f"❌ CoinDesk API hatası ({res.get('status')}): {res.get('text')}")
    else:
        df = res["df"]
        st.success(f"✅ {symbol}/{vs_currency} için {len(df)} satır veri alındı ({interval}, {days} gün).")

        # -------------------------------
        # Anomali Tespiti
        # -------------------------------
        try:
            clf = IsolationForest(contamination=float(contamination), random_state=42)
            df['Anomaly'] = clf.fit_predict(df[['Close']])
            df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
            anomalies = df[df['Anomaly'] == 1]
        except Exception as e:
            st.error(f"Anomali modelinde hata oluştu: {e}")
            anomalies = pd.DataFrame(columns=df.columns)

        # -------------------------------
        # Grafik
        # -------------------------------
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
            title=f"{symbol}/{vs_currency} Fiyat Grafiği ve Anomaliler",
            xaxis_title="Zaman",
            yaxis_title=f"Fiyat ({vs_currency})",
            template="plotly_dark",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # Tablo + CSV indir
        # -------------------------------
        st.subheader("📍 Anomali Tespit Edilen Zamanlar:")
        st.dataframe(anomalies[['Open time', 'Close']].tail(20))

        csv_buf = io.StringIO()
        anomalies[['Open time', 'Close']].to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Anomalileri CSV olarak indir",
            data=csv_buf.getvalue().encode(),
            file_name=f"{symbol}_anomalies.csv",
            mime="text/csv"
        )
