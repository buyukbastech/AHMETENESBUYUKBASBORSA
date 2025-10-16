import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Anomali Tespiti (CoinDesk API)", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>📊 Anomali Tespiti (CoinDesk API)</h1>
<p style='text-align:center;'>CoinDesk Data API üzerinden alınan fiyat verileriyle anomali tespiti yapılır.</p>
""", unsafe_allow_html=True)

# Kullanıcı ayarları
symbol = st.text_input("Kripto Sembolü (örn: BTC, ETH):", "BTC").upper()
vs_currency = st.text_input("Karşı Para (örn: USD):", "USD").upper()
interval = st.selectbox("Zaman Aralığı:", ["15m", "30m", "1h", "4h", "1d"])
days = st.slider("Kaç günlük veri alınsın:", 10, 90, 60)
contamination = st.slider("Anomali oranı:", 0.001, 0.2, 0.03, step=0.001)

# Hazır API anahtarı (kullanıcıdan istenmez)
API_KEY = "9b81bbe2f519e5e147866732d3ba26940035de9ecc90cefd9c7fb320adfc527a"
BASE_URL = "https://data-api.coindesk.com/api/v1/price/ohlc"

def get_data(symbol, vs_currency, interval, days):
    params = {
        "symbol": f"{symbol}-{vs_currency}",
        "period": interval,
        "days": days,
        "api_key": API_KEY
    }
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            return None, f"Hata kodu: {response.status_code}\n{response.text}"
        data = response.json()
        if "data" not in data or not data["data"]:
            return None, "Veri bulunamadı."
        df = pd.DataFrame(data["data"], columns=["timestamp", "Open", "High", "Low", "Close"])
        df["Open time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["Close"] = df["Close"].astype(float)
        df = df.sort_values("Open time").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)

if st.button("🚀 Veriyi Al ve Anomali Tespit Et"):
    with st.spinner("📡 Veri alınıyor..."):
        df, error = get_data(symbol, vs_currency, interval, days)
    if error:
        st.error(f"❌ CoinDesk API hatası: {error}")
    elif df is None or df.empty:
        st.warning("Veri alınamadı.")
    else:
        st.success(f"✅ {symbol}/{vs_currency} için {len(df)} satır veri alındı.")
        clf = IsolationForest(contamination=contamination, random_state=42)
        df["Anomaly"] = clf.fit_predict(df[["Close"]])
        df["Anomaly"] = df["Anomaly"].apply(lambda x: 1 if x == -1 else 0)
        anomalies = df[df["Anomaly"] == 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Open time"], y=df["Close"], mode="lines", name="Fiyat", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=anomalies["Open time"], y=anomalies["Close"], mode="markers", name="Anomaliler", marker=dict(color="red", size=8)))
        fig.update_layout(template="plotly_dark", title=f"{symbol}/{vs_currency} Fiyat Grafiği ve Anomaliler", xaxis_title="Zaman", yaxis_title=f"Fiyat ({vs_currency})")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📍 Anomaliler (Son 20):")
        st.dataframe(anomalies[["Open time", "Close"]].tail(20))

        csv = io.StringIO()
        anomalies[["Open time", "Close"]].to_csv(csv, index=False)
        st.download_button("📥 Anomalileri CSV olarak indir", data=csv.getvalue().encode(), file_name=f"{symbol}_anomalies.csv", mime="text/csv")
