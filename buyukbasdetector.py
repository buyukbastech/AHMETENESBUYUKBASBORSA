import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import time

# -----------------------------
# Ayarlar
# -----------------------------
st.set_page_config(page_title="Anomali Tespiti (Durum 2) - Binance", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>📊 Anomali Tespiti (Durum 2) - Binance Verisi</h1>
    <p style='text-align:center;'>Binance (public API) üzerinden alınan mum kapanışlarına göre anomali tespiti</p>
""", unsafe_allow_html=True)

# Denenecek alternatif base URL'ler (geo-block atlatma)
BASE_URLS = [
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.com"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AnomalyDetector/1.0; +https://github.com/)"
}

# interval -> ms mapping (used for pagination step)
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000
}

# -----------------------------
# Kullanıcı Girdileri
# -----------------------------
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("📈 Parite giriniz (örn: BTCUSDT, ETHUSDT):", "BTCUSDT").strip().upper()
with col2:
    interval = st.selectbox("⏱️ Zaman Aralığı:", ["15m", "30m", "1h", "4h", "1d"], index=0)

days = st.slider("📅 Son kaç günlük veri alınsın?", 1, 90, 60)
contamination = st.slider("🔴 Anomali oranı (IsolationForest contamination)", 0.001, 0.2, 0.03, step=0.001)

# -----------------------------
# Helper: paginate klines (handles Binance 1000-limit)
# -----------------------------
def fetch_klines_paginated(base_url: str, symbol: str, interval: str, start_ms: int, end_ms: int, headers: dict, limit: int = 1000, timeout: int = 10):
    """
    Paginate Binance /klines endpoint to fetch all data between start_ms and end_ms
    Returns list of kline arrays (same format as API)
    """
    all_klines = []
    cur_start = start_ms
    interval_ms = INTERVAL_MS.get(interval)
    if interval_ms is None:
        raise ValueError(f"Interval '{interval}' not supported for pagination.")

    while cur_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": limit
        }
        url = f"{base_url}/api/v3/klines"
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code != 200:
            # return status & message so caller can decide to try next base_url
            return {"error": True, "status_code": resp.status_code, "text": resp.text}

        data = resp.json()
        if not data:
            break

        all_klines.extend(data)

        # advance cur_start: last kline open time + interval_ms
        last_open_time = int(data[-1][0])
        # If the API returned less than limit and the last kline is at or beyond end_ms, we can break
        if last_open_time + interval_ms >= end_ms:
            break

        # avoid infinite loops: if we got the same last_open_time again, break
        cur_start = last_open_time + interval_ms
        # small sleep to be polite
        time.sleep(0.05)

        # safety: prevent extremely long loops
        if len(all_klines) > 20000:
            break

    return {"error": False, "data": all_klines}

# -----------------------------
# cache results to avoid repeated heavy calls
# -----------------------------
@st.cache_data(ttl=300)  # cached for 5 minutes
def get_klines_from_binance(symbol: str, interval: str, days: int):
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    last_error = None
    for base in BASE_URLS:
        try:
            res = fetch_klines_paginated(base, symbol, interval, start_time, end_time, HEADERS)
            if isinstance(res, dict) and res.get("error"):
                status = res.get("status_code")
                # Respect specific status responses: try next base if geo-block or forbidden or rate-limit
                last_error = (base, status, res.get("text"))
                # try next base_url
                continue
            # success
            return {"success": True, "base_url": base, "klines": res["data"]}
        except requests.exceptions.RequestException as e:
            last_error = (base, "request_exception", str(e))
            continue
        except Exception as e:
            last_error = (base, "exception", str(e))
            continue

    return {"success": False, "error": last_error}

# -----------------------------
# Main action
# -----------------------------
if st.button("🚀 Veriyi Al ve Anomali Tespit Et"):
    if not symbol:
        st.error("Lütfen geçerli bir parite girin (örn: BTCUSDT).")
    else:
        with st.spinner("📡 Veri çekiliyor — birden fazla endpoint deneniyor, sabırlı olun..."):
            result = get_klines_from_binance(symbol, interval, days)

        if not result.get("success"):
            base_err = result.get("error")
            st.error(f"❌ Binance API yanıtını veremedi. Son denenen base: {base_err}")
        else:
            base_used = result.get("base_url")
            raw_klines = result.get("klines")
            if not raw_klines:
                st.error("❌ Veri bulunamadı (boş sonuç). Parite veya zaman aralığını kontrol edin.")
            else:
                # convert to DataFrame
                df = pd.DataFrame(raw_klines, columns=[
                    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                ])
                df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
                df['Close'] = df['Close'].astype(float)

                st.success(f"✅ {symbol} için {len(df)} satır veri alındı. (kaynak: {base_used})")
                st.markdown(f"**İlk** veri: {df['Open time'].iloc[0]} — **Son** veri: {df['Open time'].iloc[-1]}")

                # Anomali tespiti
                try:
                    clf = IsolationForest(contamination=float(contamination), random_state=42)
                    df['Anomaly'] = clf.fit_predict(df[['Close']])
                    df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
                    anomalies = df[df['Anomaly'] == 1]
                except Exception as e:
                    st.error(f"Anomali modelinde hata oluştu: {e}")
                    anomalies = pd.DataFrame(columns=df.columns)

                # Grafik
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Open time'], y=df['Close'],
                    mode='lines', name='Kapanış Fiyatı', line=dict(color='cyan')
                ))
                fig.add_trace(go.Scatter(
                    x=anomalies['Open time'], y=anomalies['Close'] if not anomalies.empty else [],
                    mode='markers', name='Anomaliler',
                    marker=dict(color='red', size=8, line=dict(color='white', width=1))
                ))
                fig.update_layout(
                    title=f"{symbol} Fiyat Grafiği ve Anomaliler (kaynak: {base_used})",
                    xaxis_title="Zaman",
                    yaxis_title="Kapanış Fiyatı",
                    template="plotly_dark",
                    hovermode="x unified",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table & CSV download
                st.subheader("📍 Anomali Tespit Edilen Zamanlar (son 50):")
                st.dataframe(anomalies[['Open time', 'Close']].tail(50))

                # CSV download
                csv_buffer = io.StringIO()
                anomalies[['Open time', 'Close']].to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode('utf-8')
                st.download_button("📥 Anomalileri CSV indir", data=csv_bytes, file_name=f"{symbol}_anomalies.csv", mime="text/csv")

                # Raw data download (optional)
                raw_buf = io.StringIO()
                df.to_csv(raw_buf, index=False)
                st.download_button("📥 Tüm Veriyi CSV indir", data=raw_buf.getvalue().encode('utf-8'),
                                   file_name=f"{symbol}_klines.csv", mime="text/csv")

# Footer / help
st.markdown("---")
st.markdown("""**Notlar:**  
- Eğer Streamlit Cloud'da hâlâ `451` veya `403` görürseniz, uygulama hangi base_url'ı denediğini logs'ta görebilirsiniz.  
- Yerelde (`streamlit run buyukbasdetector.py`) çalıştırarak da test edebilirsiniz; yerel IP'den genelde bu kısıtlama yoktur.  
- Çok büyük veri çekimlerinde Binance rate limit uygulanabilir; bu yüzden aynı sorguyu sık tekrarlamaktan kaçının (uygulama sonuçları 5 dk önbelleğe alınır).  
""")
