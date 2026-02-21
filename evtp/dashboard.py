# dashboard.py
from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="EV Telemetry Predict", layout="wide")
st.title("🔧 EV Telemetry Predict")

vin = st.sidebar.text_input("VIN", "EV001")
n = st.sidebar.slider("Rows to display", 100, 5000, 1000, 100)
api_n = st.sidebar.slider("Rows to score", 1, 500, 100, 10)

# --- load raw data for charts ---
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root if dashboard is in evtp/
DB_PATH = BASE_DIR / "data" / "ev_telemetry.db"
con = con = sqlite3.connect(DB_PATH)
raw = pd.read_sql(
    "SELECT * FROM raw WHERE vin = ? ORDER BY timestamp DESC LIMIT ?",
    con,
    params=[vin, n],
    parse_dates=["timestamp"],
)
con.close()

if raw.empty:
    st.error(f"No raw data found for VIN={vin}. Run test.py to generate + ETL.")
    st.stop()

raw = raw.sort_values("timestamp")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Speed (km/h)", f"{raw['speed_kmh'].iloc[-1]:.0f}")
c2.metric("SOC (%)", f"{raw['soc_pct'].iloc[-1]:.0f}")
c3.metric("Battery Temp (°C)", f"{raw['battery_temp_c'].iloc[-1]:.1f}")
c4.metric("Inverter Temp (°C)", f"{raw['inverter_temp_c'].iloc[-1]:.1f}")

st.subheader("Speed & SOC")
st.line_chart(raw.set_index("timestamp")[["speed_kmh", "soc_pct"]])

st.subheader("Temperatures")
st.line_chart(raw.set_index("timestamp")[["battery_temp_c", "inverter_temp_c", "ambient_temp_c"]])

# --- call API for risk scores ---
st.subheader("Predicted Risk (from API)")
try:
    resp = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"vin": vin, "n": api_n},
        timeout=5,
    )
    resp.raise_for_status()
    risk = resp.json()["risk"]
    # match risk to timestamps (latest api_n rows)
    risk_df = raw.tail(api_n)[["timestamp"]].copy()
    risk_df["risk"] = list(reversed(risk))  # API returns newest-first; align with ascending timestamps
    st.line_chart(risk_df.set_index("timestamp")[["risk"]])
    st.write("Latest risk:", float(risk_df["risk"].iloc[-1]))
except Exception as e:
    st.info("Start the API first: `uvicorn evtp.service:app --reload --port 8000`")
    st.write("Error:", str(e))