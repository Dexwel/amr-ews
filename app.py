import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model import demo_data, prepare, forecast, risk_level, anomaly

st.set_page_config(page_title="AMR Early Warning", page_icon="🦠", layout="wide")
st.title("🦠 AMR Early Warning System")
st.caption("Forecasting antibiotic resistance and flagging risk early. Demo: E. coli vs ciprofloxacin.")

with st.sidebar:
    st.header("Settings")
    up = st.file_uploader("Upload CSV (date, tested, resistant)", type="csv")
    horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
    amber = st.number_input("Amber threshold (%)", 0.0, 100.0, 30.0)
    red = st.number_input("Red threshold (%)", 0.0, 100.0, 40.0)

raw = pd.read_csv(up) if up else demo_data()
if not up:
    st.info("Showing synthetic demo data. Upload your own CSV to use real data.")

df = prepare(raw)
fc, fit = forecast(df, horizon)
level, peak = risk_level(fc, amber, red)
flag, z = anomaly(df)

c1, c2, c3 = st.columns(3)
c1.metric("Latest resistance", f"{df['rate'].iloc[-1]:.1f}%")
c2.metric(f"Peak forecast ({horizon} mo)", f"{peak:.1f}%")
c3.metric("Alert level", {"GREEN": "🟢 GREEN", "AMBER": "🟠 AMBER", "RED": "🔴 RED"}[level])

if flag:
    st.warning(f"Sudden spike detected in the latest month (z = {z:.1f}). Verify the lab data.")
if level == "RED":
    st.error("Forecast exceeds the red threshold. Recommend review of prescribing guidelines and stewardship action.")
elif level == "AMBER":
    st.warning("Forecast is approaching the threshold. Increase monitoring.")
else:
    st.success("Forecast is within acceptable range.")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["rate"], label="Observed")
ax.plot(fc.index, fc["forecast"], "--", label="Forecast")
ax.fill_between(fc.index, fc["lower"], fc["upper"], alpha=0.25, label="95% interval")
ax.axhline(amber, color="orange", ls=":", label="Amber")
ax.axhline(red, color="red", ls=":", label="Red")
ax.set_ylabel("% resistant")
ax.legend()
st.pyplot(fig)

with st.expander("Forecast table and downloads"):
    st.dataframe(fc.round(1))
    st.download_button("Download forecast CSV", fc.to_csv().encode(), "forecast.csv")

st.caption("Method: damped-trend Holt-Winters exponential smoothing. Intervals are approximate. Decision-support only.")
