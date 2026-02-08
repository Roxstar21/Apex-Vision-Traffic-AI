import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="QuantumTrend Pro", page_icon="🦅", layout="wide")

# --- CSS ---
st.markdown("""<style>.stApp { background-color: #0e1117; } .prediction-card { background-color: #1c2029; padding: 20px; border-radius: 10px; border-left: 5px solid; margin-bottom: 20px; }</style>""", unsafe_allow_html=True)

# --- LOAD BRAIN ---
@st.cache_resource
def load_brain():
    paths = ["stock_predictor.h5", "backend/stock_predictor.h5"]
    for p in paths:
        if os.path.exists(p): return load_model(p)
    return None

try:
    model = load_brain()
except:
    model = None

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- SIDEBAR ---
with st.sidebar:
    st.title("QUANTUM DESK")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    run_btn = st.button("INITIATE ALGORITHM", type="primary")

# --- MAIN ---
st.title(f"💹 Market Intelligence // {ticker}")

if run_btn:
    if model is None:
        st.error("⚠️ AI Model not found. Check GitHub.")
    else:
        with st.spinner(f"📡 DOWNLOADING LIVE DATA FOR {ticker}..."):
            try:
                end = datetime.datetime.now()
                start = end - datetime.timedelta(days=730)
                data = yf.download(ticker, start=start, end=end, progress=False)
                
                # FIX: Handle MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if len(data) < 60:
                    st.error("Not enough data history.")
                else:
                    # AI PREDICTION
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(data[['Close']].values)
                    
                    x_input = scaled_data[-60:].reshape(1, 60, 1)
                    prediction = model.predict(x_input)
                    price = float(scaler.inverse_transform(prediction)[0][0])
                    
                    # TECHNICALS
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    data['RSI'] = calculate_rsi(data)
                    data = data.fillna(0)
                    
                    # --- THE ULTIMATE FIX ---
                    # Use .item() to extract the raw number from the Series
                    current_price = data['Close'].iloc[-1].item()
                    change_percent = ((price - current_price) / current_price) * 100
                    volume_val = data['Volume'].iloc[-1].item()
                    
                    # Logic Check (Now using raw floats)
                    if change_percent > 0:
                        trend_color = "#00ff00"
                        trend_msg = "🚀 BULLISH SIGNAL"
                    else:
                        trend_color = "#ff2b2b"
                        trend_msg = "🔻 BEARISH SIGNAL"

                    # DISPLAY
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("ASSET PRICE", f"${current_price:,.2f}")
                    c2.metric("AI TARGET", f"${price:,.2f}")
                    c3.metric("FORECAST", f"{change_percent:+.2f}%")
                    c4.metric("24H VOLUME", f"{int(volume_val):,}")
                    
                    st.markdown(f"""<div class="prediction-card" style="border-color: {trend_color};"><h3 style="color: {trend_color}; margin:0;">{trend_msg}</h3><p style="color: #ccc; margin-top: 5px;">Target: <b>${price:.2f}</b></p></div>""", unsafe_allow_html=True)

                    tab1, tab2 = st.tabs(["PRICE", "RSI"])
                    with tab1:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Market'))
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA', line=dict(color='#ff00ff', width=1)))
                        fig.add_trace(go.Scatter(x=[data.index[-1], "Forecast"], y=[current_price, price], mode='lines+markers', line=dict(color='yellow', dash='dot')))
                        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        st.line_chart(data['RSI'])

            except Exception as e:
                st.error(f"Analysis Failed: {e}")