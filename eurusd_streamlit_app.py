"""
Short-Term Forex & Gold Educational Analyzer
--------------------------------------------
Designed for 5m – 1h trades (scalping / short intraday)
- ENTRY, SL, TP ideas
- Multiple pairs + Gold + custom symbols
Educational only. You accept all risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/AUD": "EURAUD=X",
    "Gold (GC=F)": "GC=F",
}

def fetch_intraday(symbol, interval="15m", range_="5d"):
    """Fetch short-term data from Yahoo"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    
    if "chart" not in data or not data["chart"]["result"]:
        raise ValueError("No data returned for this symbol/interval")
    
    result = data["chart"]["result"][0]
    timestamps = result.get("timestamp")
    quote = result["indicators"]["quote"][0]
    
    if not timestamps:
        raise ValueError("Empty data")
    
    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
    }, index=pd.to_datetime(timestamps, unit="s"))
    
    df = df.dropna()
    df.index = df.index.tz_localize(None)
    return df


def add_indicators(df):
    df = df.copy()
    # Faster settings for short-term
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    ema12 = df["Close"].ewm(span=8, adjust=False).mean()
    ema26 = df["Close"].ewm(span=17, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=5, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(10).mean()
    return df


def generate_short_term_signal(df):
    latest = df.iloc[-1]
    price = float(latest["Close"])
    sma20 = float(latest["SMA_20"])
    sma50 = float(latest["SMA_50"])
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    macd_hist = float(latest["MACD_Hist"])
    atr = float(latest["ATR"])

    bullish = 0
    bearish = 0
    reasons = []
    warnings = []

    # Trend
    if price > sma20:
        bullish += 1.5
        reasons.append("Price above 20-SMA")
    else:
        bearish += 1.5
        reasons.append("Price below 20-SMA")

    if price > sma50:
        bullish += 1.0
        reasons.append("Price above 50-SMA")
    else:
        bearish += 1.0
        reasons.append("Price below 50-SMA")

    # RSI (faster)
    if rsi >= 72:
        bearish += 1.2
        warnings.append(f"RSI overbought ({rsi:.1f})")
        reasons.append(f"RSI high ({rsi:.1f})")
    elif rsi <= 28:
        bullish += 1.2
        warnings.append(f"RSI oversold ({rsi:.1f})")
        reasons.append(f"RSI low ({rsi:.1f})")
    elif 55 < rsi < 72:
        bullish += 0.6
        reasons.append(f"RSI supportive ({rsi:.1f})")
    elif 28 < rsi < 45:
        bearish += 0.6
        reasons.append(f"RSI weak ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    # MACD
    if macd > macd_signal and macd_hist > 0:
        bullish += 1.0
        reasons.append("MACD positive")
    elif macd < macd_signal and macd_hist < 0:
        bearish += 1.0
        reasons.append("MACD negative")
    else:
        reasons.append("MACD mixed")

    net = bullish - bearish

    direction = "NO CLEAR SIGNAL"
    confidence = "Low"
    entry = None
    sl = None
    tp = None
    idea = "No clear short-term bias"

    # Tighter levels for short-term
    if net >= 2.0:
        direction = "BUY BIAS (Short-term)"
        confidence = "Moderate" if net >= 2.8 else "Low-Moderate"
        entry = round(price, 5)
        sl = round(price - (1.2 * atr), 5)          # tighter SL
        risk = entry - sl
        tp = round(entry + (1.5 * risk), 5)         # 1.5R target (faster)
        idea = f"Short-term BUY idea. Hold while above {sma20:.5f}"

    elif net <= -2.0:
        direction = "SELL BIAS (Short-term)"
        confidence = "Moderate" if net <= -2.8 else "Low-Moderate"
        entry = round(price, 5)
        sl = round(price + (1.2 * atr), 5)
        risk = sl - entry
        tp = round(entry - (1.5 * risk), 5)
        idea = f"Short-term SELL idea. Hold while below {sma20:.5f}"

    return {
        "direction": direction,
        "confidence": confidence,
        "net_score": round(net, 1),
        "reasons": reasons,
        "warnings": warnings,
        "price": price,
        "idea": idea,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "sma20": round(sma20, 5),
        "sma50": round(sma50, 5),
        "rsi": round(rsi, 1),
        "atr": round(atr, 5)
    }


# ================= UI =================
st.set_page_config(page_title="Short-Term Forex Analyzer", page_icon="⚡", layout="centered")
st.title("⚡ Short-Term Forex & Gold Analyzer")
st.caption("For 5-minute to 1-hour trades • Educational only • You accept all risk")

with st.expander("⚠️ IMPORTANT RISK WARNING", expanded=True):
    st.warning("""
This tool is for **education only**.

- Signals are rule-based and designed for short-term ideas.
- Free data for 5m/15m has limitations and can be incomplete.
- ENTRY / SL / TP are calculated ideas only — **not guaranteed**.
- Short-term trading is high risk and can lose money quickly.
- You accept full responsibility for any trade you take.
""")

st.subheader("1. Choose Instrument")
selected = st.selectbox("Popular instruments", list(PAIRS.keys()))
custom = st.text_input("Or type custom Yahoo symbol", placeholder="e.g. EURUSD=X or GC=F")

symbol = custom.strip() if custom.strip() else PAIRS[selected]

st.subheader("2. Timeframe (Short-term)")
interval = st.selectbox(
    "Candle timeframe",
    options=["5m", "15m", "30m", "1h"],
    index=1,  # default 15m
    help="5m and 15m have limited history on free data"
)

# Map interval to reasonable range
range_map = {
    "5m": "5d",
    "15m": "5d",
    "30m": "10d",
    "1h": "1mo"
}
data_range = range_map.get(interval, "5d")

if st.button("Get Short-Term Signal", type="primary", use_container_width=True):
    with st.spinner(f"Fetching {interval} data for {symbol}..."):
        try:
            df = fetch_intraday(symbol, interval=interval, range_=data_range)
            df = add_indicators(df)
            df_clean = df.dropna(subset=["SMA_20", "SMA_50", "RSI", "ATR"])

            if len(df_clean) < 30:
                st.error("Not enough data for this timeframe/symbol. Try 15m or 30m, or another pair.")
            else:
                signal = generate_short_term_signal(df_clean)

                st.subheader(f"Snapshot – {symbol} ({interval})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"{signal['price']:.5f}")
                c2.metric("RSI", f"{signal['rsi']}")
                c3.metric("ATR", f"{signal['atr']*10000:.1f}")

                st.subheader("Short-Term Signal")
                if "BUY" in signal["direction"]:
                    st.success(f"**{signal['direction']}**")
                elif "SELL" in signal["direction"]:
                    st.error(f"**{signal['direction']}**")
                else:
                    st.info(f"**{signal['direction']}**")

                st.write(f"**Confidence:** {signal['confidence']}  |  Net Score: {signal['net_score']}")
                st.write(f"**Idea:** {signal['idea']}")

                if signal["entry"] is not None:
                    st.markdown("### Suggested Levels (Short-term ideas)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ENTRY", f"{signal['entry']:.5f}")
                    col2.metric("STOP LOSS", f"{signal['sl']:.5f}")
                    col3.metric("TAKE PROFIT", f"{signal['tp']:.5f}")
                    st.caption("Tighter levels for short-term trades (1.2×ATR SL, 1.5R TP). Not guaranteed.")

                if signal["warnings"]:
                    for w in signal["warnings"]:
                        st.warning(w)

                with st.expander("Why this signal?"):
                    for r in signal["reasons"]:
                        st.write(f"• {r}")
                    st.write(f"• 20-SMA: {signal['sma20']}")
                    st.write(f"• 50-SMA: {signal['sma50']}")

                st.subheader("Recent Price")
                st.line_chart(df_clean[["Close", "SMA_20", "SMA_50"]].tail(80))

                st.success("Analysis complete.")
                st.caption(f"Data range used: {data_range} | Interval: {interval}")

        except Exception as e:
            st.error(f"Could not get data for **{symbol}** on {interval}")
            st.write(str(e))
            st.info("Try:\n- Changing to 15m or 30m\n- Using EURUSD=X, GBPUSD=X, or GC=F\n- Checking your internet")

else:
    st.info("Select instrument + timeframe and tap **Get Short-Term Signal**")

st.markdown("---")
st.caption("Short-term educational tool • 5m–1h focus • You accept all trading risk")
