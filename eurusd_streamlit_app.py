"""
Multi-Pair Forex + Gold Educational Analyzer
--------------------------------------------
- Many forex pairs + Gold (using GC=F)
- Custom symbol support (closest to unlimited)
- Rule-based BUY / SELL signals
- ENTRY, SL and TP ideas
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
    "EUR/CHF": "EURCHF=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/AUD": "GBPAUD=X",
    "USD/ZAR": "USDZAR=X",
    "Gold (GC=F)": "GC=F",          # Fixed Gold symbol
}

def fetch_data(symbol, period="2y"):
    range_map = {"1y": "1y", "2y": "2y", "5y": "5y"}
    rng = range_map.get(period, "2y")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={rng}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
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
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_100"] = df["Close"].rolling(100).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df


def generate_signals(df):
    latest = df.iloc[-1]
    price = float(latest["Close"])
    sma50 = float(latest["SMA_50"])
    sma100 = float(latest["SMA_100"])
    sma200 = float(latest["SMA_200"]) if not pd.isna(latest["SMA_200"]) else None
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])
    macd_hist = float(latest["MACD_Hist"])
    atr = float(latest["ATR"])

    bullish_score = 0.0
    bearish_score = 0.0
    reasons = []
    warnings_list = []

    if price > sma50:
        bullish_score += 1.5
        reasons.append("Price above 50-day SMA")
    else:
        bearish_score += 1.5
        reasons.append("Price below 50-day SMA")

    if price > sma100:
        bullish_score += 1.0
        reasons.append("Price above 100-day SMA")
    else:
        bearish_score += 1.0
        reasons.append("Price below 100-day SMA")

    if sma200 is not None:
        if price > sma200:
            bullish_score += 0.5
            reasons.append("Price above 200-day SMA")
        else:
            bearish_score += 0.5
            reasons.append("Price below 200-day SMA")

    if rsi >= 70:
        bearish_score += 1.0
        warnings_list.append(f"RSI overbought ({rsi:.1f})")
        reasons.append(f"RSI overbought ({rsi:.1f})")
    elif rsi <= 30:
        bullish_score += 1.0
        warnings_list.append(f"RSI oversold ({rsi:.1f})")
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif 55 <= rsi < 70:
        bullish_score += 0.5
        reasons.append(f"RSI moderately strong ({rsi:.1f})")
    elif 30 < rsi <= 45:
        bearish_score += 0.5
        reasons.append(f"RSI moderately weak ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    if macd > macd_signal and macd_hist > 0:
        bullish_score += 1.0
        reasons.append("MACD positive")
    elif macd < macd_signal and macd_hist < 0:
        bearish_score += 1.0
        reasons.append("MACD negative")
    else:
        reasons.append("MACD mixed")

    net = bullish_score - bearish_score

    direction = "NO CLEAR SIGNAL (RANGE)"
    confidence = "Low"
    entry = None
    sl = None
    tp = None
    idea = "No clear directional bias – market is mixed / ranging"

    if net >= 2.0:
        direction = "BUY BIAS"
        confidence = "Moderate" if net >= 3.0 else "Low-Moderate"
        entry = round(price, 5)
        sl = round(min(sma50, price) - (1.5 * atr), 5)
        risk = entry - sl
        tp = round(entry + (2.0 * risk), 5)
        idea = f"Observational BUY idea while price holds above {sma50:.5f}"

    elif net <= -2.0:
        direction = "SELL BIAS"
        confidence = "Moderate" if net <= -3.0 else "Low-Moderate"
        entry = round(price, 5)
        sl = round(max(sma50, price) + (1.5 * atr), 5)
        risk = sl - entry
        tp = round(entry - (2.0 * risk), 5)
        idea = f"Observational SELL idea while price holds below {sma50:.5f}"

    return {
        "direction": direction,
        "confidence": confidence,
        "net_score": round(net, 1),
        "reasons": reasons,
        "warnings": warnings_list,
        "price": price,
        "idea": idea,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "key_levels": {
            "sma50": round(sma50, 5),
            "sma100": round(sma100, 5),
            "sma200": round(sma200, 5) if sma200 else None
        },
        "rsi": round(rsi, 1),
        "atr": round(atr, 5)
    }


def simple_backtest(df, forward_days=5):
    df = df.copy()
    results = []
    for i in range(len(df) - forward_days - 1):
        window = df.iloc[:i+1]
        if len(window) < 200:
            continue
        if window[["SMA_50", "SMA_100", "RSI", "MACD"]].iloc[-1].isna().any():
            continue
        sig = generate_signals(window)
        future_price = df["Close"].iloc[i + forward_days]
        current_price = df["Close"].iloc[i]
        change = future_price - current_price
        if "BUY" in sig["direction"]:
            correct = change > 0
            side = "BUY"
        elif "SELL" in sig["direction"]:
            correct = change < 0
            side = "SELL"
        else:
            continue
        results.append({"side": side, "correct": correct})
    if not results:
        return None
    res_df = pd.DataFrame(results)
    total = len(res_df)
    accuracy = (res_df["correct"].sum() / total) * 100
    buy = res_df[res_df["side"] == "BUY"]
    sell = res_df[res_df["side"] == "SELL"]
    return {
        "total_signals": total,
        "accuracy_pct": round(float(accuracy), 1),
        "buy_signals": len(buy),
        "buy_accuracy": round(float(buy["correct"].sum() / len(buy) * 100), 1) if len(buy) > 0 else None,
        "sell_signals": len(sell),
        "sell_accuracy": round(float(sell["correct"].sum() / len(sell) * 100), 1) if len(sell) > 0 else None,
        "forward_days": forward_days
    }


st.set_page_config(page_title="Forex & Gold Signal Analyzer", page_icon="📊", layout="centered")
st.title("Forex & Gold Educational Signal Analyzer")
st.caption("Educational tool • ENTRY / SL / TP ideas • You accept all risk")

with st.expander("⚠️ RISK DISCLAIMER", expanded=True):
    st.warning("""
This is for education only.  
Signals and ENTRY/SL/TP levels are rule-based and can lose money.  
No guarantees. You accept full responsibility for any trade.
""")

st.subheader("1. Choose Instrument")
selected_pair = st.selectbox("Popular instruments", list(PAIRS.keys()))
custom_symbol = st.text_input(
    "Or type any Yahoo Finance symbol (recommended for more pairs)",
    placeholder="Examples: EURUSD=X , GBPJPY=X , GC=F , BTC-USD"
)

symbol = custom_symbol.strip() if custom_symbol.strip() else PAIRS[selected_pair]

st.subheader("2. Settings")
period = st.selectbox("Historical period", ["1y", "2y", "5y"], index=1)
forward_days = st.selectbox("Backtest forward days", [5, 10, 15, 20], index=0)

if st.button("Run Analysis & Get Signal", type="primary", use_container_width=True):
    with st.spinner(f"Analyzing {symbol}..."):
        try:
            df = fetch_data(symbol, period)
            df = add_indicators(df)
            df_clean = df.dropna(subset=["SMA_50", "SMA_100", "RSI", "ATR"])

            if len(df_clean) < 50:
                st.error("Not enough data for this symbol / period.")
            else:
                signal = generate_signals(df_clean)
                bt = simple_backtest(df_clean, forward_days)

                st.subheader(f"Snapshot – {symbol}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"{signal['price']:.5f}")
                c2.metric("RSI", f"{signal['rsi']}")
                c3.metric("ATR", f"{signal['atr']*10000:.1f}")

                st.subheader("Computerized Signal")
                if "BUY" in signal["direction"]:
                    st.success(f"**SIGNAL: {signal['direction']}**")
                elif "SELL" in signal["direction"]:
                    st.error(f"**SIGNAL: {signal['direction']}**")
                else:
                    st.info(f"**SIGNAL: {signal['direction']}**")

                st.write(f"**Confidence:** {signal['confidence']}  |  Net Score: {signal['net_score']}")
                st.write(f"**Idea:** {signal['idea']}")

                if signal["entry"] is not None:
                    st.markdown("### Suggested Levels (ideas only)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ENTRY", f"{signal['entry']:.5f}")
                    col2.metric("STOP LOSS", f"{signal['sl']:.5f}")
                    col3.metric("TAKE PROFIT", f"{signal['tp']:.5f}")
                    st.caption("Calculated ideas only (not guaranteed).")

                if signal["warnings"]:
                    for w in signal["warnings"]:
                        st.warning(w)

                with st.expander("Reasons & Key Levels"):
                    for r in signal["reasons"]:
                        st.write(f"• {r}")
                    st.write(f"• 50 SMA: {signal['key_levels']['sma50']}")
                    st.write(f"• 100 SMA: {signal['key_levels']['sma100']}")
                    if signal['key_levels']['sma200']:
                        st.write(f"• 200 SMA: {signal['key_levels']['sma200']}")

                if bt:
                    st.subheader(f"Backtest ({forward_days} days)")
                    st.write(f"Signals: {bt['total_signals']} | Accuracy: {bt['accuracy_pct']}%")
                    st.write(f"BUY: {bt['buy_signals']} ({bt['buy_accuracy']}%) | SELL: {bt['sell_signals']} ({bt['sell_accuracy']}%)")

                st.subheader("Chart")
                st.line_chart(df_clean[["Close", "SMA_50", "SMA_100"]].tail(150))
                st.success("Done.")

        except Exception as e:
            st.error(f"Could not fetch data for **{symbol}**")
            st.write(f"Error details: {e}")
            st.info("Try a different symbol. Common working examples:\n- EURUSD=X\n- GBPUSD=X\n- USDJPY=X\n- GC=F (Gold)\n- BTC-USD")

else:
    st.info("Select an instrument (or type a custom symbol) and tap the button.")

st.markdown("---")
st.caption("Educational tool • You accept all trading risk")
