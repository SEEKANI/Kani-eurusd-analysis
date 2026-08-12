"""
EUR/USD Educational Analysis & Signal App (Streamlit)
Educational use only. Not financial advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def fetch_eurusd_daily(period="2y"):
    range_map = {"1y": "1y", "2y": "2y", "5y": "5y", "max": "max"}
    rng = range_map.get(period, "2y")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval=1d&range={rng}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
        "Volume": quote.get("volume", [0] * len(timestamps))
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
    price = latest["Close"]
    sma50 = latest["SMA_50"]
    sma100 = latest["SMA_100"]
    sma200 = latest["SMA_200"]
    rsi = latest["RSI"]
    macd = latest["MACD"]
    macd_signal = latest["MACD_Signal"]
    macd_hist = latest["MACD_Hist"]

    bullish_score = 0.0
    bearish_score = 0.0
    reasons = []
    warnings_list = []

    if price > sma50:
        bullish_score += 1.5
        reasons.append("Price above 50-day SMA (+1.5)")
    else:
        bearish_score += 1.5
        reasons.append("Price below 50-day SMA (+1.5 bearish)")

    if price > sma100:
        bullish_score += 1.0
        reasons.append("Price above 100-day SMA (+1.0)")
    else:
        bearish_score += 1.0
        reasons.append("Price below 100-day SMA (+1.0 bearish)")

    if not pd.isna(sma200):
        if price > sma200:
            bullish_score += 0.5
            reasons.append("Price above 200-day SMA (+0.5)")
        else:
            bearish_score += 0.5
            reasons.append("Price below 200-day SMA (+0.5 bearish)")

    if rsi >= 70:
        bearish_score += 1.0
        warnings_list.append(f"RSI overbought ({rsi:.1f}) – caution on further upside")
        reasons.append(f"RSI overbought zone ({rsi:.1f})")
    elif rsi <= 30:
        bullish_score += 1.0
        warnings_list.append(f"RSI oversold ({rsi:.1f}) – caution on further downside")
        reasons.append(f"RSI oversold zone ({rsi:.1f})")
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
        reasons.append("MACD positive & above signal (+1.0)")
    elif macd < macd_signal and macd_hist < 0:
        bearish_score += 1.0
        reasons.append("MACD negative & below signal (+1.0 bearish)")
    else:
        reasons.append("MACD mixed / transitioning")

    net = bullish_score - bearish_score

    if net >= 2.0:
        direction = "BULLISH BIAS (observational)"
        confidence = "Moderate" if net >= 3.0 else "Low-Moderate"
    elif net <= -2.0:
        direction = "BEARISH BIAS (observational)"
        confidence = "Moderate" if net <= -3.0 else "Low-Moderate"
    else:
        direction = "NEUTRAL / RANGE (observational)"
        confidence = "Low"

    return {
        "direction": direction,
        "confidence": confidence,
        "bullish_score": round(bullish_score, 1),
        "bearish_score": round(bearish_score, 1),
        "net_score": round(net, 1),
        "reasons": reasons,
        "warnings": warnings_list,
        "price": price,
        "key_levels": {
            "support_50sma": round(sma50, 5),
            "resistance_100sma": round(sma100, 5),
            "sma200": round(sma200, 5) if not pd.isna(sma200) else None
        },
        "rsi": round(rsi, 1),
        "atr": round(latest["ATR"], 5)
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

        if "BULLISH" in sig["direction"]:
            correct = change > 0
            side = "BULLISH"
        elif "BEARISH" in sig["direction"]:
            correct = change < 0
            side = "BEARISH"
        else:
            continue

        results.append({
            "date": df.index[i],
            "side": side,
            "correct": correct,
            "pct_change": (change / current_price) * 100,
            "net_score": sig["net_score"]
        })

    if not results:
        return None, None

    res_df = pd.DataFrame(results)
    total = len(res_df)
    correct_count = res_df["correct"].sum()
    accuracy = (correct_count / total) * 100 if total > 0 else 0

    bullish = res_df[res_df["side"] == "BULLISH"]
    bearish = res_df[res_df["side"] == "BEARISH"]

    summary = {
        "total_signals": total,
        "accuracy_pct": round(float(accuracy), 1),
        "bullish_signals": len(bullish),
        "bullish_accuracy": round(float(bullish["correct"].sum() / len(bullish) * 100), 1) if len(bullish) > 0 else None,
        "bearish_signals": len(bearish),
        "bearish_accuracy": round(float(bearish["correct"].sum() / len(bearish) * 100), 1) if len(bearish) > 0 else None,
        "avg_pct_change_when_correct": round(float(res_df[res_df["correct"]]["pct_change"].mean()), 3) if correct_count > 0 else None,
        "forward_days": forward_days
    }
    return summary, res_df


st.set_page_config(
    page_title="EUR/USD Educational Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("EUR/USD Educational Analyzer")
st.caption("Educational tool only • Not financial advice")

with st.expander("⚠️ IMPORTANT RISK DISCLAIMER – READ THIS", expanded=True):
    st.warning("""
**This is for education and research only.**

- It is **NOT** investment or trading advice.  
- No signal or backtest can guarantee profits.  
- Forex involves substantial risk of loss.  
- You can lose more than you put in.  
- Past results do not predict future results.  
- Always use proper risk management (including stop-losses).  
- Do your own research.
""")

st.subheader("Settings")
period = st.selectbox("Historical data period", ["1y", "2y", "5y"], index=1)
forward_days = st.selectbox(
    "Backtest: look forward how many days?",
    options=[5, 10, 15, 20],
    index=0
)

run_button = st.button("Run Analysis", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Fetching data and calculating… please wait"):
        try:
            df = fetch_eurusd_daily(period=period)
            df = add_indicators(df)
            df_clean = df.dropna(subset=["SMA_50", "SMA_100", "RSI", "ATR"]).copy()

            if len(df_clean) < 50:
                st.error("Not enough data. Try a longer period.")
            else:
                signal = generate_signals(df_clean)
                bt_summary, _ = simple_backtest(df_clean, forward_days=forward_days)

                st.subheader("Current Snapshot")
                m1, m2, m3 = st.columns(3)
                m1.metric("Close", f"{signal['price']:.5f}")
                m2.metric("RSI (14)", f"{signal['rsi']}")
                m3.metric("ATR", f"{signal['atr']*10000:.0f} pips")

                st.subheader("Observational Signal")
                st.info(
                    f"**{signal['direction']}**\n\n"
                    f"Confidence: **{signal['confidence']}**  \n"
                    f"Net score: {signal['net_score']} "
                    f"(Bull {signal['bullish_score']} / Bear {signal['bearish_score']})"
                )

                if signal["warnings"]:
                    for w in signal["warnings"]:
                        st.warning(w)

                with st.expander("Why this signal? (reasons & levels)"):
                    st.markdown("**Reasons**")
                    for r in signal["reasons"]:
                        st.write(f"• {r}")
                    st.markdown("**Key levels**")
                    st.write(f"• Support (50 SMA): `{signal['key_levels']['support_50sma']}`")
                    st.write(f"• Resistance (100 SMA): `{signal['key_levels']['resistance_100sma']}`")
                    if signal['key_levels']['sma200']:
                        st.write(f"• 200 SMA: `{signal['key_levels']['sma200']}`")

                st.subheader(f"Simple Backtest ({forward_days} days forward)")
                if bt_summary is None:
                    st.write("Not enough data for backtest.")
                else:
                    st.write(f"Directional signals tested: **{bt_summary['total_signals']}**")
                    st.write(f"Overall accuracy: **{bt_summary['accuracy_pct']}%**")
                    st.write(
                        f"Bullish: {bt_summary['bullish_signals']} signals "
                        f"({bt_summary['bullish_accuracy']}% accuracy)  \n"
                        f"Bearish: {bt_summary['bearish_signals']} signals "
                        f"({bt_summary['bearish_accuracy']}% accuracy)"
                    )
                    if bt_summary["avg_pct_change_when_correct"] is not None:
                        st.write(f"Avg move when correct: {bt_summary['avg_pct_change_when_correct']}%")

                    st.caption(
                        "This backtest only checks direction after N days. "
                        "It ignores spreads, costs, slippage, news, and stop-losses. "
                        "Accuracy near 50% is common and does **not** mean a real edge."
                    )

                st.subheader("Recent Price & Moving Averages")
                chart_df = df_clean[["Close", "SMA_50", "SMA_100"]].tail(150)
                st.line_chart(chart_df, use_container_width=True)

                st.success("Analysis complete. Remember – educational use only.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.exception(e)
else:
    st.info("Select your settings above and tap **Run Analysis**.")

st.markdown("---")
st.caption("Educational tool • Data from public sources • Not financial advice")
