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
        confidence = "Moderate" if net >= 3.0 else "Low-
