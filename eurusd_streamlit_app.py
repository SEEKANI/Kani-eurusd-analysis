"""
Kani Forex & Gold Educational Analyzer
======================================

Modes:
1. App Strategy
2. My Strategy
3. Analyze My Strategy

Risk/Reward:
- Auto
- Unlimited Custom R:R

Features:
- Forex + Gold
- 5m / 15m / 30m / 1h
- SMA / RSI / MACD / ATR
- Market-structure checks
- Custom strategy rules
- Strategy description analyzer
- Historical backtesting
- Unlimited R:R

Educational use only.
No trading execution is performed.
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Kani Forex & Gold Analyzer",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# INSTRUMENTS
# ============================================================

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


# ============================================================
# TIMEFRAME SETTINGS
# ============================================================

TIMEFRAME_RANGES = {
    "5m": "5d",
    "15m": "5d",
    "30m": "10d",
    "1h": "1mo",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_price(price, symbol):
    """
    Forex pairs and Gold do not necessarily need
    the same decimal formatting.
    """

    if symbol == "GC=F":
        return f"{price:.2f}"

    if "JPY" in symbol:
        return f"{price:.3f}"

    return f"{price:.5f}"


def safe_float(value):
    """
    Safely convert a value to float.
    """

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return None


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=60)
def fetch_intraday(symbol, interval="15m", range_="5d"):
    """
    Fetch intraday OHLC data from Yahoo Finance.
    """

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}"
    )

    params = {
        "interval": interval,
        "range": range_,
        "events": "history",
        "includeAdjustedClose": "true",
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    chart = data.get("chart", {})
    results = chart.get("result")

    if not results:
        error = chart.get("error")
        raise ValueError(
            f"No market data returned. {error}"
        )

    result = results[0]

    timestamps = result.get("timestamp")

    if not timestamps:
        raise ValueError(
            "The data source returned no timestamps."
        )

    indicators = result.get(
        "indicators",
        {}
    )

    quotes = indicators.get(
        "quote",
        []
    )

    if not quotes:
        raise ValueError(
            "The data source returned no price information."
        )

    quote = quotes[0]

    df = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
        },
        index=pd.to_datetime(
            timestamps,
            unit="s"
        )
    )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    if df.empty:
        raise ValueError(
            "No usable candles were returned."
        )

    try:
        df.index = df.index.tz_localize(None)
    except TypeError:
        pass

    df = df.sort_index()

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    df["SMA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(7)
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .rolling(7)
        .mean()
    )

    # Avoid division problems.
    rs = gain / loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema_fast = (
        df["Close"]
        .ewm(
            span=8,
            adjust=False
        )
        .mean()
    )

    ema_slow = (
        df["Close"]
        .ewm(
            span=17,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema_fast -
        ema_slow
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=5,
            adjust=False
        )
        .mean()
    )

    df["MACD_Hist"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    high_low = (
        df["High"] -
        df["Low"]
    )

    high_close = np.abs(
        df["High"] -
        df["Close"].shift(1)
    )

    low_close = np.abs(
        df["Low"] -
        df["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(10)
        .mean()
    )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

    # Shift(1) prevents the current candle from being
    # included in its own previous-high/low calculation.

    df["Previous_High_10"] = (
        df["High"]
        .shift(1)
        .rolling(10)
        .max()
    )

    df["Previous_Low_10"] = (
        df["Low"]
        .shift(1)
        .rolling(10)
        .min()
    )

    df["Previous_High_20"] = (
        df["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df["Previous_Low_20"] = (
        df["Low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_indicator_data(df):

    required = [
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "ATR",
        "Previous_High_20",
        "Previous_Low_20",
    ]

    cleaned = df.dropna(
        subset=required
    ).copy()

    return cleaned


# ============================================================
# APP STRATEGY
# ============================================================

def app_strategy_signal(df):

    latest = df.iloc[-1]

    price = float(
        latest["Close"]
    )

    sma20 = float(
        latest["SMA_20"]
    )

    sma50 = float(
        latest["SMA_50"]
    )

    rsi = float(
        latest["RSI"]
    )

    macd = float(
        latest["MACD"]
    )

    macd_signal = float(
        latest["MACD_Signal"]
    )

    macd_hist = float(
        latest["MACD_Hist"]
    )

    atr = float(
        latest["ATR"]
    )

    bullish = 0.0
    bearish = 0.0

    reasons = []
    warnings = []

    # --------------------------------------------------------
    # PRICE VS SMA 20
    # --------------------------------------------------------

    if price > sma20:

        bullish += 1.5

        reasons.append(
            "Price is above the 20-SMA."
        )

    else:

        bearish += 1.5

        reasons.append(
            "Price is below the 20-SMA."
        )

    # --------------------------------------------------------
    # PRICE VS SMA 50
    # --------------------------------------------------------

    if price > sma50:

        bullish += 1.0

        reasons.append(
            "Price is above the 50-SMA."
        )

    else:

        bearish += 1.0

        reasons.append(
            "Price is below the 50-SMA."
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi >= 72:

        bearish += 1.2

        warnings.append(
            f"RSI is overbought ({rsi:.1f})."
        )

        reasons.append(
            f"RSI is high ({rsi:.1f})."
        )

    elif rsi <= 28:

        bullish += 1.2

        warnings.append(
            f"RSI is oversold ({rsi:.1f})."
        )

        reasons.append(
            f"RSI is low ({rsi:.1f})."
        )

    elif 55 < rsi < 72:

        bullish += 0.6

        reasons.append(
            f"RSI supports bullish momentum ({rsi:.1f})."
        )

    elif 28 < rsi < 45:

        bearish += 0.6

        reasons.append(
            f"RSI shows bearish pressure ({rsi:.1f})."
        )

    else:

        reasons.append(
            f"RSI is neutral ({rsi:.1f})."
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        macd > macd_signal
        and macd_hist > 0
    ):

        bullish += 1.0

        reasons.append(
            "MACD is positive."
        )

    elif (
        macd < macd_signal
        and macd_hist < 0
    ):

        bearish += 1.0

        reasons.append(
            "MACD is negative."
        )

    else:

        reasons.append(
            "MACD is mixed."
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    net = bullish - bearish

    if net >= 2.0:

        direction = "BUY"

        confidence = (
            "Moderate"
            if net >= 2.8
            else "Low-Moderate"
        )

    elif net <= -2.0:

        direction = "SELL"

        confidence = (
            "Moderate"
            if net <= -2.8
            else "Low-Moderate"
        )

    else:

        direction = "NO TRADE"
        confidence = "Low"

    return {
        "direction": direction,
        "confidence": confidence,
        "score": round(net, 2),
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "reasons": reasons,
        "warnings": warnings,
    }


# ============================================================
# CUSTOM STRATEGY
# ============================================================

def custom_strategy_signal(
    df,
    trend_rule,
    rsi_rule,
    macd_rule,
    confirmation_count
):

    latest = df.iloc[-1]

    price = float(
        latest["Close"]
    )

    sma20 = float(
        latest["SMA_20"]
    )

    sma50 = float(
        latest["SMA_50"]
    )

    rsi = float(
        latest["RSI"]
    )

    macd = float(
        latest["MACD"]
    )

    macd_signal = float(
        latest["MACD_Signal"]
    )

    atr = float(
        latest["ATR"]
    )

    buy_confirmations = 0
    sell_confirmations = 0

    reasons = []

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if trend_rule == "Price above 20-SMA":

        if price > sma20:

            buy_confirmations += 1

            reasons.append(
                "Your 20-SMA bullish condition is satisfied."
            )

        else:

            reasons.append(
                "Your 20-SMA bullish condition is not satisfied."
            )

    elif trend_rule == "Price below 20-SMA":

        if price < sma20:

            sell_confirmations += 1

            reasons.append(
                "Your 20-SMA bearish condition is satisfied."
            )

        else:

            reasons.append(
                "Your 20-SMA bearish condition is not satisfied."
            )

    elif trend_rule == "Bullish 20/50 SMA":

        if (
            price > sma20
            and sma20 > sma50
        ):

            buy_confirmations += 1

            reasons.append(
                "Bullish 20/50 SMA structure confirmed."
            )

        else:

            reasons.append(
                "Bullish 20/50 SMA structure not confirmed."
            )

    elif trend_rule == "Bearish 20/50 SMA":

        if (
            price < sma20
            and sma20 < sma50
        ):

            sell_confirmations += 1

            reasons.append(
                "Bearish 20/50 SMA structure confirmed."
            )

        else:

            reasons.append(
                "Bearish 20/50 SMA structure not confirmed."
            )

    elif trend_rule == "No trend filter":

        reasons.append(
            "No trend filter selected."
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi_rule == "RSI above 50":

        if rsi > 50:

            buy_confirmations += 1

            reasons.append(
                f"RSI is above 50 ({rsi:.1f})."
            )

    elif rsi_rule == "RSI below 50":

        if rsi < 50:

            sell_confirmations += 1

            reasons.append(
                f"RSI is below 50 ({rsi:.1f})."
            )

    elif rsi_rule == "RSI oversold":

        if rsi <= 30:

            buy_confirmations += 1

            reasons.append(
                f"RSI is oversold ({rsi:.1f})."
            )

    elif rsi_rule == "RSI overbought":

        if rsi >= 70:

            sell_confirmations += 1

            reasons.append(
                f"RSI is overbought ({rsi:.1f})."
            )

    else:

        reasons.append(
            "No RSI filter selected."
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd_rule == "Bullish MACD":

        if macd > macd_signal:

            buy_confirmations += 1

            reasons.append(
                "Bullish MACD condition is satisfied."
            )

    elif macd_rule == "Bearish MACD":

        if macd < macd_signal:

            sell_confirmations += 1

            reasons.append(
                "Bearish MACD condition is satisfied."
            )

    else:

        reasons.append(
            "No MACD filter selected."
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if (
        buy_confirmations >= confirmation_count
        and buy_confirmations > sell_confirmations
    ):

        direction = "BUY"
        confidence = "Custom Strategy Confirmed"

    elif (
        sell_confirmations >= confirmation_count
        and sell_confirmations > buy_confirmations
    ):

        direction = "SELL"
        confidence = "Custom Strategy Confirmed"

    else:

        direction = "NO TRADE"
        confidence = "Conditions Not Met"

    return {
        "direction": direction,
        "confidence": confidence,
        "score": (
            buy_confirmations -
            sell_confirmations
        ),
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "reasons": reasons,
        "warnings": [],
    }


# ============================================================
# STRATEGY TEXT ANALYZER
# ============================================================

def analyze_strategy_description(text):

    text = text.lower().strip()

    detected = []
    missing = []
    suggestions = []

    if not text:

        return {
            "detected": [],
            "missing": [
                "No strategy description was provided."
            ],
            "suggestions": [
                "Describe your entry, confirmation, "
                "stop-loss and target rules."
            ]
        }

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "trend",
            "ema",
            "sma",
            "moving average",
            "bullish",
            "bearish"
        ]
    ):

        detected.append(
            "Trend/direction condition."
        )

    else:

        missing.append(
            "A clearly defined trend or market-direction rule."
        )

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "entry",
            "enter",
            "buy",
            "sell",
            "retest",
            "retracement",
            "pullback"
        ]
    ):

        detected.append(
            "Entry condition."
        )

    else:

        missing.append(
            "A specific entry trigger."
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if "rsi" in text:

        detected.append(
            "RSI condition."
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if "macd" in text:

        detected.append(
            "MACD condition."
        )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "break of structure",
            "market structure",
            "bos",
            "higher high",
            "lower low"
        ]
    ):

        detected.append(
            "Market-structure condition."
        )

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "liquidity",
            "liquidity sweep",
            "stop hunt"
        ]
    ):

        detected.append(
            "Liquidity condition."
        )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "stop loss",
            "stop-loss",
            "stoploss",
            "sl"
        ]
    ):

        detected.append(
            "Stop-loss rule."
        )

    else:

        missing.append(
            "A clearly defined stop-loss rule."
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "take profit",
            "take-profit",
            "takeprofit",
            "target",
            "tp",
            "risk reward",
            "risk/reward",
            "rr"
        ]
    ):

        detected.append(
            "Target/R:R rule."
        )

    else:

        missing.append(
            "A clearly defined target or R:R rule."
        )

    # --------------------------------------------------------
    # SESSION / NEWS
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "london",
            "new york",
            "asian session",
            "session"
        ]
    ):

        detected.append(
            "Trading-session condition."
        )

    else:

        suggestions.append(
            "Consider specifying which market sessions "
            "your strategy is designed for."
        )

    if any(
        phrase in text
        for phrase in [
            "news",
            "fundamental",
            "interest rate",
            "cpi",
            "nfp"
        ]
    ):

        detected.append(
            "News/fundamental filter."
        )

    else:

        suggestions.append(
            "Consider whether major economic news "
            "should invalidate a setup."
        )

    # --------------------------------------------------------
    # GENERAL SUGGESTIONS
    # --------------------------------------------------------

    if not missing:

        suggestions.append(
            "Your description contains the major "
            "components required for a structured strategy."
        )

    else:

        suggestions.append(
            "Define the missing rules before attempting "
            "to automate the strategy."
        )

    suggestions.append(
        "Avoid vague rules such as 'enter when the "
        "market looks strong'. Define measurable conditions."
    )

    return {
        "detected": detected,
        "missing": missing,
        "suggestions": suggestions
    }


# ============================================================
# AUTO R:R
# ============================================================

def calculate_auto_rr(
    df,
    direction,
    entry,
    atr
):

    if atr <= 0:

        return 2.0, [
            "ATR was invalid, so the fallback R:R is 2R."
        ]

    latest = df.iloc[-1]

    recent_high = float(
        latest["Previous_High_20"]
    )

    recent_low = float(
        latest["Previous_Low_20"]
    )

    warnings = []

    if direction == "BUY":

        distance = (
            recent_high -
            entry
        )

    elif direction == "SELL":

        distance = (
            entry -
            recent_low
        )

    else:

        return 2.0, []

    if distance <= 0:

        return 2.0, [
            "No clear structural target was found. "
            "Auto mode uses 2R."
        ]

    structural_rr = (
        distance /
        (1.2 * atr)
    )

    # Auto mode remains conservative.
    auto_rr = float(
        np.clip(
            structural_rr,
            1.0,
            5.0
        )
    )

    return round(
        auto_rr,
        1
    ), warnings


# ============================================================
# TRADE LEVEL CALCULATION
# ============================================================

def calculate_levels(
    direction,
    entry,
    atr,
    rr
):

    if atr is None or atr <= 0:

        return None

    stop_distance = (
        1.2 *
        atr
    )

    if direction == "BUY":

        sl = (
            entry -
            stop_distance
        )

        tp = (
            entry +
            (
                stop_distance *
                rr
            )
        )

    elif direction == "SELL":

        sl = (
            entry +
            stop_distance
        )

        tp = (
            entry -
            (
                stop_distance *
                rr
            )
        )

    else:

        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_distance": stop_distance,
        "rr": rr
    }


# ============================================================
# MARKET STRUCTURE CHECK
# ============================================================

def check_market_structure(
    df,
    direction,
    entry,
    sl,
    tp,
    atr
):

    latest = df.iloc[-1]

    recent_high = float(
        latest["Previous_High_20"]
    )

    recent_low = float(
        latest["Previous_Low_20"]
    )

    observations = []
    warnings_list = []

    if direction == "BUY":

        if tp > recent_high:

            observations.append(
                "TP is above the previous 20-candle high."
            )

        else:

            observations.append(
                "TP is inside the previous 20-candle range."
            )

        if sl < recent_low:

            observations.append(
                "SL is below the previous 20-candle low."
            )

        target_distance = (
            tp -
            entry
        )

    elif direction == "SELL":

        if tp < recent_low:

            observations.append(
                "TP is below the previous 20-candle low."
            )

        else:

            observations.append(
                "TP is inside the previous 20-candle range."
            )

        if sl > recent_high:

            observations.append(
                "SL is above the previous 20-candle high."
            )

        target_distance = (
            entry -
            tp
        )

    else:

        return {
            "observations": [],
            "warnings": [],
            "recent_high": recent_high,
            "recent_low": recent_low,
            "target_atr": 0
        }

    target_atr = (
        target_distance /
        atr
    )

    if target_atr >= 10:

        warnings_list.append(
            f"The selected TP is approximately "
            f"{target_atr:.1f} ATR away from entry."
        )

    elif target_atr >= 5:

        warnings_list.append(
            f"The selected TP is approximately "
            f"{target_atr:.1f} ATR away from entry. "
            "This is a relatively large target."
        )

    return {
        "observations": observations,
        "warnings": warnings_list,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "target_atr": target_atr
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest_strategy(
    df,
    strategy_name,
    custom_settings=None,
    rr=2.0,
    max_bars_forward=100
):

    data = clean_indicator_data(
        df
    ).copy()

    results = []

    # We need enough candles before each test.
    start_index = 50

    for i in range(
        start_index,
        len(data) - 1
    ):

        historical = data.iloc[
            :i + 1
        ]

        if strategy_name == "App Strategy":

            signal = app_strategy_signal(
                historical
            )

        elif strategy_name == "My Strategy":

            signal = custom_strategy_signal(
                historical,
                custom_settings["trend"],
                custom_settings["rsi"],
                custom_settings["macd"],
                custom_settings["confirmations"]
            )

        else:

            continue

        direction = signal["direction"]

        if direction not in [
            "BUY",
            "SELL"
        ]:

            continue

        entry = float(
            data.iloc[i]["Close"]
        )

        atr = float(
            data.iloc[i]["ATR"]
        )

        levels = calculate_levels(
            direction,
            entry,
            atr,
            rr
        )

        if levels is None:
            continue

        sl = levels["sl"]
        tp = levels["tp"]

        outcome = "OPEN"
        r_result = 0.0
        exit_bar = None

        future_end = min(
            i + 1 + max_bars_forward,
            len(data)
        )

        future = data.iloc[
            i + 1:future_end
        ]

        for j, candle in future.iterrows():

            high = float(
                candle["High"]
            )

            low = float(
                candle["Low"]
            )

            if direction == "BUY":

                hit_sl = low <= sl
                hit_tp = high >= tp

                if hit_sl and hit_tp:

                    # Conservative assumption when
                    # both levels are touched in one candle.
                    outcome = "LOSS"
                    r_result = -1.0
                    exit_bar = j
                    break

                if hit_sl:

                    outcome = "LOSS"
                    r_result = -1.0
                    exit_bar = j
                    break

                if hit_tp:

                    outcome = "WIN"
                    r_result = rr
                    exit_bar = j
                    break

            else:

                hit_sl = high >= sl
                hit_tp = low <= tp

                if hit_sl and hit_tp:

                    outcome = "LOSS"
                    r_result = -1.0
                    exit_bar = j
                    break

                if hit_sl:

                    outcome = "LOSS"
                    r_result = -1.0
                    exit_bar = j
                    break

                if hit_tp:

                    outcome = "WIN"
                    r_result = rr
                    exit_bar = j
                    break

        if outcome == "OPEN":

            # The target/SL wasn't reached within the
            # chosen forward window.
            outcome = "UNRESOLVED"
            r_result = 0.0

        results.append(
            {
                "Entry Time": data.index[i],
                "Direction": direction,
                "Entry": entry,
                "SL": sl,
                "TP": tp,
                "Outcome": outcome,
                "R Result": r_result,
                "Exit Time": exit_bar,
            }
        )

    if not results:

        return pd.DataFrame()

    return pd.DataFrame(
        results
    )


def summarize_backtest(results):

    if results.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "win_rate": 0.0,
            "net_r": 0.0
        }

    trades = len(
        results
    )

    wins = int(
        (
            results["Outcome"] ==
            "WIN"
        ).sum()
    )

    losses = int(
        (
            results["Outcome"] ==
            "LOSS"
        ).sum()
    )

    unresolved = int(
        (
            results["Outcome"] ==
            "UNRESOLVED"
        ).sum()
    )

    resolved = (
        wins +
        losses
    )

    win_rate = (
        wins /
        resolved *
        100
        if resolved > 0
        else 0
    )

    net_r = float(
        results["R Result"].sum()
    )

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "win_rate": win_rate,
        "net_r": net_r
    }


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "⚡ Kani Forex & Gold Analyzer"
)

st.caption(
    "5m–1h educational market analyzer • "
    "No automatic trade execution"
)


# ============================================================
# RISK WARNING
# ============================================================

with st.expander(
    "⚠️ IMPORTANT RISK WARNING",
    expanded=True
):

    st.warning(
        """
This application is for educational and analytical purposes.

Signals, entries, stop losses, take profits and backtest
results are not guarantees of future performance.

Historical results can differ substantially from live results.

Market data may be delayed, incomplete or inaccurate.

A higher R:R does not automatically mean a better setup.

No automatic trading or order execution is performed by this app.
"""
    )


# ============================================================
# STRATEGY MODE
# ============================================================

st.subheader(
    "1. Choose Strategy Mode"
)

strategy_mode = st.radio(
    "How should the analyzer determine the setup?",
    [
        "App Strategy",
        "My Strategy",
        "Analyze My Strategy"
    ],
    key="strategy_mode"
)


# ============================================================
# CUSTOM STRATEGY SETTINGS
# ============================================================

custom_settings = None

if strategy_mode == "My Strategy":

    st.markdown(
        "### ⚙️ Build Your Strategy"
    )

    trend_rule = st.selectbox(
        "Trend rule",
        [
            "No trend filter",
            "Price above 20-SMA",
            "Price below 20-SMA",
            "Bullish 20/50 SMA",
            "Bearish 20/50 SMA"
        ]
    )

    rsi_rule = st.selectbox(
        "RSI rule",
        [
            "No RSI filter",
            "RSI above 50",
            "RSI below 50",
            "RSI oversold",
            "RSI overbought"
        ]
    )

    macd_rule = st.selectbox(
        "MACD rule",
        [
            "No MACD filter",
            "Bullish MACD",
            "Bearish MACD"
        ]
    )

    confirmation_count = st.slider(
        "Minimum confirmations required",
        min_value=1,
        max_value=3,
        value=2
    )

    custom_settings = {
        "trend": trend_rule,
        "rsi": rsi_rule,
        "macd": macd_rule,
        "confirmations": confirmation_count
    }


# ============================================================
# STRATEGY DESCRIPTION
# ============================================================

strategy_description = ""

if strategy_mode == "Analyze My Strategy":

    st.markdown(
        "### 🧠 Describe Your Strategy"
    )

    strategy_description = st.text_area(
        "Write your strategy in normal language:",
        height=220,
        placeholder=(
            "Example:\n\n"
            "I first identify the overall trend. "
            "If the market is bullish, I wait for price "
            "to sweep the previous low. Then I want a "
            "bullish break of structure and a retracement "
            "before entering. My stop goes below the "
            "swing low and my target is based on R:R."
        )
    )

    st.info(
        "This version analyzes the structure of your "
        "description. A future AI layer can convert "
        "complex descriptions into more precise testable rules."
    )


# ============================================================
# RISK / REWARD
# ============================================================

st.subheader(
    "2. Risk / Reward"
)

rr_mode = st.radio(
    "How should the target be selected?",
    [
        "Auto",
        "Custom"
    ],
    horizontal=True
)

if rr_mode == "Custom":

    risk_reward = st.number_input(
        "Custom R:R",
        min_value=0.1,
        value=3.5,
        step=0.1,
        format="%.1f",
        help=(
            "There is no maximum. "
            "Examples: 1.5, 3.5, 10, 23, 50, 100."
        )
    )

else:

    risk_reward = None

    st.caption(
        "Auto mode uses nearby market structure and "
        "ATR to estimate a target."
    )


if (
    rr_mode == "Custom"
    and risk_reward >= 10
):

    st.warning(
        f"{risk_reward:.1f}R selected. "
        "The application will calculate the requested "
        "target, but such a large target may be far from "
        "current market structure."
    )


# ============================================================
# INSTRUMENT
# ============================================================

st.subheader(
    "3. Choose Instrument"
)

selected = st.selectbox(
    "Popular instruments",
    list(PAIRS.keys())
)

custom_symbol = st.text_input(
    "Or enter a Yahoo Finance symbol",
    placeholder="Example: EURUSD=X or GC=F"
)

symbol = (
    custom_symbol.strip()
    if custom_symbol.strip()
    else PAIRS[selected]
)


# ============================================================
# TIMEFRAME
# ============================================================

st.subheader(
    "4. Choose Timeframe"
)

interval = st.selectbox(
    "Candle timeframe",
    [
        "5m",
        "15m",
        "30m",
        "1h"
    ],
    index=1
)

data_range = TIMEFRAME_RANGES[
    interval
]


# ============================================================
# MAIN ANALYSIS BUTTON
# ============================================================

if st.button(
    "🔎 Analyze Market",
    type="primary",
    use_container_width=True
):

    if (
        strategy_mode ==
        "Analyze My Strategy"
        and not strategy_description.strip()
    ):

        st.error(
            "Please describe your strategy first."
        )

        st.stop()

    with st.spinner(
        f"Fetching {symbol} {interval} data..."
    ):

        try:

            df = fetch_intraday(
                symbol,
                interval,
                data_range
            )

            df = add_indicators(
                df
            )

            clean_df = clean_indicator_data(
                df
            )

        except Exception as error:

            st.error(
                f"Could not retrieve data for {symbol}."
            )

            st.code(
                str(error)
            )

            st.info(
                "Try another pair, 15m/30m timeframe, "
                "or check your internet connection."
            )

            st.stop()

    if len(clean_df) < 60:

        st.error(
            "Not enough usable candles for a reliable "
            "analysis. Try 15m or 30m."
        )

        st.stop()

    # --------------------------------------------------------
    # SELECT STRATEGY
    # --------------------------------------------------------

    if strategy_mode == "App Strategy":

        signal = app_strategy_signal(
            clean_df
        )

    elif strategy_mode == "My Strategy":

        signal = custom_strategy_signal(
            clean_df,
            custom_settings["trend"],
            custom_settings["rsi"],
            custom_settings["macd"],
            custom_settings["confirmations"]
        )

    else:

        # The current version uses the app's technical
        # engine for the market snapshot while separately
        # analyzing the user's written strategy.
        signal = app_strategy_signal(
            clean_df
        )

    # --------------------------------------------------------
    # AUTO R:R
    # --------------------------------------------------------

    if rr_mode == "Auto":

        auto_rr, auto_warnings = calculate_auto_rr(
            clean_df,
            signal["direction"],
            signal["price"],
            signal["atr"]
        )

        final_rr = auto_rr

    else:

        final_rr = float(
            risk_reward
        )

        auto_warnings = []

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        f"📊 Market Snapshot — {symbol} ({interval})"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Price",
        format_price(
            signal["price"],
            symbol
        )
    )

    c2.metric(
        "RSI",
        f"{signal['rsi']:.1f}"
    )

    c3.metric(
        "ATR",
        format_price(
            signal["atr"],
            symbol
        )
    )

    c4.metric(
        "Score",
        f"{signal['score']:.2f}"
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    st.subheader(
        "🎯 Current Analysis"
    )

    if signal["direction"] == "BUY":

        st.success(
            "🟢 BUY BIAS"
        )

    elif signal["direction"] == "SELL":

        st.error(
            "🔴 SELL BIAS"
        )

    else:

        st.info(
            "⚪ NO CLEAR SIGNAL"
        )

    st.write(
        f"**Strategy:** {strategy_mode}"
    )

    st.write(
        f"**Confidence:** {signal['confidence']}"
    )

    # --------------------------------------------------------
    # TRADE LEVELS
    # --------------------------------------------------------

    if signal["direction"] in [
        "BUY",
        "SELL"
    ]:

        levels = calculate_levels(
            signal["direction"],
            signal["price"],
            signal["atr"],
            final_rr
        )

        if levels:

            entry = levels["entry"]
            sl = levels["sl"]
            tp = levels["tp"]

            structure = check_market_structure(
                clean_df,
                signal["direction"],
                entry,
                sl,
                tp,
                signal["atr"]
            )

            st.markdown(
                "### 🎯 Calculated Levels"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "ENTRY",
                format_price(
                    entry,
                    symbol
                )
            )

            col2.metric(
                "STOP LOSS",
                format_price(
                    sl,
                    symbol
                )
            )

            col3.metric(
                "TAKE PROFIT",
                format_price(
                    tp,
                    symbol
                )
            )

            col4.metric(
                "R:R",
                f"{final_rr:.1f}R"
            )

            st.caption(
                "SL is currently based on 1.2 × ATR. "
                "The TP is calculated independently from "
                "the selected R:R."
            )

            # ------------------------------------------------
            # STRUCTURE
            # ------------------------------------------------

            st.markdown(
                "### 📐 Market Structure"
            )

            for observation in structure[
                "observations"
            ]:

                st.write(
                    f"• {observation}"
                )

            for warning in structure[
                "warnings"
            ]:

                st.warning(
                    warning
                )

            st.write(
                f"Previous 20-candle high: "
                f"{format_price(structure['recent_high'], symbol)}"
            )

            st.write(
                f"Previous 20-candle low: "
                f"{format_price(structure['recent_low'], symbol)}"
            )

            st.write(
                f"TP distance: approximately "
                f"{structure['target_atr']:.1f} ATR"
            )

    else:

        st.info(
            "No ENTRY / SL / TP generated because "
            "the selected strategy currently has no "
            "clear BUY or SELL setup."
        )

    # --------------------------------------------------------
    # AUTO R:R WARNINGS
    # --------------------------------------------------------

    for warning in auto_warnings:

        st.warning(
            warning
        )

    # --------------------------------------------------------
    # STRATEGY ANALYSIS
    # --------------------------------------------------------

    if strategy_mode == "Analyze My Strategy":

        st.subheader(
            "🧠 Your Strategy Analysis"
        )

        analysis = analyze_strategy_description(
            strategy_description
        )

        st.markdown(
            "#### What the analyzer detected"
        )

        for item in analysis[
            "detected"
        ]:

            st.write(
                f"✅ {item}"
            )

        st.markdown(
            "#### What needs more definition"
        )

        if analysis["missing"]:

            for item in analysis[
                "missing"
            ]:

                st.write(
                    f"⚠️ {item}"
                )

        else:

            st.success(
                "No major missing components were detected."
            )

        st.markdown(
            "#### Improvement suggestions"
        )

        for item in analysis[
            "suggestions"
        ]:

            st.write(
                f"💡 {item}"
            )

        st.info(
            "The written-strategy analyzer is currently "
            "rule-based. It does not yet claim to understand "
            "your strategy like a full AI model."
        )

    # --------------------------------------------------------
    # WHY?
    # --------------------------------------------------------

    with st.expander(
        "🔍 Why did the analyzer produce this result?"
    ):

        for reason in signal[
            "reasons"
        ]:

            st.write(
                f"• {reason}"
            )

        st.write(
            f"• 20-SMA: "
            f"{format_price(signal['sma20'], symbol)}"
        )

        st.write(
            f"• 50-SMA: "
            f"{format_price(signal['sma50'], symbol)}"
        )

        for warning in signal[
            "warnings"
        ]:

            st.warning(
                warning
            )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 Recent Price"
    )

    chart = clean_df[
        [
            "Close",
            "SMA_20",
            "SMA_50"
        ]
    ].tail(100)

    st.line_chart(
        chart
    )

    st.success(
        "Market analysis complete."
    )

    st.caption(
        f"Data: {data_range} | "
        f"Timeframe: {interval} | "
        f"R:R: {final_rr:.1f}R"
    )


# ============================================================
# BACKTEST SECTION
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Historical Backtest"
)

st.caption(
    "This tests the built-in/custom rule engine against "
    "the available historical candles. It is not a guarantee "
    "of future performance."
)

run_backtest = st.button(
    "Run Backtest",
    use_container_width=True
)

if run_backtest:

    if strategy_mode == "Analyze My Strategy":

        st.info(
            "The written strategy analyzer cannot yet "
            "backtest free-form text. Choose App Strategy "
            "or My Strategy for rule-based backtesting."
        )

    else:

        with st.spinner(
            "Running historical backtest..."
        ):

            try:

                backtest_df = fetch_intraday(
                    symbol,
                    interval,
                    data_range
                )

                backtest_df = add_indicators(
                    backtest_df
                )

                if rr_mode == "Auto":

                    # For backtesting, use 2R as a stable
                    # baseline rather than changing R:R
                    # dynamically during the test.
                    backtest_rr = 2.0

                    st.caption(
                        "Auto R:R backtest uses a fixed "
                        "2R baseline in this version."
                    )

                else:

                    backtest_rr = float(
                        risk_reward
                    )

                results = backtest_strategy(
                    backtest_df,
                    strategy_mode,
                    custom_settings,
                    backtest_rr
                )

                if results.empty:

                    st.warning(
                        "No qualifying setups were found "
                        "in the available historical data."
                    )

                else:

                    summary = summarize_backtest(
                        results
                    )

                    b1, b2, b3, b4 = st.columns(4)

                    b1.metric(
                        "Setups",
                        summary["trades"]
                    )

                    b2.metric(
                        "Wins",
                        summary["wins"]
                    )

                    b3.metric(
                        "Losses",
                        summary["losses"]
                    )

                    b4.metric(
                        "Win Rate",
                        f"{summary['win_rate']:.1f}%"
                    )

                    st.metric(
                        "Net R",
                        f"{summary['net_r']:.2f}R"
                    )

                    st.markdown(
                        "### Backtest Results"
                    )

                    st.dataframe(
                        results.tail(100),
                        use_container_width=True
                    )

                    st.info(
                        f"Unresolved setups: "
                        f"{summary['unresolved']}. "
                        "These are cases where neither TP nor "
                        "SL was reached inside the test window."
                    )

            except Exception as error:

                st.error(
                    "Backtest could not be completed."
                )

                st.code(
                    str(error)
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Kani Forex & Gold Analyzer • "
    "Educational software • "
    "No automatic order execution"
)
