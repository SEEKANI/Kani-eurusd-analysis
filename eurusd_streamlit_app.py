"""
KANI FOREX & GOLD ANALYZER
Educational only - no trade execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Kani Analyzer", page_icon="⚡")

PAIRS={
"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X",
"USD/CHF":"USDCHF=X","AUD/USD":"AUDUSD=X","USD/CAD":"USDCAD=X",
"NZD/USD":"NZDUSD=X","EUR/GBP":"EURGBP=X","EUR/JPY":"EURJPY=X",
"GBP/JPY":"GBPJPY=X","AUD/JPY":"AUDJPY=X","EUR/AUD":"EURAUD=X",
"Gold":"GC=F"}

RANGES={"5m":"5d","15m":"5d","30m":"10d","1h":"1mo"}

def price(x,s):
    return f"{x:.2f}" if s=="GC=F" else f"{x:.3f}" if "JPY" in s else f"{x:.5f}"

@st.cache_data(ttl=60)
def get_data(symbol,interval,rng):
    u=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    p={"interval":interval,"range":rng,"events":"history"}
    r=requests.get(u,params=p,headers={"User-Agent":"Mozilla/5.0"},timeout=20)
    r.raise_for_status()
    z=r.json()["chart"]["result"]
    if not z: raise ValueError("No market data returned.")
    z=z[0]
    q=z["indicators"]["quote"][0]
    d=pd.DataFrame({k:q[k] for k in ["open","high","low","close"]},
                   index=pd.to_datetime(z["timestamp"],unit="s"))
    d.columns=["Open","High","Low","Close"]
    return d.dropna().sort_index()

def indicators(d):
    d=d.copy()
    c=d.Close

    d["SMA20"]=c.rolling(20).mean()
    d["SMA50"]=c.rolling(50).mean()

    e1=c.ewm(span=8,adjust=False).mean()
    e2=c.ewm(span=17,adjust=False).mean()
    d["MACD"]=e1-e2
    d["MACDS"]=d.MACD.ewm(span=5,adjust=False).mean()

    ch=c.diff()
    g=ch.clip(lower=0).rolling(7).mean()
    l=(-ch.clip(upper=0)).rolling(7).mean()
    d["RSI"]=100-(100/(1+g/l.replace(0,np.nan)))

    tr=pd.concat([
        d.High-d.Low,
        (d.High-c.shift()).abs(),
        (d.Low-c.shift()).abs()
    ],axis=1).max(axis=1)

    d["ATR"]=tr.rolling(10).mean()
    d["PH"]=d.High.shift().rolling(20).max()
    d["PL"]=d.Low.shift().rolling(20).min()

    return d.dropna()

def result(side,conf,score,name,why,x):
    return {
        "direction":side,
        "confidence":conf,
        "score":score,
        "strategy":name,
        "why":why,
        "price":float(x.Close),
        "atr":float(x.ATR),
        "rsi":float(x.RSI),
        "sma20":float(x.SMA20),
        "sma50":float(x.SMA50)
    }

def app_strategy(d):
    x=d.iloc[-1]
    p=x.Close
    buy=0
    sell=0
    why=[]

    if p>x.SMA20:
        buy+=1
        why.append("Price is above the 20 SMA.")
    else:
        sell+=1
        why.append("Price is below the 20 SMA.")

    if p>x.SMA50:
        buy+=1
        why.append("Price is above the 50 SMA.")
    else:
        sell+=1
        why.append("Price is below the 50 SMA.")

    if x.RSI>55:
        buy+=1
        why.append(f"RSI supports buyers ({x.RSI:.1f}).")
    elif x.RSI<45:
        sell+=1
        why.append(f"RSI supports sellers ({x.RSI:.1f}).")

    if x.MACD>x.MACDS:
        buy+=1
        why.append("MACD is bullish.")
    else:
        sell+=1
        why.append("MACD is bearish.")

    if buy>=3:
        side="BUY"
        name="Trend Following"
    elif sell>=3:
        side="SELL"
        name="Trend Following"
    elif x.RSI<=30:
        side="BUY"
        name="Mean Reversion"
        why=["RSI is oversold."]
    elif x.RSI>=70:
        side="SELL"
        name="Mean Reversion"
        why=["RSI is overbought."]
    else:
        side="NO TRADE"
        name="No Clear Strategy"

    score=buy-sell
    conf="High" if abs(score)>=3 else "Moderate" if abs(score)>=2 else "Low"

    return result(side,conf,score,name,why,x)

def kani_strategy(d):
    x=d.iloc[-1]
    buy=0
    sell=0
    why=[]

    if x.Close>x.SMA20>x.SMA50:
        buy+=1
        why.append("Price and the 20/50 SMA structure are bullish.")

    if x.Close<x.SMA20<x.SMA50:
        sell+=1
        why.append("Price and the 20/50 SMA structure are bearish.")

    if x.RSI>50 and x.MACD>x.MACDS:
        buy+=1
        why.append("RSI and MACD confirm bullish momentum.")

    if x.RSI<50 and x.MACD<x.MACDS:
        sell+=1
        why.append("RSI and MACD confirm bearish momentum.")

    if buy>=2:
        side="BUY"
    elif sell>=2:
        side="SELL"
    else:
        side="NO TRADE"

    return result(
        side,
        "High" if abs(buy-sell)>=2 else "Moderate",
        buy-sell,
        "Kani Trend Confirmation",
        why,
        x
    )

def my_strategy(d,trend,rsi,macd,n):
    x=d.iloc[-1]
    b=0
    s=0
    why=[]

    if trend=="Above 20 SMA" and x.Close>x.SMA20:
        b+=1
        why.append("Price is above the 20 SMA.")

    if trend=="Below 20 SMA" and x.Close<x.SMA20:
        s+=1
        why.append("Price is below the 20 SMA.")

    if trend=="Bullish 20/50" and x.Close>x.SMA20>x.SMA50:
        b+=1
        why.append("Bullish 20/50 structure confirmed.")

    if trend=="Bearish 20/50" and x.Close<x.SMA20<x.SMA50:
        s+=1
        why.append("Bearish 20/50 structure confirmed.")

    if rsi=="Above 50" and x.RSI>50:
        b+=1
        why.append("RSI is above 50.")

    if rsi=="Below 50" and x.RSI<50:
        s+=1
        why.append("RSI is below 50.")

    if rsi=="Oversold" and x.RSI<=30:
        b+=1
        why.append("RSI is oversold.")

    if rsi=="Overbought" and x.RSI>=70:
        s+=1
        why.append("RSI is overbought.")

    if macd=="Bullish" and x.MACD>x.MACDS:
        b+=1
        why.append("MACD is bullish.")

    if macd=="Bearish" and x.MACD<x.MACDS:
        s+=1
        why.append("MACD is bearish.")

    side="BUY" if b>=n and b>s else "SELL" if s>=n and s>b else "NO TRADE"

    return result(
        side,
        "Custom Confirmed" if side!="NO TRADE" else "Conditions Not Met",
        b-s,
        "My Strategy",
        why,
        x
    )

def levels(side,entry,atr,rr):
    if side=="BUY":
        sl=entry-1.2*atr
        tp=entry+1.2*atr*rr
    elif side=="SELL":
        sl=entry+1.2*atr
        tp=entry-1.2*atr*rr
    else:
        return None
    return sl,tp

def strategy_text(text):
    t=text.lower()
    found=[]
    missing=[]

    checks={
        "Trend rule":["trend","sma","ema","bullish","bearish"],
        "Entry rule":["entry","enter","buy","sell","breakout","pullback","retest"],
        "Market structure":["structure","bos","higher high","lower low"],
        "Stop-loss rule":["stop loss","stop-loss","sl"],
        "Target/R:R rule":["take profit","target","tp","risk reward","r:r","rr"],
        "News/session filter":["news","cpi","nfp","london","new york","session"]
    }

    for name,words in checks.items():
        if any(w in t for w in words):
            found.append(name)
        else:
            missing.append(name)

    return found,missing

st.title("⚡ Kani Forex & Gold Analyzer")
st.caption("5m–1h market analysis • Educational only • No order execution")

with st.expander("⚠️ Risk warning",expanded=True):
    st.warning(
        "Signals are analytical ideas, not guarantees. "
        "Historical results do not predict future results. "
        "Trading can cause losses."
    )

st.subheader("1. Strategy")

mode=st.radio(
    "Choose mode",
    [
        "🤖 App Strategy",
        "🎯 Kani Strategy",
        "🧠 My Strategy",
        "🔎 Analyze My Strategy"
    ]
)

trend=rsi=macd=None
n=2
text=""

if mode=="🧠 My Strategy":

    trend=st.selectbox(
        "Trend",
        [
            "No filter",
            "Above 20 SMA",
            "Below 20 SMA",
            "Bullish 20/50",
            "Bearish 20/50"
        ]
    )

    rsi=st.selectbox(
        "RSI",
        [
            "No filter",
            "Above 50",
            "Below 50",
            "Oversold",
            "Overbought"
        ]
    )

    macd=st.selectbox(
        "MACD",
        [
            "No filter",
            "Bullish",
            "Bearish"
        ]
    )

    n=st.slider(
        "Minimum confirmations",
        1,3,2
    )

if mode=="🔎 Analyze My Strategy":

    text=st.text_area(
        "Describe your strategy",
        height=180,
        placeholder="Example: I follow the trend, wait for a pullback, confirm structure, then enter. Stop below the swing and target 3R."
    )

st.subheader("2. Risk / Reward")

rrmode=st.radio(
    "Target mode",
    ["Auto","Custom"],
    horizontal=True
)

if rrmode=="Custom":
    rr=st.number_input(
        "Custom R:R",
        min_value=0.1,
        value=3.5,
        step=0.1
    )
else:
    rr=2.0
    st.caption("Auto uses a 2R baseline.")

if rrmode=="Custom" and rr>=10:
    st.warning(
        f"{rr:g}R selected. Large targets may be difficult to reach."
    )

st.subheader("3. Market")

sel=st.selectbox(
    "Instrument",
    list(PAIRS)
)

custom=st.text_input(
    "Or Yahoo symbol",
    placeholder="EURUSD=X or GC=F"
)

symbol=custom.strip() or PAIRS[sel]

tf=st.selectbox(
    "Timeframe",
    ["5m","15m","30m","1h"],
    index=1
)

if st.button(
    "🔎 Analyze Market",
    type="primary",
    use_container_width=True
):

    if mode=="🔎 Analyze My Strategy" and not text.strip():
        st.error("Describe your strategy first.")
        st.stop()

    try:
        d=indicators(
            get_data(
                symbol,
                tf,
                RANGES[tf]
            )
        )

        if len(d)<60:
            st.error(
                "Not enough candles. Try 15m or 30m."
            )
            st.stop()

    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

    if mode=="🤖 App Strategy":
        sig=app_strategy(d)

    elif mode=="🎯 Kani Strategy":
        sig=kani_strategy(d)

    elif mode=="🧠 My Strategy":
        sig=my_strategy(
            d,
            trend,
            rsi,
            macd,
            n
        )

    else:
        sig=app_strategy(d)

    st.subheader("📊 Market Snapshot")

    a,b,c,d1=st.columns(4)

    a.metric(
        "Price",
        price(sig["price"],symbol)
    )

    b.metric(
        "RSI",
        f"{sig['rsi']:.1f}"
    )

    c.metric(
        "ATR",
        price(sig["atr"],symbol)
    )

    d1.metric(
        "Score",
        sig["score"]
    )

    st.subheader("🎯 Result")

    if sig["direction"]=="BUY":
        st.success("🟢 BUY BIAS")

    elif sig["direction"]=="SELL":
        st.error("🔴 SELL BIAS")

    else:
        st.info("⚪ NO TRADE")

    st.write(f"**Mode:** {mode}")
    st.write(f"**Strategy used:** {sig['strategy']}")
    st.write(f"**Confidence:** {sig['confidence']}")

    if sig["why"]:

        st.markdown("### 🧠 Why this strategy?")

        for w in sig["why"]:
            st.write("• "+w)

        st.info(
            "The strategy is selected from current indicator "
            "conditions. It does not guarantee the next move."
        )

    if sig["direction"]!="NO TRADE":

        lv=levels(
            sig["direction"],
            sig["price"],
            sig["atr"],
            rr
        )

        sl,tp=lv

        c1,c2,c3,c4=st.columns(4)

        c1.metric(
            "ENTRY",
            price(sig["price"],symbol)
        )

        c2.metric(
            "STOP LOSS",
            price(sl,symbol)
        )

        c3.metric(
            "TAKE PROFIT",
            price(tp,symbol)
        )

        c4.metric(
            "R:R",
            f"{rr:g}R"
        )

        st.caption(
            "SL uses 1.2 × ATR. TP follows the selected R:R."
        )

    if mode=="🔎 Analyze My Strategy":

        found,missing=strategy_text(text)

        st.markdown("### 🔎 Strategy Review")

        for x in found:
            st.write("✅ "+x)

        for x in missing:
            st.write("⚠️ Missing/unclear: "+x)

        st.info(
            "This checks whether your written strategy "
            "contains measurable components."
        )

    st.subheader("📈 Price Chart")

    st.line_chart(
        d[
            ["Close","SMA20","SMA50"]
        ].tail(100)
    )

st.markdown("---")
st.caption(
    "Kani Analyzer • Educational software • No automatic trading"
)
