"""KANI FOREX & GOLD ANALYZER — Educational only, no trade execution."""
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Kani Analyzer", page_icon="⚡")

PAIRS={"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X","USD/CHF":"USDCHF=X",
"AUD/USD":"AUDUSD=X","USD/CAD":"USDCAD=X","NZD/USD":"NZDUSD=X","EUR/GBP":"EURGBP=X",
"EUR/JPY":"EURJPY=X","GBP/JPY":"GBPJPY=X","AUD/JPY":"AUDJPY=X","EUR/AUD":"EURAUD=X","Gold":"GC=F"}
RANGES={"5m":"5d","15m":"5d","30m":"10d","1h":"1mo"}

def price(x,s): return f"{x:.2f}" if s=="GC=F" else f"{x:.3f}" if "JPY" in s else f"{x:.5f}"

@st.cache_data(ttl=60)
def get_data(symbol,interval,rng):
    u=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r=requests.get(u,params={"interval":interval,"range":rng,"events":"history"},
                   headers={"User-Agent":"Mozilla/5.0"},timeout=20)
    r.raise_for_status()
    z=r.json()["chart"]["result"]
    if not z: raise ValueError("No market data returned.")
    z=z[0]; q=z["indicators"]["quote"][0]
    d=pd.DataFrame({k:q[k] for k in ["open","high","low","close"]},index=pd.to_datetime(z["timestamp"],unit="s"))
    d.columns=["Open","High","Low","Close"]
    return d.dropna().sort_index()

def swings(d,lr=2):
    w=2*lr+1
    swh=d.High.where(d.High==d.High.rolling(w,center=True).max()).shift(lr).ffill()
    swl=d.Low.where(d.Low==d.Low.rolling(w,center=True).min()).shift(lr).ffill()
    return swh,swl

def indicators(d):
    d=d.copy(); c=d.Close
    d["SMA20"]=c.rolling(20).mean(); d["SMA50"]=c.rolling(50).mean()
    d["MACD"]=c.ewm(span=8,adjust=False).mean()-c.ewm(span=17,adjust=False).mean()
    d["MACDS"]=d.MACD.ewm(span=5,adjust=False).mean()
    ch=c.diff(); g=ch.clip(lower=0).rolling(7).mean(); l=(-ch.clip(upper=0)).rolling(7).mean()
    d["RSI"]=(100-(100/(1+g/l.replace(0,np.nan)))).fillna(100)
    tr=pd.concat([d.High-d.Low,(d.High-c.shift()).abs(),(d.Low-c.shift()).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(10).mean()
    up=d.High.diff(); dn=-d.Low.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    tr14=tr.rolling(14).sum()
    pdi=100*pd.Series(pdm,index=d.index).rolling(14).sum()/tr14
    mdi=100*pd.Series(mdm,index=d.index).rolling(14).sum()/tr14
    d["ADX"]=(((pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan))*100).rolling(14).mean()
    d["SWH"],d["SWL"]=swings(d)
    return d.dropna()

def regime(x): return "Trending" if x.ADX>=25 else "Ranging" if x.ADX<20 else "Mixed"

def score(rules):
    b=s=0; why=[]
    for cond,side,msg in rules:
        if cond:
            why.append(msg)
            if side=="B": b+=1
            else: s+=1
    return b,s,why

def result(side,conf,sc,name,why,x):
    return {"direction":side,"confidence":conf,"score":sc,"strategy":name,"why":why,
            "price":float(x.Close),"atr":float(x.ATR),"rsi":float(x.RSI)}

def app_strategy(d):
    x=d.iloc[-1]; p=x.Close; reg=regime(x)
    b,s,why=score([
        (p>x.SMA20,"B","Price above 20 SMA."),(p<=x.SMA20,"S","Price below 20 SMA."),
        (p>x.SMA50,"B","Price above 50 SMA."),(p<=x.SMA50,"S","Price below 50 SMA."),
        (x.RSI>55,"B",f"RSI supports buyers ({x.RSI:.1f})."),(x.RSI<45,"S",f"RSI supports sellers ({x.RSI:.1f})."),
        (x.MACD>x.MACDS,"B","MACD bullish."),(x.MACD<=x.MACDS,"S","MACD bearish.")])
    if b>=3 and reg!="Ranging": side,name="BUY","Trend Following"
    elif s>=3 and reg!="Ranging": side,name="SELL","Trend Following"
    elif x.RSI<=30 and reg!="Trending": side,name,why="BUY","Mean Reversion",["RSI oversold."]
    elif x.RSI>=70 and reg!="Trending": side,name,why="SELL","Mean Reversion",["RSI overbought."]
    else: side,name="NO TRADE","No Clear Strategy"
    why.append(f"Regime: {reg} (ADX {x.ADX:.1f}).")
    sc=b-s; conf="High" if abs(sc)>=3 else "Moderate" if abs(sc)>=2 else "Low"
    return result(side,conf,sc,name,why,x)

def kani_strategy(d):
    x=d.iloc[-1]; reg=regime(x)
    b,s,why=score([
        (x.Close>x.SMA20>x.SMA50,"B","Bullish 20/50 structure."),(x.Close<x.SMA20<x.SMA50,"S","Bearish 20/50 structure."),
        (x.RSI>50 and x.MACD>x.MACDS,"B","RSI+MACD bullish momentum."),(x.RSI<50 and x.MACD<x.MACDS,"S","RSI+MACD bearish momentum.")])
    side="BUY" if b>=2 and reg!="Ranging" else "SELL" if s>=2 and reg!="Ranging" else "NO TRADE"
    why.append(f"Regime: {reg} (ADX {x.ADX:.1f}).")
    return result(side,"High" if abs(b-s)>=2 else "Moderate",b-s,"Kani Trend Confirmation",why,x)

def my_strategy(d,trend,rsi,macd,n):
    x=d.iloc[-1]
    opts={"Above 20 SMA":(x.Close>x.SMA20,"B","Price above 20 SMA."),"Below 20 SMA":(x.Close<x.SMA20,"S","Price below 20 SMA."),
    "Bullish 20/50":(x.Close>x.SMA20>x.SMA50,"B","Bullish 20/50 confirmed."),"Bearish 20/50":(x.Close<x.SMA20<x.SMA50,"S","Bearish 20/50 confirmed."),
    "Above 50":(x.RSI>50,"B","RSI above 50."),"Below 50":(x.RSI<50,"S","RSI below 50."),
    "Oversold":(x.RSI<=30,"B","RSI oversold."),"Overbought":(x.RSI>=70,"S","RSI overbought."),
    "Bullish":(x.MACD>x.MACDS,"B","MACD bullish."),"Bearish":(x.MACD<x.MACDS,"S","MACD bearish.")}
    rules=[opts[v] for v in (trend,rsi,macd) if v in opts]
    b,s,why=score(rules)
    side="BUY" if b>=n and b>s else "SELL" if s>=n and s>b else "NO TRADE"
    why.append(f"Regime: {regime(x)} (ADX {x.ADX:.1f}).")
    return result(side,"Custom Confirmed" if side!="NO TRADE" else "Conditions Not Met",b-s,"My Strategy",why,x)

@st.cache_data(ttl=3600)
def yiza_daily(symbol): return indicators(get_data(symbol,"1d","1y"))

def yiza_bias_at(daily,ts):
    sub=daily[daily.index<=ts]
    if len(sub)<1: return None
    x=sub.iloc[-1]
    return "BUY" if x.Close>x.SMA20>x.SMA50 else "SELL" if x.Close<x.SMA20<x.SMA50 else None

def yiza_strategy(d,daily):
    x=d.iloc[-1]; p=d.iloc[-2]
    bias=yiza_bias_at(daily,x.name)
    if bias is None: return result("NO TRADE","Low",0,"Yiza Strategy",["Could not read the daily/weekly bias yet."],x)
    why=[f"Daily/weekly bias: {bias} only (higher-timeframe filter)."]
    swept_low=p.Low<x.SWL and x.Close>x.SWL; swept_high=p.High>x.SWH and x.Close<x.SWH
    side="NO TRADE"
    if bias=="BUY" and (swept_low or x.Close>x.SWH):
        side="BUY"; why.append("Sweep/rejection at support or break of structure confirms BUY.")
    elif bias=="SELL" and (swept_high or x.Close<x.SWL):
        side="SELL"; why.append("Sweep/rejection at resistance or break of structure confirms SELL.")
    else: why.append("No valid sweep or break of structure aligned with bias yet — wait.")
    why.append(f"Regime: {regime(x)} (ADX {x.ADX:.1f}).")
    return result(side,"High" if side!="NO TRADE" else "Low",1 if side=="BUY" else -1 if side=="SELL" else 0,"Yiza Strategy",why,x)

def levels(side,entry,atr,rr):
    if side=="BUY": return entry-1.2*atr, entry+1.2*atr*rr
    if side=="SELL": return entry+1.2*atr, entry-1.2*atr*rr
    return None

def backtest(d,engine,rr,lookahead=40):
    trades=[]
    for i in range(60,len(d)-1):
        sig=engine(d.iloc[:i+1])
        if sig["direction"]=="NO TRADE": continue
        sl,tp=levels(sig["direction"],sig["price"],sig["atr"],rr); outcome=None
        for j in range(i+1,min(i+1+lookahead,len(d))):
            hi=d.High.iloc[j]; lo=d.Low.iloc[j]
            if sig["direction"]=="BUY":
                if lo<=sl: outcome=-1.0; break
                if hi>=tp: outcome=rr; break
            else:
                if hi>=sl: outcome=-1.0; break
                if lo<=tp: outcome=rr; break
        if outcome is not None: trades.append(outcome)
    return trades

def strategy_text(text):
    t=text.lower(); found=[]; missing=[]
    checks={"Trend rule":["trend","sma","ema","bullish","bearish"],
    "Entry rule":["entry","enter","buy","sell","breakout","pullback","retest"],
    "Market structure":["structure","bos","higher high","lower low"],
    "Stop-loss rule":["stop loss","stop-loss","sl"],
    "Target/R:R rule":["take profit","target","tp","risk reward","r:r","rr"],
    "News/session filter":["news","cpi","nfp","london","new york","session"]}
    for name,words in checks.items(): (found if any(w in t for w in words) else missing).append(name)
    return found,missing

# ---UI BELOW---
st.title("⚡ Kani Forex & Gold Analyzer")
st.caption("5m–1h market analysis • Educational only • No order execution")
with st.expander("⚠️ Risk warning",expanded=True):
    st.warning("Signals are analytical ideas, not guarantees. Historical results do not predict future results. Trading can cause losses.")

st.subheader("1. Strategy")
mode=st.radio("Choose mode",["🤖 App Strategy","🎯 Kani Strategy","🧬 Yiza Strategy","🧠 My Strategy","🔎 Analyze My Strategy"])
trend=rsi=macd=None; n=2; text=""
if mode=="🧠 My Strategy":
    trend=st.selectbox("Trend",["No filter","Above 20 SMA","Below 20 SMA","Bullish 20/50","Bearish 20/50"])
    rsi=st.selectbox("RSI",["No filter","Above 50","Below 50","Oversold","Overbought"])
    macd=st.selectbox("MACD",["No filter","Bullish","Bearish"])
    n=st.slider("Minimum confirmations",1,3,2)
if mode=="🔎 Analyze My Strategy":
    text=st.text_area("Describe your strategy",height=180,
        placeholder="Example: I follow the trend, wait for a pullback, confirm structure, then enter. Stop below the swing and target 3R.")
if mode=="🧬 Yiza Strategy":
    st.caption("Daily/weekly bias → wait for a sweep/rejection or break of structure on your entry timeframe → min 3R target.")

st.subheader("2. Risk / Reward")
rrmode=st.radio("Target mode",["Auto","Custom"],horizontal=True)
rr=st.number_input("Custom R:R",min_value=0.1,value=3.5,step=0.1) if rrmode=="Custom" else 2.0
if rrmode=="Auto": st.caption("Auto uses a 2R baseline.")
if mode=="🧬 Yiza Strategy" and rr<3: rr=3.0; st.caption("Yiza Strategy enforces a minimum 3R target.")
if rrmode=="Custom" and rr>=10: st.warning(f"{rr:g}R selected. Large targets may be difficult to reach.")

st.subheader("3. Market")
sel=st.selectbox("Instrument",list(PAIRS))
custom=st.text_input("Or Yahoo symbol",placeholder="EURUSD=X or GC=F")
symbol=custom.strip() or PAIRS[sel]
tf=st.selectbox("Timeframe",["5m","15m","30m","1h"],index=1)
run_bt=st.checkbox("Also run a quick backtest on this strategy",value=False)

if st.button("🔎 Analyze Market",type="primary",use_container_width=True):
    if mode=="🔎 Analyze My Strategy" and not text.strip(): st.error("Describe your strategy first."); st.stop()
    try:
        d=indicators(get_data(symbol,tf,RANGES[tf]))
        if len(d)<80: st.error("Not enough candles. Try 15m or 30m."); st.stop()
        daily=yiza_daily(symbol) if mode=="🧬 Yiza Strategy" or run_bt else None
    except Exception as e:
        st.error(f"Data error: {e}"); st.stop()

    engines={"🤖 App Strategy":lambda s:app_strategy(s),"🎯 Kani Strategy":lambda s:kani_strategy(s),
    "🧬 Yiza Strategy":lambda s:yiza_strategy(s,daily),"🧠 My Strategy":lambda s:my_strategy(s,trend,rsi,macd,n)}
    engine=engines.get(mode,app_strategy)
    sig=engine(d)

    st.subheader("📊 Market Snapshot")
    for col,label,val in zip(st.columns(4),["Price","RSI","ATR","Score"],
        [price(sig["price"],symbol),f"{sig['rsi']:.1f}",price(sig["atr"],symbol),sig["score"]]):
        col.metric(label,val)

    st.subheader("🎯 Result")
    {"BUY":st.success,"SELL":st.error}.get(sig["direction"],st.info)(
        {"BUY":"🟢 BUY BIAS","SELL":"🔴 SELL BIAS"}.get(sig["direction"],"⚪ NO TRADE"))
    st.write(f"**Mode:** {mode}")
    st.write(f"**Strategy used:** {sig['strategy']}")
    st.write(f"**Confidence:** {sig['confidence']}")

    if sig["why"]:
        st.markdown("### 🧠 Why this strategy?")
        for w in sig["why"]: st.write("• "+w)
        st.info("The strategy is selected from current indicator conditions. It does not guarantee the next move.")

    if sig["direction"]!="NO TRADE":
        sl,tp=levels(sig["direction"],sig["price"],sig["atr"],rr)
        for col,label,val in zip(st.columns(4),["ENTRY","STOP LOSS","TAKE PROFIT","R:R"],
            [price(sig["price"],symbol),price(sl,symbol),price(tp,symbol),f"{rr:g}R"]):
            col.metric(label,val)
        st.caption("SL uses 1.2 × ATR. TP follows the selected R:R.")

    if mode=="🔎 Analyze My Strategy":
        found,missing=strategy_text(text)
        st.markdown("### 🔎 Strategy Review")
        for x in found: st.write("✅ "+x)
        for x in missing: st.write("⚠️ Missing/unclear: "+x)
        st.info("This checks whether your written strategy contains measurable components.")

    if run_bt and mode!="🔎 Analyze My Strategy":
        st.subheader("📊 Backtest (this instrument & timeframe only)")
        trades=backtest(d,engine,rr)
        if not trades: st.info("No completed signals found in this history window.")
        else:
            wins=[t for t in trades if t>0]; eq=np.cumsum(trades)
            for col,label,val in zip(st.columns(4),["Signals tested","Win rate","Avg R","Max drawdown"],
                [len(trades),f"{len(wins)/len(trades)*100:.0f}%",f"{sum(trades)/len(trades):.2f}",
                 f"{(np.maximum.accumulate(eq)-eq).max():.1f}R"]):
                col.metric(label,val)
            st.line_chart(pd.Series(eq,name="Cumulative R"))
            st.caption("Backtest replays this exact strategy bar-by-bar on the loaded history. Small sample sizes are not statistically reliable — treat as a sanity check, not proof of edge.")

    st.subheader("📈 Price Chart")
    st.line_chart(d[["Close","SMA20","SMA50"]].tail(100))

st.markdown("---")
st.caption("Kani Analyzer • Educational software • No automatic trading")
