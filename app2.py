import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 8.1: 戰略修復版", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (加入錯誤處理) ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    # 移除可能導致錯誤的特殊符號，改用標準大盤
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="🦅 Alpha 8.1 正在建立連線...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載: {t} ...")
            df = yf.Ticker(t).history(period="2y", auto_adjust=True) # 改回2年，減少數據負擔
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
            
    progress_bar.empty()
    # 確保有數據，否則回傳空 DataFrame
    try:
        return (pd.DataFrame(data['Close']).ffill(), 
                pd.DataFrame(data['High']).ffill(), 
                pd.DataFrame(data['Low']).ffill(),
                pd.DataFrame(data['Volume']).ffill())
    except:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

@st.cache_data(ttl=3600*24)
def get_fundamental_anchor(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Target_Mean': info.get('targetMeanPrice'), 
            'Forward_PE': info.get('forwardPE'),
            'Trailing_PE': info.get('trailingPE')
        }
    except: return {}

# --- 2. 機器學習引擎 (加入防呆) ---

def train_ai_model(target_ticker, df_close, df_vol, days_forecast=22):
    try:
        # 數據檢查
        if target_ticker not in df_close.columns: return None
        
        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close[target_ticker]
        
        # 技術指標
        df['Vol'] = df['Close'].pct_change().rolling(20).std()
        
        # 宏觀因子 (檢查是否存在)
        if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
        if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
            
        df['Target'] = df['Close'].shift(-days_forecast)
        df = df.dropna()
        
        if len(df) < 50: return None # 數據過少不訓練
        
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42) # 輕量化模型
        model.fit(X, y)
        
        latest_features = X.iloc[[-1]]
        return model.predict(latest_features)[0]
    except: return None

# --- 3. 核心運算 ---

def calc_kelly(trend_status, win_rate=0.55):
    if "Bull" in trend_status: win_rate += 0.1
    if "Bear" in trend_status: win_rate -= 0.15
    f_star = (win_rate * 2.0 - 1) / 1.0 # 簡化版 Kelly
    return max(0, f_star * 0.5)

def calc_targets_composite_v2(ticker, close, high, low, vol, f_data, days_forecast=22):
    if len(close) < 100: return None # 降低門檻
    
    # ATR
    try:
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = close.iloc[-1] + (atr * np.sqrt(days_forecast))
    except: t_atr = None
    
    # MC
    try:
        returns = close.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sims = [close.iloc[-1] * ((1 + mu)**days_forecast) for _ in range(10)] # 極簡化模擬
        t_mc = np.mean(sims)
    except: t_mc = None
    
    # Fib
    try:
        recent = close.iloc[-60:]
        h, l = recent.max(), recent.min()
        t_fib = h + (h - l) * 0.618 
    except: t_fib = None
    
    # Fund
    t_fund = f_data.get('Target_Mean')
    
    # AI
    try:
        t_ai = train_ai_model(ticker, close.to_frame(ticker).join(close.to_frame('^VIX',), rsuffix='_vix'), vol, days_forecast)
    except: t_ai = None
    
    # Avg
    targets = [t for t in [t_atr, t_mc, t_fib, t_ai] if t is not None]
    t_avg = sum(targets) / len(targets) if targets else None
    
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "AI": t_ai, "Avg": t_avg}

def analyze_trend(series):
    if series is None or len(series) < 60: return {"status": "資料不足", "p_now": 0, "p_2w": 0, "p_1m": 0, "p_3m": 0}
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1] if len(series) > 200 else series.rolling(50).mean().iloc[-1]
    
    status = "🛡️ 震盪"
    if p_now > sma200: status = "🔥 多頭"
    elif p_now < sma200 * 0.9: status = "🛑 空頭"
    
    # 簡單線性預測
    try:
        y = series.values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        p_1m = model.predict([[len(y)+22]])[0].item()
    except: p_1m = p_now
    
    return {"status": status, "p_now": p_now, "p_1m": p_1m}

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN APP ---
def main():
    st.title("Alpha 8.1: 戰略修復版 (Safe Mode)")
    st.caption("v8.1 | 防呆機制 | AI 輕量化 | 確保運作")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key (選填)", type="password"))
        
        st.header("💼 資產")
        default_input = """BTC-USD, 10000
AMD, 10000
NVDA, 10000"""
        user_input = st.text_area("清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產", f"${total_value:,.0f}")
        if st.button("🚀 啟動修復版", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 正在執行安全運算..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        fund_data = {t: get_fundamental_anchor(t) for t in tickers_list}

    if df_close.empty: 
        st.error("❌ 無法獲取市場數據。請檢查網路或資產代碼是否正確。")
        st.stop()

    # --- PART 1: 宏觀 ---
    st.subheader("1. 宏觀儀表 (Macro)")
    
    # 安全獲取宏觀數據
    vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
    tnx = df_close['^TNX'].iloc[-1] if '^TNX' in df_close.columns else 0
    dxy = df_close['DX-Y.NYB'].iloc[-1] if 'DX-Y.NYB' in df_close.columns else 0
    
    # 銅金比 (防呆)
    try:
        cg_ratio = (df_close['HG=F'].iloc[-1] / df_close['GC=F'].iloc[-1]) * 1000
    except: cg_ratio = 0
    
    liq_val = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💧 淨流動性", f"${liq_val:.2f}T" if liq_val else "N/A")
    c2.metric("🌪️ VIX", f"{vix:.2f}")
    c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%")
    c4.metric("🏭 銅金比", f"{cg_ratio:.2f}")

    st.markdown("---")

    # --- PART 2: 個股 ---
    st.subheader("2. 個股戰略 (AI & Targets)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        
        trend = analyze_trend(df_close[ticker])
        info = fund_data.get(ticker, {})
        targets = calc_targets_composite_v2(ticker, df_close, df_high, df_low, df_vol, info)
        obv = calc_obv(df_close[ticker], df_vol[ticker])
        
        t_avg_s = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
        
        with st.expander(f"🦅 {ticker} | {trend['status']} | 綜合目標: {t_avg_s}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1:
                st.markdown("#### 📉 價格趨勢")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_close.index[-100:], y=df_close[ticker].iloc[-100:], name='Price', line=dict(color='#00FF7F')))
                if obv is not None:
                    fig.add_trace(go.Scatter(x=df_close.index[-100:], y=obv.iloc[-100:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False))
                st.plotly_chart(fig, use_container_width=True)
            
            with k2:
                st.markdown("#### 🤖 五角定位 (1M)")
                if targets:
                    st.write(f"**ATR:** ${targets['ATR']:.2f}" if targets['ATR'] else "-")
                    st.write(f"**MC:** ${targets['MC']:.2f}" if targets['MC'] else "-")
                    st.write(f"**Fib:** ${targets['Fib']:.2f}" if targets['Fib'] else "-")
                    st.write(f"**AI:** ${targets['AI']:.2f}" if targets['AI'] else "N/A")
                    st.write(f"**Fund:** ${targets['Fund']}" if targets['Fund'] else "N/A")

            with k3:
                st.markdown("#### 🔮 趨勢")
                st.metric("1月預測", f"${trend['p_1m']:.2f}")
                st.metric("PE", f"{info.get('Forward_PE')}")

if __name__ == "__main__":
    main()