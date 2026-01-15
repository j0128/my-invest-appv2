import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 6.1: 綜合智能戰略", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .explanation-box {background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #00BFFF;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', '^IRX', 'HYG'] 
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="🦅 Alpha 6.1 正在執行綜合運算...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載: {t} ...")
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
    progress_bar.empty()
    return (pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), 
            pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill())

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

# --- 2. 核心運算 (綜合模型) ---

def calc_kelly(trend_status, win_rate=0.55, odds=2.0):
    if "Bull" in trend_status: win_rate += 0.1
    if "Bear" in trend_status: win_rate -= 0.15
    f_star = (win_rate * (odds + 1) - 1) / odds
    return max(0, f_star * 0.5)

def calc_trend_projection(series, days_future):
    """計算線性回歸預測值"""
    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.predict([[len(y) + days_future]])[0].item()

def calc_targets_composite(close, high, low, f_data, days_forecast=22):
    """
    四角定位 + 平均值 (Composite)
    """
    if len(close) < 252: return None
    
    # 1. ATR (物理極限 - 趨勢調整版)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    price_projected = calc_trend_projection(close.iloc[-126:], days_forecast) 
    t_atr = price_projected + (atr * np.sqrt(days_forecast))
    
    # 2. Monte Carlo (統計中樞 P50)
    returns = close.iloc[-252:].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    sims = []
    for _ in range(1000):
        p = close.iloc[-1]
        for _ in range(days_forecast): p *= (1 + np.random.normal(mu, sigma))
        sims.append(p)
    t_mc = np.percentile(sims, 50)
    
    # 3. Fibonacci (群眾心理 1.618)
    recent = close.iloc[-60:]
    h, l = recent.max(), recent.min()
    t_fib = h + (h - l) * 0.618 
    
    # 4. Fundamental (價值)
    t_fund = f_data.get('Target_Mean')
    
    # 計算平均值 (Composite) - 只計算技術面，因為基本面有時會缺失或極端
    valid_targets = [t for t in [t_atr, t_mc, t_fib] if t is not None]
    t_avg = sum(valid_targets) / len(valid_targets) if valid_targets else None
    
    return {
        "ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "Avg": t_avg
    }

def run_backtest_composite(close, high, low, days_ago=22):
    """
    全模組回測：回到過去，計算當時的平均預測，驗證今日誤差
    """
    if len(close) < 300: return None
    
    # 時光倒流
    idx_past = len(close) - days_ago - 1
    p_now = close.iloc[-1]
    
    # 切片: 拿到當時的數據
    c_slice = close.iloc[:idx_past+1]
    h_slice = high.iloc[:idx_past+1]
    l_slice = low.iloc[:idx_past+1]
    
    # --- 重跑模型 (當時視角) ---
    
    # 1. Past ATR
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    y = c_slice.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    pred_trend = model.predict([[len(y) + days_ago]])[0].item()
    past_atr = pred_trend + (atr * np.sqrt(days_ago))
    
    # 2. Past Fib
    recent = c_slice.iloc[-60:]
    ph, pl = recent.max(), recent.min()
    past_fib = ph + (ph - pl) * 0.618
    
    # 3. Past MC (簡化模擬)
    returns = c_slice.iloc[-252:].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    # 這裡只做一次簡單推估作為回測代表: P * (1+mu)^t
    # 或是跑 100 次小模擬
    sims = []
    for _ in range(100):
        p = c_slice.iloc[-1]
        for _ in range(days_ago): p *= (1 + np.random.normal(mu, sigma))
        sims.append(p)
    past_mc = np.percentile(sims, 50)
    
    # 計算當時的平均預測
    past_avg = (past_atr + past_fib + past_mc) / 3
    
    # 計算誤差
    err = (past_avg - p_now) / p_now
    
    return {"Past_Avg": past_avg, "Error": err, "Price_Now": p_now}

def analyze_trend_matrix(series):
    if len(series) < 126: return None
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    p_2w = model.predict([[len(y)+10]])[0].item()
    p_1m = model.predict([[len(y)+22]])[0].item()
    p_3m = model.predict([[len(y)+66]])[0].item()
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    
    status = "🛡️ 區間"
    if p_now > sma200: status = "🔥 牛市"
    elif p_now < sma200 * 0.9: status = "🛑 熊市"
    else: status = "⚠️ 弱勢"
        
    return {"p_2w": p_2w, "p_1m": p_1m, "p_3m": p_3m, "status": status}

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
    st.title("Alpha 6.1: 綜合智能戰略 (Composite Intelligence)")
    st.caption("v6.1 | 三模組平均預測 | 全系統回測 | 宏觀流動性")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        
        st.header("💼 資產")
        default_input = """BTC-USD, 10000
AMD, 10000
NVDA, 10000
PLTR, 5000"""
        user_input = st.text_area("清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產", f"${total_value:,.0f}")
        if st.button("🚀 啟動掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 正在執行 Alpha 6.1 綜合運算..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        fund_data = {t: get_fundamental_anchor(t) for t in tickers_list}

    if df_close.empty: st.error("No Data"); return

    # --- PART 1: 宏觀 ---
    st.subheader("1. 宏觀戰略 (Macro & Liquidity)")
    vix = df_close['^VIX'].iloc[-1]
    tnx = df_close['^TNX'].iloc[-1]
    liq_val = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("💧 全球流動性", f"${liq_val:.2f}T" if df_macro is not None else "N/A")
    c2.metric("🌪️ VIX", f"{vix:.2f}", delta="避險成本", delta_color="inverse")
    c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%", "定價錨")

    if df_macro is not None:
        fig_liq = px.line(df_macro, y='Net_Liquidity', title='聯準會淨流動性趨勢', color_discrete_sequence=['#00BFFF'])
        fig_liq.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_liq, use_container_width=True)
    st.markdown("---")

    # --- PART 2: 個股 ---
    st.subheader("2. 個股戰略 (Strategic Radar)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend_matrix(df_close[ticker])
        f_info = fund_data.get(ticker, {})
        
        # 計算 1個月 的目標價 (四角定位 + 平均)
        targets = calc_targets_composite(df_close[ticker], df_high[ticker], df_low[ticker], f_info, days_forecast=22)
        kelly = calc_kelly(trend['status'])
        
        # 執行全模組回測
        bt = run_backtest_composite(df_close[ticker], df_high[ticker], df_low[ticker], days_ago=22)
        obv = calc_obv(df_close[ticker], df_vol[ticker])
        
        # 標題顯示
        t_avg_display = f"${targets['Avg']:.2f}" if targets['Avg'] else "-"
        
        with st.expander(f"🦅 {ticker} | {trend['status']} | 綜合目標: {t_avg_display}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 圖表
                st.markdown("#### 📉 雙軸圖 (±6 Months)")
                fig = go.Figure()
                dates = df_close.index[-126:]
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].iloc[-126:], name='Price', line=dict(color='#00FF7F', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].rolling(200).mean().iloc[-126:], name='SMA200', line=dict(color='gray', dash='dash')))
                if obv is not None:
                    fig.add_trace(go.Scatter(x=dates, y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), 
                                  yaxis2=dict(overlaying='y', side='right', showgrid=False, title='OBV'),
                                  legend=dict(x=0, y=1.1, orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            with k2: # 預測矩陣
                st.markdown("#### 🎯 四角定位 (1M)")
                st.write(f"**1. 物理 (ATR):** ${targets['ATR']:.2f}" if targets['ATR'] else "-")
                st.write(f"**2. 統計 (MC):** ${targets['MC']:.2f}" if targets['MC'] else "-")
                st.write(f"**3. 心理 (Fib):** ${targets['Fib']:.2f}" if targets['Fib'] else "-")
                st.write(f"**4. 價值 (DCF):** ${targets['Fund']}" if targets['Fund'] else "N/A")
                
                st.divider()
                st.markdown("#### 🧪 平均模型回測")
                if bt:
                    err = bt['Error']
                    c_err = "green" if abs(err) < 0.05 else "red"
                    st.markdown(f"1月前綜合預測誤差: <span style='color:{c_err}'>{err:.1%}</span>", unsafe_allow_html=True)
                    st.caption(f"預測: ${bt['Past_Avg']:.2f} vs 現價: ${bt['Price_Now']:.2f}")

            with k3: # 未來推演
                st.markdown("#### 🔮 趨勢推演")
                st.metric("2週方向", f"${trend['p_2w']:.2f}")
                st.metric("1月方向", f"${trend['p_1m']:.2f}")
                st.metric("3月方向", f"${trend['p_3m']:.2f}")
                
                st.divider()
                st.metric("Kelly 建議", kelly)
                st.metric("Forward P/E", f"{f_info.get('Forward_PE')}")

    st.markdown("---")
    
    # --- PART 3: 說明書 ---
    st.header("3. 系統運作原理與質性說明")
    with st.container():
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        
        st.markdown("### 🎯 綜合目標 (Composite Target)")
        st.info("為了消除單一模型的盲點，本系統將 **ATR (物理)**、**Monte Carlo (機率)**、**Fibonacci (心理)** 三者的預測值進行平均，得出一個「技術共識價」。並同時顯示華爾街的 **DCF 價值目標** 作為基本面參考。")
        
        st.divider()
        st.markdown("### 🧪 全模組回測 (Time-Travel Backtest)")
        st.info("系統會自動將時間回撥至 22 個交易日 (約 1 個月) 前，使用當時的數據重新運行 ATR、MC、Fib 三大模型，計算出「當時的綜合預測價」，並與「今天的現價」進行對比。誤差 < 5% 代表模型近期極為精準。")
        
        st.divider()
        st.markdown("### 🌊 雙軸資金流")
        st.markdown("左軸 K 線代表價格，右軸黃線代表 OBV (累積能量潮)。當 OBV 趨勢向上而價格盤整時，為強烈買進訊號。")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()